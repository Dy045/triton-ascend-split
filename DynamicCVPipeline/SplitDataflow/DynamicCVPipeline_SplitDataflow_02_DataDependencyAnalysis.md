# SplitDataflow 子 Pass 逐行讲解（二）：DataDependencyAnalysis

> 源码文件：`third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.cpp`
> 头文件：`third_party/ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h`
> 流水线位置：`SplitDataflowPass` 的 7 个步骤中的 **第 2 步**

---

## 一、这个 Pass 在做什么

这是整个 SplitDataflow 的**分析核心**。它的任务是把 module 里"哪些数据在 Cube 核和 Vector 核之间流动"完整地计算出来，产出四类依赖关系：

| 依赖类型 | 方向 | 含义 |
|---|---|---|
| `VectorToCube`（V2C） | Vector → Cube | 数据由 Vector 块产生、Cube 块消费 |
| `CubeToVector`（C2V） | Cube → Vector | 数据由 Cube 块产生、Vector 块消费 |
| `CubeToCube`（C2C） | Cube → Cube | Cube 块之间的 L1 共享数据（fixpipe 直传，不过 UB） |
| Memory Dependency | 双向 | 通过内存（别名）产生的隐式依赖，需插入同步 |

这些结果统一存在 `DataDependencyInfo` 这个 MLIR Analysis 里，供下一步 `InterCoreTransferAndSync` 使用。

**本次更新要点**：
- 新增 `scf.while` 循环支持（`processIterArgDependencies` 改为基于 `LoopLikeOpInterface` + `CVPipeline::MainLoop` 统一封装 for/while）；
- 新增**嵌套 iterArg 解析**（`resolveNestedIterArgInitValue`：init 值本身是外层循环 BlockArgument 时，逐层向外追溯到真实定义）；
- 依赖分析拆分为四个更细的场景：init 核内使用、init/yield 核类型不同的"混核"场景（`insertProducerAndRecordDeps` / `insertConsumerAndRecordDeps` / `recordInitValueDeps` 三种记录方式）；
- 新增 C→C 依赖（`DependencyType::CubeToCube`），`analyzeExternalInputs` 中 Cube 之间的依赖也会被记录；
- 新增**转置折叠判断**（`isAllTransposedInVector`）：matmul 结果若只被一个 transpose 消费、且 transpose 的用户全是 Vector，可把转置折叠进 fixpipe（NZ2DN 模式）；
- 移除了旧的 `analyzeV2CMatmulABType`（matmul A/B 标注）逻辑，对齐判断移到了 `InterCoreTransferAndSync` 内部；
- 新增**去重阶段**（`deduplicateDependencies`）和循环前后两次 `createBlockInfoMap`（插入占位块后需要重建块映射）。

**本次最新拉取新增要点**：
- **C→C 依赖细分为"块间"与"块内"两类**：新增 `analyzeInternalDeps` + `intraC2CDependencies`，专门分析**同一个 Cube computeBlock 内** matmul 之间的数据链（matmul 结果直接被同块内另一个 matmul 当作输入/init 消费），与 `analyzeExternalInputs` 里的块间 C2C 分开存储；
- **`isAllTransposedInVector` 升级为返回 pair**：`std::pair<bool, std::optional<Operation *>>`，除布尔标记外还返回"消费转置结果的 yield op"（若 transpose 的用户里有 yield）。`analyzeExternalOutputs` 据此把该 yield 记录到 `c2vDependencies.back().consumerYieldOp`，供 `InterCoreTransferAndSync` 在 fixpipe 内联转置时替换 yield 上的 use；
- **`DependencyInfo` 新增三个字段**：`isSplitedIf` / `realValue`（split-if 场景，由下游 `InterCoreTransferAndSync::AnalyzeSplittedIf` 填充）、`operand`（intra-block C2C 中消费 matmul 的输入 OpOperand 指针）。

---

## 二、关键数据结构（来自头文件）

```cpp
enum class DependencyType { VectorToCube, CubeToVector, CubeToCube };

struct BlockInfo {
  int blockId;                              // 块 id
  bool isCube;                              // 是否 Cube 块
  bool isControl;                           // 是否控制流块
  llvm::SetVector<mlir::Value> inputs;      // 外部输入（被本块消费、但定义在块外）
  llvm::SmallVector<mlir::Value> outputs;   // 外部输出（本块产生、被块外消费）
  llvm::SmallVector<mlir::Operation *> Operations; // 块内所有 op
};

struct DependencyInfo {
  DependencyType type;      // 依赖方向
  mlir::Value value;        // 跨核流动的那个 value
  int producerBlockId;      // 生产者块 id（经过 LCA 对齐后）
  int consumerBlockId;      // 消费者块 id（经过 LCA 对齐后）
  int iniProducerBlockId;   // 原始生产者块 id
  int iniConsumerBlockId;   // 原始消费者块 id

  bool isAllTranspoesd = false;   // 该值是否可把转置折叠进 fixpipe（NZ2DN）

  // split-if 场景（新增）：源值被 kSplittedIf 标记时，realValue 是
  // split-if 里真正被 yield 出来的 matmul 结果（由下游 AnalyzeSplittedIf 填充）
  bool isSplitedIf = false;
  mlir::Value realValue;

  // 仅 memDependencies 使用：内存依赖的生产/消费 op
  mlir::Operation *predOp;
  mlir::Operation *nextOp;
  // 仅 iterArg yield 依赖 / 转置折叠依赖使用：消费端是 yield 时的那个 yield op
  mlir::Operation *consumerYieldOp = nullptr;

  // 仅 intra-block C2C 使用（新增）：消费 matmul 的那个输入 OpOperand 指针，
  // 用于精确区分 A/B 输入与 init（outs）
  mlir::OpOperand *operand = nullptr;
};
```

`DataDependencyInfo` 是 MLIR Analysis（继承自 `PassWrapper` 通过 `getAnalysis<>` 获取），内部持有：
- `blockInfoMap`：`block_id -> BlockInfo`
- `v2cDependencies` / `c2vDependencies` / `c2cDependencies` / `memoryDependencies`
- `intraC2CDependencies`（**新增**）：同一个 Cube computeBlock 内 matmul 之间的 C2C 依赖

关键属性常量（`Common/Utils.h`）：
- `kCoreType = "ssbuffer.core_type"`：取值 `"CUBE"` / `"VECTOR"`（可能含逗号分隔的多个值）
- `kBlockId = "ssbuffer.block_id"`

另一个关键封装是 `CVPipeline::MainLoop`（`Common/Utils.h`）：它把 `scf.for` / `scf.while` 统一抽象为"主循环"，提供 `getBody()`、`getIterArgs()`、`getLoopYieldOp()`（for 取 body 终结符，while 取 after 区块终结符）等接口，让依赖分析不必关心具体循环形态。

---

## 三、逐行讲解

### 3.1 头文件与工具函数（L23-L124）

```cpp
#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"  // MemoryDependenceGraph
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"  // annotation::MarkOp
#include "mlir/Analysis/AliasAnalysis.h"   // 别名分析，用于内存依赖
#include "mlir/Dialect/GPU/IR/GPUDialect.h" // gpu::BarrierOp
#include "mlir/Dialect/Linalg/IR/Linalg.h" // linalg.matmul 等
```

```cpp
static constexpr const char *ssbufferCoreTypeCubeAttr = "CUBE";
static constexpr const char *ssbufferCoreTypeVectorAttr = "VECTOR";
static constexpr int ND_SHAPE_LENGTH = 2;   // 2D tensor
static constexpr int SHAPE_1D_LENGTH = 1;    // 1D tensor（新增）
static constexpr int constantIntType = 32;   // 占位常量的位宽
```

工具函数 `getSsbufferCoreType`（L63-L68）：读取 op 的 `ssbuffer.core_type` 属性，返回字符串（可能为 `""`）。

工具函数 `getCoreTypeWithIndex`（L71-L85）：
```cpp
llvm::StringRef getCoreTypeWithIndex(Operation *op, int index) {
  llvm::StringRef typeStr = getSsbufferCoreType(op);
  if (typeStr.contains(", ")) {          // 多结果 op，core_type 可能是 "CUBE, VECTOR" 这种
    llvm::SmallVector<llvm::StringRef> types;
    typeStr.split(types, ", ", -1, false);
    if (index < types.size()) return types[index].trim();
    return "";
  }
  return typeStr;
}
```
- 一个 op 若有多个 result，每个 result 可能属于不同核，此时 `core_type` 属性用逗号分隔。`index` 对应 result 的编号。

新增工具函数 `updateCoreTypeAtIndex`（L87-L124）：把多值 `core_type` 字符串中第 `index` 个槽位改写为 `newCoreType`（如把 `"CUBE, VECTOR"` 的第 1 个槽改成 `"CUBE, CUBE"`）。它服务于 `insertConsumerAndRecordDeps`——当把 yield 槽位"归还"给 init 的核类型时，必须同步改写 yield op 和循环 op 上的 core_type 属性，让两侧属性保持一致。索引越界时直接触发 fallback。

### 3.2 辅助判断函数（L127-L220）

```cpp
bool DataDependencyAnalysisPass::isControlFlowOp(mlir::Operation *op) {
  if (!op) return false;
  return isa<scf::ForOp, scf::WhileOp>(op) || isa<scf::IfOp>(op) ||
         isa<scf::YieldOp, scf::ConditionOp>(op);   // 新增 scf.condition
}

bool DataDependencyAnalysisPass::isCubeOrVectorOp(mlir::Operation *op) {
  if (isa<tensor::EmptyOp, linalg::FillOp>(op)) return true; // 空张量/填充 op 不算依赖
  return false;
}
```
- `tensor::EmptyOp`、`linalg::FillOp` 是"初始化"性质的 op，它们既不真正计算、也常被 Cube/Vector 两侧都用到，因此**不能作为跨核依赖的 source**。
- 控制流判断加入了 `scf::ConditionOp`（while 的 before 区块终结符）。

```cpp
bool DataDependencyAnalysisPass::isValidShapeForDependency(mlir::Value value) {
  auto tensorTy = dyn_cast<TensorType>(value.getType());
  if (!tensorTy) return false;
  if (tensorTy.getRank() != ND_SHAPE_LENGTH) return false; // 2D
  return true;
}

bool DataDependencyAnalysisPass::isValidScalarDependency(mlir::Value value) {
  if (isa<mlir::IntegerType, mlir::FloatType>(value.getType())) {
    auto defOp = value.getDefiningOp();
    if (defOp && isa<tensor::ExtractOp>(defOp)) return true; // 从 tensor 提取出的标量
  }
  return false;
}

// 新增：1D tensor 也允许作为依赖（如 bias 一维向量）
bool DataDependencyAnalysisPass::isValid1DValueForDependency(mlir::Value value) {
  auto tensorTy = dyn_cast<TensorType>(value.getType());
  return tensorTy && tensorTy.getRank() == SHAPE_1D_LENGTH;
}
```

新增 `isAllTransposedInVector`（L172-L197，本次更新为返回 pair）：判断一个 matmul 结果是否**只被一个 `linalg.transpose` 消费，且 transpose 的所有用户都是 VECTOR**。满足时 C→V 传输可把转置折叠进 fixpipe（`FixpipeDMAMode::NZ2DN`），省掉 Vector 核上单独的转置计算。返回值从 `bool` 升级为 `std::pair<bool, std::optional<Operation *>>`——第二个元素记录**消费转置结果的 yield op**（若有），供 `analyzeExternalOutputs` 写入 `consumerYieldOp`：
```cpp
std::pair<bool, std::optional<mlir::Operation *>>
DataDependencyAnalysisPass::isAllTransposedInVector(mlir::Value value) {
  if (!isa<linalg::MatmulOp>(value.getDefiningOp()))
    return {false, std::nullopt};
  if (!llvm::hasSingleElement(value.getUsers()))       // 必须只有一个用户（transpose）
    return {false, std::nullopt};
  auto *userOp = *value.getUsers().begin();
  if (!isa<linalg::TransposeOp>(userOp))
    return {false, std::nullopt};
  for (mlir::Operation *transposeOpUser : userOp->getUsers()) {
    if (getSsbufferCoreType(transposeOpUser) != ssbufferCoreTypeVectorAttr)
      return {false, std::nullopt};                    // 用户不是 VECTOR → 不折叠
  }
  // 新增：若 transpose 的用户里有 yield，把这个 yield 返回出去
  for (mlir::Operation *transposeOpUser : userOp->getUsers()) {
    if (isa<scf::YieldOp>(transposeOpUser))
      return {true, transposeOpUser};
  }
  return {true, std::nullopt};
}
```
- **为什么需要返回 yield op**：转置折叠成 fixpipe（NZ2DN）后，原来的 `matmul → transpose → yield` 链路变成 `fixpipe → yield`，fixpipe 的 receive 值要替换 yield 上的 operand。提前把 yield 记在 `consumerYieldOp` 上，传输 pass 就能精准定位需要替换的 use。

`isValidValueForDependency`（L192-L211）是核心过滤器：
```cpp
bool DataDependencyAnalysisPass::isValidValueForDependency(mlir::Value value) {
  if (isValidScalarDependency(value)) return true;    // 标量（tensor.extract 产生）视为有效
  if (isValid1DValueForDependency(value)) return true; // 1D tensor 有效（新增）
  if (!isValidShapeForDependency(value)) return false; // 非 2D tensor 无效
  Operation *defOp = value.getDefiningOp();
  if (defOp && isCubeOrVectorOp(defOp)) return false;  // EmptyOp/FillOp 产生的无效
  return true;
}
```

`isOuterOpArg`（L214-L220）：判断 value 是否是 `BlockArgument`（函数参数 / 循环 iter_arg）。这类值没有 defining op，无法直接定位"生产者块"。

### 3.3 collectBlockInfo：收集单个块的信息（L260-L314）

```cpp
void DataDependencyAnalysisPass::collectBlockInfo(
    DataDependencyInfo &info, int blockId,
    llvm::SmallVector<mlir::Operation *> &ops) {
  if (ops.empty()) { ... return; }

  BlockInfo blockInfo;
  blockInfo.blockId = blockId;
  blockInfo.isCube = false;

  // 只要块内存在 CUBE 类型 op，就标记为 Cube 块（因为需要检查数据流）
  StringRef coreType = getSsbufferCoreType(ops[0]);
  if (coreType.contains(ssbufferCoreTypeCubeAttr)) blockInfo.isCube = true;

  blockInfo.isControl = false;
  if (isControlFlowOp(ops[0])) blockInfo.isControl = true;

  llvm::DenseSet<mlir::Operation *> opSet(ops.begin(), ops.end());

  for (auto *op : ops) {
    blockInfo.Operations.push_back(op);
    // 收集输入：operand 的定义不在本块内 → 外部输入
    for (auto operand : op->getOperands()) {
      mlir::Operation *defOp = operand.getDefiningOp();
      if (!defOp || opSet.find(defOp) == opSet.end()) {
        blockInfo.inputs.insert(operand);   // SetVector 自动去重
      }
    }
    // 收集输出：result 有块外用户 → 外部输出
    for (auto result : op->getResults()) {
      bool hasExternalUser = false;
      for (mlir::Operation *user : result.getUsers()) {
        if (opSet.find(user) == opSet.end()) { hasExternalUser = true; break; }
      }
      if (hasExternalUser) blockInfo.outputs.push_back(result);
    }
  }
  info.getBlockInfoMap()[blockInfo.blockId] = blockInfo;
}
```

**核心思想**：一个"块"是连续一组具有相同 `block_id` 的 op。块的 `inputs` = 从块外流入的值，`outputs` = 流向块外的值。这些"跨块"的值正是潜在的核间数据依赖。

### 3.4 createBlockInfoMap：遍历划分块 + createBlockInfoConstOp（L317-L357）

```cpp
void DataDependencyAnalysisPass::createBlockInfoMap(DataDependencyInfo &info) {
  int currentId = -2;                 // 初始哨兵值
  static constexpr int startCurrId = -2;
  llvm::SmallVector<mlir::Operation *> currentOps;

  module.walk([&](mlir::Operation *op) {
    auto opBlockIdOpt = CVPipeline::getOpBlockId(op);
    if (opBlockIdOpt) {
      int opBlockId = *opBlockIdOpt;
      // id 变化 → 上一个块结束
      if (opBlockId != currentId && currentId != startCurrId) {
        collectBlockInfo(info, currentId, currentOps);
        currentOps.clear();
      }
      currentId = opBlockId;
      currentOps.push_back(op);
    }
  });
  // 处理最后一组
  if (!currentOps.empty()) collectBlockInfo(info, currentId, currentOps);
}
```

- 关键假设：**相同 block_id 的 op 在 module 遍历顺序中是连续的**。这是 `PlanComputeBlock` 之前的 `ReorderOpsByBlockId` 保证的（op 已按 block_id 重排）。
- 遍历时遇到 block_id 变化，就把上一组 op 打包成一个 `BlockInfo`。

新增辅助函数 `createBlockInfoConstOp`（L341-L357）：在指定插入点创建一个 `arith.constant`（占位 op），分配新的 block_id 并设置 core_type，同时把对应的 `BlockInfo` 注册进 `blockInfoMap`。它是 `insertProducerAndRecordDeps` / `insertConsumerAndRecordDeps` 的共同基础——占位块（生产者/消费者锚点）的创建逻辑被抽成了这个函数（旧版是直接内联在 insertProducerAndRecordDeps 里）。

```cpp
mlir::Operation *DataDependencyAnalysisPass::createBlockInfoConstOp(
    OpBuilder &builder, Location loc, llvm::StringRef coreType,
    DataDependencyInfo &info) {
  int newId = CVPipeline::getAvailableBlockId(module);
  auto constOp = builder.create<arith::ConstantIntOp>(loc, 0, constantIntType);
  setOpBlockId(constOp, newId);
  setOpCoreType(constOp, coreType);
  BlockInfo blockInfo; ...
  info.getBlockInfoMap()[newId] = blockInfo;
  return constOp;
}
```

### 3.5 collectDepInfo：记录一条依赖（L359-L385）

```cpp
bool DataDependencyAnalysisPass::collectDepInfo(
    mlir::Value depvalue, DependencyType dependencyType,
    llvm::SmallVector<DependencyInfo> &dependencies, int iniProdId,
    int iniConsId, DataDependencyInfo &info, bool isAllTranspoesd) {
  DependencyInfo depInfo;
  depInfo.type = dependencyType;
  depInfo.value = depvalue;
  depInfo.iniProducerBlockId = iniProdId;   // 原始生产者 id
  depInfo.iniConsumerBlockId = iniConsId;   // 原始消费者 id

  // 找 LCA（最近公共祖先）对齐后的块 id，用于正确处理嵌套循环
  std::pair<int, int> commonLevelIds =
      findCommonLevelBlockIds(info, iniProdId, iniConsId);
  if (commonLevelIds.first == -1 || commonLevelIds.second == -1) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return false;
  }
  depInfo.producerBlockId = commonLevelIds.first;   // 对齐后的生产者 id
  depInfo.consumerBlockId = commonLevelIds.second;  // 对齐后的消费者 id
  if (isAllTranspoesd) depInfo.isAllTranspoesd = true;  // 转置可折叠进 fixpipe
  dependencies.push_back(depInfo);
  return true;
}
```

- 这里同时保存了 `ini*`（原始）和 `*`（对齐后）两组 id。原始 id 用于定位具体 op，对齐后的 id 用于插入传输/同步（因为传输要放在**同一层级的两个块之间**）。
- 返回值从 `void` 改为 `bool`，调用方据此决定是否继续后续处理（如 `consumerYieldOp` 只在成功插入后才设置）。

### 3.6 iterArg 依赖处理（L223-L257 嵌套解析 + L387-L697 主流程）

这是处理循环携带变量（iter_arg）跨核依赖的部分，是整个 Pass 最复杂的一块。新版重写后统一支持 `scf.for` 和 `scf.while`，并新增嵌套 iterArg 解析与四种依赖场景。

#### 3.6.1 resolveNestedIterArgInitValue：嵌套 iterArg 解析（L223-L257，新增）

当内层循环的 init 值本身是外层循环的 BlockArgument（即"嵌套 iterArg"）时，旧版直接跳过；新版会逐层向外追溯，找到真实的非 BlockArgument 定义：

```cpp
mlir::Value DataDependencyAnalysisPass::resolveNestedIterArgInitValue(
    mlir::Value initValue) {
  llvm::DenseSet<mlir::Value> visited;         // 防环
  mlir::Value currentValue = initValue;
  while (true) {
    if (!visited.insert(currentValue).second) break;
    auto blockArg = dyn_cast<mlir::BlockArgument>(currentValue);
    if (!blockArg) break;                       // 到达真实定义
    mlir::Operation *parentOp = blockArg.getOwner()->getParentOp();

    unsigned argIndex = blockArg.getArgNumber();
    ValueRange initArgs;
    if (auto outerFor = dyn_cast<scf::ForOp>(parentOp)) {
      if (argIndex == 0) break;                 // arg 0 是 induction var，不是携带值
      --argIndex;                              // for 的 body arg 0 是 iv，携带值从 1 开始
      initArgs = outerFor.getInitArgs();
    } else if (auto outerWhile = dyn_cast<scf::WhileOp>(parentOp)) {
      // Triton 的 while before/after 区块参数与 init 一一对应
      initArgs = outerWhile.getInits();
    } else {
      break;
    }
    if (argIndex >= initArgs.size()) break;
    currentValue = initArgs[argIndex];          // 换成外层 init，继续向上追
  }
  return currentValue;
}
```

- **关键点**：`scf.for` 的循环体第一个 BlockArgument 是 induction variable（循环计数器），携带值从第 2 个开始，所以要做 `--argIndex`；而 `scf.while` 的区块参数与 `getInits()` 一一对应，不需要偏移。这是 for/while 在 iterArg 索引上的本质差异。

#### 3.6.2 collectDiffCoreTypeUsers（L388-L409）

```cpp
llvm::SmallVector<mlir::Operation *>
DataDependencyAnalysisPass::collectDiffCoreTypeUsers(
    mlir::BlockArgument iterArg, llvm::StringRef initCoreType) {
  llvm::SmallVector<mlir::Operation *> diffUsers;
  for (mlir::Operation *user : iterArg.getUsers()) {
    if (isa<scf::YieldOp>(user)) continue;         // 跳过 yield
    if (isControlFlowOp(user)) continue;            // 跳过嵌套控制流（暂不支持）
    auto userCoreType = getCoreTypeWithIndex(user, 0);
    if (userCoreType != initCoreType && !userCoreType.empty()) {
      diffUsers.push_back(user);                    // 核类型不同 → 跨核使用
    }
  }
  return diffUsers;
}
```

#### 3.6.3 三种记录函数（L413-L553）

**insertProducerAndRecordDeps**（L413-L463）：为跨核的 iterArg 插入一个"生产者块"并记录依赖。

```cpp
void DataDependencyAnalysisPass::insertProducerAndRecordDeps(
    mlir::LoopLikeOpInterface loopOp, mlir::BlockArgument loopArg,
    llvm::StringRef initCoreType,
    llvm::SmallVector<mlir::Operation *> &diffUsers, DataDependencyInfo &info) {
  Operation *loopOperation = loopOp.getOperation();
  CVPipeline::MainLoop loop(loopOperation);            // 统一的 for/while 封装
  OpBuilder builder(loopOperation);
  builder.setInsertionPointToStart(loop.getBody());    // 循环体开头
  auto constOp = createBlockInfoConstOp(builder, loc, initCoreType, info);
  int newId = *CVPipeline::getOpBlockId(constOp);

  llvm::DenseSet<int> processedUserBlockIds;           // 同块的多个 user 只记一次
  for (auto &user : diffUsers) {
    int userBlockId = ...;
    if (!processedUserBlockIds.insert(userBlockId).second) continue;
    // init 是 VECTOR → V2C；init 是 CUBE → C2V
    DependencyType depType = (initCoreType == VECTOR) ? VectorToCube : CubeToVector;
    collectDepInfo(loopArg, depType, targetDeps, newId, userBlockId, info);
  }
}
```

- **为什么插入 `arith::ConstantIntOp` 作为占位？** 因为 iterArg（BlockArgument）没有 defining op，无法作为"生产者块"的 anchor。插入一个常量 op 充当生产者块的起点，并打上 init 的核类型标记，后续 `getBlockStartEnd` 就能定位到这个生产者块。
- 新版用 `LoopLikeOpInterface` + `MainLoop` 取代了写死的 `scf::ForOp`，while 的 after 区块同样适用。

**insertConsumerAndRecordDeps**（L465-L510，新增）：当 iterArg 只被 init 核类型的用户使用（yield 回的值来自另一种核）时，在 yield 前插入一个"消费者块"，把 yield 的值作为跨核依赖的 source：

```cpp
void DataDependencyAnalysisPass::insertConsumerAndRecordDeps(
    mlir::LoopLikeOpInterface loopOp, mlir::Value yieldedValue, int iterArgIndex,
    llvm::StringRef initCoreType, DataDependencyInfo &info) {
  scf::YieldOp yieldOp = CVPipeline::MainLoop::getLoopYieldOp(loopOperation);
  OpBuilder builder(yieldOp);
  auto constOp = createBlockInfoConstOp(builder, loc, initCoreType, info);

  // 方向：init 是 VECTOR、yielded 是 CUBE → 这条依赖是 C→V
  DependencyType depType = (initCoreType == VECTOR) ? CubeToVector : VectorToCube;
  collectDepInfo(yieldedValue, depType, targetDeps, yieldedDefBlockId, newId, info);
  targetDeps.back().consumerYieldOp = yieldOp;   // 记录 yield，传输时要替换 yield 的 use

  // 把 yield 槽位和循环结果槽位的 core_type 改回 init 核类型（属性归一化）
  updateCoreTypeAtIndex(yieldOp, iterArgIndex, initCoreType);
  updateCoreTypeAtIndex(loopOperation, iterArgIndex, initCoreType);
}
```

- `consumerYieldOp` 会被 `InterCoreTransferAndSync` 使用：传输产生的 receiveValue 除了替换普通 user 外，还要替换 yield 上的 use（`dep.consumerYieldOp->replaceUsesOfWith(...)`）。

**recordInitValueDeps**（L512-L553，新增）：当 init 值本身来自与 yield 不同的核（如外部 CUBE 输入喂给 VECTOR 主导的循环）时，直接记录一条"init 定义块 → 循环块"的依赖，不插入占位块：

```cpp
void DataDependencyAnalysisPass::recordInitValueDeps(
    mlir::LoopLikeOpInterface loopOp, mlir::Value initValue,
    llvm::StringRef yieldCoreType, DataDependencyInfo &info) {
  // 方向：yield 是 VECTOR、init 定义是 CUBE → C→V（init 值要先传进循环）
  DependencyType depType = (yieldCoreType == VECTOR) ? CubeToVector : VectorToCube;
  collectDepInfo(initValue, depType, targetDeps, initDefBlockId, loopBlockId, info);
}
```

#### 3.6.4 checkLoopYieldCoreType（L555-L577，新增）

校验函数：遍历 yield 的每个 operand，若其定义 op 是 `scf.for`/`scf.while`（即嵌套循环的结果直接被 yield），则比较 yield 槽位上的 core_type 与定义 op 对应 result 的 core_type 是否一致。不一致说明嵌套循环的结果被改判了核归属，直接 fallback。这是处理嵌套循环时的**前置合法性检查**。

#### 3.6.5 processIterArgDependencies 主流程（L580-L697）

```cpp
void DataDependencyAnalysisPass::processIterArgDependencies() {
  // 收集所有 LoopLikeOpInterface（含 for/while）
  for (mlir::LoopLikeOpInterface loopOp : loopOps) {
    if (!isa<scf::ForOp, scf::WhileOp>(loopOperation)) continue;

    CVPipeline::MainLoop loop(loopOperation);
    scf::YieldOp yieldOp = CVPipeline::MainLoop::getLoopYieldOp(loopOperation);
    if (!checkLoopYieldCoreType(yieldOp)) { fallback; return; }   // 前置校验

    // iter_arg 数量必须与 init 数量一致
    SmallVector<Value> iterArgs = loop.getIterArgs();
    if (iterArgs.size() != numIterArgs) { fallback; return; }

    for (int iterArgIndex = 0; iterArgIndex < numIterArgs; ++iterArgIndex) {
      mlir::Value initValue = loopOp.getInits()[iterArgIndex];
      mlir::BlockArgument iterArg = ...;
      mlir::Value yieldedValue = loopOp.getYieldedValues()[iterArgIndex];

      // 1D tensor 或 2D tensor 才有效
      if (!isValid1DValueForDependency(iterArg) &&
          (!isValidShapeForDependency(initValue) ||
           !isValidShapeForDependency(yieldedValue))) continue;

      Operation *initDefOp = initValue.getDefiningOp();
      if (!yieldedDefOp) continue;
      auto yieldCoreType = getCoreTypeWithIndex(loopOperation, iterArgIndex);

      // 嵌套 iterArg：init 是 BlockArgument 时向外追溯真实定义（新增）
      if (!initDefOp) {
        auto realInitValue = resolveNestedIterArgInitValue(initValue);
        auto realInitDefOp = realInitValue.getDefiningOp();
        if (!realInitDefOp || isCubeOrVectorOp(realInitDefOp)) continue;
        // 追溯到的真实 init 核类型与 yield 不一致 → 冲突，fallback
        if (getCoreTypeWithIndex(realInitDefOp, ...) != yieldCoreType) {
          fallback; return;
        }
        initDefOp = realInitDefOp;
        initValue = realInitValue;
      }
      auto initCoreType = getCoreTypeWithIndex(initDefOp, ...);

      if (initCoreType == yieldCoreType || isCubeOrVectorOp(initDefOp)) {
        // 场景 1：init 与 yield 同核（或 init 是 Empty/Fill）
        // → 只有"不同核的用户"才是跨核依赖，插入生产者占位块
        auto diffUsers = collectDiffCoreTypeUsers(iterArg, yieldCoreType);
        if (!diffUsers.empty())
          insertProducerAndRecordDeps(loopOp, iterArg, yieldCoreType, diffUsers, info);

      } else {
        // 场景 2：init 与 yield 不同核（混核携带）
        // 把 iterArg 的用户按核类型分成两组
        for (mlir::Operation *user : iterArg.getUsers()) { ... 分组 ... }

        if (!initCoreTypeUsers.empty() && !yieldCoreTypeUsers.empty()) {
          // 2a：两种核都在用 → init 值先传进来（recordInitValueDeps），
          //     再为 yield 核的用户插入生产者块
          recordInitValueDeps(loopOp, initValue, yieldCoreType, info);
          insertProducerAndRecordDeps(loopOp, iterArg, yieldCoreType,
                                       yieldCoreTypeUsers, info);
        } else if (!initCoreTypeUsers.empty() && yieldCoreTypeUsers.empty()) {
          // 2b：只有 init 核在用，yield 值来自另一种核 → 在 yield 前插入消费者块
          insertConsumerAndRecordDeps(loopOp, yieldedValue, iterArgIndex,
                                      initCoreType, info);
        } else if (initCoreTypeUsers.empty() && !yieldCoreTypeUsers.empty()) {
          // 2c：只有 yield 核在用 → 只需把 init 值传进来
          recordInitValueDeps(loopOp, initValue, yieldCoreType, info);
        }
        // 2d：都没用 → 无依赖，跳过
      }
    }
  }
}
```

四种场景小结（对应"混核携带变量"的处理矩阵）：

| 场景 | init 核用户 | yield 核用户 | 处理 |
|---|---|---|---|
| 1（同核） | — | — | 只为不同核用户插生产者块 |
| 2a | 有 | 有 | `recordInitValueDeps` + `insertProducerAndRecordDeps` |
| 2b | 有 | 无 | `insertConsumerAndRecordDeps`（yield 前插消费者块） |
| 2c | 无 | 有 | `recordInitValueDeps`（init 值先跨核传进循环） |

### 3.7 外部输入分析（V2C + C2C）（L700-L770）

`analyzeExternalInputs`：遍历所有 Cube 块的 inputs，按定义者的 core_type 分两个 case：

```cpp
  for (auto &[id, blockInfo] : blockInfoMap) {
    if (!blockInfo.isCube || blockInfo.isControl || blockInfo.inputs.empty()) continue;
    for (mlir::Value input : blockInfo.inputs) {
      if (!isValidValueForDependency(input)) continue;
      if (isOuterOpArg(input)) continue;   // BlockArgument 跳过

      Operation *defOp = input.getDefiningOp();
      auto coreType = getCoreTypeWithIndex(defOp, resultIndex);
      if (coreType == "") continue;

      // Case 1（新增）：Cube -> Cube 依赖
      // 两个 Cube 块之间也走 L1 fixpipe 直传（C2C），记录进 c2cDependencies
      if (coreType == ssbufferCoreTypeCubeAttr) {
        collectDepInfo(input, DependencyType::CubeToCube,
                       info.getC2CDependencies(), producerId, blockInfo.blockId, info);
        continue;
      }
      // Case 2：Vector -> Cube 依赖
      if (coreType == ssbufferCoreTypeVectorAttr) {
        collectDepInfo(input, DependencyType::VectorToCube, v2cDependencies,
                       producerId, blockInfo.blockId, info);
      }
    }
  }
```

- 旧版遇到 Cube 输入直接 `continue`（注释写"Cube->Cube 已在别处处理"）；新版正式把 C2C 依赖纳入分析，由 `InterCoreTransferAndSync::handleCubeToCube` 消费（matmul→trunc→matmul 链路走 L1 fixpipe + pre_quant 折叠）。

### 3.8 外部输出分析（C2V）（L773-L865）

`analyzeExternalOutputs`：找出 Cube 块产生的、被 Vector 块消费的输出：

```cpp
  for (auto &[id, blockInfo] : blockInfoMap) {
    if (!blockInfo.isCube || blockInfo.outputs.empty()) continue;
    for (mlir::Value output : blockInfo.outputs) {
      if (!isValidValueForDependency(output)) continue;
      if (isa<IntegerType, FloatType>(output.getType())) continue; // 标量跳过

      auto opResult = dyn_cast<OpResult>(output);
      unsigned resultIndex = opResult.getResultNumber();
      StringRef resultCoreType = getCoreTypeWithIndex(output.getDefiningOp(), resultIndex);
      if (resultCoreType != ssbufferCoreTypeCubeAttr) continue;

      // 本次更新：返回 pair——是否可折叠转置 + 消费转置结果的 yield op
      auto [isAllTranspoesd, transposedYieldOp] = isAllTransposedInVector(output);

      llvm::DenseSet<int> handledBlockIds;   // 去重，避免同一消费者记录多次
      for (mlir::Operation *user : output.getUsers()) {
        int outputIndex = 0;
        if (isa<scf::YieldOp>(user)) {
          // 找到 output 在 yield 的哪个 operand 位置（多槽位时）
          for (unsigned i = 0; i < user->getNumOperands(); ++i)
            if (user->getOperand(i) == output) { outputIndex = i; break; }
        }
        auto userCoreType = getCoreTypeWithIndex(user, outputIndex);
        if (userCoreType == ssbufferCoreTypeVectorAttr) {
          if (handledBlockIds.insert(consumerId).second) {  // 去重
            if (collectDepInfo(output, DependencyType::CubeToVector, c2vDependencies,
                               blockInfo.blockId, consumerId, info, isAllTranspoesd)) {
              // 本次更新：把消费转置结果的 yield 记到依赖上，供传输 pass 替换 use
              if (transposedYieldOp.has_value()) {
                c2vDependencies.back().consumerYieldOp = *transposedYieldOp;
              }
            }
          }
        }
      }
    }
  }
```

- 旧版有独立的 `analyzeV2CMatmulABType`（给依赖打 `isMatmulA`/`isMatmulB` 标注并给 matmul 设置 `ssbuffer.adep`/`ssbuffer.bdep` 属性）；新版删除了这一步，`DependencyInfo` 中的 `isMatmulA`/`isMatmulB`/`iniMatmulOp` 字段一并移除。NZ 对齐判断移到了 `InterCoreTransferAndSync` 内部（对 V2C 统一做 `computeExpectedShape`），语义更简单：**所有进入 Cube 的 2D 输入都按 NZ 对齐**，不再区分 A/B。

### 3.9 【本次更新】块内 C2C 分析：analyzeInternalDeps（L867-L922）

同一个 Cube computeBlock 内，如果 matmul 的 A/B 输入（或 init）直接来自同块内另一个 matmul 的结果，这种"Cube 核内部、但块内两个 matmul 之间"的数据链也要走 L1 fixpipe（matmul 结果不落 UB、直接喂给下一个 matmul）。它与 `analyzeExternalInputs` 记录的**块间** C2C（`c2cDependencies`）不同，单独存放在 `intraC2CDependencies`：

```cpp
// Trace an operand's defining op back to find the source matmul.
static linalg::MatmulOp resolveSameBlockMatmulProducer(mlir::Value operand,
                                                       int consumerBlockId) {
  // 沿中间 op（trunc 等 pre_quant 链）向上追溯到源头 matmul
  Operation *defOp = CVPipeline::getSourceThroughCIntermediateOps(operand);
  auto producer = dyn_cast_if_present<linalg::MatmulOp>(defOp);
  if (!producer) return nullptr;
  auto producerBlockIdOpt = CVPipeline::getOpBlockId(producer);
  if (!producerBlockIdOpt || *producerBlockIdOpt != consumerBlockId)
    return nullptr;                        // 生产者必须在同一 block 内
  return producer;
}

// Analyze C->C dependencies between the matmuls inside the same computeBlock.
void DataDependencyAnalysisPass::analyzeInternalDeps(DataDependencyInfo &info) {
  auto &blockInfoMap = info.getBlockInfoMap();
  auto &intraBlockDeps = info.getIntraC2CDependencies();

  for (auto &[blockId, blockInfo] : blockInfoMap) {
    if (!blockInfo.isCube) continue;       // 只关心 Cube 块

    // 收集块内所有 matmul
    llvm::SmallVector<linalg::MatmulOp> matmuls;
    for (mlir::Operation *op : blockInfo.Operations)
      if (auto matmulOp = dyn_cast<linalg::MatmulOp>(op))
        matmuls.push_back(matmulOp);
    if (matmuls.size() < 2) continue;      // 少于 2 个 matmul 不可能有块内链

    for (linalg::MatmulOp consumer : matmuls) {
      for (OpOperand &opOperand : consumer->getOpOperands()) {
        if (!resolveSameBlockMatmulProducer(opOperand.get(), blockId))
          continue;                        // 该输入不是同块 matmul 产生的
        DependencyInfo depInfo;
        depInfo.type = DependencyType::CubeToCube;
        depInfo.value = opOperand.get();   // 跨 matmul 流动的 value
        depInfo.operand = &opOperand;      // 精确到消费 matmul 的输入槽位
        depInfo.producerBlockId = depInfo.consumerBlockId = blockId;
        depInfo.iniProducerBlockId = depInfo.iniConsumerBlockId = blockId;
        intraBlockDeps.push_back(depInfo);
      }
    }
  }
}
```

- **`getSourceThroughCIntermediateOps`**（`Common/Utils.h`，本次更新引入）：沿 C2C 合法中间 op 链（如 `trunc` 等 pre_quant 折叠候选）向上追溯，直到遇到非中间 op。`InterCoreTransferAndSync` 里原来的 `isValidC2CIntermediateOp` 局部实现被它统一替代。
- **`operand` 指针的意义**：块内 C2C 的消费端是 matmul，它的操作数分两类——**A/B 输入**（`getDpsInputs()`）和 **init/outs**（`getDpsInits()`）。下游 `handleCubeToCube` 需要明确知道这条依赖对应哪个输入槽位，才能只替换输入 operand（matmul 结果同时被当 init 用的场景必须保留）。
- **为什么独立存放而不并入 `c2cDependencies`**：块间 C2C 的生产/消费在两个不同 block，插入 fixpipe 的插入点逻辑（LCA 对齐、块边界）与块内 C2C（生产 matmul 之后、消费 matmul 之前，同 block 内）完全不同。分开存放让 `InterCoreTransferAndSync` 能用不同的插入点策略分别处理。

### 3.10 内存依赖分析（L924-L1020）

`analyzeMemoryEffect` 使用 MLIR 的别名分析 + 自定义的 `MemoryDependenceGraph` 来发现"通过内存隐式产生的跨核依赖"：

```cpp
  auto &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
  MemoryDependenceGraph memDepGraph(module, aliasAnalysis);

  module.walk([&](mlir::Operation *op) -> WalkResult {
    if (op->getNumRegions() > 0) return WalkResult::advance(); // 跳过带 region 的 op
    if (isa<annotation::MarkOp, gpu::BarrierOp>(op)) return WalkResult::advance();

    auto currBlockIdOpt = CVPipeline::getOpBlockId(op);
    llvm::StringRef currCoreType = getSsbufferCoreType(op);
    if (!currBlockIdOpt || currCoreType.empty()) return WalkResult::advance();
    int currBlockId = ...;

    // 遍历所有"在当前 op 之前执行、且可能冲突"的 op
    for (mlir::Operation *predOp : memDepGraph.getExecBefore(op)) {
      ... 处理带 region 的 predOp（递归取 realDependency）...
      auto predCoreType = getSsbufferCoreType(predOp);
      if (predCoreType == currCoreType || predCoreType.empty()) continue; // 同核不处理
      ...
      findCommonLevelBlockIds(...)
      if (producerBlockId == consumerBlockId) continue;  // 同块不处理
      collectMemDepInfo(predCoreType, ..., memoryDependencies, predOp, op);
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);  // LCA 失败 → fallback
  }
```

- `collectMemDepInfo`（L854-L875）根据 `predCoreType` 决定依赖方向，构造一条 memory dependency（`depInfo.value` 为空，但会带上 `predOp`/`nextOp` 两个 op 指针，供 `InterCoreTransferAndSync::handleMemoryDependency` 精确打标和插同步）。

### 3.11 findCommonLevelBlockIds：LCA 对齐（L1022-L1099）

这是处理嵌套结构的核心算法。当生产者和消费者处在**不同的循环层级**时，传输必须放在"同一层级"的两个块之间：

```cpp
std::pair<int, int> DataDependencyAnalysisPass::findCommonLevelBlockIds(
    DataDependencyInfo &info, int producerBlockId, int consumerBlockId) {
  ...
  mlir::Operation *producerOp = pInfo.Operations[0];
  mlir::Operation *consumerOp = cInfo.Operations[0];
  mlir::Block *pBlock = producerOp->getBlock();
  mlir::Block *cBlock = consumerOp->getBlock();

  // 同 block → 直接返回原始 id
  if (pBlock == cBlock) return {producerBlockId, consumerBlockId};

  // 收集 producer 的祖先链
  llvm::SmallVector<mlir::Operation *> pAncestors;
  pAncestors.push_back(producerOp);
  mlir::Operation *current = producerOp->getParentOp();
  while (current) { pAncestors.push_back(current); current = current->getParentOp(); }

  // 从 consumer 向上走，找第一个共同祖先
  mlir::Operation *before = consumerOp;
  current = consumerOp;
  while (current) {
    auto it = std::find(pAncestors.begin(), pAncestors.end(), current);
    if (it != pAncestors.end()) {           // 找到共同祖先
      size_t pIndex = std::distance(pAncestors.begin(), it);
      if (pIndex == 0) break;
      mlir::Operation *pPrevOp = pAncestors[pIndex - 1]; // 共同祖先的"下一层" producer
      ...
      return {pPrevId, cPrevId};            // 返回共同祖先下一层的两个块 id
    }
    before = current;
    current = current->getParentOp();
  }
  return {-1, -1};
}
```

- **为什么需要 LCA？** 假设 Vector 块在内层循环、Cube 块在外层循环。直接把数据从内层 Vector 传到外层 Cube 是不合法的（层级不对）。LCA 找到两者最近的公共循环层级，把依赖"提升"到该层级再传输。

### 3.12 deduplicateDependencies + runOnOperation 主流程（L1127-L1195）

新增去重函数（L1127-L1138）：`std::unique` 移除 `value + iniProducerBlockId + iniConsumerBlockId` 三元组完全相同的重复依赖（iterArg 分析和外部输入分析可能记录同一条依赖两次）。

```cpp
void DataDependencyAnalysisPass::runOnOperation() {
  module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;

  auto &info = getAnalysis<DataDependencyInfo>();  // 获取 Analysis

  // Step 1: 收集块信息
  createBlockInfoMap(info);
  // Step 2: 分析 iter_arg 依赖（for + while，含嵌套 iterArg 解析）
  processIterArgDependencies();
  if (CVPipeline::hasFallbackAttr(module)) return;
  createBlockInfoMap(info);          // 重建：上一步插入了占位块，块映射需要刷新

  // Step 3: 分析外部输入（V2C + 块间 C2C）+ 外部输出（C2V，含 isAllTranspoesd）
  analyzeExternalInputs(info);
  analyzeExternalOutputs(info);

  // Step 4（本次更新新增）：分析块内 C2C 依赖（同 Cube 块内 matmul 链）
  analyzeInternalDeps(info);

  // Step 5: 分析内存依赖（memdep sync）
  analyzeMemoryEffect(info);

  // Step 6: 五类依赖各自去重
  deduplicateDependencies(info.getV2CDependencies());
  deduplicateDependencies(info.getC2VDependencies());
  deduplicateDependencies(info.getC2CDependencies());
  deduplicateDependencies(info.getMemoryDependencies());

  info.setValid(true);   // 标记分析有效
}
```

- **为什么 createBlockInfoMap 要跑两次？** `processIterArgDependencies` 会向循环体内插入占位常量块（生产者/消费者锚点），这些新块必须进入 `blockInfoMap`，后续 `analyzeExternalInputs/Outputs` 和 `findCommonLevelBlockIds` 才能识别它们。

---

## 四、整体算法总结

```
1. createBlockInfoMap：按 block_id 切块，计算每块的 inputs/outputs/isCube/isControl
2. processIterArgDependencies：处理 for/while 循环 iterArg 的跨核依赖
   ├─ resolveNestedIterArgInitValue：嵌套 iterArg 向外追溯真实定义
   ├─ 同核场景 → insertProducerAndRecordDeps（插占位生产者块）
   └─ 混核场景 → recordInitValueDeps / insertConsumerAndRecordDeps
   然后重建 blockInfoMap
3. analyzeExternalInputs：Cube 块的外部输入 → V2C + 块间 C2C 依赖
4. analyzeExternalOutputs：Cube 块输出被 Vector 消费 → C2V 依赖
   （含 isAllTranspoesd 标记 + consumerYieldOp 记录转置的 yield 消费点）
5. analyzeInternalDeps（新增）：同 Cube 块内 matmul 链 → intra-block C2C 依赖
   （用 getSourceThroughCIntermediateOps 溯源 + operand 精确到输入槽位）
6. analyzeMemoryEffect：用别名分析 + 内存依赖图找隐式跨核内存依赖
7. deduplicateDependencies：五类依赖各自去重
```

最终产出五类依赖列表（V2C/C2V/块间C2C/块内C2C/MemDep），供 `InterCoreTransferAndSync` 消费。

---

## 五、面试要点

1. **为什么 `tensor.empty`/`linalg.fill` 不能作为依赖 source？**
   因为它们是"初始化"操作，本身不承载有意义的数据流，且可能被两个核都用到。把它们当作 source 会产生虚假依赖，导致插入不必要的同步/传输。

2. **为什么 iterArg 需要插入一个占位的 `arith.constant` 作为生产者块？**
   因为 `BlockArgument`（循环 iterArg）没有 defining op，无法作为生产者块的 anchor。插入一个常量 op 并打上核类型标记，可以充当"生产者块"的定位点。

3. **LCA（最近公共祖先）对齐解决什么问题？**
   解决生产者/消费者处于不同循环嵌套层级时的传输放置问题。传输必须在同一层级的两块之间进行，LCA 找到两者最近的公共层级，把块 id 对齐到该层级。

4. **内存依赖 vs 数据依赖的区别？**
   数据依赖是显式的 SSA use-def 链（value 从 A 流到 B）；内存依赖是隐式的（通过 load/store 到同一内存地址产生别名）。后者必须借助别名分析（`AliasAnalysis`）+ 内存依赖图才能发现，且需要插入 `PIPE_MTE2` 同步保证顺序。

5. **while 和 for 在 iterArg 索引上有什么差异？**
   `scf.for` 循环体的第 0 个 BlockArgument 是 induction variable，携带值从第 1 个开始，索引需要减 1 偏移；`scf.while` 的 before/after 区块参数与 `getInits()` 一一对应，无需偏移。这就是 `resolveNestedIterArgInitValue` 中对两者区别处理的原因。

6. **"混核携带变量"（init 与 yield 核类型不同）为什么要分三种情况？**
   因为跨核数据流的方向不同：两种核都在用时，init 值要先跨核传入、且 yield 核的用户需要生产者块（2a）；只有 init 核在用时，yield 回的值才是跨核流出的，需要在 yield 前插消费者块（2b）；只有 yield 核在用时，init 值跨核传入即可（2c）。方向判断错了会导致同步插反、数据竞争。

7. **转置折叠（isAllTranspoesd）的收益是什么？**
   matmul 结果若只被一个 transpose 消费且用户全是 Vector，C→V 传输的 fixpipe 可以用 NZ2DN 模式直接在硬件搬运时完成转置，Vector 核上的 transpose 计算被完全省掉——用免费的 DMA 排布变化换取一次完整的向量核计算。

8. **为什么要把 C2C 拆成"块间"和"块内"两类？**
   它们的插入点策略完全不同：块间 C2C（`c2cDependencies`）的生产/消费在两个不同 block，fixpipe 的 L1 buffer 要按 LCA 对齐到主循环层级（跨核流水线的一部分）；块内 C2C（`intraC2CDependencies`）的生产 matmul 和消费 matmul 在同一个 Cube block 内，L1 buffer 直接插在生产 matmul 之后，且需要 `operand` 指针精确区分消费端是 A/B 输入还是 init。混在一起会让 `InterCoreTransferAndSync` 无法用统一的插入点逻辑处理。

9. **`consumerYieldOp` 为什么在"转置折叠"场景也派上用场？**
   折叠后 `matmul → transpose → yield` 变成 `fixpipe → yield`，fixpipe 的 receive 值必须替换 yield 上的 operand（否则 yield 还引用着被删掉的 transpose）。由于 `analyzeExternalOutputs` 提前把 yield 记进了依赖的 `consumerYieldOp`，传输 pass 不需要重新分析就能精准替换。这和 iterArg yield 依赖用 `consumerYieldOp` 是同一套机制——"值跨核传输后，消费端是 yield 时都要记录 yield"。 
