# SplitDataflow 子 Pass 逐行讲解（一）：AddBlockIdForControlOps

> 源码文件：`third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/AddBlockIdForControlOps.cpp`
> 流水线位置：`SplitDataflowPass` 的 7 个步骤中的 **第 1 步**

---

## 一、这个 Pass 在做什么（一句话）

给 module 中所有**控制流算子**（`scf.for`、`scf.if`、`scf.while`，以及它们的终结符 `scf.yield` / `scf.condition`）**分配一个全局递增的 `ssbuffer.block_id`**，让后续的依赖分析、数据流拆分能统一用 block_id 来标识每一个"计算块"。

在 DynamicCVPipeline 的语境里，"block" 是一个被 `ssbuffer.block_id` 标记的**计算块单元**（可能运行在 Cube 核或 Vector 核上）。前面 `PlanComputeBlock` 已经给计算类算子分配了 block_id，但**控制流算子之前没有被分配**。这个 Pass 就是补上这块，使得后面 `DataDependencyAnalysis` 能正确处理带循环/分支的场景。

**本次更新要点**：新增了 `scf.while` 循环的支持（对应 Triton 中 `while` 语句生成的 IR），并统一处理了所有控制流终结符（`scf.yield` 和 `scf.condition`）。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `ssbuffer.block_id` | 一个 `IntegerAttr` 属性，标记 op 属于哪个计算块。常量定义在 `Common/Utils.h` 中：`kBlockId = "ssbuffer.block_id"` |
| `getAvailableBlockId(module)` | 返回当前 module 中**已经使用的最大 block_id + 1**（即下一个可用 id） |
| `setOpBlockId(op, id)` | 给 op 设置 `ssbuffer.block_id` 属性（定义在 `SplitDataflow/Utils.h`，实现在 `SplitDataflow/Utils.cpp`） |
| `scf.condition` | `scf.while` 的**前区块终结符**：operand 0 是循环条件（bool），其余 operand 是循环携带值。它是 while 独有的"类 yield"终结符 |

---

## 三、逐行讲解

### 3.1 头文件与命名空间（L23-L35）

```cpp
#include "ascend/include/DynamicCVPipeline/SplitDataflow/AddBlockIdForControlOps.h"  // 本 Pass 的声明
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"                          // CVPipeline 常量/工具
#include "ascend/include/DynamicCVPipeline/SplitDataflow/Utils.h"                   // setOpBlockId
#include "llvm/Support/Debug.h"                                                     // LLVM_DEBUG 调试宏
```

```cpp
using namespace mlir;
static constexpr const char *DEBUG_TYPE = "add-block-id-for-control-ops";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)
using namespace mlir::triton;
```

- `DEBUG_TYPE` 是调试 tag，只有在编译时开启 LLVM debug 模式（`LLVM_ENABLE_ASSERTIONS`）才会输出日志。

### 3.2 Pass 入口 runOnOperation（L38-L44）

```cpp
void AddBlockIdForControlOpsPass::runOnOperation() {
  ModuleOp module = getOperation();

  // 若 module 已被标记 fallback，则直接跳过
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }
```

- `hasFallbackAttr` 检查 module 上是否有 `triton_ascend.dynamic_cv_pipeline.rc` 属性。这是整个 DynamicCVPipeline 的**兜底机制**：一旦某个环节失败，就设置该属性，后续所有 pass 看到它就提前返回，最终回退到"不做 CV 流水线"的原始编译路径（见 `AddDynamicCVPipeline.cpp` 的 `restoreModuleFromBackup`）。

### 3.3 Step 1：确定起始 maxBlockId（L46-L49）

```cpp
  // Step 1: find the max block_id
  int maxBlockId = CVPipeline::getAvailableBlockId(module) - 1;
```

- `getAvailableBlockId(module)` 返回"下一个可用 id"（= 当前最大 id + 1），所以这里 `-1` 得到**当前最大 block_id**。
- 之后从这个值继续 `++`，保证新分配的控制流 block_id 不与已有的计算块 block_id 冲突。

### 3.4 Step 2：遍历并给控制流 op 分配 block_id（L51-L82）

```cpp
  module.walk([&](Operation *op) {
    // 跳过已经有 block_id 的 op
    if (op->getAttrOfType<IntegerAttr>(CVPipeline::kBlockId)) {
      return;
    }

    if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
      maxBlockId++;
      setOpBlockId(op, maxBlockId);
    }
```

- `module.walk` 是 MLIR 的**深度优先遍历**，按程序顺序访问每个 op。
- 第一处判断：如果 op 已经带有 `ssbuffer.block_id`（`IntegerAttr`），直接 `return`（跳过该 op，但 `walk` 仍会继续遍历其嵌套 region 内的其他 op），避免重复分配。
- 第二处判断：对 `scf::ForOp`、`scf::IfOp` **和 `scf::WhileOp`** 分配新的 block_id。控制流结构本身被当作一个独立的"块"。
- **更新点**：旧版只处理 `for/if`；新版加入 `while`，是为了配合上游 Triton 中 while 循环（如 FlashAttention 的某些变体、动态长度循环）生成的 `scf.while` IR。

```cpp
    Operation *parentOp = op->getParentOp();
    bool isControlTerminator =
        (isa<scf::YieldOp>(op) && isa<scf::IfOp, scf::WhileOp>(parentOp)) ||
        (isa<scf::ConditionOp>(op) && isa<scf::WhileOp>(parentOp));
    if (isControlTerminator) {
      auto parentBlockIdOpt = CVPipeline::getOpBlockId(parentOp);

      int terminatorBlockId;
      if (parentBlockIdOpt) {
        terminatorBlockId = *parentBlockIdOpt;
      } else {
        maxBlockId++;
        terminatorBlockId = maxBlockId;
      }
      setOpBlockId(op, terminatorBlockId);
    }
  });
```

这段是对**控制流终结符**的统一处理（旧版只处理 `scf.if` 内的 yield，变量名是 `ifBlockIdOpt`/`yieldBlockId`；新版泛化为 `parentBlockIdOpt`/`terminatorBlockId`）：

| 终结符 | 父 op | 说明 |
|---|---|---|
| `scf.yield` | `scf.if` | if 的结果返回 |
| `scf.yield` | `scf.while` | while 的 after 区块结果返回 |
| `scf.condition` | `scf.while` | while 的 before 区块条件判定 + 携带值转发 |

逻辑：先看父 op 是否已有 block_id：
- 有 → 直接**复用父 op 的 block_id**（保证终结符与其控制流 op 属于同一块）；
- 没有 → 分配一个新的递增 id。

**为什么终结符要特殊处理？** 因为终结符本身不"计算"，但它的操作数来源（yield/condition 转发的 value）可能定义在另一个核的块中。给终结符分配与父控制流一致的 block_id，是为了在依赖分析中能正确追溯终结符与控制流的归属关系——`DataDependencyAnalysis` 中的 `analyzeExternalOutputs` 会用 `getCoreTypeWithIndex(user, outputIndex)` 读取终结符上的 core_type 来判断跨核依赖。

### 3.5 Pass 创建函数（L87-L96）

```cpp
std::unique_ptr<OperationPass<ModuleOp>> createAddBlockIdForControlOpsPass() {
  return std::make_unique<AddBlockIdForControlOpsPass>();
}
```

- 标准的 MLIR Pass 工厂函数，返回一个 `unique_ptr<OperationPass<ModuleOp>>`。
- 在 `SplitDataflow.cpp` 的 `runOnOperation` 里通过 `createAddBlockIdForControlOpsPass()` 被加入 `OpPassManager`。

---

## 四、算法流程总结

```
1. 若 module 有 fallback 属性 → 直接返回
2. maxBlockId = 当前最大 block_id
3. 深度优先遍历所有 op：
   a. 已有 block_id → 跳过
   b. 是 scf.for / scf.if / scf.while → 分配 ++maxBlockId
   c. 是控制流终结符（yield@if/while 或 condition@while）→ 复用父 op 的 id，或分配新 id
```

复杂度：O(N)，N 为 module 中 op 总数。

---

## 五、面试要点

1. **为什么要单独给控制流 op 分配 block_id？**
   因为依赖分析和后续的核间传输/同步需要把"生产者块"和"消费者块"用 block_id 唯一标识。控制流（循环、分支）会改变数据流的结构，必须也被纳入块划分，否则跨循环/跨分支的数据依赖无法被正确分析。

2. **`scf.condition` 和 `scf.yield` 有什么区别？为什么都要处理？**
   `scf.while` 有两个区块：before 区块以 `scf.condition` 结尾（判定条件 + 携带值转发给 after），after 区块以 `scf.yield` 结尾（结果返回/回写携带值）。两个终结符都可能"转发另一个核算出的值"，因此都需要 block_id 以便依赖分析正确归类。这是支持 while 循环后新引入的处理。

3. **为什么终结符要特殊处理（而不是统一分配新 id）？**
   终结符不产生计算，它只是"搬运"父控制流的语义。复用父 op 的 block_id 保证 yield/condition 与其控制流在同一块内，依赖分析才不会把一次循环内部转发误判为跨核依赖。

4. **fallback 机制的作用是什么？**
   这是一种保守的健壮性设计：CV 流水线是"锦上添花"的优化，任何一步失败都不应导致编译失败，而是回退到不启用流水线的路径，保证功能正确性优先。
