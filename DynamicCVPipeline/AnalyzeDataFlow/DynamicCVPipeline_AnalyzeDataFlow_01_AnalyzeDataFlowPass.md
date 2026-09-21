# DynamicCVPipeline 主 Pass 专题：数据流分析（01）

> 覆盖 `AnalyzeDataFlowPass`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L81）
>
> 源码文件：
> - [`AnalyzeDataFlow.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AnalyzeDataFlow.cpp)（编排主 Pass）
> - [`AnalyzeName.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AnalyzeDataFlow/AnalyzeName.cpp)
> - [`AnalyzeScope.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AnalyzeDataFlow/AnalyzeScope.cpp)
> - [`AnalyzeArgs.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AnalyzeDataFlow/AnalyzeArgs.cpp)
> - [`AnalyzeFlag.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AnalyzeDataFlow/AnalyzeFlag.cpp)
> - [`AnalyzeCubeContolFLowInputChain.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AnalyzeDataFlow/AnalyzeCubeContolFLowInputChain.cpp)
> - [`AnalyzeWhileConditionArgs.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AnalyzeDataFlow/AnalyzeWhileConditionArgs.cpp)
>
> 流水线位置：`SplitDataflow`（L80）之后、`AllocMultiCache`（L82）之前。此时数据流已按块拆分，本 Pass 做**最终可行性审计**：任何一项检查不通过即设置 fallback 属性，后续所有 Pass 直接返回。

---

## 一、这个专题在做什么

前面的 `PlanComputeBlock`（分块）、`ComputeBlockOpt`（块优化）、`SplitDataflow`（核间数据流拆分）已经把计算块和数据流准备好了。但**不是所有 kernel 都适合走 Dynamic CV 流水线**——有些 kernel 的结构（函数名、主循环形态、迭代参数、同步 flag、控制流输入链、while 条件）会让后续的缓存分配、控制流条件生成、内存计算分离无法正确执行。

`AnalyzeDataFlow` 是**流水线启动后的第一道审计闸门**：6 个子 Pass 分别检查一种不兼容场景，任一命中就 `setFallbackAttr` 回退标准编译，避免后续 Pass 在畸形 IR 上白跑。

**编排顺序**（AnalyzeDataFlow.cpp L46-L56）：

```cpp
pm.addPass(createAnalyzeNamePass());                    // ① 函数名黑名单
pm.addPass(createAnalyzeScopePass());                   // ② scope 主循环形态
pm.addPass(createAnalyzeArgsPass());                    // ③ 迭代参数跨块使用
pm.addPass(createAnalyzeFlagPass());                    // ④ 同步 flag 数量
pm.addPass(createAnalyzeCubeContolFLowInputChainPass()); // ⑤ CUBE 控制流输入链
pm.addPass(createAnalyzeWhileConditionArgsPass());       // ⑥ while 条件 tensor 依赖

if (failed(runPipeline(pm, module))) {                  // 任一子 pass 失败
  if (!CVPipeline::hasFallbackAttr(module))
    CVPipeline::setFallbackAttr(module, ERRCODE_FAILED); // 统一兜底
}
```

**共同模式**：6 个子 Pass 都是 **read-only 检查 + 置 fallback 属性**，不重写 IR。通过与否不产生中间产物，只影响"要不要继续走 CV 流水线"。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `setFallbackAttr(module, errcode)` | 设置 fallback 属性，下游所有 Pass 在入口检测到后直接 return。错误码区分回退原因：`ERRCODE_IGNORED`（kernel 不适合）、`ERRCODE_FAILED`（通用失败）、`ERRCODE_TUPLE_PRELOAD_FAILED`（tuple 预加载失败） |
| `scope::ScopeOp` + `TCoreType` | 昇腾 scope 方言：一个 scope 标记一块核区域。`TCoreType::VECTOR` = Vector scope，`TCoreType::CUBE` = Cube scope |
| `kTransferId`（`ssbuffer.transfer_id`） | `SplitDataflow` 注入传输 op 时打的标记，本 Pass 用它识别"真正的跨核数据传输 op" |
| `kAddFromMatmul`（`ssbuffer.add_from_matmul`） | matmul 累加标记：`to_tensor` 的用户带此标记表示数据来自 L0C 累加器复用，**不算真实的 c→v 数据交互** |
| `kMainLoop`（`ssbuffer.main_loop`） | 主循环 id：`PlanComputeBlock` 给主循环打的标记，同一 id 的 for/while 属于同一个主循环 |
| `isMainLoopOp` / `MainLoop` | 主循环判定工具（`Common/Utils.h`）：`getIterArgs()` 拿迭代参数、`getBody()` 拿循环体 |
| `BufferCountManager` | 缓冲计数工具：`IntraCore`（核内缓冲）/ `InterCore`（核间缓冲）数量 |
| `TensorArgBlockInfo` | 记录某个 tensor iter_arg 被哪些 block_id 使用、第一个使用的 block_id |
| `static_flag_id` | 同步 op（`SyncBlockSet`/`SyncBlockWait`）的 flag 编号属性，有效范围 0-14 |
| `isVectorOnlyOp` | 判断 op 是否 Vector 专属（`Common/Utils.h`） |
| `scf.while` 结构 | `before` region 的 `scf.condition` 带 condition + 前向参数；`after` region 的 `scf.yield` 回传迭代值 |

---

## 三、逐行讲解

### 3.1 AnalyzeNamePass：函数名黑名单（AnalyzeName.cpp L77-L92）

```cpp
static constexpr llvm::StringLiteral interceptrFunc[]{
    "kernel_sdpa_bwd_kv", "flash_varlen_fwd_kernel", "_sparse_decode_kernel",
    "_sparse_decode_model1_kernel", "sparse_flash_attention_grad_kernel",
    "flex_attention_backward_dkdv_kernel",
    "flex_attention_backward_dkdv_kernel_tasklist", "chunkwise_bwd_kernel_dhg",
};

static LogicalResult verifyFuncNames(ModuleOp module) {
  bool intercepted = false;
  module.walk([&](func::FuncOp funcOp) -> WalkResult {
    if (!llvm::is_contained(interceptrFunc, funcOp.getSymName()))
      return WalkResult::advance();        // 不在黑名单，继续
    intercepted = true;
    return WalkResult::interrupt();        // 命中黑名单，中断
  });
  return intercepted ? failure() : success();
}

void AnalyzeNamePass::runOnOperation() {
  ...
  if (failed(verifyFuncNames(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }
}
```

**要点**：8 个黑名单 kernel（sdpa bwd、flash varlen、sparse decode、flex attention bwd 等）经过前期验证**不适合 CV 流水线**（多为已手工优化过的 kernel），直接 `ERRCODE_IGNORED` 回退。

### 3.2 AnalyzeScopePass：scope 主循环形态审计（AnalyzeScope.cpp L199-L227）

三道检查依次执行，任一失败即 `ERRCODE_IGNORED`：

**检查 1：必须有主循环**（L200-L212）——模块内没有任何 `isMainLoopOp` → 没有多迭代循环，流水线无意义。

**检查 2：Vector scope 主循环必须有真实 CV 交互 `checkVecScopeMainLoop`（L84-L136）**：

```cpp
static bool checkTransferInteraction(mlir::Operation *op) {
  bool hasCVInteraction = false;
  // v→c 数据交互：hivm.hir.copy（Vector 传数据给 Cube）
  if (isa<hivm::CopyOp>(op)) hasCVInteraction = true;
  // c→v 数据交互：bufferization.to_tensor（Cube 结果转 tensor 给 Vector）
  if (isa<bufferization::ToTensorOp>(op)) {
    for (Operation *user : op->getUsers()) {
      if (!user->hasAttr(CVPipeline::kAddFromMatmul)) {
        hasCVInteraction = true; break;      // 非 matmul 累加 → 真交互
      }
      for (Operation *userUser : user->getUsers()) {
        if (!isa<scf::YieldOp>(userUser)) {
          hasCVInteraction = true; break;    // 累加结果有真实消费 → 真交互
        }
      }
    }
  }
  return hasCVInteraction;
}
```

- 遍历每个 **Vector scope**（`isVectorScope` 判 `TCoreType::VECTOR`，L49-L56）里的每个主循环；
- 在循环内找带 `kTransferId` 的 op（`SplitDataflow` 注入的传输 op，L108）；
- **`checkTransferInteraction` 判定"真交互"**：`CopyOp`（v→c）直接算；`ToTensorOp`（c→v）要排除 `kAddFromMatmul` 累加场景（L0C 复用不算搬数据），且累加结果若被真实消费（用户不是 yield）才算；
- **只要有一个 Vector 主循环没有任何真交互** → `allMainLoopsSatisfy=false` → 整体失败。

**检查 3：主循环不能是"单向 copy/fixpipe" `isMainLoopOnlyCopyOrFixpipe`（L147-L197）**：

```cpp
// main_loop id → (countCopy, countFixpipe)
llvm::DenseMap<int, std::pair<int, int>> idToCounts;
module.walk([&](Operation *op) -> WalkResult {
  if (!isa<scf::ForOp, scf::WhileOp>(op)) return advance();
  auto mainLoopAttr = op->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
  if (!mainLoopAttr) return advance();          // 只统计有 main_loop 标记的循环
  int id = mainLoopAttr.getInt();
  auto &counts = idToCounts[id];
  op->walk([&](Operation *innerOp) {
    if (innerOp == op || isa<scf::YieldOp>(innerOp)) return advance(); // 跳过自身和 yield
    if (isa<hivm::CopyOp>(innerOp)) ++counts.first;
    else if (isa<hivm::FixpipeOp>(innerOp)) ++counts.second;
    return advance();
  });
  return advance();
});
// 只有当每个 main_loop id 都只有 copy 或只有 fixpipe（另一种计数为 0）才回退
for (const auto &entry : idToCounts)
  if (entry.second.first != 0 && entry.second.second != 0)
    return false;                               // 至少一个 id 双向 → 流水线仍适用
return true;
```

**要点**：`CopyOp` 典型出现在 Vector scope 主循环（v→c 搬运），`FixpipeOp` 典型出现在 Cube scope 主循环（FIXPIPE 量化）。如果每个主循环 id 都**只有单向交互**，说明 CUBE/Vector 之间没有真实的数据往返依赖，流水线无收益；只要有一个 id 同时有 copy 和 fixpipe（双向往返），CV 流水线依然适用。

### 3.3 AnalyzeArgsPass：迭代参数跨块使用审计（AnalyzeArgs.cpp L200-L230）

```cpp
void AnalyzeArgsPass::runOnOperation() {
  ...
  BufferCountManager bufferCountMgr(module);
  int intraBufNum = bufferCountMgr.getBufferCountByType(DepType::IntraCore);
  int interBufNum = bufferCountMgr.getBufferCountByType(DepType::InterCore);

  // 检查 1：update-before-use（无门控，直接查）
  if (checkUpdateBeforeUseInMainLoop(module)) {
    CVPipeline::setFallbackAttr(module, ERRCODE_IGNORED);
    return;
  }

  // 检查 2：tensor iter_arg 跨块使用（仅特定缓冲数量组合触发）
  if (intraBufNum == 3 && interBufNum == 2 && checkTensorArgsInMainLoop(module)) {
    CVPipeline::setFallbackAttr(module, ERRCODE_TUPLE_PRELOAD_FAILED);
    return;
  }
}
```

**检查 1：`checkUpdateBeforeUse`（L135-L167）——"先更新后使用且跨块"**：

```cpp
// 对每个 tensor iter_arg：
Operation *defOp = CVPipeline::getLoopCarriedDefOp(iterArg, body);  // 更新定义 op
auto defBlockId = CVPipeline::getOpBlockId(defOp);
for (Operation *user : iterArg.getUsers()) {
  Operation *directChild = body->findAncestorOpInBlock(*user);     // 循环体内的直接使用
  if (!directChild) continue;
  if (!defOp->isBeforeInBlock(directChild)) continue;  // 使用必须在更新之后
  if (CVPipeline::getOpBlockId(directChild) != defBlockId)
    return true;                                        // 跨块且更新在前 → 冲突
}
```

**语义**：tensor 迭代参数在某次迭代里被"更新定义 op"（yield 前重建）先改写，随后又被**另一个 block_id** 的 op 读取（先更新后使用）。此时旧值已被覆盖，跨块读取拿到的是新值，会导致依赖顺序错乱——流水线无法处理，回退。

**检查 2：`checkTensorArgsInMainLoop`（L171-L184）——tensor iter_arg 跨块使用**（仅 `intraBufNum==3 && interBufNum==2` 的特定缓冲场景触发）：

- `collectTensorArgBlockInfo`（L61-L87）：对循环体内每个带 block_id 的 op，扫描操作数，用 `getTensorIterArgIndex` 找到对应 tensor iter_arg 下标，记录使用它的 block_id 集合 + 第一个 block_id；
- `checkMultiBlockUse`（L90-L97）：任一 tensor iter_arg 被 **≥2 个 block_id** 使用 → 跨块共享（tuple 预加载无法处理）；
- `checkUseUpdateMismatch`（L100-L122）：yield 的更新定义 op 的 block_id ≠ 该 iter_arg 的 `firstBlockId` → 更新和使用不在同一块。

**要点**：这个检查是**门控的**——只在"IntraCore=3、InterCore=2"这个具体缓冲布局下才启用。原因是该布局对应 `AllocMultiCache` 的 tuple 预加载方案，对 iter_arg 有严格的"单块使用"要求；其他布局不受此约束。

### 3.4 AnalyzeFlagPass：同步 flag 数量审计（AnalyzeFlag.cpp L50-L94）

```cpp
static bool checkFlagIdValidity(ModuleOp module) {
  module.walk([&](Operation *op) -> WalkResult {
    if (!isa<hivm::SyncBlockSetOp>(op) && !isa<hivm::SyncBlockWaitOp>(op))
      return WalkResult::advance();               // 只看同步 op
    if (auto intAttr = op->getAttrOfType<IntegerAttr>(syncFlagIdAttr)) {
      int flag = static_cast<int>(intAttr.getInt());
      if (flag < 0 || flag > 14) {                // flag 超出有效范围
        shouldReturn = true;
        invalidFlagNum++;
      }
    }
    return WalkResult::advance();
  });
  return shouldReturn;
}
```

**要点**：`static_flag_id` 有效范围 **0-14**（15 个 flag 位）。`SplitDataflow` 拆分核间传输时按 flag 分配同步槽位，若原 kernel 已有超出范围的 flag（说明传输/同步数量超过硬件 flag 容量），后续分配必然冲突 → `ERRCODE_IGNORED` 回退。

### 3.5 AnalyzeCubeControlFlowInputChainPass：CUBE 控制流输入链审计（AnalyzeCubeContolFLowInputChain.cpp L161-L176）

**背景**：Cube 核上的控制流（if/for/while）条件如果依赖 **Vector 核计算的标量**，意味着 Cube 要等待 Vector 的结果才能决定分支——跨核控制流耦合，无法流水。所以审计"CUBE scope 内控制流 op 的条件输入链上是否有 Vector 专属 op"。

```cpp
static bool hasIncompatibleOpForCondition(Value val, DenseSet<Value> &visited) {
  if (visited.contains(val)) return false;        // 环保护
  visited.insert(val);
  Operation *defOp = val.getDefiningOp();
  if (!defOp) return false;
  if (isVectorOnlyOp(defOp)) {                    // 命中 Vector 专属 op
    LDBG("Fallback reason: incompatible upstream op for control flow: " << *defOp);
    return true;
  }
  return hasIncompatibleUpstream(defOp->getOperands(), visited);  // 继续向上回溯
}

static bool checkControlFlowOpInputs(Operation *cfOp) {
  // 按控制流类型收集"影响条件"的标量操作数：
  //   scf.if   → condition
  //   scf.for  → lowerBound/upperBound/step
  //   scf.while→ 所有操作数（while 条件可能受任意循环承载变量影响，保守全收）
  //   其他     → 所有操作数（保守）
  ...
  return hasIncompatibleUpstream(scalarOperands, visited);
}

bool checkCubeControlFlowInputChain(ModuleOp module) {
  module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    if (!isCubeScope(scopeOp)) return advance();   // 只看 CUBE scope
    scopeOp.walk([&](Operation *op) -> WalkResult {
      if (!isControlFlowOp(op)) return advance();  // 只看 scf 控制流
      if (checkControlFlowOpInputs(op)) { shouldReturn = true; return interrupt(); }
      return advance();
    });
    ...
  });
  return shouldReturn;
}
```

**要点**：
- **只有 CUBE scope** 里的控制流受检（Vector scope 里条件依赖 Vector 自己没问题）；
- 条件操作数的**选择**因 op 而异：if 只查 condition，for 查界值，while 和其他 op 保守全收；
- 沿 use-def 链**递归回溯**（`visited` 防环），只要链上出现一个 `isVectorOnlyOp` 就判定不兼容 → `ERRCODE_IGNORED`。

### 3.6 AnalyzeWhileConditionArgsPass：while 条件 tensor 依赖审计（AnalyzeWhileConditionArgs.cpp L145-L165）

**背景**：主循环是 `scf.while` 时，循环条件若**传递依赖循环内部产生的 tensor**，说明每次迭代都要把 tensor 数据搬出来算标量条件——跨核搬运高频发生，流水线无法覆盖。审计从 `scf.condition` 的 condition 反向回溯。

```cpp
static bool conditionDependsOnTensor(scf::WhileOp whileOp) {
  SmallVector<Value> worklist{whileOp.getConditionOp().getCondition()};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second) continue;       // 环保护

    if (isNonScalarProducedInside(whileOp, value)) {   // ① tensor 且产自循环内部
      return true;
    }
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      expandBlockArgument(whileOp, blockArg, worklist); // ② 回溯到迭代更新点
    else
      expandOpResult(whileOp, cast<OpResult>(value), worklist); // ③ 回溯到操作数
  }
  return false;
}
```

**回溯展开规则**：
- `expandBlockArgument`（L55-L69）：before-region 参数 → after 侧 `scf.yield` 同下标操作数（迭代更新值）；after-region 参数 → `scf.condition` 的 args 同下标（前向转发值）。**外层作用域的参数不展开**（不在 while 的更新过程内）；
- `expandOpResult`（L75-L92）：op 的所有操作数 + **region terminator 的操作数**（含嵌套 if/for/while 的结果，保守追踪）；
- `isNonScalarProducedInside`（L101-L113）：值必须是 **tensor 且定义在 while 内部**（含嵌套循环的迭代参数）；循环外的 tensor 不构成问题（其计算不参与克隆，只有提取出的标量进条件链）。

**要点**：命中即 `ERRCODE_IGNORED`。判断的核心是"**是否在循环内部产生 tensor**"——循环外的 tensor 是安全的（只标量参与条件更新）。

---

## 四、流程总结

### 4.1 审计闸门流水

```
AnalyzeDataFlowPass
  ├─► ① AnalyzeNamePass        函数名命中黑名单? ──失败──► ERRCODE_IGNORED
  │       │ 通过
  │       ▼
  ├─► ② AnalyzeScopePass       无主循环 / Vector 主循环无真 CV 交互 /
  │                             所有主循环单向 copy/fixpipe? ──失败──► ERRCODE_IGNORED
  │       │ 通过
  │       ▼
  ├─► ③ AnalyzeArgsPass        update-before-use? ──► ERRCODE_IGNORED
  │       │                    缓冲门控(3,2) + iter_arg 跨块? ──► ERRCODE_TUPLE_PRELOAD_FAILED
  │       │ 通过
  │       ▼
  ├─► ④ AnalyzeFlagPass        static_flag_id 超出 0-14? ──► ERRCODE_IGNORED
  │       │ 通过
  │       ▼
  ├─► ⑤ AnalyzeCubeControlFlowInputChainPass
  │                             CUBE 控制流条件链含 Vector 专属 op? ──► ERRCODE_IGNORED
  │       │ 通过
  │       ▼
  └─► ⑥ AnalyzeWhileConditionArgsPass
                              主 while 条件依赖循环内 tensor? ──► ERRCODE_IGNORED
              │ 通过
              ▼
        继续 AllocMultiCache
```

### 4.2 检查矩阵

| 子 Pass | 检查对象 | 回退错误码 | 检查性质 |
|---|---|---|---|
| AnalyzeName | 函数名 | `ERRCODE_IGNORED` | 已知不适合的 kernel 黑名单 |
| AnalyzeScope | Vector scope 主循环的 CV 交互 + 单向传输 | `ERRCODE_IGNORED` | 主循环形态是否值得流水 |
| AnalyzeArgs | 迭代参数跨块使用/先更新后使用 | `ERRCODE_IGNORED` / `ERRCODE_TUPLE_PRELOAD_FAILED` | 数据流跨块冲突（门控） |
| AnalyzeFlag | 同步 flag 编号范围 | `ERRCODE_IGNORED` | 同步资源容量 |
| AnalyzeCubeControlFlowInputChain | CUBE 控制流条件输入链 | `ERRCODE_IGNORED` | 跨核控制流耦合 |
| AnalyzeWhileConditionArgs | while 条件 tensor 依赖 | `ERRCODE_IGNORED` | 条件更新跨核搬运 |

### 4.3 与前后 Pass 的协作

- **前置依赖**：检查依赖 `SplitDataflow` 注入的 `kTransferId`（②）、`PlanComputeBlock` 打的 `kMainLoop`（②③⑥）、块划分产生的 block_id（③）；
- **产出**：无 IR 变更，只有"通过"（继续）或"置 fallback 属性"（后续全部短路）；
- **后置**：`AllocMultiCache`（L82）等所有 Pass 入口检查 `hasFallbackAttr`，命中直接 return。

---

## 五、面试要点

1. **AnalyzeDataFlow 的目标是什么？** 它是 CV 流水线的**可行性审计闸门**：6 个子 Pass 各自检测一种"后续 Pass 无法处理"的场景（kernel 黑名单、主循环形态、迭代参数跨块、flag 溢出、CUBE 控制流依赖 Vector、while 条件依赖 tensor），命中即 fallback，避免后续在畸形 IR 上白跑。

2. **6 个子 Pass 顺序有什么讲究？** 从外到内、由粗到细：先看函数名（最粗）、再看 scope/主循环结构、再查迭代参数和 flag、最后深查控制流输入链和 while 条件（最细的 use-def 追溯）。

3. **回退错误码如何区分？** `ERRCODE_IGNORED` 表示"该 kernel 结构不适合 CV 流水线"（大多数）；`ERRCODE_FAILED` 是编排兜底；`ERRCODE_TUPLE_PRELOAD_FAILED` 特指 tuple 预加载场景下 iter_arg 跨块使用。错误码用于定位回退原因。

4. **"真实 CV 交互"怎么判定？** 只认 `SplitDataflow` 注入的带 `kTransferId` 的传输 op：`CopyOp`（v→c）直接算；`ToTensorOp`（c→v）要排除 `kAddFromMatmul`（L0C 累加复用不算搬数据），且累加结果被真实消费才算。

5. **单向 copy/fixpipe 为什么回退？** `CopyOp`（v→c）和 `FixpipeOp`（c→v FIXPIPE 通道）是相反方向的数据交互。若每个主循环 id 都只有一种（单向），说明 CUBE/Vector 间没有往返依赖，流水线无重叠收益；只要有 id 同时具备两者（双向往返），流水线适用。

6. **AnalyzeArgs 的门控是什么意思？** 检查 2（iter_arg 跨块）只在 `intraBufNum==3 && interBufNum==2` 这个特定缓冲布局下启用——该布局对应 `AllocMultiCache` 的 tuple 预加载方案，对 iter_arg 有严格单块要求；其他布局不受约束。而检查 1（update-before-use）无门控。

7. **update-before-use 为什么危险？** tensor 迭代参数在某次迭代被更新定义 op 改写后，又被**另一个 block_id** 读取。旧值已被覆盖，跨块读取拿到新值，依赖顺序错乱，流水线无法正确同步。

8. **为什么 CUBE 控制流条件不能依赖 Vector op？** Cube 核要等 Vector 核算出标量才能决定分支，跨核控制流耦合（每次迭代都要等），无法流水。所以从控制流条件操作数沿 use-def 回溯，链上出现 `isVectorOnlyOp` 就回退。while 保守全收操作数（条件可能受任意循环承载变量影响）。

9. **while 条件 tensor 依赖如何追溯？** 从 `scf.condition` 的 condition 反向 use-def 回溯：before 参数 → after yield 同下标值；after 参数 → condition args 同下标值；op 结果 → 操作数 + region terminator 操作数（嵌套控制流保守追踪）。关键是 `isNonScalarProducedInside`——只有**循环内部产生**的 tensor 才致命，循环外的不算（其计算不参与克隆）。

10. **本 Pass 与 PreCheckAvailable 的区别？** `PreCheckAvailable` 在**流水线最前**做入口检查（输入形态）；`AnalyzeDataFlow` 在 `SplitDataflow` 之后做**数据流审计**（依赖拆分后的产物：transfer_id、main_loop id、block_id、flag）。前者看"能不能进来"，后者看"拆完能不能继续"。

11. **错误码 + fallback 的机制如何保证一致性？** 任一子 Pass 置 fallback 后，编排层 `runPipeline` 失败兜底 `ERRCODE_FAILED`（若未被置过）；后续 `AllocMultiCache` 等所有 Pass 入口 `hasFallbackAttr` 直接 return，保证回退路径干净、可追溯。
