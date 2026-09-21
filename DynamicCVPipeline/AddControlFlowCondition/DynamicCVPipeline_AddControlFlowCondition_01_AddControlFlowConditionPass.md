# DynamicCVPipeline 主 Pass 专题：控制流条件注入（01）

> 覆盖 `AddControlFlowConditionPass`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L83）
>
> 源码文件：
> - [`AddControlFlowCondition.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition.cpp)（编排主 Pass）
> - [`CloneOps.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/CloneOps.cpp)（Step 0：块内 op 去共享）
> - [`ProcessArgs.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/ProcessArgs.cpp)（Step 1：共享 iter_arg 拆分）
> - [`CreateIfOps.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/CreateIfOps.cpp)（Step 2：每 block 建 if）
> - [`InitDependentMap.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/InitDependentMap.cpp)（Step 3：依赖映射 + if DAG）
> - [`UpdateLoopOps.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopOps.cpp)（Step 4：扩展循环 + PIPE_S）
> - [`UpdateConditionInfo.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.cpp)（Step 5：构造 if 条件）
> - [`UpdateLoopIterTimes.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimes.cpp)（Step 6：扩展迭代次数）
>
> 流水线位置：`AllocMultiCache`（L82）之后、`SeparateMemoryFromCompute`（L84）之前。此时多缓冲已分配、轮询结构已注入，本 Pass 把"缓冲是否可用"的**轮询条件**真正落地为 `scf.if` 上的可执行表达式，并打通 CUBE/Vector 两侧的流水线握手。

---

## 一、这个专题在做什么

`AllocMultiCache` 分配了 N 块缓冲、在每个 block 的运算外包了一层 `scf.if %true`（恒真占位），但此刻**所有 if 的条件都还是 `true`**，多缓冲轮换只是"形式上"的——producer 不会因为缓冲满而等待，consumer 也不会因为数据未到而阻塞。整个流水线随时可能踩踏缓冲。

`AddControlFlowCondition` 的职责就是**把占位 if 变成真正的同步控制流**：

1. **去共享**：保证每个 block 拥有独立的 op 和独立的状态变量（不共享），这是"每个 block 单独控流"的前提；
2. **建 if**：每个 block 一个 `scf.if`，全部运算包进 then 分支，else 分支用 loop iter_arg 作默认值兜底；
3. **建依赖**：从 `ssbuffer.cross_deps` / `ssbuffer.intra_deps` 属性还原依赖图，构建 if 块 DAG，检测循环，收集 `flowOpt` 流水线重叠候选；
4. **扩展循环**：给主循环追加三类 extra iter_arg——**block 计数器**（跟踪每 block 当前迭代）、**核内依赖计数**（跟踪每依赖组缓冲占用）、**tensor iter_arg 计数**（跟踪跨迭代张量依赖）；并插入 `PIPE_S` 核间同步（flag 15）；
5. **算条件**：基于计数变量构造 if 条件——核间用 SSBuffer 指针数组计数（输入 `loaded>0`、输出 `loaded<producer_num`），核内用 loop iter_arg 计数（输入 `>0`、输出 `<producer_num`），tensor 迭代依赖用 `==1/==0` 判断，全部 And 合并后重建 if；
6. **扩迭代**：按依赖距离 `consumer_idx - producer_idx + 1` 计算所需缓冲数，用分数交叉相乘比较取最大值，把 for 循环迭代次数扩展为 `ceil(iter * requiredBuffers / x) + ifCount`，保证依赖有足够缓冲空间。

**6 个子 Pass 顺序执行，任意失败统一 fallback**（AddControlFlowCondition.cpp L138-L143）。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `ControlFlowConditionInfo` | 全局共享状态（`AddControlFlowCondition.h`）：跨核/核内依赖映射、块计数索引、条件索引、if DAG、flowOpt 对、tensor 迭代依赖等，由主 Pass 创建并指针传给各子 Pass |
| `kIf`（`ssbuffer.if`） | `CreateIfOps` 给每个 block_id 生成的 if 标记，值为 block_id；此后所有依赖分析都以"if 块"为单位 |
| `kBlockId`（`ssbuffer.block_id`） | 块 id，标明每个 op 属于哪个 compute block，是块归属的权威属性 |
| `kClone`（`ssbuffer.clone`） | 克隆 op 标记，值为被克隆的原始 block_id，供 `cleanupClonedOps` 识别克隆产生的冗余 op |
| `kArg`（`ssbuffer.arg`）/ `kWhileArg` | 标记由共享 arg 克隆而来的计算链，标明源 arg 索引，供调试与一致性校验 |
| `IfBlockDAG` | 块级有向无环图：`producerIf -> consumerIf`，边带 `IfBlockDepKind`（CrossCore/IntraCore），用于检测循环与收集 flowOpt 候选 |
| `FlowOpt` | DAG 深度遍历：起点（入度 0）出发 DFS，深度为 3 的节点与起点构成 `flowOptIfOpPairs[target]=source` 重叠对；只有 `crossCoreBufferCount > CROSS_CORE_BUFFER_COUNT_THRESHOLD` 且 `intraCoreBufferCount > INTRA_CORE_BUFFER_COUNT_THRESHOLD` 才启用 |
| `PIPE_S` / `kPipeSFlagId=15` | 流水线预留同步 flag id（15），`UpdateLoopOps` 给主循环插入 `sync_block_wait/set` 对，保证 CUBE/Vector 流水线启动前对齐 |
| `kVectorFirst` | 主循环上的属性，决定 PIPE_S 的 wait/set 位置：vector_first 时 Vector 侧先启动，CUBE 侧循环外 SET、循环内 WAIT/SET |
| `BlockCounter` | 每个 block_id 对应一个 loop iter_arg（for 用 lowerBound 初值，while 用独立 `constant 0`），跟踪本 block 当前迭代次数 |
| `blockCounters` / `cntArgs` | `info->blockCounters[loopOp]` = 计数器在 iter_arg 中的索引列表；`info->cntArgs[ifOp]` = 该 if 用到的计数器 iter_arg |
| `innerDepConds` | 核内依赖计数变量索引列表，`UpdateConditionInfo` 据此把依赖组映射到 loop iter_arg |
| `tensorIterArgDepsMap` | 记录 tensor 型 iter_arg 的跨迭代依赖：`TensorIterArgIfOpRelation{iterArg, producer, consumers}` |
| `tensorIterArgIndicesMap` | 每个 tensor iter_arg 消费 if 对应的新 iter_arg 索引列表（一 consumer 一计数变量） |
| `VarUpdateType` | 控制变量更新类型：`DEC`（输入消费后 -1）、`INC`（输出生产后 +1），then 分支 yield 前施加 |
| `whileBlockArgMap` | while 专用：`whileOp -> block_id -> {new_arg_idx: old_arg_idx}`，把 before 区 condition 依赖的 per-block 参数映射到 after 区新 arg |
| `SSBuffer` | 核间共享内存缓冲（地址空间基址 + 偏移）：Vector 0 从地址 0 开始、Vector 1 从 1024 开始，每组间隔 4 字节存 i32 计数 |
| `VECTOR_SSBUF_OFFSET=1024` / `VALUE_SSBUF_OFFSET=4` | Vector 1 的 SSBuffer 基址偏移 1024；每个计数槽间隔 4 字节 |
| 缓冲因子 | `requiredBuffers = consumer_idx - producer_idx + 1`（跨核 runFirst 时再 -1）；最终 `new_iter = ceil(iter * requiredBuffers / x) + ifCount`，其中 `x = producerOps.size()` |
| 分数交叉相乘 | 比较 `requiredBuffers/x` 与 `max/maxX` 用 `requiredBuffers * maxX > max * x`，避免浮点精度问题 |
| `CROSS_CORE_BUFFER_COUNT_THRESHOLD` / `INTRA_CORE_BUFFER_COUNT_THRESHOLD` | 缓冲数阈值，超过才收集 flowOpt 对（InitDependentMap L749-L750） |
| `ERRCODE_FAILED` / `ERRCODE_IGNORED` | fallback 错误码：条件构造失败 → FAILED；DAG 有环（如 flowOpt 与跨核依赖成环）→ IGNORED |

---

## 三、逐行讲解

### 3.1 前置校验：verifyControlFlowPrerequisites（AddControlFlowCondition.cpp L48-L66）

```cpp
static LogicalResult verifyControlFlowPrerequisites(ModuleOp module) {
  // Check if scopeOp has ssbuffer.skip
  bool hasSkipAttr = false;
  module.walk([&](Operation *op) {
    auto scopeOp = dyn_cast<scope::ScopeOp>(op);
    if (!scopeOp) return;
    if (scopeOp->hasAttr("ssbuffer.skip")) {
      hasSkipAttr = true;
    }
  });
  if (hasSkipAttr) {
    LDBG("scopeOp has ssbuffer.skip, skip processing.");
    return failure();
  }
  return success();
}
```

**要点**：遍历 module 找 `scope::ScopeOp`，只要任意 scope 打了 `ssbuffer.skip` 就整体跳过。这是与 `AnalyzeDataFlow` 类似的"能力检查"入口——某些 kernel 由前序 Pass 判定不适合 CV 流水线。

### 3.2 主编排：runOnOperation（L68-L146）

```cpp
void AddControlFlowConditionPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }
  ...
  if (failed(verifyControlFlowPrerequisites(module))) {
    return;
  }

  PassManager pm(&getContext(), module.getOperationName());
  ControlFlowConditionInfo info;

  // Step0: Clone ops in vector/cube to ensure that each block_id has its own
  // ops without sharing
  pm.addPass(createCloneOpsPass());
  // Step1: Process shared iter_args in for ops to eliminate arg sharing across
  // block_ids
  std::unique_ptr<ProcessArgsPass> processArgsPass(new ProcessArgsPass());
  processArgsPass->setConditionInfo(&info);
  pm.addPass(std::move(processArgsPass));
  // Step2: Create if ops based on block_id
  ...
  // Step6: Update for loop iteration times based on intraCoreDependentMap
  std::unique_ptr<UpdateLoopIterTimesPass> updateLoopIterTimesPass(
      new UpdateLoopIterTimesPass());
  updateLoopIterTimesPass->setConditionInfo(&info);
  pm.addPass(std::move(updateLoopIterTimesPass));

  if (failed(runPipeline(pm, module))) {
    LDBG("Pass failed!");
    if (!CVPipeline::hasFallbackAttr(module)) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
  }
}
```

**要点**：
- **`ControlFlowConditionInfo info` 是跨子 Pass 的共享内存**：主 Pass 创建局部对象，用 `setConditionInfo(&info)` 指针传给需要它的子 Pass（`CloneOps` 不需要）。所有依赖映射、计数器索引在子 Pass 之间流动；
- **先检查 fallback 短路**：上游已回退则直接 return，避免在坏 IR 上继续改；
- **统一兜底**：任意子 Pass 失败，`runPipeline` 返回 failure，置 `ERRCODE_FAILED`。

### 3.3 legacyKernel 特判（L117-L126）

```cpp
constexpr llvm::StringLiteral legacyKernel[] = {
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim128"};
bool useLegacyConditions = false;
for (auto &op : module.getOps()) {
  auto funcOp = llvm::dyn_cast<func::FuncOp>(&op);
  if (funcOp && llvm::is_contained(legacyKernel, funcOp.getSymName())) {
    useLegacyConditions = true;
    break;
  }
}
```

**要点**：`UpdateConditionInfo` 内部会根据 `useLegacyConditions` 走不同的条件构造路径（旧 kernel 兼容逻辑）。代码在此预扫描 module 是否包含该历史 kernel 名。

---

## 四、流程总结

### 4.1 六个子 Pass 的流水

```
AddControlFlowConditionPass
  ├─► Step0: CloneOps               每个 block 克隆先前 block 的 op（反向遍历）
  │         ├─► validateBlockIdsConsecutive   校验 block_id 连续无交错
  │         ├─► cloneOpsInMainLoop            反向克隆 + 拓扑排序
  │         ├─► cleanupClonedOpsInMainLoop    删除克隆产生的冗余 op（CUBE 走 memgraph 规则）
  │         └─► validateClonedOpsInVector     Vector 主循环禁止残留 tensor 型克隆
  ├─► Step1: ProcessArgs            拆分跨 block 共享的 iter_arg
  │         ├─► updateIndependentCondsInWhileBlocks  while 的 condition 依赖链按 block 克隆
  │         └─► processSharedIterArgs         共享 arg → 每个非 owner block 一个独立新 arg
  ├─► Step2: CreateIfOps            每个 block 一个 scf.if（%true 占位）
  │         ├─► computeYieldValues            then 用块内逃逸值，else 用 loop iter_arg
  │         ├─► createIfInMainLoop            建 if + 移 op 到 then + 替换外部 uses
  │         └─► info->blockCounterNums        block 数量记录（供 UpdateLoopOps 分配计数器）
  ├─► Step3: InitDependentMap       kCrossCoreDeps/kIntraDeps 属性 → 依赖映射 + DAG
  │         ├─► initCrossCoreDependentMap     跨核 consumer->producer 组（按主循环过滤）
  │         ├─► initIntraCoreDependentMap     核内 per 主循环 consumer->producers
  │         ├─► computeProducerBufferCount    统计 producer 数（flowOpt 阈值判断用）
  │         ├─► buildIfBlockDAG               if 块间边（CrossCore/IntraCore）
  │         ├─► detectCycle                   DFS 三色检测，有环 → ERRCODE_IGNORED
  │         └─► collectFlowOptIfOpPairs       深度=3 的节点成对 → flowOpt 重叠
  ├─► Step4: UpdateLoopOps          扩展主循环 + 核间同步
  │         ├─► analyzeTensorIterArgDependencies  tensor iter_arg 的 producer/consumer if
  │         ├─► deriveBlockCountersFromIfOps      block 数 → blockCounterNums（兜底）
  │         ├─► addBlockCountersAndInnerDepConds  追加计数器/核内依赖计数 iter_arg
  │         └─► insertInterCorePipeS              插入 PIPE_S sync（flag 15）
  ├─► Step5: UpdateConditionInfo    构造每个 if 的真实条件
  │         ├─► allocSSBuffer                   核间计数缓冲（Vector0:0.., Vector1:1024..）
  │         ├─► collectDependencyBuffers        依赖组 → buffer 索引映射
  │         ├─► setCrossCoreCondition           SSBuffer 计数条件（输入>0, 输出<prod_num）
  │         ├─► setIntraCoreCondition           iter_arg 计数条件（含 tensor 迭代依赖）
  │         ├─► setFlowOptCondition             第三节点重叠条件（counter 下界/上界）
  │         └─► combineConditions               And 合并 + 重建 if + 更新 yield
  └─► Step6: UpdateLoopIterTimes    扩展循环迭代次数
            ├─► GetMainLoopIdToLoopOpMap        mainloop_id -> [CUBE loop, Vector loop]
            ├─► ComputeMainLoopTimes            per loop 计算 ifCount + requiredBuffers/x
            ├─► UpdateForLoopIteration          同 id 两侧取 max，克隆 for 换新 upperBound
            ├─► replaceForOpCounterInIfOps      用 cntArgs 替换归纳变量
            └─► UpdateWhileLoopCondition        while 条件按 per-block arg 克隆合并 OR
```

### 4.2 与前后 Pass 的协作

- **前置依赖**：`AllocMultiCache`（L82）注入的 `kCrossCoreDeps`/`kIntraDeps`/`kIntraBuffer`、`ComputeBlockOpt` 打的 `kMainLoop`/`kBlockId`、`SplitDataflow` 打的传输与 flag；
- **产出**：每个 if 带真实同步条件（SSBuffer/iter_arg 计数）、主循环带 block 计数器与 PIPE_S 同步、迭代次数已扩展、for/while 的 yield/condition 已更新；
- **后置**：`SeparateMemoryFromCompute`（L84）分离访存计算，`RemoveSsbufAttr`（L85）清理所有 `ssbuffer.*` 属性（包括本 Pass 打的 `kIf`/`kArg`/`kClone` 等）。

---

## 五、面试要点

1. **AddControlFlowCondition 的目标是什么？** `AllocMultiCache` 只分配了多缓冲、打了占位 `if %true`；本 Pass 把这些占位 if 变成可执行的同步条件（基于计数），并保证 CUBE/Vector 两侧流水线启动对齐（PIPE_S），最后按依赖距离扩展迭代次数，使依赖有足够缓冲空间。

2. **为什么拆成 6 个子 Pass？** 依赖链严格线性：先**去共享**（CloneOps）才能安全地按 block 拆分状态；再**拆 arg**（ProcessArgs）让每个 block 状态独立；再**建 if 骨架**（CreateIfOps）提供块级控制点；再**建依赖图**（InitDependentMap）指导后续；再**扩展循环**（UpdateLoopOps）提供计数器变量；再**算条件**（UpdateConditionInfo）真正落表达式；最后**扩迭代**（UpdateLoopIterTimes）保证缓冲充足。每个子 Pass 职责单一，失败可独立 fallback。

3. **为什么必须先 CloneOps 去共享？** 多缓冲轮换的前提是"每个 block 看到的是自己那份数据"。如果 op 被多个 block 共享，同一 op 的结果被多个 if 复用，计数器增减就会互相污染。反向克隆（从最后一个 block 开始）后每个 block 有独立 op 副本，才能独立控流。

4. **for 和 while 的处理差异在哪？** for 有归纳变量且循环次数静态已知：计数器初值用 lowerBound，条件用 `counter < upperBound`，迭代扩展直接改 upperBound；while 无归纳变量：计数器用独立 `constant 0`，condition 依赖的 iter_arg 要按 block 克隆到 after 区（`whileBlockArgMap`），条件表达式用 `buildWhileCounterCondition` 从 before 区 def-chain 克隆重映射。**for 是"静态扩展"，while 是"动态条件重写"**。

5. **核间 vs 核内条件为什么用不同计数器？** 核间依赖跨 CUBE/Vector 两个核，无法共享同一 loop 的 iter_arg，所以用**共享内存 SSBuffer 计数槽**（每依赖组一个，Vector 0/1 不同地址段，volatile load/store）；核内依赖在同一 loop 内，直接用 **loop iter_arg** 计数即可，省去访存。tensor 迭代依赖介于两者之间——同 loop 内但跨迭代，用一个 iter_arg 标记该迭代是否已消费/生产。

6. **SSBuffer 计数怎么保证线程安全/可见性？** load/store 都打 `kMemrefExtVolatile`（volatile）标记，防止编译器重排；两个 Vector 核用不同地址段（0- vs 1024+）天然隔离；每次迭代 then 分支末尾统一更新（输入 -1、输出 +1），保证"读时是旧值、写后是新值"。

7. **为什么需要 if DAG 和 flowOpt？** 依赖图能检测非法循环（producer 在 consumer 之后则无法排流水 → IGNORED）；flowOpt 在缓冲充足时把距离为 3 的 producer→consumer 对开启**更深的重叠**（`counter >= upperBound || counter >= lower + step*opt_num` 条件），进一步压缩流水线气泡。

8. **迭代次数为什么是 `ceil(iter * requiredBuffers / x) + ifCount`？** producer 有 `x` 个缓冲（producerOps.size()），依赖距离需要 `requiredBuffers = consumer_idx - producer_idx + 1` 块缓冲；迭代数放大 `requiredBuffers/x` 倍并向上取整，最后 `+ifCount` 是给每个 block 的 if 占用一拍的缓冲余量。用分数交叉相乘比较 `requiredBuffers/x` 避免浮点精度。

9. **整个 Pass 的 fallback 策略？** 每个子 Pass 内部失败即 `setFallbackAttr(ERRCODE_FAILED)`；DAG 有环置 `ERRCODE_IGNORED`（与 FAILED 区分：IGNORED 表示"不做流水线也正确，只是没优化"）。主 Pass 入口与子 Pass 都先检查 `hasFallbackAttr` 短路，保证幂等。
