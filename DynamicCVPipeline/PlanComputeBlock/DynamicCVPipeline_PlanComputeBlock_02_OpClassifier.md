# DynamicCVPipeline 子 Pass 逐行讲解：OpClassifierPass

> 源码文件：[`third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/OpClassifier.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/OpClassifier.cpp)（1926 行）
>
> 头文件：[`third_party/ascend/include/DynamicCVPipeline/PlanComputeBlock/OpClassifier.h`](../../../third_party/ascend/include/DynamicCVPipeline/PlanComputeBlock/OpClassifier.h)
>
> 流水线位置：`PlanComputeBlockPass` 的 **Step 1**，主 Pass 第一个子 pass

---

## 一、这个 Pass 在做什么

给模块里**每个 op 打上执行单元标签** `ssbuffer.core_type ∈ {CUBE, VECTOR, CUBE_AND_VECTOR}`，同时给所有同步 op 分配**独立 block id**。这是后续 CUBE/VECTOR 分块的前提。

核心逻辑可以归纳为**一个核心策略 + 九个步骤**：

**核心策略**：以 `linalg.matmul` 为锚点（matmul 只能在 CUBE 上跑，所以恒为 CUBE），向四周 pattern 匹配出"必然属于 CUBE"的种子 op，再从种子向上 BFS 传播 CUBE；剩下没被染色的 op 默认 VECTOR；再从 VECTOR 向上传播；对于同时被两边使用的 op（CUBE_AND_VECTOR），克隆一份分别归属。

九个步骤（`runOnOperation` L1835-L1910）：

| Step | 函数 | 做什么 |
|---|---|---|
| 1 | `markSynchronizationOp` | 同步 op 按类型/模式标 core_type，并分配独享 block id + `kExternalSync` |
| 2 | `patternMatchCUBE` | 在 matmul 周边做 pattern 匹配，收集 CUBE 种子 |
| 3 | `propagateCubeUpstream` | 从种子向上 BFS，把所有可 CUBE 化的上游 op 染成 CUBE |
| 4 | `penetrateCubeIntoForLoops` | 纯 loader 循环（体只有数据搬移）整个染成 CUBE |
| 5 | `markRemainingAsVector` | 剩余未决定 op 默认 VECTOR |
| 6 | `propagateVectorUpstream` | 从 VECTOR 向上 BFS 传播（遇到 matmul/scf 停止） |
| 7 | `handleCubeAndVector` | CUBE_AND_VECTOR 分裂：克隆 op，原 op 归 CUBE、克隆归 VECTOR |
| 8 | `handleSCFYield` | 给 scf.yield 及其父 scf op 打 `ssbuffer.core_type`（按 operand 定义 op 类型） |
| 9 | `stampToIR` | 把内存里的分类结果写回 IR 属性（跳过 linalg 内部 / 大部分 scf） |

**为什么同步 op 要先处理**：同步栅栏是分块的"墙"，先给墙定好类型和独立 id，后续所有分块算法都遵守"块不跨墙"。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `OpCoreType` | 枚举：`OP_UNDETERMINED` / `OP_CUBE_ONLY` / `OP_VECTOR_ONLY` / `OP_CUBE_AND_VECTOR`。**位掩码语义**：`markCube` 用 `|= OP_CUBE_ONLY`，`propagateVectorUpstream` 用 `|= OP_VECTOR_ONLY`，两者都置位就是 CUBE_AND_VECTOR |
| `ssbuffer.core_type` | 写回 IR 的字符串属性：`"CUBE"` / `"VECTOR"` / `"CUBE_AND_VECTOR"`，支持逗号分隔多值（用于 yield 多结果） |
| CUBE seed | 与 matmul 直接相邻、必然属于 CUBE 的 op（to_tensor/transpose/fill/empty/broadcast/store/materialize/extract_slice） |
| `inBroadcastChain` | 被并入 broadcast 链的 op 集合（expand/ext 等），CUBE 传播时跳过它们（已处理） |
| `vectorOnlyProducerCache` | 记忆化缓存：某 tensor 的定义链上是否含 VECTOR op（`hasVectorOnlyProducer`） |
| `CloneOpMap` | `原 op ↔ 克隆 op` 双向映射，CUBE_AND_VECTOR 分裂时维护，yield 处理与 stamp 时用 |
| `isExtractedLoadStoreRelated` | 判断 op 是否属于 `hivm::ExtractLoadStore` 拆出的 load/store 链（这类 op 保持 VECTOR，不参与 CUBE） |
| `kMayImplicitTransposeWithLastAxis` | 注解属性；带此注解的 to_tensor 表示"末轴隐式转置"，保持 VECTOR |
| `gpu::BarrierOp` | 通用 barrier，无条件标 CUBE_AND_VECTOR（分裂时克隆成两份） |
| `hivm::SyncBlockSetOp/WaitOp` | 带 `tcore_type` 属性，按 CUBE/VECTOR 直接标 |
| `hivm::SyncBlockOp` | 带 `SyncBlockMode`：ALL_CUBE→CUBE、ALL_VECTOR/ALL_SUB_VECTOR→VECTOR、ALL→CUBE_AND_VECTOR |
| `kExternalSync` | 主 Pass 给同步 op 打的标记（值为 1），下游识别"外部同步点" |

---

## 三、逐行讲解

### 3.1 主流程：`runOnOperation`（L1835-L1910）

```cpp
void OpClassifierPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) { return; }

  aliasAnalysis = std::make_shared<AliasAnalysis>(module);   // L1848
  memDepGraph = std::make_shared<CVPipeline::MemoryDependenceGraph>(
      module, *aliasAnalysis);                                // L1849

  initializePass(module);                                     // L1853

  if (failed(markSynchronizationOp())) {                      // Step 1
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
  if (patternMatchCUBE() != 0) { ... }                        // Step 2
  if (propagateCubeUpstream() != 0) { ... }                   // Step 3
  if (penetrateCubeIntoForLoops().failed()) { ... }           // Step 4
  if (markRemainingAsVector() != 0) { ... }                   // Step 5
  if (propagateVectorUpstream() != 0) { ... }                 // Step 6
  if (handleCubeAndVector() != 0) { ... }                     // Step 7
  if (handleSCFYield() != 0) { ... }                          // Step 8
  if (stampToIR() != 0) { ... }                               // Step 9
}
```

- 每一步失败都打 `ERRCODE_FAILED` fallback——分类是全局决策，一步出错就无法产出可信的 core_type，宁可直接回退标准编译。
- `AliasAnalysis` + `MemoryDependenceGraph` 只在这里构建一次，作为成员传给所有步骤（`getUpstreamOpsWithMemoryDeps` 需要内存边）。

### 3.2 初始化与工具（L139-L162, L118-L136, L165-L170, L172-L194）

```cpp
void OpClassifierPass::initializePass(ModuleOp module) {     // L139-L162
  opCoreTypes.clear(); allOps.clear(); cubeSeeds.clear(); ... // 重置所有容器
  module.walk([&](Operation *op) {
    if (isa<ModuleOp, func::FuncOp, func::ReturnOp>(op)) return;
    allOps.push_back(op);                                     // 收集全部 op
  });
  for (Operation *op : allOps) opCoreTypes[op] = OP_UNDETERMINED;
}
```

- `allOps` 是**静态快照**：在 walk 时收集。之后即使分裂产生新 op（Step 7），也会被 `push_back` 到 `allOps`，保证 stamp 阶段能覆盖克隆体。
- `markCube(op)`（L165-L170）：`opCoreTypes[op] |= OP_CUBE_ONLY`——位或语义，如果 op 之前已是 VECTOR，markCube 后变成 CUBE_AND_VECTOR。
- `isExtractedLoadStoreRelated`（L172-L194）：穿透 `to_tensor → view → alloc` 链，判断是否属于 `hivm::ExtractLoadStoreAttr` 标记的循环（这是显式提取的 load/store 场景，此类 op 保持 VECTOR 不走 CUBE）。
- `parseCoreTypeFromString`（L118-L136）：解析 `"CUBE,VECTOR"` 多值字符串，按 index 取第 i 个分量（Step 8 的 else 分支处理用）。

### 3.3 Step 1：同步 op 标记——`markSynchronizationOp`（L1753-L1832）

```cpp
llvm::LogicalResult OpClassifierPass::markSynchronizationOp() {
  ModuleOp module = getOperation();
  CVPipeline::ComputeBlockIdManager bm(module);               // 复用管理器，扫描已有 id

  for (Operation *op : allOps) {
    if (!isSyncOp(op)) continue;
    if (isa<gpu::BarrierOp>(op)) {
      setCoreType(op, OP_CUBE_AND_VECTOR);                    // 通用 barrier：两侧都要等
    } else if (isa<hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(op)) {
      auto tcoreAttr = op->getAttrOfType<hivm::TCoreTypeAttr>("tcore_type");
      // 按 CUBE / VECTOR 分类；CUBE_OR_VECTOR 等留给默认 VECTOR 扫尾
    } else {
      auto syncBlock = dyn_cast<hivm::SyncBlockOp>(op);
      switch (syncBlock.getSyncBlockModeAttr().getSyncMode()) {
      case SyncBlockMode::ALL_CUBE:      setCoreType(op, OP_CUBE_ONLY); break;
      case SyncBlockMode::ALL_VECTOR:    [[fallthrough]];
      case SyncBlockMode::ALL_SUB_VECTOR:setCoreType(op, OP_VECTOR_ONLY); break;
      case SyncBlockMode::ALL:           setCoreType(op, OP_CUBE_AND_VECTOR); break;
      }
    }
    // 给每个同步 op 发独立 block id（保证"墙"独享 id）
    if (llvm::failed(bm.markOpBlockId(op))) return failure();
    op->setAttr(CVPipeline::kExternalSync, ...);              // 打外部同步标记
  }
  return success();
}
```

- 三种同步 op 的类型来源不同：`gpu.barrier` 无类型信息 → 双核都等（CUBE_AND_VECTOR，Step 7 会克隆成 CUBE 版 + VECTOR 版）；`sync_block_set/wait` 看 `tcore_type`；`sync_block` 看 `SyncBlockMode`。
- **`markOpBlockId` 给每个同步 op 发一个独享 id**——同步 op 是"墙"，必须自己一组，这样任何计算块都不可能与墙同组（否则重排时栅栏会跟着块移动）。
- `SyncBlockMode::ALL`（全核同步）标 CUBE_AND_VECTOR：Step 7 分裂时，注释说明"克隆保持相同 block id，degrade pass 会按 block id 重新配对"——分裂出的两份同步 op 仍是同一个墙。

### 3.4 Step 2：pattern 匹配找 CUBE 种子——`patternMatchCUBE`（L527-L631）

```cpp
int OpClassifierPass::patternMatchCUBE() {
  for (Operation *op : allOps) {
    if (!isa<linalg::MatmulOp>(op)) continue;

    opCoreTypes[op] = OP_CUBE_ONLY;          // matmul 恒为 CUBE

    // ---- 上游（operand 侧）----
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      // 若 operand 是 loop iter_arg（无定义 op），沿 LoopLikeOpInterface 回溯到 init
      for (Value cur = operand; !def;) {
        auto blockArg = dyn_cast<BlockArgument>(cur);
        auto loopLike = blockArg ? dyn_cast_or_null<LoopLikeOpInterface>(
            blockArg.getOwner()->getParentOp()) : nullptr;
        OpOperand *init = loopLike ? loopLike.getTiedLoopInit(blockArg) : nullptr;
        if (!init) break;
        cur = init->get();
        def = cur.getDefiningOp();
      }
      matchToTensorPattern(def);    matchTransposePattern(def);
      matchFillPattern(def);        matchEmptyPattern(def);
      matchBroadcastPattern(def);
    }

    // ---- 下游（result 侧）----
    for (Value result : op->getResults()) {
      if (!result.hasOneUse()) continue;
      for (Operation *user : result.getUsers()) {
        // 若 user 是 scf.yield，穿透 yield 找真正消费者（可能穿过多层 scf）
        Operation *curUser = user;
        while (curUser) {
          if (auto yieldOp = dyn_cast<scf::YieldOp>(curUser)) {
            // 找到 prevResult 在 yield 中的槽位 → scf 对应 result → 下一个非 yield 用户
            ...
          }
          matchStorePattern(curUser);
          matchExtractSlicePattern(curUser);
          matchMaterializePattern(curUser);
          break;
        }
      }
    }
  }
  return 0;
}
```

**上游五个 pattern 各自职责**：

- `matchToTensorPattern`（L214-L242）：matmul 输入来自 `bufferization.to_tensor` → to_tensor 及其 memref 定义 op 都标 CUBE 并入种子。特判：带 `MayImplicitTransposeWithLastAxis` 注解的（末轴隐式转置）和 `ExtractLoadStore` 链上的保持 VECTOR。
- `matchTransposePattern`（L274-L306）：matmul 输入来自 `linalg.transpose` → transpose 本身标 CUBE；其输入若来自 bufferization 方言（排除 `AllocTensorOp`）或 `tensor.empty` 才继续标记为种子（上游链保守，避免 CUBE 过度扩张）。
- `matchFillPattern`（L324-L331）：matmul 的 DpsInit 来自 `linalg.fill` → fill 标 CUBE。
- `matchEmptyPattern`（L347-L354）：DpsInit 来自 `tensor.empty` → empty 标 CUBE。
- `matchBroadcastPattern`（L390-L441）：DpsInit 来自 `linalg.broadcast`（A*B+C 的 bias 表）→ 校验：结果单用户、`getBTSizeFromValidBroadcastOp` 有效且 ≤ `CACHE_TABLE_BUFFER_SIZE`(4096) → broadcast 标 CUBE。同时沿链处理 `expand_shape`（`extractMmadBiasFromPotentialUnitDimExpand`，L368-L389，处理 unit-dim expand，标 CUBE 入 `inBroadcastChain`）和 `arith.extf`（f16→f32 的扩展，标 CUBE）。

**下游三个 pattern**：`matchStorePattern`（hivm.store）、`matchExtractSlicePattern`（extract_slice + 其后的 store/materialize）、`matchMaterializePattern`（bufferization.materialize_in_destination）。作用：保证 matmul 结果写出也走 CUBE，CUBE 侧一次成型，减少跨核搬运。

### 3.5 Step 3：CUBE 向上 BFS——`propagateCubeUpstream`（L798-L840）

```cpp
int OpClassifierPass::propagateCubeUpstream() {
  std::queue<Operation *> cubeQueue;
  for (Operation *seed : cubeSeeds) { cubeVisited.insert(seed); cubeQueue.push(seed); }

  while (!cubeQueue.empty()) {
    Operation *cur = cubeQueue.front(); cubeQueue.pop();
    if (inBroadcastChain.contains(cur)) continue;    // broadcast 链已处理，跳过
    llvm::SmallVector<Operation *> upstreamOps;
    getUpstreamOpsWithMemoryDeps(cur, upstreamOps);  // SSA + 内存双维取上游

    for (Operation *def : upstreamOps) {
      if (!def || cubeVisited.count(def) || isa<linalg::MatmulOp>(def)) continue;
      if (shouldSkipCubeUpstream(def)) continue;     // 共享跳过规则
      cubeVisited.insert(def);
      opCoreTypes[def] = OP_CUBE_ONLY;
      cubeQueue.push(def);
    }
  }
  markFillOpsAsCube();   // 收尾：fill 的 outs 是 CUBE 时补标 CUBE
  return 0;
}
```

- `getUpstreamOpsWithMemoryDeps`（L692-L723）：**SSA + 内存双维**。SSA 侧取 operand 定义 op；若 operand 是 scalar iter_arg，`findIterArgUpstreamOps`（L640-L688）进一步找回该迭代变量的 init 定义和 yield 值定义（处理 for/while 的循环携带标量）；内存侧只对 `to_tensor` 生效（其它 op 已有 SSA 依赖），把 `memDefs` 中 `memref.copy` 加入上游（to_tensor 的 buffer 可能来自 copy，需要把 copy 链也 CUBE 化）。
- `shouldSkipCubeUpstream`（L772-L795）四个跳过规则：
  1. **tensor 型 arith/math op**（`isTensorArithOrMathOp` L726-L736）→ 这是 VECTOR 计算，不能 CUBE；
  2. **`tensor.extract` 且其源 tensor 定义链含 VECTOR op**（`hasVectorOnlyProducer` L743-L769，带缓存）→ extract 结果无法在 CUBE 重算；
  3. **linalg 内部 op**（`isInsideNestedLinalgRegion` L68-L78）；
  4. **ExtractLoadStore 相关**。
- `markFillOpsAsCube`（L889-L936）：补处理 fill——fill 的 DpsInit（outs）是 CUBE op（非 empty）定义，或 outs 是 CUBE scf op 的 iter_arg → fill 标 CUBE。最后 `handleFillInScfIf`（L940-L979）：fill 在 `scf.if`（无结果、仅做分支）中且 if 内全部 op 都是 CUBE → 整个 scf.if 标 CUBE 并 `propagateCubeUpstreamForOp`（L982-L1012）向上传播。

### 3.6 Step 4：纯 loader 循环渗透——`penetrateCubeIntoForLoops`（L1164-L1184）

```cpp
llvm::LogicalResult OpClassifierPass::penetrateCubeIntoForLoops() {
  llvm::SmallVector<Operation *> loaderLoops;
  getOperation().walk([&](scf::ForOp forOp) {
    if (isCubeLoaderForOp(forOp)) loaderLoops.push_back(forOp);
  });
  getOperation().walk([&](scf::WhileOp whileOp) {
    if (isCubeLoaderForWhileOp(whileOp)) loaderLoops.push_back(whileOp);
  });
  for (Operation *loop : loaderLoops) {
    Block *body = getLoopBodyBlock(loop);
    if (body) markLoopBodyAsCube(loop, body, opCoreTypes);  // 整个循环体标 CUBE
  }
  return llvm::success();
}
```

- `isCubeLoaderForOp`（L1101-L1125）：**纯 loader 循环**判定：
  1. 无 `hivm::ExtractLoadStoreAttr`；
  2. 每个结果的所有用户都是 matmul 或 CUBE op（数据只供给 CUBE）；
  3. 循环体只做数据搬移 + 标量索引计算（`isDisqualifyingLoaderOp` L1080-L1097：linalg 计算、math、tensor 型 arith 都会 disqualify）。
- `markLoopBodyAsCube`（L1151-L1162）：整个循环体（除 scf 自身）染 CUBE，循环 op 本身也标 CUBE。
- 动机：triton 里常见的"把 A 矩阵从全局内存搬进循环"的 for，体里只有 `to_tensor/load/copy`。这类循环本质是 CUBE 侧的数据加载，标成 CUBE 后整个循环作为 CUBE 计算块，减少核间交互。

### 3.7 Step 5：剩余默认 VECTOR——`markRemainingAsVector`（L849-L869）

```cpp
int OpClassifierPass::markRemainingAsVector() {
  for (Operation *op : allOps) {
    if (isa<annotation::MarkOp>(op)) {       // mark 跟随其 src 定义 op 的类型
      Value src = markOp.getSrc();
      opCoreTypes[op] = src.getDefiningOp() ? getCoreType(srcDef) : OP_VECTOR_ONLY;
      continue;
    }
    if (opCoreTypes[op] == OP_UNDETERMINED && !isa<scf::YieldOp>(op)) {
      opCoreTypes[op] = OP_VECTOR_ONLY;      // 默认 VECTOR
    }
  }
  return 0;
}
```

- **默认值语义**：CUBE 是"特权"，需要被证明（种子 + 传播）；VECTOR 是"兜底"，所有未分类的 op 都默认 VECTOR。这保证任何 op 最终都有确定归属。
- `annotation.mark` 特例：mark op 不产生计算，跟随其 src 定义 op 的类型（后续 PlanCubeBlock 的 `fuseMarkOpToDef` 也是这个逻辑）。

### 3.8 Step 6：VECTOR 向上 BFS——`propagateVectorUpstream`（L1022-L1073）

```cpp
int OpClassifierPass::propagateVectorUpstream() {
  for (Operation *op : allOps)
    if (opCoreTypes[op] == OP_VECTOR_ONLY) { vecVisited.insert(op); vecQueue.push(op); }

  while (!vecQueue.empty()) {
    Operation *cur = vecQueue.front(); vecQueue.pop();
    // 跳过 module/func/return；已是 CUBE 的 memref.copy/alloca 也不动
    ...
    for (Operation *def : upstreamOps) {
      if (!def || vecVisited.count(def)) continue;
      if (isa<linalg::MatmulOp>(def) || isa<scf::SCFDialect>(def->getDialect()))
        continue;                          // matmul 与 scf 是传播墙
      vecVisited.insert(def);
      opCoreTypes[def] |= OP_VECTOR_ONLY;  // 位或：CUBE+VECTOR → CUBE_AND_VECTOR
      vecQueue.push(def);
    }
  }
  return 0;
}
```

- 从所有 VECTOR op 出发向上传播：被 VECTOR 消费的上游数据必须也能在 VECTOR 侧生成。
- **传播墙**：`linalg.matmul`（只能 CUBE）和 scf 方言（控制流不标 VECTOR）。
- 位或语义：若上游已是 CUBE_ONLY，`|= OP_VECTOR_ONLY` 后变成 CUBE_AND_VECTOR——这就是"共享 op"的发现机制，交给 Step 7 分裂。

### 3.9 Step 7：CUBE_AND_VECTOR 分裂——`handleCubeAndVector`（L1670-L1698）+ `splitOperationForCubeAndVector`（L1565-L1658）

这是全文件最精巧的部分。核心问题：**一个 op 同时被 CUBE 用户和 VECTOR 用户使用**（如 `linalg.fill` 既喂 matmul 又喂 `arith.addf`），但两颗核不能共享同一个 op 实例——必须克隆。

```cpp
void OpClassifierPass::splitOperationForCubeAndVector(
    Operation *op, llvm::DenseSet<Operation *> &processedOps,
    llvm::DenseMap<Operation *, Operation *> &opToVectorClone) {
  if (processedOps.count(op)) return;      // 防重复
  processedOps.insert(op);
  if (opCoreTypes[op] != OP_CUBE_AND_VECTOR) return;

  // Phase 1：先递归分裂上游共享 op（深度优先）
  // 因为构建 mapping 时需要用到的"operand 的 VECTOR 克隆"必须先存在
  for (Value operand : op->getOperands()) {
    if (Operation *def = operand.getDefiningOp()) {
      if (opCoreTypes[def] == OP_CUBE_AND_VECTOR)
        splitOperationForCubeAndVector(def, processedOps, opToVectorClone);
    }
  }

  // Phase 2：构建 IRMapping，把 operand 指向其 VECTOR 克隆
  IRMapping mapping;
  for (Value operand : op->getOperands()) {
    if (Operation *def = operand.getDefiningOp()) {
      auto it = opToVectorClone.find(def);
      if (it != opToVectorClone.end()) {
        for (unsigned i = 0; i < def->getNumResults(); ++i)
          if (def->getResult(i) == operand)
            mapping.map(operand, vectorClone->getResult(i));
      }
    }
  }

  // Phase 3：克隆 op，重分类
  OpBuilder builder(op);
  Operation *vectorOp = builder.clone(*op, mapping);  // 克隆体用 VECTOR 侧输入
  opCoreTypes[op] = OP_CUBE_ONLY;        // 原 op 归 CUBE
  opCoreTypes[vectorOp] = OP_VECTOR_ONLY; // 克隆归 VECTOR
  opToVectorClone[op] = vectorOp;
  allOps.push_back(vectorOp);            // 纳入 stamp 范围
  CloneOpMap[op] = vectorOp;             // 双向记录
  CloneOpMap[vectorOp] = op;

  // Phase 4：重定向 VECTOR 用户到克隆结果
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    Value originalResult = op->getResult(i);
    Value vectorResult = vectorOp->getResult(i);
    for (OpOperand &use : originalResult.getUses()) {
      Operation *user = use.getOwner();
      if (!user || user == vectorOp) continue;
      OpCoreType coreType = getCoreType(user);
      if (llvm::isa<scf::ForOp, scf::WhileOp>(user))
        coreType = getForInitCoreType(&use);   // for/while 按 init 值类型判断
      if (coreType == OP_VECTOR_ONLY)
        usesToUpdate.push_back(&use);          // 只重定向 VECTOR 用户
    }
    for (OpOperand *use : usesToUpdate) use->set(vectorResult);
  }
}
```

- **Phase 1 为什么要递归**：假设 `addf ← fill1 ← matmul`，fill1 是 CUBE_AND_VECTOR（被 addf 和 matmul 共用）。分裂 addf 的输入时，需要把 fill1 的克隆也带出来，否则 addf 克隆体仍指向原 fill1（CUBE 版）。深度优先保证"operand 克隆先于自身克隆"。
- **Phase 4 的重定向规则**：`getForInitCoreType`（L1511-L1542）处理 scf.for/while 作为用户的情况——循环 op 本身没有明确 core_type，要看这个 use 对应哪个迭代变量、该迭代变量的 yield 值定义 op 是什么类型。这是"循环携带的 SSA 值归哪边"的关键判定。
- `handleCubeAndVector`（L1670-L1698）用 **do-while** 循环反复扫描：分裂一个 op 可能让新的 CUBE_AND_VECTOR 出现（分裂后其 operands 的处理），直到不再有新分裂。

### 3.10 Step 8：scf.yield 类型标注——`handleSCFYield`（L1468-L1509）

```cpp
int OpClassifierPass::handleSCFYield() {
  // 1. 收集所有 scf.if
  // 2. 先处理 then 块 yield（无参考），再处理 else 块 yield（以 then yield 为参考）
  for (scf::IfOp ifOp : ifOps) {
    thenYield = ifOp.getThenRegion().back().getTerminator();
    if (isa<scf::YieldOp>(thenYield)) processYieldOperation(thenYield, nullptr);
    elseYield = ifOp.getElseRegion().back().getTerminator();
    if (isa<scf::YieldOp>(elseYield)) processYieldOperation(elseYield, thenYield);
  }
  // 3. 处理其余 yield（scf.for 等）
  ...
}
```

- 每个 operand 按"其定义 op 的类型"标注：`OP_CUBE_ONLY → "CUBE"`，其余（含无定义 op 的 iter_arg）→ `"VECTOR"`。
- **else 分支以 then 分支为参考**（`handleYieldFromElseRegion` L1316-L1359）：else 的 yield 没有自己的类型来源，直接用 then yield 的 `ssbuffer.core_type` 字符串取第 i 个分量；若 operand 定义 op 有克隆（CUBE_AND_VECTOR 分裂过），把 else yield 的 operand 重定向到克隆结果。
- `processYieldOperation`（L1374-L1453）最后把组合字符串写到 yield op 和父 scf op 上（`ssbuffer.core_type`），父 scf op 的类型取"字符串含 CUBE → CUBE_ONLY，否则 VECTOR_ONLY"。
- 为什么需要这步：scf 结构作为整体会被分块/移动，必须有明确的 core_type 归属。

### 3.11 Step 9：写回 IR——`stampToIR`（L1704-L1748）

```cpp
int OpClassifierPass::stampToIR() {
  for (Operation *op : allOps) {
    if (!op || !op->getBlock()) continue;
    OpCoreType coreType = opCoreTypes[op];

    // 跳过大部分 scf（scf.condition 除外）
    if (isa<scf::SCFDialect>(op->getDialect()) && !isa<scf::ConditionOp>(op))
      continue;
    if (isa<scf::ConditionOp>(op)) coreType = OP_VECTOR_ONLY;  // condition 恒 VECTOR

    // 跳过 linalg 内部 op
    if (isInsideLinalgBlock) continue;
    if (isa<ModuleOp, func::FuncOp>(op)) continue;

    op->setAttr("ssbuffer.core_type",
                StringAttr::get(op->getContext(), coreTypeToString(coreType)));
  }
  return 0;
}
```

- 只对"有真正归属意义"的 op 写属性：scf 内部大部分 op（如 `scf.for` 的迭代变量使用处）不写；`scf.condition`（while before region 的条件判定）恒 VECTOR——条件计算属于标量/向量逻辑，不是矩阵计算。
- 至此，IR 上每个 op 都有 `ssbuffer.core_type`，同步 op 有独享 `ssbuffer.block_id`，可以进入 CUBE/VECTOR 分块。

---

## 四、算法流程总结

```
OpClassifierPass
├─ Step 1  markSynchronizationOp：同步 op 分类 + 独享 block id + kExternalSync
├─ Step 2  patternMatchCUBE：matmul 锚点，上游 5 pattern + 下游 3 pattern 收集 CUBE 种子
│          （loop iter_arg 穿透 LoopLikeOpInterface 找到真 init）
├─ Step 3  propagateCubeUpstream：种子 BFS 向上传播 CUBE（SSA+内存双维）
│          ├─ 跳过：tensor arith/math、VECTOR-producer 的 extract、linalg 内部、ExtractLoadStore
│          └─ markFillOpsAsCube（fill 的 outs 是 CUBE → CUBE）+ handleFillInScfIf（全 CUBE 的 if → CUBE）
├─ Step 4  penetrateCubeIntoForLoops：纯 loader 循环整体染 CUBE
├─ Step 5  markRemainingAsVector：剩余默认 VECTOR（mark 跟随 src 类型）
├─ Step 6  propagateVectorUpstream：VECTOR 向上 BFS，|VECTOR；matmul/scf 是传播墙
├─ Step 7  handleCubeAndVector：CUBE_AND_VECTOR 分裂（do-while）
│          ├─ 递归先分裂上游共享 op
│          ├─ 克隆 + 重分类（原 CUBE / 克隆 VECTOR）
│          └─ 重定向 VECTOR 用户（for/while 用户按 getForInitCoreType 判定）
├─ Step 8  handleSCFYield：yield 与父 scf 打 ssbuffer.core_type（else 参考 then，克隆重定向）
└─ Step 9  stampToIR：写回 IR（跳过 scf 内部 / linalg 内部；scf.condition 恒 VECTOR）
```

**分类的"方向性"**：CUBE 先证后传播（种子必须显式匹配），VECTOR 先兜底后传播（剩余即 VECTOR），最后用"克隆分裂"调和两者对共享 op 的争夺。整个顺序保证：每个 op 最终有且仅有一个明确归属（CUBE 或 VECTOR，克隆后不存在 CUBE_AND_VECTOR）。

---

## 五、面试要点

1. **为什么 matmul 恒为 CUBE？**
   昇腾 NPU 的矩阵乘只能在 Cube 核执行，这是硬件事实。所以分类算法把 matmul 当作锚点：它周围的"数据准备 op"（to_tensor/transpose/fill/broadcast）和"结果写出 op"（store/materialize/extract_slice）优先标 CUBE，让整条矩阵计算链留在 Cube 侧。

2. **为什么 VECTOR 是默认值而不是 CUBE？**
   CUBE 是稀缺资源且只能跑矩阵类计算；VECTOR 覆盖所有元素级计算。把一个 op 标为 CUBE 意味着"它能进 CUBE 流水线"，需要证据（种子 + 传播链）；标 VECTOR 永远安全（任何 op 都能在 VECTOR 上执行）。所以"有证据→CUBE，无证据→VECTOR"。

3. **CUBE_AND_VECTOR 分裂的核心难点是什么？**
   一是**克隆顺序**：必须先递归分裂上游共享 op，让 operand 的 VECTOR 克隆先存在，克隆自身时才能映射到正确的 VECTOR 侧输入；二是**用户重定向**：只把 VECTOR 用户改到克隆结果，CUBE 用户保留原结果；三是 **for/while 作为用户时无法直接看用户类型**，必须用 `getForInitCoreType` 沿循环 init/yield 链判定该 use 属于哪一侧。

4. **为什么要穿透 scf.yield 找 matmul 结果的真消费者？**
   matmul 结果常常 `yield` 出循环再被 store。若只匹配直接用户，会漏掉"循环外 store 该结果"的 CUBE 种子。穿透逻辑把 `yield → scf result → 下一个非 yield 用户` 的链走通，保证下游 store 也被正确标 CUBE。

5. **"纯 loader 循环整体染 CUBE"的判定条件？**
   ①结果只被 CUBE 消费；②循环体无真实计算（无 linalg 计算、无 math、无 tensor 型 arith），只有数据搬移和标量索引。这样把"全局→片上搬数据"的循环整体并入 CUBE，后续整个循环作为一个计算块，避免搬数据过程中的核间切换。

6. **`stampToIR` 为什么跳过 scf 内部 op 但保留 scf.condition 恒 VECTOR？**
   scf.for 的迭代逻辑（step/induction 运算）不产生矩阵数据，无 core_type 意义；`scf.condition` 是 while before region 的条件判定（标量比较），后续降级/调度需要知道它走 VECTOR。跳过大部分、特判少数，避免属性噪声。

7. **同步 op 为什么在第一步就处理且独享 block id？**
   同步是"墙"，后续所有分块都要遵守"块不跨墙"。第一步处理保证：墙的类型已知（CUBE/VECTOR/CUBE_AND_VECTOR），墙的 id 独立——任何计算块算法在读 id 时就能识别"这是墙，不能并入"。`SyncBlockMode::ALL` 分裂后两份克隆共享 id，degrade pass 按 id 重新配对，保证全核同步语义在分裂后仍成立。

8. **memory 依赖（memDepGraph）在这里解决什么问题？**
   纯 SSA 依赖不够：`to_tensor` 的 memref 可能来自一个 `memref.copy`（无 SSA 直连），如果只按 SSA 传播，copy 链不会被 CUBE 化，产生"CUBE 侧读、VECTOR 侧搬"的割裂。`getUpstreamOpsWithMemoryDeps` 对 to_tensor 额外取 `memDefs` 中的 copy 加入上游，把内存搬运也纳入 CUBE 传播。
