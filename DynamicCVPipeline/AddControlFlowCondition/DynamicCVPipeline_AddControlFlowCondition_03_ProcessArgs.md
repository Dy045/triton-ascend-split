# DynamicCVPipeline 主 Pass 专题：控制流条件注入 · 子专题 03 —— ProcessArgs（共享 iter_arg 拆分）

> 覆盖 `AddControlFlowCondition` 的 **Step 1：ProcessArgsPass**
>
> 源码文件：[`ProcessArgs.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/ProcessArgs.cpp)
>
> 流水线位置：`AddControlFlowCondition`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L83）内第二个子 Pass（主编排 L92-L94）。

---

## 一、这个子 Pass 在做什么

`CloneOps` 消除了 op 共享，但主循环的 **iter_arg（loop-carried 变量）仍可能被多个 block_id 共用**：同一个迭代参数被 block A 和 block B 一起消费、一起 yield 更新。共享 iter_arg 意味着两个 block 的"状态"纠缠在一起，后续给每个 block 注入独立计数器/依赖计数时会互相踩踏。

`ProcessArgs` 的职责是**把共享 iter_arg 拆成 per-block 独立 arg**：

1. **while 条件解耦**（`updateIndependentCondsInWhileBlocks`）：while 的 `scf.condition` 依赖的迭代参数，每个 block 克隆一份"更新计算链"，each (block_id, 原 arg) 分配一个新 iter_arg，并记录 `whileBlockArgMap[whileOp][block_id] = {new_idx: old_idx}`；
2. **共享 arg 拆分**（`processSharedIterArgs`）：被多个 block 共用的 iter_arg，非 owner block 各拿一个独立新 arg，owner 保持原 arg，新 arg 的初始值与更新链都克隆自原 arg。

**一句话**：让每个 block 拥有独立的迭代状态变量，为"每 block 单独控流"完成数据面准备。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `iter_arg` | scf.for/scf.while 的 loop-carried 迭代参数（for 的 body 区块参数，while 的 before/after 区块参数） |
| `ivOffset` | for 的 IV（归纳变量）在 block arg 0，iter_arg 从 1 开始 → `ivOffset=1`；while 无 IV → `ivOffset=0` |
| `SharedArgInfo` | 记录 `{argIndex, ownerBlockId, newArgIndex, nonOwnerBlockId}`：哪个 arg、归属哪个 block、给哪个 block 分配新 arg |
| `argIndexToBlockIds` | `argIndex -> {使用它的 block_id 集合}`；集合大小 >1 即共享 arg |
| `compOp` / `chainOps` | 产生 iter_arg 新值的顶层计算 op（yield operand 的 defining op）与它的反向定义链（`collectChainOps` 从 compOp 反向 BFS） |
| `topologicalSort` | 对克隆链拓扑排序，保证克隆顺序满足依赖 |
| `originalWhileIterArgIndices` | 每个 while 的原始 iter_arg 索引快照（`[0..numOperands)`），供后续识别"哪些 arg 被 condition 使用" |
| `condUsed` | BFS 遍历 `scf.condition` 的条件值的 def-chain，命中 before-block arg 即记为 condition 依赖的 iter_arg |
| `WhileIterArgClonePlan` | while 克隆计划：`newArgDescriptors {(blockId, newArgIdx, origIdx)}`、`compOp`、`chainOps`、`clonedPerBlock` |
| `whileBlockArgMap` | `info->whileBlockArgMap[whileOp][block_id] = {new_arg_idx: old_arg_idx}`，供 `UpdateConditionInfo` 把 before 区 condition 表达式映射到 after 区新 arg |
| `migrateBody` / `migrateWhileBodies` | 把旧 loop 的 body（for 单区 / while 双区）迁移到新 loop，block 参数按位置对齐 |
| `buildNewYieldOp` / `buildNewWhileCondition` | 重建 for 的 yield（追加克隆结果）与 while 的 condition/yield |
| `replaceMainLoopOpAndErase` | 用新 loop 替换旧 loop 的 uses，转移 `intraCoreDependentMap`/`whileBlockArgMap` 等映射后删除旧 op |

---

## 三、逐行讲解

### 3.1 入口：runOnOperation（ProcessArgs.cpp L833-L858）

```cpp
void ProcessArgsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  // 1. While-specific decoupling: snapshot original iter_args; per scf.while,
  //    clone cond-used iter_arg update chains per block_id; record in
  //    info->whileBlockArgMap.
  if (failed(updateIndependentCondsInWhileBlocks(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // 2. Process shared iter_args (adds per-block clones for args shared
  //    across block_ids). Uses originalWhileIterArgIndices captured above.
  if (failed(processSharedIterArgs(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
}
```

**要点**：先做 while 条件解耦（会替换 whileOp），再做共享 arg 拆分（复用前面记录的 `originalWhileIterArgIndices`）。顺序有依赖：while 处理期间必须先用 worklist 快照 whileOp，不能在 walk 中直接改 IR。

### 3.2 while 条件解耦：updateIndependentCondsInWhileBlocks（L533-L561）

```cpp
LogicalResult
ProcessArgsPass::updateIndependentCondsInWhileBlocks(ModuleOp module) {
  // Collect whileOps in a worklist (must NOT mutate IR during the walk —
  // processWhileIterArgsInWhileOp replaces the old whileOp).
  SmallVector<scf::WhileOp> worklist;
  module.walk([&](scf::WhileOp whileOp) {
    if (!isMainLoopOp(whileOp))
      return;
    SmallVector<unsigned> indices;
    for (unsigned i = 0; i < whileOp.getNumOperands(); ++i)
      indices.push_back(i);
    originalWhileIterArgIndices[whileOp] = indices;
    worklist.push_back(whileOp);
  });
  for (scf::WhileOp whileOp : worklist) {
    if (failed(processWhileIterArgsInWhileOp(whileOp, info)))
      return failure();
  }
  return success();
}
```

**要点**：walk 只收集、不修改；随后逐个处理。原始 arg 索引快照用于 `collectConditionUsedIterArgIndices` 定位 condition 依赖的 arg。

### 3.3 识别 condition 依赖的 iter_arg：collectConditionUsedIterArgIndices（L565-L600）

```cpp
static llvm::DenseSet<unsigned> collectConditionUsedIterArgIndices(
    scf::WhileOp whileOp, const SmallVector<unsigned> &originalIndices) {
  llvm::DenseSet<unsigned> used;
  auto cond = whileOp.getConditionOp();
  Value condValue = cond.getCondition();
  Block *beforeBlock = whileOp.getBeforeBody();

  // BFS over def-chain of condValue; stops at before-block BlockArgument
  llvm::SmallPtrSet<Value, 16> visited;
  SmallVector<Value> worklist;
  worklist.push_back(condValue);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      if (blockArg.getOwner() == beforeBlock) {
        unsigned idx = blockArg.getArgNumber();
        if (idx < originalIndices.size()) {
          used.insert(idx);
        }
      }
      continue;
    }
    Operation *defOp = v.getDefiningOp();
    if (!defOp) continue;
    for (Value operand : defOp->getOperands()) {
      worklist.push_back(operand);
    }
  }
  return used;
}
```

**要点**：从 condition 值反向 BFS 定义链，凡遇到 before 区的 BlockArgument 即记录（`idx < originalIndices.size()` 说明是 iter_arg，而非常量/外部值）。

### 3.4 while 克隆计划：planWhileIterArgDescriptors（L674-L712）

```cpp
static WhileIterArgClonePlan
planWhileIterArgDescriptors(scf::WhileOp whileOp,
                            const SmallVector<unsigned> &originalIndices,
                            const llvm::DenseSet<unsigned> &condUsed) {
  WhileIterArgClonePlan plan;
  SmallVector<int> blockIdsInOrder = getBlockIdsInOrder(whileOp);
  ...
  unsigned nextNewArgIdx = whileOp.getNumOperands();
  for (unsigned origIdx : originalIndices) {
    if (!condUsed.contains(origIdx))
      continue;
    plan.posInClonedVec[origIdx] = descIdx++;

    Operation *compOp = findYieldDefiningOp(whileOp.getAfterBody(), origIdx);
    ...
    llvm::DenseSet<Operation *> chainOps;
    collectChainOps(whileOp, compOp, chainOps);
    plan.compOp[origIdx] = compOp;
    plan.chainOps[origIdx] = chainOps;

    for (int blockId : blockIdsInOrder) {
      plan.newArgDescriptors.push_back({blockId, nextNewArgIdx++, origIdx});
    }
  }
  return plan;
}
```

**要点**：对每个 cond 依赖的 arg，**按 block 分配新 arg**（blockId × origIdx 组合）；`compOp`/`chainOps` 在 `migrateBody` **之前**计算（迁移后旧 block 参数失效，查会崩溃）。

### 3.5 while 克隆执行：processWhileIterArgsInWhileOp（L764-L831）

```cpp
LogicalResult
ProcessArgsPass::processWhileIterArgsInWhileOp(scf::WhileOp whileOp,
                                               ControlFlowConditionInfo *info) {
  ...
  auto newWhileOp =
      cast<scf::WhileOp>(createMainLoopOpWithExtras(whileOp, extraInitArgs));
  migrateWhileBodies(whileOp, newWhileOp);
  if (failed(cloneWhileBlockChains(newWhileOp, plan))) {
    return failure();
  }
  buildNewWhileCondition(whileOp, newWhileOp);
  // 按 newArgDescriptors 顺序收集克隆结果作为 extra yield values
  ...
  if (failed(buildNewYieldOp(whileOp.getAfterBody(), newWhileOp.getAfterBody(),
                             newWhileOp, extraYieldValues))) {
    return failure();
  }
  // Mirror (block_id, new_arg_idx) -> orig_idx into localWhileBlockArgMap and
  // info->whileBlockArgMap
  for (auto &desc : plan.newArgDescriptors) {
    ...
    localWhileBlockArgMap[newWhileOp][blockId][newArgIdx] = (int)origIdx;
    if (info) {
      info->whileBlockArgMap[newWhileOp][blockId][newArgIdx] = (int)origIdx;
    }
  }
  if (failed(replaceMainLoopOpAndErase(whileOp, newWhileOp, info))) {
    return failure();
  }
  return success();
}
```

**要点**：`createMainLoopOpWithExtras` 建新 while（追加 extra inits）→ 迁移双区 → 按块克隆更新链 → 重建 condition/yield → **写 `whileBlockArgMap`**（关键产出，供 Step 5 构造条件）→ 替换旧 while。

### 3.6 共享 arg 识别：prepareSharedArgsData / findSharedArgs（L312-L352, L81-L113）

```cpp
static LogicalResult prepareSharedArgsData(
    Operation *loopOp, Block *body, SmallVector<SharedArgInfo> &sharedArgsInfo,
    ...) {
  unsigned ivOffset = isa<scf::ForOp>(loopOp) ? 1 : 0;
  llvm::DenseMap<int, llvm::DenseSet<int>> argIndexToBlockIds;
  if (failed(collectArgIndexToBlockIds(body, ivOffset, argIndexToBlockIds))) {
    return failure();
  }
  SmallVector<int> idsInOrder = getBlockIdsInOrder(loopOp);
  ...
  if (failed(findSharedArgs(argIndexToBlockIds, idsInOrder, sharedArgsInfo))) {
    return failure();
  }
  if (sharedArgsInfo.empty()) {
    return success();
  }
  if (failed(buildCompInfoForSharedArgs(loopOp, body, sharedArgsInfo,
                                        sharedArgToCompOp,
                                        sharedArgToChainOps))) {
    return failure();
  }
  return success();
}
```

**要点**：
- `collectArgIndexToBlockIds`：跳过 tensor 型 iter_arg（tensor 走 `UpdateLoopOps` 的 tensor 依赖，不在此处理），只收集标量/index 型 arg 被哪些 block 使用；
- `findSharedArgs`：集合大小 >1 的 arg 是共享 arg；owner = 顺序第一个使用它的 block，其余 block 各分配一个新 arg（`extraArgCount` 递增）。

### 3.7 非 owner block 克隆链：processSharedArgsIteration / cloneChainForBlock（L253-L308, L178-L220）

```cpp
static LogicalResult processSharedArgsIteration(
    Block *newBlock, SmallVector<SharedArgInfo> &sharedArgsInfo, ...) {
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    info.iterArg = iterArgs[argIndex];

    // The migrated iter_arg (original iter_arg moved to new block)
    Value migratedIterArg = newBlock->getArgument(argIndex + ivOffset);
    // The new extra iter_arg added for this shared arg
    Value newExtraIterArg = newBlock->getArgument(extraIterArgsBase + info.newArgIndex);

    // Build argRemapping: migratedIterArg -> newExtraIterArg
    IRMapping argRemapping;
    argRemapping.map(migratedIterArg, newExtraIterArg);

    // 找到 nonOwnerBlockId 的最后 op，在它后面插克隆链
    Operation *lastOpInBlock = nullptr;
    for (Operation &op : newBlock->without_terminator()) {
      auto blockIdAttr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
      if (blockIdAttr && blockIdAttr.getInt() == info.nonOwnerBlockId) {
        lastOpInBlock = &op;
      }
    }
    ...
    if (failed(cloneChainForBlock(info, ...))) continue;
    if (failed(replaceIterArgsInBlock(info, newBlock, argRemapping, cloneBuilder)))
      continue;

    Value clonedResult = resultMapper.lookup(...);
    clonedResults.push_back(clonedResult);
  }
  return success();
}
```

**要点**：新 arg 的初值 = 原 arg 的 init（`extraInitArgs.push_back(origInits[info.argIndex])`），克隆链插到非 owner block 的末尾，克隆 op 打 `kBlockId = nonOwnerBlockId` 与 `kArg = argIndex`；`replaceIterArgsInBlock` 把该 block 内原 arg 的 uses 换成克隆结果。

### 3.8 for 路径收尾：processSharedArgsInForOp（L386-L406）

```cpp
static LogicalResult processSharedArgsInForOp(
    scf::ForOp forOp, scf::ForOp newForOp,
    SmallVector<SharedArgInfo> &sharedArgsInfo, ...) {
  Block *oldBlock = forOp.getBody();
  Block *newBlock = newForOp.getBody();
  migrateBody(oldBlock, newBlock);

  SmallVector<Value> clonedResults;
  if (failed(processSharedArgsIteration(
          newBlock, sharedArgsInfo, sharedArgToCompOp, sharedArgToChainOps,
          MainLoop(forOp).getIterArgs(), 1, clonedResults))) {
    return failure();
  }
  if (failed(buildNewYieldOp(oldBlock, newBlock, newForOp, clonedResults))) {
    return failure();
  }
  return replaceMainLoopOpAndErase(forOp, newForOp, info);
}
```

**要点**：for 路径：迁移 body → 处理共享 arg → 重建 yield（追加克隆结果）→ 替换旧 for（转移 `intraCoreDependentMap`）。

---

## 四、流程总结

### 4.1 ProcessArgs 的流水

```
ProcessArgsPass::runOnOperation
  ├─► Step 1: updateIndependentCondsInWhileBlocks（while 条件解耦）
  │     ├─► 快照 originalWhileIterArgIndices + worklist
  │     └─► 逐 whileOp：
  │           ├─► collectConditionUsedIterArgIndices   BFS 找 condition 依赖的 arg
  │           ├─► planWhileIterArgDescriptors          按 (block, arg) 规划新 arg
  │           ├─► createMainLoopOpWithExtras           建新 while（追加 extra inits）
  │           ├─► migrateWhileBodies                   迁移 before/after 双区
  │           ├─► cloneWhileBlockChains                每 block 克隆更新链
  │           ├─► buildNewWhileCondition / buildNewYieldOp  重建 condition/yield
  │           └─► 写 whileBlockArgMap + 替换旧 while
  └─► Step 2: processSharedIterArgs（共享 arg 拆分）
        └─► 逐 main_loop op：
              ├─► collectArgIndexToBlockIds            找被多 block 使用的 arg
              ├─► findSharedArgs                       定 owner + 非 owner
              ├─► buildCompInfoForSharedArgs           compOp + 反向链
              ├─► createMainLoopOpWithExtras           建新 loop（追加共享 arg 新 init）
              ├─► migrateBody                          迁移 body
              ├─► processSharedArgsIteration           每非 owner block 克隆计算链 + 替换 uses
              ├─► buildNewYieldOp                      重建 yield（追加克隆结果）
              └─► replaceMainLoopOpAndErase            替换旧 loop（转移依赖映射）
```

### 4.2 与前后子 Pass 的协作

- **前置**：`CloneOps`（Step 0）保证 op 独立；本 Pass 消费 `kBlockId` 划分归属；
- **产出**：每 block 独立 iter_arg；`info->whileBlockArgMap` 记录 while 的 per-block arg 映射；`info->intraCoreDependentMap` 转移到新 loop；
- **后置**：`CreateIfOps`（Step 2）创建 if 时 else 分支从 loop iter_arg 取默认值，此时每个 block 的 arg 已独立，取值不串扰。

---

## 五、面试要点

1. **为什么 while 要单独处理 condition？** while 的 `scf.condition` 在 before 区、依赖迭代参数；`CreateIfOps` 之后每个 block 的 if 需要自己的"是否继续迭代"判断。把 condition 依赖的更新链按 block 克隆到 after 区，再通过 `whileBlockArgMap` 把 before 区表达式重映射到每个 block 自己的 arg，才能构造 per-block 的 while 条件。

2. **为什么 compOp/chainOps 必须在 migrateBody 之前算？** 迁移 body 时旧 block 参数被替换，若迁移后再查 `findYieldDefiningOp`（基于旧 yield operand），旧 SSA 值已失效会崩溃。因此先算好计划再迁移。

3. **共享 arg 怎么拆？** 被 N 个 block 用 → 保留 owner（顺序第一个）的原 arg，其余 N-1 个 block 各加一个新 iter_arg；新 arg 初值 = 原 arg 初值，更新链克隆原 arg 的计算链，插入该 block 末尾，并把 block 内原 arg 的 uses 重定向到克隆结果。

4. **为什么跳过 tensor 型 iter_arg？** tensor 型 iter_arg 承担的是**跨迭代张量数据流**（producer if 生产、consumer if 消费），语义与标量状态变量不同，由 `UpdateLoopOps` 的 `analyzeTensorIterArgDependencies` 单独处理（每个 consumer 一个计数变量），避免标量克隆链误伤张量流。

5. **for 与 while 路径的差异？** for 单区 body 迁移 + 克隆链插块尾 + yield 重建；while 双区（before/after）迁移 + condition 与 yield 都要重建 + 额外维护 `whileBlockArgMap`。共享 arg 拆分逻辑两者共用 `processSharedArgsIteration`，仅 `ivOffset`（1 vs 0）与进入路径不同。
