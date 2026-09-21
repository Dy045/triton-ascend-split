# DynamicCVPipeline 主 Pass 专题：控制流条件注入 · 子专题 02 —— CloneOps（块内 op 去共享）

> 覆盖 `AddControlFlowCondition` 的 **Step 0：CloneOpsPass**
>
> 源码文件：[`CloneOps.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/CloneOps.cpp)
>
> 流水线位置：`AddControlFlowCondition`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L83）内第一个子 Pass（主编排 L88）。

---

## 一、这个子 Pass 在做什么

进入本 Pass 前，主循环体里的 op 是按 `ssbuffer.block_id` 分块的，但**前序 Pass（如 `AllocMultiCache`）可能注入/复用了跨 block 共享的 op**——同一 op 被多个 block_id 的 if 同时引用。共享意味着：后续给每个 block 单独注入计数器增减、单独控流时，一个 block 的修改会污染其他 block 的视图。

`CloneOps` 的职责是**消除共享**：对每个 main loop（`scf.for`/`scf.while`），按 block_id 从后往前，把"先前 block 的所有 op"克隆一份插到当前 block 的 op 之前，克隆出来的 op 打上当前 block 的 `block_id` 与源 block 的 `kClone` 标记。克隆完后清理**克隆产生的冗余 op**（同步/搬移/fixpipe 只保留一份），最后校验 Vector 主循环里不残留 tensor 型克隆。

**一句话**：让每个 block 拥有自己完整的 op 链，为后续"每 block 独立状态"铺路。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `ssbuffer.block_id`（`kBlockId`） | 每个 op 的块归属；`getLoopDirectChildBlockId` 取"主循环直接子块"的 block_id（嵌套 op 沿父链回溯到主循环直属子 op） |
| `ssbuffer.clone`（`kClone`） | 克隆 op 上记录源 block_id 的标记；`cleanupClonedOps` 依据它识别要清理的克隆 op |
| `blockOps` / `idsInOrder` | `collectOpsByBlockId` 按 block_id 收集主循环直接子 op；`getBlockIdsInOrder` 取块 id 的出现顺序（主循环体 op 的物理顺序） |
| `IRMapping` / `valueMap` | 克隆时建立 旧值→新值 映射；`cloneOpWithMapping` 先把**所有已克隆值**灌进 mapper 再 `builder.clone`，保证跨 op 依赖正确 |
| `yieldValues` | 主循环 body terminator（`scf.yield`）的 operand 集合；`updateCloneMapping` 跳过这些值不重映射（它们是 loop-carried 迭代参数，不是本块产物） |
| `topologicalSort` | 拓扑排序 `curOps`，保证克隆后 op 依赖顺序正确 |
| `MemoryDependenceGraph` | CUBE 侧清理时用别名分析（`AliasAnalysis`）构建的存储依赖图，`getExecAfter(op)` 得到"必须排在 op 之后执行"的 op 集合 |
| `sameBlockIdExecAfter` | 预计算的"同 block_id 且 exec-after 当前 op"集合；判断无结果 op 能否删除 |
| `getLoopDirectChildBlockId` | 取 op 所属主循环直接子块的 block_id：沿父链找第一个"父 op 是主循环"的祖先，取其 `kBlockId` |
| `areBlockIdsConsecutive` | 校验每个 block_id 的 op 连续成段、不交错（`[1,1,2,2]` 合法，`[1,2,1,2]` 非法） |

---

## 三、逐行讲解

### 3.1 入口：runOnOperation（CloneOps.cpp L554-L598）

```cpp
void CloneOpsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  // Validate block_ids are consecutive before cloning
  if (failed(validateBlockIdsConsecutive(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Clone ops in vector/cube so each block_id owns its ops (no sharing)
  auto walkResult = module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
      return WalkResult::advance();
    }
    if (failed(cloneOpsInMainLoop(op))) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return WalkResult::interrupt();
    }
    if (failed(cleanupClonedOpsInMainLoop(op))) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return;
  }

  // Validate no cloned tensor ops remaining in VECTOR main_loop op
  if (failed(validateClonedOpsInVector(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
}
```

**要点**：
- **三阶段**：先整体校验 block_id 连续 → 逐个 main loop 克隆+清理 → 再校验 Vector 无 tensor 克隆残留；
- 任何阶段失败即置 fallback 返回；
- `isMainLoopOp` 同时匹配 `scf.for` 与 `scf.while`（带 `ssbuffer.main_loop`）。

### 3.2 克隆核心：cloneOpsInMainLoop（L172-L197）

```cpp
LogicalResult CloneOpsPass::cloneOpsInMainLoop(Operation *op) {
  Block *bodyBlock = MainLoop(op).getBody();
  ...
  llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
  if (failed(collectOpsByBlockId(op, blockOps))) {
    return failure();
  }
  SmallVector<int> idsInOrder = getBlockIdsInOrder(op);

  for (int i = idsInOrder.size() - 1; i >= 0; --i) {
    int curId = idsInOrder[i];
    SmallVector<int> earlierIds(idsInOrder.begin(), idsInOrder.begin() + i);

    if (failed(cloneOpsForBlock(curId, blockOps[curId], earlierIds, blockOps,
                                bodyBlock))) {
      return failure();
    }
  }
  return success();
}
```

**要点**：**反向遍历**（最后一个 block 开始）。对第 i 个 block，`earlierIds = ids[0..i-1]` 是它之前的所有 block；`cloneOpsForBlock` 把它们的 op 克隆到当前 block 开头。

### 3.3 单块克隆：cloneOpsForBlock（L114-L169）

```cpp
static LogicalResult
cloneOpsForBlock(int curId, SmallVector<Operation *> &curOps,
                 const SmallVector<int> &earlierIds,
                 const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
                 Block *bodyBlock) {
  if (curOps.empty() || earlierIds.empty()) {
    return success();
  }

  // Collect all ops from earlier blocks to clone
  SmallVector<Operation *> toClone;
  for (int eid : earlierIds) {
    llvm::append_range(toClone, blockOps.lookup(eid));
  }
  ...
  llvm::DenseMap<Value, Value> valueMap;
  SmallVector<Operation *> clonedOps;
  OpBuilder builder(curOps.front());

  for (Operation *op : toClone) {
    Operation *cloned = cloneOpWithMapping(op, builder, valueMap);
    cloned->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(curId));
    if (auto origBlockIdOpt = CVPipeline::getOpBlockId(op)) {
      cloned->setAttr(
          CVPipeline::kClone,
          builder.getI32IntegerAttr(static_cast<int32_t>(*origBlockIdOpt)));
    }
    clonedOps.push_back(cloned);
  }

  curOps.insert(curOps.begin(), clonedOps.begin(), clonedOps.end());
  ...
  for (Operation *op : curOps) {
    if (failed(updateCloneMapping(op, valueMap, yieldValues))) {
      return failure();
    }
  }

  if (failed(topologicalSort(curOps))) {
    return failure();
  }
  return success();
}
```

**要点**：
- 克隆出的 op **block_id 改为 curId**（属于当前块），并打 `kClone` 记源 block_id；
- `updateCloneMapping` 递归更新 operand：不在 yieldValues 里的旧值按 valueMap 重映射；**yield operand 跳过**——它们是 loop-carried 迭代参数，引用的是"本迭代的块外值"，不应指向克隆副本；
- 最后拓扑排序保证克隆链依赖序。

### 3.4 克隆的 op 映射：updateCloneMapping / cloneOpWithMapping（L55-L110）

```cpp
static LogicalResult
updateCloneMapping(Operation *op, llvm::DenseMap<Value, Value> &valueMap,
                   const llvm::DenseSet<Value> &yieldValues) {
  ...
  for (OpOperand &operand : op->getOpOperands()) {
    // Only skip if this operand is a yield value from the main_loop body.
    Value v = operand.get();
    if (yieldValues.contains(v)) {
      continue;
    }
    auto it = valueMap.find(v);
    if (it != valueMap.end()) {
      if (it->second.getType() != v.getType()) {
        return failure();
      }
      operand.set(it->second);
    }
  }
  // 递归处理嵌套 region
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (failed(updateCloneMapping(&nestedOp, valueMap, yieldValues))) {
          return failure();
        }
      }
    }
  }
  return success();
}
```

```cpp
static Operation *cloneOpWithMapping(Operation *op, OpBuilder &builder,
                                     llvm::DenseMap<Value, Value> &valueMap) {
  IRMapping mapper;
  // Populate mapper with ALL previously cloned values (not just the current
  // op's results).
  for (const auto &entry : valueMap) {
    mapper.map(entry.first, entry.second);
  }

  Operation *cloned = builder.clone(*op, mapper);
  for (auto it : llvm::zip(op->getResults(), cloned->getResults())) {
    valueMap[std::get<0>(it)] = std::get<1>(it);
  }
  return cloned;
}
```

**要点**：`cloneOpWithMapping` 把**全部历史映射**灌进 mapper 而不是只当前 op 的 operand——因为 `builder.clone(op, mapper)` 只重映射 op 的直属 operand，深层 operand 需要 mapper 已有所有上游映射。克隆后把结果注册进 valueMap 供后续 op 使用。

### 3.5 清理冗余克隆：cleanupClonedOpsInMainLoop / cleanupClonedOps（L294-L438）

```cpp
// Cleans up cloned ops in a main-loop op.
static LogicalResult
cleanupClonedOps(Operation *mainLoopOp, Block *bodyBlock,
                 llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
                 const SmallVector<int> &idsInOrder, bool isCube,
                 std::function<MemDepGraph(Operation *)> memGraphFactory) {
  for (int i = idsInOrder.size() - 1; i >= 0; --i) {
    auto &curOps = blockOps[idsInOrder[i]];
    ...
    // Find last index of cloned ops
    int startIdx = -1;
    for (int j = curOps.size() - 1; j >= 0; --j) {
      if (curOps[j]->hasAttr(CVPipeline::kClone)) {
        startIdx = j;
        break;
      }
    }
    ...
    // Erase cloned ops bottom-to-top
    for (int j = startIdx; j >= firstClonedIdx; --j) {
      Operation *op = curOps[j];
      bool shouldErase =
          isCube ? shouldEraseOpForCube(op, sameBlockIdExecAfter, erasedOps)
                 : shouldEraseOpForVector(op);
      if (shouldErase) {
        op->erase();
        erasedOps.insert(op);
      }
    }
  }
  return validateClonedSyncOpsErased(bodyBlock);
}
```

**要点**：
- 克隆后每个 block 开头有一段**连续的克隆 op 后缀**（从 `firstClonedIdx` 到 `startIdx`），这些是先前 block 的副本；
- 其中有些克隆是**冗余**的（如 `SyncBlockWaitOp`/`SyncBlockSetOp`/`FixpipeOp` 这类同步/搬移 op，重复克隆会重复握手），必须删掉只留一份；
- **CUBE vs Vector 删除规则不同**：
  - Vector（`shouldEraseOpForVector`）：只要结果无外部使用（`use_empty`）就删；
  - CUBE（`shouldEraseOpForCube`）：更保守——同步/fixpipe 直接删；有结果且被同 block_id 后续 op 使用则保留（SSA 依赖）；无结果的 consult `MemoryDependenceGraph` 的 `exec-after` 关系，同 block 还有未删的 exec-after op 则保留。

### 3.6 CUBE 删除判定：shouldEraseOpForCube（L200-L265）

```cpp
static bool shouldEraseOpForCube(
    Operation *op,
    const llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 4>>
        &sameBlockIdExecAfter,
    const llvm::DenseSet<Operation *> &erasedOps) {
  // Rule 1: SyncBlockWaitOp, SyncBlockSetOp, FixpipeOp -> directly erase
  if (isa<SyncBlockWaitOp>(op) || isa<SyncBlockSetOp>(op) ||
      isa<hivm::FixpipeOp>(op)) {
    return true;
  }
  ...
}
```

**要点**：三条规则递进——类型规则（同步/搬移必删）→ SSA 规则（结果仍被同 block 使用则留）→ 存储依赖规则（无结果时看 memgraph exec-after）。`erasedOps` 集合让"本轮已删除的 op"不再约束当前 op，等价于每次重建。

### 3.7 校验：validateBlockIdsConsecutive / validateClonedOpsInVector（L442-L552）

```cpp
static bool areBlockIdsConsecutive(Block *bodyBlock) {
  // 收集 block_id 序列
  ...
  // Check that each block_id forms a contiguous range
  for (size_t i = 0; i < idsInOrder.size();) {
    int currentId = idsInOrder[i];
    size_t j = i;
    while (j < idsInOrder.size() && idsInOrder[j] == currentId) {
      ++j;
    }
    for (size_t k = j; k < idsInOrder.size(); ++k) {
      if (idsInOrder[k] == currentId) {
        return false;  // interleaved
      }
    }
    i = j;
  }
  return true;
}
```

```cpp
LogicalResult CloneOpsPass::validateClonedOpsInVector(ModuleOp module) {
  // 只检查 VECTOR scope 的 main_loop
  ...
  for (Operation &bodyOp : bodyBlock->without_terminator()) {
    if (!bodyOp.hasAttr(CVPipeline::kClone)) {
      continue;
    }
    if (isa<tensor::EmptyOp>(&bodyOp)) {
      continue;
    }
    bool hasTensorDep = llvm::any_of(bodyOp.getResults(), [](Value result) {
      return isa<RankedTensorType>(result.getType());
    });
    if (hasTensorDep) {
      return WalkResult::interrupt();  // VECTOR main_loop 不允许 tensor 型克隆残留
    }
  }
  ...
}
```

**要点**：克隆前保证 block_id 连续（否则 cloneOpsForBlock 的"插到块开头"会打乱语义）；克隆后 Vector 主循环里不允许残留 tensor 型克隆 op（`tensor.empty` 除外）——Vector 侧 tensor 数据流应只走缓冲，不共享克隆张量。

---

## 四、流程总结

### 4.1 CloneOps 的流水

```
CloneOpsPass::runOnOperation
  ├─► validateBlockIdsConsecutive       校验 block_id 连续无交错（失败 → fallback）
  └─► 遍历每个 main_loop op
        ├─► cloneOpsInMainLoop
        │     ├─► collectOpsByBlockId     收集块内 op
        │     └─► 反向遍历 block（从最后开始）
        │           └─► cloneOpsForBlock
        │                 ├─► 克隆所有先前 block 的 op（block_id 改 curId，打 kClone）
        │                 ├─► updateCloneMapping   重映射 operand（跳过 yieldValues）
        │                 └─► topologicalSort      拓扑排序
        ├─► cleanupClonedOpsInMainLoop
        │     └─► cleanupClonedOps（反向遍历 block）
        │           ├─► 定位连续克隆后缀 [firstClonedIdx, startIdx]
        │           ├─► CUBE: shouldEraseOpForCube（类型/SSA/memgraph 三规则）
        │           └─► Vector: shouldEraseOpForVector（use_empty）
        └─► validateClonedOpsInVector    Vector 主循环无 tensor 型克隆残留
```

### 4.2 与前后子 Pass 的协作

- **前置**：`ComputeBlockOpt`/`AllocMultiCache` 打好的 `kBlockId`、`kMainLoop`；
- **产出**：每个 block 拥有独立 op 链，克隆冗余已清理，Vector 校验通过；
- **后置**：`ProcessArgs`（Step 1）在此基础上拆分共享 iter_arg——先有独立 op 才能安全拆分状态。

---

## 五、面试要点

1. **为什么需要 CloneOps？** 前序 Pass 注入的 op（多缓冲读写、同步 op）可能被多个 block 共享。共享状态下"每 block 独立控流"无法实现——一个 block 的计数器增减会污染另一个 block。克隆让每个 block 有完整独立的 op 链。

2. **为什么反向遍历（从最后 block 开始）？** 第 i 个 block 需要克隆 `ids[0..i-1]`（在它之前的 block）。反向遍历时，前面 block 的 op 尚未被修改，`blockOps` 快照保持稳定；同时"把克隆插到 curOps 开头"不会影响更早 block 的 op 位置。

3. **克隆后为什么还要清理？** 克隆是"全量复制先前 block"，但同步/搬移类 op（sync_block_wait/set、fixpipe）**重复出现会重复握手/重复搬移**，必须只保留一份。CUBE 侧因为有存储依赖（搬移顺序敏感），需要 `MemoryDependenceGraph` 判 exec-after；Vector 侧纯 SSA，看 use_empty 即可。

4. **updateCloneMapping 为什么跳过 yieldValues？** `scf.yield` 的 operand 是 loop-carried 迭代参数（指向外部或上一次迭代的值），不是本块产物，不应被重映射到克隆副本。嵌套 if/for 内的 yield 则仍要更新（它们引用块内中间值）。

5. **CUBE 和 Vector 删除策略为什么不同？** Vector 核内 op 大多无存储副作用，SSA use 判断即可；CUBE 核有搬移/同步的存储顺序约束，必须用 `AliasAnalysis` 构建的存储依赖图确认"同 block 内没有后续依赖 op"才能删。

6. **为什么 Vector 禁止 tensor 型克隆残留？** Vector 侧张量数据流应全部走缓冲（alloc/多缓冲），克隆出的 tensor 计算链会绕过多缓冲直接共享 SSA 值，破坏"生产者写缓冲、消费者读缓冲"的流水线模型。
