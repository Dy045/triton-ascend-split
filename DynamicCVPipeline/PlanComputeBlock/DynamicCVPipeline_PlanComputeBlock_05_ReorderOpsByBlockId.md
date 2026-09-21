# DynamicCVPipeline 子 Pass 逐行讲解：ReorderOpsByBlockIdPass

> 源码文件：[`third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.cpp)（587 行）
>
> 头文件：[`third_party/ascend/include/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.h`](../../../third_party/ascend/include/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.h)
>
> 流水线位置：`PlanComputeBlockPass` 的 **Step 4**（最后一个子 pass）

---

## 一、这个 Pass 在做什么

把每个基本块里的 op，**按 block id 分组重排**：同一个计算块的 op 聚在一起，块与块之间保持拓扑依赖顺序。这是 `PlanComputeBlock` 的收尾——分块只打了"组"的标签，**重排把组变成真正的物理顺序**。

三个层次的重排：

1. **op 级**：构建 block 内所有 op 的依赖 DAG（SSA + 内存）；
2. **group 级**：把 op 级 DAG 压缩成"计算块间 DAG"，Kahn 拓扑排序得到块的执行顺序；
3. **组内**：每个计算块内部再拓扑重排，VECTOR 块把 store 类 op 沉到末尾（sink）。

同时保证**同步栅栏 fence 不变式**：原源码顺序中"在同步 op 之前/之后"的 op，重排后仍然在该同步 op 之前/之后。

**为什么需要重排**：CUBE/VECTOR 分块是"逻辑分组"，但 IR 里 op 的物理顺序还是原样。不重排的话，不同计算块的交错 op 会破坏"一块接一块执行"的流水线假设（CUBE 块要连续执行才能重叠访存，VECTOR 块要在 CUBE 之后批量处理结果）。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `BlockOpGraph` | block 内 op 级依赖图：`opIndex`（op→位置）、`preds`（前驱）、`succs`（后继），SSA + 内存双维边 |
| `EdgeHelper` | 建图辅助：`resolveToBlockOp` 把嵌套 op 映射到 block 层面的代表 op，`addEdge` 去重加边 |
| `collectBlockIds` | 收集每个 op 的 block id；容器 op（无 id 的嵌套 scf 等）从内部 op 推断，推断失败则发新 id |
| `GroupAdjacencyGraph` | group 级 DAG：节点是 block id，边是跨组依赖 |
| `computeTopologicalOrder` | Kahn 拓扑排序 group；CUBE 起始块优先，VECTOR 块按 tensor compute op 数量稳定排序 |
| `isStoreLikeWithRegion` | op（或其嵌套区域）内是否有 `hivm.store` / `materialize_in_destination` |
| `orderInOneCBlock` | 单块内重排：非 VECTOR 块保持原序；VECTOR 块 store 下沉 + Kahn |
| `kMergeComputeBlockApplied` | `MergeComputeBlockPass` 的标记：若它运行但没合并任何块，本 pass 跳过重排 |
| sync fence 校验 | debug 下验证"同步 op 两侧 op 相对顺序不变" |

---

## 三、逐行讲解

### 3.1 主流程：`runOnOperation`（L534-L582）

```cpp
void ReorderOpsByBlockIdPass::runOnOperation() {
  auto moduleOp = getOperation();
  if (CVPipeline::hasFallbackAttr(moduleOp)) { return; }

  // MergeComputeBlockPass 的标记：运行过但没合并 → 跳过重排
  if (auto applied = moduleOp->getAttrOfType<BoolAttr>(
          CVPipeline::kMergeComputeBlockApplied)) {
    moduleOp->removeAttr(CVPipeline::kMergeComputeBlockApplied);  // 消费标记
    if (!applied.getValue()) {
      LOG_DEBUG("Skip reorder: MergeComputeBlock ran but merged nothing");
      return;
    }
  }

  auto &aa = getAnalysis<AliasAnalysis>();
  auto memGraph = MemoryDependenceGraph(moduleOp, aa);
  auto bm = ComputeBlockIdManager(moduleOp);
  auto result = moduleOp.walk([&](Block *block) {
    auto *parentOp = block->getParentOp();
    // 白名单：只在 func body / scf 结构内部重排
    if (!parentOp ||
        !(isa<func::FuncOp>(parentOp) ||
          isa<scf::SCFDialect>(parentOp->getDialect()))) {
      return WalkResult::skip();
    }
    if (llvm::failed(reorderOpsInBlock(*block, memGraph, bm))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
    return;
  }
}
```

- **`kMergeComputeBlockApplied` 标记处理**：`MergeComputeBlockPass`（ComputeBlockOpt 阶段，在本 pass 之前可能运行）会写这个标记表明"我运行过，且是否真的合并了"。如果合并了，本 pass 需要重排；如果没合并（比如已经是线性顺序），直接跳过省时，且标记被消费（`removeAttr`）避免泄漏到输出 IR。
- **重排白名单**：只在 `func.func` 体和 scf 方言结构内的 block 重排。linalg 区域、其他方言的内部 block 不重排（它们的顺序由所属 op 整体移动决定）。

### 3.2 单 block 重排入口：`reorderOpsInBlock`（L470-L532）

```cpp
static llvm::LogicalResult
reorderOpsInBlock(Block &block, const MemoryDependenceGraph &memGraph,
                  ComputeBlockIdManager &bm) {
  // 排除 terminator，收集要重排的 op
  const auto allOps = llvm::to_vector(llvm::make_pointer_range(block.without_terminator()));

  const BlockOpGraph graph{allOps, &block, memGraph};      // 1. op 级 DAG
  llvm::FailureOr<DenseMap<Operation *, int>> opBlockIdOpt =
      collectBlockIds(allOps, bm);                         // 2. 收集 block id
  ...
  const auto reorderedRes = buildReorderedOps(graph, opBlockId, bm, memGraph);
  if (failed(reorderedRes)) return failure();              // 3. group 拓扑 + 组内重排

  applyReorder(block, reorderedRes.value());               // 4. moveBefore 改写 IR

  // 5. （debug）校验 sync fence 不变式
  ...
  return llvm::success();
}
```

### 3.3 op 级 DAG：`BlockOpGraph`（L129-L150）

```cpp
BlockOpGraph::BlockOpGraph(ArrayRef<Operation *> allOps, Block *block,
                           const MemoryDependenceGraph &memGraph)
    : block(block), ops(allOps) {
  for (unsigned i = 0; i < allOps.size(); ++i) {
    opIndex[allOps[i]] = i;
    preds[allOps[i]];   // 确保每个节点都有条目
    succs[allOps[i]];
  }
  EdgeHelper edges(*this, block);
  DependencyHelper depHelper{memGraph};
  for (Operation *op : allOps) {
    depHelper.forEachSource(op, [&](Operation *source) {
      Operation *def = edges.resolveToBlockOp(source);
      edges.addEdge(def, op);                              // 前驱边
    });
    depHelper.forEachUser(op, [&](Operation *user) {
      edges.addEdgeToUser(op, user);                       // 后继边
    });
  }
}
```

- 用 `DependencyHelper` 同时遍历 SSA 源/用户和内存依赖（store→load）。
- `EdgeHelper::resolveToBlockOp`（L107-L116）+ `addEdgeToUser`（L94-L100）处理**嵌套 op**：scf/linalg 内部的 op 映射到"在目标 block 中的祖先"（`getAncestorInBlock`），依赖边挂在祖先层面；同层用户由 def 侧循环覆盖，避免重复边。
- `addEdge`（L118-L127）用 `seen` 集合去重，保证 pred/succ 无重复。

### 3.4 收集 block id：`collectBlockIds`（L152-L187）

```cpp
static llvm::FailureOr<DenseMap<Operation *, int>>
collectBlockIds(ArrayRef<Operation *> allOps, ComputeBlockIdManager &bm) {
  DenseMap<Operation *, int> opBlockId;
  for (Operation *op : allOps) {
    if (llvm::failed(verifyOpBlockId(op))) return llvm::failure();  // 校验 id 属性合法
    auto blockIdOpt = getOpBlockId(op);
    if (blockIdOpt.has_value()) { opBlockId[op] = blockIdOpt.value(); continue; }

    // 容器 op（如 scf.for）本身无 id：从其内部 op 推断
    auto result = op->walk([&](Operation *nestedOp) {
      if (nestedOp != op && !llvm::isa<scf::YieldOp, linalg::FillOp>(nestedOp))
        return WalkResult::interrupt();   // 遇到第一个非 yield/fill 内部 op 就停
      auto currBlockIdOpt = getOpBlockId(nestedOp);
      if (!blockIdOpt.has_value()) blockIdOpt = getOpBlockId(nestedOp);
      if (currBlockIdOpt.has_value() && currBlockIdOpt != blockIdOpt)
        return WalkResult::interrupt();   // 内部 id 不一致 → 放弃推断
      return WalkResult::advance();
    });
    if (result.wasInterrupted() || !blockIdOpt.has_value())
      blockIdOpt = bm.getNextId();        // 推断失败 → 发新 id（独立组）
    else
      bm.updateBlockId(op, blockIdOpt.value());  // 推断成功 → 容器归入该组
    opBlockId[op] = blockIdOpt.value();
  }
  return opBlockId;
}
```

- 大多数 op 已有 `ssbuffer.block_id`（CUBE/VECTOR 分块发的）。
- **容器 op 推断**：`scf.for` 这类 op 本身可能没 id（OpClassifier 给 scf 只标了 core_type 不一定标 block id），需要从其内部 op 推断——只检查 `yield` / `linalg.fill` 这类"边界 op"，若内部某 op 的 id 与外层一致则继承，否则发独立 id。`verifyOpBlockId` 保证已有 id 格式合法。

### 3.5 group 级 DAG：`GroupAdjacencyGraph`（L211-L259）

```cpp
GroupAdjacencyGraph::GroupAdjacencyGraph(
    const BlockOpGraph &g, const DenseMap<Operation *, int> &opBlockId,
    ComputeBlockIdManager &bm) : block(g.block), bm(bm) {
  // 1. 收集去重后的 block id（保持首次出现顺序）
  DenseSet<int> seenIds;
  for (Operation *op : g.ops) {
    int id = opBlockId.at(op);
    if (seenIds.insert(id).second) groupIds.push_back(id);
  }
  unsigned n = groupIds.size();
  succs.resize(n); inDeg.assign(n, 0);
  DenseMap<int, unsigned> groupPos;      // id → 下标
  ...
  // 2. 组间边：跨组依赖 + 去重
  DenseSet<std::pair<unsigned, unsigned>> addedEdges;
  for (Operation *op : g.ops) {
    unsigned fromIdx = groupPos[opBlockId.at(op)];
    for (Operation *succ : g.succs.at(op)) {
      unsigned toIdx = groupPos[opBlockId.at(succ)];
      if (fromIdx != toIdx && addedEdges.insert({fromIdx, toIdx}).second) {
        succs[fromIdx].push_back(toIdx);
        inDeg[toIdx]++;
      }
    }
  }
}
```

- **组间边规则**：同组内依赖（`fromIdx == toIdx`）不算，跨组依赖才构成组间 DAG 的边。这就是"CUBE 块 A → VECTOR 块 B"这类调度的依据。

### 3.6 group 级拓扑：`computeTopologicalOrder`（L265-L347）

```cpp
llvm::FailureOr<SmallVector<int>>
GroupAdjacencyGraph::computeTopologicalOrder() {
  SmallVector<unsigned> ready;
  SmallVector<unsigned> startingVectorBlocks;      // 入度 0 的 VECTOR 起始块
  for (auto [i, groupId] : llvm::enumerate(groupIds)) {
    if (inDeg[i] != 0) continue;
    auto ops = bm.getOpsRefByBlockId(groupId);
    if (ops.empty() ||
        getCoreTypeOfSimpleOpOrCf(ops.front()) == CUBE_ONLY) {
      ready.push_back(i);                          // CUBE 起始块直接就绪
    } else {
      startingVectorBlocks.push_back(i);           // VECTOR 起始块进待定区
    }
  }
  // 稳定排序：tensor compute op 多的 VECTOR 块优先（有更多计算要重叠）
  constexpr size_t kPriviledgedMaxComputeOpCnt = 1;
  std::stable_partition(startingVectorBlocks.begin(), startingVectorBlocks.end(),
                        [this](unsigned idx) {
                          const auto ops = bm.getOpsRefByBlockId(groupIds[idx]);
                          auto computeOpCnt = 0;
                          for (auto op : ops)
                            if (isTensorComputeOp(op)) computeOpCnt++;
                          return computeOpCnt > kPriviledgedMaxComputeOpCnt;
                        });
  ready.append(startingVectorBlocks);
  // Kahn：不断弹出就绪组，减后继入度
  while (!ready.empty()) {
    auto cur = ready.pop_back_val();
    result.push_back(groupIds[cur]);
    for (unsigned succIdx : succs[cur])
      if (--inDeg[succIdx] == 0) ready.push_back(succIdx);
  }
  if (result.size() == n) return result;
  // 失败诊断：定位 block 所在 region
  ...
  return llvm::failure();
}
```

- **CUBE 优先 + VECTOR 计算量排序**：CUBE 起始块（入度 0）直接进就绪区；VECTOR 起始块按"tensor compute op 数量"稳定排序——计算多的 VECTOR 块优先执行，因为 CUBE 执行时 VECTOR 块的计算能与它重叠（流水线设计意图）。
- 失败时输出具体 block 位置（region 索引）便于定位，返回 failure → fallback。

### 3.7 组内重排：`orderInOneCBlock`（L359-L429）

```cpp
static SmallVector<Operation *>
orderInOneCBlock(ArrayRef<Operation *> opsInSameBlock,
                 const MemoryDependenceGraph &memGraph) {
  // 规则：
  // 1. VECTOR 块把 store 类 op 沉到末尾
  // 2. 其他（CUBE 块）保持原序
  SmallVector<Operation *> originOrder(...);
  if (llvm::any_of(opsInSameBlock, [&](Operation *op) {
        return CVPipeline::getCoreTypeOfSimpleOpOrCf(op) != VECTOR_ONLY;
      })) {
    // CUBE 块不需要沉 store：CUBE 的 store 走 FIXPIPE，
    // C->V 传输紧贴 matmul，store 与传输无冲突
    return originOrder;
  }
  if (llvm::all_of(opsInSameBlock,
                   [&](Operation *op) { return !isStoreLikeWithRegion(op); })) {
    return originOrder;                              // 无 store 类 op，直接返回
  }
  // Kahn 重排：store 沉底 + 保持原程序顺序（opIndex 决胜）
  ...
  auto comesBefore = [&](Operation *a, Operation *b) {
    bool aStore = isStoreLikeWithRegion(a);
    bool bStore = isStoreLikeWithRegion(b);
    if (aStore != bStore) return !aStore;            // 非 store 优先
    return graph.opIndex.at(a) < graph.opIndex.at(b); // 同性质按原顺序
  };
  // 每次从 ready 中选"最好"的候选（std::min_element + comesBefore）
  ...
  return ordered;
}
```

- **只有纯 VECTOR 块沉 store**。CUBE 块保持原序——注释给了两个理由：① CUBE 的 store 走 FIXPIPE（专用固定流水线），无调度冲突；② CUBE→VECTOR 传输紧贴 matmul，store 下沉没有意义。
- `isStoreLikeWithRegion`（L349-L357）：op 或其嵌套区域内含 `hivm.store` / `materialize_in_destination` 即为"store 类"。
- 组内 Kahn 用 `comesBefore` 做 ready 集选择：**非 store 优先 + 同性质按原序**（`opIndex` 决胜），保证稳定。

### 3.8 组装与落地：`buildReorderedOps`（L432-L455）+ `applyReorder`（L458-L468）

```cpp
static llvm::FailureOr<SmallVector<Operation *>> buildReorderedOps(...) {
  GroupAdjacencyGraph adjacencyGraph{graph, opBlockId, bm};
  auto groupOrderResult = adjacencyGraph.computeTopologicalOrder();  // group 级顺序
  ...
  for (int const blockId : groupOrderResult.value()) {
    SmallVector<Operation *> originOrderOp;   // 按原程序序收集该组 op
    for (Operation *op : graph.ops)
      if (opBlockId.at(op) == blockId) originOrderOp.push_back(op);
    SmallVector<Operation *> orderedInOneCBlock =
        orderInOneCBlock(originOrderOp, memGraph);   // 组内重排
    reordered.append(orderedInOneCBlock);
  }
  return reordered;
}

static void applyReorder(Block &block, ArrayRef<Operation *> reordered) {
  Operation *terminator = block.mightHaveTerminator() ? block.getTerminator() : nullptr;
  for (Operation *op : reordered) op->moveBefore(&block, block.end());  // 依次移到末尾
  if (terminator) terminator->moveBefore(&block, block.end());          // terminator 最后
}
```

- 组内重排前按**原程序序**收集成员（`graph.ops` 顺序），再做块内拓扑——块内顺序在"满足依赖 + sink store"的前提下尽量贴近源码，减少意外行为。
- `applyReorder` 用 `moveBefore` 把 op 依次移到 block 末尾，最终顺序 = 重排结果 + terminator。

---

## 四、算法流程总结

```
ReorderOpsByBlockIdPass
├─ 前置：kMergeComputeBlockApplied 标记检查（合并未发生 → 跳过）
└─ walk 每个 Block（白名单：func body / scf 结构内）
   └─ reorderOpsInBlock
      ├─ 1. BlockOpGraph：op 级 DAG（SSA + 内存，嵌套 op 映射到 block 祖先）
      ├─ 2. collectBlockIds：收集 op 的 block id
      │     └─ 容器 op（无 id）从其内部推断；失败发新 id
      ├─ 3. buildReorderedOps
      │     ├─ GroupAdjacencyGraph：op DAG → group 级 DAG（跨组边）
      │     ├─ computeTopologicalOrder：Kahn
      │     │   ├─ CUBE 起始块直接就绪
      │     │   └─ VECTOR 起始块按 tensor compute 数量稳定排序
      │     └─ 每组 orderInOneCBlock：
      │         ├─ CUBE 块 / 无 store → 原序
      │         └─ VECTOR 块 → Kahn，非 store 优先 + 原序决胜（store 沉底）
      └─ 4. applyReorder：moveBefore 改写物理顺序（terminator 兜底）
      └─ 5. （debug）sync fence 不变式校验
```

**不变式**：同步 op 在源码顺序中的"前后关系"重排后不变（每 block 校验）。这与 CUBE/VECTOR 分块阶段"块不跨同步"的约束共同保证栅栏语义。

---

## 五、面试要点

1. **为什么分块之后还要重排？**
   分块是"逻辑分组"（打标签），重排是"物理落位"（改顺序）。CV 流水线的目标是让 CUBE 块与 VECTOR 块形成流水重叠：CUBE 算 matmul 的同时，VECTOR 侧处理上一轮的输出。如果物理顺序还是"原样交错"，两块无法连续执行，重叠无从谈起。重排让"块"成为调度单元。

2. **两级 DAG 的结构是什么？为什么需要两级？**
   第一级 op 级 DAG（block 内 op 间的 SSA+内存依赖）用于：收集 block id、构建组间边、组内排序；第二级 group 级 DAG（block id 间的依赖）用于：决定块的执行顺序。两级的原因是问题本身是两层的——"op 属于哪个组"和"组之间谁先谁后"。

3. **`computeTopologicalOrder` 里 CUBE 与 VECTOR 起始块的处理策略？**
   入度 0 的 CUBE 起始块直接就绪（CUBE 是"生产者"，应该先开始）；入度 0 的 VECTOR 起始块进入待定区，按"tensor compute op 数量"稳定排序，计算多的 VECTOR 块优先。这是为了最大化流水重叠：先排布"重计算"的 VECTOR 块，让它在 CUBE 执行期间并行跑。

4. **为什么 CUBE 块不沉 store，而 VECTOR 块要沉？**
   CUBE 的 store 走 FIXPIPE（固定流水线），与 CUBE 计算无调度冲突；且 CUBE→VECTOR 的传输紧贴 matmul 结果，store 是否沉底不影响。VECTOR 块则不同：store 是访存操作，沉到末尾可以避免阻塞中间计算，让计算先行、写回靠后，为下一组 VECTOR 计算腾出执行窗口。

5. **容器 op（scf.for）的 block id 怎么来？**
   它本身可能没有 id。`collectBlockIds` 从其内部 op 推断：只查 `yield` / `linalg.fill` 这类边界 op，若内部 op 的 id 一致则容器归入该组（`updateBlockId`）；内部 id 不一致或推断失败则发新 id（容器成为独立组）。推断失败发新 id 是安全的——容器作为整体移动，不影响组内成员。

6. **`kMergeComputeBlockApplied` 标记机制的作用？**
   它是 pass 间协作的"消息通道"：`MergeComputeBlockPass`（ComputeBlockOpt 阶段）合并相邻同类型块后写该标记；本 pass 读取它决定是否重排。若合并没发生（标记为 false），说明顺序已是最优，跳过重排省一次完整图遍历；标记会被消费（remove），避免污染输出 IR。

7. **sync fence 不变式为什么重要？怎么保证？**
   同步栅栏是核间协作点，两侧 op 相对顺序一旦改变，栅栏语义就错（比如把栅栏前的读放到栅栏后，导致读到旧数据）。保证手段有两层：分块阶段"块不跨同步"（同步 op 独享 id）＋ 重排阶段"同步 op 前后关系不变"（组间 DAG 的边天然约束 + debug 校验兜底）。

8. **这个 pass 失败意味着什么？**
   组间 DAG 有环（`result.size() != n`）时拓扑排序失败——说明前面分块产生了环（正常情况 cycle detector 已防住，这里是双保险）。此时 IR 分组状态已不可信，打 `ERRCODE_FAILED` fallback 回退标准编译。
