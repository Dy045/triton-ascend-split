# DynamicCVPipeline 子 Pass 逐行讲解：PlanVectorBlockPass

> 源码文件：[`third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/PlanVectorBlockPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/PlanVectorBlockPass.cpp)（669 行）
>
> 头文件：[`third_party/ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h`](../../../third_party/ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h)
>
> 流水线位置：`PlanComputeBlockPass` 的 **Step 3**（PlanCubeBlock 之后、ReorderOpsByBlockId 之前）

---

## 一、这个 Pass 在做什么

给所有 **VECTOR 系 op** 分配 `ssbuffer.block_id`。这是比 CUBE 分块更复杂的问题：CUBE 有 matmul 做锚点，VECTOR 是海量的元素级 op，需要**纯拓扑方法**分块。

核心思路（拓扑融合 + 两端切割）：

1. **拓扑初始化**：计算每个 op 的入度（只计跨块依赖），入度为 0 的可融合 op 是候选；
2. **融合**：不断从候选队列取出 op 加入当前组，更新其用户入度，新入度 0 者入队；
3. **切割（refine）**：一组融合完后，把"不应该留在组里"的 op 切出去（`refineFuseGroup` 正向、`reverseRefineFuseGroup` 反向）——组内只保留纯 VECTOR 可融合、且依赖关系完整的 op；
4. **分段发 id**：按同步栅栏分段，每段独立 id。

**与 CUBE 分块的关系**：VECTOR 分块依赖 CUBE 已分块——切割的依据之一是"与 CUBE 块相邻的边界 op 如何处理"（`findOpsAdjacentToCube`）。VECTOR 计算必须被切到与 CUBE 边界清晰的位置，后续才能做核间传输。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `isFusableOp` | 可融合进 VECTOR 块的 op：非同步 op + `isVectorSimpleOpOrCf` + 非 terminator |
| `indegree` | 拓扑入度；`initializeIndegreeForBlock` 只计**跨块**依赖（同块依赖不算） |
| `visited` | 已访问标记（进入过候选/组），防止重复处理 |
| `candidates`（queue） | 当前可融合的候选队列（入度为 0 的可融合 op） |
| bypass（绕行） | 队列空时，把"就绪但不可融合"的 op（如 CUBE op）跳过，释放其用户的入度 |
| `refineFuseGroup` | 正向切割：找"组内下一个节点是 CUBE op"的 op，把其依赖闭包保留，其余切出 |
| `reverseRefineFuseGroup` | 反向切割：找"组内前驱是 CUBE op"的 op，把其用户闭包保留，其余切出 |
| `extractWholeFuseGroup` | 兜底规则：无 CUBE 邻接时，按 loop-carried 依赖 / tensor 完整性决定切谁 |
| `SyncWall.segmentOf` | 同步分段，块不跨墙 |

---

## 三、逐行讲解

### 3.1 主流程：`runOnOperation`（L626-L660）

```cpp
void PlanVectorBlockPass::runOnOperation() {
  auto moduleOp = getOperation();
  if (CVPipeline::hasFallbackAttr(moduleOp)) { return; }

  auto &aa = getAnalysis<AliasAnalysis>();
  auto memDepGraph = MemoryDependenceGraph(moduleOp, aa);
  auto bm = ComputeBlockIdManager(moduleOp);          // 扫描已有 id（含 CUBE 块的）

  auto result =
      moduleOp.walk<WalkOrder::PreOrder>([&](Block *block) -> WalkResult {
        if (bm.shouldInheritFromParent(block, CoreType::VECTOR_ONLY)) {
          // 纯 VECTOR scf 父 op 的子 block 继承父 id
          if (llvm::failed(bm.inheritFromParent(block))) { ... }
          return WalkResult::advance();
        }
        if (llvm::failed(planVectorBlockId(block, memDepGraph, bm))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) {
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
  }
}
```

- 与 PlanCubeBlock 镜像结构：继承分支（纯 VECTOR scf）→ 否则 `planVectorBlockId`。
- `bm` 构造时扫描到 CUBE 块已有的 id（Phase 1 发的），因此 `getBlockIdByOp` 能区分"CUBE op / 未分配 op"，这是切割算法的关键输入。

### 3.2 可融合判定：`isFusableOp`（L55-L68）

```cpp
static bool isFusableOp(Operation *op) {
  if (CVPipeline::isSyncOp(op)) return false;        // 同步 op 不可融合（墙）
  if (isVectorSimpleOpOrCf(op)) {
    if (op->getBlock()->mightHaveTerminator() &&
        op == op->getBlock()->getTerminator()) return false;  // terminator 不可融合
    return true;
  }
  return false;
}
```

- VECTOR 块只包含"简单 VECTOR op 或 scf op"（`isVectorSimpleOpOrCf`），且排除同步 op 与 terminator。terminator（yield/return）是块的结尾标记，不能成为块成员。

### 3.3 候选发现与绕行（L70-L173）

```cpp
static void
passAndCollectCandidates(Operation *nowOp, DenseMap<Operation *, int> &indegree,
                         SmallVector<Operation *> &candidates,
                         DenseMap<Operation *, bool> &visited,
                         const MemoryDependenceGraph &memGraph,
                         ComputeBlockIdManager &bm) {   // L70-L108
  DependencyHelper depHelper{memGraph};
  depHelper.forEachUserInSameBlock(nowOp, [&](Operation *user) {
    if (!bm.isSameBlock(user, nowOp)) indegree[user]--;  // 跨块用户减入度
    if (!bm.isWholeCubeReady(user, indegree)) return;    // CUBE 块要整块就绪
    if (visited[user]) return;
    if (isFusableOp(user)) {
      visited[user] = true; candidates.push_back(user);
      return;
    }
    // 不可融合用户（如 CUBE 块）→ 以其所在块为粒度继续绕过
    for (auto *cubeop : bm.getOpsInSameBlock(user)) {
      if (!visited[cubeop]) {
        visited[cubeop] = true;
        passAndCollectCandidates(cubeop, indegree, candidates, visited, memGraph, bm);
      }
    }
  });
}
```

- **绕过不可融合 op**：当用户是不可融合 op（CUBE 块成员），不能简单跳过——需要把"该 op 所在块的所有成员"都处理一遍（`getOpsInSameBlock`），因为 CUBE 块是整体调度的。`byPassNonFusable`（L110-L128）是它的顶层入口：扫描所有"整块就绪但不可融合"的 op，递归绕过。
- `findCandidates`（L156-L173）：候选空时先尝试 bypass，再把所有"入度 0 + 可融合 + 未访问"的 op 加入候选。

### 3.4 融合主循环：`planVectorBlockId`（L537-L609）

```cpp
llvm::LogicalResult
planVectorBlockId(Block *block, const MemoryDependenceGraph &memGraph,
                  ComputeBlockIdManager &bm) {
  DenseMap<Operation *, int> indegree;
  DenseMap<Operation *, bool> visited;
  initializeIndegreeForBlock(block, indegree, DependencyHelper{memGraph}, bm);
  SyncWall wall(block);                                  // 同步分段

  // 初始化 visited + 首批候选
  block->walk([&](Operation *op) {
    if (op->getBlock() == block) {
      visited[op] = false;
      if (isFusableOp(op) && indegree[op] == 0) {
        visited[op] = true; queue.push_back(op);
      }
    }
  });
  findCandidates(indegree, queue, visited, memGraph, bm);

  SmallVector<Operation *> nowFuseGroup;
  while (!queue.empty()) {
    auto nextFused = queue.front();
    if (nextFused) {
      nowFuseGroup.push_back(nextFused);                 // 融合一个
      updateCandidates(nextFused, queue, indegree, visited, memGraph);
    }
    if (queue.empty() || nextFused == nullptr) {         // 一组融合完
      refineFuseGroup(block, nowFuseGroup, visited, queue, indegree, memGraph, bm);
      reverseRefineFuseGroup(block, nowFuseGroup, visited, queue, indegree, memGraph, bm);
      // 按同步分段，每段独立发 id
      DenseMap<unsigned, SmallVector<Operation *>> segmentGroups;
      for (auto *op : nowFuseGroup) segmentGroups[wall.segmentOf(op)].push_back(op);
      for (auto &segGroup : segmentGroups)
        if (llvm::failed(bm.markOpsWithNewId(segGroup.second))) return failure();
      nowFuseGroup.clear();
      findCandidates(indegree, queue, visited, memGraph, bm);  // 找下一组
    }
  }
  return llvm::success();
}
```

- `updateCandidates`（L130-L154）：融合 `nextFused` 后，从候选移除它；遍历其同 block 用户，减入度，新入度 0 且可融合者入队。
- **一组融合完 → 先切后发 id**：融合过程中可能把不该在一起的点混进组（比如与 CUBE 边界纠缠的点），所以融合完先 `refineFuseGroup` + `reverseRefineFuseGroup` 双向切割，把组净化后再分配 id。

### 3.5 正向切割：`refineFuseGroup`（L407-L445）

```cpp
void refineFuseGroup(Block *block, SmallVector<Operation *> &nowFuseGroup,
                     ... , ComputeBlockIdManager &bm) {
  // 1. 找组内"用户是 CUBE op"的 op → toProcess
  auto toProcess = findOpsAdjacentToCube(block, nowFuseGroup, visited, memGraph, bm);
  // 2. 若无 CUBE 邻接，用兜底规则提取
  if (toProcess.empty()) {
    toProcess = extractWholeFuseGroup(block, nowFuseGroup, bm);
  }
  // 3. 兜底后仍空 → 无 op 可切（但先找候选，防止死循环）
  if (toProcess.empty()) {
    findCandidates(indegree, candidates, visited, memGraph, bm);
    if (candidates.empty()) return;   // v1→v2→yield 场景：切了还会被切回，直接保留
  }
  // 4. 收集 keepOps：toProcess 的传递依赖闭包（数据+内存+循环携带）
  auto keepOps = collectKeepOpsToCube(block, toProcess, nowFuseGroup, memGraph);
  // 5. 切掉非 keepOps，恢复 BFS 状态
  evictAndRestoreState(block, keepOps, nowFuseGroup, visited, candidates, indegree, memGraph);
}
```

- `findOpsAdjacentToCube`（L175-L225）：扫组内 op 的用户，找"不可融合且未访问"的用户（即 CUBE 块成员），取**最小 block id** 的 CUBE 块（越小的 block id 表示在 IR 中越靠前的 matmul，优先对齐前面的 CUBE）；再回扫，把"用户属于该最小 CUBE 块"的组内 op 收集为 toProcess。
- `collectKeepOpsToCube`（L227-L265）：toProcess 里每个 op，沿 `forEachSource<AcrossIterArg>` 收集其上游（含跨迭代参数），闭包内所有"组内" op 保留；特殊处理 `annotation.mark`：其 src 定义 op 在 keepOps 中时 mark 也保留（mark 必须跟班）。
- `extractWholeFuseGroup`（L320-L364）：**兜底规则**。无 CUBE 邻接时：
  - 组内无 tensor op → 全部保留（纯标量/索引组，无切割必要）；
  - 有 tensor 且 block 是 scf.for/while 体 → 若 op 的 loop-carried 依赖定义 op 属于**别的未分配块**（`bm.getBlockIdByOp(defOp) == -1` 且不在本组），说明该 op 依赖跨循环携带值，切掉它及其用户闭包（`collectAllUsersInFuseGroup`）；
  - 切完若剩余非 tensor → 返回空（整组切掉）。
- `evictAndRestoreState`（L366-L405）：状态回滚——把被切 op 从组中移除、恢复其用户的入度（`indegree[user]++`）、重置 visited，候选池重建（旧候选 + 被切 op 中入度为 0 者）。**关键**：被切 op 可以再次作为候选被后续组融合。

### 3.6 反向切割：`reverseRefineFuseGroup`（L511-L534）

```cpp
void reverseRefineFuseGroup(Block *block, SmallVector<Operation *> &nowFuseGroup,
                            ...) {
  // 1. 找组内"前驱是 CUBE op"的 op → toProcess
  auto toProcess = findOpsAdjacentFromCube(block, nowFuseGroup, visited, memGraph, bm);
  // 2. 无则直接返回（反向没有兜底规则，交给正向）
  if (toProcess.empty()) return;
  // 3. 收集 keepOps：toProcess 的传递用户闭包
  auto keepOps = collectKeepOpsFromCube(block, toProcess, nowFuseGroup, memGraph);
  // 4. 切掉非 keepOps，恢复状态
  evictAndRestoreState(block, keepOps, nowFuseGroup, visited, candidates, indegree, memGraph);
}
```

- 与正向对称：正向处理"组 → CUBE"边界（组内 op 被 CUBE 使用），反向处理"CUBE → 组"边界（组内 op 的输入来自 CUBE）。
- `findOpsAdjacentFromCube`（L447-L484）沿 `forEachSourceInSameBlock` 找 CUBE 前驱，同样取最小 block id；`collectKeepOpsFromCube`（L486-L509）沿用户方向收集闭包。
- 为什么两遍都必要：VECTOR 块与 CUBE 块可能互相纠缠（CUBE 输出喂 VECTOR、VECTOR 输出喂 CUBE），单方向切割只能解一边的边界，两遍保证块与 CUBE 的依赖关系"干净"。

---

## 四、算法流程总结

```
PlanVectorBlockPass
└─ walk 每个 Block（PreOrder）
   ├─ 继承分支：纯 VECTOR scf 父 op → 子 block inheritFromParent
   └─ planVectorBlockId
      ├─ initializeIndegreeForBlock（入度只计跨块依赖）
      ├─ 首批候选：入度 0 的可融合 op
      ├─ while 候选非空：
      │  ├─ 融合一个 → updateCandidates（减用户入度，新 0 者入队）
      │  ├─ 队列空 → 一组完成：
      │  │   ├─ refineFuseGroup（正向切割）
      │  │   │   ├─ findOpsAdjacentToCube：邻接最小 id CUBE 块 → toProcess
      │  │   │   ├─ 兜底 extractWholeFuseGroup（无 CUBE 邻接时）
      │  │   │   ├─ collectKeepOpsToCube（toProcess 上游闭包 + mark 跟班）
      │  │   │   └─ evictAndRestoreState（切掉非 keep，恢复入度/visited/候选）
      │  │   ├─ reverseRefineFuseGroup（反向切割，逻辑对称）
      │  │   └─ 按 segmentOf 分段 → markOpsWithNewId
      │  └─ findCandidates 找下一组（候选空时 byPassNonFusable 绕过 CUBE）
      └─ 失败 → fallback
```

**设计要点**：融合是"贪婪 + 可回滚"的。贪心地聚组降低块数（减少传输），但融合后必须双向净化（refine/reverse），把与 CUBE 边界纠缠的 op 切回候选池，交给下一组重新融合。被切 op 不丢失，只是推迟分组。

---

## 五、面试要点

1. **VECTOR 分块为什么比 CUBE 分块难？**
   CUBE 有 matmul 锚点 + pattern 种子，天然有"组团中心"；VECTOR 全是同质的元素级 op，没有天然中心。只能靠拓扑（入度）+ 贪心融合 + 事后切割来形成组。而且 VECTOR 组必须与 CUBE 块边界对齐（后续要插核间传输），所以切割以"CUBE 邻接"为主要依据。

2. **为什么融合完成之后才切割，而不是融合时判断？**
   融合是局部的（只看入度），无法预知整组的边界形态；只有组完整形成后，才能看到"组内哪些 op 与 CUBE 边界纠缠"。所以采用"先融合、后净化"两段式，切割把错误成员送回候选池，下一组再试。这是一种"过估计再修正"的策略。

3. **正向切割与反向切割的区别？**
   正向（refineFuseGroup）处理"组内 op 的结果被 CUBE 消费"——保留这些 op 及其上游闭包（它们是 CUBE 的输入生产者）；反向（reverseRefineFuseGroup）处理"组内 op 的输入来自 CUBE"——保留这些 op 及其下游用户闭包。双向互补，确保组内每个 op 与 CUBE 的依赖关系都闭环。

4. **`evictAndRestoreState` 为什么还要恢复入度和 visited？**
   被切出的 op 必须回到"未融合"状态：恢复其用户的入度（因为之前融合时减过），重置 visited，让它可以重新进入候选池、被下一组融合。如果不恢复，被切 op 就永远丢失了（既不在本组也不在候选），VECTOR 覆盖就不完整。

5. **`extractWholeFuseGroup` 的兜底规则在解决什么问题？**
   当组内没有 CUBE 邻接 op 时，正向切割没有依据。兜底检查 loop-carried 依赖：若组内 op 依赖了"属于其他未分配块的循环携带值"（`getLoopCarriedDefOp` 返回的 def 既不在本组也未分配 id），这个 op 的依赖跨了循环边界，必须切掉。这保证 VECTOR 块不会吞下跨循环依赖的 op。

6. **"候选空时绕过 CUBE 块"（byPassNonFusable）与 CUBE 分块的"绕行非 CUBE op"是同一个思路吗？**
   是同一思路的镜像。CUBE 分块时，CUBE 等非 CUBE 上游 → 绕过非 CUBE；VECTOR 分块时，VECTOR 等 CUBE 上游 → 绕过 CUBE（且按 CUBE 块整体绕过，因为 CUBE 块是整块调度的）。两处都是为了打破"等待链"上的死锁。

7. **为什么每个 VECTOR 组也要按同步分段发 id？**
   与 CUBE 完全相同的理由：一个拓扑波可能跨多个同步段，共用一个 id 会让块横跨栅栏，重排 fence 失效。`wall.segmentOf(op)` 分段、每段独立 `markOpsWithNewId`。

8. **这个 pass 结束后 IR 的形态是什么？**
   每个 op 都有 `ssbuffer.core_type`（CUBE/VECTOR）和 `ssbuffer.block_id`（正整数），CUBE 与 VECTOR 的 id 是统一编号的。此时"谁跟谁一组"完全确定，下一步 `ReorderOpsByBlockId` 只需要按组搬动 op 顺序即可。
