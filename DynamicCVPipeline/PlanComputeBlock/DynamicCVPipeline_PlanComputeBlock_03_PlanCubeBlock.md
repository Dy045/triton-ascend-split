# DynamicCVPipeline 子 Pass 逐行讲解：PlanCubeBlockPass

> 源码文件：[`third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/PlanCubeBlock.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/PlanCubeBlock.cpp)（595 行）
>
> 头文件：[`third_party/ascend/include/DynamicCVPipeline/PlanComputeBlock/PlanCubeBlockPass.h`](../../../third_party/ascend/include/DynamicCVPipeline/PlanComputeBlock/PlanCubeBlockPass.h)
>
> 流水线位置：`PlanComputeBlockPass` 的 **Step 2**（OpClassifier 之后、PlanVectorBlock 之前）

---

## 一、这个 Pass 在做什么

给所有 **CUBE 系 op**（`ssbuffer.core_type == CUBE_ONLY`，包括 matmul 及其数据准备链）分配 `ssbuffer.block_id`，把它们聚成**可整体调度的计算块**。

采用**两阶段（two-phase）分块**策略：

1. **Phase 1（种子扩张）**：以每个 matmul 为中心，`matchSeed` 匹配其"必然同组的输入输出"，再用 `SeedRegionPlanner` 做 BFS 扩张，把"紧密相连"的 CUBE op 聚成第一个计算块。
2. **Phase 2（拓扑补全）**：剩余 CUBE op 按拓扑顺序处理（`TopologicalPartitionPlanner`），每波（wave）互相独立的就绪 op 构成一个块，跨同步栅栏的波自动按段切分。

**设计目标**：一个计算块 = 一个核上一次执行的连续窗口。因此必须保证：
- 块内无依赖环（`DependencyCycleDetector` 保证并入不产生环）；
- 块不跨同步栅栏（`SyncWall` 保证）；
- 以 matmul 为核心，让"load → transpose → matmul → store"整链进同一块（减少核间传输）。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `isCubeSimpleOpOrCf` | CUBE op 且非同步 op 且非 terminator（`!isSyncOp && core_type==CUBE_ONLY`） |
| `matchSeed` | 从 matmul 出发，匹配合法的输入种子（transpose/to_tensor/fill/empty/broadcast/expand/extf）和输出种子（store/materialize/view/extract_slice/L0C cascade matmul） |
| `SeedRegionPlanner` | Phase 1 的 BFS 扩张器：从 seeds 出发，沿依赖边把"可并入且无环"的 CUBE op 加进 group |
| `TopologicalPartitionPlanner` | Phase 2 的拓扑调度器：处理剩余 CUBE op，一波一个块 |
| `DependencyCycleDetector` | DFS 环检测：`{block, depHelper, okSet, bm}`，检测把候选组并入后是否成环 |
| `SyncWall` | 按源码顺序标记同步栅栏位置；`segmentOf(op)` 给 op 分段，`sameSegment(a,b)` 判断两者之间有无同步 |
| `isOnlyDirectlyUse` | 判断 def 的消费是否只经过 matmul（无旁路用户），保证种子并入安全 |
| `fuseMarkOpToDef` | `annotation.mark` 是"跟班" op，并入其 src 定义 op 的块（同段、无环才并入） |
| `inheritFromParent` | 纯 CUBE scf 父 op 的子 block 直接继承父 id（见 01 篇 3.3.4） |

---

## 三、逐行讲解

### 3.1 主流程：`runOnOperation`（L554-L590）

```cpp
void mlir::triton::PlanCubeBlockPass::runOnOperation() {
  auto moduleOp = getOperation();
  if (CVPipeline::hasFallbackAttr(moduleOp)) { return; }

  auto &aa = getAnalysis<AliasAnalysis>();
  MemoryDependenceGraph memGraph{moduleOp, aa};          // 内存依赖图
  DependencyHelper depHelper{memGraph};                  // SSA+内存依赖遍历器
  auto bm = ComputeBlockIdManager(moduleOp);             // 扫描已有 id（含同步 op 的）

  auto result = moduleOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    // 纯 CUBE 的 scf 父 op 的子 block：直接继承父 id，跳过分块
    if (bm.shouldInheritFromParent(block, CoreType::CUBE_ONLY)) {
      if (llvm::failed(bm.inheritFromParent(block))) { ... return interrupt; }
      return WalkResult::advance();
    }
    if (llvm::failed(processBlockWithCubeBFS(block, depHelper, bm))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
  }
}
```

- **PreOrder walk**：外层 block 先处理。因为分块以"block 内 op"为单位，嵌套 block 在各自 walk 回调中独立处理。
- 继承分支：OpClassifier 把某个 scf.for/while 整体标成 CUBE 后，其子 block 直接 `inheritFromParent`（复用父 op 的 id），不再独立分块——整环一体。
- `memGraph` 是 `getAnalysis<AliasAnalysis>()`（pass 分析缓存）构建，与 OpClassifier 用的是不同实例（各自 pass 独立），但语义一致。

### 3.2 单 block 分块入口：`processBlockWithCubeBFS`（L494-L552）

```cpp
static llvm::LogicalResult
processBlockWithCubeBFS(Block *block, const DependencyHelper &depHelper,
                        ComputeBlockIdManager &bm) {
  llvm::DenseSet<Operation *> assigned;
  auto allDots = collectMatmulOps(block);      // 收集 block 内所有 matmul
  SyncWall wall(block);                        // 同步栅栏分段

  // Phase 1: 以 matmul 为核心的种子扩张
  for (auto *dot : allDots) {
    if (assigned.contains(dot)) continue;
    auto temBlockId = bm.getNextId();
    llvm::SmallVector<Operation *> dotSeeds = matchSeed(dot, bm, depHelper.memGraph);

    // 剪掉跨同步的种子（属于另一段的种子交给 Phase 2 独立发 id）
    llvm::erase_if(dotSeeds, [&](Operation *seed) {
      return !wall.sameSegment(seed, dot);
    });
    if (willCreateCycle(dotSeeds, depHelper.memGraph, temBlockId, bm)) {
      return llvm::failure();                  // 种子本身成环 → 失败
    }
    llvm::SmallVector<Operation *> newGroup;
    SeedRegionPlanner regionPlanner{dotSeeds, block, depHelper, assigned,
                                    newGroup, bm,    wall};
    regionPlanner.run();                        // BFS 扩张
    for (auto *op : newGroup) assigned.insert(op);
    if (llvm::failed(bm.markOpsWithNewId(newGroup))) return llvm::failure();
  }

  // Phase 2: 拓扑处理剩余 CUBE op
  TopologicalPartitionPlanner topoPlanner{block, assigned, depHelper, bm, wall};
  if (failed(topoPlanner.run())) return failure();
  fuseMarkOpToDef(block, bm, depHelper, wall);  // 收尾：mark 并入定义 op
  return llvm::success();
}
```

- 一个 matmul 一个种子组：matmul 与其直接相关的输入输出 op 先聚成"种子组"，再 BFS 扩张。`assigned` 保证每个 op 只被处理一次。
- **跨同步种子剪枝**：`matchSeed` 可能匹配到同步另一侧的 op（如 for 之外的 store），这些种子属于别的段，直接删掉——它们会在 Phase 2 拿到独立 id，从而"任何块都不跨同步"。
- `willCreateCycle`（L525 处的局部函数，基于 `DependencyCycleDetector`）：种子组内部就不允许有环，否则后续扩张全错。

### 3.3 种子匹配：`matchSeed`（L454-L488）

```cpp
static SmallVector<Operation *>
matchSeed(Operation *dotOp, ComputeBlockIdManager &bm,
          const MemoryDependenceGraph &memGraph) {
  SmallVector<Operation *> ret;
  ret.push_back(dotOp);                              // matmul 自己是第一个种子
  // ---- 输入侧 ----
  for (Value operand : dotOp->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def) continue;
    if (checkValidInputSeed(def) && isCubeSimpleOpOrCf(def) &&
        dotOp->getBlock() == def->getBlock() && bm.getBlockIdByOp(def) == -1) {
      if (CVPipeline::isOnlyDirectlyUse(def, dotOp, memGraph)) {
        ret.push_back(def);                          // 只被本 matmul 直接使用才并入
      }
    }
  }
  // ---- 输出侧：沿单用户链向后走 ----
  Operation *nowOp = dotOp;
  linalg::MatmulOp linkMatmul = llvm::dyn_cast<linalg::MatmulOp>(dotOp);
  while (nowOp->hasOneUse()) {
    auto user = *nowOp->getUsers().begin();
    if (user->getBlock() != dotOp->getBlock() || !isCubeSimpleOpOrCf(user) ||
        bm.getBlockIdByOp(user) != -1) break;
    if (checkValidUserSeed(user, linkMatmul)) {
      nowOp = user; ret.push_back(user);             // 链式追用户
    } else break;
  }
  return ret;
}
```

- `checkValidInputSeed`（L431-L436）：白名单 `transpose / to_tensor / fill / empty / broadcast / expand_shape / extf`——与 OpClassifier 的种子 pattern 保持一致（注释 `keep unify to OpClassifer`）。
- `checkValidUserSeed`（L437-L452）：输出侧白名单 `store / materialize / view / extract_slice`，外加一个特殊场景：**L0C cascade matmul**——下一个 matmul 把本 matmul 的结果当 `DpsInit[0]`（bias，L0C→L0C），这种"链条 matmul"也并入同组。
- **`isOnlyDirectlyUse`**：输入种子必须"只被本 matmul 直接使用"，否则并入会拖进无关消费链，破坏块内纯度。

### 3.4 Phase 1 扩张器：`SeedRegionPlanner`（L77-L151）

```cpp
bool SeedRegionPlanner::isEligible(Operation *op) {   // L121-L126
  if (!isCubeSimpleOpOrCf(op) || assigned.contains(op) || isMatmulOp(op))
    return false;                                      // 非 CUBE / 已分配 / matmul → 不可并入
  return !willCreateCycle(op);                         // 并入后不得产生环
}

bool SeedRegionPlanner::tryAddToGroup(Operation *op, Operation *from) {  // L128-L141
  if (!op || llvm::is_contained(group, op) || op->getBlock() != block ||
      !isEligible(op)) return false;
  // 永不跨同步：依赖边不能桥接同步点，否则 group 出现在栅栏两侧，
  // ReorderOpsByBlockId 造的 fence 会成环
  if (from && wall.hasSyncBetween(from, op)) return false;
  group.push_back(op);
  return true;
}

void SeedRegionPlanner::run() {                        // L143-L151
  size_t head = 0;
  while (head < group.size()) {
    Operation *currOp = group[head++];
    // AcrossIterArg 模式：跨迭代参数找上游源（穿透 for 迭代变量）
    depHelper.forEachSource<DependencyHelper::SourceMode::AcrossIterArg>(
        currOp, [&, this](Operation *source) { tryAddToGroup(source, currOp); });
  }
}
```

- `willCreateCycle`（L107-L114）：把 `group + op` 作为 `okSet`，DFS 检测是否存在环。**每个候选 op 都要跑一次环检测**，这是"扩张正确性"的保障——CUBE 块内若出现环，后续调度无法执行。
- `tryAddToGroup` 的 `hasSyncBetween(from, op)`：跨同步的边不允许把两个 op 拉进同一组（栅栏两侧必须分属不同块）。
- BFS 队列就是 group 本身（`head` 指针推进），天然按"先入先扩张"的宽度优先。

### 3.5 Phase 2 拓扑补全：`TopologicalPartitionPlanner`（L153-L378）

```cpp
TopologicalPartitionPlanner::TopologicalPartitionPlanner(...) {   // L183-L197
  initializeIndegreeForBlock(block, indegree, depHelper, bm);     // 入度只计跨块依赖
  block->walk([&](Operation *op) {
    if (op->getBlock() == block && isCubeSimpleOpOrCf(op) &&
        !assigned.contains(op)) nonAssignedCubeCnt++;             // 统计剩余 CUBE
  });
}

llvm::LogicalResult TopologicalPartitionPlanner::run() {          // L349-L378
  while (nonAssignedCubeCnt > 0) {
    if (failed(populateQueueWithReadyOps())) return llvm::failure();  // 入度 0 的 CUBE 入队
    if (queue.empty()) {
      if (failed(removeReadyNonCubeOps())) return llvm::failure();    // 死锁时绕过非 CUBE
      continue;
    }
    auto group = createNewGroupFromQueue();                        // 一队 = 一波
    // 跨同步的波按段切分，每段独立发 id
    llvm::DenseMap<unsigned, llvm::SmallVector<Operation *>> segmentGroups;
    for (auto *op : group) segmentGroups[wall.segmentOf(op)].push_back(op);
    for (auto &segGroup : segmentGroups)
      if (llvm::failed(bm.markOpsWithNewId(segGroup.second))) return llvm::failure();
  }
  return llvm::success();
}
```

- **拓扑波浪**：`createNewGroupFromQueue`（L322-L347）把当前所有"入度 0 且未分配"的 CUBE op 弹出组成一个 group，同时递减其用户入度；新入度 0 的继续入队。一轮 pop 完就是一个"波"（互相独立、可并行执行的计算块）。
- **按同步分段发 id**：一个波可能横跨多个同步段（比如两个独立 matmul 分别位于两个 barrier 之间），强行共用一个 id 会让块跨墙。按 `wall.segmentOf` 切分后每段独立 id，保住不变式。
- `removeReadyNonCubeOps`（L231-L258）+ `removeNonCubeOpsRecursively`（L205-L225）：**死锁绕行**。如果队列空但还有未分配 CUBE，说明有非 CUBE op 卡在前面（CUBE 等它的结果）。此时把"就绪的非 CUBE op"从依赖中绕过（递减其用户的入度），让 CUBE 解阻塞。若一轮绕行后无任何变化（`indegreeBefore == indegree && beforeVisitedSize == bypassVisited.size()`），说明真的卡死 → 返回 failure（触发 fallback）。
- `canExpandTo`（L262-L268）：`indegree == 0` 即可扩张（不检查环——拓扑顺序天然无环）。
- `dumpQueueAndIndegreeInfo`（L271-L306）：失败诊断输出（非 debug 也打印，便于定位死锁原因）。

### 3.6 收尾：`fuseMarkOpToDef`（L390-L429）

```cpp
static void fuseMarkOpToDef(Block *block, ComputeBlockIdManager &bm,
                            const DependencyHelper &depHelper, const SyncWall &wall) {
  for (auto *op : llvm::make_pointer_range(block->getOperations())) {
    if (getOpCoreType(op) != CUBE_ONLY) continue;
    auto markOp = llvm::dyn_cast<annotation::MarkOp>(op);
    if (!markOp) continue;
    auto *defOp = markOp.getSrc().getDefiningOp();
    if (!defOp) continue;
    if (!wall.sameSegment(markOp, defOp)) continue;    // 跨同步不并
    auto defBlockId = bm.getBlockIdByOp(defOp);
    if (defBlockId == -1) continue;
    auto currGroup = bm.getOpsInSameBlock(defOp);
    llvm::DenseSet<Operation *> newGroup{currGroup.begin(), currGroup.end()};
    if (newGroup.contains(markOp)) continue;
    newGroup.insert(markOp);
    DependencyCycleDetector dfs{block, depHelper, newGroup, bm};
    if (!dfs.detectCycle()) {                          // 并入后无环才更新 id
      bm.updateBlockId(markOp, defBlockId);
    }
  }
}
```

- `annotation.mark` 是"元信息 op"（标注某个 value 的来源），本身不计算。它必须**紧跟其 src 定义 op**（同一 block id），否则移动时注解与定义分离。
- 同样遵守两条铁律：**同段**（`sameSegment`）且**无环**（cycle detector）。用 `updateBlockId`（强制覆盖）而非 `markAndRecord`，因为 mark 可能已有其他 id。

---

## 四、算法流程总结

```
PlanCubeBlockPass
└─ walk 每个 Block（PreOrder）
   ├─ 继承分支：纯 CUBE scf 父 op → 子 block 直接 inheritFromParent
   └─ processBlockWithCubeBFS
      ├─ Phase 1（种子扩张）
      │   └─ 每个 matmul：
      │      ├─ matchSeed：输入白名单（transpose/to_tensor/fill/empty/broadcast/
      │      │              expand/extf，需 isOnlyDirectlyUse）+ 输出链（store/
      │      │              materialize/view/extract_slice/L0C-cascade matmul）
      │      ├─ 剪跨同步种子 → willCreateCycle 预检
      │      └─ SeedRegionPlanner.run：BFS 沿上游扩张
      │         ├─ isEligible：CUBE、未分配、非 matmul、并入无环
      │         └─ tryAddToGroup：跨同步拒并
      │         └─ 整组 markOpsWithNewId
      ├─ Phase 2（拓扑补全）
      │   └─ TopologicalPartitionPlanner.run（while 剩余 CUBE > 0）
      │      ├─ 入度 0 的 CUBE 入队（入度只计跨块依赖）
      │      ├─ 队列空 → removeReadyNonCubeOps 绕行非 CUBE（无进展 → failure）
      │      └─ createNewGroupFromQueue 一波 = 一组，按 segmentOf 切段独立发 id
      └─ fuseMarkOpToDef：mark 并入定义 op 的块（同段 + 无环）
```

**两条不变式**：① 任何 CUBE 块内部无依赖环（每个并入点都过 cycle detector）；② 任何 CUBE 块不跨同步栅栏（种子剪枝 + `tryAddToGroup` + 分段发 id 三重保证）。

---

## 五、面试要点

1. **为什么分块是"两阶段"？**
   种子扩张（Phase 1）优先把 matmul 周围紧密相关的 op 聚拢——这是"性能优先"：load→transpose→matmul→store 整链进一块，减少核间传输；拓扑补全（Phase 2）兜底处理剩余 op——这是"正确性优先"：保证每个 CUBE op 最终都有归属。先精后粗，先局部后整体。

2. **`matchSeed` 为什么要求输入种子 `isOnlyDirectlyUse`？**
   种子并入的语义是"这块要整体调度"。如果某个 to_tensor 还被其他 matmul/op 使用，把它并入本块会把无关消费链也拖进块内，块就不再是"一个执行单元"。只有"只服务本 matmul"的输入才安全并入。

3. **`willCreateCycle` 在扩张时每步都查环，代价大吗？怎么理解它的必要性？**
   每个候选 op 一次 DFS（图规模是 block 内 op 数），整体是 O(V·(V+E))。但这是必要的：CUBE 块一旦成环，后续依赖排序（ReorderOpsByBlockId 的 Kahn）必然失败。用"正确性换性能"——分块阶段宁可慢，不可错。

4. **拓扑补全为什么需要"绕行非 CUBE op"？**
   剩余 CUBE op 可能依赖某个还没分块的非 CUBE op（如一个 VECTOR 侧标量），该非 CUBE op 的入度也是 0 但它不是 CUBE，不会被处理，于是 CUBE 永远等不到入度归零 → 死锁。`removeReadyNonCubeOps` 把这些就绪的非 CUBE op 从依赖图中"绕过"（对用户减入度），CUBE 才能解阻塞。若绕行一轮毫无进展，说明存在真正的环，报 failure。

5. **"按同步分段发 id"具体解决了什么？**
   一个拓扑波里的 op 可能分居不同同步段（栅栏两侧）。若不分段，波内 op 共用一个 id，块就横跨栅栏——重排阶段栅栏 fence 会失效（op 跨墙移动）。按 `wall.segmentOf` 切分后，每段独立 id，块严格落在单侧。

6. **`annotation.mark` 为什么要"跟班"并入定义 op？**
   mark op 只是元信息注解（记录 value 来源），本身无计算、无调度价值。它必须随被注解的 value 一起移动，否则 value 被重排后注解悬空。并入条件（同段 + 无环）与正常分块完全一致，只是用 `updateBlockId` 强制覆盖。

7. **继承分支（inheritFromParent）与正常分块的分工？**
   正常分块把"块"定义为"同 block 内一起调度的 op 组"；继承把"整个纯 CUBE 循环"当作一个大块（父 op id = 块 id）。这避免了循环体内部再做细粒度分块（循环内部 op 本来就该整体跑 CUBE），也减少了 id 数量，降低后续 SplitDataflow 的处理规模。

8. **失败处理为什么统一 fallback 而不是就地修复？**
   分块是"分配决策"，一旦中途失败（种子成环、拓扑死锁），IR 上可能已经打了一半的 block id，状态不一致。此时无法保证后续 pass 正确性，直接 `ERRCODE_FAILED` 回退标准编译是最安全的选择——CV 流水线是"锦上添花"，失败必须无损回退。
