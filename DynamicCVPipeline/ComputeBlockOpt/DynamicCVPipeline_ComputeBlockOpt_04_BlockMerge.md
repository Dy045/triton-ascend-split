# DynamicCVPipeline 子 Pass 专题：块合并（04）

> 覆盖 `ComputeBlockOptPass` 阶段 D 的 4 个子 Pass：
> `MergeVectorIfBlock` / `MergeComputeBlock` / `MergeCubeBlock` / `MergeInputInitSharedCubeBlock`
>
> 源码文件：
> - [`MergeVectorIfBlockPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/MergeVectorIfBlockPass.cpp)
> - [`MergeComputeBlockPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/MergeComputeBlockPass.cpp)
> - [`MergeCubeBlock.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/MergeCubeBlock.cpp)
> - [`MergeInputInitSharedCubeBlockPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/MergeInputInitSharedCubeBlockPass.cpp)
>
> 流水线位置：`ComputeBlockOptPass`（01 篇）中阶段 D（L54-L55 的 `MergeVectorIfBlock`、L79-L84 的其余三个）

---

## 一、这个专题在做什么

阶段 B（02 篇）收紧 UB 依赖、阶段 C（03 篇）处理数据流两侧的边界 op 后，剩余的块已经"语义干净、依赖明确"。阶段 D 做**最激进的块合并**：把尽可能多的块合并成少量大块，减少块间通信次数，为流水重叠（软件流水线）创造条件。

四个子 Pass 各管一类合并场景：

| 子 Pass | 合并对象 | 核心条件 |
|---|---|---|
| `MergeVectorIfBlock` | **纯 Vector 的 `scf.if`** 与上游数据源块（+ 一个下游块） | if 内全是 VECTOR op、无嵌套控制流 |
| `MergeComputeBlock` | **CUBE 之间/周围的相邻 Vector 块**（predV→succV） | 白名单 kernel + 缓冲门控 + 无环；有环时**跨 Cube 克隆破环** |
| `MergeCubeBlock` | **CUBE 块之间** | 共享输入/输出节点 + 同深度 + 无环 + 非黑名单 kernel |
| `MergeInputInitSharedCubeBlock` | **输入/累加共享的 matmul 块** | 前一个 matmul 的结果同时是下一个的 input 和 init（L0C 复用） |

**共同的本质**：全部是"把多个 block_id 的 op 统一成一个 block_id"（`bm.updateBlockId`），合并前必须通过 `willCreateCycle` 环检测。区别在于**找合并候选的方式**：前三个用图/枚举，第四个用模式匹配。

**注意顺序**：`MergeVectorIfBlock` 在**阶段 B 开头**（L54），其余三个在阶段 D（L79-L84）。`MergeVectorIfBlock` 属于"if 块归位"（提前把纯 Vector 的 if 并入数据源块，为后续 UB 优化铺路）；阶段 D 的三兄弟是**终局合并**。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `kMergeComputeBlockApplied` | **pass 间标记**：`MergeComputeBlock`/`MergeCubeBlock` 写入"运行过 + 是否真的合并"，后续 `ReorderOpsByBlockIdPass` 读取决定是否跳过重排 |
| `kEnableMergeComputeBlockKernels` 白名单 | `MergeComputeBlock` 只对 flex_attention 系列、sdpa_bwd 系列生效（`MergeComputeBlockPass.cpp` L56-L59） |
| `kDisableMergeCubeKernel` 黑名单 | `MergeCubeBlock` 对 attention bwd / pcb06 等 kernel 禁用（`MergeCubeBlock.cpp` L48-L56） |
| `BufferCountManager` | 统计模块内 IntraCore/InterCore 缓冲数量，`MergeComputeBlock` 用它做门控（IntraCore≥3、InterCore≥2） |
| `BlockDependencyGraph` | `MergeCubeBlock` 专用的块级依赖图（含 `depth`、`isCube`、pred/succ） |
| `to_tensor` | 内存 → 张量的桥。跨 Cube 克隆专门处理"Cube 依赖 CubePre 的 to_tensor"场景 |
| `L0C`（累加器） | CUBE 片上累加缓冲。matmul 结果在 L0C 中，若直接作为下一个 matmul 的 input+init，可避免一次搬出 |
| `walkMainLoop` | `SplitIfByBlockId` 目录的共享工具：找到主循环（含 matmul 的最内层循环） |
| 同步 op 不可并入 | 所有合并 pass 遇到 sync op 一律跳过——同步必须保留独立 block_id（fence 职责） |

---

## 三、逐行讲解

### 3.1 MergeVectorIfBlockPass（MergeVectorIfBlockPass.cpp）

**做什么**：把**纯 Vector 的 `scf.if`** 并入它的**上游数据源块**，并尽量连带一个下游消费者块。`scf.if` 本身没有 core_type 标记，若独立成块会夹在数据源和消费者之间造成额外边界；如果它内部全是 Vector 计算，并入数据源块最自然。

**候选过滤 `isPureVectorIf`（L54-L73）**：

```cpp
static bool isPureVectorIf(scf::IfOp ifOp) {
  for (Region &region : ifOp->getRegions()) {
    for (Block &block : region) {
      for (Operation &op : block) {
        if (op.hasTrait<OpTrait::IsTerminator>()) continue;
        if (op.getNumRegions() > 0) return false;        // 嵌套控制流不可并入
        if (CVPipeline::getOpCoreType(&op) != CoreType::VECTOR_ONLY) return false;
      }
    }
  }
  return true;
}
```

**约束**：if 内所有非 terminator op 必须是 VECTOR 且**无嵌套 region**（嵌套 if/for 无 Vector 标记，无法合并）。

**找上游数据源 `getUpstreamBlockId`（L110-L154）**：

```cpp
Block *parent = ifOp->getBlock();
llvm::SmallDenseSet<int> ids;
auto addSource = [&](Value v) {
  Operation *def = v.getDefiningOp();
  if (!def) return;                    // block 参数不是数据源
  if (def->hasTrait<OpTrait::ConstantLike>()) return;  // 常量忽略
  if (ifOp->isAncestor(def)) return;   // if 内部产生的值不是外部源
  Operation *anc = CVPipeline::getAncestorInBlock(def, parent);
  if (!anc) return;                    // 外层作用域的值，不是兄弟块 op
  int bid = bm.getBlockIdByOp(anc);
  if (bid != -1) ids.insert(bid);
};
addSource(ifOp.getCondition());         // 条件本身是数据源
ifOp->walk(...);                        // if 内部所有 op 的操作数也是数据源
if (ids.size() != 1) return failure();  // 必须唯一
target = *ids.begin();
```

**要点**：if 的**外部数据源**（条件 + if 内部操作数中在 if 外产生的值）必须全部来自**同一个 block_id**，这个块就是合并目标。常量、block 参数、if 内部值、外层作用域值都被忽略——它们不参与"归属决策"。

**下游合并（`tryMergeIf` L221-L263）**：

```cpp
SmallVector<int> downstream = collectDownstreamBlockIds(ifOp, target, bm);  // 按程序顺序
for (int bid : downstream) {
  if (!isPureVectorOpGroup(bm.getOpsByBlockId(bid))) continue;  // 下游块也必须纯 Vector
  // 尝试 if + 下游块一起并入 target
  if (!CVPipeline::willCreateCycle(opsToUnify, memGraph, target, bm)) {
    applyMerge(ifOp, downstreamOps, target, bm);
    return;
  }
}
// Fallback：至少把 if 并入上游
if (!CVPipeline::willCreateCycle({ifOp}, memGraph, target, bm)) {
  applyMerge(ifOp, {}, target, bm);
}
```

- **`isPureVectorOpGroup`（L89-L100）**：下游块的所有 op 必须纯 Vector（用 `getCoreTypeOfSimpleOpOrCf` 递归判断控制流 op 体内）且不含 sync——**禁止把含 CUBE op 的块并入 Vector 块**（block_id 会跨两种 core，下游 pass 不期望）。
- **优先"if + 下游块"一起并**（形成一个大块），不行则退化为只并 if。
- **`applyMerge`（L156-L180）**：if 本身 + if 内部所有 op（跳过 terminator 和 sync）+ 选中的下游块 op，全部 `updateBlockId` 到 target。

**设计意图**：if 是"条件数据流分叉"，若夹在两个块之间，两条分支的数据都要跨块。把它并入数据源块后，条件计算、分支计算、汇合消费都在同一块，**一条块内完成**。

### 3.2 MergeComputeBlockPass（MergeComputeBlockPass.cpp）

**做什么**：合并 **CUBE 块之间/周围的相邻 Vector 块**（predV → succV 两个相邻 Vector 块合为一个），减少 Vector 侧块数。本 pass 是**阶段 D 最激进**的合并，带三重门控 + 跨 Cube 克隆破环。

**入口 `runOnOperation`（L492-L557）**：

```cpp
module->setAttr(kMergeComputeBlockApplied, false);      // L501 默认"没合并"

// 门控 1：白名单 kernel
for (auto funcOp : module.getOps<func::FuncOp>())
  if (llvm::is_contained(kEnableMergeComputeBlockKernels, funcOp.getSymName()))
    shouldRun = true;
if (!shouldRun) return;                                  // L514-L516

// 门控 2：缓冲数量
BufferCountManager bufMgr(module);
if (intraBufCount < 3 || interCoreBufCount < 2) return;  // L518-L527

// 只在主循环内合并
SplitIf::walkMainLoop(module, [&](Operation *loop) {
  blocksToProcess.push_back(&loop->getRegion(0).front()); ...
});
bool mergedAny = false;
for (Block *block : blocksToProcess)
  mergedAny = tryMergeInBlock(block, bm, memGraph) || mergedAny;
if (mergedAny)
  module->setAttr(kMergeComputeBlockApplied, true);      // L549-L554 真实合并 → 标记
```

**三重门控的意义**：
1. **白名单**：块合并会克隆 op、改写依赖，有风险且非每个 kernel 都受益——只对 attention 类（块间通信模式固定）开启；
2. **缓冲门控**：合并减少块数、减少通信，但流水重叠需要缓冲做多缓冲；IntraCore<3 或 InterCore<2 时缓冲不够，合并无收益；
3. **主循环限定**：只处理含 matmul 的主循环体（流水重叠只对计算密集型主循环有意义）。

**核心循环 `tryMergeInBlock`（L423-L475）**——每轮重建图、找一对合并，直到无对可并：

```cpp
while (true) {
  // Step 1: 分组建块级 DAG（groupAndBuildGraph L87-L152）
  //   computeBlocks: block_id → ComputeBlock{id, coreType, ops}
  //   succs/preds:  块间依赖邻接表
  //   blockEdges:   srcId → dstId → 触发跨块依赖的源 op 列表
  groupAndBuildGraph(block, memGraph, computeBlocks, succs, preds, blockEdges);
  if (computeBlocks.empty()) return mergedAny;

  // Step 2: 收集 VECTOR 候选（collectVectorCandidates L252-L282）
  //   VECTOR_ONLY + 有 tensor 结果 + 有 CUBE 邻居（入边或出边）
  SmallVector<int> vecCandidates = collectVectorCandidates(...);
  if (vecCandidates.size() < 2) return mergedAny;

  // Step 3: 找相邻 Vector 对（findAdjacentVectorPair L286-L300）
  //   候选中有直接 pred→succ 依赖关系的 pair
  auto pairOpt = findAdjacentVectorPair(vecCandidates, succs);
  if (!pairOpt) return mergedAny;
  int predVId = pairOpt->first; int succVId = pairOpt->second;

  // Step 4: 先试直接合并（tryDirectMerge L317-L330）
  if (tryDirectMerge(opsToMerge, memGraph, predVId, succVId, computeBlocks, bm)) {
    mergedAny = true; continue;
  }
  // Step 5: 直接合并失败（有环）→ 跨 Cube 克隆破环
  if (!tryCrossCubeCloneMerge(block, computeBlocks, preds, blockEdges,
                              predVId, succVId, opsToMerge, memGraph, bm))
    return mergedAny;
  mergedAny = true;
}
```

**`groupAndBuildGraph`（L87-L152）**——本 pass 的地基：
- walk block 内所有 op，按 block_id 分组建 `ComputeBlock`（跳过 terminator、无 block_id 的 op）；(L95-L109)
- 用 `DependencyHelper` 对每个 op 遍历源，`getAncestorInBlock` 归一化到块级，跨块依赖才建边；(L117-L151)
- **`blockEdges` 记录"触发跨块依赖的源 op"**（L139-L141，`seenDeps` 去重）——这是跨 Cube 克隆要用的**精确依赖点**。

**`collectVectorCandidates`（L252-L282）**：VECTOR_ONLY + `hasTensorResult` + 有 CUBE 邻居（succs 或 preds 中有 CUBE_ONLY 块）。**为什么必须有 CUBE 邻居**：本 pass 的目的是"合并 CUBE 周围的 Vector 块"——只有贴着 CUBE 的 Vector 块合并才有意义（远离 CUBE 的纯 Vector 链由 MergeSmallBlock 管）。

**直接合并 `tryDirectMerge`（L317-L330）**：

```cpp
if (willCreateCycle(opsToMerge, memGraph, predVId, bm)) return false;
markSubBlock(computeBlocks, predVId, succVId);     // L326 打 kSubBlockId 标记
for (Operation *op : opsToMerge)
  bm.updateBlockId(op, predVId);                   // succV 并入 predV
return true;
```

**跨 Cube 克隆破环 `tryCrossCubeCloneMerge`（L350-L420）**——本 pass 最精巧的机制：

```
问题结构：
  CubePre → to_tensor → Cube
  CubePre ... → predV → succV → Cube
  合并 succV→predV 时，predV 的输入链经过 Cube，而 Cube 又依赖 CubePre 的 to_tensor
  → 合并形成跨块依赖环
解法：把 CubePre 中 Cube 依赖的 to_tensor 链克隆一份进 Cube
  → Cube 不再依赖 CubePre 的共享 op → 环被切断
```

```cpp
// 5a. 找 succV 的 CUBE 前驱（Cube）                    L358-L364
std::optional<int> cubeId = findCubePred(succVId, computeBlocks, preds);
// 5b. 找 Cube 的 CUBE 前驱（CubePre）                  L367-L373
std::optional<int> cubePreId = findCubePred(*cubeId, computeBlocks, preds);
// 5c. 确认 Cube 依赖 CubePre 的 to_tensor（findToTensorDeps L192-L208）  L376-L390
//     —— 在 blockEdges[CubePre][Cube] 的源 op 中找 to_tensor
// 5d. 回溯 to_tensor 的全部传递依赖（collectAllDeps L173-L187）            L393-L396
SmallPtrSet<Operation *, 16> allPredecessors;
for (Operation *toTensor : toTensorOps)
  collectAllDeps(toTensor, memGraph, allPredecessors);
// 5e. 过滤出 CubePre 内的 op 作为克隆集（collectOpsToBreakCycle L334-L346） L399-L400
//     cloneOpCrossCubeDep 克隆到 Cube 前端（L212-L248）                   L404
//     —— 同时把 Cube 原 op 中引用旧值的操作数重定向到克隆值
// 5f. 克隆后再查环（L407-L412）
if (willCreateCycle(opsToMerge, memGraph, predVId, bm)) return false;
// 合并
markSubBlock(computeBlocks, predVId, succVId);
for (Operation *op : opsToMerge) bm.updateBlockId(op, predVId);
return true;
```

**为什么只克隆 CubePre 里的 op**：`collectOpsToBreakCycle`（L334-L346）从 allPredecessors 中过滤出属于 CubePre 块的 op——**其他块的依赖op不用克隆**（它们不参与这个特定环）。`cloneOpCrossCubeDep`（L212-L248）把选中的 op 克隆到 `cubeBlock.ops.front()` 之前（保持块内顺序），walk 递归 `updateBlockId` 到 Cube，再用 `IRMapping` 重定向 Cube 原 op 的操作数。

**`findToTensorDeps`（L192-L208）**：只认 `bufferization::ToTensorOp`——因为 `Cube` 依赖 `CubePre` 的**内存 → 张量**桥（CubePre 写内存、Cube 读回张量）。这正是跨块依赖环的源头。

### 3.3 MergeCubeBlockPass（MergeCubeBlock.cpp）

**做什么**：合并 **CUBE 块之间**——两个 CUBE 块如果**共享输入或输出节点**且**同深度**（软件流水线的同一层），合并成一个 CUBE 块。用 `BlockDependencyGraph` 做块级依赖分析。

**入口 `runOnOperation`（L58-L108）**：

```cpp
// 黑名单 kernel 检查（L65-L72）：命中 → kMergeComputeBlockApplied=false + 直接返回
for (auto funcOp : moduleOp.getOps<func::FuncOp>())
  if (kDisableMergeCubeKernel.contains(funcOp.getSymName())) {
    moduleOp->setAttr(kMergeComputeBlockApplied, false);
    return;
  }

// walkMainLoop 找主循环体（L82-L89），失败 → fallback
if (failed(CVPipeline::SplitIf::walkMainLoop(moduleOp, ...))) {
  setFallbackAttr(moduleOp, ERRCODE_FAILED); return;
}

bool mergedAny = false;
for (Block *block : mainLoopBlocks)
  if (failed(processBlock(block, memGraph, bm, mergedAny))) {  // 失败 → fallback
    setFallbackAttr(moduleOp, ERRCODE_FAILED); return;
  }
if (!mergedAny) moduleOp->setAttr(kMergeComputeBlockApplied, false);  // L102-L105
```

**注意**：黑名单与 `MergeComputeBlock` 的白名单**互补**——`MergeCubeBlock` 用黑名单（大多数 kernel 都允许，只有已知有问题的禁用）。合并失败 → **fallback**（与 `MergeComputeBlock` 跳过不同——CUBE 块合并是"常规路径"，失败说明结构异常）。

**`processBlock`（L110-L150）**：
- 构建 `BlockDependencyGraph graph(block, memGraph, bm)`（L120）；
- `performMerging(graph, memGraph, bm, mergedAny)`（L144）。

**`performMerging`（L152-L228）**——嵌套循环迭代合并：

```cpp
// 收集一次活着的 CUBE 块（L160-L164）；每轮合并后只需从列表删除被并掉的块
while (merged) {
  merged = false;
  // 外层 i 扮演 target，内层 j 扮演 source
  for (size_t i = 0; i < cubeBlocks.size() && !merged; ++i) {
    for (size_t j = 0; j < cubeBlocks.size(); ++j) {
      if (i == j) continue;
      BlockNode *target = cubeBlocks[i];
      BlockNode *source = cubeBlocks[j];
      if (!canMergeBlocks(target, source, graph, memGraph, bm)) continue;

      if (failed(mergeBlocks(target, source, bm))) return failure();   // 合并
      merged = true; mergeCount++;

      // 图重建：source 节点被销毁（rebuildAfterMerge 失败 → 整体 abort）
      if (failed(graph.rebuildAfterMerge(target, source))) return failure();
      cubeBlocks.erase(cubeBlocks.begin() + j);   // 删掉已合并的 source
      break;                                       // 一轮只合并一对，然后从头扫
    }
  }
}
mergedAny = mergeCount > 0;
```

**要点（L196-L217 注释很关键）**：`mergeBlocks` 已经通过 `bm` 改写了 source 的 block_id，**不是事务性的**——所以 `rebuildAfterMerge` 一旦失败，图和 bm 就处于不一致状态，必须**整个 pass 报错**（不能只跳过这对）。`erase` 之后 source 指针悬空，但只用于值比较/删除，不解引用。

**`canMergeBlocks`（L230-L264）**——三个条件全过才可并：

```cpp
// 条件 1：双方都是 CUBE 块（L236-L238）
if (!target || !source || !target->isCube || !source->isCube) return false;
// 条件 2：有公共输入或输出节点（L241-L245）
if (!hasCommonInputOrOutput(target, source, graph)) return false;
// 条件 3：合并无环（L248-L252）
if (!checkNoCycle(target, source, graph, memGraph, bm)) return false;
// 条件 4：同深度（L255-L259）
if (hasSameDepth(target, source, graph)) return true;
return false;
```

**`hasCommonInputOrOutput`（L327-L352）**：把双方的 preds/succs **过滤到"异类型块"**（`getBlocksOfDifferentType` L300-L313——只留与当前 CUBE 块类型不同的块，即 Vector 邻居），再求交。**共享同一个 Vector 上游或下游** = 这两个 CUBE 块服务于同一条 Vector 数据流 → 值得合并。

**`hasSameDepth`（L266-L298）**：
- 两个 CUBE 块本身 `depth` 必须相同；(L273-L275)
- 各自**异类型后继的最大 depth** 必须相同（`getMaxSuccDepth`，L278-L287）——合并后它们在流水线里的**输出层位**要对齐，否则合并会打乱流水层级；(L297)
- 任意一方没有异类型后继 → 仍可合并（L293-L295）。

**设计意图**：`depth` 是软件流水线的层号。合并两个 CUBE 块相当于让它们在**同一层同时计算**——共享 Vector 数据源（input）或汇合到同一 Vector 消费者（output）说明它们处理同一批数据的不同切片，合并后可共享 L0C/输入缓冲。同深度保证流水结构不被打乱。

### 3.4 MergeInputInitSharedCubeBlockPass（MergeInputInitSharedCubeBlockPass.cpp）

**做什么**：当 **前一个 matmul 的结果同时作为下一个 matmul 的 input 和 init** 时，合并两个 CUBE 块。这是 **L0C 复用**模式：matmul 结果留在 L0C 累加器里，下一个 matmul 直接把它当累加初始值 + 一个输入，省掉一次 L0C→UB 搬出。

**模式匹配 `getSharedInputInitProducer`（L46-L64）**：

```cpp
static linalg::MatmulOp getSharedInputInitProducer(linalg::MatmulOp consumer) {
  auto inits = consumer.getDpsInits();
  if (inits.empty()) return {};
  // init 必须直接来自一个 matmul
  auto initProducer = dyn_cast_if_present<linalg::MatmulOp>(inits.front().getDefiningOp());
  if (!initProducer) return {};
  // 某个 input 穿透中间 op 后也来自同一个 matmul
  for (Value input : consumer.getDpsInputs()) {
    auto inputProducer = dyn_cast_if_present<linalg::MatmulOp>(
        CVPipeline::getSourceThroughCIntermediateOps(input));  // 穿透 cast/trans 等 C 中间 op
    if (inputProducer == initProducer) return initProducer;
  }
  return {};
}
```

**要点**：
- **init 必须直接**来自 matmul（`inits.front().getDefiningOp()`）；(L51-L55)
- **input 允许穿透**：用 `getSourceThroughCIntermediateOps` 沿 C 侧中间 op（cast/transpose 等）回溯——input 可能经过精度转换才喂给下一个 matmul，**穿透后仍属同一 producer 则匹配**。
- 只检查 `DpsInputs`（matmul 的 A/B 输入），不检查 init 本身。

**合并 `tryMergeBlocks`（L68-L84）**：

```cpp
llvm::SmallVector<Operation *> consumerOps = bm.getOpsByBlockId(consumerBlockId);
if (consumerOps.empty() ||
    CVPipeline::willCreateCycle(consumerOps, memGraph, producerBlockId, bm)) {
  return failure();                       // 有环 → 失败（上层 fallback）
}
for (Operation *op : consumerOps)
  bm.updateBlockId(op, producerBlockId);  // 消费者块并入生产者块
return success();
```

**入口 `runOnOperation`（L104-L141）**：
- 先收集所有 matmul（合并会原地改 block_id，先收集防迭代器失效）；(L116-L117)
- 对每个 consumer 找共享 producer；两个块都存在且不同 → 尝试合并；(L119-L130)
- **`tryMergeBlocks` 失败 → `setFallbackAttr`（L137）**：这是唯一一个"合并失败直接 fallback"的 pass。原因：matmul 结果同时被 input 和 init 使用是**结构前提**，如果合并不成（有环）说明依赖结构异常，继续优化会产出错误分块。

**设计意图**：L0C 是 CUBE 的片上累加器。`matmul1 结果 → matmul2 的 init` 意味着 matmul2 可以直接在 L0C 里累加；同时结果又作为 input 意味着数据还在 L0C/附近可复用。合并两个块让 lowering 阶段能识别这个 L0C 复用链，**避免一次核内搬移**。

---

## 四、算法流程总结

```
04_BlockMerge（4 个子 Pass）
├─ 3.1 MergeVectorIfBlock     纯 Vector scf.if → 并入上游数据源块（+ 一个下游块）
│    纯Vector检查(无嵌套) → 上游唯一数据源(条件+内部操作数) → 下游纯Vector组
│    → 优先 if+下游一起并(无环) / 兜底只并if → applyMerge(跳过terminator/sync)
├─ 3.2 MergeComputeBlock      相邻 Vector 块(predV→succV) → 合并（白名单+缓冲门控）
│    groupAndBuildGraph(块级DAG+blockEdges) → VECTOR候选(贴CUBE+tensor结果)
│    → 找相邻对 → 直接合并(无环) / 跨Cube克隆破环(5a-5f) → 循环到无对
├─ 3.3 MergeCubeBlock         CUBE 块之间 → 合并（黑名单门控）
│    BlockDependencyGraph → 收集CUBE块 → 嵌套循环(target/source)
│    → canMerge(共享输入输出+无环+同深度) → merge → rebuildAfterMerge
│    → 重建失败则整体abort(非事务性) → 每轮一对，循环到无可并
└─ 3.4 MergeInputInitSharedCubeBlock  matmul 结果作下一个的 input+init → 合并
      init直接来自matmul + input穿透getSourceThroughCIntermediateOps同源
      → tryMergeBlocks(无环) / 有环 → fallback
```

**整体策略**：阶段 D 的四个 pass 覆盖四种合并场景——**控制流**（if 并入数据源）、**Vector 侧**（相邻 Vector 对）、**CUBE 侧**（共享邻居的 CUBE 对）、**跨 matmul**（L0C 复用链）。合并方向总是"上游 ← 下游"（消费者并入生产者）。正确性守门 = `willCreateCycle`；激进程度门控 = 白/黑名单 + 缓冲计数。合并后统一打 `kSubBlockId` 标记 + `kMergeComputeBlockApplied` 通知 Reorder。

---

## 五、面试要点

1. **MergeVectorIfBlock 的"纯 Vector + 无嵌套控制流"约束为什么必要？**
   if 内的 op 会被整体并入上游块的 block_id。若内部有 CUBE op，会造出**跨 core 类型的 block_id**（下游 DataDependencyAnalysis 的 collectBlockInfo 不期望）；若有嵌套 if/for，这些控制流 op 没有 Vector 标记，无法归类。**块合并的粒度 = 可归类的最小单元**。

2. **MergeVectorIfBlock 为什么先找"上游数据源块"，再考虑下游？**
   if 的条件和数据都来自上游，if 的归属天然绑定上游。上游唯一（`ids.size() != 1` 则拒绝）保证归属无歧义；下游是"加分项"——把 if + 下游块一起并入上游形成更大块，但下游块必须是纯 Vector 且无环，否则退回只并 if。**主从关系：上游是主，下游是次**。

3. **MergeComputeBlock 为什么加"白名单 + 缓冲门控"？它和 MergeCubeBlock 的门控有何不同？**
   `MergeComputeBlock` 用**白名单**（只对 attention 类 kernel）+ 缓冲门控（IntraCore≥3、InterCore≥2）——激进但高风险，只开启已知受益场景。`MergeCubeBlock` 用**黑名单**（大多数 kernel 允许，只有已知失败的禁用）。差异原因：Vector 块合并涉及跨 Cube 克隆（改写依赖），风险高；CUBE 块合并相对保守（只合并共享邻居的块），风险低。**风险越高，门控越严**。

4. **"跨 Cube 克隆破环"解决了什么结构问题？**
   Vector 块合并（succV 并入 predV）时，若 predV 的输入链经过 Cube、而 Cube 又依赖 CubePre 的 to_tensor（内存→张量桥），合并会把两个依赖方向捏在一起形成环。解法：把 CubePre 中 Cube 依赖的 **to_tensor 链**克隆进 Cube，Cube 改用克隆值，不再依赖 CubePre 的共享 op——**以复制换取依赖简化**。`findToTensorDeps` 只认 to_tensor，因为内存→张量桥正是跨块依赖环的源头。

5. **MergeCubeBlock 的 `rebuildAfterMerge` 失败为什么要整体 abort？**
   `mergeBlocks` 已通过 `ComputeBlockIdManager` 改写了 source 块所有 op 的 block_id，**该操作非事务性**。图重建失败后，图的边信息与 bm 的归属不一致，继续合并会基于错误的图做决策。与"跳过这一对再试"不同，这里**图状态已损坏**，只能整体失败（fallback）。这体现了"先副作用、后验证"的操作必须**要么全成、要么全败**。

6. **MergeCubeBlock 的"同深度"条件在保护什么？**
   `depth` 是软件流水线的层号。两个 CUBE 块合并 = 在同一层同时计算。共享 Vector 邻居说明它们处理同一批数据，但只有**深度相同 + 异类型后继的最大深度相同**才保证合并后流水层级对齐——否则合并会打乱"第 N 层算完 → 第 N+1 层消费"的结构。**合并不能破坏流水线的分层不变量**。

7. **MergeInputInitSharedCubeBlock 的 input 为什么要"穿透 C 中间 op"？**
   matmul 结果作为下一个 matmul 的 input 时，中间可能插了 cast/transpose 等 C 侧中间 op（精度转换、布局调整）。`getSourceThroughCIntermediateOps` 穿透这些 op 后才能确认"input 和 init 真的来自同一个 matmul"。**模式识别要穿透语义透明 op，才能看到真实的数据流**。

8. **为什么 MergeInputInitSharedCubeBlock 合并失败会 fallback，而其他合并失败只跳过？**
   该模式的成立依赖**结构前提**（matmul 结果同时作 input 和 init）。合并不成说明依赖结构异常（有环），继续优化会产出错误分块，**必须 fallback 回标准编译**。而 MergeComputeBlock/MergeCubeBlock 的候选是"启发式"的，跳过不影响正确性。**前提型 vs 启发式：前者失败即异常，后者失败即放弃**。

9. **`kMergeComputeBlockApplied` 标记在这三个 pass 间如何流转？**
   `MergeComputeBlock`/`MergeCubeBlock` 写入 `false`（默认）/`true`（真的合并）；后续 `ReorderOpsByBlockIdPass` 读取——true 则正常重排，false（运行过但没合并）则跳过重排（省一次全图遍历），随后 removeAttr 消费掉。这是**pass 间消息传递**的典型模式：用 Module 属性通信，避免重复计算。

10. **这四个 pass 与阶段 B/C 的合并有什么本质区别？**
    阶段 B/C 的合并是"局部修复"——小块并邻居、if 归位、op 链并入主块，动作小、影响局部。阶段 D 是"全局整形"——合并的是**整个计算块**（可能几十个 op），涉及克隆 op、改写依赖、图重建，动作大、影响全局。所以阶段 D 需要更严的门控（白/黑名单、缓冲计数、主循环限定）和更强的正确性保证（重建失败整体 abort、结构异常 fallback）。**越激进的变换，越需要保守的开启条件**。
