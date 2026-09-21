# DynamicCVPipeline 子 Pass 逐行讲解：ComputeBlockOptPass

> 源码文件：[`third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOptPass.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOptPass.cpp)
>
> 流水线位置：`AddDynamicCVPipelinePass::addPasses` 中 **第 4 个 pass**（`PlanComputeBlock` 之后、`AnalyzeDataFlow` 之前）
>
> 子 Pass 专题（同目录）：
> - `02_UBAndLoadOpt`：UB/加载侧优化（UnifyAllocBlock / UBUsageOpt / BroadcastUBOpt / PosMaskPattern / MergeSameSourceAxis / MergeSmallBlock / RelocateMemrefDecl）
> - `03_SinkAndPatternOpt`：下沉与模式优化（SinkI1ProducersIntoUsers / FixpipeOpt / MoveLoadIntoUser / UnifyStoreBlock / ExpSubfPattern）
> - `04_BlockMerge`：块合并（MergeVectorIfBlock / MergeComputeBlock / MergeCubeBlock / MergeInputInitSharedCubeBlock）
> - `05_SplitIfByBlockId`：scf.if 按 block id 分裂专题

---

## 一、这个 Pass 在做什么

`PlanComputeBlock` 只完成了"分块 + 排序"的**第一次分组**。这份分组是保守的（为了保证正确性，块偏碎、块间依赖偏多）。`ComputeBlockOpt` 的任务是**在保持语义不变的前提下优化分块质量**：合并碎片块、把 op 移动到更合适的块、消解块间冗余依赖，最终得到"块少、依赖干净、适合流水重叠"的分块结果。

主 Pass 是一个**大型编排壳**，依次调度 **19 个子 Pass**，每个子 Pass 之间穿插 `ReorderOpsByBlockIdPass`（重排让块的物理顺序跟上新的分组）。

```cpp
void ComputeBlockOptPass::runOnOperation() {              // L37-L92
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) { return; }

  OpPassManager pm(module.getOperationName());

  // 阶段 A：统一 load/alloc 归属
  pm.addPass(createUnifyAllocBlockPass());                // L51
  pm.addPass(createReorderOpsByBlockIdPass());            // L52

  // 阶段 B：if 合并 + UB 优化
  pm.addPass(createMergeVectorIfBlockPass());             // L54
  pm.addPass(createReorderOpsByBlockIdPass());            // L55
  pm.addPass(createReorderOpsByBlockIdPass());            // L56
  pm.addPass(createUBUsageOptPass());                     // L58
  pm.addPass(createBroadcastUBOptPass());                 // L59
  pm.addPass(createPosMaskPatternPass());                 // L60
  pm.addPass(createMergeSameSourceAxisPass());            // L61
  pm.addPass(createReorderOpsByBlockIdPass());            // L62
  pm.addPass(createMergeSmallBlockPass());                // L63
  pm.addPass(createReorderOpsByBlockIdPass());            // L64

  // 阶段 C：下沉与模式优化
  pm.addPass(createSinkI1ProducersIntoUsersPass());       // L66
  pm.addPass(createReorderOpsByBlockIdPass());            // L67
  pm.addPass(createFixpipeOptPass());                     // L69
  pm.addPass(createSplitIfByBlockIdPass());               // L70
  pm.addPass(createReorderOpsByBlockIdPass());            // L71
  pm.addPass(createMoveLoadIntoUserPass());               // L72
  pm.addPass(createUnifyStoreBlockPass());                // L73
  pm.addPass(createExpSubfPatternPass());                 // L74
  pm.addPass(createReorderOpsByBlockIdPass());            // L75
  pm.addPass(createMergeSmallBlockPass());                // L76
  pm.addPass(createReorderOpsByBlockIdPass());            // L77

  // 阶段 D：大块合并
  pm.addPass(createMergeComputeBlockPass());              // L79
  pm.addPass(createReorderOpsByBlockIdPass());            // L80
  pm.addPass(createMergeCubeBlockPass());                 // L82
  pm.addPass(createMergeInputInitSharedCubeBlockPass());  // L83
  pm.addPass(createReorderOpsByBlockIdPass());            // L84
  pm.addPass(createRelocateMemrefDeclPass());             // L85

  if (failed(runPipeline(pm, module))) {
    if (!CVPipeline::hasFallbackAttr(module)) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
    return;
  }
}
```

四个阶段的意图：

| 阶段 | 子 Pass | 意图 |
|---|---|---|
| A（统一归属） | `UnifyAllocBlock` | 把 `memref.alloc` / `scf.if(fill)` / `memref.subview` 这类"内存声明/初始化"op 统一进消费它们的计算块——声明必须随使用方走 |
| B（UB 优化） | `MergeVectorIfBlock` / `UBUsageOpt` / `BroadcastUBOpt` / `PosMaskPattern` / `MergeSameSourceAxis` / `MergeSmallBlock` | 围绕 **UB（Unified Buffer，Vector 核的片上缓冲）** 做依赖收紧：把只含 Vector 的 scf.if 并入数据源块、把 broadcast/posmask 下沉到用户块、把同源轴合并、把小块并入邻居 |
| C（下沉/模式） | `SinkI1Producers` / `FixpipeOpt` / `SplitIfByBlockId` / `MoveLoadIntoUser` / `UnifyStoreBlock` / `ExpSubfPattern` | 把 i1 生产者下沉到用户旁（i1 是小数据，就近省 UB）、fixpipe 优化 matmul-cast-store、跨 block 的 if 按 id 分裂、load 下沉到用户块、store 并入生产者块、exp-subf 模式统一 |
| D（大块合并） | `MergeComputeBlock` / `MergeCubeBlock` / `MergeInputInitSharedCubeBlock` / `RelocateMemrefDecl` | **最激进的一步**：合并相邻 Vector 块（含跨 Cube 克隆破环）、合并 CUBE 块、合并"输入/累加共享"的 matmul 块、跨同步的 memref 声明搬到消费者块 |

**为什么 Reorder 反复穿插**：每个子 Pass 都可能改写 block id 归属或改变依赖关系，只有重排后才能让"新的分组"反映到物理顺序，且下一个子 Pass 看到的顺序是"块连续"的。L55-L56 连续两次 Reorder 是刻意的——一次让块聚拢，第二次处理第一次重排引入的边界情况。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| UB（Unified Buffer） | Vector 核的片上统一缓冲，所有 Vector 计算的数据必经 UB。UB 优化 = 让"数据生产/消费"尽量靠近同一块，减少 UB 占用窗口 |
| `ssbuffer.block_id` | 计算块编号。**本 pass 所有子 pass 的本质操作都是"改写 block id"**——把 op 从一个块挪到另一个块 |
| `kMergeComputeBlockApplied` | `MergeComputeBlockPass` 写给 Module 的标记：记录"运行过 + 是否真的合并"。后续 `ReorderOpsByBlockIdPass` 读取它决定是否跳过重排 |
| `kSubBlockId`（`setSubBlockId`） | 合并时给"被并入块"打的原始 id 标记（记录块合并历史，供后续 degrade/传输 pass 使用） |
| `DependencyCycleDetector` | 复用 PlanComputeBlock 的 DFS 环检测：**任何块合并都必须先验证不产生环** |
| `BufferCountManager` | 统计模块内 InterCore/IntraCore 缓冲数量，`MergeComputeBlock` 用它做门控（缓冲太少则不值得合并） |
| `SplitIf::walkMainLoop` | SplitIfByBlockId 目录提供的工具：找到主循环（含 matmul 的最内层循环） |
| `to_tensor` | 内存 → 张量的桥。`MergeComputeBlock` 的跨 Cube 克隆专门处理"Cube 依赖 CubePre 的 to_tensor"场景 |
| FIXPIPE | CUBE 的固定流水线（matmul 结果 → cast → store 走专用硬件通道）。`FixpipeOptPass` 把这条链标成 CUBE 以启用它 |

---

## 三、逐行讲解

### 3.1 主 Pass（L37-L92）

- **L40-L42** fallback 兜底，与其余主 Pass 一致。
- **L51-L52（阶段 A）**：`UnifyAllocBlockPass` 先把"内存声明/初始化"统一归属，随后立刻 Reorder——因为 alloc 归属变了，物理顺序必须跟上（alloc 要出现在第一个使用它的块之前）。
- **L54-L56（阶段 B 开头）**：`MergeVectorIfBlockPass` 合并纯 Vector 的 scf.if 块后，**连续跑两次 Reorder**。原因：if 块合并涉及"上游数据源块、if 本身、下游消费者块"三方的块 id 变化，一次 Reorder 可能无法把块内顺序收敛到最终态（涉及嵌套 region 的块需要多轮 moveBefore）。
- **L58-L64（阶段 B 核心）**：`UBUsageOptPass` 是主 pass 注释里点名的核心——"find the smallest UB dependency location and divide the computation blocks"（找最小 UB 依赖位置来切分计算块）。它解决 UB 容量/生命周期问题：两个块若共用 UB 且生命周期重叠，就必须分开。其后 `BroadcastUBOpt`、`PosMaskPattern` 是特定 pattern 的"下沉到用户块"优化（broadcast 和 pos-mask 都产生小数据，就近消费省 UB）；`MergeSameSourceAxis` 合并同源轴的两个 Vector 消费者块；`MergeSmallBlock`（≤3 op 的小块并入邻居）。
- **L66-L77（阶段 C）**：
  - `SinkI1ProducersIntoUsersPass`：i1（布尔）产生者复制到每个使用处。i1 数据极小，复制代价低，但能让每个用户块就近拿到布尔值、避免跨块传 i1。
  - `FixpipeOptPass`：`matmul → cast → store` 模式（fixpipe 通道），把链上 op 的 core_type 改为 CUBE——让这条链走 CUBE 的 FIXPIPE 专用硬件。
  - `SplitIfByBlockIdPass`：跨多个 block id 的 `scf.if` 按 id 分裂成多个 if，每个 if 只包含一个块的计算（详见 05 篇）。
  - `MoveLoadIntoUserPass`：load 移动到唯一用户块。
  - `UnifyStoreBlockPass`：store 并入其生产者 Vector 块。
  - `ExpSubfPatternPass`：`exp` + 可选 `extf` 的 subf 模式，统一 block id。
- **L79-L85（阶段 D）**：最激进的合并阶段。
  - `MergeComputeBlockPass`：合并 CUBE 块之间/周围的相邻 Vector 块。**只对白名单 kernel 生效**（`kEnableMergeComputeBlockKernels`：flex_attention 系列、sdpa_bwd 系列），且要求 IntraCore 缓冲 ≥3、InterCore ≥2（缓冲太少合并没有意义）。通过 `kMergeComputeBlockApplied` 标记与后续 Reorder 通信。
  - `MergeCubeBlockPass`：合并 CUBE 块。
  - `MergeInputInitSharedCubeBlockPass`：一个 matmul 的结果同时作为下一个 matmul 的 input 和 init（L0C 复用场景）时合并。
  - `RelocateMemrefDeclPass`：跨同步 op 的 memref 声明搬到消费者块（声明跨同步会被重排破坏，需要就地搬家）。
- **L86-L91**：任一子 pass 失败 → `ERRCODE_FAILED` fallback。

### 3.2 典型子 Pass 内部结构（以 `MergeComputeBlockPass` 为例，L492-L557）

主 Pass 只负责排序，真正的逻辑在各子 Pass。以阶段 D 的 `MergeComputeBlockPass` 说明"一个子 Pass 的内部模式"：

```cpp
void MergeComputeBlockPass::runOnOperation() override {    // L492-L557
  if (CVPipeline::hasFallbackAttr(module)) return;

  module->setAttr(kMergeComputeBlockApplied, false);       // 默认"没合并"
  // 白名单 kernel 检查
  bool shouldRun = false;
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    if (llvm::is_contained(kEnableMergeComputeBlockKernels, funcOp.getSymName()))
      shouldRun = true;
  }
  if (!shouldRun) return;

  BufferCountManager bufMgr(module);                       // 缓冲门控
  if (intraBufCount < 3 || interCoreBufCount < 2) return;

  auto blocksToProcess = SplitIf::walkMainLoop(...);       // 只在主循环内合并
  bool mergedAny = false;
  for (Block *block : blocksToProcess)
    mergedAny = tryMergeInBlock(block, bm, memGraph) || mergedAny;
  if (mergedAny) module->setAttr(kMergeComputeBlockApplied, true);
}
```

- **门控**：白名单 kernel + 缓冲数量。`MergeComputeBlock` 是"激进但可选"的优化——只有已知能受益的 kernel（attention 类）且缓冲资源充足时才启用。
- **`tryMergeInBlock`（L423-L475）**：无限循环直到无对可合并。每轮：① 分组建块级 DAG（`groupAndBuildGraph`，含 blockEdges 记录跨块依赖源 op）；② 收集 VECTOR 候选（有 tensor 结果 + 邻接 CUBE 块）；③ 找相邻 Vector 对（predV → succV）；④ 先试直接合并（`tryDirectMerge`：无环即可）；⑤ 直接合并失败（有环）→ `tryCrossCubeCloneMerge`（L350-L420）：把 CubePre 中 Cube 依赖的 `to_tensor` 链克隆进 Cube 打破环，再合并。
- **跨 Cube 克隆破环**是本 pass 最精巧的机制：Vector 块合并会形成环（predV 的输入来自 Cube，succV 的输出喂给 Cube，Cube 又依赖 CubePre 的 to_tensor 落在 predV…），此时克隆 CubePre 的部分 op 进 Cube，切断跨块依赖，环就解开了。

### 3.3 Reorder 的"标记消费"机制（联系 01 篇 05_ReorderOpsByBlockId）

`MergeComputeBlock` 与 `ReorderOpsByBlockId` 之间有显式协作：
- `MergeComputeBlock` 合并成功后把 `kMergeComputeBlockApplied` 置 true；
- 后续 `ReorderOpsByBlockIdPass`（ComputeBlockOptPass.cpp L80）读取该标记：true → 正常重排；false（运行过但没合并）→ **跳过重排**，省一次完整图遍历；
- 标记被消费（removeAttr），不泄漏到输出 IR。

这是"pass 间消息传递"的典型模式：一个 pass 的结果信息通过 Module 属性传给下一个 pass。

---

## 四、算法流程总结

```
ComputeBlockOptPass（编排壳，19 个子 pass + 反复穿插 Reorder）
├─ 阶段 A：统一内存声明/初始化归属
│   UnifyAllocBlock → Reorder
├─ 阶段 B：UB 依赖收紧
│   MergeVectorIfBlock → Reorder ×2 → UBUsageOpt → BroadcastUBOpt
│   → PosMaskPattern → MergeSameSourceAxis → Reorder → MergeSmallBlock → Reorder
├─ 阶段 C：下沉与模式优化
│   SinkI1ProducersIntoUsers → Reorder → FixpipeOpt → SplitIfByBlockId → Reorder
│   → MoveLoadIntoUser → UnifyStoreBlock → ExpSubfPattern → Reorder
│   → MergeSmallBlock → Reorder
├─ 阶段 D：大块合并（激进、门控）
│   MergeComputeBlock（白名单 + 缓冲门控 + kMergeComputeBlockApplied 标记）
│   → Reorder → MergeCubeBlock → MergeInputInitSharedCubeBlock → Reorder
│   → RelocateMemrefDecl
└─ 失败 → ERRCODE_FAILED fallback
```

**整体策略**：先修归属（A）→ 再收紧 UB 依赖（B）→ 再处理特化 pattern（C）→ 最后激进合并（D）。每步都重排，保证"物理顺序 = 最新分组"。从"保守正确"渐进到"激进高效"，任何一步失败都不影响前面已完成的优化（子 pipeline 失败整体 fallback，但 IR 已在各自 pass 内保持一致性）。

---

## 五、面试要点

1. **ComputeBlockOpt 与 PlanComputeBlock 的分工是什么？**
   `PlanComputeBlock` 做**第一次分块**，目标是正确性（每个 op 有归属、块无环、不跨同步）；`ComputeBlockOpt` 做**分块优化**，目标是性能（块更少、依赖更少、UB 生命周期更短、匹配硬件通道）。Plan 是"画格子"，Opt 是"挪格子 + 合并格子"。

2. **为什么 ComputeBlockOpt 需要这么多子 Pass 且反复穿插 Reorder？**
   分块优化是"多目标"的：UB 占用、i1 就近、store 下沉、块合并、特殊硬件通道（FIXPIPE）……每个目标独立成一个子 Pass，便于单独调试和演进。每个子 Pass 改写 block id 后必须 Reorder，否则 IR 物理顺序与分组不一致，下一个子 Pass 基于错误的顺序做依赖分析。

3. **`MergeComputeBlock` 为什么加白名单和缓冲门控？**
   块合并是激进变换（会克隆 op、改写依赖），有风险且不是对每个 kernel 都有收益。白名单（attention 类，块间通信模式固定）保证只优化已知受益场景；缓冲门控（IntraCore≥3、InterCore≥2）保证合并后流水重叠有足够缓冲资源。**激进优化必须"保守地开启"**。

4. **"跨 Cube 克隆破环"解决了什么问题？**
   Vector 块合并（succV 并入 predV）时，如果 predV 的输入链经过 Cube → CubePre 的 to_tensor，合并会形成依赖环。解法：把 CubePre 中 Cube 依赖的 to_tensor 链克隆一份放进 Cube，Cube 不再依赖 CubePre 的共享 op，环被切断，合并得以进行。这是"以复制换取依赖简化"的经典 trade-off。

5. **UB 优化（阶段 B）的核心逻辑是什么？**
   `UBUsageOptPass` 的目标是"找最小 UB 依赖位置来切分计算块"——即分析两个块共享的 UB 缓冲，若它们的生命周期重叠会超 UB 容量，就保持块分离；若可以错开，就允许合并/靠近。`BroadcastUBOpt`/`PosMaskPattern` 是把产生小数据（broadcast/mask）的 op 下沉到用户块，让数据在块内就地消费，不占跨块缓冲。

6. **`FixpipeOptPass` 在做什么？为什么能改 core_type？**
   `matmul → cast → store` 是 FIXPIPE（CUBE 固定流水线）支持的形态。原分块时 cast 被标 VECTOR，链会变成"Cube 算 → Vector 转 → Cube/其他 存"，多一次核间往返。该 pass 把整条链标成 CUBE，让 matmul 结果在 CUBE 侧直接 cast + store，走专用硬件通道。这体现了 CV 流水线"core_type 是软决策，可以按硬件能力重标"。

7. **`kMergeComputeBlockApplied` 标记机制的设计价值？**
   它避免了一次冗余的完整重排：`MergeComputeBlock` 若没实际合并任何块（白名单外的 kernel、缓冲不足、无合并对），后续 Reorder 完全可以跳过——重排是最贵的子 pass（全图依赖 + 两级拓扑）。**用属性做 pass 间消息传递**，是 MLIR 多 pass 流水线协同的常见手法。

8. **为什么是"先 UB 优化、后块合并"的顺序？**
   UB 优化决定"哪些块必须分开"（UB 容量约束），块合并决定"哪些块可以合到一起"（减少块数）。必须先确认容量约束（哪些不能合），再去做合并（哪些能合）——反过来会合并出超 UB 容量的块。这个顺序体现了"先约束后优化"的调度原则。

9. **主 Pass 里为什么连续两次 Reorder（L55-L56）？**
   `MergeVectorIfBlockPass` 涉及三方块 id 改写（上游数据源、if 自身、下游消费者），且 if 是嵌套 region，一次 moveBefore 重排后块内顺序可能未收敛（例如先动外层再动内层才能稳定）。连续两次重排让重排操作收敛到不动点，保证后续 pass 看到稳定的块顺序。
