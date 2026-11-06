# DynamicCVPipeline 子 Pass 专题：if 内 CUBE 依赖链克隆（07）

> 覆盖 `ComputeBlockOptPass` 阶段 E 中 `CloneCubeDepInIfPass`（[ComputeBlockOptPass.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOptPass.cpp) L87）
>
> 源码文件：[`CloneCubeDepInIf.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/CloneCubeDepInIf/CloneCubeDepInIf.cpp)（639 行）
>
> 流水线位置：阶段 E 中 `CloneCubeDepInIf`（L87）之后、`SplitIfByBlockId`（L88）之前。**main-dev 新增**。
>
> 相关属性：`ssbuffer.block_id`（[Utils.h](../../../third_party/ascend/include/DynamicCVPipeline/Common/Utils.h) L39）、`ssbuffer.clone`（[Utils.h](../../../third_party/ascend/include/DynamicCVPipeline/Common/Utils.h) L75，保存"原 block_id"）。

---

## 一、这个 Pass 在做什么

`SplitIfByBlockId` 会把一个跨多个 block id 的 `scf.if` 按 block id 拆成多个 if。拆分之后，**原 if 内部"后置 CUBE 块依赖前置 CUBE 块"的跨块 SSA 边会变成跨 if 依赖**——而拆分后的每个 if 是独立区域，无法跨越 if 边界共享 SSA 值，会打断 CUBE 计算链。

`CloneCubeDepInIfPass` 在分裂**之前**先处理这个问题：对每个候选 if 的 then/else region，扫描出其中所有 CUBE 块（同一个 `ssbuffer.block_id` 的连续 CUBE op 序列），再按"从后往前"的顺序检查——如果某个后置 CUBE 块依赖了任意前置 CUBE 块，就把前置 CUBE 块里的所有 op **克隆一份**插到后置 CUBE 块的最前面，并把后置块对这些前置 op 的使用重定向到克隆体。这样每个 CUBE 块都"自带它所需的前置 CUBE 计算"，自洽且独立，后续 `SplitIfByBlockId` 分裂后不会再出现跨 if 的 CUBE 依赖。

克隆插入的 op 会打上两个标记：
- `ssbuffer.block_id` = 后置 CUBE 块的 block id（让克隆体归入后置块）；
- `ssbuffer.clone` = 原 block id（记录它来自哪个块，供后续清理与 debug）。

克隆完成后，该 Pass 还会做一次**死克隆清理**：把"整条 SSA/内存依赖链都已无同 block_id 活用户"的冗余克隆体删除（Rule 2/3），保证只留下真正被后置块消费的那部分。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `CubeBlock` | if region 内"连续且共享同一非 -1 block_id、core_type 为 CUBE"的 op 序列（最大连续运行段），是克隆/判定的基本单元 |
| `CandidateIf`（本 Pass 私有） | `ifOp` + `thenCubes`/`elseCubes` + `thenNeedsSplit`/`elseNeedsSplit`（注意：与 `SplitIfByBlockId` 的同名 `CandidateIf` 不同，本 Pass 只有 CUBE 块信息） |
| `thenNeedsSplit` / `elseNeedsSplit` | region 内**去重后的 CUBE block_id 数量 ≥ 2** 才需要本 Pass 介入（有至少两个 CUBE 块才可能产生跨块 CUBE 依赖） |
| `ssbuffer.clone`（`kClone`） | 标记"该 op 是克隆体"，值 = 原始 block_id；`cleanupSide` 只清理带此标记的 op |
| `laterCubeDependsOnEarlier` | 判断后置 CUBE 块的 op 是否通过 SSA use-def（含嵌套 region 传递）引用了任意前置 CUBE 块的 op |
| `retagNestedBlockIds` | 克隆出的 op 其嵌套 region 内的 `linalg.fill` 等也要改 block_id，使整棵子树统一锚定到后置块 |
| `cloneOpWithMapping` / `updateCloneMapping` | 复用 `CloneOps` 的 `cloneOpWithMapping` / `updateCloneMapping` 思路：带 value map 克隆 + 递归重写操作数 |
| Rule 2 / Rule 3 | 死克隆清理的两条规则（见 3.5）：有结果 op 看"是否还有同 block_id 活用户"，无结果 op 看"是否还有同 block_id 的 exec-after 活消费者" |

---

## 三、逐行讲解

### 3.1 数据结构（L63-L79）

```cpp
/// A maximal run of consecutive ops in MLIR order inside a single if-region
/// that share the same block_id AND whose core_type is CUBE.
struct CubeBlock {
  int blockId;
  SmallVector<Operation *> ops;
};

struct CandidateIf {
  scf::IfOp ifOp;
  SmallVector<CubeBlock> thenCubes;
  SmallVector<CubeBlock> elseCubes;
  bool thenNeedsSplit = false;
  bool elseNeedsSplit = false;
};
```

`CubeBlock` 是对"块"的局部刻画：只关心 region 内**连续同 block_id 的 CUBE op**。`CandidateIf` 只记录 CUBE 块信息——这与 `SplitIfByBlockId` 的 `CandidateIf`（记录全部 BlockGroup 与 yield 计划）形成对照：本 Pass 是更轻量的前置准备。

### 3.2 `collectCubeBlocksInRegion`（L112-L156）

从 then-block 或 else-block 扫描出 CUBE 块序列。关键过滤规则：

- **跳过 `scf.yield`**（L120）：yield 是终止 op，不属于任何计算块；
- **无 block_id 或 block_id == -1 的 op 跳过**（L124-L127）：未分块/无归属的 op 不参与；
- **嵌套 `scf.if` 特殊处理**（L129-L138）：如果这个嵌套 if 的 block_id 与"最近一个 CUBE 块"相同，就把它**挂到该 CUBE 块**（`cubes.back().ops.push_back(&op)`）——这样它内部的 `linalg.fill` 会随 `scf.if` 一起被深克隆，保留"填零初始化链 + 其 memref.alloc 宿主"的完整关系；
- **只有 `CUBE_ONLY` op 才锚定一个新 CUBE 块**（L143-L152）：`VECTOR_ONLY` / `UNDETERMINED` op 被**静默跳过**，不成为 CUBE 块边界。这是为了不误克隆——例如一个 if region 里混着 CUBE 块与 VECTOR 块，若 VECTOR 块（如 block_id=17）也参与，可能因为"block_id=9 依赖 block_id=17 的 tensor.empty 输出"就错误地把 block_id=5 克隆进 block_id=9。

**设计意图**：只把"纯 CUBE 同块序列"视作克隆目标，VECTOR/UNDETERMINED 一律当作透明的边界，避免跨 core_type 的误克隆。

### 3.3 `getCandidate`（L159-L180）

对每个 `scf.if`：
1. `thenCubes = collectCubeBlocksInRegion(*thenBlock)`；
2. 有 else 块才 `elseCubes = collectCubeBlocksInRegion(*elseBlock)`；
3. `thenNeedsSplit = 去重后的 then CUBE block_id 数 ≥ 2`（`distinctCount`）。

**判定口径是"去重后的 block_id 数"，不是 CUBE 块个数**：同 block_id 的 CUBE op 若被 VECTOR op 打断会分成多个 `CubeBlock`，但它们还是同一个块，不该触发克隆。只有真的存在 ≥2 个**不同** CUBE block_id 时，才可能产生"后置 CUBE 块依赖前置 CUBE 块"的跨块依赖，才需要本 Pass 介入。

### 3.4 依赖判定 + 克隆（L184-L361）

#### `laterCubeDependsOnEarlier`（L184-L225）

判断 `laterOps` 是否引用（SSA use-def）`earlierOps` 中的 op，**传递式**（含嵌套 region）：

- 对每个后置 op：遍历其 `getOpOperands()`，若 `defOp` 属于 `earlierOps` → true；
- 同时 walk 后置 op 的每个 region，检查嵌套 op 的操作数是否引用前置 op（`linalg.fill` 等嵌套 op 也可能跨块引用）。

#### `retagNestedBlockIds`（L233-L246）

克隆出一个 op 后，walk 它的所有 region，把内部嵌套 op 的 `ssbuffer.block_id` 统一改成目标 `blockId`。典型场景：克隆出的 `scf.if` 内部的 `linalg.fill` 原本 block_id 是前置块的，必须改成后置块，才能随整体一起锚定。`kClone` 标记只保留在**外层 op** 上，嵌套 op 只继承 block_id、保留原 core_type。

#### `cloneOpWithMapping`（L253-L266）

镜像 `CloneOps::cloneOpWithMapping`：先用已克隆的 `valueMap` 初始化一个 `IRMapping`（让克隆体引用"已克隆的前置值"而非原值），`builder.clone` 后把 `op` 结果 → `cloned` 结果的映射写回 `valueMap`，最后 `retagNestedBlockIds` 统一嵌套 block_id。**key：valueMap 跨多个 op 累积**，保证按 MLIR 顺序克隆时 USE-DEF 链在克隆体内部正确闭合。

#### `updateCloneMapping`（L270-L301）

镜像 `CloneOps::updateCloneMapping`：递归遍历 op 与其所有嵌套 region，把操作数中命中 `valueMap` 的值替换为克隆值。两条保护：
- 命中 `yieldValues` 的跳过（yield 的操作数是 if 的对外输出，不能被重定向到克隆体）；
- 类型不匹配则 `failure()`（value map 必须类型一致，否则是 IR 损坏信号）。

#### `cloneEarlierCubesInto`（L307-L361）

核心搬运函数——把 `earlierCubes` 的所有 op 按 MLIR 顺序克隆到 `insertBefore`（后置 CUBE 块首 op）之前，并重写 `laterOps` 的操作数指向克隆体：

```
1. toClone = 收集 earlierCubes 的所有 op，按 isBeforeInBlock 排序；
2. builder 插入点到 insertBefore；
3. for 每个 op：cloneOpWithMapping（带 valueMap 累积）→
     setAttr(kBlockId, laterBlockId)（归入后置块）
     setAttr(kClone, 原 block_id)（记录来源）
   → 收集进 newlyInserted；
4. for 每个 laterOp：updateCloneMapping（把引用前置 op 的操作数切到克隆体，yield 值跳过）。
```

### 3.5 `processSide` + `cleanupSide`（L368-L528）

#### `processSide`（L368-L419）——反向扫描 + 克隆

镜像 `CloneOps::cloneOpsInMainLoop` 的**从后往前**策略：

1. `cubes.size() < 2` 直接返回（单块无跨块依赖）；
2. 收集 `yieldValues`（if 的 yield 操作数，克隆重写时要跳过）；
3. **从最后一个 CUBE 块往第一个**遍历（`for (size_t i = cubes.size(); i-- > 0;)`，`i==0` 跳过）：
   - `earlierCubes = cubes[0..i)`；
   - `earlierOps = 所有前置块 op 的集合`；
   - 若 `laterCubeDependsOnEarlier(earlierOps, cubes[i].ops)` → `cloneEarlierCubesInto(...)` 把前置块 op 克隆进 `cubes[i]` 块首。

**为什么从后往前**：先处理后置块（克隆出它所需的前置计算），再处理更前置的块。这样更前置的块在处理时看到的 IR 是"后置块已经自洽"，不会重复克隆或顺序错乱；也对应 `cleanupSide` 的从后往前清理方向。

#### `cleanupSide`（L436-L528）——死克隆清理

克隆是"整块搬"（把前置块**所有** op 都克隆进后置块），但后置块往往只用到其中一部分，会产生冗余克隆。清理逻辑镜像 `CloneOps::shouldEraseOpForCube` 的 Rule 2/3：

- **只处理带 `ssbuffer.clone` 标记的 op**（L468），且要求有具体 block_id（L477，无 block_id 的保守当作 pinned）；
- **从后往前**遍历（L463），维护 `erasedOps` 集合屏蔽"已被级联删除的消费方"；
- **Rule 2（有结果 op）**（L509-L511）：`shouldErase = !hasLiveSameBlockIdUser(op)` —— 若没有任何**同 block_id** 的活用户消费它的结果，就删除。`sameBlockId` 要求双方都有具体且相同的 block_id，且用户不在 `erasedOps` 中；
- **Rule 3（无结果 op）**（L513-L515）：`shouldErase = !hasLiveSameBlockIdExecAfter(op)` —— 若没有**同 block_id** 的 exec-after 活消费者（用 `MemoryDependenceGraph::getExecAfter` 判定真实内存依赖，而非仅 MLIR 顺序），就删除。

级联收敛：因为从后往前删，且删除立即 `op->erase()` 并写入 `erasedOps`，下游消费者先被判定删除，上游生产者再发现"唯一消费者已被删"，也被删除——干净地收敛。

### 3.6 `processCandidate`（L530-L554）

单候选 if 的处理总控：

```
1. if (thenNeedsSplit) processSide(thenBlock, thenCubes);
2. if (elseNeedsSplit) processSide(elseBlock, elseCubes);
3. 克隆后重建 MemoryDependenceGraph memGraph(cand.ifOp, aa)；
4. if (thenNeedsSplit) cleanupSide(thenBlock, memGraph);
5. if (elseNeedsSplit) cleanupSide(elseBlock, memGraph);
```

**关键顺序**：先 processSide 克隆 → **再重建 memGraph** → 再 cleanupSide。因为 `cleanupSide` 的 Rule 3 依赖 `MemoryDependenceGraph::getExecAfter` 的真实内存依赖边，克隆出的新 op 必须参与进 memGraph，才有正确的 exec-after 关系（镜像 `CloneOps` 的 per-loop 重建策略）。

### 3.7 Pass 入口 `runOnOperation`（L578-L629）

```cpp
void CloneCubeDepInIfPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (hasFallbackAttr(module)) return;           // fallback 兜底

  auto &aa = getAnalysis<AliasAnalysis>();        // 内存依赖分析依赖别名分析

  auto mainRes = walkMainLoop(module, [&](Operation *op) -> LogicalResult {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (funcOp && is_contained(kSkippedKernels, funcOp.getSymName()))
      return success();                          // 跳过白名单外 kernel

    WalkResult walkRes = op->walk([&](scf::IfOp ifOp) -> WalkResult {
      CandidateIf cand = getCandidate(ifOp);
      if (!cand.thenNeedsSplit && !cand.elseNeedsSplit)
        return WalkResult::advance();            // 单 CUBE 块 → 跳过
      if (failed(processCandidate(cand, aa)))
        return WalkResult::interrupt();          // 失败 → 中断整个 walk
      return WalkResult::advance();
    });
    if (walkRes.wasInterrupted()) return failure();
    return success();
  });

  if (failed(mainRes)) {                         // 失败 fallback
    setFallbackAttr(module, ERRCODE_FAILED);
    return;
  }
}
```

**工作范围**：`walkMainLoop` 限定只处理主循环（含 matmul 的最内层循环）内的 if，与 `SplitIfByBlockId` 保持一致——只有主循环内的 if 才会被后续分裂，才需要本 Pass 预处理。每个候选 if 独立 `processCandidate`（内部独立重建 memGraph），失败则整体 `ERRCODE_FAILED` fallback。

---

## 四、流程总结

```
CloneCubeDepInIfPass
├─ fallback 兜底检查
├─ walkMainLoop（限定主循环）
│   └─ 每个 scf.if：
│       ├─ getCandidate
│       │    ├─ collectCubeBlocksInRegion(then) / (else)
│       │    │    └─ 只收集"连续同 block_id 的 CUBE op"，
│       │    │       嵌套 scf.if 挂到匹配的最近 CUBE 块，
│       │    │       VECTOR/UNDETERMINED 跳过
│       │    └─ distinct block_id 数 ≥ 2 → needsSplit
│       └─ needsSplit ? processCandidate :
│            ├─ processSide(后向扫描：后置块依赖前置块 → 整段克隆前置块
│            │              进后置块，重用 valueMap，重写操作数，yield 跳过)
│            ├─ 重建 MemoryDependenceGraph
│            └─ cleanupSide(从后往前删冗余克隆：Rule 2 有结果看活用户，
│                           Rule 3 无结果看 exec-after，erasedOps 屏蔽级联)
└─ 失败 → ERRCODE_FAILED fallback
```

**整体策略**：先"识别会分裂的 if（≥2 个 CUBE 块）"，再"后向克隆前置 CUBE 依赖链让后置块自洽"，最后"清理冗余克隆"。所有克隆体带 `ssbuffer.clone` 标记（值 = 原 block_id），归入后置块（`ssbuffer.block_id` = 后置块 id）。这是 `SplitIfByBlockId` 的前置保证：分裂后的每个 if 内部不再有跨块 CUBE SSA 依赖。

---

## 五、面试要点

1. **为什么需要在 `SplitIfByBlockId` 之前跑 `CloneCubeDepInIf`？**
   `SplitIfByBlockId` 会把一个 if 按 block id 拆成多个独立 if。原 if 内"后置 CUBE 块消费前置 CUBE 块结果"的 SSA 边，拆分后会变成**跨 if 边**，而 if 是独立 region，无法跨边界共享 SSA 值。本 Pass 提前把前置 CUBE 计算克隆进后置块，让每个 CUBE 块自洽，分裂后无跨块依赖。

2. **为什么 `collectCubeBlocksInRegion` 只收集 `CUBE_ONLY` op，跳过 VECTOR/UNDETERMINED？**
   避免误克隆。若把 VECTOR 块也当作边界/目标，可能因为"后置 CUBE 块依赖了某个 VECTOR 块的 tensor.empty 输出"就错误触发克隆。本 Pass 只关心 CUBE 块之间的跨块依赖（`SplitIfByBlockId` 分裂时 CUBE 块是关键），VECTOR/UNDETERMINED 一律当作透明。

3. **嵌套 `scf.if` 为什么"挂到最近的 CUBE 块"？**
   `linalg.fill` 这类初始化 op 常嵌在 `scf.if` 里，与它的 `memref.alloc` 宿主共同构成初始化链。把嵌套 if 挂到同 block_id 的 CUBE 块，克隆该 CUBE 块时就会 `deep-clone` 整条初始化链（`retagNestedBlockIds` 统一嵌套 block_id），保证"填零初始化 + alloc"关系完整，不会留下指向原块的悬空 fill。

4. **克隆体为什么要打两个标记 `ssbuffer.block_id` + `ssbuffer.clone`？**
   `block_id` = 后置块 id，让克隆体在物理/逻辑上都"属于"后置块（后续 Reorder/SplitIf 按 block_id 分组）；`clone` = 原 block id，既保留来源信息（debug / 语义追溯），又是 `cleanupSide` 的"只清理克隆体"判据——清理不能误删非克隆的原始 op。

5. **`processSide` 为什么从后往前，`cleanupSide` 也为什么从后往前？**
   `processSide` 后向处理：先让后置块自洽（克隆出前置依赖），再处理更前置块，避免重复克隆与顺序错乱。`cleanupSide` 后向删除：下游消费者先被判死，上游生产者随后发现"唯一消费者已删"也被删，用 `erasedOps` 屏蔽已完成级联，保证清理收敛且不误删还有活消费者的 op。

6. **`cleanupSide` 的 Rule 2 / Rule 3 分别覆盖什么 op，为什么分开？**
   Rule 2（有结果 op）：看"是否还有同 block_id 的活用户"——有 SSA 结果才谈得上"被另一个 op 使用"；Rule 3（无结果 op，如 store/内存副作用 op）：看"是否还有同 block_id 的 exec-after 活消费者"——这类 op 靠 `MemoryDependenceGraph` 的真实内存依赖序（而非 MLIR 顺序）来判定"后序 op 依赖它的副作用"。两规则把"数据依赖"与"内存副作用依赖"分开处理。

7. **为什么 `cloneEarlierCubesInto` 是"整段克隆"而不是"只克隆被依赖的 op"？**
   克隆要保证前置块的 USE-DEF 链在新位置闭合：后置块依赖的某个前置 op，它自己可能又依赖更靠前的 op，逐层追溯会退化成"几乎整块"。整段克隆（按 MLIR 顺序 + valueMap 累积）代价是产生冗余，但通过 `cleanupSide` 的 Rule 2/3 后向清理把冗余删掉——**"先激进克隆、再保守清理"**比"精细挑依赖"更简单可靠，也复用了 `CloneOps` 的成熟模式。

8. **本 Pass 与 `SplitIfByBlockId` 的 `CandidateIf` 同名但不同，如何区分职责？**
   本 Pass 的 `CandidateIf` 只记录 then/else 的 CUBE 块序列 + 是否 needsSplit，用于"克隆前置 CUBE 依赖"；`SplitIfByBlockId` 的 `CandidateIf` 记录完整 BlockGroup（CUBE/VECTOR/嵌套 if）+ 跨组值 + yield 槽位计划，用于"真正分裂"。两者是"准备"与"执行"的关系，共享 `walkMainLoop` 限定主循环范围与 `SplitIf` 目录的 BlockGroup 等通用结构。