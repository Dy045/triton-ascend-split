# DynamicCVPipeline 子 Pass 专题：按 block_id 分裂 if（05）

> 覆盖 `ComputeBlockOptPass` 阶段 E 中 `SplitIfByBlockIdPass`（[ComputeBlockOptPass.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOptPass.cpp) L88）
>
> 源码文件：
> - [`SplitIfByBlockId.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/SplitIfByBlockId.cpp)（Pass 主逻辑，1828 行）
> - [`Common.h`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/Common.h)（`BlockGroup` / `ScalarClosure` 声明）
> - [`ScalarClosure.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/ScalarClosure.cpp)（跨块标量收集 + 克隆）
> - [`WalkMainLoop.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/WalkMainLoop.cpp)（主循环查找工具）
>
> 流水线位置：阶段 E 中 `CloneCubeDepInIf`（L87）之后、`RelocateMemrefDecl`（L89）之前。**main-dev 从旧版阶段 C L70 移到了这里**，并在其前新增 `CloneCubeDepInIf` 作为前置准备（详见 07 篇）。

---

## 一、这个专题在做什么

阶段 B/C（02/03 篇）把 UB 依赖收紧、模式下沉之后，块已经按 `block_id` 划分完毕。但有一个结构性问题没有解决：**一个 `scf.if` 的 region 里可能混着多个 block_id 的 op**（if 是整体执行的，CUBE 块和 VECTOR 块的 op 被 if 绑死在一起，后续 `ReorderOpsByBlockId` 无法把它们分开调度）。

`SplitIfByBlockIdPass` 做的就是：**把主循环内"一个 if region 混多个 block_id 且 CUBE/VECTOR 双向交换数据"的 if 分裂成一串链式 if，每个 if 只包含一个 block_id 的 op**，每个组独立的 if 携带自己的结果签名，跨组 SSA 值通过 **yield 链**传递，跨组标量依赖通过**克隆**破坏。

分裂后，`ReorderOpsByBlockId` 才能真正按块重排——不同块（CUBE/VECTOR）的 if 可以放到不同核上并行/流水执行。

**五个阶段（在 Pass 内顺序执行）**：

| 阶段 | 函数 | 做什么 |
|---|---|---|
| 0. 分裂判定 | `getCandidate`（L479） | 按 block_id 分组后，用**双向跨核依赖**判定哪些 if 需要分裂、分裂 then 还是 else |
| 1. 预处理标量 | `preprocessScalarDependencies`（L514） | 把组内 op 依赖的**外部标量**克隆进组，破坏跨组标量依赖 |
| 2. 依赖分析 + yield 规划 | `analyzeDependencies`（L921） | 检测跨组 SSA 依赖，规划每个分裂 if 的结果签名（Case A/B + SlotKind 分类） |
| 3. 材质化 | `materializeCandidate`（L1563） | 统一循环生成链式 if：每组一个 if，then 放组 op，else 放占位/吸收另一侧，最后回填原始结果槽位 |

**失败即回退**：任意候选材质化失败 → `walkMainLoop` 返回 failure → `fallback.restore()` 回退整个 pass（标准编译路径）。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `walkMainLoop` | 递归查找**主循环**：含 matmul 的最内层 `scf.while/scf.for`。判定条件是累积 core_type == `CUBE_AND_VECTOR` 且未包含主循环（[WalkMainLoop.cpp](L34-L63)） |
| `CUBE_AND_VECTOR` 位或 | 循环的 core_type 是其 region 内所有 op core_type 的**位或**；等于 `CUBE_AND_VECTOR` 说明循环体内 CUBE/VECTOR 计算都有——这正是要拆分的混合块 |
| 跳过 kernel 黑名单 | `kSkippedKernels`（L70，当前为空数组 `{""}`），预留为已知无法正确分裂的 kernel 跳过分裂 |
| ambient op | block_id 为 -1（或 16）的 op。分组时并入当前活跃组（打上 currentId 标记） |
| **双向跨核依赖** | main-dev 新增的分裂判定：只有当 if 内同时存在 CUBE 组和 VECTOR 组，且 CUBE↔VECTOR **双向**交换数据（`cubeFeedsVector && vectorFeedsCube`）时才分裂。旧版"组数 ≥ 2 就分裂"太激进，会误拆两个纯 VECTOR（或纯 CUBE）组 |
| Case A | 原 if **无 yield**（无结果）。分裂后只有跨组值走 yield；最后组是 void if |
| Case B | 原 if **有 yield**（有结果）。分裂后最后组携带**全部原始结果类型**，非最后组带自己的产出值 |
| `SlotKind` 分类 | main-dev 新增的 Case B 原始结果槽位分类：`Delegated`/`NativeLast`/`Relay`/`External`（见 §3.8） |
| `carrierGroup(slot)` | 每个原始结果槽位由哪个分裂 if 暴露给外部：`Delegated` → 生产组自身；其余 → 最后组 |
| 统一 yield 链 | 跨组值变成 if 的 yield slot，消费者直接引用生产者 if 的结果（**jump reference**），不再用 memref 桥接 |
| void if | 纯副作用组生成无结果 if（`hasElse=false`），只保留 then 分支 |
| placeholder（占位值） | 分裂 if 的 else 分支需要 yield 与 then 相同类型的值；纯占位的 else 分支实际是**死分支**（条件为假时真正的语义由最后组 if 的 else 吸收），后续 DCE 消除 |
| `ScalarClosure` | 收集"组内 op 依赖的外部标量"，克隆进组并打上本组 block_id，破坏跨组标量依赖（标量无法走 tensor yield 链） |
| Scene 3/4 | 另一侧（未分裂侧）也有 op 的场景：**最后组 if 的 else 分支吸收这些 op**，保证条件为假时的语义 |
| `kSplittedIf` | 材质化后给分裂 if 打的**整型**标记（`SplittedIfTagGenerator` 递增）；`rearrangeIfOp` 据此移动分裂 if |
| `rearrangeIfOp` | 按 SSA 依赖 + 内存依赖（`memGraph.getExecBefore`）把分裂 if 移到最晚依赖之后，保证插入位置合法 |

---

## 三、逐行讲解

### 3.1 Pass 入口 `runOnOperation`（L1762-L1818）

```cpp
void SplitIfByBlockIdPass::runOnOperation() {
  SplittedIfTagGenerator tagGen;               // 整型分裂序列号
  ModuleOp module = getOperation();
  if (hasFallbackAttr(module)) return;         // fallback 短路
  FallbackHelper fallback{module};             // 失败时整体回退

  CVPipeline::ComputeBlockIdManager bm(module); // op ↔ block_id 双向映射

  auto mainRes = walkMainLoop(module, [&](Operation *op) {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (funcOp && llvm::is_contained(kSkippedKernels, funcOp.getSymName()))
      return llvm::success();                  // 黑名单 kernel 跳过

    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph{op, aa};  // 每个主循环重建 memGraph

    auto walkRes = op->walk([&](scf::IfOp ifOp) {
      auto candidate = getCandidate(ifOp, memGraph);     // ① 分组 + 双向跨核判定
      if (!candidate.needsSplit()) return WalkResult::advance();
      preprocessScalarDependencies(candidate);           // ② 标量克隆
      analyzeDependencies(candidate);                    // ③ 依赖 + yield 规划
      if (materializeCandidate(candidate, bm, tagGen).failed())
        return WalkResult::interrupt();                  // ④ 材质化，失败中断
      return WalkResult::advance();
    });
    if (walkRes.wasInterrupted()) return llvm::failure();
    return llvm::success();
  });

  if (llvm::failed(mainRes)) { fallback.restore(); return; }  // 整体回退

  // 成功后：把所有带 kSplittedIf 标记的 if 移到依赖之后（不删除标记）
  auto &aa = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memGraph{module, aa};
  module->walk([&](scf::IfOp ifOp) {
    if (ifOp->hasAttr(CVPipeline::kSplittedIf))
      rearrangeIfOp(ifOp, memGraph);
  });
}
```

**要点**：
- 只处理**主循环内**的 if（`walkMainLoop` 找到的循环体里 walk 所有 `scf::IfOp`），循环外的 if 不动。
- **main-dev 变化**：`memGraph` 从"全局建一次"改为"每个主循环 op 建一次"（更精确的别名分析范围），`getCandidate` 需要 memGraph 做双向跨核判定。
- 任意 if 材质化失败即**中断整个 walk** → fallback，保证不产生半分裂的畸形 IR。
- 末尾 `rearrangeIfOp` 不再移除 `kSplittedIf`（旧版会 `removeAttr`），让下游（如调试/后续 pass）仍可识别分裂产物。

### 3.2 主循环查找 `walkMainLoop`（WalkMainLoop.cpp L34-L63）

```cpp
llvm::FailureOr<WalkMainLoopResult>
walkMainLoop(Operation *op, function_ref<LogicalResult(Operation *)> pred) {
  CoreType coreType = CVPipeline::getOpCoreType(op);
  bool containsMainLoop = false;
  for (auto &region : op->getRegions()) {
    for (auto &block : region.getBlocks()) {
      for (auto &nestedOp : block) {
        auto nestedResult = walkMainLoop(&nestedOp, pred);
        if (failed(nestedResult)) return failure();
        coreType = static_cast<CoreType>(coreType | nested.coreType);  // 位或累积
        containsMainLoop = containsMainLoop || nested.containsMainLoop;
      }
    }
  }
  if (llvm::isa<scf::WhileOp, scf::ForOp>(op) &&
      coreType == CUBE_AND_VECTOR && !containsMainLoop) {
    if (pred(op).failed()) return failure();
    containsMainLoop = true;
  }
  return WalkMainLoopResult{coreType, containsMainLoop};
}
```

**要点**：从内向外递归、位或累积 core_type、`!containsMainLoop` 只找最内层主循环——与旧版一致，未变。

### 3.3 分组 `groupOpsInBlock`（L395-L477）

核心逻辑——**按 block_id 给 region 内 op 分组，嵌套 if 附着到最近组**：

```cpp
int currentId = -1;               // 嵌套/ambient op 附着的组
SmallVector<Operation *, 0> pendingAmbient;  // 第一个真实组之前的无 id op

for (auto &op : block) {
  if (isa<scf::YieldOp>(op)) continue;    // yield 不进组

  if (auto nestedIf = dyn_cast<scf::IfOp>(op)) {
    auto bid = CVPipeline::getOpBlockId(nestedIf);
    if (bid.has_value() && *bid != -1) {
      flushPending(*bid);
      getOrCreateGroup(*bid).nestedIfs.push_back(nestedIf);  // 自己成组
      currentId = *bid;
    } else if (currentId != -1) {
      getOrCreateGroup(currentId).nestedIfs.push_back(nestedIf);  // 附到当前组
    } else {
      pendingAmbient.push_back(&op);
    }
    continue;
  }

  auto bid = CVPipeline::getOpBlockId(&op);
  if (bid.has_value() && *bid != -1) {
    flushPending(*bid);
    getOrCreateGroup(*bid).ops.push_back(&op);   // 有 id → 独立成组
    currentId = *bid;
  } else if (currentId != -1) {
    op.setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(currentId));
    getOrCreateGroup(currentId).ops.push_back(&op);  // ambient 收编
  } else {
    pendingAmbient.push_back(&op);
  }
}

// 结尾：pending 刷入第一个真实组（若有组）
if (!pendingAmbient.empty() && !groups.empty()) {
  int firstBlockId = groups[0].blockId;
  for (auto *op : pendingAmbient) { ... groups[0].ops.push_back(op); }
}
return groups;
```

**要点**（与旧版一致）：每个 block_id 只建一个组；嵌套 if 优先按自己的 block_id 成组；ambient op 打当前组标记收编；最前面的 ambient 进 pending 刷入第一组。

### 3.4 分裂判定 `getCandidate` + `hasBidirectionalCrossCoreDeps`（L479-L511, L330-L374）

**这是 main-dev 的核心重写**。旧版 `getCandidate` 直接 `splitThen = thenGroups.size() >= 2`，新版本则先用 `hasBidirectionalCrossCoreDeps` 判定：

```cpp
static CandidateIf
getCandidate(scf::IfOp ifOp, const MemoryDependenceGraph &memGraph) {
  CandidateIf cand;
  cand.ifOp = ifOp;
  cand.hasYield = (ifOp->getNumResults() > 0);
  cand.selfBlockId = CVPipeline::getOpBlockId(ifOp).value_or(-1);

  // 在未合并的 raw 组上跑判定，保持原始块结构
  auto rawThenGroups = groupOpsInBlock(*ifOp.thenBlock());
  SmallVector<BlockGroup> rawElseGroups;
  if (auto elseBlk = ifOp.elseBlock())
    rawElseGroups = groupOpsInBlock(*elseBlk);

  // 只有 CUBE 与 VECTOR 组双向交换数据时才分裂
  cand.shouldSplitThen = hasBidirectionalCrossCoreDeps(rawThenGroups, memGraph);
  cand.shouldSplitElse = hasBidirectionalCrossCoreDeps(rawElseGroups, memGraph);
  if (!cand.needsSplit()) return cand;

  cand.thenGroups = std::move(rawThenGroups);
  cand.elseGroups = std::move(rawElseGroups);
  return cand;
}
```

`hasBidirectionalCrossCoreDeps`（L330-L374）三步：
1. `groupIsCube`（L255-L290）给每个组判类 CUBE_ONLY / VECTOR_ONLY / CUBE_AND_VECTOR / UNDETERMINED；要求同时存在纯 CUBE 组和纯 VECTOR 组；
2. `groupFeedsGroup`（L304-L326）沿 `DependencyHelper.forEachUser`（SSA+内存）判定 producer 组是否有 op 被 consumer 组消费；
3. 双循环遍历所有异类组对，检测 `cubeFeedsVector`（CUBE→VECTOR）与 `vectorFeedsCube`（VECTOR→CUBE），两者都成立才返回 true。

**设计意图**：纯 VECTOR 组之间、或纯 CUBE 组之间的 if 无需分裂（它们本就在同一核执行，没有跨核调度需求）；只有 CUBE/VECTOR 真正双向纠缠、必须拆开才能核间并行的情况下才分裂。

### 3.5 标量预处理 `preprocessScalarDependencies`（L514-L555）

逻辑与旧版一致：对 then/else 每个组，用 `ScalarClosure` 收集外部标量 → `capture` 克隆进组并重定向操作数。`ScalarClosure` 收集规则见下：

### 3.6 `ScalarClosure` 收集规则（ScalarClosure.cpp L30-L60）

```cpp
void ScalarClosure::collectScalarClosure(Value val) {
  if (!val || isa<TensorType>(val.getType())) return;   // tensor 走 yield 不克隆
  Operation *defOp = val.getDefiningOp();
  if (!defOp || llvm::isa<memref::AllocOp, memref::AllocaOp>(defOp)) return;  // alloc 不克隆
  auto *defBlock = defOp->getBlock();
  if (!defBlock || (defBlock != block &&
      (!includeParent || defBlock != parentBlock))) return;   // 只收 block/parentBlock
  collectOuterDependency(defOp);
  scalarOps.insert(defOp);
}
```

**收集边界**（三层过滤）：tensor 值不收（走 yield）；alloc/alloca 不收；定义块必须是 block 或 parentBlock。**main-dev 新增第二个构造 `ScalarClosure(Block*, ArrayRef<Operation*>, bool includeParent)`**，供 `ExpLoadPatternPass` 直接传 block + 关闭 includeParent 使用。`capture`（L95-L141）只克隆组内没有的标量并打组 block_id。

### 3.7 依赖分析 `analyzeDependencies`（L921-L971）

```cpp
static void analyzeDependencies(CandidateIf &candidate) {
  // Step 2.1: op → group 索引映射（含 nested ifs）
  // Step 2.2: 单趟 scanRegion 建 then/else 的 value 级 crossValueMap
  scanRegion(candidate.thenGroups, opToThenGroup, thenValueMap);
  scanRegion(candidate.elseGroups, opToElseGroup, elseValueMap);

  // 每个候选建自己的 DominanceInfo（只读，IR 修改在 materialize 之后）
  DominanceInfo dom(candidate.ifOp->getParentOp());

  // Step 2.3: 只对被判定为"该分裂"的那一侧做 yield 规划
  if (candidate.shouldSplitThen)
    planYield(candidate, /*splitThen=*/true,  candidate.thenGroups, opToThenGroup, thenValueMap, dom);
  else if (candidate.shouldSplitElse)
    planYield(candidate, /*splitThen=*/false, candidate.elseGroups, opToElseGroup, elseValueMap, dom);

  // Step 2.4: 调试 dump
  dumpYieldAugmentation(candidate);
}
```

**要点**：
- **main-dev 变化**：`splitThen` 不再来自"组数 ≥ 2"，而是 `candidate.shouldSplitThen`（已由双向跨核判定算好）；每侧独立判定至此收敛为"最多只分裂一侧"。
- `DominanceInfo dom(candidate.ifOp->getParentOp())` 是 main-dev 新增：用标准支配信息回答"原始 else 值是否支配源 if"，替代旧版 ad-hoc 的块遍历。

### 3.8 跨组依赖注册 + SlotKind 分类 + yield 规划

`addCrossGroupDependency`（L83-L101）：value → (producer 组, 消费者集合)，同组/无 def 不记录——与旧版一致。

main-dev 在 Case B 新增 **`SlotKind` 分类**（L137-L154），把"原 if 的每个结果槽位如何到达分裂链外部"分四类：

| 分类 | 条件 | 载体 | 传递方式 |
|---|---|---|---|
| `External` | 槽位值定义在分裂 region 外部（block 参数或 if 前定义） | 最后组 | `if(cond) then v else w` |
| `NativeLast` | 槽位由最后组产出 | 最后组 | 就地产出 |
| `Delegated` | 槽位由非最后组产出，且 else 值支配源 if（`domElse`） | 生产组自身 | 消费者读生产组 if 的结果 |
| `Relay` | 槽位由非最后组产出，但 else 值不可用 | 最后组 | 最后组转发生产组 if 的结果 |

`YieldAugmentation`（L172-L219）新增 `origYieldSlotKind` 与 `carrierGroup(slot)`（L212-L216）：`carrierGroup = (kind == Delegated) ? 生产组 : 最后组`。

`planYieldCaseB`（L691-L841）先按 `domElse` 判出每个槽位的 `SlotKind`，再为每个组填 `GroupOutputInfo`（Step 1 跨组产出值 + Step 2 本组产出的原始槽位），最后把 `Relay`/`External` 槽位追加给最后组（`carrierGroup(slot) == lastGi`）。`planYieldCaseA`（L845-L877）只填非最后组的跨组值，最后组 void。

### 3.9 材质化 `materializeCandidate`（L1563-L1700，main-dev 重写为统一循环）

```cpp
static LogicalResult
materializeCandidate(CandidateIf &c, ComputeBlockIdManager &bm,
                     SplittedIfTagGenerator &tagGen) {
  OpBuilder builder(c.ifOp);
  auto originalIf = c.ifOp;
  Value condition = originalIf.getCondition();
  int splittedIfTag = tagGen.next();

  bool splitThen = c.shouldSplitThen;
  auto &groups = splitThen ? c.thenGroups : c.elseGroups;
  auto otherCtx = collectOtherSideInfo(c, splitThen);

  // 分裂 else 侧时条件取反
  if (!splitThen)
    condition = builder.create<arith::XOrIOp>(loc, condition, trueVal).getResult();

  llvm::SmallDenseMap<Value, Value> crossValueReplacement;
  Operation *lastCreatedIf = originalIf.getOperation();
  SmallVector<Value> slotResults(ya.numOriginalSlots);   // 每个原始槽位的载体结果

  // 统一循环：每组一个 if（不再区分 Phase 1/Phase 2）
  for (unsigned gi = 0; gi < nGroups; ++gi) {
    auto &output = ya.groupOutputs[gi];
    bool isLastGroup = gi == nGroups - 1;
    bool hasElse = !output.isVoid() || (isLastGroup && otherCtx.hasOps);

    builder.setInsertionPointAfter(lastCreatedIf);
    scf::IfOp splittedIf = builder.create<scf::IfOp>(loc, output.outputTypes, condition, hasElse);
    splittedIf->setAttr(kBlockId, builder.getI32IntegerAttr(groups[gi].blockId));
    lastCreatedIf = splittedIf.getOperation();

    Block &thenBlock = splittedIf.getThenRegion().front();
    rewireAndMoveOps(groups[gi], crossValueReplacement, thenBlock);
    buildThenYieldForGroup(output, crossValueReplacement, thenBlock, loc, builder);

    if (hasElse) {
      Block &elseBlock = splittedIf.getElseRegion().front();
      if (isLastGroup && otherCtx.hasOps)  // Scene 3/4：吸收另一侧 op
        for (auto *op : otherCtx.ops) op->moveBefore(&elseBlock, elseBlock.end());
      if (buildElseYieldForGroup(...).failed()) return failure();
    }

    // 非最后组才发布跨组值（最后组含 Relay/External，禁止此步，见注释 hazard table）
    if (!isLastGroup)
      updateCrossValueReplacementGroup(output, splittedIf, thenBlock, crossValueReplacement);

    // 回填原始槽位：carrierGroup(slot) == gi 的槽位取本组 if 结果
    for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
      if (ya.carrierGroup(slot) != gi) continue;
      auto it = llvm::find(output.outputValues, ya.origYieldValues[slot]);
      if (output.isVoid() || it == output.outputValues.end()) return failure();
      slotResults[slot] = splittedIf.getResult(it - output.outputValues.begin());
    }

    postProcess(splittedIf, originalIf, groups[gi].blockId, splittedIfTag);
  }

  // 原 if 的结果替换为各自载体
  for (unsigned slot = 0; slot < ya.numOriginalSlots; ++slot) {
    if (!slotResults[slot]) return failure();
    originalIf.getResult(slot).replaceAllUsesWith(slotResults[slot]);
  }
  originalIf->erase();
  return success();
}
```

**main-dev 重写要点**（对比旧版 Phase1/Phase2 结构）：
- 旧版用 `for gi < nGroups-1`（非最后组）+ `if (hasYield) {...} else {...}`（最后组 Case A/B）两段；新版**统一 for 循环遍历所有组**，用 `isLastGroup` 分支局部区分。
- `hasElse` 判定统一为 `!output.isVoid() || (isLastGroup && otherCtx.hasOps)`——只有最后组且另一侧有 op 时才为空组生成 else。
- 新增 `slotResults` + `carrierGroup` 机制：每个原始槽位由**唯一载体组**产出结果，最后统一 `replaceAllUsesWith`，取代旧版"Phase 2 里一次性处理 last-if 结果"。
- `updateCrossValueReplacementGroup` **跳过最后组**（L1495-L1509 的注释详细说明原因：`External` 值替换会波及整个函数、`Relay` 会导致 use-before-def、`NativeLast` 无下游需发布）。
- `postProcess` 现在接收 `splittedIfTag`（整型序列号）。

`updateCrossValueReplacementGroup`（L1495-L1509）保持 jump reference 语义：`oldVal.replaceUsesWithIf(newVal, 排除本组 then 块内部)`。

### 3.10 占位值生成 `createPlaceholderValue`（L1068-L1244）

类型分派策略与旧版一致：tensor 追溯真实源（matmul/fill+empty/root tensor）否则 `tensor.empty + fill`；标量 `arith.constant 0`；memref 分 memory_space/stride 走 alloc/alloca + reinterpret_cast。辅助 `createMatmulPlaceHolderValue`（L1030-L1063）、`getRootTensor`（L1005-L1028）、`getRootMemRef`（L980-L1001）。

### 3.11 辅助函数族（L1246-L1337, L1339-L1465）

| 函数 | 行号 | 职责 |
|---|---|---|
| `rewireAndMoveOps` | L1246-L1309 | 替换跨组操作数后按原始顺序 moveBefore 到 then 块，打组 block_id |
| `ensureLocalValue` | L1313-L1337 | ViewLike/alloc 在 else 块本地克隆；constant 不克隆 |
| `buildThenYieldForGroup` | L1339-L1355 | 按 output.outputValues 顺序 yield（Relay 槽位从 crossValueReplacement 取值） |
| `valueDominates` | L1357-L1373 | 沿父链判断值是否支配目标块 |
| `buildElseYieldForGroup` | L1375-L1422 | Case B 原始 else 值（支配才用）+ 占位缓存 |
| `collectOtherSideInfo` | L1432-L1465 | 收集未分裂侧的 op + 原始 else yield 值（Scene 3/4 吸收用） |

### 3.12 `postProcess`（L1524-L1558，接收 splittedIfTag）

1. `setAttrs(sourceIfOp->getAttrs())` 克隆原 if 属性；
2. 从 then yield 各操作数 `getValueCoreType` join 重算 core_type 写回（无操作数则删除）；
3. 打 block_id（if 本身 + then/else yield；**不给 scf.yield terminator 打 block_id**，这是 main-dev 的修正——旧版会给 yield 打）；
4. 打 `kSplittedIf` **整型**标记（`splittedIfTag`）。

### 3.13 位置重排 `rearrangeIfOp`（L1702-L1733）

逻辑与旧版一致：沿 SSA 操作数 + `memGraph.getExecBefore` 内存依赖找到 block 内最晚依赖，`moveAfter` 修正位置。不同点在于调用后**不再移除 `kSplittedIf`**。

---

## 四、流程总结

### 4.1 单个候选 if 的处理流水

```
getCandidate（按 block_id 分组 + 双向跨核判定 shouldSplitThen/Else）
     │
     ▼
preprocessScalarDependencies ──► ScalarClosure.collect + capture（克隆外部标量）
     ▼
analyzeDependencies
     ├── scanRegion（op 级 + 值级跨组依赖收集）
     ├── planYield（Case A/B + SlotKind 分类 + carrierGroup）
     └── dumpYieldAugmentation（调试）
     ▼
materializeCandidate（统一循环）
     ├── !splitThen → 条件取反
     ├── 每组一个 if：then=组 op，else=占位；最后组吸收另一侧 op
     ├── 非最后组 updateCrossValueReplacementGroup（jump reference）
     ├── slotResults[slot] ← carrierGroup(slot) 的 if 结果
     └── replaceAllUsesWith(slotResults) + erase(originalIf)
     ▼
rearrangeIfOp（移到 SSA/内存依赖之后）
```

### 4.2 四种 if 形态（材质化产物）

| 形态 | 触发条件 | 结构 |
|---|---|---|
| void if | `output.isVoid()` 且非"最后组+另一侧有 op" | `if(cond) { ops; yield }` |
| result-bearing if | 有 output.outputValues | `if(cond) { ops; yield outs } else { placeholders; yield }` |
| Case B last-if | 原 if 有 yield，最后组 | 结果 = 原 if 全部槽位（NativeLast/Relay/External 槽位），替换原 if uses |
| Case A last-if | 原 if 无 yield，最后组 | void + 有另一侧 op 时 else 吸收 |

### 4.3 与前后 pass 的协作

- **前置 `CloneCubeDepInIf`（L87）**：先把 if 内"后置 CUBE 块依赖的前置 CUBE 块"克隆进后置块（详见 07 篇），让每个 CUBE 块自洽，`SplitIfByBlockId` 分裂后才不会产生跨 if 的 CUBE 依赖。
- **后置 `RelocateMemrefDecl`（L89）**：分裂后的 if 各自独立，跨同步的 memref 声明搬回消费者块。
- **下游 `ReorderOpsByBlockId`**：分裂 if 都打了本组 block_id，能按块重排；`AddControlFlowCondition` 等后续 pass 继续在其上插控制流。

---

## 五、面试要点

1. **为什么需要分裂 if？** 一个 if region 混多个 block_id 时，CUBE/VECTOR 的 op 被 if 整体绑死，`ReorderOpsByBlockId` 无法分开调度。分裂成链式 if 后每个 if 单 block_id，下游才能按块重排、流水。

2. **main-dev 的"双向跨核依赖"判定解决了旧版什么问题？** 旧版"组数 ≥ 2 就分裂"太激进：两个纯 VECTOR 组（或两个纯 CUBE 组）本就在同一核执行，分裂无收益甚至有害（多出链式 if 开销）。新判据要求 if 内**同时有 CUBE 和 VECTOR 组且双向交换数据**，只拆真正需要核间并行的场景。

3. **`SlotKind` 分类在解决什么？** 原 if 每个结果槽位有四种"来源 + 载体"组合（外部/最后组/生产组委托/需转发）。分类统一了"每个槽位由哪个分裂 if 暴露、怎么暴露"的规划，`carrierGroup` 据此在统一循环里填 `slotResults`，避免旧版 Phase1/Phase2 的特殊分支。
   - `Delegated`：生产组自身暴露（else 值可用，直接委托）；
   - `NativeLast`：最后组就地产出；
   - `Relay`：最后组转发生产组 if 的结果（else 值不可用）；
   - `External`：最后组以 `if(cond) then v else w` 携带外部值。

4. **为什么 `updateCrossValueReplacementGroup` 跳过最后组？** 最后组携带 `Relay`/`External` 槽位。`External` 值替换会波及整个封装函数（改写不相干块/前序迭代）；`Relay` 的生产组已发布该值，中间组体可能已读到（再替换造成 use-before-def）；`NativeLast` 则无下游需发布。所以最后组结果是原 if uses 的最终替换源，不再参与 crossValueReplacement。

5. **Case A vs Case B？** Case A 原 if 无结果 → 跨组值走新增 yield slot，最后组 void；Case B 原 if 有结果 → 最后组携带全部原始结果类型并替换原 if uses，非最后组只带自己的产出。

6. **跨组 SSA 值如何传递？** 统一 yield 链 + jump reference：跨组值变成 if 的 yield slot，`updateCrossValueReplacementGroup` 把旧值组外 uses 重定向到新 if 结果。不再用 memref 桥接。

7. **跨组标量为什么用克隆？** 标量无法走 tensor yield 链。`ScalarClosure` 收集"组内 op 依赖的外部标量"（非 tensor、非 alloc、定义块在 block/parentBlock），克隆进组并重定向操作数。

8. **失败安全如何保证？** 任意候选材质化失败 → interrupt → `walkMainLoop` failure → `fallback.restore()` 回退整个 pass，绝不产生半分裂 IR。

9. **`kSplittedIf` 从布尔改成整型有什么用？** `SplittedIfTagGenerator` 递增产生唯一序列号，便于区分同一 if 分裂出的多个 if 的先后序，调试与后续识别更精确；且 `rearrangeIfOp` 后不再移除标记，供下游追溯分裂产物。

10. **`DominanceInfo` 在这里为什么重要？** Case B 的 `Delegated` vs `Relay` 判据 `domElse` 需判断"原始 else 值是否支配源 if"。用标准 `DominanceInfo`（在源 if 的 parent op 上）回答，替代手工块遍历，避免把不可用的 else 值错误委托给生产组。

11. **与 `MergeVectorIfBlock` 的区别？** 方向相反：`MergeVectorIfBlock` 把纯 Vector 的 if **并入**上游块（消除 if 边界）；`SplitIfByBlockId` 把混合块 if **拆成**单 block_id 链（消除 if 内部混块）。一个收拢、一个拆分，服务同一目标：**让每个块/每个 if 与 block_id 一一对应**。