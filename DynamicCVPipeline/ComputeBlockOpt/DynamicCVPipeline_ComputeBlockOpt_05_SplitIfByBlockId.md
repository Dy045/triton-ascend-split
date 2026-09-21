# DynamicCVPipeline 子 Pass 专题：按 block_id 分裂 if（05）

> 覆盖 `ComputeBlockOptPass` 阶段 C 中 `SplitIfByBlockIdPass`（[ComputeBlockOptPass.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOptPass.cpp) L70）
>
> 源码文件：
> - [`SplitIfByBlockId.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/SplitIfByBlockId.cpp)（Pass 主逻辑，~1700 行）
> - [`Common.h`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/Common.h)（`BlockGroup` / `ScalarClosure` 声明）
> - [`ScalarClosure.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/ScalarClosure.cpp)（跨块标量收集 + 克隆）
> - [`WalkMainLoop.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/ComputeBlockOpt/SplitIfByBlockId/WalkMainLoop.cpp)（主循环查找工具）
>
> 流水线位置：阶段 C 中 `FixpipeOpt`（L69）之后、`MoveLoadIntoUser`（L72）之前，其后紧跟一次 `ReorderOpsByBlockId`（L71）

---

## 一、这个专题在做什么

阶段 B（02 篇）把 UB 依赖收紧后，块已经按 `block_id` 划分完毕。但有一个结构性问题没有解决：**一个 `scf.if` 的 region 里可能混着多个 block_id 的 op**（if 是整体执行的，CUBE 块和 VECTOR 块的 op 被 if 绑死在一起，后续 `ReorderOpsByBlockId` 无法把它们分开调度）。

`SplitIfByBlockIdPass` 做的就是：**把主循环内"一个 if region 混多个 block_id"的 if 分裂成一串链式 if，每个 if 只包含一个 block_id 的 op**，每个组独立的 if 携带自己的结果签名，跨组 SSA 值通过 **yield 链**传递，跨组标量依赖通过**克隆**破坏。

分裂后，`ReorderOpsByBlockId`（L71）才能真正按块重排——不同块（CUBE/VECTOR）的 if 可以放到不同核上并行/流水执行。

**四个阶段（在 Pass 内顺序执行）**：

| 阶段 | 函数 | 做什么 |
|---|---|---|
| 1. 分组 | `groupOpsInBlock`（L206） | 把 if 的 then/else region 按 block_id 分组，嵌套 if 附着到最近组 |
| 2. 预处理标量 | `preprocessScalarDependencies`（L312） | 把组内 op 依赖的**外部标量**克隆进组，破坏跨组标量依赖 |
| 3. 依赖分析 + yield 规划 | `analyzeDependencies`（L689） | 检测跨组 SSA 依赖，规划每个分裂 if 的结果签名（Case A/B） |
| 4. 材质化 | `materializeCandidate`（L1401） | 真正生成链式 if：每组一个 if，then 放组 op，else 放占位/吸收另一侧 |

**失败即回退**：任意候选材质化失败 → `walkMainLoop` 返回 failure → `fallback.restore()` 回退整个 pass（标准编译路径）。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `walkMainLoop` | 递归查找**主循环**：含 matmul 的最内层 `scf.while/scf.for`。判定条件是累积 core_type == `CUBE_AND_VECTOR` 且未包含主循环（[WalkMainLoop.cpp](L34-L63)） |
| `CUBE_AND_VECTOR` 位或 | 循环的 core_type 是其 region 内所有 op core_type 的**位或**；等于 `CUBE_AND_VECTOR` 说明循环体内 CUBE/VECTOR 计算都有——这正是要拆分的混合块 |
| 跳过 kernel 黑名单 | `parallel_deltaformer_fwd_kernel` / `chunkwise_fwd_kernel` / `parallel_nsa_fwd_kernel`（L68-L73），这三个 kernel 跳过分裂 |
| ambient op | block_id 为 -1（或 16）的 op。分组时并入当前活跃组（打上 currentId 标记） |
| Case A | 原 if **无 yield**（无结果）。分裂后只有跨组值走 yield；最后组是 void if |
| Case B | 原 if **有 yield**（有结果）。分裂后最后组携带**全部原始结果类型**，非最后组带自己的产出值 |
| 统一 yield 链 | 跨组值变成 if 的 yield slot，消费者直接引用生产者 if 的结果（**jump reference**），不再用 memref 桥接 |
| void if | 纯副作用组生成无结果 if（`hasElse=false`），只保留 then 分支 |
| placeholder（占位值） | 分裂 if 的 else 分支需要 yield 与 then 相同类型的值；纯占位的 else 分支实际是**死分支**（条件为假时真正的语义由最后组 if 的 else 吸收），后续 DCE 消除 |
| `ScalarClosure` | 收集"组内 op 依赖的外部标量"，克隆进组并打上本组 block_id，破坏跨组标量依赖（标量无法走 tensor yield 链） |
| Scene 3/4 | 另一侧（未分裂侧）也有 op 的场景：**最后组 if 的 else 分支吸收这些 op**，保证条件为假时的语义 |
| `kSplittedIf` | 材质化后给分裂 if 打的标记；`rearrangeIfOp` 根据标记把所有分裂 if 移动到依赖之后 |
| `rearrangeIfOp` | 按 SSA 依赖 + 内存依赖（`memGraph.getExecBefore`）把分裂 if 移到最晚依赖之后，保证插入位置合法 |

---

## 三、逐行讲解

### 3.1 Pass 入口 `runOnOperation`（L1664-L1718）

```cpp
ModuleOp module = getOperation();
if (hasFallbackAttr(module)) return;           // fallback 短路
FallbackHelper fallback{module};               // 失败时整体回退
CVPipeline::ComputeBlockIdManager bm(module);  // op ↔ block_id 双向映射

auto mainRes = walkMainLoop(module, [&](Operation *op) {
  // 黑名单 kernel 跳过
  if (funcOp && (name == kSkippedDeltaformerKernel || ...)) return success();
  // 对主循环内所有 scf.if：
  auto walkRes = op->walk([&](scf::IfOp ifOp) {
    auto candidate = getCandidate(ifOp);          // ① 分组
    preprocessScalarDependencies(candidate);      // ② 标量克隆
    analyzeDependencies(candidate);               // ③ 依赖 + yield 规划
    if (materializeCandidate(candidate, bm).failed())
      return WalkResult::interrupt();             // ④ 材质化，失败中断
    return WalkResult::advance();
  });
  if (walkRes.wasInterrupted()) return failure();
  return success();
});

if (llvm::failed(mainRes)) { fallback.restore(); return; }  // 整体回退

// 成功后：把所有带 kSplittedIf 标记的 if 移到依赖之后
auto &aa = getAnalysis<AliasAnalysis>();
MemoryDependenceGraph memGraph{module, aa};
module->walk([&](scf::IfOp ifOp) {
  if (ifOp->hasAttr(CVPipeline::kSplittedIf)) {
    rearrangeIfOp(ifOp, memGraph);
    ifOp->removeAttr(CVPipeline::kSplittedIf);
  }
});
```

**要点**：
- 只处理**主循环内**的 if（`walkMainLoop` 找到的循环体里 walk 所有 `scf::IfOp`），循环外的 if 不动——主循环才是 CUBE/VECTOR 并行的核心区域。
- 任意 if 材质化失败即**中断整个 walk** → fallback，保证不产生半分裂的畸形 IR。
- `walkMainLoop` 的 `pred` 返回 `failure()` 时不走 fallback 分支（因为 `walkMainLoop` 递归中失败只向上传播），但 materialize 失败走 interrupt → `wasInterrupted` → failure。

### 3.2 主循环查找 `walkMainLoop`（WalkMainLoop.cpp L34-L63）

```cpp
llvm::FailureOr<WalkMainLoopResult>
walkMainLoop(Operation *op, function_ref<LogicalResult(Operation *)> pred) {
  CoreType coreType = CVPipeline::getOpCoreType(op);
  bool containsMainLoop = false;
  // 递归遍历所有 region/block/nestedOp
  for (auto &region : op->getRegions()) {
    for (auto &block : region.getBlocks()) {
      for (auto &nestedOp : block) {
        auto nestedResult = walkMainLoop(&nestedOp, pred);
        ...
        coreType = static_cast<CoreType>(coreType | nested.coreType);  // 位或累积
        containsMainLoop = containsMainLoop || nested.containsMainLoop;
      }
    }
  }
  // 当前 op 是 while/for 且整棵树是 CUBE+VECTOR 混合且还没有主循环
  if (llvm::isa<scf::WhileOp, scf::ForOp>(op) &&
      coreType == CUBE_AND_VECTOR && !containsMainLoop) {
    if (pred(op).failed()) return failure();
    containsMainLoop = true;
  }
  return WalkMainLoopResult{coreType, containsMainLoop};
}
```

**要点**：
- **从内向外**递归：先遍历子 op 再检查自己，所以 `pred` 命中**最内层**同时满足条件的循环。
- `!containsMainLoop` 保证**只找最内层那一个主循环**——外层再遇到 `CUBE_AND_VECTOR` 循环时因 `containsMainLoop=true` 不再触发。
- `CUBE_AND_VECTOR` 位或：循环的 core_type 由体内所有 op 位或而来，等于 `CUBE_AND_VECTOR` 意味着体内 CUBE、VECTOR 计算都有，且 `pred` 实际会去 walk 该循环内所有 if 做分裂。

### 3.3 分组 `groupOpsInBlock`（L206-L288）

核心逻辑——**按 block_id 给 region 内 op 分组，嵌套 if 附着到最近组**：

```cpp
int currentId = -1;               // 嵌套/ambient op 附着的组
SmallVector<Operation *> pendingAmbient;  // 第一个真实组之前的无 id op

for (auto &op : block) {
  if (isa<scf::YieldOp>(op)) continue;    // yield 不进组

  if (auto nestedIf = dyn_cast<scf::IfOp>(op)) {
    auto bid = CVPipeline::getOpBlockId(nestedIf);
    if (bid.has_value() && *bid != -1) {
      flushPending(*bid);                    // 先把 pending 刷进目标组
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
    // ambient op（id == -1 或 16）：打上当前组标记并入组
    op.setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(currentId));
    getOrCreateGroup(currentId).ops.push_back(&op);
  } else {
    pendingAmbient.push_back(&op);              // 没有组时先挂着
  }
}

// 结尾：pending 刷入第一个真实组（若有组）
if (!pendingAmbient.empty() && !groups.empty()) {
  int firstBlockId = groups[0].blockId;
  for (auto *op : pendingAmbient) { ... groups[0].ops.push_back(op); }
}
return groups;
```

**要点**：
- **每个 block_id 只建一个组**（`idToIdx` 去重），`BlockGroup` 含 `ops`（本组 op）+ `nestedIfs`（属于本组的嵌套 if）。
- 嵌套 if 优先按**自己的 block_id** 成组（对应"前一轮分裂后同一 region 又混了不同 block_id 嵌套 if"的场景）；没有有效 id 才附到当前活跃组。
- **ambient op**（无 block_id 或 id=-1/16）被"收编"：打上当前组的 block_id 并入组——它们要么是分裂产生的边缘 op，要么是标量计算，跟着当前块走不会出错。
- 最开始的 ambient op 进 `pendingAmbient`，在遇到第一个真实组或结束时统一刷入，保证 IR 顺序语义。

`getCandidate`（L290-L309）把 then/else 两个 region 分别分组，记录 `selfBlockId`（if 自己的 block_id）和 `hasYield`（是否有结果）。

### 3.4 标量预处理 `preprocessScalarDependencies`（L312-L353）

分裂后组是独立 if，组内 op 引用的**外部标量**（定义在别的组或外层）会形成跨组依赖。标量不能像 tensor 一样走 yield 链（slot 太多、类型杂），所以直接**克隆进组**：

```cpp
auto processGroup = [](BlockGroup &group) {
  auto allOps = concat(group.ops, group.nestedIfs);
  ScalarClosure closure{group, allOps};
  closure.collect();                          // ① 收集外部标量依赖

  auto [newOps, mapper] = closure.capture(group.blockId);  // ② 克隆 + 映射
  if (newOps.empty()) return;

  allOps.append(newOps);
  group.nestedIfs.clear(); group.ops.clear();
  llvm::sort(allOps, [&closure](Operation *a, Operation *b) {
    return closure.isBefore(a, b);            // ③ 按原始顺序排序
  });

  for (Operation *op : allOps) {
    // ④ 用 mapper 把组内所有 op 的操作数重定向到克隆值
    op->walk([&](Operation *innerOp) {
      for (OpOperand &operand : innerOp->getOpOperands()) {
        Value target = mapper.lookupOrNull(operand.get());
        if (target) operand.set(target);
      }
    });
    // 恢复 ops / nestedIfs 分类
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) group.nestedIfs.push_back(ifOp);
    else group.ops.push_back(op);
  }
};
llvm::for_each(concat(cand.thenGroups, cand.elseGroups), processGroup);
```

**要点**：`capture` 只克隆组内 op **没有的**标量（`is_contained(ops, scalarOp)` 时跳过但推进插入点），克隆体打上本组 block_id（`ScalarClosure.cpp` L124-L138）。这样组内 op 的标量操作数全部变成组内定义，跨组 SSA 只剩 tensor 值（走 yield 链）。

### 3.5 `ScalarClosure` 收集规则（ScalarClosure.cpp L22-L60）

```cpp
void ScalarClosure::collectScalarClosure(Value val) {
  if (!val || isa<TensorType>(val.getType())) return;   // tensor 不走克隆（走 yield）
  Operation *defOp = val.getDefiningOp();
  if (!defOp || isa<memref::AllocOp, memref::AllocaOp>(defOp)) return;  // alloc 不克隆
  auto *defBlock = defOp->getBlock();
  // 定义块必须是本组 block（或 includeParent 时的父 block）
  if (!defBlock || (defBlock != block &&
      (!includeParent || defBlock != parentBlock))) return;
  collectOuterDependency(defOp);   // 递归收集该 def op 的标量操作数
  scalarOps.insert(defOp);
}
```

**收集边界**（三层过滤）：
1. **tensor 值不收**——tensor 跨组走 yield 链，不需要克隆；
2. **alloc/alloca 不收**——memref 分配不能克隆（会引入多余缓冲）；
3. **定义块必须是 block 或 parentBlock**——只收"组内或紧邻外层"的标量，防止无限外扩（`includeParent` 用于主循环场景，避免把更外层 memref 依赖卷进来）。

### 3.6 依赖分析 `analyzeDependencies`（L689-L733）

```cpp
// Step 2.1: 构建 op → group 索引映射（含 nested ifs）
// Step 2.2: 单趟扫描 then/else 的 crossValueMap
scanRegion(candidate.thenGroups, opToThenGroup, thenValueMap);
scanRegion(candidate.elseGroups, opToElseGroup, elseValueMap);

// Step 2.3: 对"被分裂侧"（组数≥2 的那一侧）做 yield 规划
bool splitThen = candidate.thenGroups.size() >= 2;
if (splitThen)      planYield(c, /*splitThen=*/true,  thenGroups, opToThenGroup, thenValueMap);
else if (elseGroups.size() >= 2)
                    planYield(c, /*splitThen=*/false, elseGroups, opToElseGroup, elseValueMap);

// Step 2.4: 调试 dump
dumpYieldAugmentation(candidate);
```

**要点**：
- **哪侧组数 ≥ 2 就分裂哪侧**（then 优先）。另一侧保持不动。
- `scanRegion`（L432-L495）对每组 op 的操作数、region 内层操作数、嵌套 if 操作数做 `addCrossGroupDependency`，并**反向扫描嵌套 if 的结果**（结果被别的组消费也算跨组值——walk up 找最近被 `opToGroup` 追踪的祖先）。

### 3.7 跨组依赖注册 `addCrossGroupDependency`（L88-L106）

```cpp
auto *defOp = val.getDefiningOp();
if (!defOp) return;                                  // block 参数不算
auto it = opToGroup.find(defOp);
if (it == opToGroup.end() || it->second == consumerGroup) return;  // 同组不算
auto &entry = crossValueMap[val];                    // value → (producer组, 消费者集合)
entry.first = it->second;
entry.second.insert(consumer);
```

`crossValueMap` 是 **value 级**（记录具体哪些 SSA 值跨组），区别于 `opToGroup` 的 op 级。后续 `planYield` 把它转成 `CrossGroupValue` 列表（去重 + 按 producer 组排序，L674-L680）。

### 3.8 Yield 规划 `planYield` + Case A/B（L648-L687, L501-L605, L610-L642）

`planYield` 先统一构建 `ya.crossValues`（稳定排序：producer 组序 → value 指针），再按 `c.hasYield` 分流：

**Case A（原 if 无结果，L610-L642）**：非最后组 yield 各自产生的跨组值；**最后组永远是 void**（没有原始结果）。

**Case B（原 if 有结果，L501-L605）**：
- 记录原始 yield 槽位：`numOriginalSlots`、`origYieldValues`、`origYieldProducerGroup`（每个槽位值由哪个组产出，block 参数记 -1）。
- 对**每个非最后组**：
  - Step 1：该组产出的**跨组值**（同时在 `origValToSlot` 里查它是否也是原始 yield 槽位，是则记录对应的**另一侧 yield 值**到 `origElseValues`）；
  - Step 2：该组**拥有的原始 yield 值**（`origYieldProducerGroup[slot] == gi` 且未作为跨组值加入过的）。
- 最后组信息存在 `YieldAugmentation` 顶层（`origYieldValues`/`origYieldProducerGroup`），供材质化阶段构建 last-if。

**`GroupOutputInfo` 的含义**：每个非最后组 if 的 then 要 yield 的值（`outputValues`）+ 对应 else 分支的兜底值（`origElseValues`，Case B 原始 else 侧值；否则空 → 占位）。

### 3.9 材质化 `materializeCandidate`（L1401-L1602）

```cpp
bool splitThen = c.thenGroups.size() >= 2;
auto &groups = splitThen ? c.thenGroups : c.elseGroups;
unsigned nGroups = groups.size();
auto otherCtx = collectOtherSideInfo(c, splitThen);   // 另一侧信息（op + yield 值）

builder.setInsertionPoint(originalIf);

// Step 3.2: 分裂 else 侧时条件取反（then 侧保持条件原样）
if (!splitThen) {
  auto trueVal = builder.create<arith::ConstantOp>(loc, i1(true));
  condition = builder.create<arith::XOrIOp>(loc, condition, trueVal).getResult();
}

llvm::SmallDenseMap<Value, Value> crossValueReplacement;
Operation *lastCreatedIf = originalIf.getOperation();

// Phase 1: 每个非最后组生成一个独立 if
for (unsigned gi = 0; gi < nGroups - 1; ++gi) {
  auto &output = ya.groupOutputs[gi];
  if (output.isVoid()) {
    // 纯副作用组：void if（无结果、无 else）
    splittedIf = builder.create<scf::IfOp>(loc, condition, /*hasElse=*/false);
    rewireAndMoveOps(groups[gi], crossValueReplacement, thenBlock);
    builder.create<scf::YieldOp>(loc);              // 空 yield
  } else {
    // 结果组：if 带本组输出类型签名
    splittedIf = builder.create<scf::IfOp>(loc, output.outputTypes, condition, true);
    rewireAndMoveOps(groups[gi], crossValueReplacement, thenBlock);
    buildThenYieldForGroup(output, thenBlock, loc, builder);      // then: 本组产出
    buildElseYieldForGroup(groups[gi].blockId, output, elseBlock, loc, builder);  // else: 占位
    updateCrossValueReplacementGroup(output, splittedIf, thenBlock, crossValueReplacement);
  }
  postProcess(splittedIf, originalIf, groups[gi].blockId);
}

// Phase 2: 最后组
if (c.hasYield) {
  // Case B：last-if 携带 ALL 原始结果类型
  lastIf = builder.create<scf::IfOp>(loc, lastIfTypes, condition, true);
  rewireAndMoveOps(groups[lastGi], crossValueReplacement, thenBlock);
  buildThenYieldForLastIf(lastGi, ya, crossValueReplacement, thenBlock, loc, builder);
  buildElseYieldForLastIf(groups[lastGi].blockId, otherCtx, lastIfTypes, elseBlock, loc, builder);
  // 原 if 的 uses 全部替换为 last-if 的结果
  for (unsigned ri = 0; ri < ya.numOriginalSlots; ++ri)
    originalIf.getResult(ri).replaceAllUsesWith(lastIf.getResult(ri));
} else {
  // Case A：最后组是 void if（无原始结果）
  bool hasElse = otherCtx.hasOps;
  lastIf = builder.create<scf::IfOp>(loc, condition, hasElse);
  rewireAndMoveOps(groups[lastGi], crossValueReplacement, thenBlock);
  builder.create<scf::YieldOp>(loc);
  if (hasElse) { /* else 吸收另一侧 op */ }
}
postProcess(lastIf, originalIf, groups[lastGi].blockId);

originalIf->erase();   // 删掉原 if
```

**要点**：
- **每次创建 if 后必须重置插入点**（`builder.setInsertionPointAfter(lastCreatedIf)`）——填充 then/else 会把 builder 留在那些 block 内部（注释 L1438-L1439 专门强调）。
- **jump reference**：`updateCrossValueReplacementGroup`（L1328-L1342）把旧跨组值的 uses 重定向到新 if 的结果（排除 then 块内部）并记录 `crossValueReplacement`；下游组的 `rewireAndMoveOps` 据此替换操作数，最后组 `buildThenYieldForLastIf` 据此把早期组产出的原始 yield 值**再 yield**出来（jump reference，非 passthrough 链）。
- **Scene 3/4**：另一侧有 op 时，`buildElseYieldForLastIf`/Case A 分支把它们 **move 进 last-if 的 else 块**——条件为假时执行原另一侧逻辑，语义完整。
- **void if**：纯副作用组没有产出值，`hasElse=false` 的 if 只有一个 then（代码生成的 if），不产生 SSA 值。

### 3.10 占位值生成 `createPlaceholderValue`（L839-L1019）

分裂 if 的 **else 分支**需要 yield 与 then 同类型的值，但"条件为假时这些 slot 的真正语义"由 last-if 的 else 承担，中间组的 else 是**死分支**（DCE 会消除）。占位值按类型分派：

| 类型 | 占位策略 |
|---|---|
| `RankedTensorType` | 优先 `hivm::traceDefOp<MatmulOp>` 走 `createMatmulPlaceHolderValue`；否则追溯 `getRootTensor` 的真实源（函数参数等，要求**支配**插入点且非 -1 block_id）；都不行 → `tensor.empty` |
| `FloatType` / `IntegerType` / `index` | `arith.constant 0` |
| `MemRefType` | 带 address_space（cbuf/ub）→ `memref.alloca`（特殊空间需要合法分配）；普通 memref → 静态 `alloc` / 动态 `alloca`，再套 `reinterpret_cast` 恢复 strided layout。有 memory_space 时用函数参数根 |

**为什么 memref 占位用 alloc + reinterpret_cast 而不是直接复用根值**：注释（L899-L901）解释——纯 `tensor.empty + to_memref` 会让下游 `tracebackMemRefToAlloc` 提前停在错误分支；而函数参数根（GM）只在"strided 且无 memory_space"时复用（L910-L919），保证两个分支的 provenance 一致。

**matmul 特殊处理 `createMatmulPlaceHolderValue`（L801-L831）**：
- bias 是 **block 参数**且能在 if 块内找到祖先 → 直接用 bias；
- bias 来自 `fill(tensor.empty)` → 新建同 shape 的 `tensor.empty`；
- 其他 → failure（触发材质化失败 → fallback）。

### 3.11 辅助函数族（L1023-L1324）

| 函数 | 行号 | 职责 |
|---|---|---|
| `rewireAndMoveOps` | L1023-L1083 | 把组内 op + 嵌套 if 的跨组操作数替换为 `crossValueReplacement` 新值，然后按原始 IR 顺序 `moveBefore` 到目标 then 块，统一打上组 block_id |
| `ensureLocalValue` | L1092-L1117 | 保证值能在 else 块安全 yield：`ViewLikeOpInterface`（reinterpret_cast/subview）和 alloc 克隆到 else 块本地；`arith.constant` 不克隆（占位已支配） |
| `buildThenYieldForGroup` | L1121-L1130 | 非最后组 then yield：按 `output.outputValues` 顺序 yield |
| `valueDominates` | L1134-L1150 | 沿父链检查值是否支配目标块 |
| `buildElseYieldForGroup` | L1158-L1203 | 非最后组 else yield：Case B 原始 else 值（`valueDominates` 才用，否则占位）+ 占位缓存（同类型只建一个 constant）；纯跨组槽位走 `createPlaceholderValue` |
| `collectOtherSideInfo` | L1213-L1247 | 收集未分裂侧的 op（Scene 3/4 吸收用）+ 原始另一侧 yield 值 |
| `buildThenYieldForLastIf` | L1257-L1281 | Case B last-if then yield：**本组产出或外部值直接 yield；早期组产出 → 从 `crossValueReplacement` 取新值 re-yield** |
| `buildElseYieldForLastIf` | L1285-L1324 | Case B last-if else yield：吸收另一侧 op + 原始另一侧 yield 值（`ensureLocalValue`）+ 超出的槽位占位 |

**`updateCrossValueReplacementGroup`（L1328-L1342）的关键细节**：

```cpp
oldVal.replaceUsesWithIf(newVal, [&](OpOperand &operand) {
  return !thenBlocks.contains(operand.getOwner()->getBlock());  // 排除本组 then 内部
});
```

只用 if 结果替换**组外**的 uses（本组 then 内的 uses 保留原始值），随后记录 `crossValueReplacement[oldVal] = newVal`。这保证了"组内消费自己的定义、组外消费 if 结果"的正确边界。

### 3.12 `postProcess`（L1351-L1386）

分裂 if 生成后统一收尾：
1. **克隆原 if 的 attrs**（core_type 等属性随分裂 if 保留）；
2. **重算 core_type**：从 then yield 各操作数的 `getValueCoreType` join 成字符串写回 `kCoreType`（无操作数则删除）；
3. **打 block_id**：if 本身 + then/else yield 统一设为该组 block_id；
4. **打 `kSplittedIf` 标记**：供后续 `rearrangeIfOp` 识别。

### 3.13 位置重排 `rearrangeIfOp`（L1604-L1635）

```cpp
Operation *lastDependency = nullptr;
Block *block = ifOp->getBlock();
// ① 找 if 内所有 op 的操作数定义中"在 block 内最晚"的祖先
ifOp->walk([&](Operation *nestedOp) {
  for (Value operand : nestedOp->getOperands()) {
    auto defOp = operand.getDefiningOp();
    if (!defOp || ifOp->isAncestor(defOp)) continue;     // 内部定义跳过
    auto defOpInBlock = block->findAncestorOpInBlock(*defOp);
    if (!defOpInBlock) continue;
    if (!lastDependency || lastDependency->isBeforeInBlock(defOpInBlock))
      lastDependency = defOpInBlock;
  }
});
// ② 加上内存依赖中最晚的
for (auto *memDep : memGraph.getExecBefore(ifOp)) { ... }
// ③ 移到依赖之后
if (lastDependency) ifOp->moveAfter(lastDependency);
```

**要点**：材质化把链式 if 插在原 if 位置，但原 if 可能原本就位于某些依赖 op 之前（依赖在 if 外部产生）。`rearrangeIfOp` 沿 SSA + 内存依赖把每个分裂 if 推到**所有依赖之后**，确保插入位置不破坏依赖序（这是 `moveAfter` 修正而非裸插的原因）。

---

## 四、流程总结

### 4.1 单个候选 if 的处理流水

```
getCandidate ──► groupOpsInBlock（then/else 各自按 block_id 分组）
     │
     ▼
preprocessScalarDependencies ──► ScalarClosure.collect + capture
     │                               （克隆外部标量进组，重定向操作数）
     ▼
analyzeDependencies
     ├── scanRegion（op 级 + 值级跨组依赖收集）
     ├── planYield（Case A / Case B 签名规划）
     └── dumpYieldAugmentation（调试）
     ▼
materializeCandidate
     ├── !splitThen → 条件取反
     ├── Phase 1：非最后组 → void if / result-bearing if（then=组 op，else=占位）
     │                └── updateCrossValueReplacementGroup（jump reference 注册）
     ├── Phase 2：最后组 → Case B（全原始类型）/ Case A（void）+ 吸收另一侧
     └── erase(originalIf)
     ▼
postProcess（attrs 克隆 + core_type 重算 + block_id + kSplittedIf）
     ▼
rearrangeIfOp（移到 SSA/内存依赖之后）＋ 移除 kSplittedIf
```

### 4.2 四种 if 形态（材质化产物）

| 形态 | 触发条件 | 结构 |
|---|---|---|
| void if | 非最后组且无产出（纯副作用） | `if(cond) { ops; yield }`（无 else、无结果） |
| result-bearing if | 非最后组且有跨组/原始产出 | `if(cond) { ops; yield outs } else { placeholders; yield }` |
| Case B last-if | 原 if 有 yield | `if(cond) { ops; yield 全原始槽位 } else { 另一侧 ops + 原 else 值 + 占位 }`，结果替换原 if 的 uses |
| Case A last-if | 原 if 无 yield | `if(cond) { ops; yield }`，有另一侧 op 时 else 吸收 |

### 4.3 与前后 pass 的协作

- **前置**：`FixpipeOpt`（L69）处理 FIXPIPE 量化链；`SplitIfByBlockId` 把 if 内混合块拆开，让后续 `ReorderOpsByBlockId`（L71）能真正按块重排；
- **后置**：`MoveLoadIntoUser`（L72）等按块移动 op 时不再受"if 整体"约束；
- **协作产物**：分裂 if 都打了本组 block_id，后续 `MergeSmallBlockPass`（L76）、`MergeComputeBlockPass`（L79）能按块做进一步合并/调度。

---

## 五、面试要点

1. **为什么需要分裂 if？** 一个 if region 混多个 block_id 时，CUBE/VECTOR 的 op 被 if 整体绑死，`ReorderOpsByBlockId` 无法分开调度。分裂成链式 if 后每个 if 单 block_id，下游才能按块重排、流水。

2. **主循环怎么找？** `walkMainLoop` 递归位或累积 core_type，命中"最内层且 core_type == CUBE_AND_VECTOR 的 while/for"。`CUBE_AND_VECTOR` 位或 = 循环体内 CUBE、VECTOR 计算都有，正是混合场景。只处理主循环内 if，循环外不动。

3. **Case A vs Case B 的区别？** Case A 原 if 无结果 → 跨组值走新增 yield slot，最后组 void；Case B 原 if 有结果 → 最后组携带全部原始结果类型并替换原 if 的 uses，非最后组只带自己的产出。本质是"每个分裂 if 按自己实际产出决定结果签名"（无统一 slot 规划）。

4. **跨组 SSA 值如何传递？** 统一 yield 链 + jump reference：跨组值变成 if 的 yield slot，`updateCrossValueReplacementGroup` 把旧值组外 uses 重定向到新 if 结果，最后组再 re-yield。不再用 memref 桥接（旧设计）。

5. **跨组标量为什么用克隆？** 标量无法走 tensor yield 链（slot 类型杂、数量多）。`ScalarClosure` 收集"组内 op 依赖的外部标量"（非 tensor、非 alloc、定义块在 block 或 parentBlock），克隆进组并重定向操作数，把跨组 SSA 收敛成只剩 tensor。

6. **else 分支的占位值为什么不破坏语义？** 分裂后 if 的 else 是"死分支"（条件为假时的真正语义由最后组 if 的 else 承担：Case B 吸收原 else 侧 yield 值、Scene 3/4 吸收另一侧 op）。纯占位分支 DCE 消除，保证正确。

7. **占位值类型分派**：tensor → 优先追溯真实源（函数参数/matmul/fill+empty，要求支配），否则 `tensor.empty`；标量 → `arith.constant 0`；memref → 带 memory_space 用 alloca、普通用 alloc/alloca + reinterpret_cast 恢复 layout；有地址空间时不用函数参数根。

8. **void if 是什么？** 非最后组、纯副作用、无产出值 → 生成 `hasElse=false` 的无结果 if，只保留 then 执行组内 op。

9. **分裂 else 侧为什么取反条件？** 链式 if 都复用原 if 的条件，但分裂的是 else region（组在 else 里）。对 else 侧分裂时把条件 `XOrI` 取反，等价于"原 else 体"作为 then 执行。

10. **如何保证失败安全？** 任意候选材质化失败 → `WalkResult::interrupt` → `walkMainLoop` 返回 failure → `fallback.restore()` 回退整个 pass，绝不产生半分裂 IR。

11. **rearrangeIfOp 的作用？** 材质化把链式 if 插在原 if 位置，可能插在外部依赖之前。按 SSA 操作数 + `memGraph.getExecBefore` 内存依赖找到 block 内最晚依赖，`moveAfter` 修正位置。

12. **ambient op 怎么处理？** 无 block_id（-1/16）的 op 打上当前活跃组标记并入组；最前面的 ambient op 挂 pending，遇到第一个真实组或结尾时刷入第一组——保证不产生孤立 op。

13. **与 MergeVectorIfBlock 的区别？** 方向相反：`MergeVectorIfBlock` 把纯 Vector 的 if **并入**上游块（消除 if 边界）；`SplitIfByBlockId` 把混合块 if **拆成**单 block_id 链（消除 if 内部混块）。一个收拢、一个拆分，服务同一目标：**让每个块/每个 if 与 block_id 一一对应**。
