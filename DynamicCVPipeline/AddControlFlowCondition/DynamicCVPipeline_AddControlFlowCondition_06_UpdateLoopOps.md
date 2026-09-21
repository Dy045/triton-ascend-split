# DynamicCVPipeline 主 Pass 专题：控制流条件注入 · 子专题 06 —— UpdateLoopOps（主循环扩展与核间同步）

> 覆盖 `AddControlFlowCondition` 的 **Step 4：UpdateLoopOpsPass**（独立 Pass）
>
> 源码文件：[`UpdateLoopOps.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopOps.cpp)
>
> 流水线位置：`AddControlFlowCondition`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L83）内第五个子 Pass（主编排 L99-L101）。

---

## 一、这个子 Pass 在做什么

`CreateIfOps` 造好了 if 骨架，`InitDependentMap` 造好了依赖图。但主循环本身**还缺两样东西**才能支撑真实控流：

1. **计数状态**：每个 block 需要一个"迭代计数"、每条核内依赖需要一个"依赖计数"、每个 tensor 跨迭代依赖需要一个"消费计数"——这些都必须成为主循环的 **iter_arg**（loop-carried），由 `UpdateConditionInfo` 在 if 条件里读写；
2. **核间同步**：CUBE 与 Vector 两条流水线主循环启动前必须对齐（握手），需要插入 **PIPE_S 同步**（`SyncBlockSet` / `SyncBlockWait`）。

`UpdateLoopOps` 的职责分四块：

1. **分析 tensor 迭代依赖**（`analyzeTensorIterArgDependencies`）：找 tensor 型 iter_arg 的 producer if / consumer if，记录 `info->tensorIterArgDepsMap`（每个 consumer 对应一个独立计数 iter_arg）；
2. **推导 block 计数**（`deriveBlockCountersFromIfOps`）：`blockCounterNums` 为空时从 `ssbuffer.if` 数量补推导；
3. **扩展主循环 iter_arg**（`addBlockCountersAndInnerDepConds`）：按三类 extra arg（block counters / inner dep conds / tensor iter args）重建 for/while，追加 iter_arg 与对应 init 值，并把新旧索引映射、依赖映射一并迁移到新 op；
4. **插入 PIPE_S 核间同步**（`insertInterCorePipeS`）：按 scope 类型（CUBE/Vector）与 `kVectorFirst` 标记，在主循环内外插入 `SyncBlockSet`/`SyncBlockWait`，flag id 固定为 15。

**一句话**：给主循环装上"计数器 iter_arg"和"核间握手"，让后续条件注入有状态可读、流水线启动有对齐可依。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `kPipeSFlagId = 15` | 流水线预留同步 flag id（固定 15），所有主循环的 PIPE_S SET/WAIT 共用同一 flag，全局对齐 |
| `blockCounterNums` | `info->blockCounterNums[op]`：主循环的 block 数（`CreateIfOps` 已填；本 Pass 兜底补推） |
| `blockCounters` | `info->blockCounters[op] = {iter_arg_idx...}`：每个 block 一个计数 iter_arg 的索引范围 |
| `innerDepConds` | `info->innerDepConds[op] = {iter_arg_idx...}`：每条核内依赖一个依赖计数 iter_arg 的索引范围 |
| `tensorIterArgDepsMap` | `info->tensorIterArgDepsMap[op] = {TensorIterArgIfOpRelation}`：`{iterArg, producerIf, [consumerIfs]}` |
| `tensorIterArgIndicesMap` | `info->tensorIterArgIndicesMap[op][iterArg] = {iter_arg_idx...}`：每 consumer 一个消费计数 iter_arg |
| `TensorIterArgIfOpRelation` | 一条 tensor 迭代依赖：一个 producer if + 多个 consumer if；yield 传递张量值，需按 consumer 单独计数 |
| `whileBlockArgMap` | while 专用：`whileOp -> block_id -> {new_arg_idx: old_arg_idx}`，由 `ProcessArgs` 产出、`UpdateConditionInfo` 消费；本 Pass 迁移到新 whileOp |
| `SyncBlockSetOp` / `SyncBlockWaitOp` | hivm 核间同步原语：SET（置位）在生产者侧、WAIT（等待）在消费者侧 |
| `PIPE_S`（`PipeAttr`） | 同步流水线类型，SET/WAIT 都作用于 PIPE_S |
| `kVectorFirst` | 标记：Vector 流水线先启动（反转握手方向），无该标记默认 cube_first |
| `getMainLoopBody` | 统一取主循环 body 块：for 取单 region，while 取 **after region**（before 区不承载循环体） |
| `buildNewYieldOp` / `migrateBody` | 工具函数：迁移 body 并重建 yield（追加 extra 参数） |
| `createNewForOpWithExtras` / `createNewWhileOpWithExtras` | 工具函数：创建带 extra inits 的新 loop（Utils.h） |

---

## 三、逐行讲解

### 3.1 入口：runOnOperation（UpdateLoopOps.cpp L678-L728）

```cpp
void UpdateLoopOpsPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  // info 可能为空（独立运行时），此时用本地 localInfo
  ControlFlowConditionInfo localInfo;
  ControlFlowConditionInfo *infoToUse = info ? info : &localInfo;

  // Step 1: 分析 tensor 型 iter_arg 与 ssbuffer.if 的依赖关系
  if (failed(analyzeTensorIterArgDependencies(module, infoToUse))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 2: blockCounterNums 为空时从 ssbuffer.if 数量推导
  if (infoToUse->blockCounterNums.empty()) {
    if (failed(deriveBlockCountersFromIfOps(module, infoToUse))) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
  }

  // Step 3: 扩展 for/while iter_arg（block counters + inner dep conds + tensor iter args）
  if (failed(addBlockCountersAndInnerDepConds(module, infoToUse))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 4: 插入 PIPE_S 核间同步
  if (failed(insertInterCorePipeS(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // 验证 whileBlockArgMap 在替换后仍存活（迁移检查）
  dumpWhileBlockArgMap(infoToUse->whileBlockArgMap, "...");
}
```

**要点**：四步严格有序。`whileBlockArgMap` 的 dump 放在最后，专门验证 `replaceMainLoopOpUsesAndErase` 迁移映射没有丢。

### 3.2 tensor 迭代依赖分析：analyzeTensorIterArgDependencies（L568-L676）

```cpp
LogicalResult UpdateLoopOpsPass::analyzeTensorIterArgDependencies(
    ModuleOp module, ControlFlowConditionInfo *info) {
  bool failed = false;
  module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
      return WalkResult::advance();
    }

    // iter_args：for 取 region iter args（剔除 IV）；while 取 after 区参数
    llvm::SmallVector<Value> iterArgsVec = MainLoop(op).getIterArgs();

    for (auto iterArg : iterArgsVec) {
      if (!mlir::isa<TensorType>(iterArg.getType())) {
        continue;  // 只分析 tensor 型
      }

      scf::IfOp producerIfOp = nullptr;
      llvm::SmallVector<scf::IfOp> consumerIfOps;

      for (auto &use : iterArg.getUses()) {
        Operation *user = use.getOwner();
        scf::IfOp ifOp = nullptr;
        // 沿 parent 链找最近的带 kIf 的 scf.if（不含主循环本身）
        Operation *curr = user;
        while (curr && curr != op) {
          if (auto currIf = dyn_cast<scf::IfOp>(curr)) {
            if (currIf->hasAttr(CVPipeline::kIf)) {
              ifOp = currIf;
              break;
            }
          }
          curr = curr->getParentOp();
        }
        if (!ifOp) {
          continue;  // 不在任何 ssbuffer.if 内
        }

        // producer 判定：仅当 use 是"该 if 的直接 terminator"
        // （嵌套 if/for/while 内转发 iter_arg 的 yield 都是 consumer）
        bool isProducer = isa<scf::YieldOp>(user) &&
                          user->getParentOp() == ifOp.getOperation();

        if (isProducer) {
          if (producerIfOp && producerIfOp != ifOp) {
            // 多个不同 producer → 错误
            failed = true;
            return WalkResult::interrupt();
          }
          if (!producerIfOp) {
            // 该 ifOp 之前是 consumer 则升级为 producer
            auto it = llvm::find(consumerIfOps, ifOp);
            if (it != consumerIfOps.end()) {
              consumerIfOps.erase(it);
            }
            producerIfOp = ifOp;
          }
        } else {
          if (producerIfOp == ifOp) {
            continue;  // 已是 producer 的 if 内 consumer use，忽略
          }
          if (!llvm::is_contained(consumerIfOps, ifOp)) {
            consumerIfOps.push_back(ifOp);
          }
        }
      }

      // 必须同时有 producer 和 consumer，否则跳过
      if (!producerIfOp || consumerIfOps.empty()) {
        continue;
      }
      TensorIterArgIfOpRelation relation;
      relation.iterArg = iterArg;
      relation.producer = producerIfOp;
      relation.consumers = consumerIfOps;
      info->tensorIterArgDepsMap[op].push_back(relation);
    }
    return WalkResult::advance();
  });
  return failed ? failure() : success();
}
```

**要点**：
- **producer 的唯一判定**：`iterArg` 的 use 是该 if 的**直接 `scf.yield` terminator**（`user->getParentOp() == ifOp`）才是 producer——即该 if 的 then 区把 iter_arg 作为结果 yield 出去；凡嵌套结构（if 内再套 if/for/while）转发 iter_arg 的 yield，父 op 不是 ifOp 本身，一律算 consumer；
- **一个 iter_arg 只能有一个 producer**（多个不同 producer = 错误）；consumer 可多个，且 consumer 与 producer 是同一个 if 时以 producer 为准；
- 只有 producer+consumer **同时存在**才记录依赖（单边无依赖意义）。

### 3.3 兜底推导：deriveBlockCountersFromIfOps（L51-L71）

```cpp
LogicalResult UpdateLoopOpsPass::deriveBlockCountersFromIfOps(
    ModuleOp module, ControlFlowConditionInfo *info) {
  module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
      return WalkResult::advance();
    }
    int numIfBlockIds = countUniqueIfBlockIds(op);
    if (numIfBlockIds > 0) {
      info->blockCounterNums[op] = numIfBlockIds;
    }
    return WalkResult::advance();
  });
  return success();
}
```

**要点**：`CreateIfOps` 已按 blockOps.size() 填过 `blockCounterNums`；这里仅在 map 为空（如独立运行本 Pass）时兜底，`countUniqueIfBlockIds` 数去重后的 `ssbuffer.if` 标记数。

### 3.4 扩展规划：computeMainLoopExtraArgs（L207-L224）

```cpp
static void computeMainLoopExtraArgs(
    Operation *oldLoopOp, ControlFlowConditionInfo *info, int &numBlockCounters,
    int &numInnerDepConds, int &numTensorIterArgs,
    llvm::SmallVector<TensorIterArgIfOpRelation> &depsVecCopy) {
  numBlockCounters = info->blockCounterNums.lookup(oldLoopOp);
  numInnerDepConds = info->intraCoreDependentMap.count(oldLoopOp)
                         ? (int)info->intraCoreDependentMap[oldLoopOp].size()
                         : 0;

  numTensorIterArgs = 0;
  auto tensorIterArgDepsIt = info->tensorIterArgDepsMap.find(oldLoopOp);
  if (tensorIterArgDepsIt != info->tensorIterArgDepsMap.end()) {
    depsVecCopy = tensorIterArgDepsIt->second;  // 拷贝，稍后 move 回 info
    for (auto &entry : depsVecCopy) {
      numTensorIterArgs += entry.consumers.size();  // 每 consumer 一个计数
    }
  }
}
```

**要点**：三类 extra arg 的数量——block counters = 块数；inner dep conds = 该主循环核内依赖映射的 consumer 条数（`intraCoreDependentMap[op].size()`）；tensor iter args = **所有依赖的 consumer 数之和**（每个 consumer 一个独立计数，这正是"yield 传递张量、消费需逐消费者计数"的体现）。

### 3.5 extra init 值：buildMainLoopExtraInitArgs（L238-L258）

```cpp
static void buildMainLoopExtraInitArgs(
    OpBuilder &builder, Location loc, Value forOpLowerBound, bool isWhile,
    int numBlockCounters, int numInnerDepConds, int numTensorIterArgs,
    llvm::SmallVector<Value> &extraInitArgs) {
  // 1. block counters：for 用 lowerBound（复用同一 Value）；while 每个 counter
  //    独立新建 constant i32(0)（SSA 值必须不同，下游测试依赖）
  for (int i = 0; i < numBlockCounters; ++i) {
    if (isWhile) {
      extraInitArgs.push_back(builder.create<arith::ConstantOp>(
          loc, builder.getI32Type(), builder.getI32IntegerAttr(0)));
    } else {
      extraInitArgs.push_back(forOpLowerBound);
    }
  }
  // 2. inner dep conds：全部 i32(0)
  for (int i = 0; i < numInnerDepConds; ++i) {
    extraInitArgs.push_back(builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(0)));
  }
  // 3. tensor iter args：全部 i32(1)
  for (int i = 0; i < numTensorIterArgs; ++i) {
    extraInitArgs.push_back(builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(1)));
  }
}
```

**要点**：三类初值语义各异——**block counter**：for 从 lowerBound 起计（循环每轮 +1 即块迭代序号），while 无 IV 只能独立 constant 0；**inner dep cond**：从 0 起，被依赖消费时 +1；**tensor iter arg**：从 1 起（跨迭代依赖，首轮即有前置缓冲）。

### 3.6 重建 for：createForOpAndMigrateBody（L75-L100）

```cpp
static scf::ForOp
createForOpAndMigrateBody(scf::ForOp oldForOp,
                          const llvm::SmallVector<Value> &extraInitArgs) {
  scf::ForOp newForOp = createNewForOpWithExtras(oldForOp, extraInitArgs);
  if (newForOp == oldForOp) {
    return oldForOp;  // 无 extra，原样返回
  }

  Block *oldBlock = oldForOp.getBody();
  Block *newBlock = newForOp.getBody();
  migrateBody(oldBlock, newBlock);

  // 追加 extra iter_arg 的 yield（新 block 参数在 [1+oldNumArgs, +numExtra)）
  unsigned oldNumArgs = oldForOp.getNumRegionIterArgs();
  llvm::SmallVector<Value> extras;
  for (size_t i = 0; i < extraInitArgs.size(); ++i) {
    extras.push_back(newBlock->getArgument(1 + oldNumArgs + i));
  }
  if (failed(buildNewYieldOp(oldBlock, newBlock, newForOp, extras))) {
    newForOp.erase();
    return scf::ForOp();
  }
  return newForOp;
}
```

**要点**：`1 + oldNumArgs + i` 的 `1` 是 for 的 IV（归纳变量）占位；extra 新 body 参数从 `1+oldNumArgs` 起。新 iter_arg 在 yield 里**原样传递给自己**（counter 前向自增由 `UpdateConditionInfo` 在 if 内做加减，这里 yield 传新 block 参数即可）。

### 3.7 重建 while：createWhileOpAndMigrateBody + migrateWhileRegions（L150-L190）

```cpp
static LogicalResult migrateWhileRegions(scf::WhileOp oldWhileOp,
                                         scf::WhileOp newWhileOp,
                                         unsigned numOriginalIterArgs,
                                         unsigned numExtraArgs) {
  // before 区：迁移 body + 移动 condition terminator + 扩展 condition 转发值
  Block *oldBefore = oldWhileOp.getBeforeBody();
  Block *newBefore = newWhileOp.getBeforeBody();
  migrateBody(oldBefore, newBefore);
  auto oldCond = cast<scf::ConditionOp>(oldBefore->getTerminator());
  oldCond->moveBefore(newBefore, newBefore->end());
  extendWhileCondition(oldCond, newBefore, numOriginalIterArgs, numExtraArgs);

  // after 区：迁移 body + 移动 yield terminator + 扩展 yield
  Block *oldAfter = oldWhileOp.getAfterBody();
  Block *newAfter = newWhileOp.getAfterBody();
  migrateBody(oldAfter, newAfter);
  auto oldYield = cast<scf::YieldOp>(oldAfter->getTerminator());
  oldYield->moveBefore(newAfter, newAfter->end());
  extendWhileYield(oldYield, newAfter, numOriginalIterArgs, numExtraArgs);
  return success();
}
```

```cpp
static void extendWhileCondition(scf::ConditionOp oldCond, Block *newBefore,
                                 unsigned numOriginalIterArgs,
                                 unsigned numExtraArgs) {
  // 新 before 参数在 [numOriginal, numOriginal+numExtra)，追加进 condition 转发
  for (unsigned i = 0; i < numExtraArgs; ++i) {
    newCondValues.push_back(newBefore->getArgument(numOriginalIterArgs + i));
  }
  oldCond.getArgsMutable().assign(newCondValues);
}
```

**要点**：while 双区结构——before 区的 `scf.condition` 与 after 区的 `scf.yield` 都要把新 iter_arg **透传**（condition 从 before 参数转发到 after，yield 从 after 参数转发回 before，形成循环）。while 没有 IV，`numOriginalIterArgs = getInits().size()`。

### 3.8 主扩展入口：extendMainLoopOpWithExtraArgs（L330-L378）

```cpp
static LogicalResult
extendMainLoopOpWithExtraArgs(Operation *oldLoopOp,
                              ControlFlowConditionInfo *info) {
  int numBlockCounters, numInnerDepConds, numTensorIterArgs;
  llvm::SmallVector<TensorIterArgIfOpRelation> depsVecCopy;
  computeMainLoopExtraArgs(oldLoopOp, info, numBlockCounters, numInnerDepConds,
                           numTensorIterArgs, depsVecCopy);

  int totalExtraArgs = numBlockCounters + numInnerDepConds + numTensorIterArgs;
  if (totalExtraArgs == 0) {
    return success();  // 无 extra 则不动
  }

  // 取 lowerBound（for）或标记 isWhile
  OpBuilder builder(oldLoopOp);
  Value forOpLowerBound;
  bool isWhile = false;
  if (auto forOp = dyn_cast<scf::ForOp>(oldLoopOp)) {
    forOpLowerBound = forOp.getLowerBound();
  } else if (isa<scf::WhileOp>(oldLoopOp)) {
    isWhile = true;
  } else {
    return failure();
  }

  llvm::SmallVector<Value> extraInitArgs;
  buildMainLoopExtraInitArgs(builder, oldLoopOp->getLoc(), forOpLowerBound,
                             isWhile, numBlockCounters, numInnerDepConds,
                             numTensorIterArgs, extraInitArgs);

  // 重建 loop 并迁移 body
  Operation *newOp = createMainLoopOpAndMigrateBody(oldLoopOp, extraInitArgs);
  if (!newOp) {
    return failure();
  }

  // 迁移三类索引范围 + 依赖映射到新 op
  unsigned baseIdx = getMainLoopBaseIdx(oldLoopOp, isWhile);
  recordMainLoopBlockCountersAndConds(oldLoopOp, newOp, info, baseIdx,
                                      numBlockCounters, numInnerDepConds);
  recordMainLoopTensorIterArgs(oldLoopOp, newOp, info, baseIdx,
                               numBlockCounters, numInnerDepConds,
                               numTensorIterArgs, depsVecCopy);
  transferMainLoopInfoMaps(oldLoopOp, newOp, info, isWhile);

  return replaceMainLoopOpUsesAndErase(oldLoopOp, newOp);
}
```

**要点**：整体编排——算数量 → 造 init → 重建+迁移 → **迁移所有索引/依赖映射** → 替换删除。关键在映射迁移：`blockCounters`/`innerDepConds` 记录的是新 op 上各 iter_arg 的索引（按 baseIdx 顺序排布：先 block counters、再 inner dep conds、最后 tensor iter args），`tensorIterArgDepsMap`/`intraCoreDependentMap`/`whileBlockArgMap` 全部从旧 op 转移到新 op，否则旧 op 删除后 `UpdateConditionInfo` 查不到。

### 3.9 索引迁移细节：recordMainLoopBlockCountersAndConds / recordMainLoopTensorIterArgs（L261-L306）

```cpp
static void recordMainLoopBlockCountersAndConds(
    Operation *oldLoopOp, Operation *newOp, ControlFlowConditionInfo *info,
    unsigned baseIdx, int numBlockCounters, int numInnerDepConds) {
  if (numBlockCounters > 0) {
    llvm::SmallVector<int> indices;
    for (int j = 0; j < numBlockCounters; ++j)
      indices.push_back(baseIdx + j);
    info->blockCounters.erase(oldLoopOp);
    info->blockCounters[newOp] = indices;      // 新 op 上 block counter 的 iter_arg 索引
  }
  if (numInnerDepConds > 0) {
    llvm::SmallVector<int> indices;
    for (int j = 0; j < numInnerDepConds; ++j)
      indices.push_back(baseIdx + numBlockCounters + j);
    info->innerDepConds.erase(oldLoopOp);
    info->innerDepConds[newOp] = indices;      // 紧随 block counters 之后
  }
}
```

```cpp
static void recordMainLoopTensorIterArgs(
    Operation *oldLoopOp, Operation *newOp, ControlFlowConditionInfo *info,
    unsigned baseIdx, int numBlockCounters, int numInnerDepConds,
    int numTensorIterArgs,
    llvm::SmallVector<TensorIterArgIfOpRelation> &depsVecCopy) {
  if (numTensorIterArgs == 0) {
    return;
  }
  unsigned tensorBaseIdx = baseIdx + numBlockCounters + numInnerDepConds;
  auto &newIndicesMap = info->tensorIterArgIndicesMap[newOp];
  unsigned currentIdx = tensorBaseIdx;
  for (auto &entry : depsVecCopy) {
    llvm::SmallVector<int> indices;
    for (int j = 0; j < (int)entry.consumers.size(); ++j) {
      indices.push_back(currentIdx++);          // 每 consumer 顺序占一个索引
    }
    newIndicesMap[entry.iterArg] = indices;
  }
  info->tensorIterArgIndicesMap.erase(oldLoopOp);
  info->tensorIterArgDepsMap[newOp] = std::move(depsVecCopy);
  info->tensorIterArgDepsMap.erase(oldLoopOp);
}
```

**要点**：新 iter_arg 的布局固定为 **[block counters][inner dep conds][tensor iter args]**。`tensorIterArgIndicesMap[newOp][iterArg]` 记录每个 tensor iter_arg 对应的一串 consumer 计数索引，供 `UpdateConditionInfo` 按 consumer 精确加减。

### 3.10 PIPE_S 同步插入：insertInterCorePipeS（L523-L563）

```cpp
LogicalResult UpdateLoopOpsPass::insertInterCorePipeS(ModuleOp module) {
  auto cubeCoreType = hivm::TCoreTypeAttr::get(...CUBE);
  auto vectorCoreType = hivm::TCoreTypeAttr::get(...VECTOR);
  auto setPipeType = PipeAttr::get(...PIPE_S);
  auto waitPipeType = PipeAttr::get(...PIPE_S);

  WalkResult result = module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    auto scopeTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!scopeTypeAttr) {
      return WalkResult::advance();
    }
    bool isScopeCube = (scopeTypeAttr == cubeCoreType);
    bool isScopeVector = (scopeTypeAttr == vectorCoreType);

    WalkResult innerResult = scopeOp.walk([&](Operation *op) -> WalkResult {
      if (!isMainLoopOp(op)) {
        return WalkResult::advance();
      }
      if (failed(insertPipeSForMainLoopOp(op, scopeOp, isScopeCube,
                                          isScopeVector, setPipeType,
                                          waitPipeType, kPipeSFlagId))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (innerResult.wasInterrupted()) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}
```

**要点**：按 `scope::ScopeOp` 的 `hivm.tcore_type` 判定当前核是 CUBE 还是 Vector；每个 scope 内所有主循环（for/while）都插 PIPE_S，flag id 统一 15。

### 3.11 四象限同步布局：insertPipeSForMainLoopOp（L462-L521）

```cpp
static LogicalResult
insertPipeSForMainLoopOp(Operation *loopOp, scope::ScopeOp scopeOp,
                         bool isScopeCube, bool isScopeVector, PipeAttr setPipe,
                         PipeAttr waitPipe, int flagId) {
  Block *loopBody = getMainLoopBody(loopOp);   // while 取 after 区
  bool isVectorFirst = loopOp->hasAttr(CVPipeline::kVectorFirst);

  if (isVectorFirst) {
    if (isScopeCube) {
      // vector_first + CUBE：循环前 SET + 循环内 WAIT/SET
      insertSyncOpsOutsideMainLoop(loopOp, ..., /*isBefore=*/true);
      insertSyncOpsInsideMainLoop(loopBody, ...);
    } else if (isScopeVector) {
      // vector_first + VECTOR：循环内 WAIT/SET + 循环后 WAIT
      insertSyncOpsInsideMainLoop(loopBody, ...);
      insertSyncOpsOutsideMainLoop(loopOp, ..., /*isBefore=*/false);
    }
  } else {
    // cube_first（含无 kVectorFirst 标记的默认情况）
    if (isScopeCube) {
      // cube_first + CUBE：循环内 WAIT/SET + 循环后 WAIT
      insertSyncOpsInsideMainLoop(loopBody, ...);
      insertSyncOpsOutsideMainLoop(loopOp, ..., /*isBefore=*/false);
    } else if (isScopeVector) {
      // cube_first + VECTOR：循环前 SET + 循环内 WAIT/SET
      insertSyncOpsOutsideMainLoop(loopOp, ..., /*isBefore=*/true);
      insertSyncOpsInsideMainLoop(loopBody, ...);
    }
  }
  return success();
}
```

```cpp
// 循环内：body 开头 WAIT（等对方上一轮 SET），yield 前 SET（通知对方）
static LogicalResult insertSyncOpsInsideMainLoop(...) {
  OpBuilder insertionBuilder(&loopBody->front());
  insertionBuilder.create<SyncBlockWaitOp>(loc, coreType, setPipe, waitPipe, waitFlagAttr);
  OpBuilder setBuilder(forTerminator);
  setBuilder.setInsertionPoint(forTerminator);
  setBuilder.create<SyncBlockSetOp>(loc, coreType, setPipe, waitPipe, setFlagAttr);
  return success();
}
```

**要点**：这是**四象限握手矩阵**，核心规律——**先启动方（first）在循环外"预告"同步，后启动方在循环外"收尾"确认**：
- 循环**内**统一插 `WAIT`（body 开头）+ `SET`（yield 前）：每轮迭代首尾各一次，构成逐轮握手；
- 循环**外**：`vector_first` 时 CUBE 在循环前 `SET`（预告"我要开了"）、Vector 在循环后 `WAIT`（收尾确认）；`cube_first` 则镜像——Vector 循环前 `SET`、CUBE 循环后 `WAIT`。这样两条流水线在启动前/结束后都能对齐。

### 3.12 同步原语：insertSyncOpsOutsideMainLoop（L436-L449）

```cpp
static LogicalResult insertSyncOpsOutsideMainLoop(
    Operation *loopOp, Location loc, hivm::TCoreTypeAttr coreType, PipeAttr setPipe,
    PipeAttr waitPipe, int flagId, bool isBefore) {
  OpBuilder builder(loopOp);
  if (isBefore) {
    builder.create<SyncBlockSetOp>(loc, coreType, setPipe, waitPipe, flagAttr);
  } else {
    builder.setInsertionPointAfter(loopOp);
    builder.create<SyncBlockWaitOp>(loc, coreType, setPipe, waitPipe, flagAttr);
  }
  return success();
}
```

**要点**：`isBefore=true` 在 loop 前插 SET、`isBefore=false` 在 loop 后插 WAIT；同一个 `flagId=15` 保证 SET 与 WAIT 一一配对。

---

## 四、流程总结

### 4.1 UpdateLoopOps 的流水

```
UpdateLoopOpsPass::runOnOperation
  ├─► Step 1: analyzeTensorIterArgDependencies
  │     └─► 逐主循环、逐 tensor iter_arg：
  │           ├─► 沿 parent 链找带 kIf 的 if
  │           ├─► 直接 terminator = producer；嵌套 yield = consumer
  │           ├─► 单 producer + 多 consumer → tensorIterArgDepsMap
  │           └─► 仅 producer 或仅 consumer → 跳过
  ├─► Step 2: deriveBlockCountersFromIfOps（blockCounterNums 为空才执行）
  ├─► Step 3: addBlockCountersAndInnerDepConds
  │     └─► 逐 main_loop op（info-driven）：
  │           ├─► computeMainLoopExtraArgs        三类 extra 数量
  │           ├─► buildMainLoopExtraInitArgs      初值：counter(lb/0) + dep(0) + tensor(1)
  │           ├─► createMainLoopOpAndMigrateBody  for：migrate body + yield 透传
  │           │                                    while：before/after 双区 + cond/yield 透传
  │           ├─► record 三类索引（布局 [counters][deps][tensors]）
  │           ├─► transferMainLoopInfoMaps        迁移 intraCoreDependentMap/whileBlockArgMap
  │           └─► replaceMainLoopOpUsesAndErase   类型校验 + 重定向 + 删旧
  └─► Step 4: insertInterCorePipeS
        └─► 逐 scope（按 tcore_type）：
              └─► 逐主循环 insertPipeSForMainLoopOp（flag=15）
                    ├─► vector_first+CUBE：前置 SET + 内 WAIT/SET
                    ├─► vector_first+VECTOR：内 WAIT/SET + 后置 WAIT
                    ├─► cube_first+CUBE：内 WAIT/SET + 后置 WAIT
                    └─► cube_first+VECTOR：前置 SET + 内 WAIT/SET
```

### 4.2 与前后子 Pass 的协作

- **前置**：`CreateIfOps` 填的 `blockCounterNums`；`InitDependentMap` 填的 `intraCoreDependentMap`；`ProcessArgs` 填的 `whileBlockArgMap`；
- **产出**：带三类 extra iter_arg 的新主循环；`blockCounters`/`innerDepConds`/`tensorIterArgIndicesMap`/`tensorIterArgDepsMap` 全部迁到新 op；主循环内外 PIPE_S 同步；
- **后置**：`UpdateConditionInfo` 读取上述所有映射，给每个 if 构造真实条件（读计数器、更新依赖计数）；`UpdateLoopIterTimes` 依赖新 iter_arg 布局计算缓冲因子并扩展迭代次数。

---

## 五、面试要点

1. **为什么主循环要新增三类 iter_arg？** block counter 让每个 block 知道"自己第几轮"；inner dep cond 跟踪每条核内依赖的"消费进度"（该缓冲是否已被消费可复用）；tensor iter arg 跟踪跨迭代张量依赖的消费进度（一个 producer 多 consumer，每个 consumer 独立计数）。三者分别是 if 条件成立的**轮次/复用/消费**依据。

2. **for 与 while 的 block counter 初值为何不同？** for 有 IV，直接用 `lowerBound` 做初值（每轮 IV+1，block 计数自然递增）；while 无 IV，只能每个 counter 新建独立 `constant i32(0)`。注释明确强调 while 必须**独立 SSA 值**——下游 Pass（测试/条件构造）依赖它们互不相同，复用同一 constant 会串扰。

3. **tensor iter_arg 的 producer 怎么判定？** 只有 use 是**该 if 的直接 `scf.yield` terminator**（`user->getParentOp() == ifOp`）才计为 producer——即该 if 把张量作为结果 yield 出 if 边界；嵌套 if/for/while 内转发张量的 yield 都是 consumer。一个 tensor iter_arg 有且只能有一个 producer if，多个不同 producer 直接报错。

4. **为什么 tensor 依赖按 consumer 数分配 iter_arg，而不是一个？** 张量经 yield 在 if 间传递，**每个消费者各自需要知道"producer 第几轮产出的缓冲可以读"**，消费进度彼此独立（不同 consumer 的循环位置不同），必须逐 consumer 计数才能精确判定同步条件。

5. **PIPE_S 的四象限布局逻辑是什么？** 核心是"先启动方预告、后启动方收尾"：循环内统一 WAIT（body 开头）+SET（yield 前）做逐轮握手；循环外，先启动方（有 `kVectorFirst` 的 Vector，或无标记的 CUBE）在自己的循环**前**插 SET（预告），对方在自己的循环**后**插 WAIT（确认），保证两条核间流水线启动前对齐、结束后合拢。

6. **为什么 flag id 固定 15（`kPipeSFlagId`）？** 这是流水线预留的全局同步 flag，所有主循环的 PIPE_S SET/WAIT 共用，保证不同主循环之间也能通过同一 flag 跨核互认；固定值避免与其他同步 flag 冲突。

7. **为什么重建 loop 后必须迁移所有映射？** `replaceMainLoopOpUsesAndErase` 会删除旧 op，而 `info` 里所有以 op 为键的映射（`blockCounters`、`innerDepConds`、`tensorIterArgDepsMap`/`IndicesMap`、`intraCoreDependentMap`、`whileBlockArgMap`）在旧 op 上就全失效了。必须在新 op 上重建索引（`recordMainLoop*`）并转移映射（`transferMainLoopInfoMaps`），`UpdateConditionInfo` 才能按新 op 查询——`dumpWhileBlockArgMap` 就是用来验证迁移没丢的。
