# DynamicCVPipeline 主 Pass 专题：控制流条件注入 · 子专题 07 —— UpdateConditionInfo（真实条件构造）

> 覆盖 `AddControlFlowCondition` 的 **Step 5：UpdateConditionInfoPass**（独立 Pass）
>
> 源码文件：[`UpdateConditionInfo.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.cpp)
>
> 流水线位置：`AddControlFlowCondition`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L83）内第六个子 Pass（主编排 L128-L130）。

---

## 一、这个子 Pass 在做什么

前面 4 个子 Pass 完成了：op 去共享、iter_arg 拆分、if 骨架（`scf.if %true`）、依赖图、主循环扩展（计数器 iter_arg + PIPE_S）。但**每个 if 的条件仍然是写死的 `true`**——producer 不会因缓冲满而等待、consumer 不会因数据未到而阻塞，流水线随时踩踏。

`UpdateConditionInfo` 是整套控流的"执行核心"：**把每个 if 的占位条件替换成真正的同步表达式**。

1. **分配核间 SSBuffer**（`allocSSBuffer`）：按依赖组数为 Vector 0 / Vector 1 各分配一组共享内存计数指针（地址 0 起 / 1024 起，每槽 4 字节），并置零；
2. **分析每个 if 的输入/输出依赖组**（`getInputOutputValues`）：遍历 if 内 op，找出它消费了哪些 buffer 组（输入）、生产了哪些（输出）；
3. **构造核间条件**（`setCrossCoreCondition`）：输入组计数 `> 0`（有货可读）、输出组计数 `< producer_num`（有坑可写）；并在 then 分支**更新计数**（输入 -1、输出 +1），volatile load/store 保证跨核可见；
4. **构造核内条件**（`setIntraCoreCondition`）：用 loop iter_arg 计数——输入 `> 0`（DEC）、输出 `< producer_num`（INC），tensor 迭代依赖用 `==1`（输入）/`==0`（输出）判据；
5. **flowOpt 条件**（`setFlowOptCondition`）：DAG 中深度 3 的目标 if 额外放宽：`counter >= upperBound OR counter >= lowerBound + step*opt_num`（重叠优化放宽同步）；
6. **合并并重建 if**（`combineConditions`）：核间 ∧ 核内 ∧ flowOpt ∧ 计数器条件（for：`counter < upperBound`；while：重映射的 `scf.condition`），重建带额外结果（控制变量最新值）的新 if，替换旧 if，并把 DAG/依赖映射/计数映射迁移到新 if；
7. **更新循环 yield**（`updateLoopYield`）：把控制变量的最新值写回主循环 yield，闭环整个状态流。

**一句话**：if 从"恒真占位"变成"缓冲可读可写 + 轮次合法 + 迭代继续"的同步闸门，多缓冲流水线真正转起来。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `VECTOR_SSBUF_OFFSET=1024` / `VALUE_SSBUF_OFFSET=4` | Vector 1 的 SSBuffer 基址偏移 1024；每个计数槽间隔 4 字节（i32） |
| `allocSSBuffer` | 每个 `scope::ScopeOp` 前分配两组 memref 指针（Vector 0 地址 `i*4`、Vector 1 地址 `1024+i*4`），并 store 0 初始化 |
| `crossCoreBuffers` / `intraCoreBuffers` | `{groupIdx -> {consumer -> [producers]}}`：`collectDependencyBuffers` 从依赖映射转成按 group 索引的结构（walk 顺序确定性编号） |
| `getInputOutputValues` | 遍历 if 内 op，用 consumer/producer 反查表（`consumerToGroups`/`outputToGroups`）收集该 if 的输入/输出依赖组索引 |
| `isAIC` / `isAIV` | if 位于 CUBE scope（AIC）/ Vector scope（AIV），决定 SSBuffer 指针选择与双核读写策略 |
| `computeVectorSSBufferMemrefs` | Vector 侧指针 = 基址 `groupIdx*4` + `GetSubBlockIdx * 1024`（子块索引），一条 memref 复用双核 |
| `createSsbufLoads` | volatile `memref.load`：CUBE 读 1 个指针；Vector 读 2 个（`ptrSetIdx` 0/1），打 `kMemrefExtVolatile` 标记保证跨核可见 |
| `addCrossCoreConditions` | 输入组 `load > 0`、输出组 `load < producer_num`，多组 And 合并 |
| `updateCrossCoreControlVars` | then 分支 yield 前：输入组 `load-1` 写回、输出组 `load+1` 写回（volatile） |
| `idxToVar` | `buildIdxToVarMap`：核内依赖组索引 -> loop iter_arg（控制变量），按 `innerDepConds[loopOp]` 顺序分配 |
| `collectIntraCoreInputConditions` | 核内输入：`var > 0`，标记 `DEC`（消费后 -1） |
| `collectIntraCoreOutputConditions` | 核内输出：`var < producer_limit`（该输出组 producer 数），标记 `INC`（生产后 +1） |
| `buildOutputGroups` | 按**输出 op 集合**聚合输出组：相同 producer 集合共享同一限制，合并控制变量到 `inputVars` |
| `tensorIterArgIfOpVars` | `ifOp -> {producerVars, consumerVars}`：tensor 迭代依赖的 producer if / consumer if 各自关联的计数变量 |
| `collectTensorIterArgInputConditions` | tensor 消费者：`var == 1`（本轮可消费），DEC |
| `collectTensorIterArgOutputConditions` | tensor 生产者：`var == 0`（本轮可生产），INC |
| `setFlowOptCondition` | 目标 if（DAG 深度 3）追加 `counter>=upperBound OR counter>=lowerBound+step*opt_num`，`opt_num = min(intraCoreBufferCount-1, crossCoreBufferCount)` |
| `controlVarToLatestValue` | `var -> 最新值`：控制变量在本次迭代可能被多次更新（多个 if 先后消费/生产），后续引用必须用最新值而非 iter_arg 原值 |
| `combineConditions` | 合并全部条件 + for 计数器条件 `counter < upperBound`（while 用重映射 condition），And 后重建 if |
| `createNewIfOpWithBlocks` | 重建 if：原结果 + 控制变量最新值 + 计数器新值作为新结果类型；then/else 分支各自 yield |
| `populateNewThenBlock` | then 分支：移动旧 op + yield 追加（DEC/INC 更新后的值、计数器 `+step`） |
| `populateNewElseBlock` | else 分支：yield 透传控制变量/计数器的**最新值**（不执行更新，状态保持） |
| `updateControlVarToLatestValue` | 记录本次 if 迭代后各控制变量的最新值（新 if 的扩展结果），供后续 if / yield 使用 |
| `updateLoopYield` | 主循环 yield 中控制变量位置替换为最新值（for 按 region iter arg 索引，while 按 after arg 索引） |
| `updateDAGAfterIfOpReplacement` | if 被替换后，`ifBlockDAG` / `flowOptIfOpPairs` 中的旧 if 指针全部换成新 if |
| `buildWhileCounterCondition` | while 专用：把 before 区 `scf.condition(x)` 的 x 定义链按 `whileBlockArgMap` 重映射到 after 区新 arg，克隆到当前 if 位置 |
| `cloneConditionDefChain` | 只克隆 before 区中产生 x 的 def 链，before 区 arg 通过 mapping 换成 after 值，外部值/常量原样复用 |

---

## 三、逐行讲解

### 3.1 入口：runOnOperation（UpdateConditionInfo.cpp L1919-L1940）

```cpp
void UpdateConditionInfoPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  // Step1: 分配核间 SSBuffer 指针并初始化
  SmallVector<SmallVector<Value>> ssbufferPtrs = allocSSBuffer(module);

  // Step2: 更新所有 if 的条件
  int updateResult = updateIfConds(module, ssbufferPtrs);
  if (updateResult != UPDATE_CONDITION_INFO_SUCCESS) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }
}
```

**要点**：两段式——先分配共享计数缓冲，再逐个循环、逐个 if 构造条件。所有失败统一 `ERRCODE_FAILED`。

### 3.2 分配 SSBuffer：allocSSBuffer（L242-L288）

```cpp
SmallVector<SmallVector<Value>>
UpdateConditionInfoPass::allocSSBuffer(ModuleOp module) {
  OpBuilder builder(module.getContext());
  auto i32Type = builder.getIntegerType(CONST_INT_TYPE);

  int numBuffers = countProducerGroups(info->crossCoreDependentMap);
  if (numBuffers == 0) {
    return ssbufferMemrefs;  // 无跨核依赖 → 空
  }

  module->walk([&](Operation *op) {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
      builder.setInsertionPoint(scopeOp);
      auto zeroConst = builder.create<arith::ConstantOp>(...);
      for (int i = 0; i < numBuffers; i++) {
        auto memref0 = createPointerCastOp(builder, loc, i * VALUE_SSBUF_OFFSET);
        auto memref1 = createPointerCastOp(
            builder, loc, VECTOR_SSBUF_OFFSET + i * VALUE_SSBUF_OFFSET);
        builder.create<memref::StoreOp>(..., zeroConst, memref0);
        builder.create<memref::StoreOp>(..., zeroConst, memref1);
        ssbufferVec0Memrefs.push_back(memref0.getResult());
        ssbufferVec1Memrefs.push_back(memref1.getResult());
      }
      return mlir::WalkResult::interrupt();  // 只处理第一个 scope
    }
    return mlir::WalkResult::advance();
  });

  ssbufferMemrefs.push_back(ssbufferVec0Memrefs);
  ssbufferMemrefs.push_back(ssbufferVec1Memrefs);
  return ssbufferMemrefs;
}
```

**要点**：
- 槽位数量 = 跨核依赖组的 producer 组总数（`countProducerGroups`）；
- **Vector 0 指针**：地址 `i*4`（0,4,8,...）；**Vector 1 指针**：地址 `1024+i*4`——两核计数槽互不重叠；
- 每个槽在 scope 前 store 0 初始化（计数从 0 开始，volatile 语义由后续 load/store 属性保证）；
- `interrupt` 只处理遇到的第一个 scope（跨核共享计数只需一份）。

### 3.3 主编排：updateIfConds（L1797-L1917）

```cpp
int UpdateConditionInfoPass::updateIfConds(ModuleOp module,
                                           SmallVector<SmallVector<Value>> ssbufferPtrs) {
  // 收集所有主循环（walk 校验 kMainLoop 必为 for/while）
  SmallVector<Operation *> mainLoopOps;
  ...module.walk(...)...
  if (walkResult.wasInterrupted()) return UPDATE_CONDITION_INFO_FAILED;

  // Step0: 依赖缓冲一次性收集（跨核全模块、核内逐循环）
  DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>> crossCoreBuffers;
  DenseMap<Operation *, DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>>
      intraCoreBuffersMap;
  collectDependencyBuffers(module, mainLoopOps, crossCoreBuffers, intraCoreBuffersMap);

  for (Operation *loopOp : mainLoopOps) {
    controlVarToLatestValue.clear();   // 每循环重置最新值映射

    // Step1: 取本循环的核内缓冲（可能为空）
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>> intraCoreBuffers;
    if (intraCoreBuffersMap.count(loopOp)) {
      intraCoreBuffers = intraCoreBuffersMap[loopOp];
    }
    // for/while 至少要有跨核或核内其一依赖
    if (crossCoreBuffers.empty() && intraCoreBuffers.empty()) {
      return UPDATE_CONDITION_INFO_FAILED;
    }

    // Step2: 控制变量分配（核内依赖组 -> iter_arg；tensor 依赖 -> if 映射）
    DenseMap<int, Value> idxToVar;
    if (buildIdxToVarMap(loopOp, intraCoreBuffers, idxToVar) == FAILED) return FAILED;
    if (buildTensorIterArgIfOpVarMap(loopOp) == FAILED) return FAILED;

    size_t usedCounterNum = 0;
    SmallVector<scf::IfOp> ifOps;
    if (collectSSBufferIfOps(loopOp, ifOps) == FAILED) return FAILED;
    if (validateBlockCounters(loopOp, ifOps.size()) == FAILED) return FAILED;

    // 逐个 if 更新条件
    for (scf::IfOp ifOp : ifOps) {
      // 分析输入/输出依赖组
      if (getInputOutputValues(ifOp, crossCoreBuffers, intraCoreBuffers, ...) != 0)
        return FAILED;
      // Step3: 核间条件
      Value crossCoreCond;
      if (setCrossCoreCondition(...) != 0) return FAILED;
      // Step4: 核内条件
      DenseMap<Value, VarUpdateType> varUpdateTypes;
      Value intraCoreCond;
      if (setIntraCoreCondition(...) == FAILED) return FAILED;
      // Step5: flowOpt 条件（仅 for）
      Value flowOptCond;
      if (setFlowOptCondition(ifOp, loopOp, flowOptCond) == FAILED) return FAILED;
      // Step6: 合并条件 + 重建 if
      if (combineConditions(module, crossCoreCond, intraCoreCond, flowOptCond,
                            ifOp, loopOp, usedCounterNum, varUpdateTypes) == FAILED)
        return FAILED;
    }
    // Step7: 循环 yield 更新为控制变量最新值
    if (updateLoopYield(loopOp) == FAILED) return FAILED;
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

**要点**：外层按主循环、内层按 if 顺序处理。`controlVarToLatestValue` 每循环重置；if 按 `collectSSBufferIfOps` 的 walk 顺序处理（与 DAG/计数器索引对齐）。关键设计：**一个循环内多个 if 共享状态流**——前一个 if 更新的控制变量，后一个 if 用 `controlVarToLatestValue` 读最新值，最后统一写回循环 yield。

### 3.4 依赖缓冲收集：collectDependencyBuffers（L291-L327）

```cpp
void UpdateConditionInfoPass::collectDependencyBuffers(
    ModuleOp module, SmallVector<Operation *> &mainLoopOps,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>> &crossCoreBuffers,
    DenseMap<Operation *, DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>>
        &intraCoreBuffersMap) {
  // 跨核：按 walk 顺序给每个 (consumer -> producers) 组编连续索引
  int crossCoreIdx = 0;
  module.walk([&](Operation *op) {
    auto it = info->crossCoreDependentMap.find(op);
    if (it != info->crossCoreDependentMap.end()) {
      for (SmallVector<Operation *> &producers : it->second) {
        crossCoreBuffers[crossCoreIdx][op] = producers;  // 每 group 一个索引
        crossCoreIdx++;
      }
    }
    return WalkResult::advance();
  });

  // 核内：逐主循环，同一循环内 group 编连续索引
  for (Operation *loopOp : mainLoopOps) {
    if (info->intraCoreDependentMap.count(loopOp)) {
      auto &loopDeps = info->intraCoreDependentMap[loopOp];
      DenseMap<int, ...> intraCoreBuffers;
      int intraCoreIdx = 0;
      for (auto &entry : loopDeps) {
        intraCoreBuffers[intraCoreIdx][entry.first] = entry.second;
        intraCoreIdx++;
      }
      intraCoreBuffersMap[loopOp] = intraCoreBuffers;
    }
  }
}
```

**要点**：把 InitDependentMap 的映射（consumer 键）转成**按 group 编号索引**的结构。编号顺序决定 SSBuffer 槽位（跨核）与 `innerDepConds` iter_arg（核内）的一一对应，**必须与 InitDependentMap/UpdateLoopOps 的分配顺序一致**（都是 walk / map 顺序，确定性保证）。

### 3.5 控制变量分配：buildIdxToVarMap（L355-L395）

```cpp
int UpdateConditionInfoPass::buildIdxToVarMap(
    Operation *loopOp, const ... &intraCoreBuffers, DenseMap<int, Value> &idxToVar) {
  int varIdx = 0;
  MutableArrayRef<BlockArgument> regionIterArgs;
  if (getLoopRegionIterArgs(loopOp, regionIterArgs) == FAILED) return FAILED;
  int iterArgNum = regionIterArgs.size();

  const auto &innerDepIndices = info->innerDepConds[loopOp];
  if (innerDepIndices.size() < intraCoreBuffers.size()) {
    return FAILED;  // 计数变量不够
  }

  for (const auto &entry : intraCoreBuffers) {
    int idx = entry.first;
    int argIdx = innerDepIndices[varIdx];   // 第 varIdx 个依赖组 -> 第 argIdx 个 iter_arg
    if (argIdx < 0 || argIdx >= iterArgNum) return FAILED;
    idxToVar[idx] = regionIterArgs[argIdx]; // 依赖组索引 -> 控制变量
    varIdx++;
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

**要点**：核内依赖组索引（`intraCoreBuffers` 的 key）到 loop iter_arg 的映射，靠 `innerDepConds[loopOp]`（UpdateLoopOps 记录的新 iter_arg 索引列表）按序对应。**一个依赖组一个控制变量**。

### 3.6 输入/输出依赖组分析：getInputOutputValues（L439-L527）

```cpp
int UpdateConditionInfoPass::getInputOutputValues(
    scf::IfOp ifOp, ...crossCoreBuffers, ...intraCoreBuffers,
    SmallVector<int> &crossCoreInputValues, SmallVector<int> &crossCoreOutputValues,
    SmallVector<int> &intraCoreInputValues, SmallVector<int> &intraCoreOutputValues) {
  DenseSet<int> crossCoreInputSet, crossCoreOutputSet, intraCoreInputSet, intraCoreOutputSet;

  // 构建反查表：consumer op -> [groupIdx]；producer op -> [groupIdx]
  DenseMap<Operation *, SmallVector<int>> crossCoreConsumerToGroups, crossCoreOutputToGroups;
  DenseMap<Operation *, SmallVector<int>> intraCoreConsumerToGroups, intraCoreOutputToGroups;
  buildBufferDependencyMappings(crossCoreBuffers, crossCoreConsumerToGroups, crossCoreOutputToGroups);
  buildBufferDependencyMappings(intraCoreBuffers, intraCoreConsumerToGroups, intraCoreOutputToGroups);

  // 遍历 if 内 op（除 if 自身）：
  ifOp.walk([&](Operation *op) {
    if (op == ifOp) return WalkResult::advance();
    // 该 op 是 consumer → 对应组是输入
    if (crossCoreConsumerToGroups.count(op)) {
      for (int idx : crossCoreConsumerToGroups[op]) crossCoreInputSet.insert(idx);
    }
    if (intraCoreConsumerToGroups.count(op)) {
      for (int idx : intraCoreConsumerToGroups[op]) intraCoreInputSet.insert(idx);
    }
    // 该 op 是 producer → 对应组是输出
    if (crossCoreOutputToGroups.count(op)) {
      for (int idx : crossCoreOutputToGroups[op]) crossCoreOutputSet.insert(idx);
    }
    if (intraCoreOutputToGroups.count(op)) {
      for (int idx : intraCoreOutputToGroups[op]) intraCoreOutputSet.insert(idx);
    }
    return WalkResult::advance();
  });

  // DenseSet -> SmallVector（确定性输出）
  crossCoreInputValues.assign(crossCoreInputSet.begin(), crossCoreInputSet.end());
  ... 同 crossCoreOutput / intraCoreInput / intraCoreOutput ...
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

**要点**：if 的同步条件由它**读哪些组**（输入）与**写哪些组**（输出）决定。反查表让 O(1) 判断任意 op 的组归属；一个 op 可同时是多个组的 consumer/producer。DenseSet 无序但迭代确定性（LLVM 保证相同插入序下输出稳定）。

### 3.7 核间条件总装：setCrossCoreCondition（L783-L855）

```cpp
int UpdateConditionInfoPass::setCrossCoreCondition(
    ... , scf::IfOp ifOp, SmallVector<SmallVector<Value>> ssbufferPtrs,
    Value &crossCoreCond) {
  OpBuilder builder(ifOp);

  // ========== Part 1: 判定核型 ==========
  // 沿 parent 链找 scope，读 hivm.tcore_type
  bool isAIC = false, isAIV = false;
  ... while (parentOp) { if (ScopeOp) { scopeOp = parentOp; break; } ... }
  if (scopeOp && scopeOp->hasAttr("hivm.tcore_type")) {
    if (attr == aiCAttr) isAIC = true;
    else if (attr == aivAttr) isAIV = true;
    else return FAILED;
  } else return FAILED;

  Value zeroConst = builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
  Value oneConst = builder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);

  // Vector 侧需要计算本子块的 SSBuffer 指针
  DenseMap<int, Value> VectorSSBufferPtrs;
  if (!isAIC) {
    auto result = computeVectorSSBufferMemrefs(builder, loc, scopeOp,
                                               crossCoreInputValues, crossCoreOutputValues);
    if (!result) return FAILED;
    VectorSSBufferPtrs = std::move(*result);
  }
  builder.setInsertionPoint(ifOp);

  // ========== Part 2: 构造条件 ==========
  crossCoreCond = addCrossCoreConditions(builder, loc, crossCoreInputValues,
      crossCoreOutputValues, crossCoreBuffers, isAIC, zeroConst,
      VectorSSBufferPtrs, ssbufferPtrs);

  // ========== Part 3: then 分支更新计数 ==========
  updateCrossCoreControlVars(builder, loc, ifOp, crossCoreInputValues,
      crossCoreOutputValues, isAIC, oneConst, VectorSSBufferPtrs, ssbufferPtrs);
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

**要点**：Part1 判定 if 在 CUBE（isAIC）还是 Vector（isAIV）核上——决定用固定指针（ssbufferPtrs 的哪组）还是 Vector 动态指针；Part2 生成条件；Part3 在 then 分支注入计数增减（下一小节）。

### 3.8 条件与更新：addCrossCoreConditions / updateCrossCoreControlVars / createSsbufLoads（L684-L780, L662-L681）

```cpp
Value UpdateConditionInfoPass::addCrossCoreConditions(
    ..., SmallVector<int> crossCoreInputValues, SmallVector<int> crossCoreOutputValues,
    ...&crossCoreBuffers, bool isAIC, Value zeroConst, ...) {
  Value conditions = nullptr;
  auto combineCondition = [&](Value newCond) {
    if (conditions) conditions = builder.create<arith::AndIOp>(loc, conditions, newCond);
    else conditions = newCond;
  };

  // 输入组：count > 0（有缓冲可读）
  for (int inputGroupIdx : crossCoreInputValues) {
    auto conds = createSsbufLoads(builder, loc, isAIC, inputGroupIdx, ...,
        [&](memref::LoadOp loadOp, auto) {
          return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, loadOp, zeroConst);
        });
    Value cond = conds[0];
    if (isAIC) {
      cond = builder.create<arith::AndIOp>(loc, conds[0], conds[1]);  // CUBE 双核都要有货
    }
    combineCondition(cond);
  }

  // 输出组：count < producer_num（有坑可写）
  for (int outputGroupIdx : crossCoreOutputValues) {
    int outputCount = 0;
    for (auto &entry : crossCoreBuffers[outputGroupIdx])
      outputCount += entry.second.size();   // 该组 producer 总数
    Value bufferNum = builder.create<arith::ConstantIntOp>(loc, outputCount, CONST_INT_TYPE);
    auto conds = createSsbufLoads(builder, loc, isAIC, outputGroupIdx, ...,
        [&](Value cond, auto) {
          return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, cond, bufferNum);
        });
    Value cond = conds[0];
    if (isAIC) {
      cond = builder.create<arith::AndIOp>(loc, conds[0], conds[1]);
    }
    combineCondition(cond);
  }
  return conditions;
}
```

```cpp
// volatile load：CUBE 读 1 个指针；Vector 读 2 个（ptrSetIdx 0/1），MarkOp 打 volatile
template <typename FuncTy>
auto UpdateConditionInfoPass::createSsbufLoads(
    OpBuilder &builder, Location loc, bool isAIC, int groupIdx, ...) {
  using ResTy = std::invoke_result_t<FuncTy, memref::LoadOp, int>;
  SmallVector<ResTy, kVectorCount> res;
  size_t resNum = isAIC ? kVectorCount : 1;   // CUBE 侧 2、Vector 侧 1
  for (int ptrSetIdx : llvm::seq(resNum)) {
    auto loadOp = builder.create<memref::LoadOp>(loc, getSSBufferMemref(...), ValueRange{});
    auto markOp = builder.create<annotation::MarkOp>(loc, loadOp);
    markOp->setAttr(kMemrefExtVolatile, builder.getUnitAttr());  // volatile！
    res.push_back(std::invoke(pred, loadOp, ptrSetIdx));
  }
  return res;
}
```

```cpp
// then 分支 yield 前：输入组 -1，输出组 +1（volatile store）
void UpdateConditionInfoPass::updateCrossCoreControlVars(
    ..., scf::IfOp ifOp, SmallVector<int> crossCoreInputValues,
    SmallVector<int> crossCoreOutputValues, bool isAIC, Value oneConst, ...) {
  Block *thenBlock = &ifOp.getThenRegion().front();
  auto yieldOp = cast<scf::YieldOp>(thenBlock->getTerminator());
  builder.setInsertionPoint(yieldOp);

  for (int inputGroupIdx : crossCoreInputValues) {
    createSsbufLoads(builder, loc, isAIC, inputGroupIdx, ..., [&](Value loadRes, int ptrSetIdx) {
      Value newValue = builder.create<arith::SubIOp>(loc, loadRes, oneConst);
      return builder.create<memref::StoreOp>(loc, newValue, getSSBufferMemref(...), ValueRange{});
    });
  }
  for (int outputGroupIdx : crossCoreOutputValues) {
    createSsbufLoads(builder, loc, isAIC, outputGroupIdx, ..., [&](Value loadRes, int ptrSetIdx) {
      Value newValue = builder.create<arith::AddIOp>(loc, loadRes, oneConst);
      return builder.create<memref::StoreOp>(loc, newValue, getSSBufferMemref(...), ValueRange{});
    });
  }
}
```

**要点**（跨核同步的核心语义）：
- **生产者在消费计数槽 +1、消费者在生产计数槽 -1**——各组的"已用/已还"计数跨核共享；
- **读侧**：CUBE 侧双核（ptrSet 0/1）计数必须**同时 > 0** 才放行（And），Vector 侧单核只看自己的；
- **写侧**：计数 < producer 总数才允许写（缓冲未满）；
- `volatile` 标记保证跨核 load/store 不缓存、不被重排——这是跨核同步正确性的前提；
- 条件在 if 前（`setInsertionPoint(ifOp)`），更新在 then yield 前——**先检查后执行**。

### 3.9 Vector 动态指针：computeVectorSSBufferMemrefs（L614-L660）

```cpp
std::optional<DenseMap<int, Value>>
UpdateConditionInfoPass::computeVectorSSBufferMemrefs(
    OpBuilder &builder, Location loc, Operation *scopeOp, ...) {
  // 收集输入输出组去重
  SmallVector<int> allGroupIndices; DenseSet<int> uniqueIndices;
  for (int idx : crossCoreInputValues) if (uniqueIndices.insert(idx).second) allGroupIndices.push_back(idx);
  for (int idx : crossCoreOutputValues) if (uniqueIndices.insert(idx).second) allGroupIndices.push_back(idx);

  builder.setInsertionPointToStart(&scopeOp->getRegion(0).front());
  int vec1Offset = 1024;
  Value vec1OffsetValue = builder.create<arith::ConstantIntOp>(loc, VECTOR_SSBUF_OFFSET, ADDR_INT_TYPE);
  auto subIdOp = builder.create<GetSubBlockIdxOp>(loc, builder.getIntegerType(ADDR_INT_TYPE));
  Value ssbAddrOffset = builder.create<arith::MulIOp>(loc, subIdOp, vec1OffsetValue);

  for (int groupIdx : allGroupIndices) {
    auto ssbBaseAddr = builder.create<arith::ConstantIntOp>(loc, groupIdx * VALUE_SSBUF_OFFSET, ADDR_INT_TYPE);
    auto ssbAddr = builder.create<arith::AddIOp>(loc, ssbBaseAddr, ssbAddrOffset);
    Value memref = builder.create<PointerCastOp>(loc, getSsbufMemrefType(builder), ssbAddr.getResult());
    vectorSSBufferMemrefs[groupIdx] = memref;
  }
  return vectorSSBufferMemrefs;
}
```

**要点**：Vector 侧指针 = `groupIdx*4 + GetSubBlockIdx*1024`——`GetSubBlockIdxOp` 取当前子块（Vector 0/1）索引，乘 1024 得到本子块的 SSBuffer 区基址。因此**一个 memref 在 Vector 双核上自然指向各自区域**，无需两套指针。

### 3.10 核内条件：setIntraCoreCondition（L1081-L1132）

```cpp
int UpdateConditionInfoPass::setIntraCoreCondition(
    ..., scf::IfOp ifOp, ...intraCoreBuffers,
    SmallVector<int> &intraCoreInputValues, SmallVector<int> &intraCoreOutputValues,
    DenseMap<int, Value> &idxToVar, DenseMap<Value, VarUpdateType> &varUpdateTypes,
    Value &intraCoreCond) {
  intraCoreCond = Value();
  OpBuilder builder(ifOp.getContext());
  builder.setInsertionPoint(ifOp);

  SmallVector<Value> conditions;
  DenseSet<Value> usedVarsSet;
  // 1. 核内输入条件（var > 0，DEC）
  collectIntraCoreInputConditions(builder, loc, intraCoreInputValues, idxToVar,
                                  conditions, usedVarsSet, varUpdateTypes);
  // 2. 核内输出条件（var < producer_limit，INC）
  if (collectIntraCoreOutputConditions(builder, loc, intraCoreBuffers, intraCoreOutputValues,
                                       idxToVar, conditions, usedVarsSet, varUpdateTypes) == FAILED)
    return FAILED;
  // 3. tensor 迭代依赖的消费者/生产者条件
  collectTensorIterArgInputConditions(builder, loc, ifOp, conditions, usedVarsSet, varUpdateTypes);
  collectTensorIterArgOutputConditions(builder, loc, ifOp, conditions, usedVarsSet, varUpdateTypes);

  if (!conditions.empty()) {
    intraCoreCond = conditions[0];
    for (size_t i = 1; i < conditions.size(); ++i)
      intraCoreCond = builder.create<arith::AndIOp>(loc, intraCoreCond, conditions[i]);
  }
  currentUsedVars.clear();
  for (Value var : usedVarsSet) currentUsedVars.push_back(var);
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

**要点**：核内条件 = 输入 `var>0` ∧ 输出 `var<limit` ∧ tensor 消费者 `var==1` ∧ tensor 生产者 `var==0`。`currentUsedVars`（本次 if 用到的控制变量集合）是后续重建 if 时决定"需要多少额外结果"的依据。**控制变量更新不在 if 前做，而是在 then 分支 yield 时按 `varUpdateTypes` 施加**（见 3.14）。

### 3.11 核内输入/输出条件细节（L858-L939）

```cpp
void UpdateConditionInfoPass::collectIntraCoreInputConditions(...) {
  // 每个输入组：var > 0，标记 DEC
  for (int idx : intraCoreInputValues) {
    auto varIt = idxToVar.find(idx);
    if (varIt == idxToVar.end()) continue;   // 无控制变量则跳过
    Value var = varIt->second;
    Value varToUse = var;
    auto latestIt = controlVarToLatestValue.find(var);
    if (latestIt != controlVarToLatestValue.end()) varToUse = latestIt->second;  // 最新值！
    Value cond = builder.create<arith::CmpIOp>(loc, sgt, varToUse, zeroConst);
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::DEC;
  }
}
```

```cpp
int UpdateConditionInfoPass::collectIntraCoreOutputConditions(
    ..., SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
    SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  // 先按输出 op 集合聚合（buildOutputGroups），相同 producer 集合共享 limit
  SmallVector<OutputGroupInfo> outputGroups;
  if (buildOutputGroups(intraCoreOutputValues, intraCoreBuffers, idxToVar, outputGroups) == FAILED)
    return FAILED;
  for (auto &group : outputGroups) {
    int size = group.outputs.size();   // producer 数 = 缓冲上限
    Value limitVal = builder.create<arith::ConstantIntOp>(loc, size, CONST_INT_TYPE);
    for (Value var : group.inputVars) {
      Value varToUse = ...最新值...;
      Value cond = builder.create<arith::CmpIOp>(loc, slt, varToUse, limitVal);
      conditions.push_back(cond);
      usedVarsSet.insert(var);
      varUpdateTypes[var] = VarUpdateType::INC;
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

**要点**：
- 核内输入/输出都**优先用 `controlVarToLatestValue` 的最新值**判断——同一个 var 在本轮可能已被前面的 if 更新过；
- 输出聚合 `buildOutputGroups`：多个依赖组如果共享同一 producer 集合，合并为一个限制条件（相同 producer 集合的缓冲可用上限一致），避免重复条件。

### 3.12 tensor 迭代依赖：buildTensorIterArgIfOpVarMap + 条件收集（L942-L1078）

```cpp
int UpdateConditionInfoPass::buildTensorIterArgIfOpVarMap(Operation *loopOp) {
  tensorIterArgIfOpVars.clear();
  if (!info->tensorIterArgDepsMap.count(loopOp) || !info->tensorIterArgIndicesMap.count(loopOp))
    return UPDATE_CONDITION_INFO_SUCCESS;   // 无 tensor 依赖则跳过

  auto &depsVec = info->tensorIterArgDepsMap[loopOp];
  auto &indicesMap = info->tensorIterArgIndicesMap[loopOp];
  llvm::DenseMap<scf::IfOp, llvm::DenseSet<Value>> producerVars;
  llvm::DenseMap<scf::IfOp, llvm::DenseSet<Value>> consumerVars;

  for (auto &depEntry : depsVec) {
    Value origIterArg = depEntry.iterArg;
    TensorIterArgIfOpRelation &relation = depEntry;
    if (!indicesMap.count(origIterArg)) return FAILED;
    SmallVector<int> &argIndices = indicesMap[origIterArg];
    if (relation.consumers.size() != argIndices.size()) return FAILED;

    // consumer -> 各自独立的计数变量（一一对应）
    llvm::DenseMap<scf::IfOp, Value> consumerToVar;
    for (size_t i = 0; i < relation.consumers.size(); ++i) {
      Value var;
      if (getLoopRegionIterArg(loopOp, argIndices[i], var) == FAILED) return FAILED;
      consumerToVar[relation.consumers[i]] = var;
    }

    // producer 关联所有依赖它的 consumer 的变量（作为其"产出状态"）
    if (relation.producer) {
      for (auto &[consumer, var] : consumerToVar)
        producerVars[relation.producer].insert(var);
    }
    // consumer 关联自己的变量
    for (auto &[consumer, var] : consumerToVar)
      consumerVars[consumer].insert(var);
  }

  for (auto &[producer, vars] : producerVars) tensorIterArgIfOpVars[producer].producerVars = ...;
  for (auto &[consumer, vars] : consumerVars) tensorIterArgIfOpVars[consumer].consumerVars = ...;
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

```cpp
// 消费者：var == 1 才可消费（DEC）
void UpdateConditionInfoPass::collectTensorIterArgInputConditions(...) {
  for (Value var : ifOpVars.consumerVars) {
    Value varToUse = ...最新值...;
    Value oneConst = builder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);
    Value cond = builder.create<arith::CmpIOp>(loc, eq, varToUse, oneConst);
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::DEC;
  }
}
```

```cpp
// 生产者：var == 0 才可生产（INC）
void UpdateConditionInfoPass::collectTensorIterArgOutputConditions(...) {
  for (Value var : ifOpVars.producerVars) {
    Value varToUse = ...最新值...;
    Value zeroConst = builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
    Value cond = builder.create<arith::CmpIOp>(loc, eq, varToUse, zeroConst);
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::INC;
  }
}
```

**要点**：tensor 跨迭代依赖用**精确判据**（`==1`/`==0`）而非区间判据——因为一个 producer if 对应多个 consumer，张量缓冲只能在"刚好生产 1 份待消费"时读、在"本轮无人待消费"时写。生产者 if 的产出变量 = 所有依赖它的 consumer 的计数变量（producer 必须在所有 consumer 都消费完上一轮后才能再产）。

### 3.13 flowOpt 条件：setFlowOptCondition（L1135-L1206）

```cpp
int UpdateConditionInfoPass::setFlowOptCondition(scf::IfOp currentIfOp,
                                                 Operation *loopOp, Value &flowOptCond) {
  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  if (!forOp) { flowOptCond = nullptr; return UPDATE_CONDITION_INFO_SUCCESS; }  // 仅 for

  if (!info->flowOptIfOpPairs.count(currentIfOp)) { flowOptCond = nullptr; return SUCCESS; }
  // 缓冲数超阈值才启用
  if (info->crossCoreBufferCount <= CROSS_CORE_BUFFER_COUNT_THRESHOLD ||
      info->intraCoreBufferCount <= INTRA_CORE_BUFFER_COUNT_THRESHOLD) {
    flowOptCond = nullptr; return SUCCESS;
  }

  scf::IfOp sourceIfOp = info->flowOptIfOpPairs[currentIfOp];
  if (!info->cntArgs.count(sourceIfOp)) return FAILED;   // 起点必须有计数器

  Value counter = info->cntArgs[sourceIfOp];
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  // cond1: counter >= upperBound（迭代末期强制放行）
  Value cond1 = builder.create<arith::CmpIOp>(loc, sge, counter, upperBound);
  // opt_num = min(intraCoreBufferCount-1, crossCoreBufferCount)
  int optInt = std::min(info->intraCoreBufferCount - 1, info->crossCoreBufferCount);
  Value optNum = builder.create<arith::ConstantIntOp>(loc, optInt, ...);
  Value optOffset = builder.create<arith::MulIOp>(loc, step, optNum);
  Value lowerPlusOffset = builder.create<arith::AddIOp>(loc, lowerBound, optOffset);
  Value cond2 = builder.create<arith::CmpIOp>(loc, sge, counter, lowerPlusOffset);

  // OR 合并：达到重叠窗口就放行
  flowOptCond = builder.create<arith::OrIOp>(loc, cond1, cond2);
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

**要点**：flowOpt 目标 if（DAG 深度 3）在常规同步条件之外**追加一个放行项**——当起点 if 的计数器已进入"重叠窗口"（`counter >= lowerBound + step*opt_num`）或迭代末期（`counter >= upperBound`）时，目标 if 不再等待同步（OR 语义 = 任一条满足即可）。重叠窗口大小 = `opt_num` 个迭代，取决于缓冲余量。这是"缓冲充足时提前 overlap 流水线"的开关。

### 3.14 条件合并与 if 重建：combineConditions（L1653-L1794）

```cpp
int UpdateConditionInfoPass::combineConditions(
    ModuleOp module, Value crossCoreCond, Value intraCoreCond,
    Value flowOptCond, scf::IfOp ifOp, Operation *loopOp,
    size_t &usedCounterNum, DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  SmallVector<Value> validConditions;
  Value counter;
  bool updateCounterArg = false;

  if (crossCoreCond) validConditions.push_back(crossCoreCond);
  if (intraCoreCond) validConditions.push_back(intraCoreCond);
  if (flowOptCond) validConditions.push_back(flowOptCond);

  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  auto whileOp = dyn_cast<scf::WhileOp>(loopOp);
  OpBuilder condBuilder(ifOp);

  if (forOp) {
    // 计数器条件：counter < upperBound
    if (!info->blockCounters.count(forOp)) return FAILED;
    SmallVector<int> &counterIndices = info->blockCounters[forOp];
    if (info->cntArgs.count(ifOp)) {
      counter = info->cntArgs[ifOp];        // flowOpt 起点 if 已分配过计数器
      updateCounterArg = true;
    } else {
      if (usedCounterNum >= counterIndices.size()) return FAILED;
      int argIdx = counterIndices[usedCounterNum];
      counter = forOp.getRegionIterArgs()[argIdx];
      updateCounterArg = true;
      info->cntArgs[ifOp] = counter;
      usedCounterNum++;                     // 每个 if 顺序领一个计数器
    }
    Value counterToUse = ...最新值...;
    Value counterCond = condBuilder.create<arith::CmpIOp>(loc, slt, counterToUse, upperBound);
    validConditions.push_back(counterCond);
  } else if (whileOp) {
    // while：重映射 scf.condition(x) 的 x
    Value counterCond;
    if (buildWhileCounterCondition(whileOp, ifOp, info, condBuilder,
                                   controlVarToLatestValue, counterCond) == FAILED)
      return FAILED;
    validConditions.push_back(counterCond);
  } else return FAILED;

  if (validConditions.empty()) return FAILED;

  // And 合并
  Value combinedCond = validConditions[0];
  for (size_t i = 1; i < validConditions.size(); ++i)
    combinedCond = builder.create<arith::AndIOp>(loc, combinedCond, validConditions[i]);

  Value step = updateCounterArg ? forOp.getStep() : Value();
  scf::IfOp newIfOp = createNewIfOpWithBlocks(ifOp, combinedCond, varUpdateTypes,
                                              updateCounterArg, counter, step);

  // 迁移所有指向旧 if 的映射
  updateDAGAfterIfOpReplacement(ifOp, newIfOp);
  if (updateCounterArg) {
    info->cntArgs.erase(ifOp);
    info->cntArgs[newIfOp] = counter;
  }
  if (tensorIterArgIfOpVars.count(ifOp)) { ...迁移到 newIfOp... }
  if (info->tensorIterArgDepsMap.count(loopOp)) {
    for (auto &relation : depsVec) {
      if (relation.producer.getOperation() == ifOp.getOperation()) relation.producer = newIfOp;
      for (auto &consumerIfOp : relation.consumers)
        if (consumerIfOp.getOperation() == ifOp.getOperation()) consumerIfOp = newIfOp;
    }
  }

  updateControlVarToLatestValue(newIfOp, ifOp, updateCounterArg, counter);
  ifOp.erase();
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

**要点**：
- **for 计数器条件**：`counter < upperBound`——保证 block 不超过循环上界（扩展后的迭代次数由 UpdateLoopIterTimes 负责）；
- **计数器按 if 顺序领取**：`usedCounterNum` 递增，每个 if 一个计数器（block id 顺序与 if 处理顺序一致）；flowOpt 起点 if 的计数器已在 `cntArgs` 缓存；
- **while 计数器条件**：无 iter_arg 计数器，改用重映射的 `scf.condition(x)`（见 3.15）；
- 重建新 if 后**必须迁移所有映射**（DAG、cntArgs、tensor 映射），否则后续 if 与循环 yield 查不到新 if；
- `updateControlVarToLatestValue` 记录本次迭代后各控制变量的**最新值**（新 if 的扩展结果），供后续 if 使用。

### 3.15 while 条件重映射：buildWhileCounterCondition + cloneConditionDefChain（L188-L239, L142-L183）

```cpp
static int buildWhileCounterCondition(
    scf::WhileOp whileOp, scf::IfOp ifOp, ControlFlowConditionInfo *info,
    OpBuilder &builder, const DenseMap<Value, Value> &controlVarToLatestValue,
    Value &outCond) {
  int blockId;
  if (getIfBlockId(ifOp, blockId) == FAILED) return FAILED;

  auto whileIt = info->whileBlockArgMap.find(whileOp);
  if (whileIt == info->whileBlockArgMap.end()) return FAILED;
  auto blockIt = whileIt->second.find(blockId);
  if (blockIt == whileIt->second.end()) return FAILED;

  const DenseMap<int, int> &argIdxMap = blockIt->second;  // {new(after)->old(before)}
  auto beforeArgs = whileOp.getBeforeArguments();
  auto afterArgs = whileOp.getAfterArguments();

  IRMapping mapping;
  for (auto [newArgIdx, oldArgIdx] : argIdxMap) {
    // 合法性校验...
    Value afterArg = afterArgs[newArgIdx];
    auto latestIt = controlVarToLatestValue.find(afterArg);
    if (latestIt != controlVarToLatestValue.end()) afterArg = latestIt->second;
    mapping.map(beforeArgs[oldArgIdx], afterArg);   // before 旧 arg -> after 新值
  }

  // 取 scf.condition(x) 的 x，克隆其 def 链（before 区算子）
  scf::ConditionOp condOp = whileOp.getConditionOp();
  Value beforeCond = condOp.getCondition();
  if (cloneConditionDefChain(beforeCond, whileOp.getBefore(), mapping, builder, outCond) == FAILED)
    return FAILED;
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

```cpp
static int cloneConditionDefChain(Value value, Region &beforeRegion,
                                  IRMapping &mapping, OpBuilder &builder,
                                  Value &outValue) {
  if (Value mapped = mapping.lookupOrNull(value)) { outValue = mapped; return SUCCESS; }
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getParentRegion() == &beforeRegion) return FAILED;  // 未映射的 before arg
    outValue = value; return SUCCESS;   // 外部值原样用
  }
  Operation *defOp = value.getDefiningOp();
  if (!defOp) { outValue = value; return SUCCESS; }
  if (defOp->getParentRegion() != &beforeRegion) { outValue = value; return SUCCESS; }  // 外部 op

  for (Value operand : defOp->getOperands()) {
    Value remappedOperand;
    if (cloneConditionDefChain(operand, beforeRegion, mapping, builder, remappedOperand) == FAILED)
      return FAILED;
    if (!mapping.lookupOrNull(operand)) mapping.map(operand, remappedOperand);
  }
  Operation *cloned = builder.clone(*defOp, mapping);   // 克隆 def 链到当前插入点
  outValue = cloned->getResult(cast<OpResult>(value).getResultNumber());
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

**要点**：while 的终止条件在 before 区、基于 before 区 arg。要把它变成"每个 block 自己的 if 条件"，需把 `scf.condition(x)` 的 x **克隆一份到 if 前**，且把 x 依赖的 before arg 通过 `whileBlockArgMap`（ProcessArgs 记录）映射成 after 区新 arg 的最新值。递归克隆只克隆 before 区内产生的算子，外部值（常量/闭包捕获）原样复用。

### 3.16 重建 if：createNewIfOpWithBlocks（L1606-L1649）

```cpp
scf::IfOp UpdateConditionInfoPass::createNewIfOpWithBlocks(
    scf::IfOp oldIfOp, Value combinedCond,
    DenseMap<Value, VarUpdateType> &varUpdateTypes, bool hasCounter,
    Value counter, Value step) {
  bool needsYield = !currentUsedVars.empty() || hasCounter;  // 需要额外结果
  bool oldHasElse = oldIfOp.getElseRegion().hasOneBlock();
  bool withElse = needsYield || oldHasElse;

  // 收集旧 then 的 yield operands，构造新结果类型：旧结果 + 控制变量 + 计数器
  Block &oldThenBlock = oldIfOp.getThenRegion().front();
  Operation *oldThenYieldOp = nullptr;
  SmallVector<Value> oldYieldOperands;
  collectYieldOperands(oldThenBlock, oldThenYieldOp, oldYieldOperands);
  SmallVector<Type> resultTypes = buildNewIfResultTypes(oldIfOp, hasCounter, counter);
  scf::IfOp newIfOp = builder.create<scf::IfOp>(loc, resultTypes, combinedCond, withElse);

  // 属性原样迁移（含 kIf=blockId）
  for (auto &attr : oldIfOp->getAttrs())
    newIfOp->setAttr(attr.getName(), attr.getValue());

  populateNewThenBlock(newIfOp, oldThenBlock, oldThenYieldOp, oldYieldOperands,
                       varUpdateTypes, hasCounter, counter, step);
  if (withElse) {
    populateNewElseBlock(newIfOp, oldIfOp, oldHasElse, hasCounter, counter);
  }
  // 旧结果 uses 重定向到新 if 结果
  for (size_t i = 0; i < oldIfOp.getNumResults(); ++i)
    oldIfOp.getResult(i).replaceAllUsesWith(newIfOp.getResult(i));
  return newIfOp;
}
```

**要点**：新 if = 旧 if + 额外结果（控制变量最新值 + 计数器新值）。`kIf=blockId` 属性随迁移保留——后续 Pass（UpdateLoopIterTimes）仍以 kIf 识别 if 块。

### 3.17 then/else 填充：populateNewThenBlock / populateNewElseBlock（L1494-L1603）

```cpp
void UpdateConditionInfoPass::populateNewThenBlock(
    scf::IfOp newIfOp, Block &oldThenBlock, Operation *oldThenYieldOp,
    ArrayRef<Value> oldYieldOperands,
    DenseMap<Value, VarUpdateType> &varUpdateTypes, bool hasCounter,
    Value counter, Value step) {
  Block &newThenBlock = newIfOp.getThenRegion().front();
  // 移动旧 op（跳过旧 yield）
  for (Operation &op : llvm::make_early_inc_range(oldThenBlock)) {
    if (&op != oldThenYieldOp) op.moveBefore(&newThenBlock, newThenBlock.end());
  }

  OpBuilder thenBuilder(&newThenBlock, newThenBlock.end());
  SmallVector<Value> thenYieldOperands(oldYieldOperands.begin(), oldYieldOperands.end());
  // 控制变量：DEC -> -1，INC -> +1（基于最新值）
  if (!currentUsedVars.empty()) {
    Value one = thenBuilder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);
    for (Value var : currentUsedVars) {
      Value varToUse = ...最新值...;
      Value yieldVal = varToUse;
      auto it = varUpdateTypes.find(var);
      if (it != varUpdateTypes.end()) {
        if (it->second == VarUpdateType::DEC) yieldVal = thenBuilder.create<arith::SubIOp>(loc, varToUse, one);
        else if (it->second == VarUpdateType::INC) yieldVal = thenBuilder.create<arith::AddIOp>(loc, varToUse, one);
      }
      thenYieldOperands.push_back(yieldVal);
    }
  }
  // 计数器：+step
  if (hasCounter) {
    Value newCounter = thenBuilder.create<arith::AddIOp>(loc, counter, step);
    thenYieldOperands.push_back(newCounter);
  }
  thenBuilder.create<scf::YieldOp>(loc, thenYieldOperands);
}
```

```cpp
void UpdateConditionInfoPass::populateNewElseBlock(
    scf::IfOp newIfOp, scf::IfOp oldIfOp, bool oldHasElse, bool hasCounter, Value counter) {
  Block &newElseBlock = newIfOp.getElseRegion().front();
  SmallVector<Value> oldElseYieldOperands;
  Operation *oldElseYieldOp = nullptr;
  if (oldHasElse) {
    // 移动旧 else op，收集旧 yield operands
    ...
  }
  OpBuilder elseBuilder(&newElseBlock, newElseBlock.end());
  SmallVector<Value> elseYieldOperands;
  // 旧结果 operand 换成最新值
  for (Value operand : oldElseYieldOperands) {
    Value newOperand = operand;
    auto it = controlVarToLatestValue.find(operand);
    if (it != controlVarToLatestValue.end()) newOperand = it->second;
    elseYieldOperands.push_back(newOperand);
  }
  // 控制变量：透传最新值（不更新）
  for (Value var : currentUsedVars) {
    Value varToUse = ...最新值...;
    elseYieldOperands.push_back(varToUse);
  }
  // 计数器：透传最新值（不 +step）
  if (hasCounter) { ...counterToUse... elseYieldOperands.push_back(counterToUse); }
  elseBuilder.create<scf::YieldOp>(loc, elseYieldOperands);
  if (oldElseYieldOp) oldElseYieldOp->erase();
}
```

**要点**：**then 分支"更新"、else 分支"保持"**——if 为真（执行了本 block 运算）则消费/生产计数增减、计数器 +step；if 为假（让位）则状态原样透传（用最新值，保证链上不丢更新）。

### 3.18 循环 yield 更新：updateForOpYield / updateWhileOpYield（L1277-L1390）

```cpp
int UpdateConditionInfoPass::updateForOpYield(scf::ForOp forOp) {
  if (controlVarToLatestValue.empty()) return FAILED;
  Block *forBody = forOp.getBody();
  auto yieldOp = dyn_cast<scf::YieldOp>(forBody->getTerminator());
  if (!yieldOp) return FAILED;

  SmallVector<Value> newYieldOperands(yieldOp.getOperands().begin(), yieldOp.getOperands().end());
  if (newYieldOperands.size() != forOp.getNumRegionIterArgs()) return FAILED;

  // iter_arg -> yield operand 索引
  DenseMap<Value, unsigned> iterArgToIndex;
  for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); ++j)
    iterArgToIndex[forOp.getRegionIterArgs()[j]] = j;

  // 控制变量 yield 位替换成最新值
  for (auto &entry : controlVarToLatestValue) {
    Value origVar = entry.first;
    Value latestValue = entry.second;
    auto it = iterArgToIndex.find(origVar);
    if (it == iterArgToIndex.end()) return FAILED;
    newYieldOperands[it->second] = latestValue;
  }
  OpBuilder yieldBuilder(yieldOp);
  yieldBuilder.create<scf::YieldOp>(loc, newYieldOperands);
  yieldOp.erase();
  return UPDATE_CONDITION_INFO_SUCCESS;
}
```

**要点**：循环体内所有 if 都完成后，控制变量可能被更新了多次；最终 yield 必须把每个控制变量的**最新值**写回对应 iter_arg 槽位，否则下一轮迭代读到的是未更新值。while 路径等价（按 after arg 索引）。

---

## 四、流程总结

### 4.1 UpdateConditionInfo 的流水

```
UpdateConditionInfoPass::runOnOperation
  ├─► allocSSBuffer              跨核依赖组数 -> Vector0/1 各一组计数 memref（0/1024 起，store 0）
  └─► updateIfConds
        ├─► collectDependencyBuffers     跨核全模块、核内逐循环 -> 按 group 编号
        └─► 逐 main_loop op：
              ├─► buildIdxToVarMap        核内依赖组 -> loop iter_arg（innerDepConds 对齐）
              ├─► buildTensorIterArgIfOpVarMap   tensor 依赖 -> if 的 producer/consumer 变量
              ├─► collectSSBufferIfOps + validateBlockCounters
              └─► 逐 ifOp：
                    ├─► getInputOutputValues       读哪些组/写哪些组
                    ├─► setCrossCoreCondition      核间：输入>0 ∧ 输出<producer_num + then 增减
                    ├─► setIntraCoreCondition      核内：var>0 ∧ var<limit + tensor==1/==0
                    ├─► setFlowOptCondition        深度3目标：counter>=ub OR counter>=lb+step*opt
                    └─► combineConditions          核间∧核内∧flowOpt∧counter(<ub) -> 重建 if
                          ├─► createNewIfOpWithBlocks  旧结果+控制变量+计数器结果
                          ├─► populateNewThenBlock     then: DEC/INC + counter+step
                          ├─► populateNewElseBlock     else: 透传最新值
                          ├─► 迁移 DAG/cntArgs/tensor 映射
                          └─► updateControlVarToLatestValue  记录最新值
              └─► updateLoopYield   控制变量最新值写回 yield
```

### 4.2 与前后子 Pass 的协作

- **前置**：InitDependentMap（依赖映射、DAG、flowOpt 对、缓冲数）；UpdateLoopOps（三类计数 iter_arg、innerDepConds/blockCounters/tensorIterArgIndicesMap、PIPE_S）；ProcessArgs（whileBlockArgMap）；
- **产出**：真实条件的 scf.if（带扩展结果）、SSBuffer 计数、控制变量最新值写回 yield、迁移后的 DAG/映射；
- **后置**：`UpdateLoopIterTimes` 依据 if 索引与依赖距离计算缓冲因子、扩展 for 迭代次数（保证计数器条件 `counter < upperBound` 覆盖足够轮次）。

---

## 五、面试要点

1. **核间与核内条件为什么实现不同？** 核间跨物理核，状态必须**可见共享**——用共享内存 SSBuffer 计数，volatile load/store（`kMemrefExtVolatile`）防止缓存/重排，CUBE 侧双核（ptrSet 0/1）同时校验；核内同核共享 loop iter_arg，直接读控制变量即可，无需共享内存与 volatile。tensor 跨迭代依赖因"一 producer 多 consumer"用精确判据 `==1/==0`，标量缓冲用区间判据 `>0/<limit`。

2. **为什么 CUBE 侧读 2 个计数、Vector 侧读 1 个？** Vector 双核（0/1）各自持有独立的 SSBuffer 区（基址 0 与 1024，`GetSubBlockIdx*1024` 区分），每个 Vector 核只读写自己那份；CUBE 若要从两个 Vector 核消费，必须**两个子块的计数都满足**（`conds[0] ∧ conds[1]`）才能放行。

3. **`controlVarToLatestValue` 解决什么问题？** 同一个控制变量（如某核内依赖组的计数）在一轮迭代内可能被多个 if 先后读写：第一个 if 消费（-1）、第二个 if 生产（+1）。后续任何条件判断、yield 更新都必须基于**本轮已更新过的最新值**，而非 loop iter_arg 的初始值。它贯穿整个循环处理并在每个主循环开始时清空。

4. **then 分支更新、else 分支保持的意义？** if 为真说明本 block 本轮真正执行了运算——消费了输入缓冲（-1）、产出了输出缓冲（+1）、计数器 +step（推进块轮次）；if 为假说明本 block 让位给流水线——任何状态都不能变，但必须透传最新值保证链上状态不丢。这正体现"错位执行"的流水线语义。

5. **for 与 while 的"迭代条件"来源差异？** for 有确定上界，直接用计数器 `counter < upperBound`（迭代次数由 UpdateLoopIterTimes 扩展后足够）；while 无固定次数，复用**原始终止条件**——把 before 区 `scf.condition(x)` 的 x 按 `whileBlockArgMap` 重映射并克隆 def 链到 if 前，作为本 block 的"是否继续迭代"条件。

6. **flowOpt 条件为什么是 OR 语义？** 常规同步条件（And 合并）是"全部满足才执行"的强约束；flowOpt 给 DAG 深度 3 的目标 if 加了一个**放行豁免**：起点计数器进入重叠窗口（`counter >= lb + step*opt_num`）或循环末期（`counter >= ub`）时，即使同步条件不满足也放行执行——缓冲余量（`opt_num = min(intraBufferCount-1, crossBufferCount)`）保证此时不会踩踏。OR = 常规同步 OR 重叠豁免。

7. **为什么 if 重建后必须迁移所有映射？** 后续的 if、`updateLoopYield`、`UpdateLoopIterTimes` 都通过 `kIf` 指向的 if op 与 `info` 里的映射（`ifBlockDAG`、`flowOptIfOpPairs`、`cntArgs`、`tensorIterArgDepsMap`、`tensorIterArgIfOpVars`）工作；旧 if 被 erase 后指针失效，必须全部替换成新 if，否则空悬指针崩溃或静默错位。
