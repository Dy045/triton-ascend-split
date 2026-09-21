# DynamicCVPipeline 主 Pass 专题：控制流条件注入 · 子专题 08 —— UpdateLoopIterTimes（迭代次数扩展）

> 覆盖 `AddControlFlowCondition` 的 **Step 6：UpdateLoopIterTimesPass**（独立 Pass）
>
> 源码文件：[`UpdateLoopIterTimes.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimes.cpp)、[`UpdateLoopIterTimes.h`](../../../third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimes.h)
>
> 流水线位置：`AddControlFlowCondition`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L83）内最后一个子 Pass（主编排 L133-L136）。上游是 `UpdateConditionInfo`（条件已注入）、下游直接进入 `SeparateMemoryFromCompute`。

---

## 一、这个子 Pass 在做什么

前 5 个子 Pass 已经把**条件表达式**建好了（`scf.if` 的条件是"缓冲可读可写"的同步闸门），但还有一个致命缺口：**依赖之间的缓冲距离可能大于 producer 提供的缓冲数**。

举例：producer 在 if #1、consumer 在 if #3，依赖距离 `requiredBuffers = 3 - 1 + 1 = 3`，但 producer 只有 1 块缓冲（`x = 1`）。如果主循环还是原迭代次数，第 k 轮 producer 写缓冲、第 k+1 轮 consumer 就来读——中间只隔 1 轮，缓冲被踩踏。**必须把迭代次数放大到 `ceil(iter * 3 / 1)` 才能给依赖留足"时间上的缓冲"**。

`UpdateLoopIterTimes` 的职责就是做这件事：

1. **Step1 分组**（`GetMainLoopIdToLoopOpMap`）：按 `main_loop` id 把 module 里所有主循环分成 CUBE 组和 Vector 组；
2. **Step2 算因子**（`ComputeMainLoopTimes`）：对每个 for 主循环计算 `{requiredBuffers, x}`（需要的缓冲数 / producer 拥有的缓冲数），`x` 由三类依赖分别计算后按分数最大值合并：
   - 核内依赖：`requiredBuffers = consumerIdx - producerIdx + 1`，`x = producerOps.size()`；
   - 跨核依赖：producer 在对侧核的同步 if（`syncBlockFilter` 过滤）中定位，`runFirst`（本块先执行）时再 `-1`；`comsumerIdx < producerIdx`（如 `C1→V1V2V3→C2`）属复杂情况直接返回 `{1,1}` 不扩；
   - tensor 迭代依赖：**先消费后生产**，方向相反：`requiredBuffers = producerIdx - comsumerIdx + 1`，`x = 1`；
3. **Step3 扩迭代**（`UpdateForLoopIteration`）：同 id 的 CUBE/Vector 循环**取因子最大值**（必须同步扩，保证 PIPE_S 对齐），把 for 克隆成 `new_ub = lb + step * (ceil(iter * requiredBuffers / x) + ifCount)`，迁移结果使用、更新 `cntArgs` 映射，最后统一删旧 for；
4. **Step4 换计数**（`replaceForOpCounterInIfOps`）：if 内所有 `indVar`（主循环归纳变量）的使用替换为 if 自己的计数器 `cntArgs[ifOp]`——扩迭代后 if 条件必须按"自己的第几次"判断，而非主循环第几次；
5. **Step5 改 while**（`UpdateWhileLoopCondition`）：while 没有静态 upperBound，改为把 before 区 condition 的 def 链**按 block 克隆并重映射**（`whileBlockArgMap`），每个 if block 生成独立条件，全部 OR 合并后写回 `scf.condition`。

**一句话**：这是流水线的"时间维度"收尾——把 `AddControlFlowCondition` 的同步结构从"空间多缓冲"升级为"空间 + 时间都充足"。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `IterationTimesInfo` | 每个主循环的扩展参数（头文件 L39-L44）：`ifCount`（循环内 if 数）、`requiredBuffers`（需要的缓冲数）、`x`（producer 缓冲数）、`ifOpsInThisFor`（本循环内所有 if） |
| `cmap` / `vmap` | `{main_loop_id -> [loopOp...]}`：CUBE 主循环 / Vector 主循环分组表，同一 id 两侧互为"同一流水线" |
| `infoMap` | `{loopOp -> IterationTimesInfo}`：Step2 的逐循环计算结果，Step3 消费 |
| 缓冲因子公式 | `requiredBuffers = consumer_idx - producer_idx + 1`（核内）；跨核且 `runFirst` 时再 `-1`；tensor 迭代依赖反向 `producer_idx - consumer_idx + 1` |
| 分数交叉相乘 | 比较 `requiredBuffers/x` 与 `max/maxX` 用 `requiredBuffers * maxX > max * x`，避免浮点精度 |
| 新上界公式 | `new_ub = lb + step * (ceil(iter * requiredBuffers / x) + ifCount)`，其中 `iter = ceil((ub-lb)/step)` |
| `ifCount` | 循环内 if block 数量，作为扩展迭代数的"安全余量"：给每个 if 的缓冲轮换留出一拍 |
| `collectIfOps` | 收集循环内带 `ssbuffer.if` 的 ifOp，从 1 开始编号（`ifOpIndex`），支持过滤函数 |
| `syncBlockFilter` | 只收集含 `sync_block_wait` / `sync_block_set` 的 if——跨核同步点，跨核依赖只在这些 if 上成立 |
| `findIfOpIndexInList` | 用 `isAncestor` 判断 op 属于哪个 if，返回其编号 |
| `getProducerIfOpIndex` | 一组 producer op **必须全部落在同一个 if**，否则报错（保证依赖以 if 为单位成立） |
| `runFirst` | 判断本侧第一个 if 里没有 consumer（`isRunFirst`）：本计算块先执行时缓冲少占一轮，requiredBuffers 减 1 |
| 跨核复杂情况 | `comsumerIdx < producerIdx`（如 `C1→V1V2V3→C2`）跨核依赖难以处理，直接返回 `{1,1}` 不改变迭代次数 |
| `computeNewLoopUpperBound` | 按类型生成常量/ceil-div/加减乘，逐段计算新上界 |
| `cloneForOpWithNewUpperBound` | 克隆整个 for：复制属性、映射 IV/iter_args、克隆 body、建新 yield、`replaceAllUsesWith` 迁移结果 |
| `updateCntArgsAfterClone` | for 克隆后 ifOp 指针全变，把 `info->cntArgs` 从旧 if 迁移到新 if、旧计数变量换成新变量 |
| `updateMainLoopMaps` | cmap/vmap/infoMap 中旧 for 指针替换为新 for；所有旧 for 在最后统一 `erase` |
| `replaceForOpCounterInIfOps` | 把 if 内对主循环 `indVar` 的使用替换为 `cntArgs[ifOp]`（if 专用计数器） |
| `UpdateWhileLoopCondition` | while 专用：before 区 condition 的 def 链按 block 克隆（`whileBlockArgMap` 重映射 per-block arg），逐 if 生成条件后 `OR` 合并写回 |
| `whileBlockArgMap` | `{whileOp -> block_id -> {new_arg_idx: old_arg_idx}}`：before 区条件依赖的 per-block 参数到 after 区新 arg 的映射 |

---

## 三、逐行讲解

### 3.1 入口：runOnOperation（UpdateLoopIterTimes.cpp L1268-L1333）

```cpp
void UpdateLoopIterTimesPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  // step1: 按 main_loop id 分组（CUBE / Vector 分开）
  DenseMap<int, SmallVector<Operation *>> cmap, vmap;
  ret = GetMainLoopIdToLoopOpMap(module, cmap, vmap);
  // step2: 逐循环计算 IterationTimesInfo
  DenseMap<Operation *, IterationTimesInfo> infoMap;
  ret = ComputeMainLoopTimes(cmap, infoMap);
  ret = ComputeMainLoopTimes(vmap, infoMap);
  // step3: 同 id 两侧取最大值，克隆 for 扩展迭代
  ret = UpdateForLoopIteration(cmap, vmap, infoMap);
  // step4: if 内 indVar 替换为 if 计数器
  ret = replaceForOpCounterInIfOps();
  // step5: while 条件重写
  ret = UpdateWhileLoopCondition(cmap);
  ret = UpdateWhileLoopCondition(vmap);
}
```

**要点**：
- 每个 step 失败即 `setFallbackAttr(module, ERRCODE_FAILED)`（L1286/L1295/…），不做回滚，靠上游 fallback 短路兜底；
- 五个 step 严格线性：先分组 → 再算因子 → 再扩展 → 再换计数 → 再改 while；
- Step4/Step5 都必须在 Step3 之后：`cntArgs` 和 `whileBlockArgMap` 由前序 Pass 填好，且 Step3 克隆 for 后还需把映射迁移到新指针上（`updateCntArgsAfterClone`）。

### 3.2 Step1：GetMainLoopIdToLoopOpMap（L1030-L1074）

```cpp
module.walk([&](scope::ScopeOp scopeOp) {
  // getScopeType 判断 CUBE/VECTOR
  scopeOp.walk([&](Operation *op) {
    if (op->hasAttr(CVPipeline::kMainLoop)) {
      // 只允许 ForOp / WhileOp 作为 mainloop
      auto mainLoopId = op->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
      int id = mainLoopId.getInt();
      if (isCube)  cmap[id].push_back(op);
      else if (isVector) vmap[id].push_back(op);
    }
  });
});
```

**要点**：遍历所有 `scope::ScopeOp`，按核型归入 `cmap`/`vmap`，键是 `main_loop` id。同一 id 的 CUBE 主循环与 Vector 主循环就是同一流水线的两侧，后续必须同步扩展。

### 3.3 Step2：ComputeMainLoopTimes（L1078-L1134）

```cpp
for (auto &entry : loopMap) {
  for (Operation *loopOp : entry.second) {
    if (isa<scf::WhileOp>(loopOp)) continue;  // while 不在此计算
    scf::ForOp forOp = dyn_cast<scf::ForOp>(loopOp);

    // 从 info->cntArgs 收集本循环的 if
    for (auto &[ifOp, cntVal] : info->cntArgs) {
      if (ifOp->hasAttr(CVPipeline::kIf)) {
        auto parentOp = ifOp->getParentOp();
        if (parentOp->hasAttr(CVPipeline::kMainLoop) && isa<scf::ForOp>(parentOp)) {
          if (parentOp == forOp.getOperation()) {
            iterInfo.ifCount++;                 // if 数量
            iterInfo.ifOpsInThisFor.push_back(ifOp);
          }
        } else { /* parentOp 不是 mainloop → 结构错误，return -1 */ }
      }
    }
    auto [requiredBuffers, x] = calculateFactor(forOp);
    iterInfo.requiredBuffers = requiredBuffers;
    iterInfo.x = x;
    infoMap[loopOp] = iterInfo;
  }
}
```

**要点**：`ifCount` 与 `ifOpsInThisFor` 都来自 `info->cntArgs`（Step4 `UpdateConditionInfo` 填写的 if→计数器映射），要求**每个 if 的父 op 必须是带 `main_loop` 的 for**，否则判为结构非法。`calculateFactor` 是核心，见 3.4。

### 3.4 calculateFactor（L283-L370）：三类依赖因子合并

```cpp
std::pair<int, int> UpdateLoopIterTimesPass::calculateFactor(scf::ForOp forOp) {
  // Step1: 收集本循环所有 ifOp（带 ssbuffer.if），编号从 1 开始
  collectIfOps(forOp, ifOps, ifOpIndex);

  // Step2: 核内 / 跨核依赖因子
  bool hasIntraDeps = info->intraCoreDependentMap.count(forOp) && ...;
  bool hasCrossDeps = !info->crossCoreDependentMap.empty();
  if (!hasIntraDeps && !hasCrossDeps) return {1, 1};

  if (hasIntraDeps) {
    auto [intraRequiredBuffers, intraX] =
        calculateIntraDepsFactor(ifOps, ifOpIndex, intraDeps);
    maxRequiredBuffers = intraRequiredBuffers; maxX = intraX;
  }
  if (hasCrossDeps) {
    // 只保留 consumer 在本 for 内的条目 + 只统计含 sync 的 if
    filterCrossCoreMapByForOp(forOp, crossCoreMap);
    collectIfOps(forOp, filterIfOps, filterIfOpIndex, syncBlockFilter);
    auto [crossRequiredBuffers, crossX] =
        calculateCrossDepsFactor(forOp, filterIfOps, filterIfOpIndex, ...);
    // 分数交叉相乘比较，取较大分数
    if (crossRequiredBuffers * maxX > maxRequiredBuffers * crossX) {
      maxRequiredBuffers = crossRequiredBuffers; maxX = crossX;
    }
  }
  // Step3: tensor 迭代依赖因子，同样分数比较取 max
  ...
  return {maxRequiredBuffers, maxX};
}
```

**要点**：
- 没有任何依赖直接返回 `{1, 1}`（不扩展）；
- **分数交叉相乘**（L341、L363）：比较 `cross/maxX` 与 `max/maxX` 用 `cross * maxX > max * crossX`，避免整数除法截断/浮点误差；
- 跨核依赖必须先用 `syncBlockFilter` 过滤 if 集合——只有含 `sync_block_wait/set` 的 if 才是核间握手点，核间依赖只统计这些块的距离。

#### collectIfOps / findIfOpIndexInList / getConsumerIfOpIndex / getProducerIfOpIndex（L49-L424）

```cpp
// L49: 用 isAncestor 判断 op 属于哪个 if
static int findIfOpIndexInList(Operation *op, SmallVector<scf::IfOp> &ifOps, ...) {
  for (scf::IfOp ifOp : ifOps)
    if (ifOp->isAncestor(op)) return ifOpIndex[ifOp.getOperation()];
  return -1;
}
// L184: 收集带 ssbuffer.if 的 if，index 从 1 开始；支持过滤器
bool collectIfOps(scf::ForOp forOp, ..., IfOpFilter filter = defaultIfOpFilter) {
  forOp.walk([&](Operation *op) {
    if (op->hasAttr(CVPipeline::kIf)) {
      ifOp = dyn_cast<scf::IfOp>(op);
      if (filter(ifOp)) { ifOps.push_back(ifOp); ifOpIndexMap[ifOp] = index++; }
    }
  });
}
// L387: producer 组必须全部在同一个 if
static int getProducerIfOpIndex(...) {
  for (Operation *producerOp : producerOps) {
    int currentIndex = findIfOpIndexInList(producerOp, ifOps, ifOpIndex);
    if (currentIndex != producerIfOpIndex) {
      LDBG("ProducerOps are not in the same ifOp!"); return -1;
    }
  }
}
```

**要点**：if 编号从 1 开始（L189 `int index = 1`），因此 `requiredBuffers = consumerIdx - producerIdx + 1` 恰好覆盖"从 producer 块到 consumer 块需要跨几块缓冲"。producer 组必须同块，否则依赖无法以 if 为单位满足。

### 3.5 三个因子计算函数

#### calculateIntraDepsFactor（L426-L467）——核内

```cpp
for (auto &entry : deps) {
  Operation *consumerOp = entry.first;
  int x = producerOps.size();
  int ConsumerIdx = getConsumerIfOpIndex(consumerOp, ifOps, ifOpIndex);
  int producerIdx = getProducerIfOpIndex(producerOps, ifOps, ifOpIndex);
  if (ConsumerIdx <= producerIdx) {
    LDBG("producer is after the consumer!"); return {-1, -1};  // 非法：后生产者
  }
  int requiredBuffers = ConsumerIdx - producerIdx + 1;
  if (requiredBuffers * maxX > maxRequiredBuffers * x) { ... }
}
```

**要点**：核内 producer 必须在 consumer 之前（`ConsumerIdx > producerIdx`），否则无法流水，直接失败。`x = producerOps.size()` 表示该依赖有几个 producer op（缓冲份数）。

#### calculateCrossDepsFactor（L473-L568）——跨核

```cpp
// 找对侧（cube↔vector）同 id 的主循环及其 sync if
scf::ForOp otherSideForOp = findOtherSideMainloopAndIfOps(forOp, currentIsCube, currentIsVector, otherSideIfOps, otherSideIfOpIndexMap);
bool runFirst = isRunFirst(ifOps, crossDeps);   // 本侧第一个 if 里没有 consumer → 本块先执行

for (auto &entry : crossDeps) {
  Operation *consumerOp = entry.first;
  for (auto &producerOps : entry.second) {
    int x = producerOps.size();
    if (producerOps.size() == 1) x = 1;          // 特殊非对称 buffer
    int comsumerIdx = getConsumerIfOpIndex(consumerOp, ifOps, ifOpIndex);
    int producerIdx = getProducerIfOpIndex(producerOps, otherSideIfOps, otherSideIfOpIndexMap);
    if (comsumerIdx < producerIdx) {
      LDBG("there is complex case!"); return {1, 1};   // C1→V1V2V3→C2 不扩展
    }
    int requiredBuffers = comsumerIdx - producerIdx + 1;
    if (runFirst) requiredBuffers = requiredBuffers - 1;  // 先执行省一轮
    if (requiredBuffers * maxX > maxRequiredBuffers * x) { ... }
  }
}
```

**要点**：
- producer 在对侧核的 if 中定位（编号用对侧 `otherSideIfOpIndexMap`）；
- **runFirst -1**：本侧第一个 if 没有 consumer，说明本计算块先启动，流水线启动瞬间少占一轮缓冲；
- **复杂情况返回 `{1,1}`**：`comsumerIdx < producerIdx`（consumer 的块号在对侧 producer 之前）时跨核时序难以静态推导，保守不扩展（L539-L545）；
- `x` 特判：跨核一组 producer op 通常两侧各一个 op 共享一块缓冲（`size==2`），但**非对称 buffer**（只有一侧）时强制 `x=1`（L518-L520）。

#### calculateIterDepsFactor（L577-L647）——tensor 迭代依赖

```cpp
for (TensorIterArgIfOpRelation &relation : iterArgDepsVec) {
  scf::IfOp producerIfOp = relation.producer;
  int x = 1;                                   // 迭代依赖 producer 缓冲固定 1
  int producerIdx = ifOpIndex[producerIfOp.getOperation()];
  for (scf::IfOp consumerIfOp : relation.consumers) {
    int comsumerIdx = ifOpIndex[consumerIfOp.getOperation()];
    if (producerIdx <= comsumerIdx) { ... return {-1, -1}; }  // 非法
    int requiredBuffers = producerIdx - comsumerIdx + 1;      // 方向反向！
    if (requiredBuffers * maxX > maxRequiredBuffers * x) { ... }
  }
}
```

**要点**：tensor 迭代依赖是"**先消费后生产**"（consumer 在第 k 轮消费 producer 在第 k 轮产出的**旧值**，producer 本轮再生产新值），所以公式**反向**：`producerIdx - comsumerIdx + 1`（producer 必须在 consumer 之后），`x` 固定为 1（跨迭代的 tensor 只有一个 buffer）。

### 3.6 Step3：UpdateForLoopIteration（L1198-L1266）

```cpp
DenseSet<int> allIds;  // cmap ∪ vmap 的所有 id
for (int id : allIds) {
  int maxIfCount = 0, maxRequiredBuffers = 1, maxX = 1;
  SmallVector<Operation *> sameIdForOps;
  collectForOpsAndUpdateMax(cmap, id, sameIdForOps, maxIfCount, maxRequiredBuffers, maxX, infoMap);
  collectForOpsAndUpdateMax(vmap, id, sameIdForOps, maxIfCount, maxRequiredBuffers, maxX, infoMap);
  // 同 id 两侧循环统一用最大值扩展
  for (Operation *loopOp : sameIdForOps) {
    if (maxIfCount == 0) return -1;   // 没有 if 的主循环非法
    scf::ForOp newForOp = extendForOpIterationCount(
        oldForOp, maxIfCount, maxRequiredBuffers, maxX, mapper, iterInfo.ifOpsInThisFor);
    updateMainLoopMaps(loopOp, newForOp.getOperation(), cmap, vmap, infoMap);
    allForOps.push_back(loopOp);      // 记录旧 for 待删除
  }
}
for (Operation *loopOp : allForOps) loopOp->erase();   // 最后统一删旧
```

**要点**：
- **同 id 取最大**（`collectForOpsAndUpdateMax` L1136-L1166）：CUBE/Vector 同一流水线两侧必须同步扩展，否则 PIPE_S 握手的两个主循环迭代数不一致，流水线错位；
- 扩展后旧 for 先不删，等 `cntArgs`/maps 全部更新完（`updateMainLoopMaps` L1168-L1196 做指针迁移）再统一 `erase`，避免悬垂指针。

#### extendForOpIterationCount 三部曲（L812-L845）

```cpp
// Part1 computeNewLoopUpperBound (L650-L717)：
//   rangeDiff = ub - lb
//   iterCount = ceil(rangeDiff / step)
//   scaled   = iterCount * requiredBuffers
//   ceiled   = ceil(scaled / x)
//   newIter  = ceiled + ifCount
//   totalSteps = step * newIter
//   newUpperBound = lb + totalSteps
// 即 new_ub = lb + step * (ceil(iter * requiredBuffers / x) + ifCount)

// Part2 cloneForOpWithNewUpperBound (L720-L786)：
//   建新 for（同 lb / 新 ub / 同 step / 同 initArgs）→ 复制属性
//   → mapper.map(IV, iter_args) → 旧 block arg replaceAllUsesWith 新 arg
//   → 克隆 body（不含 terminator）→ 新 yield(mapper.lookupOrDefault)
//   → oldForOp.replaceAllUsesWith(newForOp results)

// Part3 updateCntArgsAfterClone (L789-L809)：
//   for 每个旧 ifOp：新 ifOp = mapper.lookupOrDefault(旧)
//   cntArgs.erase(旧) → cntArgs[新 ifOp] = mapper.lookupOrDefault(旧 cntVal)
```

**要点**：
- **为什么克隆整个 for 而非就地改 upperBound**：`upperBound` 被 body 内算术间接使用，且克隆后 `replaceAllUsesWith` 能一次性把主循环结果的使用方迁移到新循环，再统一删旧——比就地修改安全得多；
- **Part3 是正确性关键**：for 克隆后所有 ifOp 指针都换了对象，`info->cntArgs` 不迁移的话，Step4（`replaceForOpCounterInIfOps`）会在新 if 上查不到计数器；
- 新上界公式里 `+ ifCount`：给每个 if block 占用的一拍缓冲留余量（与 01 主文档 面试要点 8 一致）。

### 3.7 Step4：replaceForOpCounterInIfOps（L980-L1026）

```cpp
getOperation().walk([&](Operation *op) {
  if (op->hasAttr(CVPipeline::kMainLoop)) {
    if (isa<scf::WhileOp>(op)) return advance();   // while 跳过
    auto forOp = dyn_cast<scf::ForOp>(op);
    Value indVar = forOp.getInductionVar();
    forOp.walk([&](scf::IfOp ifOp) {
      if (ifOp->hasAttr(CVPipeline::kIf)) {
        Value cntVal = info->cntArgs[ifOp];        // 该 if 专用计数器
        ifOp.walk([&](Operation *op) {
          for (OpOperand &operand : op->getOpOperands())
            if (operand.get() == indVar) operand.set(cntVal);
        });
      }
    });
  }
});
```

**要点**：扩迭代后主循环归纳变量 `indVar` 的语义是"主循环第几次"，而 if 条件（`counter < upperBound` 等）必须用**该 if 自己的计数器**（`UpdateLoopOps` 追加的 per-block iter_arg）。把 if 内所有 `indVar` 的使用替换为 `cntArgs[ifOp]`，使条件判断按"本 block 第几次执行"进行。

### 3.8 Step5：UpdateWhileLoopCondition（L852-L976）

```cpp
for (auto &entry : mainLoopIdMap) {
  for (Operation *loopOp : entry.second) {
    if (!isa<scf::WhileOp>(loopOp)) continue;
    auto &blockArgMap = info->whileBlockArgMap[whileOp];  // {block_id: {new: old}}
    Region &beforeRegion = whileOp.getBefore();
    Operation *terminator = beforeBlock.getTerminator();  // scf.condition
    Value originalCondition = conditionOp.getCondition();
    SmallVector<Operation *> opsToClone = beforeBlock.without_terminator();

    whileOp.walk([&](scf::IfOp ifOp) {
      if (!ifOp->hasAttr(CVPipeline::kIf)) return advance();
      int blockId = ifOp->getAttrOfType<IntegerAttr>(kIf).getInt();
      auto &argMap = blockArgMap[blockId];
      // Step1: 建 arg 重映射 mapper：before 区 old_arg → after 区 new_arg
      for (auto &argEntry : argMap) mapper.map(beforeBlock.getArgument(old), beforeBlock.getArgument(new));
      // Step2: 在 conditionOp 前克隆整个 before 区 def 链
      for (Operation *op : opsToClone) {
        Operation *clonedOp = builder.clone(*op, mapper);
        if (op->getResult(i) == originalCondition) newCondition = clonedOp->getResult(i);
      }
      // Step3: 每个 if 生成的条件 OR 合并
      combinedCondition = builder.create<arith::OrIOp>(loc, combinedCondition, newCondition);
    });
    conditionOp.getConditionMutable().assign(combinedCondition);  // Step4 写回
  }
}
```

**要点**：
- while 没有静态 upperBound，无法"扩展迭代次数"，改走**动态条件重写**：`scf.condition` 的条件是 before 区算出的布尔值，且依赖 before 区的 per-block 参数；
- 每个 if block 用自己的 `whileBlockArgMap[blockId]`（new→old arg 索引）做重映射，克隆整条 def 链生成"本 block 独立的条件"，最后 **OR 合并**（任意一个 block 满足条件 while 就继续跑）；
- 这正是 01 主文档面试要点 4 说的"**for 是静态扩展、while 是动态条件重写**"。

---

## 四、流程总结

### 4.1 处理流水

```
UpdateLoopIterTimesPass（runOnOperation L1268）
  ├─► Step1 GetMainLoopIdToLoopOpMap (L1283)
  │        module 扫描 scope 核型 → cmap{id:[CUBE for/while]}, vmap{id:[Vector for/while]}
  ├─► Step2 ComputeMainLoopTimes (L1292, L1297)     逐 for 循环
  │        ├─► 收集 if: cntArgs 中父 op==本 for 的 if → ifCount / ifOpsInThisFor
  │        └─► calculateFactor (L1123)
  │              ├─► collectIfOps(默认过滤)            if 编号 1..N
  │              ├─► calculateIntraDepsFactor (L426)   req = consumerIdx - producerIdx + 1, x = producers.size()
  │              ├─► calculateCrossDepsFactor (L473)   对侧 sync if + runFirst-1 + 复杂情况{1,1}
  │              └─► calculateIterDepsFactor (L577)    反向 req = producerIdx - comsumerIdx + 1, x = 1
  │              三路分数交叉相乘取最大 → {requiredBuffers, x}
  ├─► Step3 UpdateForLoopIteration (L1305)
  │        ├─► collectForOpsAndUpdateMax              同 id CUBE/Vector 聚合 maxIfCount + max 分数
  │        └─► extendForOpIterationCount (L1233)
  │              ├─► computeNewLoopUpperBound          new_ub = lb + step*(ceil(iter*req/x) + ifCount)
  │              ├─► cloneForOpWithNewUpperBound       克隆 body + 复制 attrs + 迁移结果 uses
  │              └─► updateCntArgsAfterClone           cntArgs 旧 ifOp → 新 ifOp/新 cntVal
  │        └─► updateMainLoopMaps + 统一 erase 旧 for
  ├─► Step4 replaceForOpCounterInIfOps (L1313)        if 内 indVar 使用 → cntArgs[ifOp]
  └─► Step5 UpdateWhileLoopCondition (L1320, L1325)   while before 区按 block 克隆条件链 + OR 合并
```

### 4.2 与前后 Pass 的协作

- **输入**：`info->intraCoreDependentMap` / `crossCoreDependentMap` / `tensorIterArgDepsMap`（InitDependentMap、UpdateLoopOps 填）；`info->cntArgs`（UpdateConditionInfo 填）；`info->whileBlockArgMap`（ProcessArgs 填）；if 上 `ssbuffer.if`、主循环上 `main_loop` 属性；
- **产出**：for 主循环迭代次数按依赖距离扩展（CUBE/Vector 同步）、if 内条件改用块计数器、while 条件改为按 block OR 的动态条件；
- **后置**：`SeparateMemoryFromCompute`（L84）分离访存计算，直接消费扩好的循环结构。

---

## 五、面试要点

1. **为什么已经建好条件还要扩迭代？** 条件只解决"缓冲是否有货/有坑"（空间），而依赖距离 `requiredBuffers` 可能大于 producer 的缓冲数 `x`（时间）：consumer 在第 k+1 轮就要读 producer 第 k 轮的数据时中间轮次不够。扩迭代 = 放大循环，把依赖在时间上拉开，等价于"时间的缓冲"。

2. **新上界公式怎么推导？** `iter = ceil((ub-lb)/step)` 是原迭代次数；producer 有 `x` 块缓冲、依赖需要 `requiredBuffers` 块，放大 `requiredBuffers/x` 倍并向上取整；再 `+ifCount`（每个 if 一轮余量）。最终 `new_ub = lb + step * (ceil(iter*requiredBuffers/x) + ifCount)`（L650-L717 逐步算术）。

3. **三类依赖的 requiredBuffers 有什么不同？** 核内 `consumerIdx - producerIdx + 1`（先生产后消费，consumer 在后）；跨核 `consumerIdx - producerIdx + 1` 但 `runFirst` 时再 `-1`（先执行块省一轮），且 producer 在对侧核的**同步 if** 中定位；tensor 迭代依赖**先消费后生产**方向反转：`producerIdx - comsumerIdx + 1`（producer 必须块号更大）。

4. **为什么用分数交叉相乘而不是除法比较？** 直接比较 `requiredBuffers/x` 会有整数截断/浮点误差；`a/x > b/y ⟺ a*y > b*x` 全程整数运算，精确无误差（L341/L363/L560/L639）。

5. **`runFirst` 是什么、为什么 -1？** `isRunFirst` 检查本侧第一个 if 是否含 consumer（L145-L158）。本计算块先执行时，流水线启动瞬间第一轮 producer 写缓冲、但 consumer 还没开始消费，依赖距离天然少占一轮，`requiredBuffers` 减 1 后迭代扩展更紧凑。

6. **跨核"复杂情况"为什么不扩展？** `comsumerIdx < producerIdx`（如 `C1→V1V2V3→C2`：consumer 在 C1、producer 在对侧 V1V2V3）时，跨核时序交错无法静态推导出安全的扩展量，直接返回 `{1,1}` 保守处理（L539-L545）——宁可少优化不可算错。

7. **为什么克隆整个 for 而不是直接改 upperBound？** for 的结果被主循环外使用、body 内大量引用 IV/iter_args，克隆后用 `IRMapping` 统一映射 + `replaceAllUsesWith` 迁移结果，比就地改算术安全；且克隆后 `updateCntArgsAfterClone` 能顺便把 if 指针/计数变量全部换成新对象。旧 for 等所有映射更新完才统一 `erase`（L1258-L1264），避免悬垂。

8. **CUBE/Vector 为什么必须取最大值同步扩展？** 同一 `main_loop` id 两侧是同一流水线（`PIPE_S` 握手同步）。若两侧迭代数不一致，一方先跑完会导致 flag 握手错位、另一方读不到同步。`collectForOpsAndUpdateMax`（L1136）把同 id 两侧的 `ifCount`/`requiredBuffers/x` 聚合取最大，两侧用同一参数扩展（L1214-L1247）。

9. **Step4 为什么把 if 内的 indVar 换成 cntArgs？** 扩迭代后主循环归纳变量是"主循环第几次"；而 if 条件（`counter<upperBound`、flowOpt 下界/上界）必须按**本 block 第几次执行**判断（if 计数器在 `UpdateLoopOps` 追加、条件在 `UpdateConditionInfo` 构造）。不替换的话条件与循环次数语义错位（L998-L1018）。

10. **for 与 while 的处理差异？** for 有静态 upperBound：直接克隆换新上界（静态扩展）；while 无归纳变量/无静态上界：保留 before 区，把 `scf.condition` 的条件 def 链**按 block 克隆 + `whileBlockArgMap` 重映射**，每个 if 生成独立条件后 **OR 合并**写回（动态条件重写，L852-L976）。这与主文档面试要点 4 的结论一致。

11. **tensor 迭代依赖为什么 `x=1` 且方向反转？** tensor iter_arg 是跨迭代传递的单个张量值（一个 buffer），producer 数恒为 1；且语义是"consumer 先消费旧值、producer 再产新值"，所以 producer 块号必须大于 consumer 块号，`requiredBuffers = producerIdx - comsumerIdx + 1`（L601-L633）。
