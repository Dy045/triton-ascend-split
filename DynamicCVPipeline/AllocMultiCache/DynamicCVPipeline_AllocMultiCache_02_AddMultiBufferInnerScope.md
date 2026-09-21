# DynamicCVPipeline 子 Pass 专题：核内多缓冲（02）

> 覆盖 `AddMultiBufferInnerScopePass`（[AddMultiBufferInnerScope.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.cpp)）
>
> 流水线位置：`AllocMultiCache`（L82）编排的第 1 步，`AddMultiBufferOuterScope` 之前。
>
> 职责：处理 **Vector scope 内部主循环** 的**核内跨块 tensor 依赖**——在循环前分配 N 块 UB 缓冲，把依赖改造为"producer 写缓冲 + consumer 读缓冲"的轮询结构，实现 CUBE↔Vector 数据流的流水重叠。

---

## 一、这个专题在做什么

Vector scope 的主循环（for/while）内部，被 `ComputeBlockOpt` 拆成了多个 block_id 的计算块（CUBE 块、Vector 块）。CUBE 块算出的 tensor，会被 Vector 块消费；Vector 块算出的 tensor，也会被 CUBE 块消费。这些**跨块的 tensor 依赖**，在单缓冲下 producer 和 consumer 只能串行。

`AddMultiBufferInnerScope` 为每个跨块 tensor 依赖做三件事：

1. **克隆**：`tensor.empty + linalg.fill`、`alloc_tensor` 这类"块内独立产生"的依赖，直接克隆到每个 consumer 块（本地一份，无需缓冲轮换）；
2. **重物化**：tensor 根节点引出的 **scalar** 依赖跨块使用时，把 scalar 计算切片克隆到每个 consumer 块（本地重算，避免 scalar 跨块同步的复杂度）；
3. **多缓冲**：真正跨块、且无法克隆/重物化的 tensor 依赖，在循环前分配 N 块 UB 缓冲，注入 `scf.if` 轮询链——producer 按奇偶写第 `iter%N` 块，consumer 按奇偶读第 `iter%N` 块。

**核心挑战**：嵌套 if/while 结构的存在让"跨块判定"和"consumer 插入位置"变得复杂——这是本专题最重要的部分。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `MainLoop` | 主循环封装（`Common/Utils.h`）：`getOperation()` 拿 for/while op、`getBody()` 拿循环体、`isWhile()`、`iterCounter`（while 注入的迭代计数） |
| `InnerBlockInfo` | 一个块的信息：`blockId`（组 key，取块内第一个有结果的 op 的 result）+ `ops`（该块所有 op） |
| `depValueMap` | `blockKey → [跨块依赖值]`，记录每个块消费了哪些跨块依赖 |
| `depUserMap` | `depVal → [使用它的 op]`，依赖的用户反向索引 |
| `BufferMap` | `depVal → [BufferPair...]`，每个依赖的 N 块缓冲（BufferPair = alloc 结果对） |
| `BufferPair` | `std::pair<Value, Value>`，缓冲的 alloc 结果（`memory_space_cast` 后的通用地址空间值） |
| `getOutermostSsbufferId` | 从 op 向上找**最外层**带 block_id 的祖先的 id；遇到 `getNumRegions()>=2` 的 op（如 scf.if）返回其 block_id。保证嵌套 op 归属外层块 |
| `kIntraDeps`（`ssbuffer.intra_deps`） | 核内依赖标记 `[groupId, 0|1]`：`1`=producer（写缓冲 copy），`0`=consumer（读缓冲 to_tensor/if 包装） |
| `kIntraBuffer` | 标记多缓冲引入的 `scf.if`/`copy`/`to_tensor` op |
| `kDepMark`（`ssbuffer.dep_mark`） | 标量依赖标记：producer 和跨块 consumer 打同一个 mark，供后续 Pass 识别 |
| `kBlockId` | 新注入 op 必须继承所在块 id，保证块归属一致 |
| `kIterCounter` | while 主循环的迭代计数 loop-carried 变量（i32），`insertWhileCounterOps` 在 body 末尾 `+1` |
| `BufferCountManager` | `getBufferCountByType(IntraCore)` 拿核内缓冲块数 N |
| `cloneDepsToConsumers` | 通用克隆框架：模式检查 + 克隆回调，把匹配的依赖克隆到每个跨块 consumer |

---

## 三、逐行讲解

### 3.1 主循环收集（L63-L88）

```cpp
// 单块内收集 main_loop op
static int collectMainLoopsInBlock(Block &block, SmallVector<Operation *> &mainLoops) {
  int count = 0;
  for (Operation &op : block)
    if (isMainLoopOp(&op)) { mainLoops.push_back(&op); count++; }
  return count;
}
// 递归收集（跨 region）
static int collectMainLoopsRecursively(Region &region, SmallVector<Operation *> &mainLoops) {
  int totalCount = 0;
  for (Block &block : region) {
    totalCount += collectMainLoopsInBlock(block, mainLoops);
    for (Operation &op : block)
      for (auto &nestedRegion : op.getRegions())
        totalCount += collectMainLoopsRecursively(nestedRegion, mainLoops);
  }
  return totalCount;
}
```

**要点**：`isMainLoopOp`（`PlanComputeBlock` 打的 `kMainLoop` 属性）识别主循环。递归收集覆盖嵌套 region（scope 内所有层级的 for/while）。

### 3.2 最外层 block_id 判定：getOutermostSsbufferId（L95-L110）

```cpp
static std::optional<int64_t> getOutermostSsbufferId(Operation *op) {
  std::optional<int64_t> result;
  for (Operation *current = op; current; current = current->getParentOp()) {
    if (current->hasAttr(kMainLoop))                    // 到主循环边界
      return result.has_value() ? result : -1;
    if (current->getNumRegions() >= 2)                  // 多 region op（scf.if/while）→ 归属其 block_id
      return getOpBlockId(current);
    if (auto curId = getOpBlockId(current); curId.has_value())
      result = curId;                                   // 记住已看到的最深 id，外层可覆盖
  }
  return result;
}
```

**核心语义**：**跨块判定必须用"最外层归属"而不是"自身 block_id"**。例：`scf.if` 在 block 4 内，其 then 分支里的 subview 在 block 3 —— 若直接用自身 block_id（3），会把"同一 if 内的操作"误判为"跨块消费"；用 `getOutermostSsbufferId`（找多 region 祖先 if 的 block_id = 4）则正确归位。

**边界规则**：
- 遇到 `kMainLoop` 祖先 → 返回已见的最深 id（或 -1 表示无 id）——主循环是归属边界；
- 遇到 `getNumRegions() >= 2` 的 op（scf.if、scf.while、scf.for）→ 立即返回该 op 的 block_id（多 region op 是"块容器"，内部 op 都归属它）；
- 普通 op → 记录其 block_id 作为候选，继续向外（外层容器可覆盖）。

### 3.3 块信息收集：collectInnerBlockInfo（L331-L394）

```cpp
static int collectInnerBlockInfo(const MainLoop &loop,
                                 DenseMap<Value, InnerBlockInfo> &blocks,
                                 DenseMap<Value, SmallVector<Value>> &depValueMap,
                                 SmallVector<Operation *> &allOps, bool &i1Found) {
  depValueMap.clear();
  Block *body = loop.getBody();
  collectNestedOps(body, allOps);                       // 收集循环体所有嵌套 op

  llvm::MapVector<int, SmallVector<Operation *>> opsById;
  if (groupOpsBySsbufferId(allOps, opsById) != 0) return -1;  // 按 block_id 分组
  if (opsById.empty()) return 0;                        // 无块 → 无事可做

  // 建立 output → block_id 映射
  DenseMap<Value, int> outputToBlockId;
  for (auto &p : opsById)
    for (Operation *op : p.second)
      for (auto res : op->getResults())
        outputToBlockId[res] = p.first;

  // 对每个块：keyOp 取第一个有结果的 op，groupKey = keyOp 的 result(0)
  for (auto &p : opsById) {
    ...keyOp 查找...
    InnerBlockInfo bi; bi.blockId = groupKey; bi.ops = p.second;
    blocks[groupKey] = bi;
    for (Operation *op : bi.ops)
      for (Value operand : op->getOperands())
        collectDepValue(operand, body, op, outputToBlockId, depValueMap, groupKey, i1Found);
  }

  // 补充：多 region consumer 的 yield 操作数也收集（yield 消费了跨块依赖）
  for (auto &blockPair : blocks)
    for (Operation *op : blockPair.second.ops)
      forEachYieldedCrossBlockDep(op, outputToBlockId, [&](Value operand) {
        if (!llvm::is_contained(depValueMap[blockKey], operand))
          depValueMap[blockKey].push_back(operand);
      });
  return 0;
}
```

**要点**：
- `groupOpsBySsbufferId`（L253-L288）按 `kBlockId` 分组并**去重**（多结果 op 如 scf.if 会被 `opsByValue` 插入多次，去重避免重复 dep_mark），无结果但有 block_id 的 op 也注册；
- 依赖收集用 `collectDepValue`（L190-L224）：BlockArgument（body 参数，主循环的 iter_arg）直接记录；op 结果查 `outputToBlockId`，用 `getOutermostSsbufferId` 判断**是否跨块**（同块跳过）；**i1 tensor 依赖置 `i1Found`**（跨块 i1 tensor 无法多缓冲，触发 fallback）；
- 多 region consumer（scf.if/while）的 **yield 操作数**也是依赖消费点，`forEachYieldedCrossBlockDep`（L307-L324）专门补采（yield 值被跨块依赖、且 op 本身不是直接操作数时）。

### 3.4 依赖用户索引：buildDepUserMap（L430-L451）

```cpp
DenseMap<Value, SmallVector<Operation *>> buildDepUserMap(...) {
  // First pass: 块内所有 op 的操作数
  for (auto &p : blocks)
    for (Operation *op : p.second.ops)
      for (Value operand : op->getOperands())
        depUserMap[operand].push_back(op);
  // Second pass: 不在块里的 yield op → 把父多 region op 当 consumer
  for (Operation *op : allOps)
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op))
      processYieldNotInBlocks(yieldOp, blocks, depUserMap);  // L410-L428
  return depUserMap;
}
```

**要点**：`processYieldNotInBlocks`（L410-L428）处理 `depVal` 只出现在 yield、不作为直接操作数的情况——把父多 region op（≥2 regions）登记为该 depVal 的 consumer（`depUserMap[operand].push_back(parentOp)`）。`isYieldAlreadyProcessed`（L397-L405）防重复。

### 3.5 依赖值筛选（L453-L565）

```cpp
// ① empty+fill 模式：depVal 的 def 是 fill，其 outs 是 tensor.empty 或 alloc_tensor
static bool isEmptyFillPattern(Value depVal) { ... }

// ② alloc_tensor 模式
static bool isAllocTensorPattern(Value depVal) {
  return isa_and_nonnull<bufferization::AllocTensorOp>(depVal.getDefiningOp());
}

// ③ 需要分配缓冲的 value 列表：非 empty/alloc_tensor/已克隆的 ShapedType depVal
SmallVector<Value> collectBufferValues(depValueMap, clonedDepVals) { ... }

// ④ 需要 dep_mark 的 scalar 依赖（含 tensor.empty，父 op 是主循环）
SmallVector<Value> collectScalarDeps(depValueMap, depUserMap) {
  // 对每个 depVal：非 BlockArgument、非 ShapedType（或 empty 且父是 main_loop）、
  // 有跨块 consumer（getOutermostSsbufferId ≠ producerId）→ 加入 scalarValueList
  ...
}
```

**要点**：
- **三种依赖各自走不同路径**：`empty+fill` → 克隆；`alloc_tensor` → 克隆；其他 tensor → 分配多缓冲；scalar → dep_mark 标记；
- `collectScalarDeps` 中 `tensor::EmptyOp` 被当 scalar 处理（加 dep_mark 而非缓冲），且要求其父 op 是 main_loop（循环内定义的 empty 才能标记）；
- `isEmptyFillPattern` 同时支持 `tensor.empty` 和 `bufferization.alloc_tensor` 作为 fill 的 outs（新/旧两种形式）。

### 3.6 tensor 根标量依赖重物化（L587-L746）

**背景**：一个跨块 tensor 依赖的根节点（如 `tensor.extract`/`arith.extf`）会引出标量值，这些标量若跨块使用，无法用 tensor 多缓冲覆盖（缓冲是整块 tensor 级别）。解决方案：**把标量计算链克隆到每个 consumer 块，本地重算**。

```cpp
// ① 深度优先构建 scalar 计算切片：递归停到 tensor 操作数
static void buildScalarSlice(Value root, const MainLoop &mainLoop,
                             SmallVector<Operation *> &sliceInOrder, ...) {
  Operation *def = root.getDefiningOp();
  if (!def || !isOpInMainLoop(def, mainLoop)) return;   // 循环外/无 def → 停
  if (!visited.insert(def).second) return;              // 防环
  for (Value dep : collectOpDependencies(def)) {        // 操作数 + region 捕获值
    if (isa<TensorType>(dep.getType())) { boundaryTensors.insert(dep); continue; }  // tensor 边界
    ...
    buildScalarSlice(dep, mainLoop, ...);               // 递归
  }
  sliceInOrder.push_back(def);                          // 后序 → 依赖在前
}

// ② 把切片克隆到每个跨块 consumer 块，并改写用户
static bool rematerializeScalarDep(Value root, int producerId, ...) {
  // 按 consumer block_id 分组（跳过 producer 同块）
  llvm::MapVector<int, SmallVector<Operation *>> usersByBlock;
  for (Operation *user : root.getUsers()) {
    ...getAncestorInBlock(user, body)...                // 循环体内直接祖先
    auto userId = getOpBlockId(user) 或 getOpBlockId(bodyAnc);
    if (!userId || *userId == producerId) continue;     // 同块跳过
    usersByBlock[*userId].push_back(user);
  }
  for (auto &entry : usersByBlock) {
    // 插到最早 consumer 前，克隆 slice，全部打 userBlockId
    OpBuilder builder(insertPt);
    IRMapping map;
    for (Operation *op : sliceInOrder) {
      Operation *cloned = builder.clone(*op, map);
      cloned->walk([&](Operation *o) { o->setAttr(kBlockId, ...userBlockId); });
    }
    Value clonedRoot = map.lookupOrDefault(root);
    for (Operation *user : users) user->replaceUsesOfWith(root, clonedRoot);  // 改写
  }
}

// ③ 扫描主循环找候选 root：跨块 scalar 操作数，其 def 在主循环内
static void rematerializeTensorRootedScalarDeps(const MainLoop &mainLoop) {
  for (Operation *op : allOps) {
    if (无 block_id) continue;
    for (Value operand : op->getOperands()) {
      if (isa<ShapedType>(operand.getType())) continue;  // 只要 scalar 操作数
      defOp 在主循环内 且 producerId ≠ userId → roots.insert(operand);
    }
  }
  for (Value root : roots) {
    buildScalarSlice(root, ...);
    if (boundaryTensors.empty()) continue;   // 纯 scalar/memref 链 → 保持原 dep_mark 处理
    rematerializeScalarDep(root, ...);
  }
}
```

**要点**：
- **重物化只针对 tensor 根的标量链**（`boundaryTensors` 非空）：纯 scalar 链走原 dep_mark 路径；
- 克隆整条 slice 后**全部打上 consumer 块的 block_id**，保证块归属；
- `getAncestorInBlock`（L619-L624）找"直接属于循环体的祖先 op"，用于定位插入点和判定跨块。

### 3.7 迭代计数获取：getIterCount / computeBufferIndex（L748-L832, L1017-L1036）

```cpp
static Value getIterCount(OpBuilder &builder, const MainLoop &loop, Location loc, ...) {
  if (loop.isWhile())
    return loop.iterCounter;                 // while：直接复用注入的计数器（3.10 节）
  auto forOp = cast<scf::ForOp>(loop.getOperation());
  Value iv = forOp.getInductionVar(), lb = forOp.getLowerBound(), step = forOp.getStep();
  // 优化：lb==0 且 step==1 → 直接用 iv
  //      lb==0 且 step!=1 → iv / step
  //      通用 → (iv - lb) / step
  ...
  // 类型归一化到 i32（index 用 IndexCast，窄/宽整型用 Ext/Trunc）
  ...
}

// 缓冲索引：iterCount % N
static Value computeBufferIndex(OpBuilder &builder, const MainLoop &loop, Location loc, int N, ...) {
  Value iterCount = getIterCount(...);
  Value Nval = builder.create<arith::ConstantIntOp>(loc, N, 32);
  Value bufIdx = builder.create<arith::RemSIOp>(loc, iterCount, Nval);
  // 新 op 打 kBlockId
  return bufIdx;
}
```

**要点**：
- while 直接复用 `loop.iterCounter`（3.10 的注入计数器），for 用归纳变量推导 `(iv - lb) / step`；
- 结果归一化到 i32（多缓冲索引运算统一位宽）；
- 新 op 打 `kBlockId` 保证归属。

### 3.8 if-else 链构建：buildIfChain（L851-L1014）

```cpp
// N==2：简单嵌套（idx==0 → buf[0]，else → buf[1]）
static int buildIfChainTwoBuffers(...) {
  Value zero = 常量 0; Value firstCond = (indexVal == 0);
  auto firstIf = builder.create<scf::IfOp>(loc, types, firstCond, true, true);
  // then: createOpFn(buffers[0].second) + yield
  // else: createOpFn(buffers[1].second) + yield
}

// N>2：if-else-if 链（idx==0 → buf[0]，idx==1 → buf[1]，... else buf[N-1]）
static int buildIfChainMultiBuffers(...) {
  // rootIf: idx==0 → buf[0]
  // else 里嵌 nestedIf: idx==i → buf[i]，直到最后 else 用 buf[N-1]
  // 每层 else 结束 yield 传递 nestedIf.getResults()
}

static int buildIfChain(...) {
  if (N == 2) return buildIfChainTwoBuffers(...);
  return buildIfChainMultiBuffers(...);
}
```

**要点**：
- `createOpFn`（回调）在 then/else 分支里创建**实际操作**（producer 是 `hivm.copy`，consumer 是 `bufferization.to_tensor`）；
- `yieldFn` 返回分支结果（有结果时），无结果（producer copy）则空 yield；
- 所有新 op（常量、比较、if）打 `kBlockId`；`newOps` 收集全部新 op 供后续统一打标。

### 3.9 producer/consumer 逻辑插入（L1038-L1144）

```cpp
// producer：depVal 写进缓冲
static SmallVector<Operation *> insertProducerLogic(OpBuilder &builder, Value depVal,
                                                    SmallVector<BufferPair> &buffers, ...) {
  int N = buffers.size();
  if (N == 1) {                              // 单缓冲：直接 copy
    Operation *producerOp = builder.create<hivm::CopyOp>(loc, {}, depVal, buffers[0].second);
    if (groupId >= 0) producerOp->setAttr(kIntraDeps, {groupId, crossCoreProducerId});
    return newOps;
  }
  Value bufIdx = computeBufferIndex(builder, loop, loc, N, &newOps);
  buildIfChain(builder, loc, bufIdx, buffers, newOps, outIfOps,
               [&](b, l, buffer) { return b.create<hivm::CopyOp>(l, {}, depVal, buffer); },
               nullptr);                     // 无结果 → 空 yield
  if (groupId >= 0)
    for (Operation *op : newOps)
      if (isa<hivm::CopyOp>(op)) op->setAttr(kIntraDeps, {groupId, crossCoreProducerId});
  return newOps;
}

// consumer：从缓冲 to_tensor 读回
static int insertConsumerLogic(OpBuilder &builder, Value depVal,
                               SmallVector<BufferPair> &buffers, ..., int groupId = -1, int blockId = -1) {
  int N = buffers.size();
  if (N == 1) {                              // 单缓冲：直接 to_tensor
    Operation *consumerOp = handleSingleBufferConsumer(builder, loc, buffers);
    if (groupId >= 0) consumerOp->setAttr(kIntraDeps, {groupId, 0});
    return 0;
  }
  Value readIdx = computeBufferIndex(builder, loop, loc, N, &newOps, blockId);
  auto tensorType = RankedTensorType::get(memrefShape, elemType);
  buildIfChain(builder, loc, readIdx, buffers, newOps, outIfOps,
               [&](b, l, buffer) { return createToTensorOp(b, l, tensorType, buffer); },
               [&](b, l, op) { return cast<ToTensorOp>(op).getResult(); },
               resultTypes, blockId);
  if (groupId >= 0 && !outIfOps.empty())
    outIfOps.front()->setAttr(kIntraDeps, {groupId, 0});   // consumer 标记打在最外层 if
  return 0;
}
```

**要点**：
- **producer 标记**（`[groupId, 1]`）打在真正的写 op（`hivm.copy`）上；**consumer 标记**（`[groupId, 0]`）打在 `scf.if` 包装或 `to_tensor` 上（单缓冲时直接在 to_tensor 上）；
- `crossCoreProducerId` 是常量（1）——producer 角色；
- consumer 的 if 链有结果（tensor 类型），分支 yield 返回 to_tensor 结果，外层 `outIfOps` 的第一个 if 就是"选缓冲的开关"。

### 3.10 while 迭代计数器注入：setupWhileIterArgCounter + insertWhileCounterOps（L1898-L2015）

```cpp
// ① 重建 whileOp，追加 i32 counter（初值 0）
static std::pair<Value, scf::WhileOp> setupWhileIterArgCounter(const MainLoop &loop, OpBuilder &builder) {
  auto oldWhile = cast<scf::WhileOp>(loop.getOperation());
  Value zero = preBuilder.create<arith::ConstantIntOp>(loc, 0, 32);   // 插在 oldWhile 前
  SmallVector<Value> newInits(oldWhile.getInits().begin(), oldWhile.getInits().end());
  newInits.push_back(zero);                                            // 追加 counter 初值
  SmallVector<Type> newResultTypes(oldWhile.getResultTypes().begin(), oldWhile.getResultTypes().end());
  newResultTypes.push_back(i32Type);                                   // 追加 counter 结果类型
  Value counterIterArg;
  auto newWhile = cb.create<scf::WhileOp>(loc, newResultTypes, newInits,
    [&](bb, bl, iterArgs) { buildBeforeRegion(oldWhile, bb, bl, iterArgs); },   // 克隆 before
    [&](ab, al, iterArgs) { buildAfterRegion(oldWhile, ab, al, iterArgs, counterIterArg); }); // 克隆 after
  for (auto attr : oldWhile->getAttrs()) newWhile->setAttr(attr.getName(), attr.getValue());  // 保持属性
  newWhile->setAttr(kIterCounter, cb.getUnitAttr());                  // 标记"已注入计数器"
  for (unsigned i = 0; i < oldWhile.getNumResults(); ++i)
    oldWhile.getResult(i).replaceAllUsesWith(newWhile.getResult(i));  // 替换 uses
  oldWhile.erase();
  return {counterIterArg, newWhile};
}

// ② after-region 构建：旧 op 克隆 + yield 追加 counterIterArg 占位
static void buildAfterRegion(scf::WhileOp oldWhile, OpBuilder &ab, Location al,
                             ValueRange iterArgs, Value &counterIterArgOut) {
  counterIterArgOut = iterArgs[numOrig];          // 记录 counter 迭代参数
  // 克隆旧 after 所有非 yield op，yield 操作数追加 counterIterArgOut 占位
  ...
}

// ③ body 末尾插入 counter+1，替换 yield 占位
static void insertWhileCounterOps(const MainLoop &mainLoop) {
  // 找 counter 消费 op 的 block_id（第一个用 iterCounter 的 op）
  // 在 doBlock 里找最后一个带该 block_id 的 op，插到其后：
  Value one = ConstantIntOp(1, bitWidth);
  Value nextCounter = AddIOp(counter, one);
  // yield 中 counterIterArg 占位替换为 nextCounter
  ...
}
```

**要点**：
- `buildBeforeRegion`（L1834-L1862）：克隆旧 before 的所有非 condition op（`IRMapping` 映射旧参数到新 iterArgs），重建 `scf.condition`（操作数 + 追加 counter 前向参数）；
- `buildAfterRegion`（L1867-L1895）：克隆旧 after 的非 yield op，yield 操作数**追加 counterIterArg 占位**（真正的 `+1` 由 `insertWhileCounterOps` 稍后插入替换）；
- `insertWhileCounterOps`（L1951-L2015）：必须**在 `processTensorDependencies` 之后**执行——此时 counter 的消费 op（轮询 if 链）已生成，才能定位 counter 的 block_id 归属；
- 计数更新 `kIterCounter` 标记，op 继承消费块 block_id。

### 3.11 核心编排：addInnerMultiBuffer（L2017-L2159）

```cpp
static int addInnerMultiBuffer(MainLoop mainLoop, OpBuilder &builder,
                               scope::ScopeOp vectorScope, int &groupId,
                               bool &i1Found, bool &memrefFound) {
  // ① while 主循环且 bufNum>1：先注入迭代计数器（bufNum==1 跳过避免死 iter_arg）
  if (mainLoop.isWhile()) {
    int bufNum = BufferCountManager(...).getBufferCountByType(IntraCore);
    if (bufNum > kBufferCountOne) {
      auto [counter, newWhile] = setupWhileIterArgCounter(mainLoop, globalBuilder);
      mainLoop.op = newWhile; mainLoop.body = newWhile.getAfterBody(); mainLoop.iterCounter = counter;
    }
  }

  // ② 收集块信息 + 依赖（i1Found 检查）
  if (collectInnerBlockInfo(mainLoop, blocks, depValueMap, allOps, i1Found) != 0) return -1;
  if (blocks.empty()) return -1;
  if (hasMemrefDepValue(depValueMap)) { memrefFound = true; return -1; }   // memref 依赖不支持

  // ③ Phase 1：初始 depUserMap + 克隆 empty+fill
  DenseMap<Value, SmallVector<Operation *>> initialDepUserMap = buildDepUserMap(...);
  DenseSet<Value> phase1ClonedDepVals;
  cloneEmptyFillsInBlocks(mainLoop, blocks, depValueMap, initialDepUserMap, globalBuilder, &phase1ClonedDepVals);

  // ④ tensor 根标量依赖重物化
  rematerializeTensorRootedScalarDeps(mainLoop);

  // ⑤ Phase 2：重新收集（clone/重物化产生的新跨块引用）
  blocks.clear(); depValueMap.clear(); allOps.clear();
  collectInnerBlockInfo(mainLoop, blocks, depValueMap, allOps, i1Found);
  if (i1Found) return -1;                                   // Phase 2 也可能发现 i1
  if (blocks.empty()) return -1;
  // 丢弃 clone 引入的 memref 依赖（Phase 2 产物）
  for (auto &p : depValueMap) llvm::erase_if(p.second, [&](Value v) { if (isa<MemRefType>(...)) return true; });

  auto depUserMap = buildDepUserMap(blocks, allOps, depValueMap);

  // ⑥ 克隆 alloc_tensor 到 consumer 块
  cloneAllocTensorsInBlocks(mainLoop, blocks, depValueMap, depUserMap, globalBuilder);

  // ⑦ 收集需要缓冲的 value，循环前分配 N 块 UB 缓冲
  auto valueList = collectBufferValues(depValueMap, phase1ClonedDepVals);
  auto bufferMap = insertBuffersBeforeLoop(mainLoop, valueList, builder, groupId);

  // ⑧ 标量依赖打 dep_mark
  auto scalarValueList = collectScalarDeps(depValueMap, depUserMap);
  markScalarDeps(scalarValueList, depUserMap, globalBuilder, 1);

  // ⑨ 处理跨块 tensor 依赖（producer 写 + consumer 读）
  processTensorDependencies(mainLoop, blocks, depValueMap, depUserMap, bufferMap, globalBuilder, groupId, phase1ClonedDepVals);

  // ⑩ while 主循环：插入 counter+1
  insertWhileCounterOps(mainLoop);
  return 0;
}
```

**要点**：
- **两阶段依赖收集**是核心设计：Phase 1 克隆 empty+fill 后，克隆 fill 的 `ins` 链可能引入 producer 侧 tensor 引用；标量重物化也产生新链；**Phase 2 重跑收集**才能让新引用进入多缓冲管线；
- `cloneEmptyFillsInBlocks`（L1626-L1668）用通用 `cloneDepsToConsumers`（L1546-L1621）：按 consumer block_id 分组、跳过 producer 块、克隆 `empty/fill/ins` 链并全部打 consumer 块 block_id；
- `insertBuffersBeforeLoop`（L1769-L1809）：循环前（parent block 中 loop 之前）为每个 depVal 分配 N 块 `AddressSpace::UB` memref（`memref.alloc` + `memref.memory_space_cast` 到通用空间 0u）；
- `processTensorDependencies`（L1691-L1767）：跳过 empty/alloc_tensor/已克隆/BlockArgument，校验 `allUsersSameBlock`（全同块则无需多缓冲），否则调 `processDepVal` 并 `groupId++`。

### 3.12 单个依赖处理：processDepVal（L1410-L1543）

```cpp
static int processDepVal(Value depVal, const MainLoop &loop, BufferMap &bufferMap, ...) {
  SmallVector<BufferPair> &buffers = bufferMap[depVal];
  // 读取 module 级 kInsertionOptimization 属性（Python 侧可动态开关插入位置优化）
  bool enableOpt = mod->hasAttr(CVPipeline::kInsertionOptimization);

  // ===== producer 插入 =====
  Operation *producerAnchor = depDefinedOp;
  if (enableOpt) {   // 优化开：插到 producer 块最后（depDefinedOp 所在 block_id 的最后一个 op 后）
    if (auto prodId = getOpBlockId(depDefinedOp); prodId.has_value())
      if (Operation *lastInRegion = findLastOpWithBlockIdInBlock(depDefinedOp, *prodId))
        producerAnchor = lastInRegion;
  }
  producedBuffers.setInsertionPointAfter(producerAnchor);
  auto producerNewOps = insertProducerLogic(producedBuffers, depVal, buffers, loop, groupId);
  addBlockAttrForOps(producerNewOps, producerId, globalBuilder);   // 打 producer 块 block_id
  // 多缓冲：scf.if 打 kIntraBuffer；单缓冲：所有新 op 打
  ...

  // ===== consumer 插入（两种路径）=====
  DenseMap<int, SmallVector<Operation *>> opsByBlockId;
  for (Operation *depUser : depUsers) {
    auto userBlockId = getOutermostSsbufferId(depUser);
    if (!userBlockId.has_value() || *userBlockId == producerId) continue;   // 跳过 producer 块
    if (isMultiRegionConsumerFromYield(depUser, depVal)) {
      // 多 region consumer（yield 消费 depVal）：在 region 里插缓冲选择 if
      processMultiRegionAllYields(consumedBuilder, depVal, buffers, loop, depUser, *userBlockId, groupId);
    } else {
      opsByBlockId[*userBlockId].push_back(depUser);   // 普通 consumer：按块批量
    }
  }

  // 按 Block 再分组（getOutermostSsbufferId 可能合并不同 region 的 op 到同一 block_id，
  // 但 scf.if 的 then/else 结果只在各自 region 可见 → SSA dominance 约束必须按 Block 处理）
  for (auto &blockPair : opsByBlockId) {
    DenseMap<Block *, SmallVector<Operation *>> opsByBlock;
    for (Operation *op : opsInBlock) opsByBlock[op->getBlock()].push_back(op);
    for (auto &regionPair : opsByBlock) {
      Operation *firstOp = opsInRegion.front();
      Operation *consumerAnchor = firstOp;
      if (enableOpt) {   // 优化开：插到 consumer 块第一个 op 前
        ...findFirstOpWithBlockIdInBlock...
      }
      consumedBuilder.setInsertionPoint(consumerAnchor);
      processNormalConsumerBlock(consumedBuilder, depVal, buffers, loop, opsInRegion, userBlockId, groupId, globalBuilder);
    }
  }
  return 0;
}
```

**要点**：
- **插入位置优化**（`kInsertionOptimization`）：producer 插到 producer 块**末尾**（`findLastOpWithBlockIdInBlock`，L1377-L1390），consumer 插到 consumer 块**开头**（`findFirstOpWithBlockIdInBlock`，L1395-L1407）——保证缓冲写/读紧跟块边界，最大化重叠；关闭时保持"紧贴 depVal 定义后/紧贴 user 前"；
- `isMultiRegionConsumerFromYield`（L1276-L1286）：depUser 是多 region op 且 depVal 不是其直接操作数（是 yield 消费）→ 走 `processMultiRegionAllYields`（L1334-L1372）：遍历 region 1 起的各 region，在 region 头插缓冲选择 if，替换 yield 操作数；
- **按 Block 再分组**是关键正确性细节：`getOutermostSsbufferId` 会把 scf.if then/else 里不同 region 的 op 归到同一个 block_id，但一个缓冲选择 if 的结果只在**自己的 region** 可见——跨 region 共享会破坏 SSA dominance，所以必须再按 `Block*` 细分；
- `processNormalConsumerBlock`（L1290-L1327）：一个块只生成**一份**缓冲选择 if（`insertConsumerLogic`），块内所有 op 共享其结果，替换所有 `depVal` 操作数。

### 3.13 主循环扫描 + 入口：runOnOperation（L2170-L2260）

```cpp
void AddMultiBufferInnerScopePass::runOnOperation() {
  auto module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;

  // 扫描已有 kIntraDeps 的最大 groupId（避免与 InterCoreTransferAndSync 的 C2C 冲突）
  int groupId = 0;
  module.walk([&](Operation *op) {
    if (auto attr = op->getAttrOfType<ArrayAttr>(CVPipeline::kIntraDeps))
      if (!attr.empty())
        if (auto intAttr = dyn_cast<IntegerAttr>(attr[0]))
          groupId = std::max(groupId, static_cast<int>(intAttr.getInt()) + 1);
  });

  auto walkResult = module.walk([&](scope::ScopeOp scope) -> WalkResult {
    // Step 1: 必须有 coreType 属性
    auto coreTypeAttr = scope->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr) return WalkResult::advance();
    // Step 2: 必须 VECTOR scope
    hivm::TCoreType coreType = coreTypeAttr.getTcoretype();
    if (coreType != hivm::TCoreType::VECTOR) return WalkResult::advance();
    // Step 3: 收集 scope 内所有 main_loop
    SmallVector<Operation *> mainLoops;
    int foundCount = collectMainLoopsRecursively(scope.getBodyRegion(), mainLoops);
    if (foundCount == 0) return WalkResult::advance();
    // Step 4: 逐个处理 main_loop
    for (Operation *loopOp : mainLoops) {
      MainLoop mainLoop(loopOp);
      if (findNestedMainloop(mainLoop))          // 嵌套 main_loop 不允许
        return WalkResult::interrupt();
      bool i1Found = false, memrefFound = false;
      int ret = addInnerMultiBuffer(mainLoop, builder, scope, groupId, i1Found, memrefFound);
      if (i1Found)    { setFallbackAttr(module, ERRCODE_IGNORED); return interrupt(); }
      if (memrefFound){ setFallbackAttr(module, ERRCODE_IGNORED); return interrupt(); }
      if (ret != 0)   { setFallbackAttr(module, ERRCODE_FAILED);  return interrupt(); }
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    if (!CVPipeline::hasFallbackAttr(module))
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
}
```

**要点**：
- **只处理 VECTOR scope**（CUBE scope 内部无跨块 tensor 依赖需要核内多缓冲）；
- **嵌套 main_loop 不允许**：`findNestedMainloop`（L227-L236）在主循环体内再找带 `kMainLoop` 的 for/while → 结构复杂，直接回退；
- `i1Found`/`memrefFound` 每循环**重置**（只对当前 main_loop 生效），命中分别回退 `ERRCODE_IGNORED`；
- `groupId` 扫描已有 `kIntraDeps` 避免与 `SplitDataflow` 的 C2C fixpipe 传输冲突。

---

## 四、流程总结

### 4.1 单个 main_loop 的处理流水

```
addInnerMultiBuffer(mainLoop)
  ├─► while && bufNum>1 → setupWhileIterArgCounter（注入 i32 迭代计数器）
  ├─► collectInnerBlockInfo       收集块 + 跨块依赖（含 i1 检查）
  ├─► memref 依赖？→ fallback
  ├─► Phase 1
  │     ├─► buildDepUserMap
  │     ├─► cloneEmptyFillsInBlocks   empty+fill → 克隆到每个 consumer 块
  │     └─► rematerializeTensorRootedScalarDeps  tensor 根标量 → 克隆重算到 consumer 块
  ├─► Phase 2
  │     ├─► collectInnerBlockInfo（重新收集，发现 clone 引入的新跨块引用）
  │     ├─► 丢弃 clone 引入的 memref 依赖
  │     ├─► cloneAllocTensorsInBlocks  alloc_tensor → 克隆到 consumer 块
  │     ├─► insertBuffersBeforeLoop    循环前分配 N 块 UB 缓冲
  │     ├─► markScalarDeps             标量依赖打 dep_mark
  │     └─► processTensorDependencies  每个跨块 tensor 依赖：
  │            producer 写分支（copy into buffer[i]） + consumer 读分支（to_tensor buffer[i]）
  └─► insertWhileCounterOps         while body 末尾 counter += 1
```

### 4.2 依赖处理矩阵

| 依赖类型 | 处理方式 | 标记 |
|---|---|---|
| `tensor.empty + linalg.fill` | 克隆到每个 consumer 块（含 ins 链） | 克隆 op 打 consumer 块 block_id |
| `bufferization.alloc_tensor` | 克隆到每个 consumer 块 | 克隆 op 打 consumer 块 block_id |
| 其他跨块 tensor | 循环前分配 N 块 UB 缓冲 + if 轮询链 | `kIntraDeps=[gid,1]`（copy）/ `[gid,0]`（if/to_tensor）+ `kIntraBuffer` |
| tensor 根标量 | 克隆计算 slice 到 consumer 块重算 | 克隆 op 打 consumer 块 block_id |
| 纯标量跨块 | 打 `kDepMark` | producer/consumer 同 mark |
| i1 tensor 跨块 | 不支持 → fallback `ERRCODE_IGNORED` | — |
| memref 依赖 | 不支持 → fallback `ERRCODE_IGNORED` | — |

### 4.3 与前后 Pass 的协作

- **前置**：`ComputeBlockOpt` 的 block_id 划分；`SplitDataflow` 的 C2C 传输（`kIntraDeps` 起始 groupId 避开）；
- **产出**：核内多缓冲结构（alloc 组 + producer/consumer if 链）+ 各类标记；
- **后置**：`AddMultiBufferOuterScope` 复用 while 计数器（`kIterCounter` 已在 while 上）；`AddControlFlowCondition` 用 `kIntraDeps` 生成同步。

---

## 五、面试要点

1. **Inner Scope 处理什么？** Vector scope 主循环**内部**的跨块 tensor 依赖——CUBE 块和 Vector 块之间的数据流。在循环前分配 N 块 UB 缓冲，producer/consumer 按迭代奇偶轮换，消除单缓冲串行。

2. **getOutermostSsbufferId 为什么是核心？** 跨块判定必须用"最外层归属"。scf.if 内嵌 op 的自身 block_id 可能是内层块的，若直接用它判断会**误判跨块**（把同 if 内的操作当成跨块 consumer）。找多 region 祖先的 block_id 才正确。

3. **为什么要两阶段收集依赖？** Phase 1 克隆 empty+fill 后，克隆 fill 的 `ins` 链会引用 producer 侧 tensor（新跨块引用）；标量重物化也产生新链。Phase 2 重跑收集才能让这些新引用进入多缓冲管线。

4. **三种依赖的处理策略？** 能克隆的（empty+fill、alloc_tensor）**克隆到 consumer 块本地一份**；tensor 根标量**克隆计算切片重算**；真正跨块的 tensor 才**分配多缓冲轮换**。每种选择都是为了减少跨块同步开销。

5. **为什么多 region consumer 的 yield 消费要单独处理？** scf.if/while 的 yield 值如果来自跨块依赖，depVal 不是 op 的直接操作数（而是 region 内 yield 的），常规操作数收集看不到。`processMultiRegionAllYields` 在 region 头插缓冲选择 if 并改写 yield 操作数。

6. **为什么 consumer 要按 Block 再分组？** `getOutermostSsbufferId` 会把 scf.if then/else 两个 region 的 op 归到同一 block_id，但一个缓冲选择 if 的结果只在**自己的 region** 可见，跨 region 共享会破坏 SSA dominance。所以按 `Block*` 细分，每个 Block 生成独立的选择 if。

7. **while 怎么获得迭代计数？** while 没有 for 的归纳变量。`setupWhileIterArgCounter` 重建 whileOp 追加 i32 counter（初值 0），`insertWhileCounterOps` 在 body 末尾插 `counter+=1` 并替换 yield 占位。必须等 `processTensorDependencies` 生成轮询 if（counter 消费点）后再定位插入位置。

8. **插入位置优化是什么？** `kInsertionOptimization` 属性控制：开 → producer 写逻辑插到 producer 块末尾、consumer 读逻辑插到 consumer 块开头（最大化重叠）；关 → 保持"紧贴定义后/紧贴使用前"。保证缓冲写读紧跟块边界。

9. **i1 / memref 依赖为什么回退？** i1 tensor（mask/条件 tensor）跨块无法用 UB 缓冲正确轮换；memref 依赖说明 IR 已缓冲化（本 Pass 在 bufferization 前工作，只处理 tensor 级依赖）。命中 `ERRCODE_IGNORED`。

10. **groupId 为什么从已有最大 +1 开始？** `kIntraDeps` 的 groupId 是核内依赖组的唯一标识，`SplitDataflow` 的 C2C fixpipe 传输可能已用过部分 id。扫描已有 id 取 max+1，避免新组 id 冲突导致下游 Pass 混淆依赖归属。
