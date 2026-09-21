# SplitDataflow 子 Pass 逐行讲解（四）：InterCoreTransferAndSync

> 源码文件：`third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/InterCoreTransferAndSync.cpp`（约 2000 行，SplitDataflow 中最大的一步）
> 流水线位置：`SplitDataflowPass` 的 7 个步骤中的 **第 3 步**（最核心、最复杂的一步）

---

## 一、这个 Pass 在做什么

`InterCoreTransferAndSync` 消费上一步 `DataDependencyAnalysis` 产出的依赖列表，在 IR 中**实际插入核间数据传输指令和同步指令**：

- **Vector → Cube 传输**：用 `hivm.copy` + `ConvertLayout`（nZ→ND）+ `memref.memory_space_cast` + `bufferization.to_tensor`；标量走 SSBuffer；
- **Cube → Vector 传输**：用 `hivm.fixpipe`（NZ2ND / **NZ2DN** 模式）+ `memory_space_cast` + `to_tensor`；
- **Cube → Cube 传输（新增）**：用 `hivm.fixpipe`（NZ2NZ 模式）写共享 L1 buffer，支持把 trunc 等量化算子**折叠进 fixpipe 的 pre_quant**；
- **同步**：插入 `SyncBlockSetOp`（set flag）和 `SyncBlockWaitOp`（wait flag）成对；
- **内存依赖同步**：插入基于 `PIPE_MTE2` 的 set/wait 对；
- **flag 复用**：**超限才触发**，调用 `FlagIdReuseManager` 压缩 flag id；
- **伪 op 清理（新增）**：`removeVectorPseudoOps` 移除 VECTOR 侧 `add_from_matmul` 标记的加零伪 op；
- **C→V direct store 同步（新增）**：`processCubeToVectorDirectStoreSync` 为"fixpipe 出来的数据不经过 Vector 计算、直接 store"的模式补一对 set/wait；
- **形状对齐**：对传入 Cube 的 tensor 做 32B 对齐的 ND→NZ 归一化（不够则 padding）。

这是整个 CV 流水线"拆分后如何重新连接"的关键一步。

**本次更新要点**：
- 新增 **C→C 传输**（`handleCubeToCube` + `createC2CSharedL1Buffer`，NZ2NZ + pre_quant 折叠 + channelSplit）；
- **C→C 细分为块间/块内**（本次最新）：`isIntraC2C` 判定 + 块内 buffer 插生产 matmul 后 + 只替换 A/B 输入 operand；
- 新增 **split-if 场景支持**（`AnalyzeSplittedIf` / `findYieldMatmulInSplitIf` / `getOpBeforeYield`：溯源 realValue + 同步区间收敛到 yield 前）；
- 新增 **C→V direct store 模式**（`isStoreDirectlyInUserChain` 判断 + `matchCubeToVectorDirectStorePattern` 匹配 + `processCubeToVectorDirectStoreSync` 补同步）；
- 新增 **VECTOR 伪 op 清理**（`removeVectorPseudoOps`）；
- 新增 **NZ2DN 转置折叠**（`dep.isAllTranspoesd`：matmul 结果只被 transpose 消费且全在 Vector 时，转置折叠进 fixpipe）；
- 新增 **1D tensor 传输支持**（不插 ConvertLayout，直接 memref cast）；
- 新增 **精确插入点分析**（`analyzeConsumerReadInsertPoint` / `getConsumerWaitPoint` / `getCopyPointBeforeStore`：wait 尽量贴近真实读取点、nd2nz 避开与 store 抢 MTE3 pipe）；
- 新增 **while 循环支持**（`findMainLoopforTransfer` 匹配 `scf::ForOp, scf::WhileOp`；direct store 匹配支持 for/while/if）；
- 新增 **依赖排序**（`sortDependencies`：按消费者出现顺序稳定处理）；
- flag 复用改为**超限才触发**（详见 03 篇）；
- 标量同步改用 `PIPE_S`（与 tensor flag 空间隔离）；
- `computeExpectedShape` 简化：不再区分 matmul A/B 场景，统一按 NZ 对齐计算。

---

## 二、关键背景：昇腾硬件 pipe 与地址空间

| 概念 | 说明 |
|---|---|
| `PIPE_FIX` / `PIPE_V` | Cube→Vector 方向使用的 pipe（fixpipe 走 FIX，vector 端走 V） |
| `PIPE_MTE1/MTE2/MTE3/M` | Vector→Cube 方向使用的 pipe（搬运引擎） |
| `PIPE_S` | 标量同步专用 pipe，与 tensor flag 空间隔离 |
| `AddressSpace::L1` | L1 缓存（V→C 传输、C2C 共享缓冲的中间缓冲） |
| `AddressSpace::UB` | Unified Buffer（C→V 传输的中间缓冲） |
| `AddressSpace::L0C` | Cube 单元的输出缓存（fixpipe 的源头） |
| `DataLayout::nZ` / `ND` / `nZ→nZ` | 昇腾数据排布格式，nZ 是 Cube 友好的分形布局 |
| `FixpipeDMAMode::NZ2ND/NZ2DN/NZ2NZ` | fixpipe 的三种搬运模式：ND 输出 / 转置输出 / 保持 NZ |
| `TCoreType::CUBE/VECTOR` | 两种计算核 |

---

## 三、逐行讲解

### 3.1 对齐计算与打标工具函数（L69-L133）

```cpp
static uint64_t getElemBytesForAlign(Type t);        // 元素字节数（f16=2, f32=4...）
static uint64_t getBlockElemsFor32BAlign(Type t) {  // 32B 对齐所需元素数
  constexpr uint64_t kAlignBytes = 32;
  uint64_t elemBytes = getElemBytesForAlign(elemType);
  if (elemBytes == 0) return 0;         // 不支持类型
  if (elemBytes >= kAlignBytes) return 1;
  return kAlignBytes / elemBytes;        // f32→8 个, f16→16 个
}
```

打标函数族：
- `attachCommonTags(op, blockId, coreType)`：`ssbuffer.block_id` + `ssbuffer.core_type`；
- `attachTransferTags(...)`：额外加 `ssbuffer.transfer_id`（同一传输的生产/消费 op 共享一个 id，后续按 id 查找）；
- `attachMemCrossDeps` / `attachCrossCoreDeps`：`ssbuffer.memCrossDeps` / `ssbuffer.crossCoreDeps` = `[transferId, seqId]`，seqId 是 `crossCoreProducerId`（生产端序号）或 `crossCoreConsumerId`（消费端序号）——把内存依赖的原 op 也挂上"跨核依赖"标记；
- `attachAnalyzeFlagIdTag(op)`：`ssbuffer.analyze_flag_id`（UnitAttr），标记本 sync op 参与 flag 复用分析。

### 3.2 C→C 有效性判断（L166-L199，本次更新）

```cpp
static bool hasAnyMatmulABInputUser(Value value, int consumerBlockId) {
  // 消费者块内至少有一个 matmul 把 value 用作 A/B 输入（DpsInputs）
}

static bool isValidC2CMatmulDependency(Value value, int consumerBlockId) {
  // 沿 use-def 链向上溯源（跳过可折叠进 pre_quant 的中间 op），
  // 最终必须到达一个 matmul；且消费端有 matmul 把它当 A/B 输入
  Operation *defOp = CVPipeline::getSourceThroughCIntermediateOps(value);
  if (!isa_and_nonnull<linalg::MatmulOp>(defOp)) return false;
  return hasAnyMatmulABInputUser(value, consumerBlockId);
}
```

- **本次更新**：把原来的局部 `isValidC2CIntermediateOp` + while 循环，替换为统一的 `CVPipeline::getSourceThroughCIntermediateOps`（`Common/Utils.h`）——与 `DataDependencyAnalysis` 的 `resolveSameBlockMatmulProducer` 共用同一套"跳过中间 op 溯源"逻辑，避免两处重复定义漂移；
- 合法 C2C 链：`matmul -> (intermediate_op)* -> matmul`，中间 op 只允许"有 FixpipePreQuantMode"的量化算子。

### 3.3 getBlockStartEnd：定位块首尾 op（L201-L247）

```cpp
std::pair<mlir::Operation *, mlir::Operation *>
InterCoreTransferAndSyncPass::getBlockStartEnd(int targetId, mlir::ModuleOp module) {
  // PreOrder walk 找到第一个属于 targetId 的 op → 它所在的 Block
  // 然后在该 Block 内线性扫描，收集 [start, end] 连续区间
}
```

- 假设同一 block 内属于同一 block_id 的 op 是连续的（由前置 `ReorderOpsByBlockId` 保证）。

### 3.4 isOuterLayerDependency：外层内存依赖去重（L283-L336）

```cpp
bool InterCoreTransferAndSyncPass::isOuterLayerDependency(
    size_t depIndex, Operation *currProdEnd, Operation *currConsStart,
    llvm::SmallVector<DependencyInfo> &memDependencies) {
  // 对每条同类型其他依赖 otherDep：
  //   otherProdEnd 在 currProdEnd 之后（!otherProdEnd->isBeforeInBlock(currProdEnd)）
  //   且 currConsStart 在 otherConsStart 之前（!currConsStart->isBeforeInBlock(otherConsStart)）
  //   → otherDep 被"包"在当前依赖内部 → 当前是外层依赖，跳过
  // 两个完全相同的生产/消费块时：索引更小的先处理，后面的跳过
}
```

- 嵌套循环中，同一对块的内存依赖会在多层各报一次，只有最内层（真正紧邻读写点）的那条需要插同步。

### 3.5 computeExpectedShape：NZ 对齐形状计算（L338-L388，简化）

```cpp
SmallVector<int64_t> InterCoreTransferAndSyncPass::computeExpectedShape(mlir::Value depValue) {
  auto tensorTy = dyn_cast<TensorType>(depValue.getType());
  if (!tensorTy || tensorTy.getRank() != 2) { setFallbackAttr(...); return {}; }
  int64_t M = tensorTy.getDimSize(0), N = tensorTy.getDimSize(1);
  int64_t nWidth = getBlockElemsFor32BAlign(tensorTy.getElementType());
  int mRound = NzDimWidth;   // 16
  int nRound = nWidth;       // 32B / 元素大小
  int64_t newM = ((M + mRound - 1) / mRound) * mRound;   // M 向上取整到 16
  int64_t newN = ((N + nRound - 1) / nRound) * nRound;   // N 向上取整到 32B
  return {newM, newN};
}
```

- **本次更新**：删除了旧的 `isMatmulA/isMatmulB/isOnlyDepInMatmul` 分支（matmul A/B 对齐判断移到了 `DataDependencyAnalysis`/其他地方），现在统一只按 NZ 格式的对齐规则计算。

### 3.6 getCopyPointBeforeStore：避开 store 的插入点（L389-L411，新增）

```cpp
// insert copyop before store to avoid mte3 blocking (store and V->C use the
// same PIPE)
mlir::Operation *InterCoreTransferAndSyncPass::getCopyPointBeforeStore(
    Value depValue, Operation *vectorEndOp, int iniProducerBlockId) {
  Operation *curr = vectorEndOp;
  while (curr) {
    if (blockIdOpt != iniProducerBlockId) break;       // 出块停止
    if (curr == depValue.getDefiningOp()) break;        // 到定义点停止
    if (CVPipeline::isStoreLike(curr)) {
      firstStoreOpAfterProducer = curr->getPrevNode();   // store 前一个 op
    }
    curr = curr->getPrevNode();                          // 向前扫描
  }
  return firstStoreOpAfterProducer;
}
```

- **为什么**：V→C 的 copy 和 store 都走 **MTE3 pipe**。若 nd2nz 的 copy 插在 store 之后，会跟 store 抢同一个 pipe 造成阻塞；提前到 store 之前可以并行。

### 3.7 alignShapeByInsertSlice：padding 到对齐形状（L412-L466）

形状不满 NZ 对齐要求时：`arith.constant(0)` + `tensor.empty` + `linalg.fill`（填零）+ `tensor.insert_slice`（把原数据贴进左上角），全部打 VECTOR 标。插入点：BlockArgument 在生产块尾之后，否则在定义 op 之后。

### 3.8 Nd2NzNormalize：ND→NZ 归一化（L468-L561）

```cpp
void InterCoreTransferAndSyncPass::Nd2NzNormalize(OpBuilder &builder,
                                                  DependencyInfo &dep, Location loc) {
  Value origValue = dep.value;
  // Step 0: ndnzValueMapping 查重，处理过直接返回（本次更新：改名 ndnzValueMapping）
  // Step 1: computeExpectedShape 计算对齐形状
  // Step 2: isExpectedShape 不等 → alignShapeByInsertSlice padding
  // Step 3: 插入 nd2nz 三步曲：
  auto [newProdStart, newProdEnd] = getBlockStartEnd(dep.producerBlockId, module);
  if (dep.iniProducerBlockId == dep.producerBlockId) {
    auto producerPoint = getCopyPointBeforeStore(newValue, newProdEnd, dep.iniProducerBlockId);
    if (producerPoint) newProdEnd = producerPoint;    // 避开 store（MTE3 冲突）
  }
  builder.setInsertionPointAfter(newProdEnd);
  auto reshape3DOp  = builder.create<tensor::ReshapeOp>(loc, type3D, newValue, ...);   // {M, N/blk, blk}
  auto transposeOp  = builder.create<linalg::TransposeOp>(..., {1,0,2});              // {N/blk, M, blk}
  auto reshape4DOp  = builder.create<tensor::ReshapeOp>(loc, typeFinal, ...);         // {N/blk, M/16, 16, blk}
  // 全部 attachCommonTags(..., originBlockId, "VECTOR")
  ndnzValueMapping[origValue] = reshape4DOp.getResult();
}
```

- `ND` 是普通连续排布；`NZ` 是 Cube 要求的**分形排布**，reshape+transpose+reshape 三步实现布局转换。
- **sub-block 打标（本次更新）**：若 `dep.value` 的定义 op 带 `ssbuffer.subBlock` 标记（说明该 op 位于子块内），归一化插入的所有新 op（reshape3D/reshape3Dcst/transpose/reshape4D/reshape4Dcst/emptyTrans）都会用 `CVPipeline::setSubBlockId` 打上同一个 subBlock id——保证归一化产生的 op 属于同一个子块，后续调度/同步能按子块对齐。

### 3.9 传输缓冲与插入点工具（L563-L691）

- `annotateTightlyCoupledBuffer`（L563）：给 alloc 挂 `annotation.mark`（effects=[read,write]）+ `HIVMTightlyCoupledBufferAttr`（编号 markAllocIndex），生产/消费两个 memref 在硬件上映射到同一物理 buffer 的两侧视图；
- `findMainLoopforTransfer`（L580-L593）：向上找最近的**主循环**（`isa<scf::ForOp, scf::WhileOp>`，**本次更新加入 while**），缓冲分配到循环外；找不到且两端不同父块 → fallback；
- `createTransferAllocs`（L594-L638）：有主循环时在循环外分配**一对** buffer（prod/cons 各一，都打 tightly-coupled 标记 + transfer_id）；无主循环时在生产点后分配两个；
- `analyzeConsumerReadInsertPoint`（L639-L662，新增）：PreOrder walk 找**消费者块内第一个真正读取 srcValue 的 op**——wait 的理想插入点（比块首更精确，减少不必要等待）；
- `getConsumerWaitPoint`（L663-L691）：按 `transfer_id` 找 `ConvertLayoutOp`/`MemorySpaceCastOp`/`LoadOp`，作为传输插入后的 wait 锚点。

### 3.10 insertVectorToCubeTransfer：插入 V→C 传输（L692-L836）

```cpp
Operation *InterCoreTransferAndSyncPass::insertVectorToCubeTransfer(
    OpBuilder &builder, Value srcValue, Value normalizedValue,
    Operation *vectorEndOp, Operation *cubeStartOp, Location loc,
    int transferIndex, DependencyInfo &dep, bool is1DTensor,   // 新增 is1DTensor
    Operation **consumedDataOp) {
  if (isScalarDependency(dep.value)) {
    // 标量：SSBuffer 路径
    auto addrOpt = ssbufferManager.writeToSSBuffer(srcValue, builder, writeOps);
    auto loadedValueOpt = ssbufferManager.readFromSSBuffer(addr, builder, readOps);
    // 写/读 op 各自 attachTransferTags + attachCrossCoreDeps
  } else {
    // 张量：alloc L1 缓冲对 + copy + （非 1D 时）ConvertLayout(nZ→ND) + cast + to_tensor
    auto [vecAllocOp, cubeAllocOp] = createTransferAllocs(..., hivm::AddressSpace::L1, ...);
    auto copyOp = builder.create<hivm::CopyOp>(loc, {}, normalizedValue, vecAllocOp->getResult(0));
    builder.setInsertionPoint(cubeStartOp);
    Value memValue = cubeAllocOp->getResult(0);
    if (!is1DTensor) {                       // 1D tensor（如 bias）跳过布局转换
      auto convertLayoutOp = builder.create<hivm::ConvertLayoutOp>(
          loc, newAllocType, memValue, nzLayout, ndLayout);
      memValue = convertLayoutOp.getResult();
    }
    auto memspaceCastOp = builder.create<memref::MemorySpaceCastOp>(...);
    auto toTensorOp = builder.create<bufferization::ToTensorOp>(...);
  }

  // 替换消费端所有 use（含 consumerYieldOp——yield 消费点也要换，本次更新）
  if (dep.consumerYieldOp) dep.consumerYieldOp->replaceUsesOfWith(srcValue, receiveValue);
  for (Operation *user : users)
    if (userBlockId == dep.iniConsumerBlockId) user->replaceUsesOfWith(srcValue, receiveValue);
  if (consumedDataOp) *consumedDataOp = receiveOp;
  return sendOp;
}
```

### 3.11 insertCubeToVectorTransfer：插入 C→V 传输（L837-L928）

```cpp
Operation *InterCoreTransferAndSyncPass::insertCubeToVectorTransfer(...) {
  // 转置折叠：matmul 结果若只被一个 transpose 消费且全在 Vector（isAllTranspoesd），
  // 目标形状转置为 {N, M}，fixpipe 用 NZ2DN 模式直接转置输出
  auto targetShape = dep.isAllTranspoesd ? std::vector<int64_t>{N, M}
                                         : std::vector<int64_t>{M, N};
  auto [cubeAllocOp, vecAllocOp] = createTransferAllocs(..., hivm::AddressSpace::UB, ...);
  auto dmaModeAttr = FixpipeDMAModeAttr::get(ctx,
      dep.isAllTranspoesd ? FixpipeDMAMode::NZ2DN : FixpipeDMAMode::NZ2ND);

  auto fixpipeOp = builder.create<hivm::FixpipeOp>(
      loc, {}, srcValue, cubeAllocOp->getResult(0), {}, dmaModeAttr, ...);
  // Vector 侧：memory_space_cast + to_tensor

  if (dep.isAllTranspoesd) {
    // 找到那个 transpose 用户，后续替换它的结果而非 matmul 的结果
    for (auto *userOp : srcValue.getUsers())
      if (isa<linalg::TransposeOp>(userOp)) { srcValue = userOp->getResults()[0]; break; }
  }
  // 替换消费端 use（含 consumerYieldOp）
}
```

- **NZ2DN 转置折叠（新增）**：Vector 侧本来要单独做一次 transpose；折叠进 fixpipe 后硬件搬运时直接转置，省掉 Vector 核的计算。

### 3.12 getTransferPipeConfig：pipe 配置（L930-L987，更新）

```cpp
TransferPipeConfig InterCoreTransferAndSyncPass::getTransferPipeConfig(
    Operation *transferOp, bool isStoreDirectly) {          // 新增参数
  if (isa<hivm::FixpipeOp>(transferOp)) {                   // C→V
    if (isStoreDirectly) {          // direct store：Vector 端是 MTE3 store
      config.forReadTPipe = pipeFixAttr;   config.forReadPipe = pipeMte3Attr;
      config.forWriteTPipe = pipeMte3Attr; config.forWritePipe = pipeFixAttr;
    } else {                         // 普通：Vector 端走 V pipe
      config.forReadTPipe = pipeFixAttr;   config.forReadPipe = pipeVAttr;
      config.forWriteTPipe = pipeVAttr;    config.forWritePipe = pipeFixAttr;
    }
  } else if (isa<hivm::CopyOp>(transferOp)) {                // V→C
    config.forReadTPipe = pipeMte3Attr;  config.forReadPipe = pipeMte1Attr;
    config.forWriteTPipe = pipeMAttr;   config.forWritePipe = pipeMte3Attr;
  } else if (isa<memref::StoreOp>(transferOp)) {             // 标量
    // Scalar sync uses PIPE_S to stay isolated from tensor flag space.
    config 全部用 pipeSAttr;      // 标量同步用 PIPE_S，与张量 flag 空间隔离
  }
}
```

### 3.13 isStoreDirectlyInUserChain：direct store 判断（L988-L1024，新增）

```cpp
bool InterCoreTransferAndSyncPass::isStoreDirectlyInUserChain(Value toTensorValue) {
  llvm::SmallVector<Value> workList = {toTensorValue};
  llvm::DenseSet<Value> visited;
  bool hasStore = false;
  while (!workList.empty()) {
    Value currVal = workList.pop_back_val();
    if (visited.count(currVal)) continue;
    visited.insert(currVal);
    for (Operation *user : currVal.getUsers()) {
      if (CVPipeline::isStoreLike(user)) { hasStore = true; continue; }
      // ViewLike / zero-add / for_may_not_exec 视为"透明"，继续下钻
      if (CVPipeline::isViewLike(user) || CVPipeline::isZeroAdd(user) ||
          user->hasAttr(CVPipeline::kForMayNotExec)) {
        for (Value result : user->getResults()) workList.push_back(result);
      }
      // 其他计算 op：说明中间有 Vector 张量计算，不是 direct store
    }
  }
  return hasStore && ...;   // 有 store 且中途没有真正的 Vector 计算
}
```

- fixpipe 到 UB 的值如果只经过 view/加零这类"透明" op 就直接 store，说明 Vector 核没有做任何有效计算——此时同步的 wait 端应该贴着 store（MTE3），而不是 V pipe。

### 3.14 insertInterCoreSync：插入同步（L1025-L1144，更新）

```cpp
void InterCoreTransferAndSyncPass::insertInterCoreSync(
    OpBuilder &builder, Operation *transferOp, Operation *consumerStartOp,
    Operation *consumerEndOp, int flag, Location loc, int transferIndex,
    FlagIdReuseManager &flagIdReuseManager, Operation *consumedDataOp,
    bool isStoreDirectly) {                                  // 新增参数
  auto config = getTransferPipeConfig(transferOp, isStoreDirectly);

  // 读同步：传输后 set，消费前 wait
  builder.setInsertionPointAfter(transferOp);
  auto setOpForRead = builder.create<SyncBlockSetOp>(loc, config.srcCoreAttr, config.forReadTPipe, config.forReadPipe, flagId);
  builder.setInsertionPoint(consumerStartOp);
  auto waitOpForRead = builder.create<SyncBlockWaitOp>(loc, config.dstCoreAttr, ...);

  if (mainLoopOp) {
    // 写同步（ping-pong）：写前 wait、用后 set、循环边界 set/wait
    auto waitOpForWrite = ...;  // 在 transferOp 之前
    auto setOpForWrite  = ...;  // 在 consumerEndOp 之后
    auto setOpForStart  = ...;  // 在 mainLoopOp 之前
    auto waitOpForEnd   = ...;  // 在 mainLoopOp 之后

    attachAnalyzeFlagIdTag(六个 op 全部打标);      // 本次更新：统一打标
    // E2: 注册全部三组 set->wait 配对（读、写、循环边界）
    flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForRead, waitOpForRead);
    flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForWrite, waitOpForWrite);
    flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForStart, waitOpForEnd);
    // E4: read-wait 连到它保护的数据消费点
    flagIdReuseManager.insertRelationBetweenSetAndWait(waitOpForRead, consumedDataOp);
    return;
  }
  // 无主循环：一对 set/wait + 同样打标和 E2/E4 注册
}
```

- **读同步**：生产者写完 set，消费者读取前 wait；
- **写同步**（仅主循环内）：生产者写新数据前 wait（等消费者用完旧数据），消费者用完 set。这就是 **ping-pong 双缓冲同步**；
- E2/E4 关系边与 `kAnalyzeFlagId` 打标是为 flag 复用分析准备的（详见 03 篇）。

### 3.15 insertMemDepSync：内存依赖同步（L1145-L1337）

```cpp
void InterCoreTransferAndSyncPass::insertMemDepSync(
    ..., Operation *producerOp, Operation *consumerOp, int flag, ...,
    bool isCubeToVector, FlagIdReuseManager &flagIdReuseManager) {
  hivm::PIPE srcPipe = isCubeToVector ? PIPE_FIX : PIPE_MTE3;
  hivm::PIPE dstPipe = PIPE_MTE2;
  // producerOp 之后 set（srcCore），consumerOp 之前 wait（dstCore）
  attachAnalyzeFlagIdTag(setOp); attachAnalyzeFlagIdTag(waitOp);
  flagIdReuseManager.insertRelationBetweenSetAndWait(setOp, waitOp);  // E2
}
```

- 内存依赖固定经 `PIPE_MTE2`（搬运引擎）同步，保证跨核 load/store 顺序。

### 3.16 C→V direct store 三件套（L1376-L1521，新增）

**模式定义**：Cube fixpipe 出来的数据经 UB 到达 Vector，**中间没有任何 Vector 张量计算**，直接被 store 走。此时需要一对额外同步保护 SCF region 之后的 MTE3 store。

```cpp
// 1. findVectorToTensorInScfOp：在 scf op 的 region 里找
//    "scf.yield 直接 yield 一个 VECTOR 侧 to_tensor，且其 memref 可回溯到
//     tightly_coupled_buffer 标记的 alloc" —— 即 fixpipe 的 UB 输出被 yield 出循环
std::optional<VectorToTensorInfo> findVectorToTensorInScfOp(Operation *scfOp);

// 2. findStoreLikeAfterScfOp：从 scf result 出发，穿过 view-like op 找 store
Operation *findStoreLikeAfterScfOp(Value scfResult);

// 3. matchCubeToVectorDirectStorePattern：组合判断
std::optional<CubeToVectorDirectStoreInfo>
InterCoreTransferAndSyncPass::matchCubeToVectorDirectStorePattern(Operation *scfOp) {
  if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp>(scfOp)) return std::nullopt;  // for/while/if
  auto anchor = findVectorToTensorInScfOp(scfOp);
  // yield 的结果索引 → scf result；沿 result 找 store
  // 命中返回 {tightlyCoupledBufferId, storeOp}
}

// 4. processCubeToVectorDirectStoreSync：为每个命中的模式补同步
void InterCoreTransferAndSyncPass::processCubeToVectorDirectStoreSync(...) {
  module.walk([&](Operation *op) {
    auto match = matchCubeToVectorDirectStorePattern(op);
    ...
    int flagId = flagManager.acquireId();
    int syncTransferId = transferIndex++;
    // set 插在 SCF op 之后（跳过之前 handler 已追加的 sync op）
    auto setOp = builder.create<SyncBlockSetOp>(loc, cubeAttr, pipeFixAttr, pipeMte3Attr, flagIdAttr);
    // wait 插在 store 之前（materialize_in_destination 前）
    auto waitOp = builder.create<SyncBlockWaitOp>(loc, vecAttr, pipeFixAttr, pipeMte3Attr, flagIdAttr);
    attachAnalyzeFlagIdTag(...);  // 参与复用分析
    flagIdReuseManager.insertRelationBetweenSetAndWait(setOp, waitOp);
  });
}
```

- set 端 `block_id` 取 SCF op 的、core_type = CUBE；wait 端取 store 的、core_type = VECTOR；
- **为什么需要**：fixpipe（FIX pipe）与 store（MTE3 pipe）在不同 pipe 上，硬件不会自动排序，必须显式 flag 保证"Cube 侧 fixpipe 完成后 Vector 才能 store"。

### 3.17 removeVectorPseudoOps：VECTOR 伪 op 清理（L1339-L1375，新增）

```cpp
void InterCoreTransferAndSyncPass::removeVectorPseudoOps() {
  module.walk([&](Operation *op) {
    if (!isa<arith::AddFOp, arith::AddIOp>(op)) return;
    // (1) 必须带 ssbuffer.add_from_matmul 标记且是 VECTOR 核
    if (!op->hasAttr(CVPipeline::kAddFromMatmul)) return;
    if (coreAttr != CVPipeline::kCoreTypeVector) return;
    // (2) 一个操作数必须是 zero-fill tensor，另一个是真正的数据流来源
    Value lhs = op->getOperand(0), rhs = op->getOperand(1);
    Value keptOperand = nullptr;
    if (CVPipeline::isZeroFillValue(lhs)) keptOperand = rhs;
    else if (CVPipeline::isZeroFillValue(rhs)) keptOperand = lhs;
    else return;
    // x + 0 = x：结果直接替换为保留的操作数，伪 op 删除
    op->getResult(0).replaceAllUsesWith(keptOperand);
    op->erase();
  });
}
```

- **背景**：上游为了 SSA 形态合法，可能给 yield 塞了 `addf(x, zero)` 这种"加零"伪 op 让数据看似在 Vector 侧被计算过。CV 拆分完成后这些 op 是纯开销，统一清理掉。

### 3.18 分析辅助函数（L1956-L2018，新增）

```cpp
static bool isConcretePipe(hivm::PIPE pipe);      // 非 UNASSIGNED/ALL/NUM
static std::optional<hivm::TCoreType> getAnalyzeCoreType(Operation *op);
  // 先看 hivm.tcore_type 属性，再看 ssbuffer.core_type 字符串
static std::optional<hivm::PIPE> getCopyPipeForAnalyze(hivm::CopyOp copyOp);
  // 按 copy 的 src/dst 地址空间推断 pipe：
  //   UB→UB = V, L0C→GM = FIX, GM→L1 = MTE2, UB→L1 = MTE3
  //   纯 buffer 语义直接用 copyOp.getPipe()
  //   SplitDataflow 插的 tensor→L1 copy 用 MTE3（vector 侧）
```

- 这些函数服务于 `insertAnalyzeFlagRelations`（flag 复用关系图构建），详见 03 篇。

### 3.19 三个 handle 函数（L1574-L1954）

```cpp
handleVectorToCube(...)   // 处理一条 V→C 依赖：
  normalizedVal = ndnzValueMapping[dep.value]                    // 归一化结果
  [prodStart, prodEnd] / [consStart, consEnd] = getBlockStartEnd(...)
  // 精确化插入点（新增）：
  if (dep.consumerBlockId == dep.iniConsumerBlockId)
    consStart = analyzeConsumerReadInsertPoint(srcValue, ...);    // wait 贴近真实读点
  if (dep.iniProducerBlockId == dep.producerBlockId) {
    prodEnd = getCopyPointBeforeStore(...);                       // copy 避开 store
    // sub-block 修正（本次更新）：源 op 若带 sub_block 标记，
    //   prodEnd 取源 op 所在子块的末尾（归一化 copy 落在子块内而非子块外）
    if (srcDefOp 带 subBlockId) prodEnd = getSubBlockStartEnd(srcDefOp).second;
  }
  transferOp = insertVectorToCubeTransfer(...)
  flagId = flagManager.acquireId()
  // 插入传输后重新取块端点（块内多了新 op），再精确化 wait 点：
  newConsStart = getConsumerWaitPoint(transferIndex);
  insertInterCoreSync(builder, transferOp, newConsStart, newConsEnd, flagId, ...)
  transferIndex++

handleCubeToVector(...)   // 处理一条 C→V 依赖：
  AnalyzeSplittedIf(dep)                            // 新增：识别 split-if 场景，填充 dep.isSplitedIf / realValue
  // sub-block 修正（本次更新）：若消费端真实读取点带 sub_block 标记，
  //   consStart 收敛到该子块的起点、newConsEnd 收敛到子块终点
  //   （同步区间与子块对齐，避免跨子块误同步）
  consumerPoint = analyzeConsumerReadInsertPoint(srcValue, iniConsumerBlockId);
  if (consumerPoint && getSubBlockId(consumerPoint)) {
    consStart   = getSubBlockStartEnd(consumerPoint).first;
    newConsEnd  = getSubBlockStartEnd(consumerPoint).second;
  }
  // split-if 场景下（源值来自被 kSplittedIf 标记的 op）：
  //   prodEnd 直接指到 realValue（split-if 内真正 yield 出来的 matmul）的 defining op
  //   consStart 用 analyzeConsumerReadInsertPoint 精确到真实读取点
  transferOp = insertCubeToVectorTransfer(...)
  flagId = flagManager.acquireId()
  bool isStoreDirectly = isStoreDirectlyInUserChain(consumedDataOp->getResult(0));  // 新增
  if (dep.isSplitedIf) {
    // 同步点修正：wait 插到真实读取点，且 set/wait 区间截止到 yield 之前的那个 op
    // （getOpBeforeYield 返回 block 终结符的前一个 op），避免同步越过 yield
    newConsStart = getConsumerWaitPoint(transferIndex);
    newConsEnd   = getOpBeforeYield(newconsumerPoint);
  }
  insertInterCoreSync(..., isStoreDirectly)                       // 影响 pipe 配置

handleMemoryDependency(...)  // 处理一条内存依赖：
  if (isOuterLayerDependency(...)) return success();  // 跳过外层依赖（内层已覆盖）
  attachMemCrossDeps / attachCrossCoreDeps            // 给原 op 打跨核标记（新增）
  flagId = flagManager.acquireId()
  insertMemDepSync(...)
```

### 3.20 C→C 传输：handleCubeToCube（L1728-L1896，本次更新：块间/块内区分）

```cpp
// C->C Shared L1 buffer allocation（L1732，本次更新：新增 isIntraC2C 参数）
Operation *createC2CSharedL1Buffer(..., bool isIntraC2C) {
  // 块间 C2C（isIntraC2C=false）：主循环存在 → 在循环外分配（遵循块级流水线）
  // 块内 C2C（isIntraC2C=true）  ：不管有没有主循环，都直接插在生产 matmul 之后
  if (mainLoopOp && !isIntraC2C) { setInsertionPoint(mainLoopOp); alloc; setInsertionPointAfter(prodEnd); }
  else                           { setInsertionPointAfter(prodEnd); alloc; }
}

LogicalResult InterCoreTransferAndSyncPass::handleCubeToCube(OpBuilder &builder,
                                                              DependencyInfo &dep) {
  // 0. 判定块间/块内（新增）：生产与消费的原始块 id 相同 → 块内 C2C
  bool isIntraC2C = (dep.iniProducerBlockId == dep.iniConsumerBlockId);

  // 1. 量化折叠：matmul -> trunc -> matmul 模式下，
  //    fixpipe 源用 trunc 之前的 matmul 结果，trunc 折叠为 pre_quant
  std::optional<FixpipePreQuantMode> quantMode =
      CVPipeline::getFixpipePreQuantMode(truncOp);
  if (quantMode) fixpipeSrcValue = truncOp->getOperand(0);

  // 2. 共享 L1 buffer 分配（块间/块内插入点不同）
  auto *allocOp = createC2CSharedL1Buffer(..., isIntraC2C);

  // 3. 生产端：fixpipe NZ2NZ 写 L1（可选 pre_quant + channelSplit）
  auto dmaModeAttr = FixpipeDMAModeAttr::get(ctx, FixpipeDMAMode::NZ2NZ);
  bool channelSplit = isChannelSplitNeeded(transferTensorType);  // 8 elem/block 时分流
  auto fixpipeOp = builder.create<hivm::FixpipeOp>(..., fixpipeSrcValue, allocOp, ...);
  if (!isIntraC2C)                                            // 仅块间打 kIntraDeps
    fixpipeOp->setAttr(CVPipeline::kIntraDeps, {intraDepsGroupId, crossCoreProducerId});

  // 4. 消费端：memory_space_cast + to_tensor 读 L1
  if (!isIntraC2C)                                            // 仅块间打 kIntraDeps
    memspaceCastOp->setAttr(CVPipeline::kIntraDeps, {intraDepsGroupId, crossCoreConsumerId});

  // 5. 替换消费端 use：matmul 只换 A/B 输入，不换 outs/init
  //    （同一值同时做输入和 init 时，init 继续直接用原 matmul 结果）
  for (Operation *user : users) {
    if (userBlockId == dep.iniConsumerBlockId) {
      if (user == fixpipeOp) continue;
      if (auto matmulUser = dyn_cast<linalg::MatmulOp>(user)) {
        for (i : DpsInputs) if (getOpOperand(i) == transferValue) setOperand(i, toTensorResult);
      } else {
        user->replaceUsesOfWith(transferValue, toTensorOp.getResult());
      }
    }
  }
  // 6. trunc 已折叠，删除
  if (truncOp && quantMode) truncOp->erase();
  intraDepsGroupId++;
}
```

- **C→C 的意义**：Cube 核间的 matmul 链（如 `matmul → trunc → matmul`）中间结果不必绕道 UB/Vector 核，用 fixpipe NZ2NZ 直接在 **L1 层面直传**，同时把量化算子折叠进 fixpipe 的 pre_quant，一次搬运完成"传输+量化"；
- **注意**：C→C 不走 `insertInterCoreSync`（不分配 flag），同步依赖 L1 fixpipe 的天然顺序，用 `kIntraDeps` 属性标记组内依赖关系（`intraDepsGroupId` 成对：生产 fixpipe 与消费 memspace_cast）；
- **块间 vs 块内（本次更新）**：
  - **块间 C2C**：生产/消费在不同 Cube block，buffer 必须分配到**主循环外**（若两端有主循环包裹），遵循块级流水线；需要打 `kIntraDeps` 标记以便下游调度按组对齐；
  - **块内 C2C**：同一 block 内前一个 matmul 直接喂后一个 matmul，buffer 直接插在**生产 matmul 之后**、不上升到循环外；同块天然顺序存在，**不需要** `kIntraDeps` 分组标记；
  - 替换时区分**输入（A/B）与 init（outs）**：块内场景下 matmul 结果常同时作为下一个 matmul 的输入和 init（如累加），只能替换 DpsInputs 槽位，init 保留原引用，否则会破坏累加语义。

### 3.20.1 split-if 场景：AnalyzeSplittedIf / findYieldMatmulInSplitIf（L1523-L1571，本次更新）

```cpp
// 找到 split-if 里真正被 yield 出来的那个 matmul 结果（L1526）
linalg::MatmulOp findYieldMatmulInSplitIf(Value splittedIfResult) {
  auto ifOp = splittedIfResult.getDefiningOp<scf::IfOp>();
  unsigned resultIndex = splittedIfResult.cast<OpResult>().getResultNumber();
  auto yieldOp = ifOp.thenYield();                        // split-if 的 then 块 yield
  return dyn_cast<linalg::MatmulOp>(
      yieldOp.getOperand(resultIndex).getDefiningOp());
}

// split-if 场景分析，填充 dep.isSplitedIf / dep.realValue（L1546）
void AnalyzeSplittedIf(DependencyInfo &dep) {
  auto srcOp = dep.value.getDefiningOp();
  if (!srcOp || !srcOp->hasAttr(CVPipeline::kSplittedIf)) return;   // 源值必须被 split-if 标记
  // 在源 op 之后线性扫描，找到同 consumerBlockId 且同样带 kSplittedIf 的 scf.if
  for (Operation *nextOp = srcOp->getNextNode(); nextOp; nextOp = nextOp->getNextNode()) {
    auto op = dyn_cast<scf::IfOp>(nextOp);
    if (op && CVPipeline::getOpBlockId(op) == dep.consumerBlockId) { consIfOp = op; break; }
  }
  // 两边的 kSplittedIf 编号必须一致（同一对 split 的 if/else）
  if (srcOp->getAttrOfType<IntegerAttr>(kSplittedIf) !=
      consIfOp->getAttrOfType<IntegerAttr>(kSplittedIf)) return;
  if (auto matmulOp = findYieldMatmulInSplitIf(dep.value)) {
    dep.isSplitedIf = true;
    dep.realValue = matmulOp->getResult(0);   // 真正的 matmul 结果
  }
}
```

- **为什么要溯源到 split-if 内部**：split-if 把原 block 拆成 if/else 两个子块，`dep.value` 是 `scf.if` 的 **result**（在 if 块外），真正的 matmul 在 if 的 then 块内被 yield 出来。fixpipe 只能直接从 matmul 结果搬运，所以必须钻到 split-if 内部找到 `realValue`，否则插入点（prodEnd）会落在 if 之前、搬运到的是错误的数据；
- **同步区间修正**（见 3.19）：split-if 场景下 `insertInterCoreSync` 的 wait 插到真实读取点，`newConsEnd` 用 `getOpBeforeYield` 收敛到 yield 终结符的前一个 op——避免 set/wait 区间把 split-if 的 yield 也包进去，造成同步语义错误；
- `getOpBeforeYield`（L138）：取 op 所在 block 的终结符（必须是 `scf.yield`），返回其前一个 op。

### 3.21 sortDependencies：依赖排序（L2094-L2142，新增）

```cpp
void InterCoreTransferAndSyncPass::sortDependencies(
    llvm::SmallVector<DependencyInfo> &dependencies, mlir::ModuleOp module) {
  // Step 1: PreOrder walk 给每个 op 一个单调递增序号
  // Step 2: getFirstConsumerOp：dep.value 的用户中 block_id == consumerBlockId
  //         且序号最小的那个（消费块内最早的读取点）
  // Step 3: 按"消费者 op 出现顺序"排序（稳定的插入顺序）
}
```

- 保证依赖处理顺序与消费点在 IR 中的位置一致 → 插入的传输/同步顺序确定、可复现。

### 3.22 processDependencies 主流程（L2144-L2311，更新）

```cpp
LogicalResult InterCoreTransferAndSyncPass::processDependencies(
    FlagIdManager &flagManager, FlagIdReuseManager &flagIdReuseManager) {
  auto &info = getAnalysis<DataDependencyInfo>();
  if (!info.isValid()) return failure();

  // Step 1: V→C（先归一化再插传输）
  auto &V2CDependencies = info.getV2CDependencies();
  sortDependencies(V2CDependencies, module);                         // 新增排序
  for (auto &dep : V2CDependencies)
    if (!isScalarDependency(dep.value) && !is1DTensorDependency(dep.value))  // 1D 免归一化
      Nd2NzNormalize(builder, dep, loc);
  for (auto &dep : V2CDependencies) handleVectorToCube(...);

  // Step 2: C→V
  auto &C2VDependencies = info.getC2VDependencies();
  sortDependencies(C2VDependencies, module);
  for (auto &dep : C2VDependencies) handleCubeToVector(...);

  // Step 3.1: C→C 块间（新增）
  auto &C2CDependencies = info.getC2CDependencies();
  sortDependencies(C2CDependencies, module);
  for (auto &dep : C2CDependencies) {
    if (!isValidC2CMatmulDependency(dep.value, dep.consumerBlockId)) continue;
    if (failed(handleCubeToCube(builder, dep))) return failure();
  }
  // Step 3: C→C 块内（intraC2CDependencies，本次更新）
  auto &intraC2CDependencies = info.getIntraC2CDependencies();
  sortDependencies(intraC2CDependencies, module);
  for (auto &dep : intraC2CDependencies) {
    if (!isValidC2CMatmulDependency(dep.value, dep.consumerBlockId)) continue;
    // 块内 C2C 必须通过 operand 精确定位消费 matmul 的输入槽位：
    // 只处理 A/B 输入（getNumDpsInputs 以内），init（outs）槽位跳过
    if (!dep.operand) continue;
    auto consumerMatmul = dyn_cast<linalg::MatmulOp>(dep.operand->getOwner());
    if (!consumerMatmul ||
        dep.operand->getOperandNumber() >= consumerMatmul.getNumDpsInputs())
      continue;
    if (failed(handleCubeToCube(builder, dep))) return failure();
  }

  // Step 4: 内存依赖
  for (size_t i = 0; i < memDependencies.size(); ++i)
    handleMemoryDependency(builder, dep, i, memDependencies, flagManager, flagIdReuseManager);

  // Step 5: flag 复用（超限才触发，本次更新）
  if (!flagManager.checkCurrentId()) {
    auto analyzeFlagIdOps = insertAnalyzeFlagRelations(module, flagIdReuseManager);
    auto remapResult = flagIdReuseManager.reuseInterCoreTransferFlagIds(analyzeFlagIdOps);
    remapInterCoreTransferFlagIds(remapResult);
  }

  // Step 6: 伪 op 清理（新增）——必须在 direct store 同步之前
  removeVectorPseudoOps();

  // Step 7: C→V direct store 同步（新增）
  processCubeToVectorDirectStoreSync(builder, flagManager, flagIdReuseManager);
  return success();
}
```

- `insertAnalyzeFlagRelations`（L1670-L1782）/ `remapInterCoreTransferFlagIds`（L1784-L1806）详见 03 篇；
- **顺序讲究**：flag 复用在 direct store 同步**之前**（后者新拿的 flag 不参与本轮复用）；伪 op 清理在 direct store 匹配**之前**（去掉 add_from_matmul 伪 op 后 yield 链路更干净，direct store 模式更容易匹配上）。

### 3.23 runOnOperation（L2313-L2333）

```cpp
void InterCoreTransferAndSyncPass::runOnOperation() {
  module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;

  FlagIdManager flagManager(module);       // Phase 1: flag 分配器（本地变量）
  FlagIdReuseManager flagIdReuseManager;   //        复用管理器

  if (failed(processDependencies(flagManager, flagIdReuseManager))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);  // Phase 2
    return;
  }
}
```

---

## 四、整体流程总结

```
1. 读取 DataDependencyInfo（V2C / C2V / C2C / MemDeps 四类依赖）
2. V→C 依赖：
   a. sortDependencies 按消费点排序
   b. 非 2D 标量且非 1D → ND→NZ 归一化（padding + reshape/transpose/reshape，插入点避开 store）
   c. 插入 copy + convertLayout + to_tensor（标量走 SSBuffer）
   d. 插入 set/wait 同步（读同步 + 主循环内的写同步 ping-pong）
3. C→V 依赖：
   a. AnalyzeSplittedIf 识别 split-if（钻到 if 内部溯源 realValue）（本次更新）
   b. fixpipe（NZ2ND / NZ2DN 转置折叠）+ memory_space_cast + to_tensor
   c. isStoreDirectlyInUserChain 判断 direct store → 选对应 pipe 配置
   d. 插入 set/wait 同步（split-if 场景 wait 贴读取点、区间收敛到 yield 前）
4. C→C 依赖（新增）：共享 L1 buffer + fixpipe NZ2NZ（pre_quant 折叠 + channelSplit）
   4a. 先处理块间 C2C（buffer 分到主循环外 + 打 kIntraDeps）
   4b. 再处理块内 C2C（buffer 插生产 matmul 后 + operand 定位 A/B 输入槽位）（本次更新）
5. 内存依赖：isOuterLayerDependency 去重后插 PIPE_MTE2 同步
6. 若 flag 超限：建关系图（E1-E4）→ 图着色复用 → 重映射
7. removeVectorPseudoOps：清理 add_from_matmul 加零伪 op（新增）
8. processCubeToVectorDirectStoreSync：为 direct store 模式补 FIX→MTE3 同步（新增）
```

---

## 五、面试要点

1. **V→C 和 C→V 为什么用不同的传输指令？**
   硬件上两个方向搬运路径不同：V→C 走搬运引擎（`hivm.copy` + MTE pipe），C→V 走 `hivm.fixpipe`（NZ2ND 模式，从 L0C 直接到 UB）。

2. **C→C 为什么又是一种做法？**
   Cube 核间的 matmul 链不必绕 UB/Vector，fixpipe NZ2NZ 直接在 L1 层直传，还能把 trunc 量化折叠进 pre_quant——"传输+量化"一次完成。且 C2C 不需要 flag 同步，依赖 L1 顺序，用 `kIntraDeps` 组号标记。

2.5 **C→C 为什么还分块间/块内？**
   - **块间 C2C**：生产/消费在不同 block，插入点要对齐到 LCA 层级，L1 buffer 分配到主循环外（跨迭代复用），必须打 `kIntraDeps` 组标记；
   - **块内 C2C**：同一 block 内 `matmul → matmul` 直链，buffer 插在生产 matmul 后即可、不上升到循环外，同块天然有序无需分组标记；
   - 关键难点在**替换**：同一个值常同时作为下一个 matmul 的 A/B 输入和 init（累加），必须用 `OpOperand` 指针精确区分，只替换输入槽位，init 保留原引用——否则累加语义被破坏。这也是 `DataDependencyAnalysis` 阶段就要保存 `dep.operand` 的原因。

2.8 **split-if 场景为什么特殊？**
   split-if 把原 block 拆成 if/else，`dep.value` 是 `scf.if` 的 result（在 if 外），真正的 matmul 在 then 块内被 yield。直接拿 `dep.value` 当 fixpipe 源会把插入点放在 if 之前、搬到错误数据；必须用 `AnalyzeSplittedIf` 溯源到 `realValue`，并用 `getOpBeforeYield` 把同步区间收敛到 yield 前。

3. **为什么需要 ND→NZ 归一化？**
   Cube 单元要求数据以 NZ 分形布局存放（对齐 16 和 32B），Vector 产生的 ND 布局必须经 reshape+transpose+reshape 转换；不够对齐时先 padding（fill 0 + insert_slice）。

4. **ping-pong 同步的机制是什么？**
   主循环内除了"生产者 set / 消费者 wait"（读同步），还有"生产者 wait / 消费者 set"（写同步）。生产者写第 N+1 次前等消费者用完第 N 次的数据，实现双缓冲重叠。

5. **为什么 wait 要精确插入（analyzeConsumerReadInsertPoint）而不是块首？**
   wait 挡住的是其后面所有 op。插在块首会让消费者块内不相关的 op 也被阻塞，降低并行度；贴着真实读取点插入，前面的 op 可以与生产者核并行执行。

6. **direct store 模式是什么？为什么要单独补同步？**
   Cube fixpipe 结果经 UB 直接 store（无 Vector 计算）时，fixpipe 走 FIX pipe、store 走 MTE3 pipe，硬件不保证两个 pipe 间的顺序，必须显式 flag：SCF region 后 set（CUBE），store 前 wait（VECTOR）。

7. **伪 op（add_from_matmul）是什么？为什么要清理？**
   上游为 SSA 合法性插入的 `x + 0` 加零 op，让数据"看似"在 Vector 侧计算过。拆分完成后是纯开销，`removeVectorPseudoOps` 把 `x + 0` 恒等替换为 `x` 并删除。

8. **内存依赖为什么单独处理？**
   它不是显式 SSA 数据流而是 load/store 到同一地址的隐式别名，必须借助别名分析发现，插 `PIPE_MTE2` 同步保证跨核访存顺序；嵌套循环下多层重复报告的依赖只处理最内层（isOuterLayerDependency）。

9. **为什么要 sortDependencies？**
   按消费点在 IR 中的位置排序处理，保证插入的传输/同步相对顺序稳定可复现，避免处理顺序不同导致非确定性结果。

10. **while 循环这次是怎么支持的？**
    `findMainLoopforTransfer` 把 `scf::WhileOp` 与 `scf::ForOp` 同等对待（缓冲分配到循环外、ping-pong 同步挂循环边界）；direct store 模式匹配也覆盖 for/while/if 三种 SCF 结构。

11. **sub-block 是什么？为什么传输插入点要跟它对齐？**
    分块循环调度（MultiCache）时，一个 block 内部还可能按数据分块切成若干 sub-block（`ssbuffer.subBlock`）。传输（归一化 copy、fixpipe 的 set/wait）若插在 sub-block 之外，会跟子块内的 op 跨子块相互阻塞；所以凡是源值/消费点带 subBlock 标记的场景，`prodEnd`/`consStart`/`newConsEnd` 都要收敛到对应子块的起点/终点（`getSubBlockStartEnd`），且新插入的 op 也要继承同一个 subBlock id。
