# DynamicCVPipeline 子 Pass 专题：核间多缓冲（03）

> 覆盖 `AddMultiBufferOuterScopePass`（[AddMultiBufferOuterScope.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.cpp)）
>
> 流水线位置：`AllocMultiCache`（L82）编排的第 2 步，`AddMultiBufferInnerScope` 之后。
>
> 职责：处理 **循环外层 scope 的核间传输**——`SplitDataflow` 已把 CUBE↔Vector 的数据交互拆成带 `kTransferId` 的传输组（wait + transfer + set + alloc/mark），本 Pass 给这些传输组分配**双缓冲**并注入**轮询控制流**：每次迭代奇偶轮换原缓冲/输出缓冲，使传输与计算重叠。

---

## 一、这个专题在做什么

`SplitDataflow` 把核间数据交互拆成了**显式传输组**：每个 `kTransferId` 对应一个完整传输（sender 侧 wait→transfer→set，receiver 侧同样），配一块核间缓冲（sender 一块、receiver 一块）和一个同步 flag。此时还是**单缓冲**：sender 写完、receiver 读完后 sender 才能重写——跨迭代串行。

`AddMultiBufferOuterScope` 把核间传输升级为**双缓冲流水**：

1. **分组**：按 `kTransferId` 收集所有传输 op，识别每组的方向（C→V / V→C）、原 flag、sender/receiver 链、缓冲 alloc/mark；
2. **分配输出缓冲**：为每组 sender/receiver 各分配一块**输出缓冲**（TCB 紧耦合缓冲），配一组输出 flag 同步 op；
3. **注入轮询**：在主循环体内构造 `scf.if` 条件链——奇偶为 0 走原缓冲（原 flag），奇偶为 1 走输出缓冲（输出 flag），把 wait/transfer/set 三个 op 都包进条件分支，实现缓冲轮换。

**核心设计**：同一 `(originalFlag, direction)` 的传输组**共享一个输出 flag**（同步位对齐，节省 flag 资源）；每个循环只构造**一份共享轮询条件**（`(iter/step)%2==0`），所有组的 scf.if 都复用它。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `kTransferId`（`ssbuffer.transfer_id`） | `SplitDataflow` 打的传输组标记，本 Pass 按它分组所有传输 op |
| `TransferGroupInfo` | 一个传输组的完整信息：`tid`、`originalFlag`、`outputFlag`、`isCtoV`、`tcbId`、sender/receiver 链（`TransferOpChain`）、extraSync（循环外同步对）、sender/receiver 缓冲（`BufferInfo`） |
| `TransferOpChain` | 一侧传输链：`waitOp` + `transferOp` + `setOp` + `toTensorOp`（receiver 侧 tensor 边界） |
| `TransferOpChainInfo` | `collectTransferChains` 的原始产物（sender/receiver 两套） |
| `ExtraSyncInfo` | 循环外（父 op 无 main_loop）的同步对：`setOp`/`waitOp` |
| `BufferInfo` | 一侧的缓冲：`allocOp` + `markOp`（annotation.mark 标记紧耦合缓冲） |
| sender / receiver | **sender** = 写核间缓冲的传输 op（Write effect 或无结果）；**receiver head** = 第一个以值为单位读核间缓冲的 op |
| `kCrossCoreDeps`（`ssbuffer.cross_core_deps`） | 核间依赖标记 `[tid, 0|1]`：`1`=producer（写），`0`=consumer（读），供下游 `AddControlFlowCondition` 生成同步 |
| `kReservedPipeFlagId = 15` | flag 15 预留流水线同步（PIPE_S），传输可用 0-14 |
| `FlagIdManager` | flag 分配器：`acquireId()` 分配，`MAX_FLAG_ID`=14，`INVALID_FLAG_ID` 表示耗尽 |
| `TCB`（`hivm.tightly_coupled_buffer`） | 紧耦合缓冲，带 id 属性，核间传输的物理缓冲，每个传输组分配新 id |
| 轮询条件 | `(iter/step)%2==0`（for）或 `counter%2==0`（while），true=原缓冲、false=输出缓冲 |
| `parentOpHasMainLoopAttr` | 同步 op 的祖先里有没有 main_loop → 区分"循环内链"和"循环外 extraSync" |
| `isInsideVectorScope` | op 是否在 Vector scope → 方向推导依据（sender 在 Vector → V→C） |

---

## 三、逐行讲解

### 3.1 入口编排：runOnOperation（L1530-L1647）

```cpp
void AddMultiBufferOuterScopePass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;   // 上游已回退 → 短路

  // 缓冲模式：InterCore 缓冲数 > 1 → 双缓冲
  int interCoreBufNum = BufferCountManager(module).getBufferCountByType(
      BufferCountManager::DepType::InterCore);
  bool isDoubleBuf = (interCoreBufNum > 1);

  // 预处理：双缓冲才需要 while 迭代计数器
  if (isDoubleBuf) preInjectWhileOpToggles(module);   // 见 3.9

  // ---- Step 1: 收集传输组 ----
  FlagIdManager flagIdMgr(module);
  DenseMap<int, SmallVector<Operation *>> opsByTid;
  collectOpsByTransferId(module, opsByTid);
  DenseMap<int, TransferGroupInfo> groups;
  if (collectTransferGroupData(module, opsByTid, flagIdMgr, groups)) {   // 失败 → fallback
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // ---- flag 预算检查 ----
  // ① 输入 flag > 15 → 本 Pass 无法产生产出 → ERRCODE_IGNORED
  // ② 输出 flag 超过 MAX_FLAG_ID(14) → 降级单缓冲
  std::set<int> usedFlags; module.walk([&](Operation *op) { /* 收集所有同步 flag */ });
  bool inputOverBudget = false;
  for (int f : usedFlags) if (f > kReservedPipeFlagId) inputOverBudget = true;
  if (inputOverBudget) { setFallbackAttr(module, ERRCODE_IGNORED); return; }
  if (isDoubleBuf) {
    int maxOutputFlag = 各组 outputFlag 的最大值;
    if (maxOutputFlag > FlagIdManager::MAX_FLAG_ID) isDoubleBuf = false;  // 降级单缓冲
  }

  // ---- 标记 llvm.load/store volatile 的 crossDeps（双缓冲模式）----
  if (isDoubleBuf) {
    collectLoadStoreOpsByTransferId(module, loadStoreByTid);
    tagLoadStoreOpsWithCrossDeps(loadStoreByTid);
  }

  // ---- Step 2: 创建输出缓冲 ----
  if (isDoubleBuf) {
    if (createOutputBuffers(groups, module)) { setFallbackAttr(module, ERRCODE_FAILED); return; }
    // ---- Step 3: 注入轮询控制流 ----
    int pollingRc = CVPipeline::ERRCODE_FAILED;
    if (failed(addPollingControlFlow(groups, pollingRc))) {
      setFallbackAttr(module, pollingRc);   // ERRCODE_FAILED 或 ERRCODE_IGNORED
      return;
    }
  } else {
    LDBG("[Step 2-3] Skipped (single-buffer mode).");
  }
}
```

**要点**：
- **`InterCore` 缓冲数决定是否双缓冲**：>1 才做 Step 2/3，否则整个 Pass 只做 Step 1 收集（无实际改动）；
- **flag 预算两道闸**：输入 flag 超过 15（流水线预留位）→ kernel 本身 flag 超量，`ERRCODE_IGNORED`；输出 flag 估算超 14 → 资源不足，**降级单缓冲**（不报错，保底正确性）；
- load/store volatile 的 `crossDeps` 标记（见 3.6）只对双缓冲模式有意义。

### 3.2 Step 1a：按 transfer_id 分组（L213-L239）

```cpp
static int collectOpsByTransferId(ModuleOp module,
                                  DenseMap<int, SmallVector<Operation *>> &opsByTid) {
  module.walk([&](Operation *op) {
    if (!op->hasAttr(mlir::CVPipeline::kTransferId)) return;
    int tid = getTransferId(op);
    if (tid >= 0) opsByTid[tid].push_back(op);
  });
  return 0;
}
```

**要点**：`getTransferId`（L62-L66）读 `ssbuffer.transfer_id` 属性。所有带标记的 op（wait/transfer/set/alloc/mark/load/store）按 tid 聚组。负 tid 丢弃。

### 3.3 Step 1b：单组信息构建 buildTransferGroupData（L498-L579）

```cpp
static int buildTransferGroupData(int tid, const SmallVector<Operation *> &ops,
                                  TransferGroupInfo &info) {
  info.tid = tid;

  // 1. 确定原 flag：组里第一个带 flag 的同步 op
  for (Operation *op : ops)
    if (isa<SyncBlockSetOp>(op) || isa<SyncBlockWaitOp>(op))
      if (int f = getFlagFromSyncOp(op); f >= 0) { info.originalFlag = f; break; }

  // 2. 循环外同步对（父 op 无 main_loop）
  ExtraSyncInfo extraInfo;
  collectExtraSync(ops, info.originalFlag, extraInfo);      // 见 3.4
  info.extraSyncSetOp = extraInfo.setOp;
  info.extraSyncWaitOp = extraInfo.waitOp;

  // 3. 循环内传输链（父 op 有 main_loop）
  TransferChainInfo chainInfo;
  collectTransferChains(ops, info.originalFlag, chainInfo);  // 见 3.5
  info.senderChain = chainInfo.sender;
  info.receiverChain = chainInfo.receiver;

  // 3.5 链完整性检查：sender/receiver 都要 wait+set 齐全，
  //     只包半握手（producer 换 flag 而 consumer 不换）会死锁 → 整组降级单缓冲
  auto chainComplete = [](const TransferOpChain &chain) {
    return !chain.transferOp || (chain.waitOp && chain.setOp);
  };
  if (!chainComplete(info.senderChain) || !chainComplete(info.receiverChain)) {
    info.senderChain = TransferOpChain();
    info.receiverChain = TransferOpChain();
  }

  // 4. 方向：sender 在 Vector scope → V→C；否则 C→V；无 sender 用 receiver 反推
  if (info.senderChain.transferOp)
    info.isCtoV = !isInsideVectorScope(info.senderChain.transferOp);
  else if (info.receiverChain.transferOp)
    info.isCtoV = isInsideVectorScope(info.receiverChain.transferOp);

  // 5. 收集缓冲 alloc/mark 对（必须跑在传输链收集之后，才能识别真正的核间缓冲，
  //    忽略核内局部缓冲如 CUBE 侧 cc）
  if (collectBufferAllocs(ops, info)) return -1;

  // 6. 输出 flag 在 collectTransferGroupData 统一分配（同 flag+方向共享）
  return 0;
}
```

**要点**：
- **链完整性检查**是关键正确性保护：双缓冲轮换要求 producer 和 consumer **同步换 flag**。若一侧缺 wait/set（只包半握手），轮换会导致握手不匹配死锁 → 整组清空链（保持单缓冲）；
- **方向推导**：sender 执行侧决定方向。sender 在 Vector scope 说明数据从 Vector 发出（V→C），在 CUBE scope 则是 C→V；无 sender 时用 receiver 侧反推（receiver 在 Vector → 数据进 Vector → V→C）；
- `getFlagFromSyncOp`（L43-L54）兼容 `flag_id` / `static_flag_id` / `flag` 三种属性名。

### 3.4 循环外同步对：collectExtraSync（L368-L418）

```cpp
static int collectExtraSync(const SmallVector<Operation *> &ops,
                            int originalFlag, ExtraSyncInfo &info) {
  // 收集父 op 无 main_loop 的 SyncBlockSet / SyncBlockWait
  for (Operation *op : ops) {
    if (!(isa<SyncBlockSetOp>(op) || isa<SyncBlockWaitOp>(op))) continue;
    bool hasMainLoop = parentOpHasMainLoopAttr(op);   // findMainLoopOp 找祖先
    if (!hasMainLoop) { extraSets / extraWaits 分类入桶; }
  }
  // 按原 flag 精确配对
  for (auto *setOp : extraSets)
    for (auto *waitOp : extraWaits)
      if (flag(setOp) == originalFlag && flag(waitOp) == originalFlag) {
        info.setOp = setOp; info.waitOp = waitOp; return 0;
      }
  // 兜底：无精确匹配用第一对
  ...
}
```

**要点**：`parentOpHasMainLoopAttr`（L87-L92）用 `findMainLoopOp`（L81-L84）向上找带 `kMainLoop` 的祖先。**extraSync 是循环外**的同步（如 scope 级初始化同步），它们的输出 flag 同步 op 要插在**循环外**对应位置（`createOutputBufferForGroup` 中处理）。

### 3.5 循环内传输链：collectTransferChains（L425-L495）

```cpp
static int collectTransferChains(const SmallVector<Operation *> &ops,
                                 int originalFlag, TransferChainInfo &info) {
  // 组缓冲：带 transfer_id 的 alloc（sender/receiver 各一块物理缓冲）
  SmallVector<Value> groupBuffers;
  for (Operation *op : ops)
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) groupBuffers.push_back(allocOp.getResult());

  for (Operation *op : ops) {
    if (同步 op || !op->getBlock()) continue;
    if (alloc/mark) continue;                 // 数据而非传输行为
    if (!parentOpHasMainLoopAttr(op)) continue; // 只看循环内的传输 op
    // 找第一个操作数是组缓冲的操作数
    Value bufferOperand = ...isGroupBuffer(operand) 查找...;
    if (!bufferOperand) continue;

    if (op->getNumResults() == 0 || hasWriteEffectOn(op, bufferOperand)) {
      // 写组缓冲 → sender
      if (info.sender.transferOp) continue;   // 只取第一个
      info.sender.transferOp = op;
      info.sender.bufferOperand = bufferOperand;
      info.sender.waitOp = findSyncOpWithFlag(block, op, originalFlag, false, true);  // 向前找 wait
      info.sender.setOp   = findSyncOpWithFlag(block, op, originalFlag, true,  false); // 向后找 set
    } else if (!info.receiver.transferOp && op->getNumResults() > 0) {
      // 读组缓冲 → receiver head
      info.receiver.transferOp = op;
      info.receiver.bufferOperand = bufferOperand;
      info.receiver.waitOp = findSyncOpWithFlag(block, op, originalFlag, false, true);
      info.receiver.setOp   = findSyncOpWithFlag(block, op, originalFlag, true,  false);
      info.receiver.toTensorOp = findChainTensorTerminal(op);   // 沿单 use memref 链找 tensor 边界
    }
  }
  return 0;
}
```

**要点**：
- **sender 判定**：无结果（如 `hivm.copy` 写缓冲）或 `hasWriteEffectOn`（L108-L119，MemoryEffectOpInterface 声明对缓冲 Write）→ 写侧；
- **receiver 判定**：有结果且读组缓冲 → 读侧 head；`findChainTensorTerminal`（L126-L149）沿**单 use memref 链**（不 fork、不跨 block、每 op 单结果）找到第一个产出 tensor 的 op（如 `bufferization.to_tensor`）——这是 receiver 的 tensor 边界，轮询包装要把整条链包进 if；
- `findSyncOpWithFlag`（L163-L206）：块内双向搜索同 flag 的 wait/set，是定位握手点的关键工具。

### 3.6 llvm.load/store volatile 的 crossDeps 标记（L314-L365）

```cpp
static int tagLoadStoreOpsWithCrossDeps(DenseMap<int, SmallVector<Operation *>> &loadStoreByTid) {
  for (auto &p : loadStoreByTid) {
    int tid = p.first;
    for (auto *op : p.second) {
      if (auto storeOp = dyn_cast<mlir::LLVM::StoreOp>(op)) {
        // producer：crossDeps = {tid, 1}，打在 store 的 ptr 定义 op 上
        Value ptr = storeOp.getOperand(1);
        if (auto *ptrDefOp = ptr.getDefiningOp())
          ptrDefOp->setAttr(kCrossCoreDeps, {tid, 1});
      } else if (auto loadOp = dyn_cast<mlir::LLVM::LoadOp>(op)) {
        // consumer：crossDeps = {tid, 0}，打在 load 自身
        op->setAttr(kCrossCoreDeps, {tid, 0});
      }
    }
  }
}
```

**要点**：volatile load/store 是核间搬移的 LLVM 层表现形式（volatile 语义保证跨核可见）。**producer 标记打在 store 的指针定义 op**（store 自身不打，因为 store 的"写"行为属于 ptr 的产出），consumer 标记打在 load 自身。供下游 `AddControlFlowCondition` 识别真正的读写依赖。

### 3.7 Step 1c：分组汇总 + 输出 flag 共享（L582-L622）

```cpp
static int collectTransferGroupData(ModuleOp module, opsByTid, flagIdMgr, groups) {
  for (auto &p : opsByTid) {
    TransferGroupInfo info;
    if (buildTransferGroupData(p.first, p.second, info)) continue;
    if (info.senderChain.transferOp || info.receiverChain.transferOp)
      groups[p.first] = info;                 // 只有链完整的组才进入
  }

  // 输出 flag 复用：同一 (originalFlag, direction) 共享一个输出 flag
  std::map<std::pair<int, bool>, int> outputFlagByKey;   // key = (originalFlag, isCtoV)
  for (auto &p : groups)
    outputFlagByKey.try_emplace({p.second.originalFlag, p.second.isCtoV},
                                FlagIdManager::INVALID_FLAG_ID);
  for (auto &entry : outputFlagByKey) {
    for (int attempt = 0; attempt < kMaxFlagAttempts; ++attempt) {
      int64_t pf = flagIdMgr.acquireId();
      if (pf == FlagIdManager::INVALID_FLAG_ID || pf != entry.first.first) {
        entry.second = static_cast<int>(pf);  // 分配新 flag（跳过与原 flag 相同值）
        break;
      }
    }
  }
  for (auto &p : groups)
    p.second.outputFlag = outputFlagByKey[{p.second.originalFlag, p.second.isCtoV}];
  return 0;
}
```

**要点**：
- **输出 flag 共享**：多个传输共用同一原始 flag + 方向时，它们用同一块输出 flag —— 因为同方向的传输同步位天然对齐，共享 flag 不会冲突（省 flag 资源，flag 0-14 只有 15 个）；
- `acquireId()` 分配时**跳过与原 flag 相同的 id**（避免输出 flag 撞原 flag）；
- `std::map` 按键排序遍历保证**确定性分配**（同一输入多次运行结果一致）。

### 3.8 Step 2：创建输出缓冲（L630-L801）

```cpp
static int createOutputBuffers(DenseMap<int, TransferGroupInfo> &groups, ModuleOp module) {
  // 收集已有 TCB id，新 id 从 max+1 开始
  std::set<int> usedTcbIds;
  module.walk([&](Operation *op) {
    if (auto tcbAttr = op->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>("hivm.tightly_coupled_buffer"))
      if (auto id = tcbAttr.getId(); id.has_value()) usedTcbIds.insert(id.value());
  });
  int nextTcbId = usedTcbIds.empty() ? 0 : *usedTcbIds.rbegin() + 1;

  for (auto &p : groups) {
    TransferGroupInfo &g = p.second;
    g.tcbId = allocateNewTcbId(nextTcbId, usedTcbIds);   // 分配 TCB id
    nextTcbId = g.tcbId + 1;
    createOutputBufferForGroup(g, builder);              // 见下
  }
  return 0;
}

static int createOutputBufferForGroup(TransferGroupInfo &g, OpBuilder &builder) {
  // ① sender 输出缓冲
  createOutputBufferPair(g.senderBuf.allocOp, g.tid, g.tcbId,
                         g.senderInputBuffer, g.senderOutputBuffer, builder, true);
  // ② receiver 输出缓冲
  createOutputBufferPair(g.receiverBuf.allocOp, g.tid, g.tcbId,
                         g.receiverInputBuffer, g.receiverOutputBuffer, builder, false);
  // ③ 输出同步 set 插在 extraSync.setOp 位置（sender scope）
  if (g.extraSyncSetOp) createOutputSyncSetOp(g.extraSyncSetOp, g.outputFlag, g.tid, builder);
  // ④ 输出同步 wait 插在 extraSync.waitOp（或 receiverChain.waitOp）位置（receiver scope）
  Operation *outputWaitInsertOp = g.extraSyncWaitOp ? g.extraSyncWaitOp : g.receiverChain.waitOp;
  if (outputWaitInsertOp) createOutputSyncWaitOp(outputWaitInsertOp, g.outputFlag, g.tid, builder);
  return 0;
}

// 单侧输出缓冲：alloc + mark（紧耦合缓冲标记）
static int createOutputBufferPair(Operation *inputAllocOp, int tid, int tcbId,
                                  Value &inputBuffer, Value &outputBuffer, OpBuilder &builder, bool isSender) {
  inputBuffer = inputAllocOp->getResult(0);
  auto memRefType = dyn_cast<MemRefType>(inputBuffer.getType());
  builder.setInsertionPointAfter(inputAllocOp);
  auto outputAlloc = builder.create<memref::AllocOp>(loc, memRefType);
  outputAlloc->setAttr(kBlockId, getBlockId(inputAllocOp));   // 继承原块 id
  outputAlloc->setAttr(kTransferId, tid);
  outputBuffer = outputAlloc.getResult();
  auto outputMark = builder.create<annotation::MarkOp>(loc, outputBuffer);
  outputMark->setAttr("effects", {"write", "read"});
  outputMark->setAttr(kBlockId, ...);
  outputMark->setAttr(kTransferId, tid);
  outputMark->setAttr("hivm.tightly_coupled_buffer", HIVMTightlyCoupledBufferAttr::get(ctx, tcbId));
  return 0;
}
```

**要点**：
- **每个传输组分配独立 TCB id**（紧耦合缓冲），输出缓冲 `alloc` 克隆原缓冲类型，紧跟原 alloc 之后；
- **输出同步 op**：`createOutputSyncSetOp`（L700-L710）克隆原 setOp 但 flag 换成 `outputFlag`，插在原 extraSync 后；wait 同理。这些是**循环外**的输出 flag 同步（轮询 if 是循环内的，见 Step 3）；
- 输出 alloc **不带 crossDeps**（alloc 是缓冲创建 op 而非行为 op）——producer 标记打在 if 内的 transfer 克隆上，consumer 标记打在 if 包装上（见 3.11）。

### 3.9 预处理：while 迭代计数器注入（L1502-L1524, L840-L956）

```cpp
// 主循环 while 且含传输 op → 注入计数器
static void preInjectWhileOpToggles(ModuleOp module) {
  SmallVector<scf::WhileOp> whileOps;
  module.walk([&](scf::WhileOp whileOp) {
    if (!CVPipeline::isMainLoopOp(whileOp)) return;
    bool hasTransferOps = false;
    whileOp.walk([&](Operation *op) {
      if (op->hasAttr(mlir::CVPipeline::kTransferId)) { hasTransferOps = true; return interrupt(); }
      return advance();
    });
    if (hasTransferOps) whileOps.push_back(whileOp);
  });
  for (auto whileOp : whileOps) ensureWhileOpHasCounter(whileOp);   // 见下
}

// 确保 while 有 i32 迭代计数器
static Value ensureWhileOpHasCounter(scf::WhileOp whileOp) {
  if (whileOp->hasAttr(CVPipeline::kIterCounter)) {
    // 已注入（InnerScope 注入过）→ 复用：把 body 末尾的 counter 更新移到 body 头部，
    // 重新打第一个 block_id（轮询链要在 body 头部构造，counter 更新必须在轮询 remsi 前）
    ...
    return after.getArgument(after.getNumArguments() - 1);
  }
  // 未注入 → 重建 whileOp（同 InnerScope 的 setupWhileIterArgCounter 模式）
  //   init 追加 0，resultTypes 追加 i32
  //   before: 克隆旧 op + condition 追加 counter 前向参数
  //   after:  counter+1 插在 body 头部（为轮询链做准备），克隆旧 op，yield 追加 nextCounter
  //   counterOne/counterAdd 打第一个 body block_id + kIterCounter 标记
  ...
  return counterIterArg;
}
```

**要点**：
- **计数器更新位置**：`ensureWhileOpHasCounter` 把 `counter+1` 放在 **body 头部**（而非 InnerScope 的 body 末尾）——轮询链（remsi/cmpi）要在 body 头部围绕 counter 构造，`+1` 必须在 remsi **之前**（先算本迭代计数、再自增），这样 `counter%2` 才是本迭代的奇偶；
- **复用 vs 重建**：`kIterCounter` 已存在（InnerScope 已注入）→ 只做"更新位置归一化 + 重新打块 id"；否则整体重建；
- 更新 op 的 block_id 取 **body 第一个带 block_id 的 op**（CloneOps 连续性）。

### 3.10 Step 3a：轮询条件构造 prepareLoopPolling（L1316-L1383）

```cpp
static Value prepareLoopPolling(Operation *loopOp, Operation *waitOp, OpBuilder &builderOut) {
  int bid = getBlockId(waitOp);
  int tid = getTransferId(waitOp);

  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    builderOut.setInsertionPointToStart(forOp.getBody());   // body 头部
    // block_id 取 body 第一个带块 id 的 op（CloneOps 连续性），没有则用 wait 的
    std::optional<int> pollBlockId = getFirstBlockId(forOp.getBody());
    if (!pollBlockId) pollBlockId = bid;
    Value cond = createPollingCondition(forOp, condBuilder, *pollBlockId, tid);
    return cond;
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    if (!whileOp->hasAttr(CVPipeline::kIterCounter)) return Value();  // 非主循环 while → 无计数器，失败
    Block &after = whileOp.getAfter().front();
    Value counter = after.getArgument(after.getNumArguments() - 1);
    // 定位 counter 更新（body 头部）
    Operation *lastCounterUpdate = ...找 kIterCounter op...;
    bool counterAtHead = (lastCounterUpdate 在 body 头部);
    builderOut.setInsertionPointToStart(&after);          // body 头部插轮询链
    // c2 = 常量 2; rem = counter % 2
    // c0 = 常量 0; cond = (rem == 0)
    // c0+cmpi 插在 counter 更新之后（+1 落在 remsi 和 cmpi 之间，后端顺序已验证）
    if (counterAtHead) condBuilder.setInsertionPointAfter(lastCounterUpdate);
    ...
    return condOp->getResult(0);
  }
  return Value();   // 意外循环类型 → 调用方 ERRCODE_IGNORED
}

// for 的轮询条件：(iter/step) % 2 == 0
static Value createPollingCondition(scf::ForOp forOp, OpBuilder &builder, int blockId, int tid) {
  Value iterVar = forOp.getInductionVar();
  Value step = forOp.getStep();
  auto divOp = builder.create<arith::DivSIOp>(loc, iterVar, step);   // iter/step
  auto c2Val = builder.create<arith::ConstantIntOp>(loc, 2, bitWidth);
  auto remOp = builder.create<arith::RemSIOp>(loc, divOp, c2Val);    // % 2
  auto c0Val = builder.create<arith::ConstantIntOp>(loc, 0, bitWidth);
  auto cmpOp = builder.create<arith::CmpIOp>(loc, eq, remOp, c0Val); // == 0
  // 所有 op 打 ssbuffer 标记（block_id + transfer_id）
  return cmpOp.getResult();
}
```

**要点**：
- **for 与 while 轮询条件不同**：for 用 `(iter/step)%2==0`（归纳变量推导迭代序号）；while 用 `counter%2==0`（注入的循环承载计数）；
- **while 的 op 顺序有讲究**：`counter%2` 的 remsi 必须在 counter 更新（+1）**之前**（否则算的是下一迭代的奇偶），而 c0+cmpi 放在 +1 **之后**（后端验证过的指令顺序）——`+1` 恰好落在 remsi 和 cmpi 之间；
- 所有轮询 op 打 `kBlockId` + `kTransferId`，保证块归属与传输组关联。

### 3.11 Step 3b：共享轮询条件 getOrCreateLoopCond（L1390-L1403）

```cpp
static Value getOrCreateLoopCond(Operation *loopOp, Operation *anchorWait,
                                 DenseMap<Operation *, Value> &condCache) {
  auto it = condCache.find(loopOp);
  if (it != condCache.end()) return it->second;    // 同一循环复用同一条件
  OpBuilder builder(loopOp->getContext());
  Value cond = prepareLoopPolling(loopOp, anchorWait, builder);
  if (cond) condCache[loopOp] = cond;
  return cond;
}
```

**要点**：`(iter%2)==0` 对所有传输组**完全相同**（都按同一迭代奇偶轮换），所以**每个循环只需构造一份条件**，所有组的 scf.if 复用。`anchorWait` 是循环内**最早的 wait**（保证条件 op 支配所有组的 if 包装）。

### 3.12 Step 3c：循环锚定 addPollingControlFlow（L1413-L1493）

```cpp
static LogicalResult addPollingControlFlow(DenseMap<int, TransferGroupInfo> &groups, int &errCode) {
  // 每循环选最早 wait 作为锚（共享条件必须支配所有组的 scf.if）
  DenseMap<Operation *, Operation *> loopAnchor;
  for (auto &[tid, group] : groups) {
    auto track = [&](Operation *waitOp) {
      Operation *loop = resolveLoopOp(waitOp);          // 最近的 for/while 祖先
      Operation *anchor = loopAnchor.lookup(loop);
      if (!anchor || (同 block && waitOp 更早)) loopAnchor[loop] = waitOp;
    };
    track(group.senderChain.waitOp);
    track(group.receiverChain.waitOp);
  }

  DenseMap<Operation *, Value> condCache;
  for (auto &[tid, group] : groups) {
    // sender 循环 + 共享条件
    Operation *senderWaitParent = resolveLoopOp(group.senderChain.waitOp);
    Value senderCond = getOrCreateLoopCond(senderWaitParent, loopAnchor[senderWaitParent], condCache);
    if (!senderCond) { errCode = ERRCODE_IGNORED; return failure(); }   // 意外循环类型

    // 处理 sender 链（producer）
    if (failed(processTransferChain(group.senderChain, senderCond,
                                    group.senderOutputBuffer, group.outputFlag,
                                    true /*isProducer*/, senderBuilder))) {
      errCode = ERRCODE_FAILED; return failure();
    }
    // 处理 receiver 链（consumer，可能在不同循环）
    if (group.receiverChain.waitOp) {
      Operation *receiverWaitParent = resolveLoopOp(group.receiverChain.waitOp);
      if (receiverWaitParent == senderWaitParent) {
        // 同一循环：复用 senderCond
        processTransferChain(group.receiverChain, senderCond, group.receiverOutputBuffer,
                             group.outputFlag, false, senderBuilder);
      } else {
        // 不同循环：各自循环构造条件
        Value receiverCond = getOrCreateLoopCond(receiverWaitParent, loopAnchor[receiverWaitParent], condCache);
        processTransferChain(group.receiverChain, receiverCond, group.receiverOutputBuffer,
                             group.outputFlag, false, receiverBuilder);
      }
    }
  }
  return success();
}
```

**要点**：
- **sender/receiver 可能在不同循环**（C→V 时 sender 在 CUBE 主循环、receiver 在 Vector 主循环）→ 各自循环构造轮询条件；
- **锚 wait 保证支配性**：条件构造在循环体头部、anchor 是最早的 wait，因此条件 SSA 值支配所有组的 if 包装；
- `resolveLoopOp`（L1407-L1411）找最近的 for/while 祖先（sync 可能在 scf.if 内；while 嵌套在 for 里解析为 while）。

### 3.13 链包装：processTransferChain（L1245-L1311）

```cpp
static LogicalResult processTransferChain(TransferOpChain &chain, Value cond,
                                          Value outputBuffer, int outputFlag,
                                          bool isProducer, OpBuilder &builder) {
  if (!chain.waitOp) return failure();

  // 1. waitOp 包进 if：then=clone 原 wait，else=新建 outputFlag 的 wait
  chain.waitOp = wrapSyncOpWithScfIf<hivm::SyncBlockWaitOp>(
      chain.waitOp, cond, outputFlag, builder,
      [&](OpBuilder &b, Location l) -> Operation * {
        return b.create<hivm::SyncBlockWaitOp>(l, waitOp.getTcoreType(), waitOp.getTpipe(),
                                               waitOp.getPipe(), b.getI64IntegerAttr(outputFlag)).getOperation();
      });

  // 2. transferOp 包进 if（then=原缓冲，else=outputBuffer）
  if (chain.transferOp) {
    int bid = getBlockId(chain.transferOp);
    int tid = getTransferId(chain.transferOp);
    if (!isProducer && chain.toTensorOp) {
      // receiver 有 tensor 边界：整条链（transfer→trailing ops→to_tensor）包进 if 返回 tensor
      chain.transferOp = wrapReceiverChainWithScfIf(chain.transferOp, chain.toTensorOp, cond,
                                                    chain.bufferOperand, outputBuffer, bid, tid, builder);
      chain.toTensorOp = nullptr;
    } else {
      bool hasExternalUses = transferOp 有外部 uses;
      chain.transferOp = hasExternalUses
          ? wrapTransferOpWithScfIfYield(...)    // 有外部 use：if 带 yield 返回结果
          : wrapTransferOpWithScfIfSimple(...);  // 无外部 use：纯副作用
    }
  }

  // 3. setOp 包进 if：then=clone 原 set，else=新建 outputFlag 的 set
  if (chain.setOp) {
    chain.setOp = wrapSyncOpWithScfIf<hivm::SyncBlockSetOp>(chain.setOp, cond, outputFlag, builder, ...);
  }
  return success();
}
```

**要点**：**wait / transfer / set 三者必须一起轮换**——这是双缓冲正确性的核心。任何一侧只换一半（如 producer 换 flag、consumer 不换）都会握手失配死锁。

### 3.14 三种 scf.if 包装（L988-L1242）

```cpp
// ① 同步 op 包装：then=clone 原 op（原 flag），else=createAltFn（outputFlag）
template <typename OpTy>
static Operation *wrapSyncOpWithScfIf(Operation *op, Value cond, int outputFlag,
                                      OpBuilder &builder, std::function<Operation *(OpBuilder &, Location)> createAltFn) {
  auto ifOp = builder.create<scf::IfOp>(loc, TypeRange{}, cond, true);
  ifOp->setAttr(kBlockId, getBlockId(op));
  ifOp->setAttr("ssbuffer.cross_buffer", 1);
  auto thenBuilder = ifOp.getThenBodyBuilder();
  Operation *cloned = thenBuilder.clone(*op);          // then: 原 op
  auto elseBuilder = ifOp.getElseBodyBuilder();
  Operation *altOp = createAltFn(elseBuilder, loc);    // else: outputFlag 版本
  // 复制 ssbuffer 标记到 cloned/altOp
  op->replaceAllUsesWith(ifOp.getOperation());
  op->erase();
  return ifOp.getOperation();
}

// ② transfer op（有外部 use）：if 带结果，then=原缓冲 else=outputBuffer，yield 返回
static Operation *wrapTransferOpWithScfIfYield(Operation *transferOp, Value cond, Value bufferOperand,
                                               Value outputBuffer, int bid, int tid, bool isProducer, OpBuilder &builder) {
  auto ifOp = builder.create<scf::IfOp>(loc, transferOp->getResultTypes(), cond, true);
  // then: clone(transferOp) 原缓冲 + yield
  // else: clone(transferOp, {bufferOperand→outputBuffer}) + yield
  if (isProducer) {
    // producer 标记打在克隆的 transferOp 上（真正的行为 op）：[tid, 1]
    thenCloned->setAttr(kCrossCoreDeps, {tid, 1});
    elseCloned->setAttr(kCrossCoreDeps, {tid, 1});
  }
  ifOp->setAttr(kBlockId, bid);
  ifOp->setAttr(kTransferId, tid);
  ifOp->setAttr("ssbuffer.cross_buffer", 1);
  // 替换 transferOp 的所有结果 uses
  return ifOp.getOperation();
}

// ③ receiver 链（transfer→trailing ops→to_tensor）：整条包进 if 返回 tensor
static Operation *wrapReceiverChainWithScfIf(Operation *transferOp, Operation *toTensorOp, Value cond,
                                             Value bufferOperand, Value outputBuffer, int bid, int tid, OpBuilder &builder) {
  // 收集 transferOp 到 toTensorOp 之间的 trailing ops（结果流入 to_tensor 的中间 op）
  SmallVector<Operation *> trailingOps;
  ...
  auto tensorType = toTensorOp->getResult(0).getType();
  auto ifOp = builder.create<scf::IfOp>(loc, tensorType, cond, true);
  // then: clone 整条链（原缓冲）+ yield tensor
  //   clonedTransfer 要去掉 crossDeps（consumer 角色由 ifOp 包装承担）
  // else: clone 整条链（outputBuffer 替换 bufferOperand）+ yield tensor
  ifOp->setAttr(kBlockId, bid);
  ifOp->setAttr(kTransferId, tid);
  ifOp->setAttr("ssbuffer.cross_buffer", 1);
  // consumer 角色单一来源：ifOp 包装打 [tid, 0]
  ifOp->setAttr(kCrossCoreDeps, {tid, 0});
  // 从外到内替换/删除：toTensorOp → trailingOps(逆序) → transferOp
  toTensorOp->getResult(0).replaceAllUsesWith(ifOp.getResult(0));
  ...
  return ifOp.getOperation();
}
```

**要点**：
- **三种包装的标记哲学**：producer 角色（`[tid,1]`）打在 if 内**真正的行为 op**（fixpipe/copy 克隆）上；consumer 角色（`[tid,0]`）打在 **ifOp 包装**上——因为 receiver 的 if 是"选缓冲并产出 tensor"的控制结构，消费行为属于它；scf.if 本身只是轮询分发结构，不打 crossDeps；
- receiver 链克隆时**必须剥离** clone() 继承来的 crossDeps（原 transferOp 可能带上游 `[tid,0]` 标记），避免双重角色；
- `wrapReceiverChainWithScfIf` 的删除顺序：从外到内（toTensor → trailing → transfer），避免 use-after-free。

---

## 四、流程总结

### 4.1 核间双缓冲完整流水

```
AddMultiBufferOuterScopePass
  ├─► isDoubleBuf = (InterCore 缓冲数 > 1)
  ├─► [预处理] 双缓冲时 preInjectWhileOpToggles
  │       └─► 含传输 op 的 while 主循环 → ensureWhileOpHasCounter
  │              ├─► 已有 kIterCounter（InnerScope 注入）→ 归一化到 body 头部 + 重打块 id
  │              └─► 无 → 重建 while 追加 i32 counter
  ├─► [Step 1] 收集传输组
  │       ├─► collectOpsByTransferId      按 kTransferId 分组
  │       └─► collectTransferGroupData    每组：原 flag / extraSync / 传输链 / 方向 / 缓冲
  │              ├─► 链完整性检查（缺 wait/set → 降级单缓冲）
  │              ├─► 方向推导（sender 在 Vector → V→C）
  │              └─► 输出 flag 共享（同 originalFlag+方向 复用）
  ├─► flag 预算检查
  │       ├─► 输入 flag > 15 → ERRCODE_IGNORED
  │       └─► 输出 flag > 14 → 降级单缓冲
  ├─► 标记 llvm.load/store volatile 的 crossDeps
  ├─► [Step 2] createOutputBuffers
  │       └─► 每组：sender/receiver 输出 alloc + mark（新 TCB id）
  │            + 输出 flag 的 set/wait（插在 extraSync 位置）
  └─► [Step 3] addPollingControlFlow
          ├─► 每循环选最早 wait 为锚
          ├─► 每循环构造共享轮询条件（for: (iter/step)%2==0；while: counter%2==0）
          └─► 每组：sender 链 + receiver 链各自包装
                 wait → if(then=原, else=outputFlag)
                 transfer → if(then=原缓冲, else=outputBuffer)（producer 打 [tid,1]）
                 receiver 有 to_tensor → 整条链包 if 返回 tensor（if 打 [tid,0]）
                 set → if(then=原, else=outputFlag)
```

### 4.2 双缓冲轮换语义（单次迭代）

```
迭代 i（i%2==0）:
  原缓冲路径:  wait(原flag) → transfer(原缓冲) → set(原flag)
迭代 i（i%2==1）:
  输出缓冲路径: wait(输出flag) → transfer(输出缓冲) → set(输出flag)
```

- producer 迭代 i 写原缓冲、i+1 写输出缓冲、i+2 再写原缓冲……consumer 同步轮换读取；
- 因 producer 和 consumer 错开一拍，producer 写第 i+1 块时 consumer 还在读第 i 块 → 流水重叠。

### 4.3 与前后 Pass 的协作

- **前置**：`SplitDataflow` 打的 `kTransferId`（分组）、`kCrossCoreDeps`（load/store 标记前提）、原 flag/缓冲 alloc/mark；`AddMultiBufferInnerScope` 注入的 `kIterCounter`（while 计数器复用）；
- **产出**：输出缓冲（新 alloc + TCB + 输出 flag 同步）+ 轮询 scf.if 结构 + `kCrossCoreDeps` 标记（producer 在 transfer 克隆、consumer 在 receiver if、load/store 单独打）；
- **后置**：`AddControlFlowCondition`（L83）消费 `kCrossCoreDeps` 生成真正同步控制流；`SeparateMemoryFromCompute` 识别缓冲读写。

---

## 五、面试要点

1. **Outer Scope 处理什么？** 循环外层 scope 的**核间传输**（CUBE↔Vector 的数据交互，`SplitDataflow` 已拆成带 `kTransferId` 的传输组）。给它分配双缓冲 + 注入轮询控制流，让跨核传输与计算重叠。

2. **传输组怎么识别？** 按 `kTransferId` 属性分组。每组包含 wait/transfer/set 同步链 + alloc/mark 缓冲。**sender** = 写组缓冲的 op（Write effect 或无结果）；**receiver head** = 第一个以值为单位读组缓冲的 op；receiver 沿单 use memref 链找 `to_tensor` 作为 tensor 边界。

3. **方向如何推导？** sender transfer op 所在 scope：Vector scope → `V→C`；否则 `C→V`。无 sender 时用 receiver 的 scope 反推（receiver 在 Vector → 数据进 Vector → V→C）。方向决定输出 flag 复用键 `(originalFlag, isCtoV)`。

4. **输出 flag 为什么共享？** flag 资源只有 0-14（15 个，15 预留流水线）。同一 `(originalFlag, direction)` 的传输组同步位天然对齐，共享一个输出 flag 不冲突还省资源。分配时跳过与原 flag 相同的 id。

5. **链完整性检查解决什么问题？** 双缓冲轮换要求 producer 和 consumer **同步换 flag**。若一侧缺 wait/set（只包半握手），轮换会造成握手不匹配死锁。所以链不完整的组整体降级单缓冲。

6. **for 和 while 的轮询条件差异？** for 用归纳变量推导 `(iter/step)%2==0`；while 没有归纳变量，靠 `ensureWhileOpHasCounter` 注入 i32 counter，条件为 `counter%2==0`。while 的 op 顺序有讲究：`counter%2` 的 remsi 在 `counter+1` **之前**（算本迭代奇偶），c0+cmpi 在 `+1` 之后（后端验证过的指令顺序）。

7. **为什么每个循环只构造一份共享轮询条件？** 所有传输组都按**同一迭代奇偶**轮换（`%2==0` 对所有组相同），一份条件即可；锚 wait（循环内最早 wait）保证条件 SSA 值支配所有组的 scf.if 包装。sender 和 receiver 若在不同循环，则各构造一份。

8. **三种 scf.if 包装有什么区别？** ① `wrapSyncOpWithScfIf`：同步 op（wait/set），then=原 flag 克隆、else=新建 outputFlag 版本，无结果；② `wrapTransferOpWithScfIfYield`：有外部 use 的 transfer，if 带 yield 返回结果；`wrapTransferOpWithScfIfSimple`：无外部 use 的纯副作用 transfer；③ `wrapReceiverChainWithScfIf`：receiver 整条链（transfer→trailing ops→to_tensor）包进 if 返回 tensor。

9. **producer/consumer 角色标记打在哪？** producer 角色（`[tid,1]`）打在 if 内**真正的行为 op**（fixpipe/copy 克隆）上；consumer 角色（`[tid,0]`）打在 **receiver 的 ifOp 包装**上（if 是选缓冲并产出 tensor 的控制结构）。scf.if 本身只带 `ssbuffer.cross_buffer` 不做行为标记。receiver 链克隆必须剥离继承的 crossDeps 避免双重角色。

10. **与 Inner Scope 的分工？** Inner 处理**同一核内**主循环内部的跨块 tensor 依赖（UB 缓冲，`kIntraDeps`）；Outer 处理**跨核**的传输组（TCB 双缓冲，`kCrossCoreDeps` + 输出 flag）。Outer 复用 Inner 注入的 while 计数器（`kIterCounter`），先内后外保证计数器先就位。

11. **flag 预算两道闸的作用？** 输入 flag 超过 15（流水线预留位）说明 kernel 同步数量超硬件容量 → `ERRCODE_IGNORED`（上游问题，本 Pass 无法补救）；输出 flag 估算超过 14 → 资源不足 → **降级单缓冲**（保底正确性，不报错）。
