# SplitDataflow 子 Pass 逐行讲解（三）：FlagIdReuse

> 源码文件：`third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/FlagIdReuse.cpp`
> 头文件：`third_party/ascend/include/DynamicCVPipeline/SplitDataflow/FlagIdReuse.h`
> 关联文件：`third_party/ascend/lib/DynamicCVPipeline/Common/FlagIdManager.cpp`（flag 分配与上限判断）、`third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/InterCoreTransferAndSync.cpp`（调用方）
> 流水线位置：**不是独立的 Pass**，而是 `InterCoreTransferAndSync`（第 3 步）内部使用的辅助管理器类 `FlagIdReuseManager`

---

## 一、这个模块在做什么

`FlagIdReuseManager` 解决的是**同步 flag 数量受限**的问题。

昇腾硬件上可用于核间同步的 flag 数量是有限的（`FlagIdManager` 里定义了 `MAX_FLAG_ID = 14`；多缓冲场景下更紧张，`MULTI_MAX_FLAG_ID = 7`）。当依赖很多时，如果每条依赖都分配一个全新 flag，会超过硬件上限。因此需要**复用 flag id**：只要两个同步操作在时间上不会重叠（一个完全"释放"后才轮到另一个"获取"），它们就可以共用一个 flag id。

这个复用问题本质上是**图着色（graph coloring）**问题：
- 每个"传输/同步"是一个节点；
- 两个同步若"可能同时活跃"（发生冲突/干扰），则它们之间连一条边；
- 给图着色，颜色编号 = 复用的 flag id，相邻节点不能同色。

**本次更新要点**：
- **复用改为条件触发（超限才触发）**：只有当分配出去的 flag id 超过硬件上限（`flagManager.checkCurrentId()` 返回 false）时，才会执行复用分析；未超限时完全不介入，避免不必要的重编号；
- **`FlagIdManager` 独立成 Common 类**：flag id 的分配（`acquireId`）与上限判断（`checkCurrentId`）被抽到 `Common/FlagIdManager.h/.cpp`，并新增多缓冲上限 `MULTI_MAX_FLAG_ID = 7`；
- **新增 `ssbuffer.analyze_flag_id` 标记属性**（`kAnalyzeFlagId`）：所有因核间传输而创建的 sync op 都会被打上该 UnitAttr，只有带标记的 op 才参与复用分析与重映射，分析完成后标记被清除；
- **关系图的构建拆成两处**：E2（set→wait 配对）和 E4（wait→被保护数据）在**插入同步时**注册进 `relations`；E1（同 pipe FIFO）和 E3（数据依赖）在**分析时**由 `insertAnalyzeFlagRelations` 统一扫描补全；
- **新增重映射落盘函数 `remapInterCoreTransferFlagIds`**：把着色结果写回各 sync op 的 `static_flag_id` 属性；
- 新增单测 `flag_reuse_over_limit.mlir`（超限才复用）、`flag_reuse_cross_direction.mlir`（V→C→V 长链场景）。

---

## 二、核心概念

| 概念 | 说明 |
|---|---|
| `SyncBlockSetOp` | 同步的"生产端"操作：设置 flag（类似信号量的 signal） |
| `SyncBlockWaitOp` | 同步的"消费端"操作：等待 flag（类似信号量的 wait） |
| `static_flag_id` | `SyncBlockSet/Wait` 上的静态 flag id 属性 |
| `FlagIdManager` | Common 下的 flag id 分配器：`acquireId()` 递增分配；`checkCurrentId()` 判断是否超上限 |
| `MAX_FLAG_ID` / `MULTI_MAX_FLAG_ID` | 单缓冲上限 14 / 多缓冲（inter-core buffer count > 1）上限 7 |
| `ssbuffer.analyze_flag_id` | UnitAttr 标记：本 sync op 是核间传输引入的、可参与 flag 复用 |
| 干扰图（interference graph） | 节点=flag，边=两个 flag 生命周期重叠 |
| Welsh-Powell 算法 | 贪心图着色：按度数降序着色，每个节点取最小可用颜色 |

硬件 pipe 背景（来自 `getFlagDirection` 注释）：Cube→Vector 传输（fixpipe）走 FIX/V pipe，Vector→Cube 传输（copy）走 MTE*/M pipe，两类 pipe 互不相交。

---

## 三、复用的完整触发链路（新增，重点）

复用逻辑分布在三个时间点，理解这条链路才能讲清楚"flag 复用是怎么跑起来的"。

### 3.1 触发时机：超限才触发（InterCoreTransferAndSync.cpp L1967-L1973）

```cpp
if (!flagManager.checkCurrentId()) {          // 超过硬件上限才进入
  llvm::SmallVector<mlir::Operation *> analyzeFlagIdOps =
      insertAnalyzeFlagRelations(module, flagIdReuseManager);   // 补全关系图
  DenseMap<int, int> remapResult =
      flagIdReuseManager.reuseInterCoreTransferFlagIds(analyzeFlagIdOps);  // 着色
  remapInterCoreTransferFlagIds(remapResult);  // 写回 static_flag_id
}
```

- 这段代码位于 `processDependencies()` 的**尾部**：所有 V2C / C2V / C2C / 内存依赖的传输与同步都已插入完毕，此时 `flagManager.currentMaxId` 已是最终值；
- `checkCurrentId()` 为 false（即超限）才会做复用；未超限时 flag id 保持原样，**零开销跳过**整个分析；
- 复用发生在 `removeVectorPseudoOps()` 和 `processCubeToVectorDirectStoreSync()` **之前**——direct-store 同步是之后才插入的，它新拿的 flag 不参与本轮复用。

### 3.2 上限判断：FlagIdManager::checkCurrentId（Common/FlagIdManager.cpp L64-L72）

```cpp
bool FlagIdManager::checkCurrentId() {
  BufferCountManager::DepType depType = BufferCountManager::DepType::InterCore;
  BufferCountManager bufferCountMgr(module);
  int outerBufferCount = bufferCountMgr.getBufferCountByType(depType);
  if (outerBufferCount > 1) {
    return currentMaxId <= MULTI_MAX_FLAG_ID;   // 多缓冲：上限 7
  }
  return currentMaxId <= MAX_FLAG_ID;           // 单缓冲：上限 14
}
```

- **为什么多缓冲上限更低？** 多缓冲（multi-buffer，inter-core 传输目标有 >1 份缓存）意味着同一份逻辑 flag 在流水线中会有多个"实例"同时活跃（每个缓冲副本一套 set/wait），硬件 flag 资源被成倍占用，所以阈值收紧到 7。
- `FlagIdManager` 构造时通过 `scanExistingFlags` 扫描 module 里已有 sync op 的 `static_flag_id`/`flag` 属性，把 `currentMaxId` 初始化为历史最大值；`acquireId()` 则是 `++currentMaxId` 简单递增。
- 这就是"**每条依赖先无脑分配新 flag，最后统一看超不超限、超了才做复用**"的总体策略：分配期简单可靠，复用期集中处理。

### 3.3 打标阶段：attachAnalyzeFlagIdTag（InterCoreTransferAndSync.cpp L130-L133）

```cpp
static void attachAnalyzeFlagIdTag(Operation *op) {
  MLIRContext *ctx = op->getContext();
  op->setAttr(CVPipeline::kAnalyzeFlagId, UnitAttr::get(ctx));
}
```

- 每次为核间传输创建 `SyncBlockSetOp`/`SyncBlockWaitOp` 时都会调用它（如 `insertInterCoreSync` 中 6 个 sync op 全部打标：`setOpForRead/waitOpForRead/waitOpForWrite/setOpForWrite/setOpForStart/waitOpForEnd`；`insertMemDepSync`、C2V direct store 等同理）；
- **作用**：把"本 Pass 自己插入的 sync op"和"IR 里原本就存在的 sync op"区分开——只有前者才允许被重映射。这样复用分析不会误伤外部（如手写 IR、其他 pass 留下）的 flag；
- 该标记是临时属性，`remapInterCoreTransferFlagIds` 会统一移除（`RemoveAttributes.cpp` 里也注册了兜底清理）。

### 3.4 建图阶段：insertAnalyzeFlagRelations（InterCoreTransferAndSync.cpp L1670-L1782）

这个函数只在**超限后**被调用，负责把 happens-before 关系图补全，并收集参与分析的 op：

```cpp
llvm::SmallVector<mlir::Operation *>
InterCoreTransferAndSyncPass::insertAnalyzeFlagRelations(
    mlir::ModuleOp module, FlagIdReuseManager &flagIdReuseManager) {
  // E1 (per-pipe FIFO) is isolated per MLIR block: ops on one (core, pipe)
  llvm::DenseMap<Block *,
                 llvm::SmallDenseMap<hivm::TCoreType,
                                     llvm::SmallDenseMap<hivm::PIPE, OpVector>>>
      sequenceOpMap;                    // block -> core -> pipe -> op 序列
  llvm::DenseSet<mlir::Operation *> relationOpSet;
  llvm::SmallVector<mlir::Operation *> relationOps;
  llvm::SmallVector<mlir::Operation *> analyzeFlagIdOps;  // 带 kAnalyzeFlagId 的 op
  ...
```

**Step 1：单遍 module.walk 收集**（L1709-L1754）
- 遇到 `SyncBlockSetOp`/`SyncBlockWaitOp`：读取 `tcore_type` + pipe 属性，按 (block, core, pipe) 归类进 `sequenceOpMap`（E1 素材）；若带 `kAnalyzeFlagId` 标记则收进 `analyzeFlagIdOps`；
- 其他 op：若能通过 `getAnalyzeCoreType` 判断出核类型，则视为"关系图节点"（`noteRelationOp`）；
- 对实现了 `hivm::OpPipeInterface` 的 op（copy、macro op 等），按其 in/out pipe 同样归入 `sequenceOpMap`——这些真实计算 op 也会占用 pipe，必须纳入 FIFO 序列。

**Step 2：E3 数据依赖边**（L1756-L1768）

```cpp
for (Operation *op : relationOps) {
  for (Value operand : op->getOperands()) {
    Operation *definingOp = operand.getDefiningOp();
    if (!definingOp) {
      if (auto blockArgument = dyn_cast<BlockArgument>(operand)) {
        definingOp = blockArgument.getOwner()->getParentOp();  // BlockArgument 归到父 op
      }
    }
    if (definingOp && relationOpSet.contains(definingOp)) {
      insertRelation(definingOp, op);      // def -> use 的 happens-before 边
    }
  }
}
```

- BlockArgument 没有 defining op，这里把它的"定义者"归到**持有该参数的父 op**（如循环），使跨循环携带值的依赖也能连上。

**Step 3：E1 同 pipe FIFO 边**（L1770-L1780）

```cpp
for (auto &blockEntry : sequenceOpMap) {
  for (auto &coreEntry : blockEntry.second) {
    for (auto &pipeEntry : coreEntry.second) {
      auto &ops = pipeEntry.second;
      for (size_t i = 0; i + 1 < ops.size(); ++i) {
        insertRelation(ops[i], ops[i + 1]);   // 相邻两个 op 连边
      }
    }
  }
}
```

- 同一 MLIR block 内、同一 (core, pipe) 上的 op 天然按序执行（硬件 pipe 是 FIFO），相邻连边即可。

**E2/E4 边在插入同步时就已经注册**（`insertInterCoreSync` L1003-L1015）：

```cpp
// E2: register every set->wait pair of this transfer, not just the
// loop start/end pair. Each pair is the only proof of cross-core
// ordering for the sync ops it connects.
flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForRead, waitOpForRead);
flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForWrite, waitOpForWrite);
flagIdReuseManager.insertRelationBetweenSetAndWait(setOpForStart, waitOpForEnd);
// E4: link the read-wait to the consumed data it guards so the sync
// op is threaded into the downstream dataflow graph.
flagIdReuseManager.insertRelationBetweenSetAndWait(waitOpForRead, consumedDataOp);
```

- **E2**：一次多缓冲传输会产生 3 组 set→wait 配对（读、写、循环边界），全部注册——注释强调"每一对都是它所连接的 sync op 跨核顺序的唯一证明"，漏掉任何一对都会让 `opPrecedes` 丢失可达路径；
- **E4**：读端 wait 连到它保护的数据消费点，把同步 op "缝进"数据流图，这样下游数据依赖的传递路径能经过同步点。

### 3.5 落盘阶段：remapInterCoreTransferFlagIds（InterCoreTransferAndSync.cpp L1784-L1806）

```cpp
void InterCoreTransferAndSyncPass::remapInterCoreTransferFlagIds(
    llvm::DenseMap<int, int> &remapResult) {
  module.walk([&](mlir::Operation *op) {
    if (!llvm::isa<hivm::SyncBlockSetOp>(op) &&
        !llvm::isa<hivm::SyncBlockWaitOp>(op)) {
      return;
    }
    bool trackedForReuse = op->hasAttr(CVPipeline::kAnalyzeFlagId);
    op->removeAttr(CVPipeline::kAnalyzeFlagId);        // 标记使命完成，统一清除
    if (!trackedForReuse || remapResult.empty()) {
      return;                                          // 未打标 / 无重映射 → 不动
    }
    if (auto intAttr = op->getAttrOfType<mlir::IntegerAttr>("static_flag_id")) {
      int flagId = static_cast<int>(intAttr.getInt());
      auto it = remapResult.find(flagId);
      if (it == remapResult.end()) {
        return;                                        // 该 id 不在重映射表里 → 不动
      }
      auto newFlagAttr = mlir::IntegerAttr::get(intAttr.getType(), it->second);
      op->setAttr("static_flag_id", newFlagAttr);      // 旧 id → 新 id
    }
  });
}
```

- 三重过滤保证只改自己插入的 op：是 sync op、带 `kAnalyzeFlagId`、id 出现在 remap 表中；
- 所有 sync op 的标记属性（无论是否被重映射）都会被移除，保持 IR 干净。

---

## 四、FlagIdReuseManager 逐行讲解

### 4.1 数据结构与插入关系（L48-L58）

```cpp
void FlagIdReuseManager::insertRelationBetweenSetAndWait(Operation *before,
                                                         Operation *after) {
  if (!before || !after) { ... return; }   // 空指针保护
  relations[before].push_back(after);       // 建立 before -> after 的有向边
  return;
}
```

- `relations` 是一个 `DenseMap<Operation*, SmallVector<Operation*>>`，构成一张**有向图**，边表示"happens-before"（顺序）关系。
- 四种顺序关系（E1-E4）的来源：
  - **E1**：同一 (core, pipe) 上的 FIFO 顺序（同一管道天然有序）——分析期由 `insertAnalyzeFlagRelations` 补全；
  - **E2**：set → wait 配对（信号量语义）——插入同步时注册；
  - **E3**：数据依赖（data flow）——分析期补全；
  - **E4**：read-wait → 被保护的数据（同步 op 与被它保护的数据消费点之间）——插入同步时注册。

### 4.2 主入口 reuseInterCoreTransferFlagIds（L60-L69）

```cpp
DenseMap<int, int> FlagIdReuseManager::reuseInterCoreTransferFlagIds(
    const llvm::SmallVector<Operation *> &syncOps) {
  DenseMap<int, int> remapResult;    // 返回值：旧 flagId -> 新 flagId
  if (syncOps.empty()) return remapResult;

  preworkForAnalyze(syncOps);         // 预处理：建立 flagIdToOps 和 opOrder
  return colorInterferenceGraph();    // 着色并生成重映射
}
```

- 入参 `syncOps` 即 `insertAnalyzeFlagRelations` 收集的 `analyzeFlagIdOps`（只有打标的核间 sync op）。

### 4.3 预处理 preworkForAnalyze（L71-L86）

```cpp
void FlagIdReuseManager::preworkForAnalyze(
    const llvm::SmallVector<Operation *> &syncOps) {
  flagIdToOps.clear();
  opOrder.clear();
  // syncOps arrive in module-walk (program) order, so the index is a stable
  // program-order rank used to pick the earliest set / latest wait of a flag.
  int order = 0;
  for (auto op : syncOps) {
    opOrder[op] = order++;            // 记录每个 sync op 的程序顺序序号
    auto flagId = getFlagId(op);      // 读取 static_flag_id
    if (flagId == -1) continue;       // 动态 flagId 跳过
    flagIdToOps[flagId].push_back(op); // flagId -> 使用该 flag 的所有 op
  }
}
```

- `opOrder` 用程序顺序作为稳定的排序依据，用于判断两个 op 的先后关系。
- 注意：`syncOps` 传进来时本身是 module-walk 顺序，索引天然就是程序序。

### 4.4 找最早 set 和最晚 wait（L88-L112）

```cpp
Operation *FlagIdReuseManager::getEarliestSet(int flagId) {
  Operation *earliest = nullptr;
  for (auto op : flagIdToOps[flagId]) {
    if (!llvm::isa<SyncBlockSetOp>(op)) continue;   // 只看 set
    if (!earliest || opOrder[op] < opOrder[earliest]) earliest = op;
  }
  return earliest;
}

Operation *FlagIdReuseManager::getLatestWait(int flagId) {
  Operation *latest = nullptr;
  for (auto op : flagIdToOps[flagId]) {
    if (!llvm::isa<SyncBlockWaitOp>(op)) continue;  // 只看 wait
    if (!latest || opOrder[op] > opOrder[latest]) latest = op;
  }
  return latest;
}
```

- "释放"（release）= 最晚的 wait 完成；"获取"（acquire）= 最早的 set 开始。
- 这两个函数给出的是**生命周期边界**（头文件注释：lifecycle boundary of a flag group）。

### 4.5 顺序判断 opPrecedes（L114-L130）

```cpp
bool FlagIdReuseManager::opPrecedes(Operation *p, Operation *q) {
  if (!p || !q) return false;
  if (p == q) return true;
  // Same block: MLIR static program order is an exact, sound ordering and
  // covers straight-line chains and sibling loops without needing a graph edge.
  if (p->getBlock() == q->getBlock()) return p->isBeforeInBlock(q);
  // Cross block: only the real happens-before edges (E1 per-pipe FIFO, E2
  // set->wait, E3 data, E4 read-wait->consumed data) may prove ordering.
  llvm::SmallSet<Operation *, CVPipeline::INIT_SIZE> visited;
  return hasPath(visited, p, q);   // DFS 找 p -> q 的路径
}
```

- 同一 block 内用 `isBeforeInBlock`（静态顺序精确可靠，覆盖直线代码和兄弟循环，不需要图边）；跨 block 则只能靠图上的 DFS 判断可达性。

### 4.6 flagReleasedBefore：判断 flag 释放顺序（L132-L141）

```cpp
bool FlagIdReuseManager::flagReleasedBefore(int before, int after) {
  Operation *release = getLatestWait(before);  // before 的释放点
  Operation *acquire = getEarliestSet(after);  // after 的获取点
  if (!release || !acquire) return false;
  return opPrecedes(release, acquire);          // 释放早于获取 → 可复用
}
```

### 4.7 getFlagDirection：判断传输方向（L143-L164）

```cpp
FlagIdReuseManager::SyncDir FlagIdReuseManager::getFlagDirection(int flagId) {
  for (auto *op : flagIdToOps[flagId]) {
    PipeAttr srcPipe, dstPipe;
    if (auto setOp = llvm::dyn_cast<SyncBlockSetOp>(op)) {
      srcPipe = setOp.getTpipeAttr(); dstPipe = setOp.getPipeAttr();
    } else if (auto waitOp = llvm::dyn_cast<SyncBlockWaitOp>(op)) { ... }
    for (PipeAttr pipe : {srcPipe, dstPipe}) {
      if (pipe && (pipe.getPipe() == hivm::PIPE::PIPE_FIX ||
                   pipe.getPipe() == hivm::PIPE::PIPE_V)) {
        return SyncDir::CubeToVector;   // FIX/V pipe → Cube→Vector
      }
    }
  }
  return SyncDir::VectorToCube;
}
```

- A transfer's data-movement direction, read off its sync pipes：FIX/V pipe → C2V（fixpipe），否则 → V2C（copy 走 MTE*/M）。两类 pipe 互不相交，所以看任意一个 pipe 即可定方向。

### 4.8 flagsInterfere：判断两个 flag 是否冲突（L166-L176）

```cpp
bool FlagIdReuseManager::flagsInterfere(int lhs, int rhs) {
  // Opposite-direction transfers (Vector->Cube vs Cube->Vector) execute
  // concurrently on the two cores, so their flag lifetimes always overlap
  if (getFlagDirection(lhs) != getFlagDirection(rhs)) return true;
  // 同方向：除非一个可证明"先释放"、另一个"后获取"，否则都视为冲突（保守）
  return !flagReleasedBefore(lhs, rhs) && !flagReleasedBefore(rhs, lhs);
}
```

- 这是**保守策略**：只要无法证明"不重叠"，就当作冲突，宁可不复用也不冒险，保证正确性。

### 4.9 colorInterferenceGraph：图着色（L178-L247）

```cpp
DenseMap<int, int> FlagIdReuseManager::colorInterferenceGraph() {
  // 1. 收集所有 flagId，按首次出现顺序排序（确定性）
  llvm::SmallVector<int> flagIds;
  for (auto &[flagId, ops] : flagIdToOps) if (!ops.empty()) flagIds.push_back(flagId);
  auto firstOrder = [&](int flagId) { ... 最小 opOrder ... };
  llvm::sort(flagIds, [&](int a, int b) { return firstOrder(a) < firstOrder(b); });

  // 2. 建无向干扰图
  DenseMap<int, llvm::SmallVector<int>> adj;
  for (size_t i = 0; i < flagIds.size(); ++i)
    for (size_t j = i + 1; j < flagIds.size(); ++j)
      if (flagsInterfere(flagIds[i], flagIds[j])) {
        adj[flagIds[i]].push_back(flagIds[j]);
        adj[flagIds[j]].push_back(flagIds[i]);
      }

  // 3. Welsh-Powell：度数高的先着色（同度数时先出现的 flag 优先）
  llvm::sort(order, [&](int a, int b) {
    if (adj[a].size() != adj[b].size()) return adj[a].size() > adj[b].size();
    return firstOrder(a) < firstOrder(b);
  });

  // 4. 贪心着色：每个节点取邻居未使用的最小颜色
  DenseMap<int, int> rawColor;
  for (int flagId : order) {
    llvm::SmallSet<int, CVPipeline::INIT_SIZE> usedByNeighbours;
    for (int nb : adj[flagId])
      if (rawColor.find(nb) != rawColor.end()) usedByNeighbours.insert(rawColor[nb]);
    int color = 1;
    while (usedByNeighbours.contains(color)) ++color;
    rawColor[flagId] = color;
  }

  // 5. 压缩重编号：按程序顺序把颜色映射为 1,2,3,...
  DenseMap<int, int> colorToCompact;
  int nextCompact = 1;
  DenseMap<int, int> remapResult;
  for (int flagId : flagIds) {
    int color = rawColor[flagId];
    if (!colorToCompact.contains(color)) colorToCompact[color] = nextCompact++;
    remapResult[flagId] = colorToCompact[color];
  }
  return remapResult;
}
```

- Welsh-Powell 贪心着色得到的是近似最优解（最小着色是 NP-hard），但足够用。
- 最后一步"压缩重编号"让输出的 flag id 从 1 开始连续递增，节省 flag 资源且结果稳定（注释：output flag ids are minimal and stably ordered）。

### 4.10 hasPath：DFS 可达性（L249-L265）

```cpp
bool FlagIdReuseManager::hasPath(
    llvm::SmallSet<Operation *, CVPipeline::INIT_SIZE> &visited,
    Operation *from, Operation *to) {
  if (from == to) return true;
  if (visited.contains(from)) return false;
  visited.insert(from);
  for (auto nxt : relations[from])
    if (hasPath(visited, nxt, to)) return true;
  return false;
}
```

- 标准 DFS 判 reachability，`visited` 集合避免环导致的无限递归。

### 4.11 getFlagId：读取 static_flag_id（L267-L288）

```cpp
int FlagIdReuseManager::getFlagId(Operation *op) {
  if (auto setOp = llvm::dyn_cast<SyncBlockSetOp>(op)) {
    if (auto staticFlagId = setOp.getStaticFlagId()) return staticFlagId->getInt();
    ... return -1;  // 动态 flagId 跳过
  }
  if (auto waitOp = llvm::dyn_cast<SyncBlockWaitOp>(op)) { ... }
  return -1;
}
```

- 只有**静态** flag id（编译期已知）才能参与复用分析；动态 flag id 直接跳过（返回 -1）。

---

## 五、算法流程总结

```
【插入期】每条核间依赖分配新 flag（FlagIdManager::acquireId 递增）
    └─ 同步 op 创建时：打 kAnalyzeFlagId 标记 + 注册 E2（set→wait 对）/ E4（wait→数据）边

【判断期】所有依赖处理完后：
    flagManager.checkCurrentId()？
    ├─ 未超限（单缓冲 ≤14 / 多缓冲 ≤7）→ 直接结束，flag id 保持原样
    └─ 超限 → 进入复用：
        1. insertAnalyzeFlagRelations：补全 E1（同 pipe FIFO）/ E3（数据依赖）边，
           收集带 kAnalyzeFlagId 标记的 sync op 列表
        2. reuseInterCoreTransferFlagIds：
           a. 预处理：建立 flagId -> ops 映射 和 op -> 程序顺序序号
           b. 建干扰图：
              - 方向不同的 flag → 一定冲突（连边）
              - 方向相同的 flag → 若无法证明"先释放后获取" → 冲突（连边）
           c. Welsh-Powell 贪心着色
           d. 压缩重编号 → 输出 remapResult（旧 id -> 新 id）
        3. remapInterCoreTransferFlagIds：按表写回 static_flag_id，清除标记
```

---

## 六、面试要点

1. **为什么要复用 flag id？**
   硬件 flag 数量有限（单缓冲 MAX=14，多缓冲 MAX=7），依赖数量可能远超这个数。必须复用：生命周期不重叠的同步可以共用一个 flag。

2. **为什么"超限才触发"复用，而不是每次都做？**
   复用（图着色 + 全模块扫描重映射）有编译期开销，而且会把原本互不相同的 flag id 重排，降低 IR 可读性和调试便利性。只在真正超过硬件上限时才启动，是典型的"**快路径/慢路径**"设计：绝大多数 kernel 依赖数不超过上限，走零开销快路径。

3. **为什么用图着色而不是简单线性分配？**
   复用 flag 的核心约束是"活跃区间不重叠"，这正是**寄存器分配/区间图着色**的经典模型。图着色能精确表达任意复杂的重叠关系。

4. **为什么方向相反的传输一定冲突（连边）？**
   因为 Cube→Vector 和 Vector→Cube 分别由两个核并发执行，它们的 flag 生命周期在时间上必然重叠，无法证明串行，所以保守地视为冲突。

5. **保守策略的意义？**
   "无法证明不重叠就当作冲突"保证了正确性优先——宁可少复用（多用几个 flag）也不会让两个并发传输错误地共享 flag 导致数据竞争。

6. **E1-E4 四种 happens-before 边分别是什么？为什么需要全部四种？**
   E1 同 pipe FIFO、E2 set→wait 信号量配对、E3 数据依赖、E4 wait→被保护数据。`opPrecedes` 判断"释放早于获取"依赖图可达性，缺任何一种边都可能丢失真实的顺序证明，导致误判冲突（正确性不受影响，但复用率下降）。尤其是 E2 注释强调"每一对 set→wait 都要注册，不只是循环首尾那对"。

7. **`ssbuffer.analyze_flag_id` 标记的作用？**
   划定复用分析的**作用域**：只有本 Pass 插入的 sync op 才参与重映射，避免误改 IR 中已有的外部同步；同时它是临时属性，重映射完成后统一清除。

8. **多缓冲场景为什么上限是 7 而不是 14？**
   多缓冲意味着同一逻辑传输有多份缓冲副本，流水线推进时多套 set/wait 同时活跃，硬件 flag 占用成倍增加，因此阈值收紧（`MULTI_MAX_FLAG_ID`），`checkCurrentId` 通过 `BufferCountManager` 读取 inter-core buffer 数量来选择阈值。
