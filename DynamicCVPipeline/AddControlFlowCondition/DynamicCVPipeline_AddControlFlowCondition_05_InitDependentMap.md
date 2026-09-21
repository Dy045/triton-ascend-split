# DynamicCVPipeline 主 Pass 专题：控制流条件注入 · 子专题 05 —— InitDependentMap（依赖图构建）

> 覆盖 `AddControlFlowCondition` 的 **Step 3：InitDependentMapPass**
>
> 源码文件：[`InitDependentMap.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/AddControlFlowCondition/InitDependentMap.cpp)
>
> 流水线位置：`AddControlFlowCondition`（[AddDynamicCVPipeline.cpp](../../../third_party/ascend/lib/DynamicCVPipeline/AddDynamicCVPipeline.cpp) L83）内第四个**独立**子 Pass（主编排 L101-L106，`setConditionInfo` 传入共享 `info`）。

---

## 一、这个子 Pass 在做什么

`CreateIfOps` 给每个 block_id 搭好了 `scf.if %true` 骨架，但**哪些 if 之间有数据依赖、依赖是跨核还是核内、需要几份缓冲**这些信息还是"散布在 op 属性里"的状态。`UpdateConditionInfo` 构造真实同步条件之前，必须先把它固化成可查询的数据结构。

`InitDependentMap` 的职责是**把 `ssbuffer.crossCoreDeps` / `ssbuffer.intraDeps` 属性解析成依赖映射，并构建块级依赖 DAG**：

1. **跨核依赖映射**：consumer op -> 每个依赖 group 的 producers 列表（`info->crossCoreDependentMap`），只保留同一主循环内的对；
2. **核内依赖映射**：每个主循环 -> 循环内 consumer -> producers（`info->intraCoreDependentMap`），consumer 限定在该主循环内（不含嵌套主循环）；
3. **缓冲数**：统计 producer 列表最大长度作为 `crossCoreBufferCount` / `intraCoreBufferCount`（核内为空时回退到 `BufferCountManager`）；
4. **if 块 DAG**：把 op 级依赖映射到 if 块级（`info->ifBlockDAG`，带 CrossCore/IntraCore 边类型）；
5. **环检测**：DFS 三色标记检测 DAG 是否有环，有环则该 kernel 不可流水线化；
6. **flowOpt 对**：缓冲数超阈值时，DAG 深度遍历找"距离起点深度恰为 3"的节点，记录 `flowOptIfOpPairs`（开启重叠优化）。

**一句话**：从属性到结构化依赖图，是后续一切缓冲/条件/循环扩展计算的"事实来源"。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `kCrossCoreDeps`（`ssbuffer.crossCoreDeps`） | 跨核依赖属性，`AllocMultiCache` 打上。扁平数组：`[group, role, group, role, ...]`，role: 1=producer、0=consumer |
| `kIntraDeps`（`ssbuffer.intraDeps`） | 核内依赖属性，同上格式 |
| `depsByGroup` | `group -> [(op, role)]`：按 group 分组收集带依赖属性的 op 及其角色（一个 op 可出现在多个 group / 双重角色） |
| `ConsumerProducerMap` | `consumer op -> SmallVector<SmallVector<Operation*>>`：每个内层 vector 是一个 group 的 producers（一个 consumer 可消费多个 group） |
| `isConsumerInMainLoop` | 沿 parent 链上溯：命中**其他**主循环 → 该 consumer 属于嵌套主循环，跳过；命中目标主循环 → 收入 consumers |
| `filterMemCrossCoreDepsByMainLoop` | 跨核过滤：consumer 与整组 producers 必须在**同一个**主循环内，否则丢弃该 group（producers 之间主循环不一致则报错） |
| `collectMainLoopById` / `findMainLoopIdContainingOp` | 收集所有 `kMainLoop` 标记的 for/while 及其 id；反查某个 op 所在的 main_loop id |
| `computeProducerBufferCount` | 取依赖映射中 producer 列表最大长度作为缓冲数；核内映射为空时回退 `BufferCountManager` 的 IntraCore 值 |
| `ifBlockDAG` | `producerIf -> [(consumerIf, IfBlockDepKind)]`：块级有向图，`IfBlockDepKind` ∈ {CrossCore, IntraCore} |
| `findIfOpContainingOp` | 沿 parent 链找最近带 `kIf` 的 `scf.if`（深度上限 100），把 op 级依赖映射到 if 级 |
| `DfsState` 三色标记 | Unvisited / Visiting / Done，DFS 检测环：遇到 Visiting 即回边 |
| `flowOptIfOpPairs` | `targetIf -> startIf`：从入度 0 起点 DFS，深度恰好 = 3 的节点成为 pair，用于流水线重叠优化 |
| `CROSS_CORE_BUFFER_COUNT_THRESHOLD` / `INTRA_CORE_BUFFER_COUNT_THRESHOLD` | flowOpt 启用阈值，缓冲数须同时超过才收集 flowOpt 对 |
| `ERRCODE_IGNORED` | 环检测失败时设置的 fallback 码：视为"kernel 不适合"，可忽略回退（与 `ERRCODE_FAILED` 的"管线内部错误"语义不同） |

---

## 三、逐行讲解

### 3.1 入口：runOnOperation（InitDependentMap.cpp L704-L763）

```cpp
void InitDependentMapPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  // Step 1: 初始化跨核依赖映射
  if (initCrossCoreDependentMap(module, info) != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
  // Step 2: 初始化核内依赖映射
  if (initIntraCoreDependentMap(module, info) != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LLVM_DEBUG(printDependentMaps(info));

  // Step 4: 计算 producer 缓冲数（供 flowOpt 阈值判断）
  computeProducerBufferCount(info, module);

  // Step 4: 构建 if 块 DAG
  if (buildIfBlockDAG(module, info) != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
  // Step 5: 检测 DAG 环
  if (detectCycle(info) != 0) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }
  // Step 6: 缓冲数超阈值才收集 flowOpt 对
  if (info->crossCoreBufferCount > CROSS_CORE_BUFFER_COUNT_THRESHOLD &&
      info->intraCoreBufferCount > INTRA_CORE_BUFFER_COUNT_THRESHOLD) {
    if (collectFlowOptIfOpPairs(module, info) != 0) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
    LLVM_DEBUG(printDAGInfo(info));
  }
}
```

**要点**：6 步严格有序，任一步失败即 fallback 终止。注意环检测用 `ERRCODE_IGNORED`（依赖有环 = kernel 结构不适合流水线化，属预期回退），其余用 `ERRCODE_FAILED`（内部错误）。

### 3.2 属性解析：collectDepsByGroup（L79-L116）

```cpp
static int
collectDepsByGroup(Operation *rootOp, const char *attrName,
                   llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>>
                       &depsByGroup) {
  int ret = 0;
  int depSize = 2;      // 一条记录 = [group, role]
  int groupOffset = 0;
  int roleOffset = 1;

  rootOp->walk([&](Operation *op) {
    auto depsAttr = op->getAttrOfType<ArrayAttr>(attrName);
    if (!depsAttr)
      return;

    // 属性长度必须是偶数（成对），否则格式错误
    if (depsAttr.empty() || depsAttr.size() % depSize != 0) {
      ret = -1;
      return;
    }

    for (size_t i = 0; i < depsAttr.size(); i += depSize) {
      Attribute groupAttr = depsAttr[i + groupOffset];
      Attribute roleAttr = depsAttr[i + roleOffset];
      if (!isa<IntegerAttr>(groupAttr) || !isa<IntegerAttr>(roleAttr)) {
        ret = -1;
        return;
      }
      int group = cast<IntegerAttr>(groupAttr).getInt();
      int role = cast<IntegerAttr>(roleAttr).getInt();
      depsByGroup[group].push_back({op, role});
    }
  });
  return ret;
}
```

**要点**：walk 整个模块，凡带目标属性（cross/intra）的 op 都按 `(group, role)` 记录。一个 op 有多个 group 记录 → 在多个 group 里各占一份；`walk` 保证确定性顺序。

### 3.3 跨核映射：buildCrossCoreProducerConsumerMapping（L120-L151）

```cpp
static int buildCrossCoreProducerConsumerMapping(
    llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>> &depsByGroup,
    ConsumerProducerMap &result) {
  for (auto &groupEntry : depsByGroup) {
    auto &ops = groupEntry.second;

    SmallVector<Operation *> producers;
    SmallVector<Operation *> consumers;
    for (auto &opRole : ops) {
      Operation *op = opRole.first;
      int role = opRole.second;
      if (role == CVPipeline::crossCoreProducerId) {
        if (!llvm::is_contained(producers, op))
          producers.push_back(op);
      } else if (role == CVPipeline::crossCoreConsumerId) {
        if (!llvm::is_contained(consumers, op))
          consumers.push_back(op);
      } else {
        return -1;  // 未知 role
      }
    }

    // 每个 consumer 都拿到该 group 的完整 producers 列表
    for (Operation *consumer : consumers) {
      result[consumer].push_back(producers);
    }
  }
  return 0;
}
```

**要点**：按 group 把 ops 分成 producers/consumers（去重），然后**每个 consumer 追加一份该 group 的 producers 列表**。若一个 consumer 同时出现在多个 group，`result[consumer]` 就有多个内层列表——这正是 `ConsumerProducerMap`（外层 consumer、内层每 group 一份 producers）的结构由来。

### 3.4 跨核主循环过滤：filterMemCrossCoreDepsByMainLoop（L234-L304）

```cpp
static int
filterMemCrossCoreDepsByMainLoop(ModuleOp module,
                                 ConsumerProducerMap &initialDepsMap,
                                 ConsumerProducerMap &filteredDepsMap) {
  // Step 1: 收集所有 main_loop op 及其 id
  llvm::DenseMap<Operation *, int> mainLoopById;
  if (collectMainLoopById(module, mainLoopById) != 0)
    return -1;

  // Step 2: 逐 consumer、逐 group 过滤
  for (auto &entry : initialDepsMap) {
    Operation *consumer = entry.first;
    int consumerMainLoopId = findMainLoopIdContainingOp(consumer, mainLoopById);
    if (consumerMainLoopId == -1) {
      continue;  // consumer 不在任何主循环 → 跳过
    }

    for (SmallVector<Operation *> &producers : entry.second) {
      int producerMainLoopId =
          findMainLoopIdContainingOp(producers[0], mainLoopById);
      if (producerMainLoopId == -1) {
        continue;
      }

      // 该 group 所有 producers 必须在同一主循环
      for (size_t i = 1; i < producers.size(); i++) {
        if (findMainLoopIdContainingOp(producers[i], mainLoopById) !=
            producerMainLoopId) {
          return -1;  // 同组 producers 跨主循环 → 错误
        }
      }

      // consumer 与 producers 必须在同一主循环，否则丢弃该 group
      if (consumerMainLoopId != producerMainLoopId) {
        continue;
      }
      filteredDepsMap[consumer].push_back(producers);
    }
  }
  return 0;
}
```

**要点**：
- **过滤粒度是"每个 producer group"**：多 group consumer 可保留部分 group、丢弃部分 group（注释特别强调）；
- 跨核依赖本质是"同一主循环的两个核之间"的同步，因此 consumer 与整组 producers 必须同属一个 main_loop；组内 producers 跨主循环视为错误（配置问题）。

### 3.5 核内映射：initIntraCoreDependentMap + buildIntraCoreProducerConsumerMapping（L341-L378, L155-L199）

```cpp
int initIntraCoreDependentMap(ModuleOp module, ControlFlowConditionInfo *info) {
  // 全模块收集 kIntraDeps 属性（一次 walk）
  llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>> allIntraDepsByGroup;
  if (collectDepsByGroup(module, CVPipeline::kIntraDeps.data(),
                         allIntraDepsByGroup) != 0)
    return -1;

  int ret = 0;
  module.walk([&](Operation *op) {
    if (!op->hasAttr(CVPipeline::kMainLoop))
      return;
    if (!isMainLoopOp(op)) {
      ret = -1;
      return;
    }

    // 对该主循环构建映射（consumer 限定在循环内）
    llvm::DenseMap<Operation *, SmallVector<Operation *>> depMap;
    if (buildIntraCoreProducerConsumerMapping(allIntraDepsByGroup, depMap, op) != 0) {
      ret = -1;
      return;
    }
    if (!depMap.empty()) {
      info->intraCoreDependentMap[op] = depMap;  // 以 main_loop op 为键
    }
  });
  return ret;
}
```

```cpp
static int buildIntraCoreProducerConsumerMapping(
    llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>> &depsByGroup,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &result,
    Operation *mainLoop = nullptr) {
  for (auto &groupEntry : depsByGroup) {
    // producers：全部保留
    // consumers：mainLoop != nullptr 时用 isConsumerInMainLoop 过滤
    for (auto &opRole : ops) {
      if (role == CVPipeline::crossCoreProducerId) {
        producers.push_back(op);
      } else if (role == CVPipeline::crossCoreConsumerId) {
        if (mainLoop != nullptr) {
          // 只保留"位于本主循环内且不在嵌套主循环"的 consumer
          if (isConsumerInMainLoop(op, mainLoop, consumers) != 0)
            return -1;
        } else {
          consumers.push_back(op);
        }
      }
    }
    // 过滤后无 consumer → 跳过该 group
    if (mainLoop != nullptr && consumers.empty())
      continue;

    for (Operation *consumer : consumers) {
      result[consumer] = producers;
    }
  }
  return 0;
}
```

**要点**：与跨核的关键差异——核内映射以 **main_loop op 为键**（`info->intraCoreDependentMap[op]`），且 consumer 用 `isConsumerInMainLoop` 排除嵌套主循环（沿 parent 上溯命中其他 main_loop 即 0 跳过）。`isConsumerInMainLoop`（L53-L73）两种"返回 0"：consumer 在目标循环内（push）或属于嵌套主循环（skip），需区分。

### 3.6 缓冲数统计：computeProducerBufferCount（L450-L482）

```cpp
static void computeProducerBufferCount(ControlFlowConditionInfo *info,
                                       ModuleOp module) {
  // 跨核：producers 列表最大长度
  info->crossCoreBufferCount = 0;
  for (auto &entry : info->crossCoreDependentMap) {
    for (SmallVector<Operation *> &producers : entry.second) {
      info->crossCoreBufferCount =
          std::max(info->crossCoreBufferCount, (int)producers.size());
    }
  }

  // 核内：所有主循环的 producers 列表最大长度
  info->intraCoreBufferCount = 0;
  for (auto &loopEntry : info->intraCoreDependentMap) {
    for (auto &entry : loopEntry.second) {
      info->intraCoreBufferCount =
          std::max(info->intraCoreBufferCount, (int)entry.second.size());
    }
  }

  // 核内映射为空 → 回退 BufferCountManager 的 IntraCore 值
  if (info->intraCoreBufferCount == 0) {
    BufferCountManager bufferCountMgr(module);
    info->intraCoreBufferCount = bufferCountMgr.getBufferCountByType(
        BufferCountManager::DepType::IntraCore);
  }
}
```

**要点**：缓冲数 = 一个依赖组里 producer 的最大个数（多少个 producer 共用一条链就需要多少份缓冲轮换）。核内映射为空（没有显式核内依赖属性）时回退到 `BufferCountManager` 配置值，保证流程继续。

### 3.7 if 块 DAG：buildIfBlockDAG（L498-L562）

```cpp
static int buildIfBlockDAG(ModuleOp module, ControlFlowConditionInfo *info) {
  // Step 1: 跨核依赖 → CrossCore 边
  for (auto &entry : info->crossCoreDependentMap) {
    Operation *consumerOp = entry.first;
    scf::IfOp consumerIf = findIfOpContainingOp(consumerOp);
    if (!consumerIf)
      return -1;

    for (SmallVector<Operation *> &producers : entry.second) {
      for (Operation *producerOp : producers) {
        scf::IfOp producerIf = findIfOpContainingOp(producerOp);
        if (!producerIf)
          return -1;

        // 同一 if 内 producer+consumer 不合法
        if (producerIf == consumerIf)
          return -1;

        addIfBlockEdge(info, producerIf, consumerIf, IfBlockDepKind::CrossCore);
      }
    }
  }

  // Step 2: 核内依赖 → IntraCore 边（结构同上）
  for (auto &loopEntry : info->intraCoreDependentMap) {
    for (auto &consumerProducers : loopEntry.second) {
      ...
      addIfBlockEdge(info, producerIf, consumerIf, IfBlockDepKind::IntraCore);
    }
  }
  return 0;
}
```

```cpp
static void addIfBlockEdge(ControlFlowConditionInfo *info, scf::IfOp producerIf,
                           scf::IfOp consumerIf, IfBlockDepKind kind) {
  auto &edges = info->ifBlockDAG[producerIf];
  // 去重：producer -> consumer 边只加一次
  bool seen = llvm::any_of(edges, [&](const auto &e) {
    return e.first == consumerIf;
  });
  if (!seen) {
    edges.push_back({consumerIf, kind});
  }
}
```

**要点**：
- 依赖从 op 级**上卷**到 if 级：`findIfOpContainingOp` 沿 parent 链找最近带 `kIf` 的 if（深度上限 100，超限告警）；
- **同一 if 内 producer+consumer 非法**——`AllocMultiCache`/`CreateIfOps` 设计上每个 block 一个 if，同块依赖应已被 `CloneOps` 消解，出现即说明上游有问题；
- `addIfBlockEdge` 去重：同一个 producer→consumer 对只保留一条边（首个 kind 生效，后续同 pair 不同 kind 会被丢弃——跨核/核内同时存在时以先到者为准）。

### 3.8 环检测：detectCycle + dfsCycle（L565-L609）

```cpp
enum class DfsState : uint8_t { Unvisited, Visiting, Done };

static bool
dfsCycle(scf::IfOp node, dag, llvm::DenseMap<scf::IfOp, DfsState> &state) {
  state[node] = DfsState::Visiting;
  auto it = dag.find(node);
  if (it != dag.end()) {
    for (auto &edge : it->second) {
      scf::IfOp neighbor = edge.first;
      auto s = state.lookup(neighbor);
      if (s == DfsState::Visiting)
        return true;                       // 回边 → 有环
      if (s == DfsState::Unvisited && dfsCycle(neighbor, dag, state))
        return true;
    }
  }
  state[node] = DfsState::Done;
  return false;
}

static int detectCycle(ControlFlowConditionInfo *info) {
  // 收集全部节点（key + value）
  llvm::DenseMap<scf::IfOp, DfsState> state;
  for (auto &entry : info->ifBlockDAG) {
    state.try_emplace(entry.first, DfsState::Unvisited);
    for (auto &edge : entry.second) {
      state.try_emplace(edge.first, DfsState::Unvisited);
    }
  }
  // 从每个未访问节点 DFS
  for (auto &entry : state) {
    if (entry.second == DfsState::Unvisited) {
      if (dfsCycle(entry.first, info->ifBlockDAG, state))
        return -1;
    }
  }
  return 0;
}
```

**要点**：经典 DFS 三色标记。Visiting 状态再次被访问 = 回边 = 环。DAG 有环意味着"producer 依赖消费者产出的缓冲、消费者又依赖 producer 的产出"形成死锁，无法注入同步条件，直接 fallback（`ERRCODE_IGNORED`）。

### 3.9 flowOpt 对收集：collectFlowOptIfOpPairs（L612-L680）

```cpp
static int collectFlowOptIfOpPairs(ModuleOp module,
                                   ControlFlowConditionInfo *info) {
  // Step 1: 收集 DAG 全部节点
  llvm::DenseSet<scf::IfOp> allNodes;
  for (auto &entry : info->ifBlockDAG) {
    allNodes.insert(entry.first);
    for (auto &edge : entry.second)
      allNodes.insert(edge.first);
  }

  // Step 2: 计算入度（cross/intra 每条边都 +1）
  llvm::DenseMap<scf::IfOp, int> inDegree;
  for (auto &entry : info->ifBlockDAG)
    for (auto &edge : entry.second)
      inDegree[edge.first]++;

  // Step 3: 入度 0 即起点
  llvm::SmallVector<scf::IfOp> startNodes;
  for (auto node : allNodes)
    if (inDegree.lookup(node) == 0)
      startNodes.push_back(node);

  // Step 4: 逐起点 DFS 求最大深度，深度恰 = 3 的节点成对
  constexpr int targetDepth = 3;
  for (scf::IfOp start : startNodes) {
    llvm::DenseMap<scf::IfOp, int> depth;
    llvm::SmallVector<std::pair<scf::IfOp, int>> worklist;
    worklist.push_back({start, 1});
    depth[start] = 1;

    while (!worklist.empty()) {
      auto [node, nodeDepth] = worklist.pop_back_val();
      auto it = info->ifBlockDAG.find(node);
      if (it == info->ifBlockDAG.end())
        continue;
      for (auto &edge : it->second) {
        scf::IfOp neighbor = edge.first;
        // CrossCore 边深度 +1，IntraCore 边深度 +0
        int step = (edge.second == IfBlockDepKind::CrossCore) ? 1 : 0;
        int candidate = nodeDepth + step;
        auto inserted = depth.try_emplace(neighbor, candidate);
        if (inserted.second) {
          worklist.push_back({neighbor, candidate});
        } else if (candidate > inserted.first->second) {
          inserted.first->second = candidate;   // 取所有路径最大值
          worklist.push_back({neighbor, candidate});
        }
      }
    }

    for (auto &entry : depth) {
      if (entry.second == targetDepth) {
        info->flowOptIfOpPairs[entry.first] = start;  // target -> source
      }
    }
  }
  return 0;
}
```

**要点**：
- **深度计分规则**：跨核边 +1、核内边 +0——跨核一次 = 一次完整的 buffer 轮换周期，核内同步不增加流水级数；
- **深度取"所有路径的最大值"**（非首次到达值），保证从起点到节点存在一条恰好 3 级跨核链即命中；
- 命中节点 → `flowOptIfOpPairs[target] = start`：距离 3 级的这对 if 启动流水线重叠优化（`UpdateConditionInfo` 据此跳过同步、提前 overlap）。

---

## 四、流程总结

### 4.1 InitDependentMap 的流水

```
InitDependentMapPass::runOnOperation
  ├─► Step 1: initCrossCoreDependentMap
  │     ├─► collectDepsByGroup(kCrossCoreDeps)       属性 → (group, op, role)
  │     ├─► buildCrossCoreProducerConsumerMapping     consumer -> [每 group producers]
  │     └─► filterMemCrossCoreDepsByMainLoop          consumer 与整组 producers 同主循环
  ├─► Step 2: initIntraCoreDependentMap
  │     ├─► collectDepsByGroup(kIntraDeps)            全模块一次
  │     └─► 逐 main_loop op：
  │           └─► buildIntraCoreProducerConsumerMapping   consumer 限本循环内（isConsumerInMainLoop）
  ├─► computeProducerBufferCount                       最大 producers.size()；空则回退 BufferCountManager
  ├─► buildIfBlockDAG                                  op 级 → if 级（findIfOpContainingOp），Cross/Intra 边
  ├─► detectCycle                                      DFS 三色，有环 → ERRCODE_IGNORED
  └─► collectFlowOptIfOpPairs（仅超阈值）
        ├─► 入度 0 = 起点
        └─► 每起点 DFS：Cross +1 / Intra +0，深度恰 3 → flowOptIfOpPairs[target]=start
```

### 4.2 与前后子 Pass 的协作

- **前置**：`AllocMultiCache` 打的 `kCrossCoreDeps`/`kIntraDeps` 属性；`CreateIfOps` 打的 `kIf` 标记（DAG 节点身份）；
- **产出**：`crossCoreDependentMap`、`intraCoreDependentMap`（供 UpdateLoopOps/UpdateLoopIterTimes 计算缓冲因子、插入同步）、`ifBlockDAG`、`flowOptIfOpPairs`、`crossCoreBufferCount`/`intraCoreBufferCount`（供 UpdateConditionInfo 分配 SSBuffer 与开关 flowOpt）；
- **后置**：`UpdateLoopOps` 读取依赖映射插入 PIPE_S 同步；`UpdateConditionInfo` 读取映射 + DAG + flowOpt 对构造真实条件。

---

## 五、面试要点

1. **为什么依赖映射要区分跨核/核内，且存储结构不同？** 跨核是"两个核的两个主循环之间"的同步，consumer 与 producers 必须在同一 main_loop（`filterMemCrossCoreDepsByMainLoop`），结构是 `consumer -> 每 group 一份 producers`；核内是"同一循环内 block 之间"，结构是 `main_loop -> consumer -> producers`，consumer 用 `isConsumerInMainLoop` 排除嵌套主循环。不同维度决定不同缓冲/同步策略。

2. **`isConsumerInMainLoop` 为什么返回 0 有两种含义？** 沿 parent 链上溯：命中**其他**主循环（consumer 在嵌套主循环里，跳过）或命中**目标**主循环（收入 consumers）都返回 0，只有找不到任何主循环才返回 -1。返回 0 的成功但"收入与否"要区分。

3. **同组 producers 必须同主循环，为什么？** 一个依赖 group 的多个 producer 是"同一缓冲槽位的轮换生产者"（如 matmul 的多个切分），它们必须来自同一个主循环（同一核、同一流水线），否则该依赖组无意义且下游无法统一插入同步。

4. **if 块 DAG 为什么要求 producer 和 consumer 不在同一 if？** 每个 block 一个 if，同块内 producer→consumer 是块内自依赖，应在 `CloneOps` 阶段已消除（op 独立化）。DAG 里出现同块自环说明克隆/分组逻辑有误，直接失败更安全。

5. **flowOpt 深度计分为什么 CrossCore +1、IntraCore +0？** 流水线重叠的"级数"由跨核同步次数决定：跨核边代表一次完整的核间 buffer 轮换（同步开销大），核内边只是同核内的前后关系，不增加流水线深度。深度 = 3 表示从起点经 3 次跨核依赖到目标，此时缓冲数已足够支撑重叠。

6. **环检测失败为什么用 `ERRCODE_IGNORED` 而不是 `ERRCODE_FAILED`？** 依赖有环是 kernel 本身的拓扑特征（不可流水线化），属于"预期内的不适用"，上层按可忽略回退处理；`ERRCODE_FAILED` 留给格式错误、缺属性等内部异常，两者 fallback 语义不同（见主编排文档的 errcode 处理）。

7. **缓冲数回退到 BufferCountManager 的动机？** 核内显式依赖属性可能为空（该循环没有属性标注的核内依赖），但流程仍需一个合理默认缓冲数来继续 flowOpt 阈值判断；回退到配置管理的 IntraCore 值保证决策一致。
