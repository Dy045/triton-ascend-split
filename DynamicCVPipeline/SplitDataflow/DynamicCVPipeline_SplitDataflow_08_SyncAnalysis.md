# SplitDataflow 子分析专题（八）：SyncAnalysis —— 循环维度 flag 复用分析

> 源码文件：[`SyncAnalysis.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/SyncAnalysis.cpp)（182 行，**main-dev 新增**）
> 头文件：[`SyncAnalysis.h`](../../../third_party/ascend/include/DynamicCVPipeline/SplitDataflow/SyncAnalysis.h)（37 行）
>
> 流水线位置：不是独立注册的 Pass，而是 `DataDependencyAnalysisPass::runOnOperation` 末尾调用的**分析函数**（`analyzeSyncReuse`），产出结果直接写回 `DependencyInfo`，供下游 `InterCoreTransferAndSync` 使用。

---

## 一、这个分析在做什么

在旧版里，flag id 由 `InterCoreTransferAndSync` 在处理每条依赖时用 `FlagIdManager::acquireId()` 现场分配，再由 `FlagIdReuseManager` 在"flag 超限"时才做图着色压缩。main-dev 把**flag id 的分配与复用决策提前到分析阶段**：

`SyncAnalysis` 的核心目标是——**让处于不同循环 region 的依赖复用同一个 flag id**，从而减少 flag 数量。它的思路是：

1. 给每条依赖打上"它属于哪个循环深度 / 哪个循环 region"（`analyzeDepLoopInfo` 填 `loopRegion` + `loopDepth`）；
2. 在同一循环深度内，把所有依赖按"循环 region"分组；**依赖最多的那个 region 是"主导 region"**，它的 flag id 原样保留；
3. 其他 region 的依赖**轮流复用主导 region 的 flag id**（round-robin），并标记 `reuseStartEnd = true`；
4. 最后把全部 flag id 重映射为从 0 开始的连续数值。

`reuseStartEnd` 标记的含义：被复用 flag 的依赖需要额外的 **start/end 同步对**（在循环开始前 set、循环结束后 wait），因为同一个 flag id 在多个 region 之间复用后，必须靠循环边界上的同步来"重设"信号，避免不同 region 之间的 set/wait 互相串扰。这正是 `InterCoreTransferAndSync::insertInterCoreSync` 里新增 `if (!dep.reuseStartEnd)` 分支的依据。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `loopRegion` / `loopDepth` | 每条依赖所属的最内层循环 op 及其嵌套深度，由 `analyzeDepLoopInfo` 填充（`loopDepth` 取自 `ssbuffer.loop_depth` 属性，见 `Utils.cpp` 的 `setOpLoopDepth`/`getOpLoopDepth`） |
| `regionID` | group 用的 key：`CVPipeline::getOpBlockId(dep->loopRegion)`，即该循环 op 的 block_id |
| 主导 region | 同一循环深度内依赖数量最多的 region，其 flag id 被当作"master flags"保留 |
| `reuseStartEnd` | 依赖的 flag id 被其他 region 复用时置 `true`；需要额外的 start/end 边界同步 |
| `masterFlags` | 主导 region 中去重后的 flag id 列表，其他 region 轮流复用它们 |
| `isOutlineReuseBlacklistKernel` | 黑名单 kernel（`kernel_da_fwd_u`）跳过复用（`Utils.cpp` 新增） |
| `remapFlagIdsToContinuous` | 把分散的 flag id 重映射为 0..n 连续编号，保序且保留"共享 id"关系 |

---

## 三、逐行讲解

### 3.1 `analyzeDepLoopInfo`（L38-L62）：填充循环信息

```cpp
void analyzeDepLoopInfo(ModuleOp module, DependencyInfo &dep) {
  // Step 1: PreOrder walk 找到第一个 block_id == dep.producerBlockId 的 op，
  //         作为"生产者块"的锚点 op
  // Step 2: 从该 op 向上逐层找最近的外层 scf::ForOp / scf::WhileOp
  //         → dep.loopRegion = parent; dep.loopDepth = getOpLoopDepth(parent);
}
```

- **定位生产者块**：用 `dep.producerBlockId` 找 block 里第一个 op，而不是遍历全部——依赖的生产者块就是该依赖"属于哪个 block"的判断依据；
- **找最内层循环**：找到第一个 `scf::ForOp`/`scf::WhileOp` 祖先即停。`loopDepth` 通过 `getOpLoopDepth` 读 op 上预先打好的 `ssbuffer.loop_depth` 属性（由 `Utils.cpp::setOpLoopDepth` 在别处打上：从 op 向上数 scf.for/while 的层数 + 1）；
- 注意：这个函数在 `DataDependencyAnalysisPass::collectMemDepInfo` 里被调用（每条内存依赖都会被填上循环信息），而 V2C/C2V 依赖的循环信息是在 `analyzeSyncReuse` 里"就地"推断的（见下）。

### 3.2 `assignDepFlagIds`（L71-L132）：region 分组 + 复用决策

```cpp
void assignDepFlagIds(SmallVector<DependencyInfo> &deps) {
  // Step 1: 按 loopDepth 分组（跳过 loopRegion 为空或 depth==0 的依赖）
  DenseMap<int, SmallVector<DependencyInfo*>> loopDepthGroups;
  for (auto &dep : deps)
    if (dep.loopRegion && dep.loopDepth != 0)
      loopDepthGroups[dep.loopDepth].push_back(&dep);

  for (auto &[loopDepth, depList] : loopDepthGroups) {
    // Step 2: 同深度内再按 regionID（循环 op 的 block_id）分组
    DenseMap<int, SmallVector<DependencyInfo*>> regionGroups;
    for (auto *dep : depList)
      regionGroups[getOpBlockId(dep->loopRegion).value_or(-1)].push_back(dep);

    // Step 3: 找依赖最多的 region = 主导 region
    int maxRegion = ...;   // 依赖数最多的 region key
    size_t maxSize = ...;

    // Step 4: 收集主导 region 的去重 flag id = masterFlags
    SmallVector<int> masterFlags;
    DenseSet<int> seen;
    for (auto *dep : regionGroups[maxRegion])
      if (seen.insert(dep->flagId).second) masterFlags.push_back(dep->flagId);

    if (masterFlags.empty()) continue;

    // Step 5: 主导 region 保留，其他 region 轮流复用 masterFlags
    for (auto &[region, group] : regionGroups) {
      if (region == maxRegion) {
        for (auto *dep : group) dep->reuseStartEnd = false;   // 主导：不额外同步
        continue;
      }
      size_t idx = 0;
      for (auto *dep : group) {
        dep->flagId = masterFlags[idx % masterFlags.size()];   // round-robin 复用
        dep->reuseStartEnd = true;                             // 复用：需要 start/end 同步
        ++idx;
      }
    }
  }
}
```

**核心思想**：
- **只在同一 `loopDepth` 内复用**：不同循环深度的 flag 生命周期不同，混用会串扰；同深度的不同 region 天然是"并在循环里互不嵌套"的关系，具备复用安全前提。
- **主导 region 决定 master flags**：选择依赖最多的 region 作为"信号基准"，它的 flag id 尽可能分散（不做任何复用），保证自身同步语义最稳；其余 region 去"蹭"这些 id。
- **round-robin 复用**：非主导 region 的依赖依次取 `masterFlags[idx % N]`，把多个 region 的依赖压缩到同一批 flag id 上。
- **`reuseStartEnd` 标记**：被复用的依赖（非主导 region）必须额外补一对 start/end 边界同步（见 3.4 与 doc 04），否则同一 flag 在多个 region 间同时 set/wait 会信号错乱。

### 3.3 `remapFlagIdsToContinuous`（L136-L153）：连续化重映射

```cpp
void remapFlagIdsToContinuous(SmallVector<SmallVector<DependencyInfo>*> &depLists) {
  DenseMap<int, int> idMap;
  int nextId = 0;
  for (auto *list : depLists)
    for (auto &dep : *list) {
      int old = dep.flagId;
      if (!idMap.count(old)) { idMap[old] = nextId++; }   // 首次出现 → 分配新连续 id
      dep.flagId = idMap[old];                            // 共享 id 的依赖映射到同一个新 id
    }
}
```

- 经过 `assignDepFlagIds` 复用后，flag id 可能稀疏散落（被复用成 masterFlags 里的 id），这一步把它们**压缩成 0..n 连续编号**；
- 关键：**共享同一个 flag id 的依赖必须映射到同一个新 id**（`idMap` 保证），这保持了"同一 flag 保护同一组 set/wait"的语义。

### 3.4 `analyzeSyncReuse`（L157-L182）：入口

```cpp
void analyzeSyncReuse(DataDependencyInfo &info, ModuleOp module) {
  int flag_id = 0;
  // Step 1: 给三类依赖顺序分配初始 flag id（0,1,2,...）
  for (auto &dep : info.getV2CDependencies())  dep.flagId = flag_id++;
  for (auto &dep : info.getC2VDependencies())  dep.flagId = flag_id++;
  for (auto &dep : info.getMemoryDependencies()) dep.flagId = flag_id++;

  // Step 2: 黑名单 kernel 直接返回（不做复用）
  if (isOutlineReuseBlacklistKernel(module)) return;

  // Step 3: 分别对三类依赖做 region 分组复用
  assignDepFlagIds(info.getV2CDependencies());
  assignDepFlagIds(info.getC2VDependencies());
  assignDepFlagIds(info.getMemoryDependencies());

  // Step 4: 三类依赖合并连续化重映射
  SmallVector<SmallVector<DependencyInfo>*> depLists = {V2C, C2V, MemDeps};
  remapFlagIdsToContinuous(depLists);
}
```

- **顺序分配**：先给三条依赖列表各自顺序编号，保证每个依赖一开始有唯一 id；
- **黑名单跳过**：`kernel_da_fwd_u` 这种 kernel 不参与复用（`reuseStartEnd` 全为 false，行为退化为旧版"每依赖一 flag"）；
- **复用后连续化**：四条列表里只有 V2C/C2V/MemDeps 参与复用与重映射（C2C 不走 flag 同步，天然没有 flag id 需求，见 doc 04）。

---

## 四、算法流程总结

```
analyzeSyncReuse(info, module)
├─ 顺序分配初始 flag id（V2C → C2V → MemDeps）
├─ isOutlineReuseBlacklistKernel ? 直接返回（不复用）
├─ assignDepFlagIds（每类依赖各做一次）
│   ├─ 按 loopDepth 分组
│   ├─ 同深度内按 loopRegion 的 block_id 分组
│   ├─ 依赖最多的 region = 主导 region → 收集 masterFlags
│   └─ 其他 region round-robin 复用 masterFlags + 标记 reuseStartEnd=true
└─ remapFlagIdsToContinuous（三类合并，压缩为 0..n 连续 id，保留共享关系）
```

**整体策略**：把"flag 复用"从 `InterCoreTransferAndSync` 的运行时压缩（flag 超限才触发）前移为**分析阶段的确定性决策**——用循环 region 分组复用 flag id，并用 `reuseStartEnd` 标记告诉下游"哪些依赖需要额外的 start/end 边界同步"。这比旧版的"超限后才图着色"更前置、更可预测，且把复用粒度对齐到"循环 region"这一硬件天然边界。

---

## 五、面试要点

1. **为什么把 flag 复用从 `InterCoreTransferAndSync` 前移到 `DataDependencyAnalysis` 末尾？**
   旧版在 flag 超限时才用图着色压缩，是"事后补救"；新版在分析阶段就根据循环结构确定性地分配/复用 flag id，下游 `InterCoreTransferAndSync` 直接 `dep.flagId` 取用，不再需要 `FlagIdManager::acquireId` 的现场分配 + 事后压缩两阶段。分析前移让 flag id 的分配决策单一、可控、可预测。

2. **为什么"只有同一 `loopDepth` 内才复用"？**
   不同循环深度的依赖，其 flag 的活跃区间（set 到 wait 之间）跨越不同层级，若跨深度复用会互相覆盖；同深度的不同 region 是"并列"关系（互不嵌套），各自的 set/wait 区间天然错开，是复用安全的最小前提。

3. **"主导 region"为什么选依赖最多的那个？**
   主导 region 的 flag id 原样保留（不复用），自身同步语义最稳。选依赖最多的 region 当基准，能让"最复杂的那部分区域"保持最干净的信号、不因复用而引入额外约束；其余较小 region 去蹭它的 id，复用收益最大、风险最小。

4. **`reuseStartEnd = true` 到底让下游做什么？**
   它告诉 `InterCoreTransferAndSync`：这个依赖的 flag 被其他 region 复用了，光有"传输后 set / 消费前 wait"不够，还必须在**循环开始前 set、循环结束后 wait**——用边界同步把"上一个 region 的残留信号"隔离掉，否则复用同一 flag 的两个 region 会串扰（见 doc 04 的 `insertInterCoreSync` 新增分支）。

5. **`remapFlagIdsToContinuous` 为什么必须保留"共享 id"关系？**
   flag 复用的前提是"多个依赖复用同一个 id，因此共用同一组 flag 信号"。连续化只是把稀疏 id（经过 round-robin 复用后可能变成 masterFlags 里的任意值）压缩成 0..n，但**共享同一 id 的依赖必须仍映射到同一新 id**——否则会破坏"同一 flag 保护同一组 set/wait"的语义，导致同步错乱。

6. **C2C 依赖为什么不参与 flag 复用/分配？**
   C2C（Cube→Cube）走 L1 fixpipe 直传，依赖的是 L1 的天然顺序，**根本不插入 set/wait flag 同步**（用 `kIntraDeps` 属性标记组内依赖），所以 `analyzeSyncReuse` 只给 V2C/C2V/MemDeps 三类分配 flag id。

7. **黑名单 `isOutlineReuseBlacklistKernel` 的作用？**
   对 `kernel_da_fwd_u` 这类已知 kernel，复用 flag 会导致边界同步（outline start/end）产生额外开销甚至语义问题，直接禁用复用——退化为"每依赖一 flag"的旧行为。这体现了"激进优化要保守开启"的原则：复用是收益与风险的权衡，对不适用场景必须能一键回退。