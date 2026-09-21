# DynamicCVPipeline 子 Pass 逐行讲解：PreCheckAvailablePass

> 源码文件：
> - 主 Pass：[`third_party/ascend/lib/DynamicCVPipeline/PreCheckAvailable.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PreCheckAvailable.cpp)
> - 子 Pass：[`PreCheckAvailable/PreCheckBlacklist.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PreCheckAvailable/PreCheckBlacklist.cpp)、[`PreCheckAvailable/PreCheckMatmul.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PreCheckAvailable/PreCheckMatmul.cpp)、[`PreCheckAvailable/PreCheckDisablePreload.cpp`](../../../third_party/ascend/lib/DynamicCVPipeline/PreCheckAvailable/PreCheckDisablePreload.cpp)
>
> 流水线位置：`AddDynamicCVPipelinePass::addPasses` 中 **第 1 个 pass**（入口处，最先执行）

---

## 一、这个 Pass 在做什么

在投入昂贵的 CV 拆分流水线之前，先做**三重低成本预检**，判断"这个 kernel 是否适合、以及以什么参数跑动态 CV 流水线"：

1. **黑名单检查**（`PreCheckBlacklist`）：如果 IR 里已经出现 `scope.scope` 之类的 op，说明该 kernel 已被 Ascend 专用路径优化过（scope 已拆分完毕），动态 CV 流水线不应再插手 → `ERRCODE_IGNORED` 回退；
2. **Matmul 存在性检查**（`PreCheckMatmul`）：CV 流水线的价值来自 Cube 计算并行化。如果整个模块一个 `linalg.matmul` 都没有（纯 vector 计算）→ `ERRCODE_IGNORED` 回退；
3. **预加载参数检查**（`PreCheckDisablePreload`）：如果命中已知的"不支持 3-preload"函数名单，把 buffer 计数从默认的 3/2 降级为 2/1，避免后续 `AllocMultiCache` 用错参数。

主 Pass 是纯编排：任一子 Pass 失败或设置 fallback，就直接把 `ERRCODE_IGNORED` 挂到模块上，终止整个流水线（由 `AddDynamicCVPipeline` 统一回退标准编译）。

**为什么放在第 1 步**：它要尽早、廉价地拦住"不该做 CV 流水线"的输入，避免后面的 StandardizeOp、PlanComputeBlock 等几十个 Pass 白跑。

---

## 二、关键背景概念

| 概念 | 说明 |
|---|---|
| `ERRCODE_IGNORED` | fallback 错误码之一："kernel 不适合做动态 CV 流水线"，不是 Pass 出错，是主动放弃 |
| `hasFallbackAttr(module)` | 检查模块是否已挂 fallback 属性（`triton_ascend.dynamic_cv_pipeline.rc`）。任一子 Pass 设置后，后续 Pass 看到都直接返回 |
| `scope.scope` | Scope 方言的 op，标志 IR 已经被 Ascend 的 scope 拆分路径处理过 |
| `linalg.matmul` | 矩阵乘 op。它是 CV 流水线的"原料"，没有它就没有 Cube 计算可并行 |
| 3-preload | 双缓冲升级到三缓冲：`intra` 侧 3 份、`inter` 侧 2 份（buffer count 3/2） |
| `kIntraBufCount` | `"ssbuffer.intra_buf_count"`，intra-core 缓冲份数模块属性 |
| `kInterCoreBufCount` | `"ssbuffer.inter_core_buf_count"`，inter-core 缓冲份数模块属性 |
| `BufferCountManager` | 读写模块级 buffer 计数（`DepType::{IntraCore, InterCore, LoadStore}`）的管理器 |

---

## 三、逐行讲解

### 3.1 主 Pass：`PreCheckAvailablePass`（PreCheckAvailable.cpp L44-L79）

```cpp
void PreCheckAvailablePass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) { return; }   // 兜底

  PassManager pm(&getContext(), module.getOperationName());
  pm.addPass(createPreCheckBlacklistPass());      // 1. 黑名单 op
  pm.addPass(createPreCheckMatmulPass());         // 2. matmul 存在性
  pm.addPass(createPreCheckDisablePreloadPass()); // 3. 预加载参数降级

  if (failed(runPipeline(pm, module))) {
    // 任一子 Pass 失败（内部已设 fallback）→ 统一兜底为 IGNORED
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }
}
```

- 与 `StandardizeOpPass` 一样是**编排壳**：3 个子 Pass 各自独立注册、可单独用 `mlir-opt` 调试。
- `runPipeline` 失败时用 `ERRCODE_IGNORED`（主动放弃），而不是 `ERRCODE_FAILED`（内部错误）——语义上"这个输入本就不该走 CV 流水线"。
- 注册（L70-L79）：`registerPreCheckAvailablePasses()` 把 3 个子 Pass + 主 Pass 全部注册进 pass 库。

### 3.2 子 Pass 1：`PreCheckBlacklistPass`（PreCheckBlacklist.cpp L34-L83）

```cpp
// 黑名单：出现这些 op 说明已被 Ascend 优化路径处理过
static const llvm::SmallVector<llvm::StringRef> kBlacklistOpNames = {
    "scope.scope",
};

void PreCheckBlacklistPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) { return; }

  Operation *foundBlacklistOp = nullptr;
  module.walk([&](Operation *op) -> WalkResult {
    llvm::StringRef opName = op->getName().getStringRef();
    if (llvm::is_contained(kBlacklistOpNames, opName)) {
      foundBlacklistOp = op;
      return WalkResult::interrupt();      // 找到即中断遍历
    }
    return WalkResult::advance();
  });

  if (!foundBlacklistOp) { return; }       // 没有 → 通过

  CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
}
```

- 只有一种黑名单 op：`scope.scope`。它是"该 kernel 已经完成 scope 拆分"的信号——动态 CV 流水线本身就是干 scope 拆分的，面对已经拆好的 IR 没有意义，直接回退。
- `getDependentDialects`（L47-L50）声明 `scope::ScopeDialect`，保证 walk 时能识别该 op 的名字。
- 用 `WalkResult::interrupt()` 提前终止，命中一个就够了（O(N) 但可提前退出）。

### 3.3 子 Pass 2：`PreCheckMatmulPass`（PreCheckMatmul.cpp L38-L60）

```cpp
void PreCheckMatmul::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) { return; }

  linalg::MatmulOp firstMatmulOp = nullptr;
  module.walk([&](linalg::MatmulOp matmulOp) -> WalkResult {
    firstMatmulOp = matmulOp;
    return WalkResult::interrupt();
  });

  if (firstMatmulOp) { return; }           // 有 matmul → 通过

  CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
}
```

- 只要找到**至少一个** matmul 就通过。`interrupt()` 同样提前终止。
- 没有 matmul 意味着这个 kernel 是纯 vector 计算（elementwise、reduction 等），CV 拆分没有 Cube 侧可并行，回退。

### 3.4 子 Pass 3：`PreCheckDisablePreload`（PreCheckDisablePreload.cpp L38-L95）

```cpp
// 这些函数已知不支持 3-preload（显式多缓冲）
static const llvm::SmallVector<llvm::StringRef> kBlacklistFuncNames = {
    "_attn_bwd", "kernel_sdpa_fwd", "_swa_paged_decode_kernel",
    "_mqa_logits_kernel", "paged_decode_fd_kernel"};

void PreCheckDisablePreload::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) { return; }

  func::FuncOp foundBlacklistFunc = nullptr;
  module.walk([&](func::FuncOp func) -> WalkResult {
    if (llvm::is_contained(kBlacklistFuncNames, func.getSymName())) {
      foundBlacklistFunc = func;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!foundBlacklistFunc) { return; }     // 名单外 → 保持默认 3-preload

  // 特例：kernel_sdpa_fwd 显式配置了 3-preload（intra=3, inter=2）时保持
  if (foundBlacklistFunc.getSymName() == "kernel_sdpa_fwd") {
    auto intra = module->getAttrOfType<IntegerAttr>(CVPipeline::kIntraBufCount);
    auto inter = module->getAttrOfType<IntegerAttr>(CVPipeline::kInterCoreBufCount);
    if (intra && inter && intra.getInt() == 3 && inter.getInt() == 2) {
      return;                               // 显式 3-preload → 不降级
    }
  }

  // 降级：禁用 3-preload，退回双缓冲
  BufferCountManager bufferCountManager(module);
  bufferCountManager.setBufferCount(BufferCountManager::DepType::IntraCore, 2);
  bufferCountManager.setBufferCount(BufferCountManager::DepType::InterCore, 1);
}
```

- 名单里的 5 个函数是 Attention 系列的 kernel（flash-attn 前向/反向、mqa、paged_decode 等），这些 kernel 的 buffer 依赖复杂，3-preload（3 份 buffer 交替）可能超 UB 容量或破坏同步。
- `setBufferCount` 会往模块属性 `ssbuffer.intra_buf_count` / `ssbuffer.inter_core_buf_count` 写入 2 / 1，`AllocMultiCache` 按此分配多缓冲。
- **特例**：`kernel_sdpa_fwd` 如果前端已显式指定 `intra=3, inter=2`，说明该路径经过验证可行，不降级。
- 注意：`_attn_bwd` 也依赖 `AddDynamicCVPipeline` 失败重试时的 buffer 修改（见主流程：重试时同样设置 2/1），两处策略一致。

---

## 四、算法流程总结

```
PreCheckAvailablePass（编排壳）
├─ PreCheckBlacklist    : 存在 scope.scope        → ERRCODE_IGNORED 回退
├─ PreCheckMatmul       : 无任何 linalg.matmul    → ERRCODE_IGNORED 回退
└─ PreCheckDisablePreload: 命中名单函数
                          └─ kernel_sdpa_fwd 且已显式 3-preload → 保持
                          └─ 否则 buffer count 降级为 IntraCore=2 / InterCore=1
```

复杂度：三个子 Pass 均为单次 `module.walk`，O(N)。这是整个流水线里最便宜的 pass 之一，放在最前面。

---

## 五、面试要点

1. **为什么要有 PreCheck？**
   动态 CV 流水线由 10 个 Pass 组成，成本高且要求输入形态特定。先用 O(N) 的 walk 把"不该做的输入"（已 scope 优化过的、纯 vector 的）拦在入口，避免无效计算和潜在的误变换。这是典型的"前置快速失败（fail fast）"设计。

2. **`ERRCODE_IGNORED` 和 `ERRCODE_FAILED` 有什么区别？**
   `IGNORED` 表示"主动放弃：kernel 不适合"，是预期内路径，`AddDynamicCVPipeline` 只发 warning 不附加完整 IR；`FAILED` 表示"流水线内部出错"，通常意味着上游分析有 bug。两者都触发标准编译回退，但日志语义和调试价值不同。

3. **为什么 `scope.scope` 出现就要回退？**
   `scope.scope` 是 Ascend 专用路径已经做完 scope 拆分的标志。动态 CV 流水线的核心目标就是 scope 拆分（`SeparateCVScope`），对已拆分的 IR 再拆会产生重复、错误的 scope。识别方式是 op 名字黑名单，简单可靠。

4. **为什么没有 matmul 就不做 CV 流水线？**
   CV（Cube/Vector）并行化的收益全部来自"矩阵乘在 Cube 上跑、其他在 Vector 上跑"。纯 vector kernel 没有 Cube 侧计算可调度，拆分后核间同步的开销只会拖慢，不如直接走标准编译。

5. **3-preload 降级是怎么实现的？为什么是这几个函数？**
   通过 `BufferCountManager::setBufferCount` 把模块属性 `ssbuffer.intra_buf_count` 改为 2、`ssbuffer.inter_core_buf_count` 改为 1，后续 `AllocMultiCache` 按此分配。名单是 flash-attention 系列 kernel（`_attn_bwd`、`kernel_sdpa_fwd` 等），这些 kernel 的 L1/UB buffer 占用高，3 份缓冲会超容量或引入额外同步错误；`kernel_sdpa_fwd` 显式验证过 3-preload 可行为特例。

6. **PreCheck 和 `AddDynamicCVPipeline` 失败重试的关系？**
   主流程重试时（tuple-preload 失败后）也会把 buffer count 设为 2/1，与 `PreCheckDisablePreload` 的降级目标一致——都指向"特殊 kernel 用双缓冲"。区别是：PreCheck 是输入驱动（函数名单），重试是运行时驱动（实际失败）。
