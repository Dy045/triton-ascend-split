# SplitDataflow 子 Pass 逐行讲解（五）：MarkMainLoop

> 源码文件：`third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/MarkMainLoop.cpp`
> 流水线位置：`SplitDataflowPass` 的 7 个步骤中的 **第 4 步**

---

## 一、这个 Pass 在做什么

标记 module 中的"主循环"（main loop）：找出包含 `hivm.fixpipe` 或 `hivm.copy` 的 `scf.for` / `scf.while` 循环，给它们打上 `ssbuffer.main_loop` 属性。这个属性后续会被 `SeparateCVScope`、`AddControlFlowCondition`、`AllocMultiCache` 等 pass 用来识别"核心计算循环"，从而决定多缓冲、同步、流水线控制的具体策略。

**本次更新要点**：
- 新增 **`scf.while` 循环支持**（与 `scf.for` 同等对待，适配 while 形态的主循环）；
- 新增 **`isL1Fixpipe` 过滤**：目标是 L1 的 fixpipe（即 C→C 传输，见 04 篇 `handleCubeToCube`）**不算**主循环候选——C2C 走 L1 直传，不经过跨核流水线，不需要 main_loop 化；
- 收集判断统一走 `CVPipeline::isMainLoopOp` 工具函数（`Common/Utils.h`：for/while 且带 kMainLoop 属性）。

---

## 二、关键背景

| 概念 | 说明 |
|---|---|
| `hivm::FixpipeOp` | Cube 的搬运指令：C→V（写 UB）或 C→C（写 L1） |
| `hivm::CopyOp` | Vector→Cube 的搬运指令 |
| `ssbuffer.main_loop` | `kMainLoop` 属性，值为整数（主循环编号） |
| `isL1Fixpipe` | fixpipe 目标地址空间是 L1 → C2C 直传，跳过 |
| `CVPipeline::isMainLoopOp` | `isa<scf::ForOp, scf::WhileOp> && hasAttr(kMainLoop)` |

关键洞察：经过 `InterCoreTransferAndSync` 之后，**跨核数据传输（fixpipe/copy）所在的循环，就是需要做流水线优化的核心循环**；而 C→C 的 L1 fixpipe 是核内直传，不属于此类。

---

## 三、逐行讲解

### 3.1 头文件与入口（L23-L47）

```cpp
#include "ascend/include/DynamicCVPipeline/SplitDataflow/MarkMainLoop.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"  // hivm::FixpipeOp, hivm::CopyOp
#include "mlir/IR/Operation.h"
#include "llvm/Support/Debug.h"
```

```cpp
void MarkMainLoopPass::runOnOperation() {
  ModuleOp module = getOperation();
  if (CVPipeline::hasFallbackAttr(module)) return;   // fallback 兜底

  int mainLoopIdCounter = 0;
  SmallVector<Operation *> mainLoops;      // 本次更新：类型放宽为 Operation*
                                             // （既要装 ForOp 也要装 WhileOp）
```

### 3.2 isL1Fixpipe：C→C fixpipe 判定（L50-L61，新增）

```cpp
// Find all candidate main loops (ForOp + WhileOp)
auto isL1Fixpipe = [](Operation *op) -> bool {
  auto fixpipeOp = dyn_cast<hivm::FixpipeOp>(op);
  if (!fixpipeOp) return false;
  auto dstType = dyn_cast<MemRefType>(fixpipeOp.getDst().getType());
  if (!dstType) return false;
  auto addrSpaceAttr =
      dyn_cast_or_null<hivm::AddressSpaceAttr>(dstType.getMemorySpace());
  return addrSpaceAttr &&
         addrSpaceAttr.getAddressSpace() == hivm::AddressSpace::L1;
};
```

- 读 fixpipe **目的 memref 的 memory space**：是 L1 → 这是 `handleCubeToCube` 插的核内直传（matmul 链共享 L1 buffer），不是跨核流水线的一部分。

### 3.3 第一步：找出候选主循环（L63-L74，更新）

```cpp
module.walk([&](Operation *op) {
  if (!isa<hivm::FixpipeOp, hivm::CopyOp>(op)) return;

  if (isL1Fixpipe(op)) return;                        // 新增：跳过 C2C 直传

  if (auto forOp = op->getParentOfType<scf::ForOp>())
    mainLoops.push_back(forOp);                        // 最近 for 祖先
  if (auto whileOp = op->getParentOfType<scf::WhileOp>())
    mainLoops.push_back(whileOp);                      // 新增：最近 while 祖先
});
```

- `getParentOfType<T>()` 向上查找最近的 T 类型祖先。
- 注意：同一个循环可能包含多个 fixpipe/copy，`mainLoops` 里可能有**重复项**（靠后面 `hasAttr` 去重）。
- **while 支持**：Triton 中 `while` 语句生成的 `scf.while`（如 FlashAttention 变体的动态长度循环）同样是候选主循环。

### 3.4 第二步：给主循环打编号（L76-L84）

```cpp
for (Operation *loopOp : mainLoops) {
  if (!loopOp->hasAttr(CVPipeline::kMainLoop)) {
    // 用整数属性记录主循环编号（从 0 开始递增）
    loopOp->setAttr(CVPipeline::kMainLoop,
                    Builder(module.getContext()).getI32IntegerAttr(mainLoopIdCounter));
    mainLoopIdCounter++;
  }
}
```

- `hasAttr` 检查避免 `mainLoops` 重复项导致的重复编号。

### 3.5 第三步：嵌套循环只保留最内层（L86-L107，更新）

```cpp
// Remove main_loop attribute from outer loops if nested loops both have it
// Keep only the innermost main_loop
SmallVector<Operation *> allMainLoops;
module.walk([&](Operation *loopOp) {
  if (CVPipeline::isMainLoopOp(loopOp)) {       // 本次更新：统一工具函数
    allMainLoops.push_back(loopOp);             // （for/while 通用判断）
  }
});

for (Operation *loopOp : allMainLoops) {
  bool hasNestedMainLoop = false;
  loopOp->walk([&](Operation *nestedLoopOp) {
    if (nestedLoopOp != loopOp && CVPipeline::isMainLoopOp(nestedLoopOp)) {
      hasNestedMainLoop = true;
    }
  });
  if (hasNestedMainLoop) {                       // 外层里还有内层主循环
    loopOp->removeAttr(CVPipeline::kMainLoop);   // → 移除外层标记
  }
}
```

- **为什么只保留最内层？** 真正的"数据搬运 + 计算"流水线发生在最内层循环。外层循环的搬运往往只是内层的"外壳"，都标记会导致后续流水线 pass 在错误的层级做多缓冲/同步。

### 3.6 Pass 创建（L112-L119）

```cpp
std::unique_ptr<OperationPass<ModuleOp>> createMarkMainLoopPass() {
  return std::make_unique<MarkMainLoopPass>();
}
```

---

## 四、算法流程总结

```
1. 找出所有包含 fixpipe/copy 的 for/while 循环（候选主循环）
   - fixpipe 目标是 L1（C2C 直传）的跳过
2. 给候选主循环按出现顺序编号（0,1,2,...）
3. 若嵌套循环都带 main_loop，只保留最内层
```

复杂度：O(N × depth)，N 为 op 数，depth 为嵌套深度。

---

## 五、面试要点

1. **如何识别"主循环"？**
   以"是否包含跨核传输指令（fixpipe/copy）"为标志。这些指令是 `InterCoreTransferAndSync` 刚插入的，它们所在的循环必然是数据需要跨核流动、值得流水线化的核心循环。

2. **为什么 C→C 的 L1 fixpipe 要排除？**
   C2C fixpipe 是 Cube 核**内部**的 L1 直传（matmul 链共享 L1 buffer），没有跨核流水线需求；若把它所在循环也标记成 main_loop，会让后续多缓冲/同步 pass 在不需要的循环上做文章，浪费 buffer 资源甚至破坏正确性。判断依据是 fixpipe 目标 memref 的 `AddressSpace == L1`。

3. **为什么嵌套时只保留最内层？**
   流水线优化（多缓冲、ping-pong、同步）应作用于真正反复搬运数据的最内层循环。外层只是包围结构，若也标记，后续 pass 会在错误层级插入缓冲和同步。

4. **while 循环是怎么支持的？**
   候选收集时对每个传输 op 同时查 `getParentOfType<scf::ForOp>()` 和 `getParentOfType<scf::WhileOp>()`；后续判断统一用 `isMainLoopOp`（for/while 都认），三段逻辑天然覆盖 while。
