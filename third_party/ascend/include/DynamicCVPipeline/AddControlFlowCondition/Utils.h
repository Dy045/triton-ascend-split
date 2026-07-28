/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UTILS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UTILS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
namespace triton {

// Collect all nested ops within an operation's regions
LogicalResult collectAllNestedOps(Operation *op,
                                  llvm::DenseSet<Operation *> &regionOps);

// Returns the body block (single block of loop body ops + scf.yield
// terminator) of a main-loop op (scf.for or scf.while). Returns nullptr
// if `op` is neither.
Block *getMainLoopBodyBlock(Operation *op);

// Group operations by their block_id attribute. `op` must be a main-loop
// op (scf.for or scf.while) carrying ssbuffer.main_loop. Returns failure
// otherwise.
LogicalResult
collectOpsByBlockId(Operation *op,
                    llvm::DenseMap<int, SmallVector<Operation *>> &blockOps);

// Topological sort of operations based on operand dependencies
LogicalResult topologicalSort(llvm::DenseSet<Operation *> &ops,
                              llvm::DenseMap<Operation *, int> *opOrder,
                              SmallVectorImpl<Operation *> &sorted);

LogicalResult topologicalSort(SmallVector<Operation *> &ops);

// Get block_ids in order of appearance in the main-loop body (forOp body
// or whileOp after-region body). Returns empty if `op` is neither.
SmallVector<int> getBlockIdsInOrder(Operation *op);

// Count unique ssbuffer.if values inside a main-loop op (scf.for or scf.while
// carrying ssbuffer.main_loop), walking all nested ops. Returns 0 if none.
int countUniqueIfBlockIds(Operation *loopOp);

// Get the block_id of the immediate child of the main-loop (scf.for or
// scf.while carrying ssbuffer.main_loop) that contains op. For scf.while,
// "body" means the after-region block.
std::optional<int> getLoopDirectChildBlockId(Operation *op);

// Find the tcb group id that contains value v
int findTcbGroupId(
    Value v,
    llvm::DenseMap<int, SmallVector<Value>> &tightlyCoupledBufferGroups);

// Set isCube/isVector based on the scope's tcore_type attribute
// Returns failure if scopeOp does not have tcore_type attribute
LogicalResult getScopeType(Operation *scopeOp, bool &isCube, bool &isVector);

// Check if op is a scf.if whose body only contains hivm.hir.sync_block_wait,
// hivm.hir.sync_block_set and hivm.fixpipe ops (excluding terminators).
// Returns false if op is not a scf.if or contains any other op.
bool isIfOpWithOnlySyncOps(Operation *op);

} // namespace triton
} // namespace mlir
#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UTILS_H
