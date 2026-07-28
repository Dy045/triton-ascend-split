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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/ProcessArgs.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "ProcessArgs";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace mlir;
using namespace triton;

// Collects mapping from iter_arg index to block_ids that use it. `ivOffset`
// is 1 for scf.for (IV at block arg 0; iter_args start at 1), 0 for
// scf.while (no IV; iter_args start at 0). Compares against
// body->getArguments() directly (rather than loopOp.getRegionIterArgs()) since
// for scf.while those are different blocks' args and SSA equality is
// block-local.
static LogicalResult collectArgIndexToBlockIds(
    Block *body, unsigned ivOffset,
    llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds) {
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: loop body is invalid or has no terminator\n");
    return failure();
  }

  for (Operation &op : body->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
    if (!blockIdAttr)
      continue;
    int blockId = blockIdAttr.getInt();

    for (OpOperand &operand : op.getOpOperands()) {
      Value v = operand.get();
      for (BlockArgument iterArg : body->getArguments()) {
        int argIdx = iterArg.getArgNumber();
        if (argIdx < (int)ivOffset) {
          // scf.for's IV at block arg 0 — never an iter_arg.
          continue;
        }
        // Skip tensor-type iter_args, only process scalar and index types
        if (mlir::isa<TensorType>(iterArg.getType())) {
          continue;
        }
        if (v == iterArg) {
          argIndexToBlockIds[argIdx - (int)ivOffset].insert(blockId);
        }
      }
    }
  }
  return success();
}

// Finds iter_args used by multiple block_ids (shared args).
// Determines owner block (first in order) and creates SharedArgInfo for each
// non-owner. Each non-owner block gets its own extra iter_arg.
static LogicalResult findSharedArgs(
    const llvm::DenseMap<int, llvm::DenseSet<int>> &argIndexToBlockIds,
    const SmallVector<int> &idsInOrder,
    SmallVector<SharedArgInfo> &sharedArgsInfo) {
  int extraArgCount = 0;
  for (auto &p : argIndexToBlockIds) {
    int argIndex = p.first;
    const llvm::DenseSet<int> &blockIds = p.second;

    if (blockIds.size() <= 1)
      continue;

    int ownerBlockId = -1;
    for (int id : idsInOrder) {
      if (blockIds.contains(id)) {
        ownerBlockId = id;
        break;
      }
    }
    if (ownerBlockId == -1)
      continue;

    // Each non-owner block for this argIndex gets its own extra iter_arg
    for (int bid : blockIds) {
      if (bid != ownerBlockId) {
        sharedArgsInfo.push_back(
            SharedArgInfo(argIndex, ownerBlockId, extraArgCount, bid));
        extraArgCount++;
      }
    }
  }
  return success();
}

// Returns the operation that defines the iter_arg at `argIndex` in `body`'s
// scf.yield (top of the update chain). Returns nullptr if the operand is out
// of bounds or is a BlockArgument (i.e., the chain doesn't have a defining
// op at this point). Shared by both the shared-iter-args and the
// while-cond-iter-args paths.
static Operation *findYieldDefiningOp(Block *body, unsigned argIndex) {
  auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
  if (argIndex >= yieldOp.getNumOperands()) {
    return nullptr;
  }
  return yieldOp.getOperand(argIndex).getDefiningOp();
}

// Collects all operations in the computation chain by backward traversal
// from compOp, scoped to ops inside `loopOp`'s body.
static void collectChainOps(Operation *loopOp, Operation *compOp,
                            llvm::DenseSet<Operation *> &chainOps) {
  SmallVector<Operation *> worklist;
  worklist.push_back(compOp);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (chainOps.contains(op))
      continue;
    chainOps.insert(op);

    for (Value operand : op->getOperands()) {
      if (auto *defOp = operand.getDefiningOp()) {
        if (defOp->getParentOp() == loopOp && !chainOps.contains(defOp)) {
          worklist.push_back(defOp);
        }
      }
    }
  }
}

// Builds computation info (compOp and chainOps) for each shared arg.
// `body` is the loop body (forOp body or whileOp after-body) used to locate
// the scf.yield terminator.
static LogicalResult buildCompInfoForSharedArgs(
    Operation *loopOp, Block *body, SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation *>> &sharedArgToChainOps) {
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    if (sharedArgToCompOp.contains(argIndex))
      continue;

    Operation *compOp = findYieldDefiningOp(body, argIndex);
    if (!compOp) {
      continue;
    }

    sharedArgToCompOp[argIndex] = compOp;

    llvm::DenseSet<Operation *> chainOps;
    collectChainOps(loopOp, compOp, chainOps);
    sharedArgToChainOps[argIndex] = chainOps;
  }
  return success();
}

// Creates a new scf.for or scf.while op with one extra iter_arg per index in
// `origIndices`. Each new init arg's value is the same as the existing
// iter_arg it shadows (at the same index). scf.for uses its single-region
// builder API; scf.while additionally creates before+after blocks and grows
// result types in lockstep with the new iter_args. Both copy attrs from the
// old op. Returns nullptr if `op` is neither scf.for nor scf.while. Caller
// casts the result. Exceeds the 50-line helper limit (~55 lines) per the
// "single entry, differentiated processing" principle — splitting back into
// per-op-type factories would defeat the goal. `origIndices` is computed by
// the caller: from `sharedArgsInfo` for the shared-iter-args path, or from
// `WhileIterArgClonePlan::newArgDescriptors` for the while-cond-iter-args
// path.
static Operation *createNewLoopOp(Operation *op,
                                  ArrayRef<unsigned> origIndices) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    OpBuilder builder(forOp);
    SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                   forOp.getInitArgs().end());
    for (unsigned origIdx : origIndices) {
      newInitArgs.push_back(forOp.getInitArgs()[origIdx]);
    }
    scf::ForOp newForOp = builder.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    for (auto &attr : forOp->getAttrs()) {
      newForOp->setAttr(attr.getName(), attr.getValue());
    }
    return newForOp;
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    OpBuilder builder(whileOp);
    SmallVector<Value> newInits(whileOp.getInits().begin(),
                                whileOp.getInits().end());
    SmallVector<Type> newResultTypes(whileOp->getResultTypes().begin(),
                                     whileOp->getResultTypes().end());
    for (unsigned origIdx : origIndices) {
      Value init = whileOp.getInits()[origIdx];
      newInits.push_back(init);
      newResultTypes.push_back(init.getType());
    }
    scf::WhileOp newWhileOp = builder.create<scf::WhileOp>(
        whileOp.getLoc(), newResultTypes, newInits);

    SmallVector<Type> argTypes;
    argTypes.reserve(newInits.size());
    for (Value v : newInits) {
      argTypes.push_back(v.getType());
    }
    SmallVector<Location> argLocs(newInits.size(), whileOp.getLoc());
    builder.createBlock(&newWhileOp.getBefore(), {}, argTypes, argLocs);
    builder.createBlock(&newWhileOp.getAfter(), {}, argTypes, argLocs);

    for (auto &attr : whileOp->getAttrs()) {
      newWhileOp->setAttr(attr.getName(), attr.getValue());
    }
    return newWhileOp;
  }

  return nullptr;
}

// Migrates operations from old block to new block.
// Redirects block arguments to new block arguments and moves all ops.
static void migrateBody(Block *oldBlock, Block *newBlock) {
  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlock->getArgument(i).replaceAllUsesWith(newBlock->getArgument(i));
  }

  for (Operation &op :
       llvm::make_early_inc_range(oldBlock->without_terminator())) {
    op.moveBefore(newBlock, newBlock->end());
  }
}

// Clones the computation chain for a non-owner block.
// Topologically sorts the chain and clones each op with remapped operands.
// argRemapping: maps migrated iter_arg Value -> new extra iter_arg Value.
// resultMapper: maps original op results -> cloned op results.
// clonedArgIdx: unique index for this non-owner block's clone (used as
// ssbuffer.arg).
static LogicalResult
cloneChainForBlock(SharedArgInfo &info, Operation *compOp,
                   const llvm::DenseSet<Operation *> &chainOps, Block *newBlock,
                   IRMapping &argRemapping, OpBuilder &cloneBuilder,
                   IRMapping &resultMapper, int clonedArgIdx) {
  if (!compOp || chainOps.empty()) {
    return failure();
  }

  SmallVector<Operation *> sortedChain(chainOps.begin(), chainOps.end());
  if (failed(topologicalSort(sortedChain))) {
    return failure();
  }

  for (Operation *op : sortedChain) {
    IRMapping opMapper;
    for (OpOperand &operand : op->getOpOperands()) {
      Value oldVal = operand.get();
      Value newVal = oldVal;
      if (argRemapping.contains(oldVal)) {
        newVal = argRemapping.lookup(oldVal);
      } else if (resultMapper.contains(oldVal)) {
        // Operand is a result from earlier in the owner chain, use cloned
        // result
        newVal = resultMapper.lookup(oldVal);
      }
      opMapper.map(oldVal, newVal);
    }

    if (resultMapper.contains(op->getResult(0)))
      continue;

    Operation *cloned = cloneBuilder.clone(*op, opMapper);
    cloned->setAttr(CVPipeline::kBlockId,
                    cloneBuilder.getI32IntegerAttr(info.nonOwnerBlockId));
    cloned->setAttr(CVPipeline::kArg,
                    cloneBuilder.getI32IntegerAttr(info.argIndex));

    resultMapper.map(op->getResult(0), cloned->getResult(0));
    cloneBuilder.setInsertionPointAfter(cloned);
  }
  return success();
}

// Replaces iter_arg uses in non-owner block with the cloned iter_arg.
// argRemapping maps migrated iter_arg Value -> new extra iter_arg Value.
static LogicalResult replaceIterArgsInBlock(SharedArgInfo &info,
                                            Block *newBlock,
                                            IRMapping &argRemapping,
                                            OpBuilder &cloneBuilder) {
  for (Operation &op : newBlock->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
    if (!blockIdAttr || blockIdAttr.getInt() != info.nonOwnerBlockId)
      continue;

    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      Value operand = op.getOperand(i);
      if (argRemapping.contains(operand)) {
        Value newVal = argRemapping.lookup(operand);
        op.setOperand(i, newVal);
        op.setAttr(CVPipeline::kArg,
                   cloneBuilder.getI32IntegerAttr(info.argIndex));
      }
    }
  }
  return success();
}

// Processes each shared arg: finds insertion point, clones chain, replaces
// iter_args. `ivOffset` is 1 for scf.for (IV at block arg 0) and 0 for
// scf.while (no IV).
static LogicalResult processSharedArgsIteration(
    Block *newBlock, SmallVector<SharedArgInfo> &sharedArgsInfo,
    const llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
    const llvm::DenseMap<int, llvm::DenseSet<Operation *>> &sharedArgToChainOps,
    ValueRange iterArgs, unsigned ivOffset, SmallVector<Value> &clonedResults) {
  unsigned numOriginalIterArgs = iterArgs.size();
  unsigned extraIterArgsBase = ivOffset + numOriginalIterArgs;

  int clonedArgIdx = clonedResults.size();
  for (auto &info : sharedArgsInfo) {
    int argIndex = info.argIndex;
    info.iterArg = iterArgs[argIndex];

    // The migrated iter_arg (original iter_arg moved to new block)
    Value migratedIterArg = newBlock->getArgument(argIndex + ivOffset);
    // The new extra iter_arg added for this shared arg
    unsigned newExtraBlockArgIdx = extraIterArgsBase + info.newArgIndex;
    Value newExtraIterArg = newBlock->getArgument(newExtraBlockArgIdx);

    // Build argRemapping: migratedIterArg -> newExtraIterArg
    IRMapping argRemapping;
    argRemapping.map(migratedIterArg, newExtraIterArg);

    Operation *lastOpInBlock = nullptr;
    for (Operation &op : newBlock->without_terminator()) {
      auto blockIdAttr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
      if (blockIdAttr && blockIdAttr.getInt() == info.nonOwnerBlockId) {
        lastOpInBlock = &op;
      }
    }

    OpBuilder cloneBuilder(newBlock, newBlock->end());
    if (lastOpInBlock) {
      cloneBuilder.setInsertionPointAfter(lastOpInBlock);
    }

    IRMapping resultMapper;
    if (failed(cloneChainForBlock(info, sharedArgToCompOp.lookup(argIndex),
                                  sharedArgToChainOps.lookup(argIndex),
                                  newBlock, argRemapping, cloneBuilder,
                                  resultMapper, clonedArgIdx))) {
      continue;
    }

    if (failed(replaceIterArgsInBlock(info, newBlock, argRemapping,
                                      cloneBuilder))) {
      continue;
    }

    Value clonedResult =
        resultMapper.lookup(sharedArgToCompOp.lookup(argIndex)->getResult(0));
    clonedResults.push_back(clonedResult);
    clonedArgIdx++;
  }
  return success();
}

// Prepares all shared args data: collects arg->blockId mapping, finds shared
// args, and builds computation info for each shared arg.
static LogicalResult prepareSharedArgsData(
    Operation *loopOp, Block *body, SmallVector<SharedArgInfo> &sharedArgsInfo,
    llvm::DenseMap<int, Operation *> &sharedArgToCompOp,
    llvm::DenseMap<int, llvm::DenseSet<Operation *>> &sharedArgToChainOps) {
  if (!body || !body->mightHaveTerminator()) {
    LDBG("[Error]: loop body is invalid or has no terminator\n");
    return failure();
  }

  // ivOffset: 1 for scf.for (IV at block arg 0), 0 for scf.while (no IV).
  unsigned ivOffset = isa<scf::ForOp>(loopOp) ? 1 : 0;

  llvm::DenseMap<int, llvm::DenseSet<int>> argIndexToBlockIds;
  if (failed(collectArgIndexToBlockIds(body, ivOffset, argIndexToBlockIds))) {
    return failure();
  }

  SmallVector<int> idsInOrder = getBlockIdsInOrder(loopOp);
  if (idsInOrder.empty() && !getMainLoopBodyBlock(loopOp)) {
    LDBG("[Error]: loopOp is neither scf::ForOp nor scf::WhileOp\n");
    return failure();
  }
  if (failed(findSharedArgs(argIndexToBlockIds, idsInOrder, sharedArgsInfo))) {
    return failure();
  }

  if (sharedArgsInfo.empty()) {
    return success();
  }

  LDBG("[INFO]: Found " << sharedArgsInfo.size()
                        << " shared iter_args to process\n");

  if (failed(buildCompInfoForSharedArgs(loopOp, body, sharedArgsInfo,
                                        sharedArgToCompOp,
                                        sharedArgToChainOps))) {
    return failure();
  }

  return success();
}

// Builds new scf.yield in `newBlock`: copies operands from oldBlock's
// scf.yield, then appends `extraYieldValues`, creates the new yield, and
// erases the old one. Used by both the shared-iter-args path (where
// extraYieldValues is the per-non-owner-block cloned results) and the
// while-cond-iter-args path (where it's the per-(blockId, cond-used iter_arg)
// cloned results gathered from WhileIterArgClonePlan::clonedPerBlock).
static LogicalResult buildNewYieldOp(Block *oldBlock, Block *newBlock,
                                     Operation *newOp,
                                     ArrayRef<Value> extraYieldValues) {
  auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> yieldOperands;
  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    yieldOperands.push_back(oldYield.getOperand(i));
  }
  for (Value v : extraYieldValues) {
    yieldOperands.push_back(v);
  }
  OpBuilder builder = OpBuilder::atBlockEnd(newBlock);
  builder.create<scf::YieldOp>(newOp->getLoc(), yieldOperands);
  oldYield.erase();
  return success();
}

// Replaces all uses of the old main-loop op with the new op's results, erases
// the old op, and transfers its intraCoreDependentMap entry to the new op.
static LogicalResult replaceMainLoopOpAndErase(Operation *oldOp,
                                               Operation *newOp,
                                               ControlFlowConditionInfo *info) {
  if (oldOp->getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < oldOp->getNumResults(); ++i) {
      newResults.push_back(newOp->getResult(i));
    }
    oldOp->replaceAllUsesWith(newResults);
  }

  // Transfer intraCoreDependentMap entry from oldOp to newOp.
  if (info) {
    if (info->intraCoreDependentMap.count(oldOp)) {
      info->intraCoreDependentMap[newOp] = info->intraCoreDependentMap[oldOp];
      info->intraCoreDependentMap.erase(oldOp);
    }
  }

  oldOp->erase();
  return success();
}

// Forward declaration removed: replaceMainLoopOpAndEraseAndTransferWhileIdx
// was folded into the unified processSharedIterArgsInLoop orchestrator (its
// originalWhileIterArgIndices transfer is inlined there).

// Main entry point for processing shared iter_args in a single main-loop op
// (scf.for or scf.while).
//
// Pipeline (common to both): prepareSharedArgsData → createNew*Op →
// migrateBody → processSharedArgsIteration → build terminator(s) →
// replaceMainLoopOpAndErase. forOp and whileOp differ only in op factory (1 vs
// 2 regions migrated), iteration target (body + ivOffset), and terminator(s).
// Builds the new scf.condition in the new while op's before region: cond
// value preserved from the old scf.condition (its defining op was moved into
// the new before block by migrateBody and its block-arg uses remapped).
// Forwarded operands must include all of newWhileOp's before-block args
// (incl. extra iter_args), so we cannot reuse the old scf.condition verbatim.
static void buildNewWhileCondition(scf::WhileOp whileOp,
                                   scf::WhileOp newWhileOp) {
  auto oldCond = whileOp.getConditionOp();
  Value origCond = oldCond.getCondition();

  OpBuilder beforeBuilder(newWhileOp.getBeforeBody(),
                          newWhileOp.getBeforeBody()->end());
  SmallVector<Value> forwardedValues;
  for (BlockArgument arg : newWhileOp.getBeforeArguments()) {
    forwardedValues.push_back(arg);
  }
  beforeBuilder.create<scf::ConditionOp>(whileOp.getLoc(), origCond,
                                         forwardedValues);
  oldCond.erase();
}

// Derives the body block to inspect for shared-iter_arg analysis and the IV
// offset (1 for scf.for, 0 for scf.while). Returns false if `op` is neither.
static bool getOpIterParams(Operation *op, Block *&inspectBody,
                            unsigned &ivOffset) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    inspectBody = forOp.getBody();
    ivOffset = 1; // scf.for has the IV at block arg 0.
    return true;
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    inspectBody = whileOp.getAfterBody();
    ivOffset = 0; // scf.while has no IV.
    return true;
  }
  return false;
}

// Single entry point for processing shared iter_args in a main-loop op
// (scf.for or scf.while). Pipeline: prepare data → create new op → migrate
// body → process iteration → build terminator(s) → replace old op. The
// forOp vs whileOp branches handle the differences: scf.for has one body
// region and a single yield; scf.while has before+after regions, needs the
// new scf.condition built, and must also transfer the
// originalWhileIterArgIndices entry to the new op (so downstream passes
// keyed on the old whileOp still see it).
LogicalResult
ProcessArgsPass::processSharedIterArgsInLoop(Operation *op,
                                             ControlFlowConditionInfo *info) {
  Block *inspectBody = nullptr;
  unsigned ivOffset = 0;
  if (!getOpIterParams(op, inspectBody, ivOffset)) {
    LDBG("[Error]: op with ssbuffer.main_loop is neither scf::ForOp nor "
         "scf::WhileOp\n");
    return failure();
  }

  SmallVector<SharedArgInfo> sharedArgsInfo;
  llvm::DenseMap<int, Operation *> sharedArgToCompOp;
  llvm::DenseMap<int, llvm::DenseSet<Operation *>> sharedArgToChainOps;
  if (failed(prepareSharedArgsData(op, inspectBody, sharedArgsInfo,
                                   sharedArgToCompOp, sharedArgToChainOps))) {
    return failure();
  }
  if (sharedArgsInfo.empty()) {
    return success();
  }

  SmallVector<unsigned> origIndices;
  origIndices.reserve(sharedArgsInfo.size());
  for (const auto &info : sharedArgsInfo) {
    origIndices.push_back(info.argIndex);
  }

  Operation *newOp = createNewLoopOp(op, origIndices);
  if (!newOp) {
    return failure();
  }

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    auto newForOp = cast<scf::ForOp>(newOp);
    Block *oldBlock = forOp.getBody();
    Block *newBlock = newForOp.getBody();
    migrateBody(oldBlock, newBlock);

    SmallVector<Value> clonedResults;
    if (failed(processSharedArgsIteration(
            newBlock, sharedArgsInfo, sharedArgToCompOp, sharedArgToChainOps,
            forOp.getRegionIterArgs(), 1, clonedResults))) {
      return failure();
    }
    if (failed(buildNewYieldOp(oldBlock, newBlock, newForOp, clonedResults))) {
      return failure();
    }
    return replaceMainLoopOpAndErase(forOp, newForOp, info);
  }

  auto whileOp = cast<scf::WhileOp>(op);
  auto newWhileOp = cast<scf::WhileOp>(newOp);
  migrateBody(whileOp.getBeforeBody(), newWhileOp.getBeforeBody());
  migrateBody(whileOp.getAfterBody(), newWhileOp.getAfterBody());

  SmallVector<Value> clonedResults;
  if (failed(processSharedArgsIteration(
          newWhileOp.getAfterBody(), sharedArgsInfo, sharedArgToCompOp,
          sharedArgToChainOps, whileOp.getRegionIterArgs(), 0,
          clonedResults))) {
    return failure();
  }
  buildNewWhileCondition(whileOp, newWhileOp);
  if (failed(buildNewYieldOp(whileOp.getAfterBody(), newWhileOp.getAfterBody(),
                             newWhileOp, clonedResults))) {
    return failure();
  }
  // Transfer originalWhileIterArgIndices entry (was done by the now-removed
  // replaceMainLoopOpAndEraseAndTransferWhileIdx wrapper).
  if (originalWhileIterArgIndices.count(whileOp)) {
    originalWhileIterArgIndices[newWhileOp] =
        originalWhileIterArgIndices[whileOp];
    originalWhileIterArgIndices.erase(whileOp);
  }
  // Transfer whileBlockArgMap entry so downstream passes still see the
  // (block_id, new_arg_idx) -> old_arg_idx mapping after the old whileOp is
  // erased. Mirrors the intraCoreDependentMap transfer in
  // replaceMainLoopOpAndErase.
  if (localWhileBlockArgMap.count(whileOp)) {
    localWhileBlockArgMap[newWhileOp] =
        std::move(localWhileBlockArgMap[whileOp]);
    localWhileBlockArgMap.erase(whileOp);
  }
  if (info && info->whileBlockArgMap.count(whileOp)) {
    info->whileBlockArgMap[newWhileOp] =
        std::move(info->whileBlockArgMap[whileOp]);
    info->whileBlockArgMap.erase(whileOp);
  }
  return replaceMainLoopOpAndErase(whileOp, newWhileOp, info);
}

// Walks module to find for/while ops with ssbuffer.main_loop attribute and
// dispatches each into processSharedIterArgsInLoop.
LogicalResult ProcessArgsPass::processSharedIterArgs(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(CVPipeline::kMainLoop)) {
      return WalkResult::advance();
    }
    if (failed(processSharedIterArgsInLoop(op, info))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    return failure();
  }
  return success();
}

// ----- Per-block update chain for scf.while iter_args used in scf.condition
// -----
//
// For each scf.while op with main_loop, every block_id's run of consecutive
// ops in the do block maintains its own iter_arg value (one new arg per
// block per original iter_arg used in scf.condition). Clones the update
// chain into the end of each block_id's run, builds a new scf.while with one
// extra iter_arg per (block_id, original iter_arg), extends scf.condition
// and scf.yield, and records whileBlockArgMap[whileop][block_id][new_arg_idx]
// = old_arg_idx in info for downstream passes.

// Snapshots original iter_args of every scf.while op with main_loop, then
// processes each one: clone the update chain of every iter_arg used in
// scf.condition into the end of each block_id's run of consecutive ops in
// the do body, and add a new iter_arg per (block_id, original arg) pair.
// Recorded in info->whileBlockArgMap for downstream passes.
LogicalResult
ProcessArgsPass::updateIndependentCondsInWhileBlocks(ModuleOp module) {
  // Phase 1: snapshot original iter_args. Must run before any other
  // processing so we can later identify which iter_args were referenced by
  // scf.condition in the input.
  module.walk([&](scf::WhileOp whileOp) {
    if (!whileOp->hasAttr(CVPipeline::kMainLoop))
      return;
    SmallVector<unsigned> indices;
    for (unsigned i = 0; i < whileOp.getNumOperands(); ++i)
      indices.push_back(i);
    originalWhileIterArgIndices[whileOp] = indices;
  });

  // Phase 2: process each whileOp. Collect while ops first; we must NOT
  // mutate the IR during the walk (processWhileIterArgsInWhileOp replaces
  // the old whileOp with a new one).
  SmallVector<scf::WhileOp> worklist;
  module.walk([&](scf::WhileOp whileOp) {
    if (whileOp->hasAttr(CVPipeline::kMainLoop))
      worklist.push_back(whileOp);
  });
  for (scf::WhileOp whileOp : worklist) {
    if (failed(processWhileIterArgsInWhileOp(whileOp, info)))
      return failure();
  }

  // Dump whileBlockArgMap (new_whileop -> block_id -> (new_arg_idx ->
  // old_arg_idx)). Uses the pass-local map so it's observable even when
  // --process-args runs standalone (where `info` may be null).
  LDBG("[INFO]: whileBlockArgMap contents (new_whileop -> block_id -> "
       "(new_arg_idx -> old_arg_idx)):\n");
  for (auto &whileEntry : localWhileBlockArgMap) {
    scf::WhileOp w = whileEntry.first;
    LDBG("  whileOp @" << w.getLoc() << ":\n");
    for (auto &blockEntry : whileEntry.second) {
      int blockId = blockEntry.first;
      for (auto &argEntry : blockEntry.second) {
        int newArgIdx = argEntry.first;
        int oldArgIdx = argEntry.second;
        LDBG("    block_id=" << blockId << " new_arg_idx=" << newArgIdx
                             << " -> old_arg_idx=" << oldArgIdx << "\n");
      }
    }
  }
  return success();
}

// Returns the set of original iter_arg indices that contribute to the cond
// value of scf.condition for `whileOp`. Walks the def-chain of the cond
// value (operand 0) back to before-block BlockArguments; only those are
// "cond-used" and need per-block clones from
// updateIndependentCondsInWhileBlocks. Restricts to originalIndices;
// iter_args added by processSharedIterArgs are ignored.
//
// (Previously this iterated forwarded operands 1..N, which incorrectly
// treated every forwarded iter_arg as cond-used and cloned their update
// chains per block_id even when they had no influence on the cond
// computation. Forwarded args that don't feed the cond value do not need
// per-block copies.)
static llvm::DenseSet<unsigned> collectConditionUsedIterArgIndices(
    scf::WhileOp whileOp, const SmallVector<unsigned> &originalIndices) {
  llvm::DenseSet<unsigned> used;
  auto cond = whileOp.getConditionOp();
  Value condValue = cond.getCondition();
  Block *beforeBlock = whileOp.getBeforeBody();

  // BFS over the def-chain of condValue. A value stops being followed when
  // it's a before-block BlockArgument (collected) or when it has no defining
  // op in this region (e.g. a parent-region BlockArgument — not an iter_arg
  // of this whileOp, so ignored).
  llvm::SmallPtrSet<Value, 16> visited;
  SmallVector<Value> worklist;
  worklist.push_back(condValue);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;

    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      if (blockArg.getOwner() == beforeBlock) {
        unsigned idx = blockArg.getArgNumber();
        if (idx < originalIndices.size()) {
          used.insert(idx);
        }
      }
      continue;
    }

    Operation *defOp = v.getDefiningOp();
    if (!defOp)
      continue;
    for (Value operand : defOp->getOperands()) {
      worklist.push_back(operand);
    }
  }
  return used;
}

// (findYieldCompOpForWhile folded into findYieldDefiningOp — see top of
// file. Callers now pass `whileOp.getAfterBody()` directly.)

// Returns the last op in `body` whose `ssbuffer.block_id` matches `blockId`
// (i.e., end of the block's run of consecutive ops).
static Operation *findLastOpWithBlockId(Block *body, int blockId) {
  Operation *last = nullptr;
  for (Operation &op : body->without_terminator()) {
    auto attr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId);
    if (attr && attr.getInt() == blockId) {
      last = &op;
    }
  }
  return last;
}

// Builds a clone of the update chain after the last op with `blockId` in
// `body`. Annotates with ssbuffer.while_arg = originalArgIndex and
// ssbuffer.block_id = blockId. Remaps the original iter_arg operand to the
// new iter_arg; chain-internal operands remap to the corresponding clones.
// Returns the cloned compOp result.
static Value
cloneUpdateChainForWhileBlock(scf::WhileOp whileOp, Block *body,
                              Operation *compOp, int blockId,
                              unsigned originalArgIndex, unsigned newArgIndex,
                              const llvm::DenseSet<Operation *> &chainOps) {
  if (!compOp || chainOps.empty()) {
    return Value();
  }

  SmallVector<Operation *> sortedChain(chainOps.begin(), chainOps.end());
  if (failed(topologicalSort(sortedChain))) {
    return Value();
  }

  // Find insertion point: after the last op with this block_id.
  Operation *lastOp = findLastOpWithBlockId(body, blockId);
  OpBuilder builder(body, body->end());
  if (lastOp) {
    builder.setInsertionPointAfter(lastOp);
  }
  // If no op has this block_id, the builder is already positioned at
  // body->end() (set by the constructor), which appends before any future
  // terminator. Avoid setInsertionPoint(body->getTerminator()) since the
  // terminator may not exist yet at this point in the pipeline.

  IRMapping resultMapper;
  for (Operation *op : sortedChain) {
    IRMapping opMapper;
    for (OpOperand &operand : op->getOpOperands()) {
      Value oldVal = operand.get();
      Value newVal = oldVal;
      // Remap the original iter_arg to the new iter_arg for this block;
      // chain-internal operands remap to the corresponding cloned value.
      auto blockArg = dyn_cast<BlockArgument>(oldVal);
      if (blockArg && blockArg.getOwner() == body &&
          (unsigned)blockArg.getArgNumber() == originalArgIndex) {
        newVal = body->getArgument(newArgIndex);
      } else if (resultMapper.contains(oldVal)) {
        newVal = resultMapper.lookup(oldVal);
      }
      opMapper.map(oldVal, newVal);
    }

    if (resultMapper.contains(op->getResult(0)))
      continue;

    Operation *cloned = builder.clone(*op, opMapper);
    cloned->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(blockId));
    cloned->setAttr(CVPipeline::kWhileArg,
                    builder.getI32IntegerAttr(originalArgIndex));
    resultMapper.map(op->getResult(0), cloned->getResult(0));
    builder.setInsertionPointAfter(cloned);
  }

  return resultMapper.lookup(compOp->getResult(0));
}

// For each original iter_arg used in scf.condition, plan one new iter_arg
// per (block_id, originalArgIndex). Computes compOp + def-chain here, BEFORE
// migrateBody moves ops into the new while op's after body: Operation*
// pointers stay valid across migrateBody (moveBefore relinks rather than
// destroys), and collectChainOps must be scoped to the OLD whileOp where the
// ops currently live. Re-querying after migrateBody is unsafe (was the
// source of an IRMapping::lookup crash). `clonedPerBlock[blockId]` is
// pre-sized with placeholder Values that the clone phase fills in.
static LogicalResult planWhileIterArgDescriptors(
    scf::WhileOp whileOp, const SmallVector<unsigned> &originalIndices,
    const llvm::DenseSet<unsigned> &condUsed, WhileIterArgClonePlan &plan) {
  SmallVector<int> blockIdsInOrder = getBlockIdsInOrder(whileOp);
  if (blockIdsInOrder.empty())
    return success();
  unsigned nextNewArgIdx = whileOp.getNumOperands();
  unsigned descIdx = 0;
  for (unsigned origIdx : originalIndices) {
    if (!condUsed.contains(origIdx))
      continue;
    plan.posInClonedVec[origIdx] = descIdx++;

    Operation *compOp = findYieldDefiningOp(whileOp.getAfterBody(), origIdx);
    if (!compOp) {
      LDBG("[WARN]: no compOp for while iter_arg idx=" << origIdx);
      continue;
    }

    llvm::DenseSet<Operation *> chainOps;
    collectChainOps(whileOp, compOp, chainOps);
    plan.compOp[origIdx] = compOp;
    plan.chainOps[origIdx] = chainOps;

    for (int blockId : blockIdsInOrder) {
      plan.clonedPerBlock[blockId].push_back(Value()); // placeholder
      plan.newArgDescriptors.push_back({blockId, nextNewArgIdx++, origIdx});
    }
  }
  return success();
}

// (createNewWhileOpForIterArgs folded into createNewLoopOp — the two were
// 100% identical; createNewLoopOp now takes ArrayRef<unsigned> origIndices
// and the cond-iter-args caller pre-computes the indices from
// plan.newArgDescriptors.)

// Clones update chains into newAfter for each (block_id, original arg index)
// using newAfter's block args for remapping. Reuses the compOp/chainOps
// computed before migrateBody (the ops now live in newAfter, with operands
// already referencing newAfter's block args). Records cloned compOp results
// into plan.clonedPerBlock[blockId] at position plan.posInClonedVec[origIdx].
static LogicalResult cloneWhileBlockChains(scf::WhileOp newWhileOp,
                                           WhileIterArgClonePlan &plan) {
  Block *newAfter = newWhileOp.getAfterBody();
  // Derive (blockId, origIdx) -> newArgIdx and the ordered blockId list from
  // newArgDescriptors (first-appearance order matches getBlockIdsInOrder).
  llvm::DenseMap<std::pair<int, unsigned>, unsigned> blockOrigToNewArg;
  llvm::DenseSet<int> seenBlocks;
  SmallVector<int> blockIdsInOrder;
  for (auto &desc : plan.newArgDescriptors) {
    int blockId;
    unsigned newArgIdx, origIdx;
    std::tie(blockId, newArgIdx, origIdx) = desc;
    blockOrigToNewArg[{blockId, origIdx}] = newArgIdx;
    if (seenBlocks.insert(blockId).second)
      blockIdsInOrder.push_back(blockId);
  }

  // Iterate unique origIdx in first-appearance order across newArgDescriptors;
  // planWhileIterArgDescriptors pushes (blockId, origIdx) tuples in
  // originalIndices × blockIdsInOrder order, so first-appearance order on
  // origIdx matches the original `for origIdx : originalIndices` loop.
  llvm::DenseSet<unsigned> seenOrig;
  for (auto &desc : plan.newArgDescriptors) {
    unsigned origIdx;
    std::tie(std::ignore, std::ignore, origIdx) = desc;
    if (!seenOrig.insert(origIdx).second)
      continue;
    Operation *compOp = plan.compOp.lookup(origIdx);
    if (!compOp)
      continue;
    const llvm::DenseSet<Operation *> &chainOps = plan.chainOps.lookup(origIdx);

    for (int blockId : blockIdsInOrder) {
      unsigned newArgIdx = blockOrigToNewArg.lookup({blockId, origIdx});
      Value cloned = cloneUpdateChainForWhileBlock(
          newWhileOp, newAfter, compOp, blockId, origIdx, newArgIdx, chainOps);
      plan.clonedPerBlock[blockId][plan.posInClonedVec.lookup(origIdx)] =
          cloned;
    }
  }
  return success();
}

// Records whileBlockArgMap[newWhileOp][block_id][new_arg_idx] = old_arg_idx
// into localWhileBlockArgMap (always) and info->whileBlockArgMap (when info
// is set), so the map is observable when --process-args runs standalone.
static void recordWhileBlockArgMap(
    scf::WhileOp newWhileOp, const WhileIterArgClonePlan &plan,
    ControlFlowConditionInfo *info,
    llvm::DenseMap<scf::WhileOp, llvm::DenseMap<int, llvm::DenseMap<int, int>>>
        &localWhileBlockArgMap) {
  for (auto &desc : plan.newArgDescriptors) {
    int blockId;
    unsigned newArgIdx, origIdx;
    std::tie(blockId, newArgIdx, origIdx) = desc;
    localWhileBlockArgMap[newWhileOp][blockId][newArgIdx] = (int)origIdx;
    if (info) {
      info->whileBlockArgMap[newWhileOp][blockId][newArgIdx] = (int)origIdx;
    }
  }
}

// (createNewWhileOpForIterArgs folded into createNewLoopOp — the two were
// 100% identical; createNewLoopOp now takes ArrayRef<unsigned> origIndices
// and the cond-iter-args caller pre-computes the indices from
// plan.newArgDescriptors.)
// Per (block_id, cond-used iter_arg) pair: clone the update chain into the
// new after body, extend the scf.while, and record the new arg mapping.
LogicalResult
ProcessArgsPass::processWhileIterArgsInWhileOp(scf::WhileOp whileOp,
                                               ControlFlowConditionInfo *info) {
  auto it = originalWhileIterArgIndices.find(whileOp);
  if (it == originalWhileIterArgIndices.end())
    return success();
  const SmallVector<unsigned> &originalIndices = it->second;
  llvm::DenseSet<unsigned> condUsed =
      collectConditionUsedIterArgIndices(whileOp, originalIndices);
  if (condUsed.empty())
    return success();

  WhileIterArgClonePlan plan;
  if (failed(planWhileIterArgDescriptors(whileOp, originalIndices, condUsed,
                                         plan)) ||
      plan.newArgDescriptors.empty()) {
    return success();
  }

  // Gather origIdx from plan.newArgDescriptors in order — same shadowing rule
  // as the shared-iter-args path (one new init arg per orig iter_arg to
  // shadow), so we route through the unified createNewLoopOp.
  SmallVector<unsigned> origIndices;
  origIndices.reserve(plan.newArgDescriptors.size());
  for (auto &desc : plan.newArgDescriptors) {
    unsigned origIdx;
    std::tie(std::ignore, std::ignore, origIdx) = desc;
    origIndices.push_back(origIdx);
  }
  auto newWhileOp = cast<scf::WhileOp>(createNewLoopOp(whileOp, origIndices));
  migrateBody(whileOp.getBeforeBody(), newWhileOp.getBeforeBody());
  migrateBody(whileOp.getAfterBody(), newWhileOp.getAfterBody());
  if (failed(cloneWhileBlockChains(newWhileOp, plan))) {
    return failure();
  }
  buildNewWhileCondition(whileOp, newWhileOp);
  // Pre-compute the extra yield values from plan.clonedPerBlock in
  // newArgDescriptors order; appended to the new scf.yield operands by
  // buildNewYieldOp.
  SmallVector<Value> extraYieldValues;
  extraYieldValues.reserve(plan.newArgDescriptors.size());
  for (auto &desc : plan.newArgDescriptors) {
    int blockId;
    unsigned newArgIdx, origIdx;
    std::tie(blockId, newArgIdx, origIdx) = desc;
    extraYieldValues.push_back(plan.clonedPerBlock.lookup(
        blockId)[plan.posInClonedVec.lookup(origIdx)]);
  }
  if (failed(buildNewYieldOp(whileOp.getAfterBody(), newWhileOp.getAfterBody(),
                             newWhileOp, extraYieldValues))) {
    return failure();
  }
  recordWhileBlockArgMap(newWhileOp, plan, info, localWhileBlockArgMap);
  if (failed(replaceMainLoopOpAndErase(whileOp, newWhileOp, info))) {
    return failure();
  }
  return success();
}

void ProcessArgsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("before processArgs:\n" << module << "\n");

  // 1. While-specific decoupling: snapshot original iter_args, then for each
  //    scf.while op clone the update chain of every iter_arg used in
  //    scf.condition into the end of each block_id's run of consecutive ops
  //    in the do body, and add a new iter_arg per (block_id, original arg)
  //    pair. Recorded in info->whileBlockArgMap for downstream passes.
  if (failed(updateIndependentCondsInWhileBlocks(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // 2. Process shared iter_args (adds per-block clones for args shared
  //    across block_ids). Uses originalWhileIterArgIndices captured above.
  if (failed(processSharedIterArgs(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LDBG("after processArgs:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createProcessArgsPass() {
  return std::make_unique<ProcessArgsPass>();
}

} // namespace triton
} // namespace mlir
