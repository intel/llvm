//===- ParallelLICM.cpp - Parallel Loop Invariant Code Motion -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "parallel-licm"

using namespace mlir;
using namespace polygeist;

namespace {
struct ParallelLICM : public ParallelLICMBase<ParallelLICM> {
  void runOnOperation() override;
};
} // namespace

/// Represents the memory effects associated with an operation.
struct OperationMemoryEffects {
  OperationMemoryEffects(Operation &op) {
    if (op.getName().hasInterface<MemoryEffectOpInterface>()) {
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      cast<MemoryEffectOpInterface>(op).getEffects(effects);

      // Collect the memory effects of the operation.
      for (MemoryEffects::EffectInstance &effect : effects)
        TypeSwitch<MemoryEffects::Effect *>(effect.getEffect())
            .Case<MemoryEffects::Read>(
                [&](auto) { readResources.push_back(effect.getResource()); })
            .Case<MemoryEffects::Write>(
                [&](auto) { writeResources.push_back(effect.getResource()); })
            .Case<MemoryEffects::Free>(
                [&](auto) { freeResources.push_back(effect.getResource()); })
            .Case<MemoryEffects::Allocate>([&](auto) {
              allocateResources.push_back(effect.getResource());
            });
    }
  }

  bool readsFromMemory() const { return !readResources.empty(); }
  bool writesToMemory() const { return !writeResources.empty(); }
  bool freesMemory() const { return !freeResources.empty(); }
  bool allocatesMemory() const { return !allocateResources.empty(); }

  /// Returns true if the given operation \p other has memory effects that
  /// conflict this class memory effects, and false otherwise. For example:
  ///  case 1:
  ///   - this: read resource
  ///   - other: allocate or write the same resource
  ///  case 2:
  ///   - this: write resource
  ///   - other: allocate, read or write the same resource
  bool conflictsWith(Operation &other) {
    if (other.hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      for (Region &region : other.getRegions())
        for (Operation &innerOp : region.getOps())
          if (conflictsWith(innerOp))
            return true;
      return false;
    }

    if (!other.getName().hasInterface<MemoryEffectOpInterface>())
      return true;

    // The given operation conflicts with the memory effect of this operation if
    // it allocates or writes a resource that is read by this operation.
    for (SideEffects::Resource *res : readResources) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      cast<MemoryEffectOpInterface>(other).getEffectsOnResource(res, effects);
      for (const MemoryEffects::EffectInstance &effect : effects) {
        if (isa<MemoryEffects::Allocate>(effect.getEffect()) ||
            isa<MemoryEffects::Write>(effect.getEffect()))
          return true;
      }
    }

    // The given operation conflicts with the memory effect of this operation if
    // it allocates or reads a resource that is written by this operation.
    for (SideEffects::Resource *res : writeResources) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      cast<MemoryEffectOpInterface>(other).getEffectsOnResource(res, effects);
      for (const MemoryEffects::EffectInstance &effect : effects) {
        if (isa<MemoryEffects::Allocate>(effect.getEffect()) ||
            isa<MemoryEffects::Read>(effect.getEffect()) ||
            isa<MemoryEffects::Write>(effect.getEffect()))
          return true;
      }
    }

    // The given operation conflicts with the memory effect of this operation if
    // it allocates, writes or reads a resource that is written by freed by this
    // operation.
    for (SideEffects::Resource *res : freeResources) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      cast<MemoryEffectOpInterface>(other).getEffectsOnResource(res, effects);
      for (const MemoryEffects::EffectInstance &effect : effects) {
        if (isa<MemoryEffects::Allocate>(effect.getEffect()))
          return true;
        if (isa<MemoryEffects::Write>(effect.getEffect()))
          return true;
        if (isa<MemoryEffects::Read>(effect.getEffect()))
          return true;
      }
    }

    return false;
  }

  const SmallVector<SideEffects::Resource *> &getReadResources() const {
    return readResources;
  };
  const SmallVector<SideEffects::Resource *> &getWriteResources() const {
    return writeResources;
  };
  const SmallVector<SideEffects::Resource *> &getFreeResources() const {
    return freeResources;
  };
  const SmallVector<SideEffects::Resource *> &getAllocateResources() const {
    return allocateResources;
  };

private:
  SmallVector<SideEffects::Resource *> readResources;
  SmallVector<SideEffects::Resource *> writeResources;
  SmallVector<SideEffects::Resource *> freeResources;
  SmallVector<SideEffects::Resource *> allocateResources;
};

/// Returns true if the Operation \p op can be hoisted out of the given loop \p
/// loop. The \p willBeMoved argument represents operations that are known to be
/// loop invariant (and therefore will be moved outside of the loop).
static bool canBeHoisted(Operation &op, LoopLikeOpInterface loop,
                         const SmallPtrSetImpl<Operation *> &willBeMoved) {
  // Returns true if the given value can be moved outside of the loop, and false
  // otherwise. A value can be moved outside of the loop if its operand are
  // defined by operations that can themselves be moved, or are already outside
  // of the loop.
  auto canBeMoved = [&](Value value) {
    if (auto BA = value.dyn_cast<BlockArgument>())
      if (willBeMoved.count(BA.getOwner()->getParentOp()))
        return true;
    Operation *definingOp = value.getDefiningOp();
    if ((definingOp && willBeMoved.count(definingOp)) ||
        loop.isDefinedOutsideOfLoop(value))
      return true;
    return false;
  };

  // The operation cannot be hoisted is any of its operands cannot be moved
  // outside of the loop.
  if (llvm::any_of(op.getOperands(),
                   [&](Value value) { return !canBeMoved(value); }))
    return false;

  // If the operation allocates memory in the loop it cannot be hoisted.
  OperationMemoryEffects memoryEffects(op);
  if (memoryEffects.allocatesMemory())
    return false;

  // This lambda function determines whether the given operation has memory
  // effects that 'conflict' with other operations in the same loop.
  std::function<bool(Operation &, LoopLikeOpInterface)> hasConflictsInLoop =
      [&](Operation &op, LoopLikeOpInterface loop) {
        // Check for conflicts with other previous operations in the same block.
        for (Operation *it = op.getPrevNode();
             it != nullptr && !willBeMoved.count(it); it = it->getPrevNode()) {
          if (memoryEffects.conflictsWith(op))
            return true;
        }

        // Check for conflicts with the parent operation.
        if (op.getParentOp() == loop)
          return false;
        if (hasConflictsInLoop(*op.getParentOp(), loop))
          return true;

        // If the parent operation is not guaranteed to execute its
        // (single-block) region once, walk the block.
        bool conflict = false;
        if (!isa<scf::IfOp, AffineIfOp, memref::AllocaScopeOp>(op)) {
          op.walk([&](Operation *in) {
            if (!willBeMoved.count(in) && memoryEffects.conflictsWith(*in)) {
              conflict = true;
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
        }

        return conflict;
      };

  if ((memoryEffects.readsFromMemory() || memoryEffects.writesToMemory() ||
       memoryEffects.freesMemory()) &&
      hasConflictsInLoop(op, loop))
    return false;

  // Recurse into the regions for this op and check whether the contained ops
  // can be hoisted. We can inductively assume that this op will have its block
  // args available outside the loop
  SmallPtrSet<Operation *, 2> willBeMoved2(willBeMoved.begin(),
                                           willBeMoved.end());
  willBeMoved2.insert(&op);

  for (Region &region : op.getRegions()) {
    for (Operation &innerOp : region.getOps()) {
      if (!canBeHoisted(innerOp, loop, willBeMoved2))
        return false;
      willBeMoved2.insert(&innerOp);
    }
  }

  return true;
}

void moveParallelLoopInvariantCode(scf::ParallelOp loopLike) {
  Region &loopBody = loopLike.getLoopBody();

  // We use two collections here as we need to preserve the order for insertion
  // and this is easiest.
  SmallPtrSet<Operation *, 8> willBeMovedSet;
  SmallVector<Operation *, 8> opsToMove;

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.
  for (Block &block : loopBody) {
    for (Operation &op : block.without_terminator()) {
      if (!canBeHoisted(op, loopLike, willBeMovedSet))
        continue;

      //      opsToMove.push_back(&op);
      willBeMovedSet.insert(&op);
    }
  }

  // Move all operations we found to be invariant outside of the loop.
  if (!opsToMove.empty()) {
    OpBuilder b(loopLike);
    Value cond = nullptr;
    for (auto pair : llvm::zip(loopLike.getLowerBound(),
                               loopLike.getUpperBound(), loopLike.getStep())) {
      auto val = b.create<arith::CmpIOp>(
          loopLike.getLoc(), arith::CmpIPredicate::sle,
          b.create<arith::AddIOp>(loopLike.getLoc(), std::get<0>(pair),
                                  std::get<2>(pair)),
          std::get<1>(pair));
      if (cond == nullptr)
        cond = val;
      else
        cond = b.create<arith::AndIOp>(loopLike.getLoc(), cond, val);
    }

    auto ifOp = b.create<scf::IfOp>(loopLike.getLoc(), TypeRange(), cond);
    loopLike->moveBefore(ifOp.thenYield());

    for (auto *op : opsToMove)
      loopLike.moveOutOfLoop(op);
  }

  LLVM_DEBUG(loopLike.print(llvm::dbgs() << "\n\nModified loop:\n"));
}

// TODO affine parallel licm
void moveParallelLoopInvariantCode(AffineParallelOp loopLike) {
  auto &loopBody = loopLike.getLoopBody();

  // We use two collections here as we need to preserve the order for insertion
  // and this is easiest.
  SmallPtrSet<Operation *, 8> willBeMoved;
  SmallVector<Operation *, 8> opsToMove;

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.
  for (auto &block : loopBody) {
    for (auto &op : block.without_terminator()) {
      if (canBeHoisted(op, loopLike, willBeMoved)) {
        opsToMove.push_back(&op);
        willBeMoved.insert(&op);
      }
    }
  }

  // For all instructions that we found to be invariant, move outside of the
  // loop.
  if (opsToMove.size()) {
    OpBuilder b(loopLike);

    // TODO properly fill exprs and eqflags
    SmallVector<AffineExpr, 2> exprs;
    SmallVector<bool, 2> eqflags;

    for (auto step : llvm::enumerate(loopLike.getSteps())) {
      for (auto ub : loopLike.getUpperBoundMap(step.index()).getResults()) {
        SmallVector<AffineExpr, 4> symbols;
        for (unsigned idx = 0;
             idx < loopLike.getUpperBoundsMap().getNumSymbols(); ++idx)
          symbols.push_back(getAffineSymbolExpr(
              idx + loopLike.getLowerBoundsMap().getNumSymbols(),
              loopLike.getContext()));

        SmallVector<AffineExpr, 4> dims;
        for (unsigned idx = 0; idx < loopLike.getUpperBoundsMap().getNumDims();
             ++idx)
          dims.push_back(
              getAffineDimExpr(idx + loopLike.getLowerBoundsMap().getNumDims(),
                               loopLike.getContext()));

        ub = ub.replaceDimsAndSymbols(dims, symbols);

        for (auto lb : loopLike.getLowerBoundMap(step.index()).getResults()) {

          // Bound is whether this expr >= 0, which since we want ub > lb, we
          // rewrite as follows.
          exprs.push_back(ub - lb - step.value());
          eqflags.push_back(false);
        }
      }
    }

    SmallVector<Value> values;
    auto lb_ops = loopLike.getLowerBoundsOperands();
    auto ub_ops = loopLike.getUpperBoundsOperands();
    for (unsigned idx = 0; idx < loopLike.getLowerBoundsMap().getNumDims();
         ++idx)
      values.push_back(lb_ops[idx]);
    for (unsigned idx = 0; idx < loopLike.getUpperBoundsMap().getNumDims();
         ++idx)
      values.push_back(ub_ops[idx]);
    for (unsigned idx = 0; idx < loopLike.getLowerBoundsMap().getNumSymbols();
         ++idx)
      values.push_back(lb_ops[idx + loopLike.getLowerBoundsMap().getNumDims()]);
    for (unsigned idx = 0; idx < loopLike.getUpperBoundsMap().getNumSymbols();
         ++idx)
      values.push_back(ub_ops[idx + loopLike.getUpperBoundsMap().getNumDims()]);

    auto iset = IntegerSet::get(
        /*dim*/ loopLike.getLowerBoundsMap().getNumDims() +
            loopLike.getUpperBoundsMap().getNumDims(),
        /*symbols*/ loopLike.getLowerBoundsMap().getNumSymbols() +
            loopLike.getUpperBoundsMap().getNumSymbols(),
        exprs, eqflags);
    auto ifOp = b.create<AffineIfOp>(loopLike.getLoc(), TypeRange(), iset,
                                     values, /*else*/ false);
    loopLike->moveBefore(ifOp.getThenBlock()->getTerminator());
  }
  for (auto *op : opsToMove)
    loopLike.moveOutOfLoop(op);
  LLVM_DEBUG(loopLike.print(llvm::dbgs() << "\n\nModified loop:\n"));
}

void ParallelLICM::runOnOperation() {
  getOperation()->walk([&](LoopLikeOpInterface loopLike) {
    LLVM_DEBUG(loopLike.print(llvm::dbgs() << "\nOriginal loop:\n"));
    moveLoopInvariantCode(loopLike);
    if (auto par = dyn_cast<scf::ParallelOp>((Operation *)loopLike))
      moveParallelLoopInvariantCode(par);
    else if (auto par = dyn_cast<AffineParallelOp>((Operation *)loopLike))
      moveParallelLoopInvariantCode(par);
  });
}

std::unique_ptr<Pass> mlir::polygeist::createParallelLICMPass() {
  return std::make_unique<ParallelLICM>();
}
