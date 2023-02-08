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
    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffects(effects);

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
  /// conflict with the memory effects summarized in this class, and false
  /// otherwise.
  bool conflictsWith(Operation &other) const {
    // Check all the nested operations if 'other' has recursive side effects.
    bool hasRecursiveEffects =
        other.hasTrait<OpTrait::HasRecursiveMemoryEffects>();
    if (hasRecursiveEffects) {
      for (Region &region : other.getRegions())
        for (Operation &innerOp : region.getOps())
          if (conflictsWith(innerOp))
            return true;
      return false;
    }

    // If the given operation has side effects, characterize them and check
    // whether they might prevent hoisting.
    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(other)) {
      // Check whether the given operation writes or allocates a resource read
      // by the operation associated with this class.
      for (SideEffects::Resource *res : readResources) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memEffect.getEffectsOnResource(res, effects);
        for (const MemoryEffects::EffectInstance &effect : effects) {
          if (isa<MemoryEffects::Allocate, MemoryEffects::Write>(
                  effect.getEffect()))
            return true;
        }
      }

      // Check whether the given operation read. writes or allocates a resources
      // that is written by the operation associated with this class.
      for (SideEffects::Resource *res : writeResources) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memEffect.getEffectsOnResource(res, effects);
        for (const MemoryEffects::EffectInstance &effect : effects) {
          if (isa<MemoryEffects::Allocate, MemoryEffects::Read,
                  MemoryEffects::Write>(effect.getEffect()))
            return true;
        }
      }

      // Check whether the given operation read, writes or allocates a resources
      // that is freed by the operation associated with this class.
      for (SideEffects::Resource *res : freeResources) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memEffect.getEffectsOnResource(res, effects);
        for (const MemoryEffects::EffectInstance &effect : effects) {
          if (isa<MemoryEffects::Allocate, MemoryEffects::Write,
                  MemoryEffects::Read>(effect.getEffect()))
            return true;
        }
      }
    }

    return false;
  }

  const ArrayRef<SideEffects::Resource *> getReadResources() const {
    return readResources;
  };
  const ArrayRef<SideEffects::Resource *> getWriteResources() const {
    return writeResources;
  };
  const ArrayRef<SideEffects::Resource *> getFreeResources() const {
    return freeResources;
  };
  const ArrayRef<SideEffects::Resource *> getAllocateResources() const {
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
template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                          T, scf::ParallelOp, AffineParallelOp>::value>>
static bool canBeHoisted(Operation &op, T loop,
                         const SmallPtrSetImpl<Operation *> &willBeMoved) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing: " << op << "\n");

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

  // Ensure operands can be hoisted.
  if (llvm::any_of(op.getOperands(),
                   [&](Value value) { return !canBeMoved(value); })) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "cannot be hoisted: operand(s) can't be hoisted\n");
    return false;
  }

  // If the operation has no side effects it can be hoisted.
  if (isMemoryEffectFree(&op)) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "can be hoisted: operation has no side effects\n");
    return true;
  }

  // Operations with unknown side effects cannot be hoisted.
  if (!isa<MemoryEffectOpInterface>(op) &&
      !op.hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "cannot be hoisted: unknown side effects\n");
    return false;
  }

  // Do not hoist operations that allocate memory.
  const OperationMemoryEffects memoryEffects(op);
  if (memoryEffects.allocatesMemory()) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "cannot be hoisted: operation allocates memory\n");
    return false;
  }

  // Determine whether the given operation has memory effects that 'conflict'
  // with other operations in the same loop.
  std::function<bool(Operation &, LoopLikeOpInterface)> hasConflictsInLoop =
      [&](Operation &op, LoopLikeOpInterface loop) {
        // Check for conflicts with other previous operations in the same block.
        for (Operation *it = op.getPrevNode();
             it != nullptr && !willBeMoved.count(it); it = it->getPrevNode()) {
          if (memoryEffects.conflictsWith(*it)) {
            LLVM_DEBUG(llvm::dbgs().indent(2)
                       << "conflicting operation: " << *it << "\n");
            return true;
          }
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

  // If the operation has side effects, check whether other operation in the
  // loop prevent hosting it.
  if ((memoryEffects.readsFromMemory() || memoryEffects.writesToMemory() ||
       memoryEffects.freesMemory()) &&
      hasConflictsInLoop(op, loop)) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "cannot be hoisted: found conflicting operation\n");
    return false;
  }

  // Recurse into the regions for this op and check whether the contained ops
  // can be hoisted. We can inductively assume that this op will have its block
  // args available outside the loop.
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

  LLVM_DEBUG(llvm::dbgs().indent(2) << "can be hoisted: no conflicts found\n");

  return true;
}

// Populate \p opsToMove with operations that can be hoisted out of the given
// loop \p loop.
template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                          T, scf::ParallelOp, AffineParallelOp>::value>>
static void collectHoistableOperation(T loop,
                                      SmallVectorImpl<Operation *> &opsToMove) {
  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there. These regions might have semantics unknown to this
  // rewriting. If the nested regions are loops, they will have been processed.
  SmallPtrSet<Operation *, 8> willBeMoved;
  for (Block &block : loop.getLoopBody()) {
    for (Operation &op : block.without_terminator()) {
      if (!canBeHoisted(op, loop, willBeMoved))
        continue;
      opsToMove.push_back(&op);
      willBeMoved.insert(&op);
    }
  }
}

void moveParallelLoopInvariantCode(scf::ParallelOp loop) {
  SmallVector<Operation *, 8> opsToMove;
  collectHoistableOperation(loop, opsToMove);
  if (opsToMove.empty())
    return;

  // Move all operations we found to be invariant outside of the loop.
  OpBuilder b(loop);
  Value cond;
  for (auto pair :
       llvm::zip(loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
    const Value val = b.create<arith::CmpIOp>(
        loop.getLoc(), arith::CmpIPredicate::sle,
        b.create<arith::AddIOp>(loop.getLoc(), std::get<0>(pair),
                                std::get<2>(pair)),
        std::get<1>(pair));
    cond = cond ? static_cast<Value>(
                      b.create<arith::AndIOp>(loop.getLoc(), cond, val))
                : val;
  }

  auto ifOp = b.create<scf::IfOp>(loop.getLoc(), TypeRange(), cond);
  loop->moveBefore(ifOp.thenYield());

  for (Operation *op : opsToMove)
    loop.moveOutOfLoop(op);

  LLVM_DEBUG({
    loop.print(llvm::dbgs() << "\nModified loop:\n");
    llvm::dbgs() << "\n----------------\n";
  });
}

void moveParallelLoopInvariantCode(AffineParallelOp loop) {
  SmallVector<Operation *, 8> opsToMove;
  collectHoistableOperation(loop, opsToMove);
  if (opsToMove.empty())
    return;

  // Move all operations we found to be invariant outside of the loop.
  OpBuilder b(loop);

  // TODO properly fill exprs and eqflags
  SmallVector<AffineExpr, 2> exprs;
  SmallVector<bool, 2> eqflags;

  for (auto step : llvm::enumerate(loop.getSteps())) {
    for (AffineExpr ub : loop.getUpperBoundMap(step.index()).getResults()) {
      SmallVector<AffineExpr, 4> symbols;
      for (unsigned idx = 0; idx < loop.getUpperBoundsMap().getNumSymbols();
           ++idx)
        symbols.push_back(getAffineSymbolExpr(
            idx + loop.getLowerBoundsMap().getNumSymbols(), loop.getContext()));

      SmallVector<AffineExpr, 4> dims;
      for (unsigned idx = 0; idx < loop.getUpperBoundsMap().getNumDims(); ++idx)
        dims.push_back(getAffineDimExpr(
            idx + loop.getLowerBoundsMap().getNumDims(), loop.getContext()));

      ub = ub.replaceDimsAndSymbols(dims, symbols);

      for (AffineExpr lb : loop.getLowerBoundMap(step.index()).getResults()) {
        // Bound is whether this expr >= 0, which since we want ub > lb, we
        // rewrite as follows.
        exprs.push_back(ub - lb - step.value());
        eqflags.push_back(false);
      }
    }
  }

  SmallVector<Value> values;
  OperandRange lb_ops = loop.getLowerBoundsOperands(),
               ub_ops = loop.getUpperBoundsOperands();

  std::copy(lb_ops.begin(),
            lb_ops.begin() + loop.getLowerBoundsMap().getNumDims(),
            std::back_inserter(values));
  std::copy(ub_ops.begin(),
            ub_ops.begin() + loop.getUpperBoundsMap().getNumDims(),
            op std::back_inserter(values));
  std::copy(lb_ops.begin() + loop.getLowerBoundsMap().getNumDims(),
            lb_ops.end(), std::back_inserter(values));
  std::copy(ub_ops.begin() + loop.getUpperBoundsMap().getNumDims(),
            ub_ops.end(), std::back_inserter(values));

  auto iset = IntegerSet::get(
      /*dim*/ loop.getLowerBoundsMap().getNumDims() +
          loop.getUpperBoundsMap().getNumDims(),
      /*symbols*/ loop.getLowerBoundsMap().getNumSymbols() +
          loop.getUpperBoundsMap().getNumSymbols(),
      exprs, eqflags);
  auto ifOp = b.create<AffineIfOp>(loop.getLoc(), TypeRange(), iset, values,
                                   /*else*/ false);
  loop->moveBefore(ifOp.getThenBlock()->getTerminator());

  for (Operation *op : opsToMove)
    loop.moveOutOfLoop(op);

  LLVM_DEBUG(loop.print(llvm::dbgs() << "\n\nModified loop:\n"));
}

void ParallelLICM::runOnOperation() {
  getOperation()->walk([&](LoopLikeOpInterface loop) {
    LLVM_DEBUG({
      llvm::dbgs() << "----------------\n";
      loop.print(llvm::dbgs() << "Original loop:\n");
      llvm::dbgs() << "\n";
    });

    moveLoopInvariantCode(loop);
    TypeSwitch<Operation *>((Operation *)loop)
        .Case<scf::ParallelOp>(
            [&](auto loop) { moveParallelLoopInvariantCode(loop); })
        .Case<AffineParallelOp>(
            [&](auto loop) { moveParallelLoopInvariantCode(loop); });
  });
}

std::unique_ptr<Pass> mlir::polygeist::createParallelLICMPass() {
  return std::make_unique<ParallelLICM>();
}
