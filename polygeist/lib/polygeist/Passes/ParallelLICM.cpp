//===- ParallelLICM.cpp - Parallel Loop Invariant Code Motion -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
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

/// Represents the side effects associated with an operation.
class OperationSideEffects {
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const OperationSideEffects &);

public:
  OperationSideEffects(const Operation &op, const AliasAnalysis &aliasAnalysis)
      : op(op), aliasAnalysis(aliasAnalysis) {
    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffects(effects);

      // Classify the side effects of the operation.
      for (MemoryEffects::EffectInstance EI : effects) {
        TypeSwitch<const MemoryEffects::Effect *>(EI.getEffect())
            .Case<MemoryEffects::Read>(
                [&](auto) { readResources.push_back(EI); })
            .Case<MemoryEffects::Write>(
                [&](auto) { writeResources.push_back(EI); })
            .Case<MemoryEffects::Free>(
                [&](auto) { freeResources.push_back(EI); })
            .Case<MemoryEffects::Allocate>(
                [&](auto) { allocateResources.push_back(EI); });
      }
    }
  }

  const Operation &getOperation() const { return op; };

  bool readsFromResource() const { return !readResources.empty(); }
  bool writesToResource() const { return !writeResources.empty(); }
  bool freesResource() const { return !freeResources.empty(); }
  bool allocatesResource() const { return !allocateResources.empty(); }

  /// Collects the side effects the symbol associated with this class has on the
  /// given resource \p res. The results is stored into \p effects.
  void getEffectsOnResource(
      SideEffects::Resource *res,
      SmallVectorImpl<MemoryEffects::EffectInstance> &effects) const {
    assert(res && "Expecting a valid resource");
    assert(effects.empty() && "Expecting an empty vector");
    cast<MemoryEffectOpInterface>(op).getEffectsOnResource(res, effects);
  }

  /// Returns true if the given operation \p other has side effects that
  /// conflict with the side effects summarized in this class, and false
  /// otherwise.
  bool conflictsWith(const Operation &other) const;

  /// Returns the first operation in the given block \p block with side effects
  /// that conflict with the side effects summarized in this class, and a
  /// std::nullopt if none is found. If \p point is given, only operations that
  /// appear before it are considered.
  Optional<Operation *>
  conflictsWithOperationInBlock(Block &block, Operation *point = nullptr) const;

  /// Returns the first operation in the given region \p rgn with side effects
  /// that conflict with the side effects summarized in this class, and a
  /// std::nullopt if none is found.
  Optional<Operation *> conflictsWithOperationInRegion(Region &rgn) const;

  /// Returns the first operation in the given loop \p loop with side effects
  /// that conflict with the side effects summarized in this class, and a
  /// std::nullopt if none is found.
  Optional<Operation *>
  conflictsWithOperationInLoop(LoopLikeOpInterface loop) const;

private:
  const Operation &op; /// Operation associated with the side effects.
  const AliasAnalysis &aliasAnalysis; /// Alias Analysis reference.
  /// Side effects associated with reading resources.
  SmallVector<MemoryEffects::EffectInstance> readResources;
  /// Side effects associated with writing resources.
  SmallVector<MemoryEffects::EffectInstance> writeResources;
  /// Side effects associated with freeing resources.
  SmallVector<MemoryEffects::EffectInstance> freeResources;
  /// Side effects associated with allocating resources.
  SmallVector<MemoryEffects::EffectInstance> allocateResources;
};

[[maybe_unused]] inline llvm::raw_ostream &
operator<<(llvm::raw_ostream &OS, const OperationSideEffects &ME) {
  auto printResources = [&OS](
                            StringRef title,
                            ArrayRef<MemoryEffects::EffectInstance> resources) {
    auto delimit = [](raw_ostream &OS, bool &isFirst) {
      if (!isFirst)
        OS << ", ";
      isFirst = false;
    };

    OS.indent(2) << title << ": ";
    bool isFirst = true;
    for (const MemoryEffects::EffectInstance &EI : resources) {
      delimit(OS, isFirst);
      OS << "{" << EI.getResource()->getName() << ", " << EI.getValue() << "}";
    }
    OS << "\n";
  };

  bool isSideEffectFree = !ME.readsFromResource() && !ME.writesToResource() &&
                          !ME.freesResource() && !ME.allocatesResource();

  OS << "Operation: " << ME.getOperation() << "\n";
  if (isSideEffectFree)
    OS.indent(2) << "is side effects free.\n";
  else {
    if (ME.readsFromResource())
      printResources("read resources", ME.readResources);
    if (ME.writesToResource())
      printResources("write resources", ME.writeResources);
    if (ME.freesResource())
      printResources("free resources", ME.freeResources);
    if (ME.allocatesResource())
      printResources("allocate resources", ME.allocateResources);
  }

  return OS;
}

class LoopGuardBuilder {
public:
  static std::unique_ptr<LoopGuardBuilder> create(LoopLikeOpInterface loop);

  LoopGuardBuilder(LoopLikeOpInterface loop) : builder(loop){};
  LoopGuardBuilder(const LoopGuardBuilder &) = delete;
  LoopGuardBuilder(LoopGuardBuilder &&) = delete;
  void operator=(const LoopGuardBuilder &) = delete;
  void operator=(LoopGuardBuilder &&) = delete;
  virtual ~LoopGuardBuilder() = default;

  virtual void guardLoop() const = 0;

protected:
  mutable OpBuilder builder;
};

class SCFForGuardBuilder : public LoopGuardBuilder {
public:
  SCFForGuardBuilder(scf::ForOp loop) : LoopGuardBuilder(loop), loop(loop) {}

  void guardLoop() const final;

private:
  Value createGuardExpr() const;
  scf::IfOp createGuard() const;
  void replaceUsesOfLoopReturnValues(scf::IfOp &) const;

  mutable scf::ForOp loop;
};

class SCFParallelGuardBuilder : public LoopGuardBuilder {
public:
  SCFParallelGuardBuilder(scf::ParallelOp loop)
      : LoopGuardBuilder(loop), loop(loop) {}

  void guardLoop() const final;

private:
  Value createGuardExpr() const;
  scf::IfOp createGuard() const;
  void replaceUsesOfLoopReturnValues(scf::IfOp &) const;

  mutable scf::ParallelOp loop;
};

class AffineForGuardBuilder : public LoopGuardBuilder {
public:
  AffineForGuardBuilder(AffineForOp loop)
      : LoopGuardBuilder(loop), loop(loop) {}

  void guardLoop() const final;

private:
  IntegerSet createGuardExpr() const;
  AffineIfOp createGuard() const;
  void replaceUsesOfLoopReturnValues(AffineIfOp &) const;

  mutable AffineForOp loop;
};

class AffineParallelGuardBuilder : public LoopGuardBuilder {
public:
  AffineParallelGuardBuilder(AffineParallelOp loop)
      : LoopGuardBuilder(loop), loop(loop) {}

  void guardLoop() const final;

private:
  IntegerSet createGuardExpr() const;
  AffineIfOp createGuard() const;
  void replaceUsesOfLoopReturnValues(AffineIfOp &) const;

  mutable AffineParallelOp loop;
};

} // namespace

//===----------------------------------------------------------------------===//
// OperationSideEffects
//===----------------------------------------------------------------------===//

bool OperationSideEffects::conflictsWith(const Operation &other) const {
  if (&op == &other)
    return false;

  // Check all the nested operations if 'other' has recursive side effects.
  bool hasRecursiveEffects =
      const_cast<Operation &>(other)
          .hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (hasRecursiveEffects) {
    for (Region &region : const_cast<Operation &>(other).getRegions())
      for (Operation &innerOp : region.getOps())
        if (conflictsWith(innerOp))
          return true;
    return false;
  }

  // If the given operation has side effects, check whether they conflict with
  // the side effects summarized in this class.
  if (auto MEI = dyn_cast<MemoryEffectOpInterface>(other)) {
    OperationSideEffects sideEffects(other, aliasAnalysis);

    // Checks for a conflicts on the given resource 'res' by applying the
    // supplied predicate function 'hasConflict'.
    auto checkForConflict =
        [&sideEffects](
            SideEffects::Resource *res,
            std::function<bool(const MemoryEffects::EffectInstance &EI)>
                hasConflict) {
          SmallVector<MemoryEffects::EffectInstance> effects;
          sideEffects.getEffectsOnResource(res, effects);
          return llvm::any_of(
              effects, [hasConflict](const MemoryEffects::EffectInstance &EI) {
                return hasConflict(EI);
              });
        };

    [[maybe_unused]] auto printConflictingSideEffects =
        [](const MemoryEffects::EffectInstance &EI, AliasResult aliasRes) {
          llvm::dbgs() << "Found conflicting side effect: {"
                       << EI.getResource()->getName() << ", " << EI.getValue()
                       << "}\n";
          llvm::dbgs().indent(2) << "aliasResult: " << aliasRes << "\n";
        };

    // Check whether the given operation 'other' writes (or allocates, or frees)
    // a resource that is read by the operation associated with this class.
    if (llvm::any_of(
            readResources, [&](const MemoryEffects::EffectInstance &readRes) {
              auto hasConflict = [&](const MemoryEffects::EffectInstance &EI) {
                if (isa<MemoryEffects::Read>(EI.getEffect()))
                  return false;

                AliasResult aliasRes =
                    const_cast<AliasAnalysis &>(aliasAnalysis)
                        .alias(EI.getValue(), readRes.getValue());
                if (aliasRes.isNo())
                  return false;

                LLVM_DEBUG(printConflictingSideEffects(EI, aliasRes));
                return true;
              };

              return checkForConflict(readRes.getResource(), hasConflict);
            })) {
      return true;
    }

    // Check whether the given operation 'other' allocates, reads, writes or
    // frees a resource that is written by the operation associated with this
    // class.
    if (llvm::any_of(
            writeResources, [&](const MemoryEffects::EffectInstance &writeRes) {
              auto hasConflict = [&](const MemoryEffects::EffectInstance &EI) {
                AliasResult aliasRes =
                    const_cast<AliasAnalysis &>(aliasAnalysis)
                        .alias(EI.getValue(), writeRes.getValue());
                if (aliasRes.isNo())
                  return false;

                LLVM_DEBUG(printConflictingSideEffects(EI, aliasRes));
                return true;
              };

              return checkForConflict(writeRes.getResource(), hasConflict);
            })) {
      return true;
    }

    // Check whether the given operation 'other'  allocates, reads, writes or
    // frees a resource that is freed by the operation associated with this
    // class.
    if (llvm::any_of(
            freeResources, [&](const MemoryEffects::EffectInstance &freeRes) {
              auto hasConflict = [&](const MemoryEffects::EffectInstance &EI) {
                AliasResult aliasRes =
                    const_cast<AliasAnalysis &>(aliasAnalysis)
                        .alias(EI.getValue(), freeRes.getValue());
                if (aliasRes.isNo())
                  return false;

                LLVM_DEBUG(printConflictingSideEffects(EI, aliasRes));
                return true;
              };

              return checkForConflict(freeRes.getResource(), hasConflict);
            })) {
      return true;
    }
  }

  return false;
}

Optional<Operation *>
OperationSideEffects::conflictsWithOperationInBlock(Block &block,
                                                    Operation *point) const {
  for (Operation &other : block) {
    if (point && !other.isBeforeInBlock(point))
      break;
    if (conflictsWith(other))
      return &other;
  }
  return std::nullopt;
}

Optional<Operation *>
OperationSideEffects::conflictsWithOperationInRegion(Region &rgn) const {
  for (Block &block : rgn) {
    Optional<Operation *> conflictingOp = conflictsWithOperationInBlock(block);
    if (conflictingOp.has_value())
      return conflictingOp.value();
  }
  return std::nullopt;
}

Optional<Operation *> OperationSideEffects::conflictsWithOperationInLoop(
    LoopLikeOpInterface loop) const {
  return conflictsWithOperationInRegion(loop.getLoopBody());
}

//===----------------------------------------------------------------------===//
// LoopGuardBuilder
//===----------------------------------------------------------------------===//

std::unique_ptr<LoopGuardBuilder>
LoopGuardBuilder::create(LoopLikeOpInterface loop) {
  return TypeSwitch<Operation *, std::unique_ptr<LoopGuardBuilder>>(
             (Operation *)loop)
      .Case<scf::ForOp>(
          [](auto loop) { return std::make_unique<SCFForGuardBuilder>(loop); })
      .Case<scf::ParallelOp>([](auto loop) {
        return std::make_unique<SCFParallelGuardBuilder>(loop);
      })
      .Case<AffineForOp>([](auto loop) {
        return std::make_unique<AffineForGuardBuilder>(loop);
      })
      .Case<AffineParallelOp>([](auto loop) {
        return std::make_unique<AffineParallelGuardBuilder>(loop);
      })
      .Default([](auto) { return nullptr; });
}

//===----------------------------------------------------------------------===//
// SCFForLoopGuardBuilder
//===----------------------------------------------------------------------===//

void SCFForGuardBuilder::guardLoop() const {
  scf::IfOp ifOp = createGuard();
  loop->moveBefore(ifOp.thenYield());
  replaceUsesOfLoopReturnValues(ifOp);

  bool yieldsResults = !loop->getResults().empty();
  if (!yieldsResults)
    ifOp.elseBlock()->erase();
}

Value SCFForGuardBuilder::createGuardExpr() const {
  return builder.create<arith::CmpIOp>(
      loop.getLoc(), arith::CmpIPredicate::sle,
      builder.create<arith::AddIOp>(loop.getLoc(), loop.getLowerBound(),
                                    loop.getStep()),
      loop.getUpperBound());
}

scf::IfOp SCFForGuardBuilder::createGuard() const {
  TypeRange types(loop->getResults());
  return builder.create<scf::IfOp>(
      loop.getLoc(), types, createGuardExpr(),
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, loop.getResults());
      },
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, loop.getInitArgs());
      });
}

void SCFForGuardBuilder::replaceUsesOfLoopReturnValues(scf::IfOp &ifOp) const {
  // Replace uses of the loop return value(s) with the value(s) yielded by the
  // if operation.
  for (auto it : llvm::zip(loop.getResults(), ifOp.getResults()))
    std::get<0>(it).replaceUsesWithIf(std::get<1>(it), [&](OpOperand &op) {
      Block *useBlock = op.getOwner()->getBlock();
      return useBlock != ifOp.thenBlock();
    });
}

//===----------------------------------------------------------------------===//
// SCFParallelLoopGuardBuilder
//===----------------------------------------------------------------------===//

void SCFParallelGuardBuilder::guardLoop() const {
  scf::IfOp ifOp = createGuard();
  loop->moveBefore(ifOp.thenYield());
  replaceUsesOfLoopReturnValues(ifOp);
}

Value SCFParallelGuardBuilder::createGuardExpr() const {
  Value cond;
  for (auto pair :
       llvm::zip(loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
    const Value val = builder.create<arith::CmpIOp>(
        loop.getLoc(), arith::CmpIPredicate::sle,
        builder.create<arith::AddIOp>(loop.getLoc(), std::get<0>(pair),
                                      std::get<2>(pair)),
        std::get<1>(pair));
    cond = cond ? static_cast<Value>(
                      builder.create<arith::AndIOp>(loop.getLoc(), cond, val))
                : val;
  }
  return cond;
}

scf::IfOp SCFParallelGuardBuilder::createGuard() const {
  return builder.create<scf::IfOp>(loop.getLoc(), TypeRange(),
                                   createGuardExpr());
}

void SCFParallelGuardBuilder::replaceUsesOfLoopReturnValues(
    scf::IfOp &ifOp) const {
  // TODO
}

//===----------------------------------------------------------------------===//
// AffineForLoopGuardBuilder
//===----------------------------------------------------------------------===//

void AffineForGuardBuilder::guardLoop() const {
  AffineIfOp ifOp = createGuard();

  bool yieldsResults = !loop.getResults().empty();
  if (yieldsResults) {
    ifOp.getThenBodyBuilder().create<AffineYieldOp>(loop.getLoc(),
                                                    loop.getResults());
    ifOp.getElseBodyBuilder().create<AffineYieldOp>(loop.getLoc(),
                                                    loop.getIterOperands());
  }

  loop->moveBefore(ifOp.getThenBlock()->getTerminator());
  replaceUsesOfLoopReturnValues(ifOp);
}

IntegerSet AffineForGuardBuilder::createGuardExpr() const {
  SmallVector<AffineExpr, 2> exprs;
  SmallVector<bool, 2> eqflags;

  const AffineMap lbMap = loop.getLowerBoundMap();
  const AffineMap ubMap = loop.getUpperBoundMap();

  for (AffineExpr ub : ubMap.getResults()) {
    SmallVector<AffineExpr, 4> symbols;
    for (unsigned idx = 0; idx < ubMap.getNumSymbols(); ++idx)
      symbols.push_back(
          getAffineSymbolExpr(idx + lbMap.getNumSymbols(), loop.getContext()));

    SmallVector<AffineExpr, 4> dims;
    for (unsigned idx = 0; idx < ubMap.getNumDims(); ++idx)
      dims.push_back(
          getAffineDimExpr(idx + lbMap.getNumDims(), loop.getContext()));

    ub = ub.replaceDimsAndSymbols(dims, symbols);

    for (AffineExpr lb : lbMap.getResults()) {
      // Bound is whether this expr >= 0, which since we want ub > lb,
      // we rewrite as follows.
      exprs.push_back(ub - lb - loop.getStep());
      eqflags.push_back(false);
    }
  }

  return IntegerSet::get(
      /*dim*/ lbMap.getNumDims() + ubMap.getNumDims(),
      /*symbols*/ lbMap.getNumSymbols() + ubMap.getNumSymbols(), exprs,
      eqflags);
}

AffineIfOp AffineForGuardBuilder::createGuard() const {
  SmallVector<Value> values;
  OperandRange lb_ops = loop.getLowerBoundOperands(),
               ub_ops = loop.getUpperBoundOperands();

  std::copy(lb_ops.begin(),
            lb_ops.begin() + loop.getLowerBoundMap().getNumDims(),
            std::back_inserter(values));
  std::copy(ub_ops.begin(),
            ub_ops.begin() + loop.getUpperBoundMap().getNumDims(),
            std::back_inserter(values));
  std::copy(lb_ops.begin() + loop.getLowerBoundMap().getNumDims(), lb_ops.end(),
            std::back_inserter(values));
  std::copy(ub_ops.begin() + loop.getUpperBoundMap().getNumDims(), ub_ops.end(),
            std::back_inserter(values));

  bool yieldsResults = !loop.getResults().empty();
  TypeRange types(loop.getResults());
  return builder.create<AffineIfOp>(loop.getLoc(), types, createGuardExpr(),
                                    values, yieldsResults ? true : false);
}

void AffineForGuardBuilder::replaceUsesOfLoopReturnValues(
    AffineIfOp &ifOp) const {
  // Replace uses of the loop return value(s) with the value(s) yielded by the
  // if operation.
  for (auto it : llvm::zip(loop.getResults(), ifOp.getResults()))
    std::get<0>(it).replaceUsesWithIf(std::get<1>(it), [&](OpOperand &op) {
      Block *useBlock = op.getOwner()->getBlock();
      return useBlock != ifOp.getThenBlock();
    });
}

//===----------------------------------------------------------------------===//
// AffineParallelLoopGuardBuilder
//===----------------------------------------------------------------------===//

void AffineParallelGuardBuilder::guardLoop() const {
  AffineIfOp ifOp = createGuard();
  loop->moveBefore(ifOp.getThenBlock()->getTerminator());
  replaceUsesOfLoopReturnValues(ifOp);
}

IntegerSet AffineParallelGuardBuilder::createGuardExpr() const {
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
        // Bound is whether this expr >= 0, which since we want ub > lb,
        // we rewrite as follows.
        exprs.push_back(ub - lb - step.value());
        eqflags.push_back(false);
      }
    }
  }

  return IntegerSet::get(
      /*dim*/ loop.getLowerBoundsMap().getNumDims() +
          loop.getUpperBoundsMap().getNumDims(),
      /*symbols*/ loop.getLowerBoundsMap().getNumSymbols() +
          loop.getUpperBoundsMap().getNumSymbols(),
      exprs, eqflags);
}

AffineIfOp AffineParallelGuardBuilder::createGuard() const {
  SmallVector<Value> values;
  OperandRange lb_ops = loop.getLowerBoundsOperands(),
               ub_ops = loop.getUpperBoundsOperands();

  std::copy(lb_ops.begin(),
            lb_ops.begin() + loop.getLowerBoundsMap().getNumDims(),
            std::back_inserter(values));
  std::copy(ub_ops.begin(),
            ub_ops.begin() + loop.getUpperBoundsMap().getNumDims(),
            std::back_inserter(values));
  std::copy(lb_ops.begin() + loop.getLowerBoundsMap().getNumDims(),
            lb_ops.end(), std::back_inserter(values));
  std::copy(ub_ops.begin() + loop.getUpperBoundsMap().getNumDims(),
            ub_ops.end(), std::back_inserter(values));

  return builder.create<AffineIfOp>(loop.getLoc(), TypeRange(),
                                    createGuardExpr(), values,
                                    /*else*/ false);
}

void AffineParallelGuardBuilder::replaceUsesOfLoopReturnValues(
    AffineIfOp &ifOp) const {
  // TODO
}

/// Determine whether any operation in the \p loop has a conflict with the
/// given operation \p op that prevents hoisting the operation out of the
/// loop. Operations that are already known to have no hoisting preventing
/// conflicts in the loop are given in \p willBeMoved.
static bool hasConflictsInLoop(Operation &op, LoopLikeOpInterface loop,
                               const SmallPtrSetImpl<Operation *> &willBeMoved,
                               const AliasAnalysis &aliasAnalysis) {
  const OperationSideEffects sideEffects(op, aliasAnalysis);

  Optional<Operation *> conflictingOp =
      TypeSwitch<Operation *, Optional<Operation *>>((Operation *)loop)
          .Case<scf::ParallelOp, AffineParallelOp>([&](auto loop) {
            // Check for conflicts with (only) other previous operations
            // in the same block.
            Operation *point = &op;
            return sideEffects.conflictsWithOperationInBlock(*op.getBlock(),
                                                             point);
          })
          .Default([&](auto loop) {
            // Check for conflicts with all other operations in the same
            // block.
            return sideEffects.conflictsWithOperationInBlock(*op.getBlock());
          });

  if (conflictingOp.has_value()) {
    if (!willBeMoved.count(*conflictingOp))
      return true;
    LLVM_DEBUG(llvm::dbgs() << "Related operation will be hoisted\n");
  }

  // Check whether the parent operation has conflicts on the loop.
  if (op.getParentOp() == loop)
    return false;
  if (hasConflictsInLoop(*op.getParentOp(), loop, willBeMoved, aliasAnalysis))
    return true;

  // If the parent operation is not guaranteed to execute its
  // (single-block) region once, walk the block.
  bool conflict = false;
  if (!isa<scf::IfOp, AffineIfOp, memref::AllocaScopeOp>(op)) {
    op.walk([&](Operation *in) {
      if (!willBeMoved.count(in) && sideEffects.conflictsWith(*in)) {
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "conflicting operation: " << *in << "\n");
        conflict = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

  return conflict;
}

/// Returns true if the Operation \p op can be hoisted out of the given loop
/// \p loop. The \p willBeMoved argument represents operations that are known
/// to be loop invariant (and therefore will be moved outside of the loop).
static bool canBeHoisted(Operation &op, LoopLikeOpInterface loop,
                         const SmallPtrSetImpl<Operation *> &willBeMoved,
                         const AliasAnalysis &aliasAnalysis) {
  // Returns true if the given value can be moved outside of the loop, and
  // false otherwise. A value cannot be moved outside of the loop if its
  // operands are not defined outside of the loop and cannot themselves be
  // moved.
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
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2)
          << "cannot be hoisted: operand(s) can't be hoisted\n";
    });
    return false;
  }

  // If the operation has no side effects it can be hoisted.
  if (isMemoryEffectFree(&op)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2) << "can be hoisted: has no side effects\n";
    });
    return true;
  }

  // Operations with unknown side effects cannot be hoisted.
  if (!isa<MemoryEffectOpInterface>(op) &&
      !op.hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2) << "cannot be hoisted: unknown side effects\n";
    });
    return false;
  }

  // Do not hoist operations that allocate a resource.
  const OperationSideEffects sideEffects(op, aliasAnalysis);
  if (sideEffects.allocatesResource()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2)
          << "cannot be hoisted: operation allocates a resource\n";
    });
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << sideEffects);

  // If the operation has side effects, check whether other operations in the
  // loop prevent hosting it.
  if ((sideEffects.readsFromResource() || sideEffects.writesToResource() ||
       sideEffects.freesResource()) &&
      hasConflictsInLoop(op, loop, willBeMoved, aliasAnalysis)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Cannot be hoisted: found conflicting operation\n");
    return false;
  }

  // Recurse into the regions for this op and check whether the contained ops
  // can be hoisted. We can inductively assume that this op will have its
  // block args available outside the loop.
  SmallPtrSet<Operation *, 2> willBeMoved2(willBeMoved.begin(),
                                           willBeMoved.end());
  willBeMoved2.insert(&op);

  for (Region &region : op.getRegions()) {
    for (Operation &innerOp : region.getOps()) {
      if (!canBeHoisted(innerOp, loop, willBeMoved2, aliasAnalysis))
        return false;
      willBeMoved2.insert(&innerOp);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Can be hoisted: no conflicts found\n");

  return true;
}

// Populate \p opsToMove with operations that can be hoisted out of the given
// loop \p loop.
static void
collectHoistableOperations(LoopLikeOpInterface loop,
                           const AliasAnalysis &aliasAnalysis,
                           SmallVectorImpl<Operation *> &opsToMove) {
  // Do not use walk here, as we do not want to go into nested regions and
  // hoist operations from there. These regions might have semantics unknown
  // to this rewriting. If the nested regions are loops, they will have been
  // processed.
  SmallPtrSet<Operation *, 8> willBeMoved;
  for (Block &block : loop.getLoopBody()) {
    for (Operation &op : block.without_terminator()) {
      if (!canBeHoisted(op, loop, willBeMoved, aliasAnalysis))
        continue;
      opsToMove.push_back(&op);
      willBeMoved.insert(&op);
    }
  }
}

static size_t moveLoopInvariantCode(LoopLikeOpInterface loop,
                                    const AliasAnalysis &aliasAnalysis) {
  Operation *loopOp = loop;
  if (!isa<scf::ParallelOp, AffineParallelOp, AffineForOp>(loopOp))
    return 0;

  SmallVector<Operation *, 8> opsToMove;
  collectHoistableOperations(loop, aliasAnalysis, opsToMove);
  if (opsToMove.empty())
    return 0;

  LoopGuardBuilder::create(loop)->guardLoop();

  size_t numOpsHoisted = 0;
  for (Operation *op : opsToMove) {
    loop.moveOutOfLoop(op);
    ++numOpsHoisted;
  }

  return numOpsHoisted;
}

void ParallelLICM::runOnOperation() {
  AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();

  [[maybe_unused]] auto getParentFunction = [](LoopLikeOpInterface loop) {
    Operation *parentOp = loop;
    do {
      parentOp = parentOp->getParentOp();
    } while (parentOp && !isa<func::FuncOp>(parentOp));
    assert(parentOp && "Failed to find parent function");
    return parentOp;
  };

  getOperation()->walk([&](LoopLikeOpInterface loop) {
    LLVM_DEBUG({
      llvm::dbgs() << "----------------\n";
      loop.print(llvm::dbgs() << "Original loop:\n");
      llvm::dbgs() << "\n";
      llvm::dbgs() << "in function " << *getParentFunction(loop) << "\n";
    });

    // First use MLIR LICM to hoist simple operations.
    {
      size_t numOpHoisted = moveLoopInvariantCode(loop);

      LLVM_DEBUG({
        if (numOpHoisted)
          loop.print(llvm::dbgs() << "Loop after MLIR LICM:\n");
        llvm::dbgs() << "\nHoisted " << numOpHoisted << " operation(s)\n";
        llvm::dbgs() << "----------------\n";
        llvm::dbgs() << "in function " << *getParentFunction(loop) << "\n";
        assert(mlir::verify(getParentFunction(loop)).succeeded());
      });
    }

    // Now use this pass to hoist more complex operations.
    {
      numOpHoisted = moveLoopInvariantCode(loop, aliasAnalysis);

      LLVM_DEBUG({
        if (numOpHoisted)
          loop.print(llvm::dbgs() << "Loop after Parallel LICM:\n");
        llvm::dbgs() << "\nHoisted " << numOpHoisted << " operation(s)\n";
        llvm::dbgs() << "----------------\n";
        llvm::dbgs() << "in function " << *getParentFunction(loop) << "\n";
        assert(mlir::verify(getParentFunction(loop)).succeeded());
      });
    }
  });
}

std::unique_ptr<Pass> mlir::polygeist::createParallelLICMPass() {
  return std::make_unique<ParallelLICM>();
}
