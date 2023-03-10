//===- LICM.cpp - Loop Invariant Code Motion ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "licm"
#define REPORT_DEBUG_TYPE DEBUG_TYPE "-report"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_LICM
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

namespace {

struct LICM : public mlir::polygeist::impl::LICMBase<LICM> {
  using LICMBase<LICM>::LICMBase;

  void runOnOperation() override;
};

/// Represents the side effects associated with an operation.
class OperationSideEffects {
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const OperationSideEffects &);

public:
  OperationSideEffects(const Operation &op, const AliasAnalysis &aliasAnalysis,
                       const DominanceInfo &domInfo)
      : op(op), aliasAnalysis(aliasAnalysis), domInfo(domInfo) {
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
  const DominanceInfo &domInfo;       /// Dominance information reference.

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
    OS.indent(2) << "=> is side effects free.\n";
  else {
    if (ME.readsFromResource())
      printResources("=> read resources", ME.readResources);
    if (ME.writesToResource())
      printResources("=> write resources", ME.writeResources);
    if (ME.freesResource())
      printResources("=> free resources", ME.freeResources);
    if (ME.allocatesResource())
      printResources("=> allocate resources", ME.allocateResources);
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

class SCFLoopGuardBuilder : public LoopGuardBuilder {
public:
  SCFLoopGuardBuilder(LoopLikeOpInterface loop) : LoopGuardBuilder(loop) {}

protected:
  scf::IfOp guard(LoopLikeOpInterface loop) const;

private:
  virtual Value createGuardExpr() const = 0;
  virtual scf::IfOp createGuard() const = 0;
  virtual void replaceUsesOfLoopReturnValues(scf::IfOp &) const {}
};

class SCFForGuardBuilder : public SCFLoopGuardBuilder {
public:
  SCFForGuardBuilder(scf::ForOp loop) : SCFLoopGuardBuilder(loop), loop(loop) {}

  void guardLoop() const final;

private:
  Value createGuardExpr() const final;
  scf::IfOp createGuard() const final;
  void replaceUsesOfLoopReturnValues(scf::IfOp &) const final;

  mutable scf::ForOp loop;
};

class SCFParallelGuardBuilder : public SCFLoopGuardBuilder {
public:
  SCFParallelGuardBuilder(scf::ParallelOp loop)
      : SCFLoopGuardBuilder(loop), loop(loop) {}

  void guardLoop() const final;

private:
  Value createGuardExpr() const final;
  scf::IfOp createGuard() const final;

  mutable scf::ParallelOp loop;
};

class AffineLoopGuardBuilder : public LoopGuardBuilder {
public:
  AffineLoopGuardBuilder(LoopLikeOpInterface loop) : LoopGuardBuilder(loop) {}

protected:
  AffineIfOp guard(LoopLikeOpInterface loop) const;

private:
  virtual IntegerSet createGuardExpr() const = 0;
  virtual AffineIfOp createGuard() const = 0;
  virtual void yieldResults(AffineIfOp &) const {}
  virtual void replaceUsesOfLoopReturnValues(AffineIfOp &) const {}
};

class AffineForGuardBuilder : public AffineLoopGuardBuilder {
public:
  AffineForGuardBuilder(AffineForOp loop)
      : AffineLoopGuardBuilder(loop), loop(loop) {}

  void guardLoop() const final;

private:
  IntegerSet createGuardExpr() const final;
  AffineIfOp createGuard() const final;
  void yieldResults(AffineIfOp &) const final;
  void replaceUsesOfLoopReturnValues(AffineIfOp &) const final;

  mutable AffineForOp loop;
};

class AffineParallelGuardBuilder : public AffineLoopGuardBuilder {
public:
  AffineParallelGuardBuilder(AffineParallelOp loop)
      : AffineLoopGuardBuilder(loop), loop(loop) {}

  void guardLoop() const final;

private:
  IntegerSet createGuardExpr() const final;
  AffineIfOp createGuard() const final;

  mutable AffineParallelOp loop;
};

} // namespace

//===----------------------------------------------------------------------===//
// OperationSideEffects
//===----------------------------------------------------------------------===//

bool OperationSideEffects::conflictsWith(const Operation &other) const {
  if (&op == &other || isMemoryEffectFree(const_cast<Operation *>(&other)))
    return false;

  // Conservatively assume operations with unknown side effects might write to
  // any memory.
  if (!isa<MemoryEffectOpInterface>(other) &&
      !const_cast<Operation &>(other)
           .hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "=> found conflict due to operation with unknown side effects:\n";
      llvm::dbgs().indent(2) << other << "\n";
    });
    return true;
  }

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
    OperationSideEffects sideEffects(other, aliasAnalysis, domInfo);

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
        [](const MemoryEffects::EffectInstance &EI, AliasResult aliasRes,
           const Operation &other) {
          llvm::dbgs().indent(2)
              << "=> found conflicting side effect with: " << other << "\n";
          llvm::dbgs().indent(2) << "=> aliasResult: " << aliasRes << "\n";
        };

    // Check whether the given operation 'other' allocates, writes, or frees a
    // resource that is read by the operation associated with this class.
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

                LLVM_DEBUG(printConflictingSideEffects(EI, aliasRes, other));
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

                // An aliased read operation doesn't prevent hoisting if it is
                // dominated by the write operation.
                if (isa<MemoryEffects::Read>(EI.getEffect()) &&
                    domInfo.dominates(const_cast<Operation *>(&op),
                                      const_cast<Operation *>(&other))) {
                  LLVM_DEBUG({
                    printConflictingSideEffects(EI, aliasRes, other);
                    llvm::dbgs().indent(2)
                        << "=> aliased write operation dominates the "
                           "read operation\n";
                  });
                  return false;
                }

                LLVM_DEBUG(printConflictingSideEffects(EI, aliasRes, other));
                return true;
              };

              return checkForConflict(writeRes.getResource(), hasConflict);
            })) {
      return true;
    }

    // Check whether the given operation 'other' allocates, reads, writes or
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

                LLVM_DEBUG(printConflictingSideEffects(EI, aliasRes, other));
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
      });
}

//===----------------------------------------------------------------------===//
// SCFLoopGuardBuilder
//===----------------------------------------------------------------------===//

scf::IfOp SCFLoopGuardBuilder::guard(LoopLikeOpInterface loop) const {
  scf::IfOp ifOp = createGuard();
  loop->moveBefore(ifOp.thenYield());
  replaceUsesOfLoopReturnValues(ifOp);
  return ifOp;
}

//===----------------------------------------------------------------------===//
// SCFForLoopGuardBuilder
//===----------------------------------------------------------------------===//

void SCFForGuardBuilder::guardLoop() const {
  scf::IfOp ifOp = guard(loop);
  bool yieldsResults = !loop->getResults().empty();
  if (!yieldsResults)
    ifOp.elseBlock()->erase();
}

Value SCFForGuardBuilder::createGuardExpr() const {
  return builder.create<arith::CmpIOp>(loop.getLoc(), arith::CmpIPredicate::slt,
                                       loop.getLowerBound(),
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

void SCFParallelGuardBuilder::guardLoop() const { guard(loop); }

Value SCFParallelGuardBuilder::createGuardExpr() const {
  Value cond;
  for (auto pair :
       llvm::zip(loop.getLowerBound(), loop.getUpperBound(), loop.getStep())) {
    const Value val =
        builder.create<arith::CmpIOp>(loop.getLoc(), arith::CmpIPredicate::slt,
                                      std::get<0>(pair), std::get<1>(pair));
    cond = cond ? static_cast<Value>(
                      builder.create<arith::AndIOp>(loop.getLoc(), cond, val))
                : val;
  }
  return cond;
}

scf::IfOp SCFParallelGuardBuilder::createGuard() const {
  TypeRange types(loop->getResults());
  return builder.create<scf::IfOp>(
      loop.getLoc(), types, createGuardExpr(),
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, loop.getResults());
      },
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, loop.getInitVals());
      });
}

//===----------------------------------------------------------------------===//
// AffineLoopGuardBuilder
//===----------------------------------------------------------------------===//

AffineIfOp AffineLoopGuardBuilder::guard(LoopLikeOpInterface loop) const {
  AffineIfOp ifOp = createGuard();
  yieldResults(ifOp);
  loop->moveBefore(ifOp.getThenBlock()->getTerminator());
  replaceUsesOfLoopReturnValues(ifOp);
  return ifOp;
}

//===----------------------------------------------------------------------===//
// AffineForLoopGuardBuilder
//===----------------------------------------------------------------------===//

void AffineForGuardBuilder::guardLoop() const { guard(loop); }

IntegerSet AffineForGuardBuilder::createGuardExpr() const {
  SmallVector<AffineExpr, 2> exprs;
  SmallVector<bool, 2> eqflags;

  const AffineMap lbMap = loop.getLowerBoundMap(),
                  ubMap = loop.getUpperBoundMap();

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
      exprs.push_back(ub - lb - 1);
      eqflags.push_back(false);
    }
  }

  return IntegerSet::get(
      /*dim*/ lbMap.getNumDims() + ubMap.getNumDims(),
      /*symbols*/ lbMap.getNumSymbols() + ubMap.getNumSymbols(), exprs,
      eqflags);
}

void AffineForGuardBuilder::yieldResults(AffineIfOp &ifOp) const {
  bool yieldsResults = !loop.getResults().empty();
  if (yieldsResults) {
    ifOp.getThenBodyBuilder().create<AffineYieldOp>(loop.getLoc(),
                                                    loop.getResults());
    ifOp.getElseBodyBuilder().create<AffineYieldOp>(loop.getLoc(),
                                                    loop.getIterOperands());
  }
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
                                    values, yieldsResults);
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

void AffineParallelGuardBuilder::guardLoop() const { guard(loop); }

IntegerSet AffineParallelGuardBuilder::createGuardExpr() const {
  SmallVector<AffineExpr, 2> exprs;
  SmallVector<bool, 2> eqflags;

  const AffineMap lbMap = loop.getLowerBoundsMap(),
                  ubMap = loop.getUpperBoundsMap();

  for (auto step : llvm::enumerate(loop.getSteps())) {
    for (AffineExpr ub : loop.getUpperBoundMap(step.index()).getResults()) {
      SmallVector<AffineExpr, 4> symbols;
      for (unsigned idx = 0; idx < ubMap.getNumSymbols(); ++idx)
        symbols.push_back(getAffineSymbolExpr(idx + lbMap.getNumSymbols(),
                                              loop.getContext()));

      SmallVector<AffineExpr, 4> dims;
      for (unsigned idx = 0; idx < ubMap.getNumDims(); ++idx)
        dims.push_back(
            getAffineDimExpr(idx + lbMap.getNumDims(), loop.getContext()));

      ub = ub.replaceDimsAndSymbols(dims, symbols);

      for (AffineExpr lb : loop.getLowerBoundMap(step.index()).getResults()) {
        // Bound is whether this expr >= 0, which since we want ub > lb, we
        // rewrite as follows.
        exprs.push_back(ub - lb - step.value());
        eqflags.push_back(false);
      }
    }
  }

  return IntegerSet::get(
      /*dim*/ lbMap.getNumDims() + ubMap.getNumDims(),
      /*symbols*/ lbMap.getNumSymbols() + ubMap.getNumSymbols(), exprs,
      eqflags);
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

/// Determine whether any operation in the \p loop has a conflict with the
/// given operation \p op that prevents hoisting the operation out of the
/// loop. Operations that are already known to have no hoisting preventing
/// conflicts in the loop are given in \p willBeMoved.
static bool hasConflictsInLoop(Operation &op, LoopLikeOpInterface loop,
                               const SmallPtrSetImpl<Operation *> &willBeMoved,
                               const AliasAnalysis &aliasAnalysis,
                               const DominanceInfo &domInfo) {
  const OperationSideEffects sideEffects(op, aliasAnalysis, domInfo);

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
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "can be hoisted: conflicting operation will be hoisted\n");
  }

  // Check whether the parent operation has conflicts on the loop.
  if (op.getParentOp() == loop)
    return false;
  if (hasConflictsInLoop(*op.getParentOp(), loop, willBeMoved, aliasAnalysis,
                         domInfo))
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
                         const AliasAnalysis &aliasAnalysis,
                         const DominanceInfo &domInfo) {
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

  // Operations with unknown side effects cannot be hoisted.
  if (!isa<MemoryEffectOpInterface>(op) &&
      !op.hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2)
          << "**** cannot be hoisted: unknown side effects\n\n";
    });
    return false;
  }

  // Ensure operands can be hoisted.
  if (llvm::any_of(op.getOperands(),
                   [&](Value value) { return !canBeMoved(value); })) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2)
          << "**** cannot be hoisted: operand(s) can't be hoisted\n\n";
    });
    return false;
  }

  // If the operation has no side effects it can be hoisted.
  if (isMemoryEffectFree(&op)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2) << "**** can be hoisted: has no side effects\n\n";
    });
    return true;
  }

  // Do not hoist operations that allocate a resource.
  const OperationSideEffects sideEffects(op, aliasAnalysis, domInfo);
  if (sideEffects.allocatesResource()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Operation: " << op << "\n";
      llvm::dbgs().indent(2)
          << "**** cannot be hoisted: operation allocates a resource\n\n";
    });
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << sideEffects);

  // If the operation has side effects, check whether other operations in the
  // loop prevent hosting it.
  if ((sideEffects.readsFromResource() || sideEffects.writesToResource() ||
       sideEffects.freesResource()) &&
      hasConflictsInLoop(op, loop, willBeMoved, aliasAnalysis, domInfo)) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "**** cannot be hoisted: found conflicting operation\n\n");
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
      if (!canBeHoisted(innerOp, loop, willBeMoved2, aliasAnalysis, domInfo))
        return false;
      willBeMoved2.insert(&innerOp);
    }
  }

  LLVM_DEBUG(llvm::dbgs().indent(2)
             << "**** can be hoisted: no conflicts found\n\n");

  return true;
}

// Populate \p opsToMove with operations that can be hoisted out of the given
// loop \p loop.
static void collectHoistableOperations(
    LoopLikeOpInterface loop, const AliasAnalysis &aliasAnalysis,
    const DominanceInfo &domInfo, SmallVectorImpl<Operation *> &opsToMove) {
  // Do not use walk here, as we do not want to go into nested regions and
  // hoist operations from there. These regions might have semantics unknown
  // to this rewriting. If the nested regions are loops, they will have been
  // processed.
  SmallPtrSet<Operation *, 8> willBeMoved;
  for (Block &block : loop.getLoopBody()) {
    for (Operation &op : block.without_terminator()) {
      if (!canBeHoisted(op, loop, willBeMoved, aliasAnalysis, domInfo))
        continue;
      opsToMove.push_back(&op);
      willBeMoved.insert(&op);
    }
  }
}

static size_t moveLoopInvariantCode(LoopLikeOpInterface loop,
                                    const AliasAnalysis &aliasAnalysis,
                                    const DominanceInfo &domInfo) {
  Operation *loopOp = loop;
  if (!isa<scf::ForOp, scf::ParallelOp, AffineParallelOp, AffineForOp>(loopOp))
    return 0;

  SmallVector<Operation *, 8> opsToMove;
  collectHoistableOperations(loop, aliasAnalysis, domInfo, opsToMove);
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

void LICM::runOnOperation() {
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  AliasAnalysis &aliasAnalysis = getAnalysis<AliasAnalysis>();
  aliasAnalysis.addAnalysisImplementation(sycl::AliasAnalysis(relaxedAliasing));

  getOperation()->walk([&](LoopLikeOpInterface loop) {
    LLVM_DEBUG({
      llvm::dbgs() << "----------------\n";
      loop.print(llvm::dbgs() << "Original loop:\n");
      llvm::dbgs() << "\nIn:\n"
                   << *loop->getParentOfType<FunctionOpInterface>() << "\n";
    });

    // First use MLIR LICM to hoist simple operations.
    {
      size_t OpHoisted = moveLoopInvariantCode(loop);

      LLVM_DEBUG({
        llvm::dbgs() << "\nMLIR LICM hoisted " << OpHoisted
                     << " operation(s).\n";
        if (OpHoisted) {
          loop.print(llvm::dbgs() << "Loop after MLIR LICM:\n");
          llvm::dbgs() << "\nIn:\n"
                       << *loop->getParentOfType<FunctionOpInterface>() << "\n";
          assert(mlir::verify(loop->getParentOfType<FunctionOpInterface>())
                     .succeeded());
        }
        llvm::dbgs() << "----------------\n";
      });
    }

    // Now use this pass to hoist more complex operations.
    {
      size_t OpHoisted = moveLoopInvariantCode(loop, aliasAnalysis, domInfo);
      numOpHoisted += OpHoisted;

      LLVM_DEBUG({
        llvm::dbgs() << "\nLICM hoisted " << OpHoisted << " operation(s).\n";
        if (OpHoisted) {
          loop.print(llvm::dbgs() << "Loop after LICM:\n");
          llvm::dbgs() << "\nIn:\n"
                       << *loop->getParentOfType<FunctionOpInterface>() << "\n";
          assert(mlir::verify(loop->getParentOfType<FunctionOpInterface>())
                     .succeeded());
        }
        llvm::dbgs() << "----------------\n";
      });

      DEBUG_WITH_TYPE(REPORT_DEBUG_TYPE, {
        if (OpHoisted)
          llvm::dbgs() << "LICM: hoisted " << OpHoisted
                       << " operations(s) in : "
                       << loop->getParentOfType<FunctionOpInterface>().getName()
                       << "\n";
      });
    }
  });
}

std::unique_ptr<Pass> mlir::polygeist::createLICMPass() {
  return std::make_unique<LICM>();
}

std::unique_ptr<Pass>
mlir::polygeist::createLICMPass(const LICMOptions &options) {
  return std::make_unique<LICM>(options);
}
