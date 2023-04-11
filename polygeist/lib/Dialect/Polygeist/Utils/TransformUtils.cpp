//===- TransformUtils.cpp - Polygeist Transform Utilities  ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utilities functions
//===----------------------------------------------------------------------===//

static Block &getThenBlock(RegionBranchOpInterface ifOp) {
  return ifOp->getRegion(0).front();
}

static Block &getElseBlock(RegionBranchOpInterface ifOp) {
  return ifOp->getRegion(1).front();
}

// Replace uses of the loop \p loop return value(s) with the value(s) yielded by
// the \p ifOp operation.
static void replaceUsesOfLoopReturnValues(LoopLikeOpInterface loop,
                                          RegionBranchOpInterface ifOp) {
  assert(ifOp && "Expected valid ifOp");
  for (auto [loopVal, ifVal] :
       llvm::zip(loop->getResults(), ifOp->getResults()))
    loopVal.replaceUsesWithIf(ifVal, [&](OpOperand &op) {
      Block *useBlock = op.getOwner()->getBlock();
      return useBlock != &getThenBlock(ifOp);
    });
}

static void createThenBody(LoopLikeOpInterface loop, scf::IfOp ifOp) {
  loop->moveBefore(&getThenBlock(ifOp).front());
}

static void createThenBody(LoopLikeOpInterface loop, AffineIfOp ifOp) {
  OpBuilder thenBodyBuilder = ifOp.getThenBodyBuilder();
  if (!loop->getResults().empty())
    thenBodyBuilder.create<AffineYieldOp>(loop.getLoc(), loop->getResults());
  loop->moveBefore(&getThenBlock(ifOp).front());
}

namespace {

struct SCFIfBuilder {
  static scf::IfOp createIfOp(Value condition, Operation::result_range results,
                              OpBuilder &builder, Location loc) {
    assert(condition && "Expecting a valid condition");
    return builder.create<scf::IfOp>(
        loc, condition,
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, results);
        },
        [&](OpBuilder &b, Location loc) {
          b.create<scf::YieldOp>(loc, results);
        });
  }
};

struct AffineIfBuilder {
  static AffineIfOp createIfOp(IntegerSet ifCondSet,
                               SmallVectorImpl<Value> &setOperands,
                               Operation::result_range results,
                               OpBuilder &builder, Location loc) {
    TypeRange types(results);
    return builder.create<AffineIfOp>(loc, types, ifCondSet, setOperands, true);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// LoopVersionBuilder
//===----------------------------------------------------------------------===//

void LoopVersionBuilder::versionLoop(
    const LoopVersionCondition &versionCond) const {
  OpBuilder builder(loop);

  if (versionCond.hasSCFCondition()) {
    scf::IfOp ifOp =
        SCFIfBuilder::createIfOp(versionCond.getSCFCondition(),
                                 loop->getResults(), builder, loop.getLoc());
    createThenBody(loop, ifOp);
    createElseBody(ifOp);
    replaceUsesOfLoopReturnValues(loop, ifOp);
  } else {
    assert(versionCond.hasAffineCondition() && "Expecting an affine condition");
    const auto &affineCond = versionCond.getAffineCondition();
    AffineIfOp ifOp = AffineIfBuilder::createIfOp(
        affineCond.ifCondSet, affineCond.setOperands, loop->getResults(),
        builder, loop.getLoc());
    createThenBody(loop, ifOp);
    createElseBody(ifOp);
    replaceUsesOfLoopReturnValues(loop, ifOp);
  }
}

void LoopVersionBuilder::createElseBody(scf::IfOp ifOp) const {
  Operation &origYield = getElseBlock(ifOp).back();
  OpBuilder elseBodyBuilder = ifOp.getElseBodyBuilder();
  Operation *clonedLoop = elseBodyBuilder.clone(*loop.getOperation());
  elseBodyBuilder.create<scf::YieldOp>(loop.getLoc(), clonedLoop->getResults());
  origYield.erase();
}

void LoopVersionBuilder::createElseBody(AffineIfOp ifOp) const {
  OpBuilder elseBodyBuilder = ifOp.getElseBodyBuilder();
  Operation *clonedLoop = elseBodyBuilder.clone(*loop.getOperation());
  if (!clonedLoop->getResults().empty())
    elseBodyBuilder.create<AffineYieldOp>(loop.getLoc(),
                                          clonedLoop->getResults());
}

//===----------------------------------------------------------------------===//
// LoopGuardBuilder
//===----------------------------------------------------------------------===//

std::unique_ptr<LoopGuardBuilder>
LoopGuardBuilder::create(LoopLikeOpInterface loop) {
  return TypeSwitch<Operation *, std::unique_ptr<LoopGuardBuilder>>(loop)
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

void LoopGuardBuilder::guardLoop(RegionBranchOpInterface ifOp) const {
  createThenBody(ifOp);
  createElseBody(ifOp);
  replaceUsesOfLoopReturnValues(loop, ifOp);
}

//===----------------------------------------------------------------------===//
// SCFLoopGuardBuilder
//===----------------------------------------------------------------------===//

void SCFLoopGuardBuilder::guardLoop() const {
  OpBuilder builder(loop);
  Value condition = createCondition();
  RegionBranchOpInterface ifOp = SCFIfBuilder::createIfOp(
      condition, loop->getResults(), builder, loop.getLoc());
  LoopGuardBuilder::guardLoop(ifOp);
}

Value SCFLoopGuardBuilder::createCondition() const {
  OpBuilder builder(loop);
  Value cond;
  for (auto [lb, ub] : llvm::zip(getLowerBounds(), getUpperBounds())) {
    const Value val = builder.create<arith::CmpIOp>(
        loop.getLoc(), arith::CmpIPredicate::slt, lb, ub);
    cond = cond ? static_cast<Value>(
                      builder.create<arith::AndIOp>(loop.getLoc(), cond, val))
                : val;
  }
  return cond;
}

void SCFLoopGuardBuilder::createThenBody(RegionBranchOpInterface ifOp) const {
  ::createThenBody(loop, cast<scf::IfOp>(ifOp));
}

void SCFLoopGuardBuilder::createElseBody(RegionBranchOpInterface ifOp) const {
  Operation &origYield = getElseBlock(ifOp).back();
  bool yieldsResults = !loop->getResults().empty();
  OpBuilder elseBodyBuilder = cast<scf::IfOp>(ifOp).getElseBodyBuilder();
  if (yieldsResults) {
    elseBodyBuilder.create<scf::YieldOp>(loop.getLoc(), getInitVals());
    origYield.erase();
  } else
    getElseBlock(ifOp).erase();
}

//===----------------------------------------------------------------------===//
// AffineLoopGuardBuilder
//===----------------------------------------------------------------------===//

void AffineLoopGuardBuilder::guardLoop() const {
  OpBuilder builder(loop);
  SmallVector<Value> setOperands;
  IntegerSet ifCondSet = createCondition(setOperands);
  RegionBranchOpInterface ifOp = AffineIfBuilder::createIfOp(
      ifCondSet, setOperands, loop->getResults(), builder, loop.getLoc());
  LoopGuardBuilder::guardLoop(ifOp);
}

void AffineLoopGuardBuilder::createThenBody(
    RegionBranchOpInterface ifOp) const {
  ::createThenBody(loop, cast<AffineIfOp>(ifOp));
}

void AffineLoopGuardBuilder::createElseBody(
    RegionBranchOpInterface ifOp) const {
  bool yieldsResults = !loop->getResults().empty();
  OpBuilder elseBodyBuilder = cast<AffineIfOp>(ifOp).getElseBodyBuilder();
  if (yieldsResults)
    elseBodyBuilder.create<AffineYieldOp>(loop.getLoc(), getInitVals());
  else
    getElseBlock(ifOp).erase();
}

IntegerSet AffineLoopGuardBuilder::createCondition(
    SmallVectorImpl<Value> &setOperands) const {
  OperandRange lb_ops = getLowerBoundsOperands(),
               ub_ops = getUpperBoundsOperands();
  const AffineMap lbMap = getLowerBoundsMap(), ubMap = getUpperBoundsMap();

  std::copy(lb_ops.begin(), lb_ops.begin() + lbMap.getNumDims(),
            std::back_inserter(setOperands));
  std::copy(ub_ops.begin(), ub_ops.begin() + ubMap.getNumDims(),
            std::back_inserter(setOperands));
  std::copy(lb_ops.begin() + lbMap.getNumDims(), lb_ops.end(),
            std::back_inserter(setOperands));
  std::copy(ub_ops.begin() + ubMap.getNumDims(), ub_ops.end(),
            std::back_inserter(setOperands));

  SmallVector<AffineExpr, 4> dims;
  for (unsigned idx = 0; idx < ubMap.getNumDims(); ++idx)
    dims.push_back(
        getAffineDimExpr(idx + lbMap.getNumDims(), loop.getContext()));

  SmallVector<AffineExpr, 4> symbols;
  for (unsigned idx = 0; idx < ubMap.getNumSymbols(); ++idx)
    symbols.push_back(
        getAffineSymbolExpr(idx + lbMap.getNumSymbols(), loop.getContext()));

  SmallVector<AffineExpr, 2> exprs;
  getConstraints(exprs, dims, symbols);
  SmallVector<bool, 2> eqFlags(exprs.size(), false);

  return IntegerSet::get(
      /*dim*/ lbMap.getNumDims() + ubMap.getNumDims(),
      /*symbols*/ lbMap.getNumSymbols() + ubMap.getNumSymbols(), exprs,
      eqFlags);
}

//===----------------------------------------------------------------------===//
// SCFForGuardBuilder
//===----------------------------------------------------------------------===//

SCFForGuardBuilder::SCFForGuardBuilder(scf::ForOp loop)
    : SCFLoopGuardBuilder(loop) {}

scf::ForOp SCFForGuardBuilder::getLoop() const {
  return cast<scf::ForOp>(loop);
}

OperandRange SCFForGuardBuilder::getInitVals() const {
  return getLoop().getInitArgs();
}

OperandRange SCFForGuardBuilder::getLowerBounds() const {
  return getLoop().getODSOperands(0);
}

OperandRange SCFForGuardBuilder::getUpperBounds() const {
  return getLoop().getODSOperands(1);
}

//===----------------------------------------------------------------------===//
// SCFParallelGuardBuilder
//===----------------------------------------------------------------------===//

SCFParallelGuardBuilder::SCFParallelGuardBuilder(scf::ParallelOp loop)
    : SCFLoopGuardBuilder(loop) {}

scf::ParallelOp SCFParallelGuardBuilder::getLoop() const {
  return cast<scf::ParallelOp>(loop);
}

OperandRange SCFParallelGuardBuilder::getInitVals() const {
  return getLoop().getInitVals();
}

OperandRange SCFParallelGuardBuilder::getLowerBounds() const {
  return getLoop().getLowerBound();
}

OperandRange SCFParallelGuardBuilder::getUpperBounds() const {
  return getLoop().getUpperBound();
}

//===----------------------------------------------------------------------===//
// AffineForGuardBuilder
//===----------------------------------------------------------------------===//

AffineForGuardBuilder::AffineForGuardBuilder(AffineForOp loop)
    : AffineLoopGuardBuilder(loop) {}

AffineForOp AffineForGuardBuilder::getLoop() const {
  return cast<AffineForOp>(loop);
}

void AffineForGuardBuilder::getConstraints(SmallVectorImpl<AffineExpr> &exprs,
                                           ArrayRef<AffineExpr> dims,
                                           ArrayRef<AffineExpr> symbols) const {
  for (AffineExpr ub : getLoop().getUpperBoundMap().getResults()) {
    ub = ub.replaceDimsAndSymbols(dims, symbols);
    for (AffineExpr lb : getLoop().getLowerBoundMap().getResults()) {
      // Bound is whether this expr >= 0, which since we want ub > lb, we
      // rewrite as follows.
      exprs.push_back(ub - lb - 1);
    }
  }
}

OperandRange AffineForGuardBuilder::getInitVals() const {
  return getLoop().getIterOperands();
}

OperandRange AffineForGuardBuilder::getLowerBoundsOperands() const {
  return getLoop().getLowerBoundOperands();
}

OperandRange AffineForGuardBuilder::getUpperBoundsOperands() const {
  return getLoop().getUpperBoundOperands();
}

AffineMap AffineForGuardBuilder::getLowerBoundsMap() const {
  return getLoop().getLowerBoundMap();
}

AffineMap AffineForGuardBuilder::getUpperBoundsMap() const {
  return getLoop().getUpperBoundMap();
}

//===----------------------------------------------------------------------===//
// AffineParallelGuardBuilder
//===----------------------------------------------------------------------===//

AffineParallelGuardBuilder::AffineParallelGuardBuilder(AffineParallelOp loop)
    : AffineLoopGuardBuilder(loop) {}

AffineParallelOp AffineParallelGuardBuilder::getLoop() const {
  return cast<AffineParallelOp>(loop);
}

void AffineParallelGuardBuilder::getConstraints(
    SmallVectorImpl<AffineExpr> &exprs, ArrayRef<AffineExpr> dims,
    ArrayRef<AffineExpr> symbols) const {
  for (auto step : llvm::enumerate(getLoop().getSteps()))
    for (AffineExpr ub :
         getLoop().getUpperBoundMap(step.index()).getResults()) {
      ub = ub.replaceDimsAndSymbols(dims, symbols);
      for (AffineExpr lb :
           getLoop().getLowerBoundMap(step.index()).getResults()) {
        // Bound is whether this expr >= 0, which since we want ub > lb, we
        // rewrite as follows.
        exprs.push_back(ub - lb - 1);
      }
    }
}

mlir::Operation::operand_range AffineParallelGuardBuilder::getInitVals() const {
  return getLoop().getMapOperands();
}

OperandRange AffineParallelGuardBuilder::getLowerBoundsOperands() const {
  return getLoop().getLowerBoundsOperands();
}

OperandRange AffineParallelGuardBuilder::getUpperBoundsOperands() const {
  return getLoop().getUpperBoundsOperands();
}

AffineMap AffineParallelGuardBuilder::getLowerBoundsMap() const {
  return getLoop().getLowerBoundsMap();
}

AffineMap AffineParallelGuardBuilder::getUpperBoundsMap() const {
  return getLoop().getUpperBoundsMap();
}

//===----------------------------------------------------------------------===//
// Loop Tools
//===----------------------------------------------------------------------===//

void LoopTools::guardLoop(LoopLikeOpInterface loop) {
  LoopGuardBuilder::create(loop)->guardLoop();
}

void LoopTools::versionLoop(LoopLikeOpInterface loop,
                            const LoopVersionCondition &versionCond) {
  LoopVersionBuilder(loop).versionLoop(versionCond);
}
