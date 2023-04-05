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
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// LoopVersionBuilder
//===----------------------------------------------------------------------===//

LoopVersionBuilder::LoopVersionBuilder(LoopLikeOpInterface loop)
    : builder(loop), loop(loop) {}

void LoopVersionBuilder::versionLoop() {
  createIfOp();
  createThenBody();
  createElseBody();
  replaceUsesOfLoopReturnValues();
}

void LoopVersionBuilder::replaceUsesOfLoopReturnValues() const {
  // Replace uses of the loop return value(s) with the value(s) yielded by the
  // if operation.
  for (auto [loopVal, ifVal] :
       llvm::zip(loop->getResults(), ifOp->getResults()))
    loopVal.replaceUsesWithIf(ifVal, [&](OpOperand &op) {
      Block *useBlock = op.getOwner()->getBlock();
      return useBlock != &getThenBlock(ifOp);
    });
}

//===----------------------------------------------------------------------===//
// SCFLoopVersionBuilder
//===----------------------------------------------------------------------===//

SCFLoopVersionBuilder::SCFLoopVersionBuilder(LoopLikeOpInterface loop)
    : LoopVersionBuilder(loop) {}

scf::IfOp SCFLoopVersionBuilder::getIfOp() const {
  assert(ifOp && "Expected valid ifOp");
  return cast<scf::IfOp>(ifOp);
}

void SCFLoopVersionBuilder::createIfOp() {
  ifOp = builder.create<scf::IfOp>(
      loop.getLoc(), createCondition(),
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, loop->getResults());
      },
      [&](OpBuilder &b, Location loc) {
        b.create<scf::YieldOp>(loc, loop->getResults());
      });
}

void SCFLoopVersionBuilder::createThenBody() const {
  loop->moveBefore(&*getThenBlock(ifOp).begin());
}

void SCFLoopVersionBuilder::createElseBody() const {
  Operation &origYield = getElseBlock(ifOp).back();
  OpBuilder elseBodyBuilder = getIfOp().getElseBodyBuilder();
  Operation *clonedLoop = elseBodyBuilder.clone(*loop.getOperation());
  elseBodyBuilder.create<scf::YieldOp>(loop.getLoc(), clonedLoop->getResults());
  origYield.erase();
}

//===----------------------------------------------------------------------===//
// AffineLoopVersionBuilder
//===----------------------------------------------------------------------===//

AffineLoopVersionBuilder::AffineLoopVersionBuilder(LoopLikeOpInterface loop)
    : LoopVersionBuilder(loop) {}

AffineIfOp AffineLoopVersionBuilder::getIfOp() const {
  assert(ifOp && "Expected valid ifOp");
  return cast<AffineIfOp>(ifOp);
}

void AffineLoopVersionBuilder::createIfOp() {
  TypeRange types(loop->getResults());
  SmallVector<Value> values;
  const IntegerSet &set = createCondition(values);
  ifOp = builder.create<AffineIfOp>(loop.getLoc(), types, set, values, true);
}

void AffineLoopVersionBuilder::createThenBody() const {
  OpBuilder thenBodyBuilder = getIfOp().getThenBodyBuilder();
  if (!loop->getResults().empty())
    thenBodyBuilder.create<AffineYieldOp>(loop.getLoc(), loop->getResults());
  loop->moveBefore(&*getThenBlock(ifOp).begin());
}

void AffineLoopVersionBuilder::createElseBody() const {
  OpBuilder elseBodyBuilder = getIfOp().getElseBodyBuilder();
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

SCFLoopGuardBuilder::SCFLoopGuardBuilder(LoopLikeOpInterface loop)
    : LoopGuardBuilder(), SCFLoopVersionBuilder(loop) {}

void SCFLoopGuardBuilder::createElseBody() const {
  Operation &origYield = getElseBlock(ifOp).back();
  bool yieldsResults = !loop->getResults().empty();
  OpBuilder elseBodyBuilder = getIfOp().getElseBodyBuilder();
  if (yieldsResults) {
    elseBodyBuilder.create<scf::YieldOp>(loop->getLoc(), getInitVals());
    origYield.erase();
  } else
    getElseBlock(getIfOp()).erase();
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

Value SCFForGuardBuilder::createCondition() const {
  return builder.create<arith::CmpIOp>(loop.getLoc(), arith::CmpIPredicate::slt,
                                       getLoop().getLowerBound(),
                                       getLoop().getUpperBound());
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

Value SCFParallelGuardBuilder::createCondition() const {
  Value cond;
  for (auto [lb, ub] :
       llvm::zip(getLoop().getLowerBound(), getLoop().getUpperBound())) {
    const Value val = builder.create<arith::CmpIOp>(
        loop.getLoc(), arith::CmpIPredicate::slt, lb, ub);
    cond = cond ? static_cast<Value>(
                      builder.create<arith::AndIOp>(loop.getLoc(), cond, val))
                : val;
  }
  return cond;
}

//===----------------------------------------------------------------------===//
// AffineLoopGuardBuilder
//===----------------------------------------------------------------------===//

void AffineLoopGuardBuilder::createElseBody() const {
  bool yieldsResults = !loop->getResults().empty();
  OpBuilder elseBodyBuilder = getIfOp().getElseBodyBuilder();
  if (yieldsResults)
    elseBodyBuilder.create<AffineYieldOp>(loop.getLoc(), getInitVals());
  else
    getElseBlock(getIfOp()).erase();
}

IntegerSet
AffineLoopGuardBuilder::createCondition(SmallVectorImpl<Value> &values) const {
  OperandRange lb_ops = getLowerBoundsOperands(),
               ub_ops = getUpperBoundsOperands();
  const AffineMap lbMap = getLowerBoundsMap(), ubMap = getUpperBoundsMap();

  std::copy(lb_ops.begin(), lb_ops.begin() + lbMap.getNumDims(),
            std::back_inserter(values));
  std::copy(ub_ops.begin(), ub_ops.begin() + ubMap.getNumDims(),
            std::back_inserter(values));
  std::copy(lb_ops.begin() + lbMap.getNumDims(), lb_ops.end(),
            std::back_inserter(values));
  std::copy(ub_ops.begin() + ubMap.getNumDims(), ub_ops.end(),
            std::back_inserter(values));

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
  SmallVector<bool, 2> eqflags(exprs.size(), false);

  return IntegerSet::get(
      /*dim*/ lbMap.getNumDims() + ubMap.getNumDims(),
      /*symbols*/ lbMap.getNumSymbols() + ubMap.getNumSymbols(), exprs,
      eqflags);
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
