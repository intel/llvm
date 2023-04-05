//===- Utils.cpp - Polygeist Transform Utility functions ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// LoopVersionBuilder
//===----------------------------------------------------------------------===//

LoopVersionBuilder::LoopVersionBuilder(LoopLikeOpInterface loop)
    : builder(loop), loop(loop) {}

//===----------------------------------------------------------------------===//
// SCFLoopVersionBuilder
//===----------------------------------------------------------------------===//

SCFLoopVersionBuilder::SCFLoopVersionBuilder(LoopLikeOpInterface loop)
    : LoopVersionBuilder(loop) {}

scf::IfOp SCFLoopVersionBuilder::getIfOp() const {
  assert(ifOp && "Expected valid ifOp");
  return cast<scf::IfOp>(ifOp);
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

//===----------------------------------------------------------------------===//
// SCFLoopGuardBuilder
//===----------------------------------------------------------------------===//

SCFLoopGuardBuilder::SCFLoopGuardBuilder(LoopLikeOpInterface loop)
    : LoopGuardBuilder(), SCFLoopVersionBuilder(loop) {}

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
