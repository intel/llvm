//===- InnerSerialization.cpp - Inner Serialization Pass --------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_INNERSERIALIZATION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

namespace {
struct InnerSerialization
    : public mlir::polygeist::impl::InnerSerializationBase<InnerSerialization> {
  void runOnOperation() override;
};
} // namespace

struct ParSerialize : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp nextParallel,
                                PatternRewriter &rewriter) const override {
    if (!(nextParallel->getParentOfType<scf::ParallelOp>()
          // || nextParallel->getParentOfType<AffineParallelOp>()
          ))
      return failure();

    SmallVector<Value> inds;
    scf::ForOp last = nullptr;
    for (auto tup :
         llvm::zip(nextParallel.getLowerBound(), nextParallel.getUpperBound(),
                   nextParallel.getStep(), nextParallel.getInductionVars())) {
      last =
          rewriter.create<scf::ForOp>(nextParallel.getLoc(), std::get<0>(tup),
                                      std::get<1>(tup), std::get<2>(tup));
      inds.push_back(last.getInductionVar());
      rewriter.setInsertionPointToStart(last.getBody());
    }
    rewriter.eraseOp(last.getBody()->getTerminator());
    rewriter.mergeBlocks(&nextParallel.getRegion().front(), last.getBody(),
                         inds);

    rewriter.eraseOp(nextParallel);
    return success();
  }
};

void InnerSerialization::runOnOperation() {
  mlir::RewritePatternSet rpl(getOperation()->getContext());
  rpl.add<ParSerialize>(getOperation()->getContext());
  GreedyRewriteConfig config;
  config.maxIterations = 47;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
}

std::unique_ptr<Pass> mlir::polygeist::createInnerSerializationPass() {
  return std::make_unique<InnerSerialization>();
}
