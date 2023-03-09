//===- RaiseToAffine.cpp - Polygeist CFG to Affine Pass ---------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "raise-to-affine"
#define REPORT_DEBUG_TYPE DEBUG_TYPE "-report"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_SCFRAISETOAFFINE
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;
using namespace mlir::arith;
using namespace polygeist;

namespace {
struct RaiseSCFToAffine
    : public mlir::polygeist::impl::SCFRaiseToAffineBase<RaiseSCFToAffine> {
  void runOnOperation() override;
};
} // namespace

/// Attempt to transform an scf.ForOp loop to an AffineForOp loop.
struct ForOpRaising : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  AffineMap getMultiSymbolIdentity(Builder &B, unsigned rank) const {
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i)
      dimExprs.push_back(B.getAffineSymbolExpr(i));
    return AffineMap::get(/*dimCount=*/0, /*symbolCount=*/rank, dimExprs,
                          B.getContext());
  }
  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const final {
    LLVM_DEBUG({
      llvm::dbgs() << "\n----------------\n";
      loop.print(llvm::dbgs() << "Original loop:\n");
      llvm::dbgs() << "\n";
    });

    if (!isValidSymbol(loop.getStep())) {
      LLVM_DEBUG(llvm::dbgs() << "Failed: step is not valid\n");
      return failure();
    }

    auto getBounds = [](TypedValue<IndexType> bound, CmpIPredicate cmpPred,
                        SmallVectorImpl<Value> &bounds) {
      SmallVector<Value> todo = {bound};
      while (todo.size()) {
        Value cur = todo.back();
        todo.pop_back();
        if (isValidIndex(cur)) {
          bounds.push_back(cur);
          continue;
        }
        if (auto selOp = cur.getDefiningOp<SelectOp>()) {
          if (auto cmp = selOp.getCondition().getDefiningOp<CmpIOp>()) {
            if (cmp.getLhs() == selOp.getTrueValue() &&
                cmp.getRhs() == selOp.getFalseValue() &&
                cmp.getPredicate() == cmpPred) {
              todo.push_back(cmp.getLhs());
              todo.push_back(cmp.getRhs());
              continue;
            }
          }
        }
        return false;
      }
      return true;
    };

    SmallVector<Value> lbs;
    if (!getBounds(loop.getLowerBound(), CmpIPredicate::sge, lbs)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed: cannot get lower bounds\n");
      return failure();
    }

    SmallVector<Value> ubs;
    if (!getBounds(loop.getUpperBound(), CmpIPredicate::sle, ubs)) {
      LLVM_DEBUG(llvm::dbgs() << "Failed: cannot get upper bounds\n");
      return failure();
    }

    bool rewrittenStep = false;
    if (!loop.getStep().getDefiningOp<ConstantIndexOp>()) {
      if (ubs.size() != 1 || lbs.size() != 1) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed: more than one lower/upper bounds\n");
        return failure();
      }
      ubs[0] = rewriter.create<DivUIOp>(
          loop.getLoc(),
          rewriter.create<SubIOp>(loop.getLoc(), loop.getUpperBound(),
                                  loop.getLowerBound()),
          loop.getStep());
      lbs[0] = rewriter.create<ConstantIndexOp>(loop.getLoc(), 0);
      rewrittenStep = true;
    }

    DominanceInfo DI(getAffineScope(loop)->getParentOp());
    OpBuilder builder(loop);

    AffineMap lbMap = getMultiSymbolIdentity(builder, lbs.size());
    fully2ComposeAffineMapAndOperands(rewriter, &lbMap, &lbs, DI);
    canonicalizeMapAndOperands(&lbMap, &lbs);
    lbMap = removeDuplicateExprs(lbMap);

    AffineMap ubMap = getMultiSymbolIdentity(builder, ubs.size());
    fully2ComposeAffineMapAndOperands(rewriter, &ubMap, &ubs, DI);
    canonicalizeMapAndOperands(&ubMap, &ubs);
    ubMap = removeDuplicateExprs(ubMap);

    ConstantIndexOp cstOp = loop.getStep().getDefiningOp<ConstantIndexOp>();
    AffineForOp affineLoop = rewriter.create<AffineForOp>(
        loop.getLoc(), lbs, lbMap, ubs, ubMap, cstOp ? cstOp.value() : 1,
        loop.getIterOperands());

    auto mergedYieldOp =
        cast<scf::YieldOp>(loop.getRegion().front().getTerminator());

    Block &newBlock = affineLoop.getRegion().front();

    // The terminator is added if the iterator args are not provided.
    // see the ::build method.
    if (affineLoop.getNumIterOperands() == 0) {
      Operation *affineYieldOp = newBlock.getTerminator();
      rewriter.eraseOp(affineYieldOp);
    }

    SmallVector<Value> vals;
    rewriter.setInsertionPointToStart(&affineLoop.getRegion().front());
    for (Value arg : affineLoop.getRegion().front().getArguments()) {
      if (rewrittenStep && arg == affineLoop.getInductionVar())
        arg = rewriter.create<AddIOp>(
            loop.getLoc(), loop.getLowerBound(),
            rewriter.create<MulIOp>(loop.getLoc(), arg, loop.getStep()));
      vals.push_back(arg);
    }
    assert(vals.size() == loop.getRegion().front().getNumArguments());
    rewriter.mergeBlocks(&loop.getRegion().front(),
                         &affineLoop.getRegion().front(), vals);

    rewriter.setInsertionPoint(mergedYieldOp);
    rewriter.create<AffineYieldOp>(mergedYieldOp.getLoc(),
                                   mergedYieldOp.getOperands());
    rewriter.eraseOp(mergedYieldOp);

    DEBUG_WITH_TYPE(REPORT_DEBUG_TYPE, {
      llvm::dbgs()
          << "RaiseSCFForToAffine: raised scf::ForOp to AffineForOp in: "
          << loop->getParentOfType<FunctionOpInterface>().getName() << "\n";
    });

    LLVM_DEBUG({
      affineLoop.print(llvm::dbgs() << "\nNew Loop:\n");
      llvm::dbgs() << "\n----------------\n";
    });

    rewriter.replaceOp(loop, affineLoop.getResults());

    return success();
  }
};

struct ParallelOpRaising : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  void canonicalizeLoopBounds(PatternRewriter &rewriter,
                              AffineParallelOp forOp) const {
    SmallVector<Value, 4> lbOperands(forOp.getLowerBoundsOperands());
    SmallVector<Value, 4> ubOperands(forOp.getUpperBoundsOperands());

    auto lbMap = forOp.getLowerBoundsMap();
    auto ubMap = forOp.getUpperBoundsMap();

    auto *scope = getAffineScope(forOp)->getParentOp();
    DominanceInfo DI(scope);

    fully2ComposeAffineMapAndOperands(rewriter, &lbMap, &lbOperands, DI);
    canonicalizeMapAndOperands(&lbMap, &lbOperands);

    fully2ComposeAffineMapAndOperands(rewriter, &ubMap, &ubOperands, DI);
    canonicalizeMapAndOperands(&ubMap, &ubOperands);

    forOp.setLowerBounds(lbOperands, lbMap);
    forOp.setUpperBounds(ubOperands, ubMap);
  }

  LogicalResult matchAndRewrite(scf::ParallelOp loop,
                                PatternRewriter &rewriter) const final {
    OpBuilder builder(loop);

    if (loop.getResults().size())
      return failure();

    if (!llvm::all_of(loop.getLowerBound(), isValidIndex))
      return failure();

    if (!llvm::all_of(loop.getUpperBound(), isValidIndex))
      return failure();

    SmallVector<int64_t> steps;
    for (auto step : loop.getStep())
      if (auto cst = step.getDefiningOp<ConstantIndexOp>())
        steps.push_back(cst.value());
      else
        return failure();

    ArrayRef<AtomicRMWKind> reductions;
    SmallVector<AffineMap> bounds;
    for (size_t i = 0; i < loop.getLowerBound().size(); i++)
      bounds.push_back(AffineMap::get(
          /*dimCount=*/0, /*symbolCount=*/loop.getLowerBound().size(),
          builder.getAffineSymbolExpr(i)));
    AffineParallelOp affineLoop = rewriter.create<AffineParallelOp>(
        loop.getLoc(), loop.getResultTypes(), reductions, bounds,
        loop.getLowerBound(), bounds, loop.getUpperBound(),
        steps); //, loop.getInitVals());

    canonicalizeLoopBounds(rewriter, affineLoop);

    auto mergedYieldOp =
        cast<scf::YieldOp>(loop.getRegion().front().getTerminator());

    Block &newBlock = affineLoop.getRegion().front();

    // The terminator is added if the iterator args are not provided.
    // see the ::build method.
    if (affineLoop.getResults().size() == 0) {
      auto *affineYieldOp = newBlock.getTerminator();
      rewriter.eraseOp(affineYieldOp);
    }

    SmallVector<Value> vals;
    for (Value arg : affineLoop.getRegion().front().getArguments())
      vals.push_back(arg);

    rewriter.mergeBlocks(&loop.getRegion().front(),
                         &affineLoop.getRegion().front(), vals);

    rewriter.setInsertionPoint(mergedYieldOp);
    rewriter.create<AffineYieldOp>(mergedYieldOp.getLoc(),
                                   mergedYieldOp.getOperands());
    rewriter.eraseOp(mergedYieldOp);

    rewriter.replaceOp(loop, affineLoop.getResults());

    return success();
  }
};

void RaiseSCFToAffine::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.insert<ForOpRaising, ParallelOpRaising>(&getContext());

  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                     config);
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRaiseSCFToAffinePass() {
  return std::make_unique<RaiseSCFToAffine>();
}
} // namespace polygeist
} // namespace mlir
