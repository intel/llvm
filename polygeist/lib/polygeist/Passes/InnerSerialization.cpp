#include "PassDetails.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arith;
using namespace polygeist;

namespace {
struct InnerSerialization : public InnerSerializationBase<InnerSerialization> {
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
