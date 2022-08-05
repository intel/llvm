#include "PassDetails.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace polygeist;

namespace {
struct AffineReductionPass : public AffineReductionBase<AffineReductionPass> {
  void runOnOperation() override;
};
} // end namespace.

namespace {

struct AffineForReductionIter : public OpRewritePattern<AffineForOp> {
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  bool isInCurrentAffineFor(Operation *op, AffineForOp forOp) const {
    auto *parentOp = op->getParentOp();
    auto maybeParentFor = dyn_cast_or_null<AffineForOp>(parentOp);
    if (maybeParentFor && maybeParentFor == forOp)
      return true;
    return false;
  }

  bool areInSameAffineFor(AffineLoadOp load, AffineStoreOp store,
                          AffineForOp forOp) const {
    return isInCurrentAffineFor(load.getOperation(), forOp) &&
           isInCurrentAffineFor(store.getOperation(), forOp);
  }

  template <typename T>
  bool haveSameIndices(AffineLoadOp load, T storeOrLoad) const {
    static_assert(llvm::is_one_of<T, AffineLoadOp, AffineStoreOp>::value,
                  "applies to only AffineLoadOp or AffineStoreOp");
    SmallVector<Value, 4> loadIndices(load.getIndices());
    SmallVector<Value, 4> storeOrLoadIndices = storeOrLoad.getIndices();
    if (loadIndices.size() != storeOrLoadIndices.size())
      return false;
    return std::equal(loadIndices.begin(), loadIndices.end(),
                      storeOrLoadIndices.begin());
  }

  template <typename T> bool areCompatible(AffineLoadOp load, T store) const {
    static_assert(llvm::is_one_of<T, AffineLoadOp, AffineStoreOp>::value,
                  "applies to only AffineLoadOp or AffineStoreOp");
    if (load.getMemRef() != store.getMemRef()) {
      return false;
    }
    return haveSameIndices<T>(load, store);
  }

  bool checkDominance(Operation *a, Operation *b) const {
    DominanceInfo dom(a);
    return dom.properlyDominates(a, b);
  }

  bool checkDominance(Operation *a, ArrayRef<Operation *> bs) const {
    bool res = true;
    for (auto *b : bs)
      if (!checkDominance(b, a)) {
        res = false;
        break;
      }
    return res;
  }

  bool hasAllDimsReduced(ArrayRef<Value> indices, Value indVar) const {
    if (llvm::all_of(indices,
                     [indVar](Value index) { return index != indVar; }))
      return true;
    return false;
  }

  bool hasParentOp(Operation *a, Operation *b) const {
    Operation *currOp = a;
    while (Operation *parentOp = currOp->getParentOp()) {
      if (isa<mlir::AffineForOp>(parentOp) && parentOp == b)
        return true;
      currOp = parentOp;
    }
    return false;
  }

  LogicalResult matchAndRewrite(AffineForOp forOp,
                                PatternRewriter &rewriter) const override {

    Block *block = forOp.getBody();
    SmallVector<std::pair<Operation *, Operation *>, 0> candidateOpsInFor;
    SmallVector<SmallVector<Operation *>> loadsInFor;
    block->walk([&](Operation *operation) {
      if (auto load = dyn_cast<AffineLoadOp>(operation)) {
        SmallVector<Value, 4> indices(load.getIndices());
        // skip load if all dimensions are not reduced.
        if (!hasAllDimsReduced(indices, forOp.getInductionVar()))
          return WalkResult::advance();
        // locate possible compatible stores.
        Value memref = load.getMemRef();
        SmallVector<AffineStoreOp> candidateStores;
        SmallVector<Operation *> otherStores;
        SmallVector<Operation *> otherLoads;
        for (auto *user : memref.getUsers()) {
          if (auto store = dyn_cast<AffineStoreOp>(user)) {
            if (areInSameAffineFor(load, store, forOp) &&
                areCompatible<AffineStoreOp>(load, store)) {
              candidateStores.push_back(store);
            } else if (areCompatible<AffineStoreOp>(load, store) &&
                       hasParentOp(store.getOperation(), forOp.getOperation()))
              otherStores.push_back(store);
          }
          if (auto otherLoad = dyn_cast<AffineLoadOp>(user)) {
            if (areCompatible<AffineLoadOp>(load, otherLoad) &&
                load != otherLoad &&
                hasParentOp(otherLoad.getOperation(), forOp.getOperation()))
              otherLoads.push_back(otherLoad);
          }
        }
        // require a single store within the current for. The load must dominate
        // the single store. There must be no other stores in the current for.
        if ((candidateStores.size() == 1) &&
            checkDominance(load.getOperation(), candidateStores[0].getOperation()) &&
            otherStores.size() == 0 /*
            checkDominance(candidateStores[0].getOperation(), otherStores)*/) {
          candidateOpsInFor.push_back(std::make_pair(
              load.getOperation(), candidateStores[0].getOperation()));
          loadsInFor.push_back(otherLoads);
        }
      }
      return WalkResult::advance();
    });

    // no work to do.
    if (!candidateOpsInFor.size())
      return failure();

    // llvm::errs() << "------------\n";
    // llvm::errs() << "#candidateOpsInFor: " << candidateOpsInFor.size() <<
    // "\n";

    /*
     llvm::errs() << "candidateOpsInFor\n";
     for (auto pair : candidateOpsInFor) {
       std::get<0>(pair)->dump();
       std::get<1>(pair)->dump();
     }
     llvm::errs() << "-for-\n";
     */
    // forOp.dump();
    // llvm::errs() << "------------\n";

    // move the load outside the loop. All the load indexes are
    // not used in the current for (see hasAllDimReduced).
    // The load result are passed to the new forOp as iter args.
    SmallVector<Value, 4> newIterArgs;
    llvm::append_range(newIterArgs, forOp.getRegionIterArgs());
    rewriter.setInsertionPoint(forOp);
    for (auto pair : candidateOpsInFor) {
      auto *movedLoad = rewriter.clone(*std::get<0>(pair));
      newIterArgs.push_back(movedLoad->getResult(0));
    }

    // create the for.
    AffineForOp newForOp = rewriter.create<AffineForOp>(
        forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
        forOp.getStep(), newIterArgs);

    // remove load operation inside the for.
    size_t i = 0;
    size_t origNumRegionArgs = forOp.getNumRegionIterArgs();
    for (auto pair : candidateOpsInFor) {
      std::get<0>(pair)->getResult(0).replaceAllUsesWith(
          newForOp.getBody()->getArguments()[i + origNumRegionArgs + 1]);
      rewriter.eraseOp(std::get<0>(pair));
      ++i;
    }

    Block *newBlock = newForOp.getBody();
    Block *oldBlock = forOp.getBody();
    SmallVector<Value, 4> newBlockTransferArgs;
    newBlockTransferArgs.push_back(newForOp.getInductionVar());
    for (size_t i = 0; i < origNumRegionArgs; i++)
      newBlockTransferArgs.push_back(newForOp.getRegionIterArgs()[i]);
    assert(oldBlock->getNumArguments() == newBlockTransferArgs.size() &&
           "unexpected argument size mismatch");
    rewriter.mergeBlocks(oldBlock, newBlock, newBlockTransferArgs);

    auto cloneFilteredTerminator = [&](AffineYieldOp mergedTerminator) {
      SmallVector<Value, 4> newOperands;
      llvm::append_range(newOperands, mergedTerminator.getOperands());
      // store operands are now returned.
      for (auto pair : candidateOpsInFor) {
        newOperands.push_back(std::get<1>(pair)->getOperand(0));
        // rewriter.eraseOp(std::get<1>(pair));
      }
      mergedTerminator.operandsMutable().assign(newOperands);
    };

    auto mergedYieldOp = cast<AffineYieldOp>(newBlock->getTerminator());
    cloneFilteredTerminator(mergedYieldOp);

    // prepare for new yielded value for 'replaceOp'.
    SmallVector<Value, 4> newYieldedRes;
    SmallVector<Value, 4> newRes(newForOp.getResults());
    int additionalRes =
        newForOp.getResults().size() - forOp.getResults().size();
    assert(additionalRes >= 0 && "must be >= 0");
    newRes.insert(newRes.end(), newRes.begin(), newRes.end() - additionalRes);

    // propagate results new forOp to downstream loads if any,
    // otherwise insert a store right after the for. The stored
    // element is the result of the for.
    assert(candidateOpsInFor.size() == loadsInFor.size());
    i = 0;

    DominanceInfo DT;
    PostDominanceInfo PDT;
    for (auto pair : candidateOpsInFor) {
      auto store = cast<AffineStoreOp>(std::get<1>(pair));

      auto loads = loadsInFor[i];
      for (auto *load : loads) {
        if (PDT.postDominates(store, load)) {
          load->getResult(0).replaceAllUsesWith(
              newForOp.getBody()->getArguments()[i + origNumRegionArgs + 1]);
        } else if (DT.dominates(store, load)) {
          load->getResult(0).replaceAllUsesWith(store.getOperand(0));
        } else {

          assert(0 && "illegal behavior");
        }
      }

      rewriter.setInsertionPointAfter(newForOp);
      rewriter.create<AffineStoreOp>(
          newForOp.getLoc(),
          newForOp.getResults()[forOp.getResults().size() + i],
          store.getMemRef(), store.getAffineMap(), store.getIndices());
      rewriter.eraseOp(std::get<1>(pair));
      ++i;
    }

    rewriter.replaceOp(forOp, newYieldedRes);
    return success();
  }
};

} // end namespace.

void AffineReductionPass::runOnOperation() {
  mlir::RewritePatternSet rpl(getOperation()->getContext());
  rpl.add<AffineForReductionIter>(getOperation()->getContext());
  GreedyRewriteConfig config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(rpl), config);
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> detectReductionPass() {
  return std::make_unique<AffineReductionPass>();
}
} // namespace polygeist
} // namespace mlir
