//===- AffineReduction.cpp - Affine Reduction Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "polygeist/Passes/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-reduction"

using namespace mlir;
using namespace polygeist;

namespace {
struct AffineReductionPass : public AffineReductionBase<AffineReductionPass> {
  void runOnOperation() override;
};

struct AffineForReductionIter : public OpRewritePattern<AffineForOp> {
  using OpRewritePattern<AffineForOp>::OpRewritePattern;

  /// Returns true if the given operation \p Op is immediately contained in the
  /// \p ForOp loop, and false otherwise.
  bool isInAffineFor(Operation *Op, const AffineForOp ForOp) const {
    assert(Op && "Expecting valid operation");
    auto MaybeParentFor = dyn_cast_or_null<AffineForOp>(Op->getParentOp());
    return (MaybeParentFor && MaybeParentFor == ForOp);
  }

  /// Returns true if the \p Load and \p Store operations are in the given
  /// \p ForOp loop, and false otherwise.
  bool areInSameAffineFor(AffineLoadOp Load, AffineStoreOp Store,
                          const AffineForOp ForOp) const {
    return isInAffineFor(Load, ForOp) && isInAffineFor(Store, ForOp);
  }

  /// Returns true if operation \p A is nested within the \p ForOp loop, and
  /// false otherwise.
  bool isNestedInForOp(Operation *A, AffineForOp &ForOp) const {
    Operation *CurrOp = A;
    while (Operation *ParentOp = CurrOp->getParentOp()) {
      if (ParentOp == ForOp)
        return true;

      CurrOp = ParentOp;
    }

    return false;
  }

  /// Returns true if the given \p Load and \p StoreOrLoad operations have the
  /// same subscript indices, and false otherwise.
  template <typename T>
  bool haveSameIndices(AffineLoadOp Load, T StoreOrLoad) const {
    static_assert(
        llvm::is_one_of<T, AffineLoadOp, AffineStoreOp>::value,
        "template parameter should be 'AffineLoadOp' or 'AffineStoreOp'");
    const SmallVector<Value, 4> LoadIndices(Load.getIndices());
    const SmallVector<Value, 4> StoreOrLoadIndices(StoreOrLoad.getIndices());
    return std::equal(LoadIndices.begin(), LoadIndices.end(),
                      StoreOrLoadIndices.begin(), StoreOrLoadIndices.end());
  }

  /// Returns true if the given \p Load and \p LoadOrStore operations have the
  /// same base operand and the same subscript indices, and false otherwise.
  template <typename T>
  bool areCompatible(AffineLoadOp Load, T StoreOrLoad) const {
    static_assert(
        llvm::is_one_of<T, AffineLoadOp, AffineStoreOp>::value,
        "template parameter should be 'AffineLoadOp' or 'AffineStoreOp'");
    if (Load.getMemRef() != StoreOrLoad.getMemRef())
      return false;

    return haveSameIndices<T>(Load, StoreOrLoad);
  }

  /// Returns true if the operation \p A properly dominates \p B and false
  /// otherwise.
  bool properlyDominates(Operation *A, Operation *B) const {
    DominanceInfo Dom(A);
    return Dom.properlyDominates(A, B);
  }

  /// Returns true if all operations in \P Bs properly dominate operation \p A
  /// and false otherwise.
  bool allProperlyDominate(const ArrayRef<Operation *> Bs, Operation *A) const {
    return llvm::all_of(Bs,
                        [&](Operation *B) { return properlyDominates(B, A); });
  }

  /// Retuns true if none of the indices in \p Indices are \p IndVar, and false
  /// otherwise.
  bool hasAllDimsReduced(const ArrayRef<Value> Indices, Value IndVar) const {
    return llvm::none_of(Indices,
                         [IndVar](Value Index) { return Index == IndVar; });
  }

  /// Represent a pair of affine load and store operations that satisfy all
  /// requirements for a reduction.
  struct CandidateReductionOp {
    CandidateReductionOp(AffineLoadOp &Load, AffineStoreOp &Store,
                         SmallVector<AffineLoadOp> &OtherLoads)
        : Load(Load), Store(Store), OtherLoads(OtherLoads) {
      // TODO: add safety checks here ?
    }

    AffineLoadOp getLoad() const { return Load; }
    AffineStoreOp getStore() const { return Store; }
    const SmallVector<AffineLoadOp> &getOtherLoads() const {
      return OtherLoads;
    }

  private:
    /// The reduction load in the loop nest.
    AffineLoadOp Load;
    /// The reduction store in the same loop nest.
    AffineStoreOp Store;
    /// Compatible loads in the same loop nest.
    SmallVector<AffineLoadOp> OtherLoads;
  };

  /// Collect candidate reduction operations in the given loop \p ForOp.
  /// Candidate reductions are stored into \p CandidateOps, and affine load
  /// operations with the same base operand and subscript indices as the
  /// candidate load are stored into \p LoadsInLoops.
  WalkResult
  collectCandidates(AffineForOp ForOp,
                    SmallVectorImpl<CandidateReductionOp> &CandidateOps) const {
    assert(CandidateOps.empty() && "Expecting an empty vector");

    WalkResult Result = ForOp.getBody()->walk([&](Operation *Op) {
      // We are only interested in affine load operations.
      if (!isa<AffineLoadOp>(Op))
        return WalkResult::advance();

      // Skip the load if any of its subscript indices are the loop induction
      // variable (i.e. the load is not loop invariant).
      auto Load = cast<AffineLoadOp>(Op);
      SmallVector<Value, 4> Indices(Load.getIndices());
      if (!hasAllDimsReduced(Indices, ForOp.getInductionVar()))
        return WalkResult::advance();

      // Locate possible compatible stores (stores that have the same base
      // operand and subscript indices as the load), and collect all other loads
      // that have the same subscript and base symbol.
      SmallVector<AffineStoreOp> CandidateStores;
      SmallVector<AffineLoadOp> OtherLoads;
      for (Operation *User : Load.getMemRef().getUsers()) {
        bool InterruptWalk = false;
        TypeSwitch<Operation *>(User)
            .Case<AffineLoadOp>([&](auto OtherLoad) {
              if (areCompatible(Load, OtherLoad) && Load != OtherLoad &&
                  isNestedInForOp(OtherLoad, ForOp))
                OtherLoads.push_back(OtherLoad);
            })
            .Case<AffineStoreOp>([&](auto Store) {
              if (areCompatible(Load, Store)) {
                if (areInSameAffineFor(Load, Store, ForOp))
                  CandidateStores.push_back(Store);
                else if (isNestedInForOp(Store, ForOp))
                  InterruptWalk = true;
              }
            });

        if (InterruptWalk)
          return WalkResult::interrupt();
      }

      // Require a single store within the current loop. The load must dominate
      // the single store.
      if (CandidateStores.size() != 1 ||
          !properlyDominates(Load, CandidateStores[0]))
        return WalkResult::interrupt();

      CandidateOps.push_back(
          CandidateReductionOp(Load, CandidateStores[0], OtherLoads));

      return WalkResult::advance();
    });

    return Result;
  }

  LogicalResult matchAndRewrite(AffineForOp ForOp,
                                PatternRewriter &Rewriter) const override {
    SmallVector<CandidateReductionOp> CandidateOps;
    WalkResult Result = collectCandidates(ForOp, CandidateOps);
    if (Result.wasInterrupted() || CandidateOps.empty())
      return failure();

    LLVM_DEBUG({
      llvm::dbgs() << "------------\n";
      llvm::dbgs() << "Found " << CandidateOps.size()
                   << " candidate reduction operations in loop:\n";
      for (CandidateReductionOp &CandidateOp : CandidateOps) {
        llvm::dbgs().indent(2) << "Load: ";
        CandidateOp.getLoad()->dump();
        llvm::dbgs().indent(2) << "Store: ";
        CandidateOp.getStore()->dump();
      }

      llvm::dbgs() << "Loop:\n";
      ForOp.dump();
      llvm::dbgs() << "------------\n";
    });

    // Move the load outside the loop(recall that the load indexes are not used
    // in the current for (see hasAllDimReduced)). The load result is passed to
    // the new forOp as iter args.
    SmallVector<Value, 4> NewIterArgs;
    llvm::append_range(NewIterArgs, ForOp.getRegionIterArgs());
    Rewriter.setInsertionPoint(ForOp);
    for (CandidateReductionOp &Op : CandidateOps) {
      Operation *MovedLoad = Rewriter.clone(*Op.getLoad());
      NewIterArgs.push_back(MovedLoad->getResult(0));
    }

    // Create the new loop.
    AffineForOp NewForOp = Rewriter.create<AffineForOp>(
        ForOp.getLoc(), ForOp.getLowerBoundOperands(), ForOp.getLowerBoundMap(),
        ForOp.getUpperBoundOperands(), ForOp.getUpperBoundMap(),
        ForOp.getStep(), NewIterArgs);

    // Remove the load operation inside the new for loop.
    size_t ArgNo = 0, OrigNumRegionArgs = ForOp.getNumRegionIterArgs();
    for (CandidateReductionOp &Op : CandidateOps) {
      Op.getLoad()->getResult(0).replaceAllUsesWith(
          NewForOp.getBody()->getArguments()[ArgNo + OrigNumRegionArgs + 1]);
      Rewriter.eraseOp(Op.getLoad());
      ++ArgNo;
    }

    Block *NewBlock = NewForOp.getBody(), *OldBlock = ForOp.getBody();
    SmallVector<Value, 4> NewBlockTransferArgs;
    NewBlockTransferArgs.push_back(NewForOp.getInductionVar());
    for (size_t ArgNo = 0; ArgNo < OrigNumRegionArgs; ++ArgNo)
      NewBlockTransferArgs.push_back(NewForOp.getRegionIterArgs()[ArgNo]);
    assert(OldBlock->getNumArguments() == NewBlockTransferArgs.size() &&
           "unexpected argument size mismatch");
    Rewriter.mergeBlocks(OldBlock, NewBlock, NewBlockTransferArgs);

    auto CloneFilteredTerminator =
        [CandidateOps](AffineYieldOp MergedTerminator) {
          SmallVector<Value, 4> NewOperands;
          llvm::append_range(NewOperands, MergedTerminator.getOperands());
          // store operands are now returned.
          for (const CandidateReductionOp &Op : CandidateOps)
            NewOperands.push_back(Op.getStore()->getOperand(0));
          MergedTerminator.getOperandsMutable().assign(NewOperands);
        };

    auto MergedYieldOp = cast<AffineYieldOp>(NewBlock->getTerminator());
    CloneFilteredTerminator(MergedYieldOp);

    // Prepare for new yielded value for 'replaceOp'.
    SmallVector<Value, 4> NewYieldedRes, NewRes(NewForOp.getResults());
    size_t AdditionalRes =
        NewForOp.getResults().size() - ForOp.getResults().size();
    assert(AdditionalRes >= 0 && "must be >= 0");
    NewRes.insert(NewRes.end(), NewRes.begin(), NewRes.end() - AdditionalRes);

    // Propagate results new forOp to downstream loads if any, otherwise insert
    // a store right after the for. The stored element is the result of the for.
    {
      DominanceInfo DT;
      PostDominanceInfo PDT;
      size_t ArgNo = 0;
      for (CandidateReductionOp &Op : CandidateOps) {
        for (Operation *Load : Op.getOtherLoads()) {
          if (PDT.postDominates(Op.getStore(), Load))
            Load->getResult(0).replaceAllUsesWith(
                NewForOp.getBody()
                    ->getArguments()[ArgNo + OrigNumRegionArgs + 1]);
          else if (DT.dominates(Op.getStore(), Load))
            Load->getResult(0).replaceAllUsesWith(Op.getStore().getOperand(0));
          else
            llvm_unreachable("illegal behavior");
        }

        Rewriter.setInsertionPointAfter(NewForOp);
        Rewriter.create<AffineStoreOp>(
            NewForOp.getLoc(),
            NewForOp.getResults()[ForOp.getResults().size() + ArgNo],
            Op.getStore().getMemRef(), Op.getStore().getAffineMap(),
            Op.getStore().getIndices());
        Rewriter.eraseOp(Op.getStore());
        ++ArgNo;
      }
    }

    Rewriter.replaceOp(ForOp, NewYieldedRes);
    return success();
  }
};

} // end namespace.

void AffineReductionPass::runOnOperation() {
  RewritePatternSet Rpl(getOperation()->getContext());
  Rpl.add<AffineForReductionIter>(getOperation()->getContext());
  GreedyRewriteConfig Config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(Rpl), Config);
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> detectReductionPass() {
  return std::make_unique<AffineReductionPass>();
}
} // namespace polygeist
} // namespace mlir
