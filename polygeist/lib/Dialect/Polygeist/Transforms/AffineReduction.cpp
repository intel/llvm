//===- AffineReduction.cpp - Affine Reduction Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "affine-reduction"
#define REPORT_DEBUG_TYPE DEBUG_TYPE "-report"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_AFFINEREDUCTION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

namespace {
class AffineReductionPass
    : public mlir::polygeist::impl::AffineReductionBase<AffineReductionPass> {
public:
  void runOnOperation() override;
};

/// Represent a pair of affine load and store operations that satisfy all
/// requirements for a reduction.
class ReductionOp {
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ReductionOp &);

public:
  ReductionOp(AffineLoadOp &Load, AffineStoreOp &Store,
              SmallVector<AffineLoadOp> &OtherLoads)
      : Load(Load), Store(Store), OtherLoads(OtherLoads) {}

  AffineLoadOp getLoad() const { return Load; }
  AffineStoreOp getStore() const { return Store; }
  ArrayRef<AffineLoadOp> getOtherLoads() const { return OtherLoads; }

private:
  /// The reduction load in the loop nest.
  AffineLoadOp Load;
  /// The reduction store in the same loop nest.
  AffineStoreOp Store;
  /// Compatible loads in the same loop nest.
  SmallVector<AffineLoadOp> OtherLoads;
};

[[maybe_unused]] inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                                      const ReductionOp &CR) {
  OS.indent(2) << "Load: ";
  CR.getLoad()->dump();
  OS.indent(2) << "Store: ";
  CR.getStore()->dump();
  OS.indent(2) << "Other Loads: ";
  for (const AffineLoadOp &Op : CR.getOtherLoads())
    Op->dump();

  return OS;
}

class AffineForReductionIter : public OpRewritePattern<AffineForOp> {
  Pass::Statistic &numReductionsDetected;

public:
  AffineForReductionIter(MLIRContext *context,
                         Pass::Statistic &numReductionsDetected)
      : OpRewritePattern<AffineForOp>(context),
        numReductionsDetected(numReductionsDetected) {}

  /// Returns true if the given operation \p Op is immediately contained in the
  /// \p ForOp loop, and false otherwise.
  bool isInAffineFor(Operation *Op, const AffineForOp ForOp) const {
    assert(Op && "Expecting valid operation");
    return Op->getParentOp() == ForOp;
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
  template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                            T, AffineLoadOp, AffineStoreOp>::value>>
  bool haveSameIndices(AffineLoadOp Load, T StoreOrLoad) const {
    const Operation::operand_range LoadIndices = Load.getIndices();
    const Operation::operand_range StoreOrLoadIndices =
        StoreOrLoad.getIndices();
    return std::equal(LoadIndices.begin(), LoadIndices.end(),
                      StoreOrLoadIndices.begin(), StoreOrLoadIndices.end());
  }

  /// Returns true if the given \p Load and \p LoadOrStore operations have the
  /// same base operand and the same subscript indices, and false otherwise.
  template <typename T, typename = std::enable_if<llvm::is_one_of<
                            T, AffineLoadOp, AffineStoreOp>::value>>
  bool areCompatible(AffineLoadOp Load, T StoreOrLoad) const {
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

  /// Returns true if all operations in \p Bs properly dominate operation \p A
  /// and false otherwise.
  bool allProperlyDominate(ArrayRef<Operation *> Bs, Operation *A) const {
    return llvm::all_of(Bs,
                        [&](Operation *B) { return properlyDominates(B, A); });
  }

  /// Collect candidate reduction operations in the given loop \p ForOp.
  /// Candidate reductions are stored into \p CandidateOps, and affine load
  /// operations with the same base operand and subscript indices as the
  /// candidate load are stored into \p LoadsInLoops.
  WalkResult
  collectCandidates(AffineForOp ForOp,
                    SmallVectorImpl<ReductionOp> &CandidateOps) const {
    assert(CandidateOps.empty() && "Expecting an empty vector");

    LLVM_DEBUG({
      llvm::dbgs() << "\n---------- Affine Reduction Detection ----------\n";
      ForOp.dump();
      llvm::dbgs() << "\n";
    });

    WalkResult Result = ForOp.getBody()->walk([&](AffineLoadOp Load) {
      LLVM_DEBUG(llvm::dbgs() << "Load: " << Load << "\n");

      // Ensure operands are loop invariant.
      if (llvm::any_of(Load.getOperands(), [&ForOp](Value value) {
            return !ForOp.isDefinedOutsideOfLoop(value);
          })) {
        LLVM_DEBUG({
          llvm::dbgs().indent(2) << "Skip: operand(s) not loop invariant\n";
        });
        return WalkResult::advance();
      }

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
                  ForOp->isProperAncestor(OtherLoad))
                OtherLoads.push_back(OtherLoad);
            })
            .Case<AffineStoreOp>([&](auto Store) {
              if (areCompatible(Load, Store)) {
                if (areInSameAffineFor(Load, Store, ForOp))
                  CandidateStores.push_back(Store);
                else if (ForOp->isProperAncestor(Store)) {
                  LLVM_DEBUG(llvm::dbgs().indent(2)
                             << "Interrupting - found incompatible store: "
                             << Store << "\n");
                  InterruptWalk = true;
                }
              }
            });

        if (InterruptWalk)
          return WalkResult::interrupt();
      }

      if (CandidateStores.empty()) {
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "Skip - no compatible store found\n");
        return WalkResult::advance();
      }

      // Require a single store within the current loop.
      if (CandidateStores.size() != 1) {
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "Interrupting - more than one compatible store found\n");
        return WalkResult::interrupt();
      }

      // The load must dominate the single store.
      if (!properlyDominates(Load, CandidateStores[0])) {
        LLVM_DEBUG(llvm::dbgs().indent(2)
                   << "Interrupting - store doesn't dominate load: "
                   << CandidateStores[0] << "\n");
        return WalkResult::interrupt();
      }

      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "Found compatible store: " << CandidateStores[0] << "\n");

      CandidateOps.push_back(ReductionOp(Load, CandidateStores[0], OtherLoads));

      return WalkResult::advance();
    });

    return Result;
  }

  LogicalResult matchAndRewrite(AffineForOp ForOp,
                                PatternRewriter &Rewriter) const override {
    SmallVector<ReductionOp> ReductionOps;
    WalkResult Result = collectCandidates(ForOp, ReductionOps);
    if (Result.wasInterrupted() || ReductionOps.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "No reduction found\n";
        llvm::dbgs() << "------------------------------------------------\n";
      });
      return failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "\nFound " << ReductionOps.size() << " reduction(s):\n";
      for (ReductionOp &Op : ReductionOps)
        llvm::dbgs() << Op << "\n";
      llvm::dbgs() << "\n";
    });

    // Move the load outside the loop (recall that the load is loop invariant).
    // The load result is passed to the new 'forOp' as iter args.
    SmallVector<Value, 4> NewIterArgs(ForOp.getRegionIterArgs());
    Rewriter.setInsertionPoint(ForOp);
    for (const ReductionOp &Op : ReductionOps) {
      Operation *MovedLoad = Rewriter.clone(*Op.getLoad());
      LLVM_DEBUG(llvm::dbgs() << "New Loop:\n" << *MovedLoad << "\n");
      NewIterArgs.push_back(MovedLoad->getResult(0));
    }

    // Create the new loop.
    auto NewForOp = Rewriter.create<AffineForOp>(
        ForOp.getLoc(), ForOp.getLowerBoundOperands(), ForOp.getLowerBoundMap(),
        ForOp.getUpperBoundOperands(), ForOp.getUpperBoundMap(),
        ForOp.getStep(), NewIterArgs);

    // Remove the load operation inside the new loop.
    size_t ArgNo = 0, OrigNumRegionArgs = ForOp.getNumRegionIterArgs();
    for (const ReductionOp &Op : ReductionOps) {
      Op.getLoad()->getResult(0).replaceAllUsesWith(
          NewForOp.getBody()->getArguments()[ArgNo + OrigNumRegionArgs + 1]);
      Rewriter.eraseOp(Op.getLoad());
      ++ArgNo;
    }

    Block *NewBlock = NewForOp.getBody(), *OldBlock = ForOp.getBody();
    SmallVector<Value, 4> NewBlockTransferArgs;
    NewBlockTransferArgs.push_back(NewForOp.getInductionVar());
    const Block::BlockArgListType &IterArgs = NewForOp.getRegionIterArgs();
    for (size_t ArgNo = 0; ArgNo < OrigNumRegionArgs; ++ArgNo)
      NewBlockTransferArgs.push_back(IterArgs[ArgNo]);
    assert(OldBlock->getNumArguments() == NewBlockTransferArgs.size() &&
           "unexpected argument size mismatch");
    Rewriter.mergeBlocks(OldBlock, NewBlock, NewBlockTransferArgs);

    auto CloneFilteredTerminator =
        [ReductionOps](AffineYieldOp MergedTerminator) {
          SmallVector<Value, 4> NewOperands;
          llvm::append_range(NewOperands, MergedTerminator.getOperands());
          // store operands are now returned.
          for (const ReductionOp &Op : ReductionOps)
            NewOperands.push_back(Op.getStore()->getOperand(0));
          MergedTerminator.getOperandsMutable().assign(NewOperands);
        };

    auto MergedYieldOp = cast<AffineYieldOp>(NewBlock->getTerminator());
    CloneFilteredTerminator(MergedYieldOp);

    // Prepare for new yielded value for 'replaceOp'.
    SmallVector<Value, 4> NewYieldedRes, NewRes(NewForOp.getResults());
    const unsigned NewForOpNumRes{NewForOp.getNumResults()};
    const unsigned ForOpNumRes{ForOp.getNumResults()};
    assert(NewForOpNumRes >= ForOpNumRes &&
           "new for must cannot have less arguments than old one");
    const size_t AdditionalRes{NewForOpNumRes - ForOpNumRes};
    NewRes.insert(NewRes.end(), NewRes.begin(), NewRes.end() - AdditionalRes);

    // Propagate results new forOp to downstream loads if any, otherwise insert
    // a store right after the for. The stored element is the result of the for.
    {
      DominanceInfo DT;
      PostDominanceInfo PDT;
      size_t ArgNo = 0;
      SmallVector<AffineStoreOp> NewStores;

      for (ReductionOp &Op : ReductionOps) {
        for (const AffineLoadOp &Load : Op.getOtherLoads()) {
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
        auto Store = Rewriter.create<AffineStoreOp>(
            NewForOp.getLoc(),
            NewForOp.getResults()[ForOp.getResults().size() + ArgNo],
            Op.getStore().getMemRef(), Op.getStore().getAffineMap(),
            Op.getStore().getIndices());
        NewStores.push_back(Store);
        Rewriter.eraseOp(Op.getStore());
        ++ArgNo;
      }

      LLVM_DEBUG({
        NewForOp.dump();
        for (AffineStoreOp &Store : NewStores)
          Store.dump();
        llvm::dbgs() << "------------------------------------------------\n";
      });
    }

    numReductionsDetected += ReductionOps.size();

    [[maybe_unused]] auto getParentFunction = [](LoopLikeOpInterface loop) {
      Operation *parentOp = loop;
      do {
        parentOp = parentOp->getParentOp();
      } while (parentOp && !isa<func::FuncOp>(parentOp));
      assert(parentOp && "Failed to find parent function");
      return cast<func::FuncOp>(parentOp);
    };

    DEBUG_WITH_TYPE(
        REPORT_DEBUG_TYPE, if (!ReductionOps.empty()) {
          llvm::dbgs() << "AffineReduction: detected " << ReductionOps.size()
                       << " reduction(s) in: "
                       << getParentFunction(ForOp).getSymName() << "\n";
        });

    Rewriter.replaceOp(ForOp, NewYieldedRes);
    return success();
  }
};

} // end namespace.

void AffineReductionPass::runOnOperation() {
  RewritePatternSet RPS(getOperation()->getContext());
  RPS.add<AffineForReductionIter>(getOperation()->getContext(), numReductions);
  GreedyRewriteConfig Config;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(RPS), Config);
}

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> detectReductionPass() {
  return std::make_unique<AffineReductionPass>();
}
} // namespace polygeist
} // namespace mlir
