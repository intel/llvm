//===--- OpenMPOpt.cpp - OpenMP Support -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include "polygeist/Ops.h"
#include "polygeist/Passes/Passes.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace mlir;
using namespace mlir::func;
using namespace mlir::arith;
using namespace polygeist;

namespace {
struct OpenMPOpt : public OpenMPOptPassBase<OpenMPOpt> {
  void runOnOperation() override;
};
} // namespace

/// Merge any consecutive parallel's
///
///    omp.parallel {
///       codeA();
///    }
///    omp.parallel {
///       codeB();
///    }
///
///  becomes
///
///    omp.parallel {
///       codeA();
///       omp.barrier
///       codeB();
///    }
bool isReadOnly(Operation *Op) {
  bool HasRecursiveEffects = Op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (HasRecursiveEffects) {
    for (Region &Region : Op->getRegions()) {
      for (auto &Block : Region) {
        for (auto &NestedOp : Block)
          if (!isReadOnly(&NestedOp))
            return false;
      }
    }
    return true;
  }

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto EffectInterface = dyn_cast<MemoryEffectOpInterface>(Op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> Effects;
    EffectInterface.getEffects(Effects);
    if (!llvm::all_of(Effects, [](const MemoryEffects::EffectInstance &It) {
          return isa<MemoryEffects::Read>(It.getEffect());
        })) {
      return false;
    }
    return true;
  }
  return false;
}

bool mayReadFrom(Operation *Op, Value Val) {
  bool HasRecursiveEffects = Op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (HasRecursiveEffects) {
    for (Region &Region : Op->getRegions()) {
      for (auto &Block : Region) {
        for (auto &NestedOp : Block)
          if (mayReadFrom(&NestedOp, Val))
            return true;
      }
    }
    return false;
  }

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto EffectInterface = dyn_cast<MemoryEffectOpInterface>(Op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> Effects;
    EffectInterface.getEffects(Effects);
    for (auto It : Effects) {
      if (!isa<MemoryEffects::Read>(It.getEffect()))
        continue;
      if (mayAlias(It, Val))
        return true;
    }
    return false;
  }
  return true;
}

Value getBase(Value V);
bool isStackAlloca(Value V);
bool isCaptured(Value V, Operation *PotentialUser = nullptr,
                bool *SeenUse = nullptr);

bool mayWriteTo(Operation *Op, Value Val, bool IgnoreBarrier) {
  bool HasRecursiveEffects = Op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  if (HasRecursiveEffects) {
    for (Region &Region : Op->getRegions()) {
      for (auto &Block : Region) {
        for (auto &NestedOp : Block)
          if (mayWriteTo(&NestedOp, Val, IgnoreBarrier))
            return true;
      }
    }
    return false;
  }

  if (IgnoreBarrier && isa<polygeist::BarrierOp>(Op))
    return false;

  // If the op has memory effects, try to characterize them to see if the op
  // is trivially dead here.
  if (auto EffectInterface = dyn_cast<MemoryEffectOpInterface>(Op)) {
    // Check to see if this op either has no effects, or only allocates/reads
    // memory.
    SmallVector<MemoryEffects::EffectInstance, 1> Effects;
    EffectInterface.getEffects(Effects);
    for (auto It : Effects) {
      if (!isa<MemoryEffects::Write>(It.getEffect()))
        continue;
      if (mayAlias(It, Val))
        return true;
    }
    return false;
  }

  // Calls which do not use a derived pointer of a known alloca, which is not
  // captured can not write to said memory.
  if (isa<LLVM::CallOp, func::CallOp>(Op)) {
    auto Base = getBase(Val);
    bool SeenUse = false;
    if (isStackAlloca(Base) && !isCaptured(Base, Op, &SeenUse) && !SeenUse)
      return false;
  }
  return true;
}

struct CombineParallel : public OpRewritePattern<omp::ParallelOp> {
  using OpRewritePattern<omp::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::ParallelOp NextParallel,
                                PatternRewriter &Rewriter) const override {
    Block *Parent = NextParallel->getBlock();
    if (NextParallel == &Parent->front())
      return failure();

    // Only attempt this if there is another parallel within the function, which
    // is not contained within this operation.
    bool NonContained = false;
    NextParallel->getParentOfType<FuncOp>()->walk([&](omp::ParallelOp Other) {
      if (!NextParallel->isAncestor(Other))
        NonContained = true;
    });
    if (!NonContained)
      return failure();

    omp::ParallelOp PrevParallel;
    SmallVector<Operation *> PrevOps;

    bool Changed = false;

    for (Operation *PrevOp = NextParallel->getPrevNode(); 1;) {
      if ((PrevParallel = dyn_cast<omp::ParallelOp>(PrevOp)))
        break;
      // We can move this into the parallel if it only reads
      if (isReadOnly(PrevOp) &&
          llvm::all_of(PrevOp->getResults(), [NextParallel](Value V) {
            return llvm::all_of(V.getUsers(), [NextParallel](Operation *User) {
              return NextParallel->isAncestor(User);
            });
          })) {
        auto *PrevIter =
            (PrevOp == &Parent->front()) ? nullptr : PrevOp->getPrevNode();
        Rewriter.setInsertionPointToStart(&NextParallel.getRegion().front());
        auto *Replacement = Rewriter.clone(*PrevOp);
        Rewriter.replaceOp(PrevOp, Replacement->getResults());
        Changed = true;
        if (!PrevIter)
          return success();
        PrevOp = PrevIter;
        continue;
      }
      return success(Changed);
    }

    // TODO analyze if already has barrier at the end
    bool PreBarrier = false;
    Rewriter.setInsertionPointToEnd(&PrevParallel.getRegion().front());
    if (!PreBarrier)
      Rewriter.replaceOpWithNewOp<omp::BarrierOp>(
          PrevParallel.getRegion().front().getTerminator(), TypeRange());
    Rewriter.mergeBlocks(&NextParallel.getRegion().front(),
                         &PrevParallel.getRegion().front());
    Rewriter.eraseOp(NextParallel);
    return success();
  }
};

struct ParallelForInterchange : public OpRewritePattern<omp::ParallelOp> {
  using OpRewritePattern<omp::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(omp::ParallelOp NextParallel,
                                PatternRewriter &Rewriter) const override {
    Block *Parent = NextParallel->getBlock();
    if (Parent->getOperations().size() != 2)
      return failure();

    auto PrevFor = dyn_cast<scf::ForOp>(NextParallel->getParentOp());
    if (!PrevFor || PrevFor->getResults().size())
      return failure();

    Rewriter.setInsertionPoint(PrevFor);
    auto NewParallel = Rewriter.create<omp::ParallelOp>(NextParallel.getLoc());
    Rewriter.createBlock(&NewParallel.getRegion());
    Rewriter.setInsertionPointToEnd(&NewParallel.getRegion().front());
    auto NewFor =
        Rewriter.create<scf::ForOp>(PrevFor.getLoc(), PrevFor.getLowerBound(),
                                    PrevFor.getUpperBound(), PrevFor.getStep());
    auto *Yield = NextParallel.getRegion().front().getTerminator();
    NewFor.getRegion().takeBody(PrevFor.getRegion());
    Rewriter.mergeBlockBefore(&NextParallel.getRegion().front(),
                              NewFor.getBody()->getTerminator());
    Rewriter.setInsertionPoint(NewFor.getBody()->getTerminator());
    Rewriter.create<omp::BarrierOp>(NextParallel.getLoc());

    Rewriter.setInsertionPointToEnd(&NewParallel.getRegion().front());
    Rewriter.clone(*Yield);
    Rewriter.eraseOp(Yield);
    Rewriter.eraseOp(NextParallel);
    Rewriter.eraseOp(PrevFor);

    return success();
  }
};

struct ParallelIfInterchange : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp PrevIf,
                                PatternRewriter &Rewriter) const override {
    if (PrevIf->getResults().size())
      return failure();

    omp::ParallelOp NextParallel = nullptr;
    if (auto *ThenB = PrevIf.thenBlock()) {
      if (ThenB->getOperations().size() != 2)
        return failure();
      NextParallel = dyn_cast<omp::ParallelOp>(&ThenB->front());
    }
    if (!NextParallel)
      return failure();

    omp::ParallelOp ElseParallel = nullptr;
    if (auto *ElseB = PrevIf.elseBlock()) {
      if (ElseB->getOperations().size() != 2)
        return failure();
      ElseParallel = dyn_cast<omp::ParallelOp>(&ElseB->front());
      if (!ElseParallel)
        return failure();
    }

    Rewriter.setInsertionPoint(PrevIf);
    auto NewParallel = Rewriter.create<omp::ParallelOp>(NextParallel.getLoc());
    Rewriter.createBlock(&NewParallel.getRegion());
    Rewriter.setInsertionPointToEnd(&NewParallel.getRegion().front());
    auto NewIf = Rewriter.create<scf::IfOp>(
        PrevIf.getLoc(), PrevIf.getCondition(), /*hasElse*/ ElseParallel);
    auto *Yield = NextParallel.getRegion().front().getTerminator();
    Rewriter.setInsertionPoint(NewIf.thenYield());

    auto AllocScope =
        Rewriter.create<memref::AllocaScopeOp>(PrevIf.getLoc(), TypeRange());
    Rewriter.inlineRegionBefore(NextParallel.getRegion(),
                                AllocScope.getRegion(),
                                AllocScope.getRegion().begin());
    Rewriter.setInsertionPointToEnd(&AllocScope.getRegion().front());
    Rewriter.create<memref::AllocaScopeReturnOp>(AllocScope.getLoc());

    if (ElseParallel) {
      Rewriter.eraseOp(ElseParallel.getRegion().front().getTerminator());
      Rewriter.setInsertionPoint(NewIf.elseYield());
      auto AllocScope =
          Rewriter.create<memref::AllocaScopeOp>(PrevIf.getLoc(), TypeRange());
      Rewriter.inlineRegionBefore(ElseParallel.getRegion(),
                                  AllocScope.getRegion(),
                                  AllocScope.getRegion().begin());
      Rewriter.setInsertionPointToEnd(&AllocScope.getRegion().front());
      Rewriter.create<memref::AllocaScopeReturnOp>(AllocScope.getLoc());
    }

    Rewriter.setInsertionPointToEnd(&NewParallel.getRegion().front());
    Rewriter.clone(*Yield);
    Rewriter.eraseOp(Yield);
    Rewriter.eraseOp(PrevIf);
    return success();
  }
};

void OpenMPOpt::runOnOperation() {
  mlir::RewritePatternSet RPL(getOperation()->getContext());
  RPL.add<CombineParallel, ParallelForInterchange, ParallelIfInterchange>(
      getOperation()->getContext());
  GreedyRewriteConfig Config;
  Config.maxIterations = 47;
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(RPL), Config);
}

std::unique_ptr<Pass> mlir::polygeist::createOpenMPOptPass() {
  return std::make_unique<OpenMPOpt>();
}
