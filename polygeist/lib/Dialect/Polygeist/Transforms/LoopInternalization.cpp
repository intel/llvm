//===- LoopInternalization.cpp - Promote memory access to local memory ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tiles perfect loop nests to 'prefetch' memory accesses in shared
// local memory.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Polygeist/Analysis/MemoryAccessAnalysis.h"
#include "mlir/Dialect/Polygeist/Transforms/Passes.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "loop-internalization"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_LOOPINTERNALIZATION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;
using namespace mlir::polygeist;

static llvm::cl::list<unsigned> LoopInternalizationTileSizes(
    DEBUG_TYPE "-tile-sizes", llvm::cl::CommaSeparated,
    llvm::cl::desc("Tile sizes used in LoopInternalization"));

namespace {

//===----------------------------------------------------------------------===//
// Utilities functions
//===----------------------------------------------------------------------===//

/// A loop is a candidate when it is the outermost affine or scf for loop.
bool isCandidate(LoopLikeOpInterface loop) {
  if (!isa<affine::AffineForOp, scf::ForOp>(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not affine or scf for loop\n");
    return false;
  }

  if (!LoopTools::isOutermostLoop(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not outermost loop\n");
    return false;
  }

  if (!LoopTools::isPerfectLoopNest(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not perfect loop nest\n");
    return false;
  }

  return true;
}

template <typename T,
          typename = std::enable_if_t<llvm::is_one_of<
              T, affine::AffineForOp, scf::ForOp, LoopLikeOpInterface>::value>>
LogicalResult getTileSizes(const SmallVector<T> &nestedLoops,
                           SmallVectorImpl<Value> &tileSizes) {
  // TODO: calculate proper tile sizes.
  OpBuilder builder(nestedLoops.front());
  for (auto tileSize : LoopInternalizationTileSizes)
    tileSizes.push_back(builder.create<arith::ConstantIndexOp>(
        builder.getUnknownLoc(), tileSize));
  if (nestedLoops.size() != tileSizes.size()) {
    Value one =
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
    tileSizes.resize(nestedLoops.size(), one);
  }
  return success();
}

LogicalResult tile(MutableArrayRef<affine::AffineForOp> nestedLoops,
                   ArrayRef<Value> tileSizes,
                   SmallVectorImpl<affine::AffineForOp> &tiledNest) {
  return tilePerfectlyNestedParametric(nestedLoops, tileSizes, &tiledNest);
}
LogicalResult tile(SmallVectorImpl<scf::ForOp> &nestedLoops,
                   ArrayRef<Value> tileSizes,
                   SmallVectorImpl<scf::ForOp> &tiledNest) {
  tile(nestedLoops, tileSizes, nestedLoops.back());
  tiledNest = nestedLoops;
  return success();
}

//===----------------------------------------------------------------------===//
// LoopInternalization
//===----------------------------------------------------------------------===//

struct LoopInternalization
    : public polygeist::impl::LoopInternalizationBase<LoopInternalization> {
  using LoopInternalizationBase<LoopInternalization>::LoopInternalizationBase;

  void runOnOperation() final {
    Operation *module = getOperation();
    ModuleAnalysisManager mam(module, /*passInstrumentor=*/nullptr);
    AnalysisManager am = mam;
    auto &memAccessAnalysis =
        am.getAnalysis<MemoryAccessAnalysis>().initialize(relaxedAliasing);

    auto walkResult = module->walk([&](FunctionOpInterface func) {
      DataFlowSolver solver;
      solver.load<dataflow::DeadCodeAnalysis>();
      solver.load<dataflow::IntegerRangeAnalysis>();
      if (failed(solver.initializeAndRun(func)))
        return WalkResult::interrupt();

      LLVM_DEBUG(llvm::dbgs() << "LoopInternalization: Visiting Function "
                              << func.getName() << "\n");
      func->walk<WalkOrder::PreOrder>([&](LoopLikeOpInterface loop) {
        LLVM_DEBUG(llvm::dbgs()
                   << "LoopInternalization: Visiting Loop " << loop << "\n");

        if (!isCandidate(loop))
          return;

        transform(loop, memAccessAnalysis, solver);
      });

      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      signalPassFailure();
  }

private:
  void transform(LoopLikeOpInterface loop,
                 const MemoryAccessAnalysis &memAccessAnalysis,
                 DataFlowSolver &solver) const {
    TypeSwitch<Operation *>(loop).Case<affine::AffineForOp, scf::ForOp>(
        [&](auto loop) { transform(loop, memAccessAnalysis, solver); });
  }

  template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                            T, affine::AffineForOp, scf::ForOp>::value>>
  void transform(T loop, const MemoryAccessAnalysis &memAccessAnalysis,
                 DataFlowSolver &solver) const {
    FunctionOpInterface func =
        loop->template getParentOfType<FunctionOpInterface>();

    LoopLikeOpInterface innermostLoop = *LoopTools::getInnermostLoop(loop);

    SmallVector<T> nestedLoops;
    LoopTools::getPerfectlyNestedLoops(nestedLoops, loop);

    SmallVector<Value> tileSizes;
    if (getTileSizes(nestedLoops, tileSizes).failed())
      return;

    SmallVector<T> tiledNest;
    LogicalResult res = tile(nestedLoops, tileSizes, tiledNest);
    LLVM_DEBUG({
      if (res.succeeded())
        llvm::dbgs() << "Tiled loop: " << tiledNest.front() << "\n";
      else
        llvm::dbgs() << "Tile NOT performed\n";
    });
    // TODO: promote loop accesses to local memory.
  }
};

} // namespace

std::unique_ptr<Pass> polygeist::createLoopInternalizationPass() {
  return std::make_unique<LoopInternalization>();
}
