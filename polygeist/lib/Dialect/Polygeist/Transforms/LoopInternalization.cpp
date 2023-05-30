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
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
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

// Only visit kernel body function with nd_item argument.
bool isCandidate(FunctionOpInterface func) {
  if (!polygeist::isPotentialKernelBodyFunc(func))
    return false;

  if (func.getNumArguments() == 0 ||
      !sycl::isPtrType<sycl::NdItemType>(func.getArgumentTypes().back()))
    return false;

  return true;
}

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

  // TODO: check uniformity.

  return true;
}

template <typename T,
          typename = std::enable_if_t<llvm::is_one_of<
              T, affine::AffineForOp, scf::ForOp, LoopLikeOpInterface>::value>>
void getTileSizes(const SmallVector<T> &nestedLoops,
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
}

LogicalResult tile(MutableArrayRef<affine::AffineForOp> nestedLoops,
                   ArrayRef<Value> tileSizes,
                   SmallVectorImpl<affine::AffineForOp> &tiledNest) {
  SmallVector<affine::AffineForOp> newNestedLoops;
  unsigned numLoops = nestedLoops.size();
  LogicalResult res =
      tilePerfectlyNestedParametric(nestedLoops, tileSizes, &newNestedLoops);
  tiledNest = SmallVector<affine::AffineForOp>(
      newNestedLoops.begin() + numLoops, newNestedLoops.end());
  return res;
}
LogicalResult tile(ArrayRef<scf::ForOp> nestedLoops, ArrayRef<Value> tileSizes,
                   SmallVectorImpl<scf::ForOp> &tiledNest) {
  tiledNest = tile(nestedLoops, tileSizes, nestedLoops.back());
  return success();
}

void createLocalBarrier(OpBuilder builder) {
  // TODO: Use gpu.barrier, require GPUToSPIRV conversion in the pipeline.
  builder.create<spirv::ControlBarrierOp>(
      builder.getUnknownLoc(), spirv::Scope::Workgroup, spirv::Scope::Workgroup,
      spirv::MemorySemantics::SequentiallyConsistent |
          spirv::MemorySemantics::WorkgroupMemory);
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

      if (!isCandidate(func))
        return WalkResult::advance();

      LLVM_DEBUG(llvm::dbgs()
                 << "LoopInternalization: Visiting candidate function "
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
    getTileSizes(nestedLoops, tileSizes);

    SmallVector<T> tiledNest;
    LogicalResult res = tile(nestedLoops, tileSizes, tiledNest);
    LLVM_DEBUG({
      if (res.succeeded())
        llvm::dbgs() << "Tiled loop: " << tiledNest.front() << "\n";
      else
        llvm::dbgs() << "Tile NOT performed\n";
    });
    // TODO: promote loop accesses to local memory.
    loop = tiledNest.front();
    OpBuilder builder(loop);
    builder.setInsertionPointToStart(loop->getBlock());
    createLocalBarrier(builder);
    builder.setInsertionPointAfter(loop);
    createLocalBarrier(builder);
  }
};

} // namespace

std::unique_ptr<Pass> polygeist::createLoopInternalizationPass() {
  return std::make_unique<LoopInternalization>();
}
std::unique_ptr<Pass> polygeist::createLoopInternalizationPass(
    const LoopInternalizationOptions &options) {
  return std::make_unique<LoopInternalization>(options);
}
