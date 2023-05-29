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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Polygeist/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
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

static llvm::cl::list<unsigned> LoopInternalizationTileSizes(
    DEBUG_TYPE "-tile-sizes", llvm::cl::CommaSeparated,
    llvm::cl::desc("Tile sizes used in LoopInternalization"));

namespace {
/// Collect perfectly nested loops starting from \p root.  Loops are
/// perfectly nested if each loop is the first and only non-terminator operation
/// in the parent loop.
template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                          T, affine::AffineForOp, scf::ForOp>::value>>
void getPerfectlyNestedLoops(SmallVector<T> &nestedLoops, T root) {
  for (unsigned i = 0; i < std::numeric_limits<unsigned>::max(); ++i) {
    nestedLoops.push_back(root);
    assert(root.getLoopBody().hasOneBlock() && "Expecting single block");
    Block &body = root.getLoopBody().front();
    if (body.begin() != std::prev(body.end(), 2))
      return;

    root = dyn_cast<T>(&body.front());
    if (!root)
      return;
  }
}

bool isOutermostLoop(LoopLikeOpInterface loop) {
  return !loop->getParentOfType<LoopLikeOpInterface>();
}

/// A loop is a candidate when it is the outermost affine or scf for loop.
bool isCandidate(LoopLikeOpInterface loop) {
  if (!isOutermostLoop(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not outermost loop\n");
    return false;
  }

  if (!isa<affine::AffineForOp, scf::ForOp>(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not affine or scf for loop\n");
    return false;
  }

  // TODO: check uniformity.

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

template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                          T, affine::AffineForOp, scf::ForOp>::value>>
void transform(T loop) {
  FunctionOpInterface func =
      loop->template getParentOfType<FunctionOpInterface>();
  SmallVector<T> nestedLoops;
  getPerfectlyNestedLoops(nestedLoops, loop);
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
  OpBuilder builder(loop);
  builder.setInsertionPointToStart(tiledNest.front()->getBlock());
  createLocalBarrier(builder);
  builder.setInsertionPointAfter(tiledNest.front());
  createLocalBarrier(builder);
}

void transform(LoopLikeOpInterface loop) {
  TypeSwitch<Operation *>(loop).Case<affine::AffineForOp, scf::ForOp>(
      [&](auto loop) { transform(loop); });
}

struct LoopInternalization
    : public polygeist::impl::LoopInternalizationBase<LoopInternalization> {
  void runOnOperation() override {
    getOperation()->walk([&](LoopLikeOpInterface loop) {
      LLVM_DEBUG({
        FunctionOpInterface func = loop->getParentOfType<FunctionOpInterface>();
        llvm::dbgs() << "LoopInternalization: Visiting Function "
                     << func.getName() << "\n Loop: ";
        loop.dump();
      });

      if (!isCandidate(loop))
        return;

      transform(loop);
    });
  }
};
} // namespace

std::unique_ptr<Pass> polygeist::createLoopInternalizationPass() {
  return std::make_unique<LoopInternalization>();
}
