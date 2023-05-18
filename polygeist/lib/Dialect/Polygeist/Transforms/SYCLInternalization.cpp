//===- SYCLInternalization.cpp - Promote loop access to local memory ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass attempts to tile loops that are considered profitable to have some
// of their loop memory access to local memory.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Polygeist/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sycl-internalization"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_SYCLINTERNALIZATION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// SYCLInternalizationPass
//===----------------------------------------------------------------------===//

namespace {
/// Collect perfectly nested loops starting from \p root.  Loops are
/// perfectly nested if each loop is the first and only non-terminator operation
/// in the parent loop.
template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                          T, affine::AffineForOp, scf::ForOp>::value>>
void getPerfectlyNestedLoops(SmallVector<T> &nestedLoops, T root) {
  for (unsigned i = 0; i < std::numeric_limits<unsigned>::max(); ++i) {
    nestedLoops.push_back(root);
    Block &body = root.getLoopBody().front();
    if (body.begin() != std::prev(body.end(), 2))
      return;

    root = dyn_cast<T>(&body.front());
    if (!root)
      return;
  }
}

bool isOutermostLoop(LoopLikeOpInterface loop) {
  return !loop->getParentRegion()->getParentOfType<LoopLikeOpInterface>();
}

bool isCandidate(LoopLikeOpInterface loop) {
  if (!isOutermostLoop(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not outermost loop\n");
    return false;
  }

  if (!isa<affine::AffineForOp, scf::ForOp>(loop)) {
    LLVM_DEBUG(llvm::dbgs() << "not candidate: not affine or scf for loop\n");
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
  Value one =
      builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
  tileSizes.resize(nestedLoops.size(), one);
  return success();
}

LogicalResult tile(MutableArrayRef<affine::AffineForOp> nestedLoops,
                   ArrayRef<Value> tileSizes) {
  return tilePerfectlyNestedParametric(nestedLoops, tileSizes);
}
LogicalResult tile(MutableArrayRef<scf::ForOp> nestedLoops,
                   ArrayRef<Value> tileSizes) {
  tile(nestedLoops, tileSizes, nestedLoops.back());
  return success();
}

template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                          T, affine::AffineForOp, scf::ForOp>::value>>
LogicalResult transform(T loop) {
  SmallVector<T> nestedLoops;
  getPerfectlyNestedLoops(nestedLoops, loop);
  SmallVector<Value> tileSizes;
  if (getTileSizes(nestedLoops, tileSizes).failed())
    return failure();
  if (tile(nestedLoops, tileSizes).failed())
    return failure();
  // TODO: promote loop accesses to local memory.
  return success();
}

void transform(LoopLikeOpInterface loop) {
  TypeSwitch<Operation *>(loop).Case<affine::AffineForOp, scf::ForOp>(
      [&](auto loop) {
        LogicalResult res = transform(loop);
        assert(res.succeeded() && "Expecting transform to be successful");
      });
}

struct SYCLInternalization
    : public polygeist::impl::SYCLInternalizationBase<SYCLInternalization> {
  void runOnOperation() override {
    LoopLikeOpInterface loop = getOperation();
    LLVM_DEBUG(llvm::dbgs()
               << "SYCLInternalization: Visiting " << loop << "\n");

    if (!isCandidate(loop))
      return;

    transform(loop);
  }
};
} // namespace

std::unique_ptr<Pass> polygeist::createSYCLInternalization() {
  return std::make_unique<SYCLInternalization>();
}
