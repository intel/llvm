//===- TrivialUse.cpp - Remove trivial use instruction ---------------- -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"

#define DEBUG_TYPE "trivial-use"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_REMOVETRIVIALUSE
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;
using namespace polygeist;

namespace {
struct RemoveTrivialUse
    : public mlir::polygeist::impl::RemoveTrivialUseBase<RemoveTrivialUse> {
  void runOnOperation() override;
};

} // end anonymous namespace

namespace mlir {
namespace polygeist {
std::unique_ptr<Pass> createRemoveTrivialUsePass() {
  return std::make_unique<RemoveTrivialUse>();
}
} // namespace polygeist
} // namespace mlir

void RemoveTrivialUse::runOnOperation() {
  getOperation()->walk([&](polygeist::TrivialUseOp bidx) { bidx.erase(); });
}
