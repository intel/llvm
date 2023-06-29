//=== SYCLRaiseHost.cpp - Raise host constructs to SYCL dialect operations ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass attempts to detect instruction sequences of interest in the MLIR
// (mostly LLVM dialect) for the SYCL host side and raise them to types and
// operations from the SYCL dialect to facilitate analysis in other passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/Polygeist/Utils/TransformUtils.h"
#include "mlir/Dialect/Polygeist/Utils/Utils.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_SYCLRAISEHOSTCONSTRUCTS
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

namespace {

class SYCLRaiseHostConstructsPass
    : public polygeist::impl::SYCLRaiseHostConstructsBase<
          SYCLRaiseHostConstructsPass> {
public:
  using polygeist::impl::SYCLRaiseHostConstructsBase<
      SYCLRaiseHostConstructsPass>::SYCLRaiseHostConstructsBase;

  void runOnOperation() override;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//

namespace {
struct RaiseKernelName : public OpRewritePattern<LLVM::AddressOfOp> {
public:
  using OpRewritePattern<LLVM::AddressOfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AddressOfOp op,
                                PatternRewriter &rewriter) const final {
    // Get the global this operation uses
    SymbolTableCollection symbolTable;
    auto global = symbolTable.lookupNearestSymbolFrom<LLVM::GlobalOp>(
        op, op.getGlobalNameAttr());
    if (!global)
      return failure();

    // Get a reference to the kernel the global references
    std::optional<SymbolRefAttr> ref = getKernelRef(global);
    if (!ref)
      return failure();

    rewriter.replaceOpWithNewOp<sycl::SYCLHostGetKernelOp>(op, op.getType(),
                                                           *ref);
    return success();
  }

private:
  /// If the input global contains a constant string representing the name of a
  /// SYCL kernel, returns a reference to the `gpu.func` implementing this
  /// kernel.
  ///
  /// The string will contain a trailing character we need to get rid of before
  /// searching.
  static std::optional<SymbolRefAttr> getKernelRef(LLVM::GlobalOp op) {
    // Check the operation has a value
    std::optional<Attribute> attr = op.getValue();
    if (!attr)
      return std::nullopt;

    // Check it is a string
    auto strAttr = dyn_cast<StringAttr>(*attr);
    if (!strAttr)
      return std::nullopt;

    // Drop the trailing `0` character
    StringRef name = strAttr.getValue().drop_back();

    // Search the `gpu.func` in the device module
    SymbolTableCollection symbolTable;
    auto ref =
        SymbolRefAttr::get(op->getContext(), DeviceModuleName,
                           FlatSymbolRefAttr::get(op->getContext(), name));
    auto kernel = symbolTable.lookupNearestSymbolFrom<gpu::GPUFuncOp>(op, ref);

    // If it was found and it is a kernel, return the reference
    return kernel && kernel.isKernel() ? std::optional<SymbolRefAttr>(ref)
                                       : std::nullopt;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SYCLRaiseHostConstructsPass
//===----------------------------------------------------------------------===//

void SYCLRaiseHostConstructsPass::runOnOperation() {
  Operation *scopeOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet rewritePatterns{context};
  rewritePatterns.add<RaiseKernelName>(context);
  FrozenRewritePatternSet frozen(std::move(rewritePatterns));

  if (failed(applyPatternsAndFoldGreedily(scopeOp, frozen)))
    signalPassFailure();
}

std::unique_ptr<Pass> mlir::polygeist::createSYCLHostRaisingPass() {
  return std::make_unique<SYCLRaiseHostConstructsPass>();
}

std::unique_ptr<Pass> mlir::polygeist::createSYCLHostRaisingPass(
    const polygeist::SYCLRaiseHostConstructsOptions &options) {
  return std::make_unique<SYCLRaiseHostConstructsPass>(options);
}
