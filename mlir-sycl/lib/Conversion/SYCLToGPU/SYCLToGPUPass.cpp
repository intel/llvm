//===- SYCLToGPUPass.cpp - SYCL to GPU Passes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR SYCL ops into GPU ops
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToGPU/SYCLToGPUPass.h"

#include "mlir/Conversion/SYCLToGPU/SYCLToGPU.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"

using namespace mlir;
using namespace mlir::sycl;

namespace mlir {
#define GEN_PASS_DEF_CONVERTSYCLTOGPU
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DEF_CONVERTSYCLTOGPU
} // namespace mlir

namespace {
/// A pass converting MLIR SYCL operations into LLVM dialect.
class ConvertSYCLToGPUPass
    : public impl::ConvertSYCLToGPUBase<ConvertSYCLToGPUPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertSYCLToGPUPass::runOnOperation() {
  auto *context = &getContext();

  RewritePatternSet patterns(context);
  ConversionTarget target(*context);

  populateSYCLToGPUConversionPatterns(patterns);

  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<gpu::GPUDialect>();
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<SYCLDialect>();

  target
      .addIllegalOp<SYCLWorkGroupIDOp, SYCLNumWorkItemsOp, SYCLWorkGroupSizeOp,
                    SYCLLocalIDOp, SYCLGlobalIDOp, SYCLSubGroupIDOp,
                    SYCLNumSubGroupsOp, SYCLSubGroupSizeOp>();

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::sycl::createConvertSYCLToGPUPass() {
  return std::make_unique<ConvertSYCLToGPUPass>();
}
