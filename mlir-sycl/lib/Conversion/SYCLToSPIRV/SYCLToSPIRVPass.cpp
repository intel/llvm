//===- SYCLToSPIRVPass.cpp - SYCL to SPIRV Passes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR SYCL ops into SPIRV ops
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToSPIRV/SYCLToSPIRVPass.h"

#include "mlir/Conversion/SYCLToSPIRV/SYCLToSPIRV.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"

using namespace mlir;
using namespace mlir::sycl;

namespace mlir {
#define GEN_PASS_DEF_CONVERTSYCLTOSPIRV
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DEF_CONVERTSYCLTOSPIRV
} // namespace mlir

namespace {
/// A pass converting MLIR SYCL operations into LLVM dialect.
class ConvertSYCLToSPIRVPass
    : public impl::ConvertSYCLToSPIRVBase<ConvertSYCLToSPIRVPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertSYCLToSPIRVPass::runOnOperation() {
  auto *context = &getContext();
  auto module = getOperation();

  module.walk([&](gpu::GPUModuleOp gpuModule) {
    // We walk the different GPU modules looking for different SPIRV target
    // environment definitions. Currently, this does not affect the behavior of
    // this pass.
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    auto targetAttr = spirv::lookupTargetEnvOrDefault(gpuModule);
    SPIRVTypeConverter typeConverter(targetAttr);

    populateSYCLToSPIRVConversionPatterns(typeConverter, patterns);

    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<spirv::SPIRVDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<SYCLDialect>();

    target.addIllegalOp<SYCLGlobalOffsetOp, SYCLNumWorkGroupsOp,
                        SYCLSubGroupLocalIDOp, SYCLSubGroupMaxSizeOp>();

    if (failed(applyPartialConversion(gpuModule, target, std::move(patterns))))
      signalPassFailure();
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::sycl::createConvertSYCLToSPIRVPass() {
  return std::make_unique<ConvertSYCLToSPIRVPass>();
}
