//===- GENToSPIRV.cpp - GEN to SPIRV dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GENToSPIRV/GENToSPIRV.h"

#include "mlir/Dialect/GEN/IR/GENDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGENTOSPIRV
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "gen-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::GEN::populateGENToSPIRVPatterns(
    [[maybe_unused]] SPIRVTypeConverter &,
    [[maybe_unused]] RewritePatternSet &) {}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertGENToSPIRVPass
    : public impl::ConvertGENToSPIRVBase<ConvertGENToSPIRVPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
    std::unique_ptr<SPIRVConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    SPIRVTypeConverter typeConverter(targetAttr);

    // Fail hard when there are any remaining GEN ops.
    target->addIllegalDialect<GEN::GENDialect>();

    RewritePatternSet patterns(&getContext());
    GEN::populateGENToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<>> mlir::GEN::createConvertGENToSPIRVPass() {
  return std::make_unique<ConvertGENToSPIRVPass>();
}
