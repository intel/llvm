//===- SYCLToLLVM.cpp - SYCL to LLVM Patterns -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert SYCL dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToLLVM/SYCLToLLVM.h"

#include <optional>

#include "DPCPP.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVM.h"
#include "mlir/Conversion/SYCLToSPIRV/SYCLToSPIRV.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Polygeist/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h"
#include "mlir/Dialect/SYCL/IR/SYCLDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "sycl-to-llvm"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSYCLTOLLVM
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DEF_CONVERTSYCLTOLLVM
} // namespace mlir

using namespace mlir;
using namespace mlir::sycl;

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateSYCLToLLVMTypeConversion(sycl::Implementation implementation,
                                            LLVMTypeConverter &typeConverter) {
  switch (implementation) {
  case sycl::Implementation::DPCPP:
    dpcpp::populateSYCLToLLVMTypeConversion(typeConverter);
    break;
  }
}

void mlir::populateSYCLToLLVMConversionPatterns(
    sycl::Implementation implementation, sycl::LoweringTarget target,
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  switch (implementation) {
  case sycl::Implementation::DPCPP:
    dpcpp::populateSYCLToLLVMConversionPatterns(target, typeConverter,
                                                patterns);
    dpcpp::populateSYCLToLLVMTypeConversion(typeConverter);
    break;
  }
}

namespace {
class ConvertSYCLToLLVMPass
    : public impl::ConvertSYCLToLLVMBase<ConvertSYCLToLLVMPass> {
public:
  using impl::ConvertSYCLToLLVMBase<
      ConvertSYCLToLLVMPass>::ConvertSYCLToLLVMBase;

  void runOnOperation() override;
};
} // namespace

void ConvertSYCLToLLVMPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering to LLVM...\n");

  auto &context = getContext();
  auto module = getOperation();

  // TODO: As we may have device modules with different index widths, we may
  // need to revamp how we run this.

  LowerToLLVMOptions options(&context);
  options.useBarePtrCallConv = true;
  options.useOpaquePointers = true;
  if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
    options.overrideIndexBitwidth(indexBitwidth);
  LLVMTypeConverter converter(&context, options);

  RewritePatternSet patterns(&context);

  constexpr auto clientAPI = spirv::ClientAPI::OpenCL;

  // Keep these at the top; these should be run before the rest of
  // function conversion patterns.
  populateReturnOpTypeConversionPattern(patterns, converter);
  populateCallOpTypeConversionPattern(patterns, converter);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
  polygeist::populateBareMemRefToLLVMConversionPatterns(converter, patterns);
  populateSYCLToSPIRVConversionPatterns(converter, patterns);

  populateSYCLToLLVMConversionPatterns(syclImplementation, syclTarget,
                                       converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);

  populateVectorToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateSPIRVToLLVMTypeConversion(converter, clientAPI);
  populateSPIRVToLLVMConversionPatterns(converter, patterns, clientAPI);

  LLVMConversionTarget target(context);
  target.addIllegalDialect<SYCLDialect>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();

  LLVM_DEBUG(llvm::dbgs() << "Module after LLVM lowering:\n"; module.dump(););
}
