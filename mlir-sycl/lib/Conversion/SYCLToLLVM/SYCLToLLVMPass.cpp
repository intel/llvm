//===- SYCLToLLVMPass.cpp - SYCL to LLVM Passes ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR SYCL ops into LLVM ops
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToLLVM/SYCLToLLVMPass.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SYCLToLLVM/SYCLToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"

using namespace mlir;

namespace mlir {
#define GEN_PASS_DEF_CONVERTSYCLTOLLVM
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DEF_CONVERTSYCLTOLLVM
} // namespace mlir

namespace {
/// A pass converting MLIR SYCL operations into LLVM dialect.
class ConvertSYCLToLLVMPass
    : public impl::ConvertSYCLToLLVMBase<ConvertSYCLToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertSYCLToLLVMPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  LLVMTypeConverter converter(&getContext());

  RewritePatternSet patterns(context);

  sycl::populateSYCLToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);

  ConversionTarget target(*context);
  target.addIllegalDialect<sycl::SYCLDialect>();
  target.addLegalDialect<LLVM::LLVMDialect>();

  target.addLegalOp<ModuleOp>();
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::sycl::createConvertSYCLToLLVMPass() {
  return std::make_unique<ConvertSYCLToLLVMPass>();
}
