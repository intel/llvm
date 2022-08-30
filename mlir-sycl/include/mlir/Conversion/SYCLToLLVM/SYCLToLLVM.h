//===- SYCLToLLVM.h - SYCL to LLVM Patterns ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert SYCL dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLTOLLVM_SYCLTOLLVM_H
#define MLIR_CONVERSION_SYCLTOLLVM_SYCLTOLLVM_H

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;

namespace sycl {
template <typename SYCLOp>
class SYCLToLLVMConversion : public OpConversionPattern<SYCLOp> {
public:
  SYCLToLLVMConversion(MLIRContext *context, LLVMTypeConverter &typeConverter,
                       PatternBenefit benefit = 1)
      : OpConversionPattern<SYCLOp>(typeConverter, context, benefit),
        typeConverter(typeConverter) {}

protected:
  LLVMTypeConverter &typeConverter;
};

/// Populates type conversions with additional SYCL types.
void populateSYCLToLLVMTypeConversion(LLVMTypeConverter &typeConverter);

/// Populates the given list with patterns that convert from SYCL to LLVM.
void populateSYCLToLLVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns);

} // namespace sycl
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOLLVM_SYCLTOLLVM_H
