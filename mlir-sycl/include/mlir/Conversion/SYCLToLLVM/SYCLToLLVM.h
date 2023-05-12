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

#include <memory>

#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h"

namespace mlir {
class Pass;
class LLVMTypeConverter;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTSYCLTOLLVM
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DECL_CONVERTSYCLTOLLVM

/// Populates type conversions with additional SYCL types.
void populateSYCLToLLVMTypeConversion(sycl::Implementation implementation,
                                      LLVMTypeConverter &typeConverter);

/// Populates the given list with patterns that convert from SYCL to LLVM.
void populateSYCLToLLVMConversionPatterns(sycl::Implementation implementation,
                                          sycl::LoweringTarget target,
                                          LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOLLVM_SYCLTOLLVM_H
