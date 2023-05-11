//===- DPCPP.h - SYCL to LLVM Patterns for the DPC++ implementation-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLTOLLVM_DPCPP_H
#define MLIR_CONVERSION_SYCLTOLLVM_DPCPP_H

#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h"

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
namespace dpcpp {
/// Populates type conversions with additional SYCL types.
void populateSYCLToLLVMTypeConversion(LLVMTypeConverter &typeConverter);

/// Populates the given list with patterns that convert from SYCL to LLVM.
void populateSYCLToLLVMConversionPatterns(sycl::LoweringTarget target,
                                          LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns);
} // namespace dpcpp
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOLLVM_DPCPP_H
