//===- SYCLToGPU.h - SYCL to GPU Patterns -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert SYCL dialect to GPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLTOGPU_SYCLTOGPU_H
#define MLIR_CONVERSION_SYCLTOGPU_SYCLTOGPU_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class TypeConverter;
namespace sycl {

/// Populates the given list with patterns that convert from SYCL to GPU.
void populateSYCLToGPUConversionPatterns(RewritePatternSet &patterns);

} // namespace sycl
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOGPU_SYCLTOGPU_H
