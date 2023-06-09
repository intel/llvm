//===- SYCLToMath.h - SYCL to Math Patterns ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert SYCL dialect to Math dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLTOMATH_SYCLTOMATH_H
#define MLIR_CONVERSION_SYCLTOMATH_SYCLTOMATH_H

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class TypeConverter;

#define GEN_PASS_DECL_CONVERTSYCLTOMATH
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DECL_CONVERTSYCLTOMATH

/// Populates the given list with patterns that convert from SYCL to Math.
void populateSYCLToMathConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOMATH_SYCLTOMATH_H
