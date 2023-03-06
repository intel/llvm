//===- SYCLToSPIRV.h - SYCL to SPIRV Patterns -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert SYCL dialect to SPIRV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLTOSPIRV_SYCLTOSPIRV_H
#define MLIR_CONVERSION_SYCLTOSPIRV_SYCLTOSPIRV_H

#include <memory>

namespace mlir {
class Pass;
class RewritePatternSet;
class SPIRVTypeConverter;

#define GEN_PASS_DECL_CONVERTSYCLTOSPIRV
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DECL_CONVERTSYCLTOSPIRV

/// Populates the given list with patterns that convert from SYCL to SPIRV.
void populateSYCLToSPIRVConversionPatterns(SPIRVTypeConverter &typeConverter,
                                           RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOSPIRV_SYCLTOSPIRV_H
