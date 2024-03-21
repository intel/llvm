//===- GENToSPIRV.h - Convert GEN to SPIRV dialect -----*- C++ ----------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GENTOSPIRV_GENTOSPIRV_H
#define MLIR_CONVERSION_GENTOSPIRV_GENTOSPIRV_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class SPIRVTypeConverter;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTGENTOSPIRV
#include "mlir/Conversion/Passes.h.inc"

namespace GEN {
void populateGENToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                RewritePatternSet &patterns);

std::unique_ptr<OperationPass<>> createConvertGENToSPIRVPass();
} // namespace GEN
} // namespace mlir

#endif // MLIR_CONVERSION_GENTOSPIRV_GENTOSPIRV_H
