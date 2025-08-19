//===- SPIRVBuiltInToNVVM.h - SPIRVBuiltIn to NVVM Patterns
//-----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert SPIRVBuiltIn dialect to NVVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SPIRVBUILTINTONVVM_SPIRVBUILTINTONVVM_H
#define MLIR_CONVERSION_SPIRVBUILTINTONVVM_SPIRVBUILTINTONVVM_H

#include <memory>

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTSPIRVBUILTINTONVVM
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DECL_CONVERTSPIRVBUILTINTONVVM

} // namespace mlir

#endif // MLIR_CONVERSION_SPIRVBUILTINTONVVM_SPIRVBUILTINTONVVM_H
