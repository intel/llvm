//===- SYCLToSPIRVPass.h - SYCL to SPIRV Passes -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert SYCL dialect to SPIRV dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLTOSPIRV_SYCLTOSPIRVPASS_H
#define MLIR_CONVERSION_SYCLTOSPIRV_SYCLTOSPIRVPASS_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;

#define GEN_PASS_DECL_CONVERTSYCLTOSPIRV
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DECL_CONVERTSYCLTOSPIRV

namespace sycl {

/// Creates a pass to convert SYCL operations to the SPIRV dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertSYCLToSPIRVPass();

} // namespace sycl
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOSPIRV_SYCLTOSPIRVPASS_H
