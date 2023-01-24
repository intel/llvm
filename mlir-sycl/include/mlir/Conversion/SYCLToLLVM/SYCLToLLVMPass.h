//===- SYCLToLLVMPass.h - SYCL to LLVM Passes -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert SYCL dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLTOLLVM_SYCLTOLLVMPASS_H
#define MLIR_CONVERSION_SYCLTOLLVM_SYCLTOLLVMPASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;

#define GEN_PASS_DECL_CONVERTSYCLTOLLVM
#include "mlir/Conversion/SYCLPasses.h.inc"

namespace sycl {

/// Creates a pass to convert SYCL operations to the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertSYCLToLLVMPass();

} // namespace sycl
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOLLVM_SYCLTOLLVMPASS_H
