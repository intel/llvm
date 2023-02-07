//===- SYCLPasses.h - Conversion Pass Construction and Registration -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLPASSES_H
#define MLIR_CONVERSION_SYCLPASSES_H

#include "mlir/Conversion/SYCLToGPU/SYCLToGPUPass.h"
#include "mlir/Conversion/SYCLToLLVM/SYCLToLLVMPass.h"

namespace mlir {
namespace sycl {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Conversion/SYCLPasses.h.inc"

} // namespace sycl
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLPASSES_H
