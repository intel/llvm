//===- GENDialect.h - MLIR GEN dialect -----------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the GEN dialect in MLIR, containing Intel GEN operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GEN_IR_GENDIALECT_H
#define MLIR_DIALECT_GEN_IR_GENDIALECT_H

#include "mlir/IR/Dialect.h"

#include "mlir/Dialect/GEN/IR/GENOpsDialect.h.inc"

namespace mlir {
namespace GEN {

/// GEN memory space identifiers following SPIRV storage class convention
/// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/docs/SPIRVRepresentationInLLVM.rst#address-spaces
///
enum class GENStorageClass {
  Function = 0,        // OpenCL workitem address space
  CrossWorkgroup = 1,  // OpenCL Global memory
  UniformConstant = 2, // OpenCL Constant memory
  Workgroup = 3,       // OpenCL Local memory
  Generic = 4          // OpenCL Generic memory
};

} // namespace GEN
} // namespace mlir

#endif // MLIR_DIALECT_GEN_IR_GENDIALECT_H
