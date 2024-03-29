//===- GENOpsInterfaces.h - GEN Interfaces ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op interfaces for the GEN dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GEN_IR_GENOPSINTERFACES_H_
#define MLIR_DIALECT_GEN_IR_GENOPSINTERFACES_H_

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace GEN {
namespace detail {

/// Verifies the operation receives an i32 argument, returns an index and that
/// the input is non-constant or in the range [0, 2].
LogicalResult verify3DNDRangeOpInterface(Operation *op);

} // namespace detail
} // namespace GEN
} // namespace mlir

#include "mlir/Dialect/GEN/IR/GENOpsInterfaces.h.inc"

#endif // MLIR_DIALECT_GEN_IR_GENOPSINTERFACES_H_
