//===--- SYCLOps.h --------------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_IR_SYCLOPS_H
#define MLIR_DIALECT_SYCL_IR_SYCLOPS_H

#include "mlir/Dialect/SYCL/IR/SYCLHostTraits.h"
#include "mlir/Dialect/SYCL/IR/SYCLTraits.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"

#include "mlir/IR/BuiltinOps.h"

#include "mlir/Dialect/SYCL/IR/SYCLMethodOpInterface.h"

namespace mlir {
namespace sycl {

// Return true if the operation \p op belongs to the SYCL MLIR dialect.
inline bool isSYCLOperation(Operation *op) {
  if (!op || !op->getDialect())
    return false;
  return isa<sycl::SYCLDialect>(op->getDialect());
}

} // namespace sycl
} // namespace mlir

/// Include the auto-generated header file containing the declarations of the
/// sycl operations.
#define GET_OP_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOps.h.inc"

#endif // MLIR_DIALECT_SYCL_IR_SYCLOPS_H
