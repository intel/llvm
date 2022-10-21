//===--- SYCLOps.h --------------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_OPS_H_
#define MLIR_SYCL_OPS_H_

#include "mlir/Dialect/SYCL/IR/SYCLOpTraits.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"

#include "mlir/IR/BuiltinOps.h"

/// Include the header file containing the declaration of the sycl operation
/// interfaces.
#include "mlir/Dialect/SYCL/IR/SYCLOpInterfaces.h.inc"

namespace mlir {
namespace sycl {
template <typename T>
using isSYCLMethod = std::is_base_of<SYCLMethodOpInterface::Trait<T>, T>;
} // namespace sycl
} // namespace mlir

/// Include the auto-generated header file containing the declarations of the
/// sycl operations.
#define GET_OP_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOps.h.inc"

#endif // MLIR_SYCL_OPS_H_
