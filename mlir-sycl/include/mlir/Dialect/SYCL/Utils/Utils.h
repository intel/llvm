//===- Utils.h - General SYCL transformation utilities ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various transformation utilities for
// the SYCL dialect. These are not passes by themselves but are used
// either by passes, optimization sequences, or in turn by other transformation
// utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_UTILS_UTILS_H
#define MLIR_DIALECT_SYCL_UTILS_UTILS_H

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"

namespace mlir {
namespace sycl {

/// Return type of the accessor of \p op.
inline AccessorType getAccessorType(SYCLAccessorSubscriptOp op) {
  return sycl::AccessorPtrValue(op.getAcc()).getAccessorType();
}

sycl::SYCLIDGetOp createSYCLIDGetOp(TypedValue<MemRefType> id, unsigned index,
                                    OpBuilder builder, Location loc);

sycl::SYCLAccessorSubscriptOp
createSYCLAccessorSubscriptOp(sycl::AccessorPtrValue accessor,
                              TypedValue<MemRefType> id, OpBuilder builder,
                              Location loc);

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_UTILS_UTILS_H
