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
  return AccessorPtrValue(op.getAcc()).getAccessorType();
}

/// Create sycl.id.get with result type \resTy, id \p id and index \p index.
SYCLIDGetOp createSYCLIDGetOp(Type resTy, TypedValue<MemRefType> id,
                              unsigned index, OpBuilder builder, Location loc);

/// Create sycl.range.get with result type \p resTy, range \p range and index \p
/// index.
SYCLRangeGetOp createSYCLRangeGetOp(Type resTy, TypedValue<MemRefType> range,
                                    unsigned index, OpBuilder builder,
                                    Location loc);

/// Construct sycl.id with id type \p idTy and indexes \p indexes.
TypedValue<MemRefType> createSYCLIDConstructorOp(IDType idTy,
                                                 ValueRange indexes,
                                                 OpBuilder builder,
                                                 Location loc);

/// Create sycl.accessor.subscript with accessor \p accessor and id \p id.
SYCLAccessorSubscriptOp createSYCLAccessorSubscriptOp(AccessorPtrValue accessor,
                                                      TypedValue<MemRefType> id,
                                                      OpBuilder builder,
                                                      Location loc);

/// Create sycl.work_group_size with rank \p numDims.
sycl::SYCLWorkGroupSizeOp createWorkGroupSize(unsigned numDims,
                                              OpBuilder builder, Location loc);

/// Populate \p wgSizes with workgroup size per dimensionality, given the rank
/// \p numDims.
void populateWorkGroupSize(SmallVectorImpl<Value> &wgSizes, unsigned numDims,
                           OpBuilder builder, Location loc);

/// Populate \p localIDs with local id per dimensionality, given the rank
/// \p numDims.
void populateLocalID(SmallVectorImpl<Value> &localIDs, unsigned numDims,
                     OpBuilder builder, Location loc);

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_UTILS_UTILS_H
