//===- Utils.cpp - Utilities to support the SYCL dialect ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for the SYCL dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Utils/Utils.h"

namespace mlir {
namespace sycl {

SYCLIDGetOp createSYCLIDGetOp(TypedValue<MemRefType> id, unsigned index,
                              OpBuilder builder, Location loc) {
  const Value indexOp = builder.create<arith::ConstantIntOp>(loc, index, 32);
  const auto resTy = builder.getIndexType();
  return builder.create<SYCLIDGetOp>(
      loc, MemRefType::get(ShapedType::kDynamic, resTy), id, indexOp);
}

TypedValue<MemRefType> constructSYCLID(IDType idTy, ArrayRef<Value> indexes,
                                       OpBuilder builder, Location loc) {
  const unsigned numDims = idTy.getDimension();
  assert(numDims == indexes.size() &&
         "Expecting the size of indexes to be the id dimension");
  auto id = builder.create<memref::AllocaOp>(loc, MemRefType::get(1, idTy));
  const Value zeroIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (unsigned dim = 0; dim < idTy.getDimension(); ++dim) {
    Value idGetOp = createSYCLIDGetOp(id, dim, builder, loc);
    builder.create<memref::StoreOp>(loc, indexes[dim], idGetOp, zeroIndex);
  }
  return id;
}

SYCLAccessorSubscriptOp createSYCLAccessorSubscriptOp(AccessorPtrValue accessor,
                                                      TypedValue<MemRefType> id,
                                                      OpBuilder builder,
                                                      Location loc) {
  const AccessorType accTy = accessor.getAccessorType();
  assert(accTy.getDimension() != 0 && "Dimensions cannot be zero");
  const auto MT = MemRefType::get(
      ShapedType::kDynamic, accTy.getType(), MemRefLayoutAttrInterface(),
      builder.getI64IntegerAttr(targetToAddressSpace(accTy.getTargetMode())));
  return builder.create<SYCLAccessorSubscriptOp>(loc, MT, accessor, id);
}

} // namespace sycl
} // namespace mlir
