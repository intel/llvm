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

using namespace mlir;
using namespace mlir::sycl;

sycl::SYCLIDGetOp sycl::createSYCLIDGetOp(TypedValue<MemRefType> id,
                                          unsigned index, OpBuilder builder,
                                          Location loc) {
  const Value indexOp = builder.create<arith::ConstantIntOp>(loc, index, 32);
  const auto resTy = builder.getIndexType();
  return builder.create<sycl::SYCLIDGetOp>(
      loc, MemRefType::get(ShapedType::kDynamic, resTy), id, indexOp);
}

sycl::SYCLAccessorSubscriptOp
sycl::createSYCLAccessorSubscriptOp(sycl::AccessorPtrValue accessor,
                                    TypedValue<MemRefType> id,
                                    OpBuilder builder, Location loc) {
  const sycl::AccessorType accTy = accessor.getAccessorType();
  assert(accTy.getDimension() != 0 && "Dimensions cannot be zero");
  const auto MT = MemRefType::get(
      ShapedType::kDynamic, accTy.getType(), MemRefLayoutAttrInterface(),
      builder.getI64IntegerAttr(targetToAddressSpace(accTy.getTargetMode())));
  return builder.create<sycl::SYCLAccessorSubscriptOp>(loc, MT, accessor, id);
}
