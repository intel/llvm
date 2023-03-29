//===- ConstantFolder.cc -----------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ConstantFolder.h"

using namespace mlir;
using namespace mlirclang;

Value ConstantFolder::foldFPCast(Location Loc, Type PromotionType,
                                 arith::ConstantOp C) {
  assert(isa<FloatType>(PromotionType) && "Expecting FP type");
  const auto Attr = cast<FloatAttr>(C.getValueAttr());
  const auto NewAttr =
      FloatAttr::get(PromotionType, Attr.getValue().convertToDouble());
  return Builder.createOrFold<arith::ConstantOp>(Loc, NewAttr, PromotionType);
}
