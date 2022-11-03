//===- ValueCategory.h -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_MLIR_VALUE_CATEGORY
#define CLANG_MLIR_VALUE_CATEGORY

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/WithColor.h"

// Represents a rhs or lhs value.
class ValueCategory {
public:
  mlir::Value val;
  bool isReference;

  template <typename OpTy>
  ValueCategory Cast(mlir::OpBuilder &builder, mlir::Type PromotionType) const {
    if (val.getType() == PromotionType)
      return *this;
    return {builder.createOrFold<OpTy>(val.getLoc(), PromotionType, val),
            false};
  }

public:
  ValueCategory() : val(nullptr), isReference(false) {}
  ValueCategory(std::nullptr_t) : val(nullptr), isReference(false) {}
  ValueCategory(mlir::Value val, bool isReference);

  // TODO: rename to 'loadVariable'? getValue seems to generic.
  mlir::Value getValue(mlir::OpBuilder &builder) const;
  void store(mlir::OpBuilder &builder, ValueCategory toStore,
             bool isArray) const;
  // TODO: rename to storeVariable?
  void store(mlir::OpBuilder &builder, mlir::Value toStore) const;
  ValueCategory dereference(mlir::OpBuilder &builder) const;

  ValueCategory FPTrunc(mlir::OpBuilder &builder,
                        mlir::Type PromotionType) const;

  ValueCategory FPExt(mlir::OpBuilder &builder, mlir::Type PromotionType) const;
};

#endif /* CLANG_MLIR_VALUE_CATEGORY */
