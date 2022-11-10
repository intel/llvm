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

// Represents a rhs or lhs value.
class ValueCategory {
private:
  template <typename OpTy>
  ValueCategory Cast(mlir::OpBuilder &Builder, mlir::Type PromotionType) const {
    if (val.getType() == PromotionType)
      return *this;
    return {
        Builder.createOrFold<OpTy>(Builder.getUnknownLoc(), PromotionType, val),
        false};
  }

public:
  mlir::Value val;
  bool isReference;

public:
  ValueCategory() : val(nullptr), isReference(false) {}
  ValueCategory(std::nullptr_t) : val(nullptr), isReference(false) {}
  ValueCategory(mlir::Value val, bool isReference);

  // TODO: rename to 'loadVariable'? getValue seems to generic.
  mlir::Value getValue(mlir::OpBuilder &Builder) const;
  void store(mlir::OpBuilder &Builder, ValueCategory toStore,
             bool isArray) const;
  // TODO: rename to storeVariable?
  void store(mlir::OpBuilder &Builder, mlir::Value toStore) const;
  ValueCategory dereference(mlir::OpBuilder &Builder) const;

  ValueCategory FPTrunc(mlir::OpBuilder &Builder,
                        mlir::Type PromotionType) const;

  ValueCategory FPExt(mlir::OpBuilder &Builder, mlir::Type PromotionType) const;
  ValueCategory IntCast(mlir::OpBuilder &Builder, mlir::Type PromotionType,
                        bool IsSigned) const;
  ValueCategory SIToFP(mlir::OpBuilder &Builder,
                       mlir::Type PromotionType) const;
  ValueCategory UIToFP(mlir::OpBuilder &Builder,
                       mlir::Type PromotionType) const;
  ValueCategory FPToUI(mlir::OpBuilder &Builder,
                       mlir::Type PromotionType) const;
  ValueCategory FPToSI(mlir::OpBuilder &Builder,
                       mlir::Type PromotionType) const;
};

#endif /* CLANG_MLIR_VALUE_CATEGORY */
