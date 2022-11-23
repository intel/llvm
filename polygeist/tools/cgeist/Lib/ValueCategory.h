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
  ValueCategory Cast(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Type PromotionType) const {
    if (val.getType() == PromotionType)
      return *this;
    return {Builder.createOrFold<OpTy>(Loc, PromotionType, val), false};
  }

  ValueCategory ICmp(mlir::OpBuilder &builder, mlir::Location Loc,
                     mlir::arith::CmpIPredicate predicate,
                     mlir::Value RHS) const;
  ValueCategory FCmp(mlir::OpBuilder &builder, mlir::Location Loc,
                     mlir::arith::CmpFPredicate predicate,
                     mlir::Value RHS) const;

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

  ValueCategory SubIndex(mlir::OpBuilder &Builder, mlir::Location Loc,
                         mlir::Type Type, mlir::Value Index,
                         bool IsInBounds = false) const;
  ValueCategory InBoundsSubIndex(mlir::OpBuilder &Builder, mlir::Location Loc,
                                 mlir::Type Type, mlir::Value Index) const;

  ValueCategory GEP(mlir::OpBuilder &Builder, mlir::Location Loc,
                    mlir::Type Type, mlir::ValueRange IdxList,
                    bool IsInBounds = false) const;
  ValueCategory InBoundsGEP(mlir::OpBuilder &Builder, mlir::Location Loc,
                            mlir::Type Type, mlir::ValueRange IdxList) const;

  ValueCategory GEPOrSubIndex(mlir::OpBuilder &Builder, mlir::Location Loc,
                              mlir::Type Type, mlir::ValueRange IdxList,
                              bool IsInBounds = false) const;
  ValueCategory InBoundsGEPOrSubIndex(mlir::OpBuilder &Builder,
                                      mlir::Location Loc, mlir::Type Type,
                                      mlir::ValueRange IdxList) const;

  ValueCategory FPTrunc(mlir::OpBuilder &Builder, mlir::Location Loc,
                        mlir::Type PromotionType) const;

  ValueCategory FPExt(mlir::OpBuilder &Builder, mlir::Location Loc,
                      mlir::Type PromotionType) const;
  ValueCategory IntCast(mlir::OpBuilder &Builder, mlir::Location Loc,
                        mlir::Type PromotionType, bool IsSigned) const;
  ValueCategory SIToFP(mlir::OpBuilder &Builder, mlir::Location Loc,
                       mlir::Type PromotionType) const;
  ValueCategory UIToFP(mlir::OpBuilder &Builder, mlir::Location Loc,
                       mlir::Type PromotionType) const;
  ValueCategory FPToUI(mlir::OpBuilder &Builder, mlir::Location Loc,
                       mlir::Type PromotionType) const;
  ValueCategory FPToSI(mlir::OpBuilder &Builder, mlir::Location Loc,
                       mlir::Type PromotionType) const;
  ValueCategory PtrToInt(mlir::OpBuilder &Builder, mlir::Location Loc,
                         mlir::Type DestTy) const;
  ValueCategory IntToPtr(mlir::OpBuilder &Builder, mlir::Location Loc,
                         mlir::Type DestTy) const;
  ValueCategory BitCast(mlir::OpBuilder &Builder, mlir::Location Loc,
                        mlir::Type DestTy) const;
  ValueCategory MemRef2Ptr(mlir::OpBuilder &Builder, mlir::Location Loc) const;

  ValueCategory Splat(mlir::OpBuilder &Builder, mlir::Location Loc,
                      mlir::Type VecTy) const;

  ValueCategory ICmpNE(mlir::OpBuilder &builder, mlir::Location Loc,
                       mlir::Value RHS) const;
  ValueCategory FCmpUNE(mlir::OpBuilder &builder, mlir::Location Loc,
                        mlir::Value RHS) const;

  ValueCategory SDiv(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS, bool IsExact = false) const;
  ValueCategory ExactSDiv(mlir::OpBuilder &Builder, mlir::Location Loc,
                          mlir::Value RHS) const;
  ValueCategory Neg(mlir::OpBuilder &Builder, mlir::Location Loc,
                    bool HasNUW = false, bool HasNSW = false) const;
  ValueCategory Add(mlir::OpBuilder &Builder, mlir::Location Loc,
                    mlir::Value RHS, bool HasNUW = false,
                    bool HasNSW = false) const;
  ValueCategory FAdd(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS) const;
  ValueCategory Sub(mlir::OpBuilder &Builder, mlir::Location Loc,
                    mlir::Value RHS, bool HasNUW = false,
                    bool HasNSW = false) const;
  ValueCategory FSub(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS) const;
};

#endif /* CLANG_MLIR_VALUE_CATEGORY */
