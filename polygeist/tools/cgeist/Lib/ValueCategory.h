//===- ValueCategory.h -------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_MLIR_VALUE_CATEGORY
#define CLANG_MLIR_VALUE_CATEGORY

#include "Lib/ConstantFolder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/Optional.h"
#include <type_traits>

// Represents a rhs or lhs value.
class ValueCategory {
private:
  template <typename OpTy>
  ValueCategory Cast(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Type PromotionType,
                     std::optional<mlir::Type> ElemTy = std::nullopt) const {
    if (val.getType() == PromotionType)
      return *this;
    if (const auto C = val.getDefiningOp<mlir::arith::ConstantOp>()) {
      mlirclang::ConstantFolder Folder{Builder};
      if (const mlir::Value FoldedRes =
              Folder.fold<OpTy>(Loc, PromotionType, C))
        return {FoldedRes, false};
    }
    // If we're casting a pointer to an integer, the result won't be a reference
    // anymore.
    bool ResIsReference =
        !std::is_same_v<OpTy, mlir::LLVM::PtrToIntOp> && isReference;
    return {Builder.createOrFold<OpTy>(Loc, PromotionType, val), ResIsReference,
            ElemTy};
  }

  mlir::Type getPointerType(mlir::Type ElementType,
                            unsigned AddressSpace) const;

public:
  mlir::Value val;
  bool isReference;
  /// Holds the index the lvalue to a vector element refers to.
  llvm::Optional<mlir::Value> Index{std::nullopt};
  /// Holds the element type of memrefs or pointers. This is particularly
  /// important with opaque pointers.
  std::optional<mlir::Type> ElementType{std::nullopt};

  mlir::Type getElemTy() const {
    assert(ElementType && "No element type defined");
    return ElementType.value();
  }

public:
  ValueCategory() : val(nullptr), isReference(false) {}
  ValueCategory(std::nullptr_t) : val(nullptr), isReference(false) {}
  ValueCategory(mlir::Value val, bool isReference,
                std::optional<mlir::Type> elementType = std::nullopt);
  ValueCategory(mlir::Value Val, mlir::Value Index,
                std::optional<mlir::Type> elementType = std::nullopt);

  static ValueCategory getNullValue(mlir::OpBuilder &Builder,
                                    mlir::Location Loc, mlir::Type Type);
  static ValueCategory getUndefValue(mlir::OpBuilder &Builder,
                                     mlir::Location Loc, mlir::Type Type);

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
                         mlir::Type DestTy, mlir::Type ElemType) const;
  ValueCategory BitCast(mlir::OpBuilder &Builder, mlir::Location Loc,
                        mlir::Type DestTy, mlir::Type ElemTy) const;
  ValueCategory MemRef2Ptr(mlir::OpBuilder &Builder, mlir::Location Loc) const;
  ValueCategory
  Ptr2MemRef(mlir::OpBuilder &Builder, mlir::Location Loc,
             llvm::ArrayRef<int64_t> Shape = mlir::ShapedType::kDynamic,
             mlir::MemRefLayoutAttrInterface Layout = {}) const;

  ValueCategory Splat(mlir::OpBuilder &Builder, mlir::Location Loc,
                      mlir::Type VecTy) const;

  ValueCategory ICmp(mlir::OpBuilder &builder, mlir::Location Loc,
                     mlir::arith::CmpIPredicate predicate,
                     mlir::Value RHS) const;
  ValueCategory FCmp(mlir::OpBuilder &builder, mlir::Location Loc,
                     mlir::arith::CmpFPredicate predicate,
                     mlir::Value RHS) const;

  ValueCategory ICmpNE(mlir::OpBuilder &builder, mlir::Location Loc,
                       mlir::Value RHS) const;
  ValueCategory FCmpUNE(mlir::OpBuilder &builder, mlir::Location Loc,
                        mlir::Value RHS) const;

  ValueCategory Mul(mlir::OpBuilder &Builder, mlir::Location Loc,
                    mlir::Value RHS, bool HasNUW = false,
                    bool HasNSW = false) const;
  ValueCategory FMul(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS) const;

  ValueCategory UDiv(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS, bool IsExact = false) const;
  ValueCategory SDiv(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS, bool IsExact = false) const;
  ValueCategory ExactSDiv(mlir::OpBuilder &Builder, mlir::Location Loc,
                          mlir::Value RHS) const;

  ValueCategory ExactUDiv(mlir::OpBuilder &Builder, mlir::Location Loc,
                          mlir::Value RHS) const;
  ValueCategory FDiv(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS) const;

  ValueCategory URem(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS) const;
  ValueCategory SRem(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS) const;

  ValueCategory LShr(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS, bool IsExact = false) const;
  ValueCategory AShr(mlir::OpBuilder &Builder, mlir::Location Loc,
                     mlir::Value RHS, bool IsExact = false) const;
  ValueCategory Shl(mlir::OpBuilder &Builder, mlir::Location Loc,
                    mlir::Value RHS, bool HasNUW = false,
                    bool HasNSW = false) const;

  ValueCategory FNeg(mlir::OpBuilder &Builder, mlir::Location Loc) const;
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

  ValueCategory And(mlir::OpBuilder &Builder, mlir::Location Loc,
                    mlir::Value RHS) const;
  ValueCategory Or(mlir::OpBuilder &Builder, mlir::Location Loc,
                   mlir::Value RHS) const;
  ValueCategory Xor(mlir::OpBuilder &Builder, mlir::Location Loc,
                    mlir::Value RHS) const;

  ValueCategory Insert(mlir::OpBuilder &Builder, mlir::Location Loc,
                       mlir::Value V, llvm::ArrayRef<int64_t> Indices) const;
  ValueCategory InsertElement(mlir::OpBuilder &Builder, mlir::Location Loc,
                              mlir::Value V, mlir::Value Idx) const;
  ValueCategory Extract(mlir::OpBuilder &Builder, mlir::Location Loc,
                        llvm::ArrayRef<int64_t> Indices) const;
  ValueCategory ExtractElement(mlir::OpBuilder &Builder, mlir::Location Loc,
                               mlir::Value Idx) const;
  ValueCategory Shuffle(mlir::OpBuilder &Builder, mlir::Location Loc,
                        mlir::Value V2, llvm::ArrayRef<int64_t> Indices) const;
  ValueCategory Reshape(mlir::OpBuilder &Builder, mlir::Location Loc,
                        llvm::ArrayRef<int64_t> Shape) const;
};

#endif /* CLANG_MLIR_VALUE_CATEGORY */
