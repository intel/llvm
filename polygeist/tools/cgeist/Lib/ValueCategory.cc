// Copyright (C) Codeplay Software Limited

//===- ValueCategory.cc ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValueCategory.h"
#include "Lib/TypeUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "polygeist/Ops.h"
#include "utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::arith;

ValueCategory::ValueCategory(mlir::Value val, bool isReference)
    : val(val), isReference(isReference) {
  assert(val && "null value");
  if (isReference) {
    if (!(val.getType().isa<MemRefType>() ||
          val.getType().isa<LLVM::LLVMPointerType>())) {
      llvm::errs() << "val: " << val << "\n";
    }
    assert((val.getType().isa<MemRefType>() ||
            val.getType().isa<LLVM::LLVMPointerType>()) &&
           "Reference value must have pointer/memref type");
  }
}

mlir::Value ValueCategory::getValue(mlir::OpBuilder &builder) const {
  assert(val && "must be not-null");
  if (!isReference)
    return val;
  auto loc = builder.getUnknownLoc();
  if (val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
    return builder.create<mlir::LLVM::LoadOp>(loc, val);
  }
  if (auto mt = val.getType().dyn_cast<mlir::MemRefType>()) {
    assert(mt.getShape().size() == 1 && "must have shape 1");
    auto c0 = builder.create<ConstantIndexOp>(loc, 0);
    return builder.create<memref::LoadOp>(loc, val,
                                          std::vector<mlir::Value>({c0}));
  }
  llvm_unreachable("type must be LLVMPointer or MemRef");
}

void ValueCategory::store(mlir::OpBuilder &builder, mlir::Value toStore) const {
  assert(isReference && "must be a reference");
  assert(val && "expect not-null");
  if (toStore.getType().isInteger(1)) {
    // Ad-hoc extension of booleans
    auto ElementType =
        llvm::TypeSwitch<Type, Type>(val.getType())
            .Case<LLVM::LLVMPointerType>(
                [](auto Ty) -> Type { return Ty.getElementType(); })
            .Case<MemRefType>(
                [](auto Ty) -> Type { return Ty.getElementType(); })
            .Default([](auto) -> Type { llvm_unreachable("Unhandled type"); });
    toStore = builder.createOrFold<arith::ExtUIOp>(builder.getUnknownLoc(),
                                                   ElementType, toStore);
  }
  auto loc = builder.getUnknownLoc();
  if (auto pt = val.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    if (auto p2m = toStore.getDefiningOp<polygeist::Pointer2MemrefOp>()) {
      if (pt.getElementType() == p2m.getSource().getType())
        toStore = p2m.getSource();
      else if (auto nt = p2m.getSource().getDefiningOp<LLVM::NullOp>()) {
        if (pt.getElementType().isa<LLVM::LLVMPointerType>())
          toStore =
              builder.create<LLVM::NullOp>(nt.getLoc(), pt.getElementType());
      }
    }
    if (toStore.getType() != pt.getElementType()) {
      if (auto mt = toStore.getType().dyn_cast<MemRefType>()) {
        if (auto spt =
                pt.getElementType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
          if (mt.getElementType() != spt.getElementType()) {
            // llvm::errs() << " func: " <<
            // val.getDefiningOp()->getParentOfType<FuncOp>() << "\n";
            mlirclang::warning()
                << "potential store type mismatch:\n"
                << "val: " << val << " tosval: " << toStore << "\n"
                << "mt: " << mt << "spt: " << spt << "\n";
          }
          toStore =
              builder.create<polygeist::Memref2PointerOp>(loc, spt, toStore);
        }
      }
    } else { // toStore.getType() == pt.getElementType()
      assert(toStore.getType() == pt.getElementType() && "expect same type");
      builder.create<mlir::LLVM::StoreOp>(loc, toStore, val);
    }
    return;
  }
  if (auto mt = val.getType().dyn_cast<MemRefType>()) {
    assert(mt.getShape().size() == 1 && "must have size 1");
    if (auto PT = toStore.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      if (auto MT = val.getType()
                        .cast<MemRefType>()
                        .getElementType()
                        .dyn_cast<mlir::MemRefType>()) {
        assert(MT.getShape().size() == 1);
        assert(MT.getShape()[0] == -1);
        assert(MT.getElementType() == PT.getElementType());
        toStore = builder.create<polygeist::Pointer2MemrefOp>(loc, MT, toStore);
      }
    }
    if (auto RHS = toStore.getType().dyn_cast<mlir::MemRefType>()) {
      if (auto LHS =
              mt.getElementType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
        assert(LHS.getElementType() == RHS.getElementType() &&
               "Store types mismatch");
        assert(LHS.getAddressSpace() == RHS.getMemorySpaceAsInt() &&
               "Store address spaces mismatch");
        toStore =
            builder.create<polygeist::Memref2PointerOp>(loc, LHS, toStore);
      }
    }
    assert(toStore.getType() ==
               val.getType().cast<MemRefType>().getElementType() &&
           "expect same type");
    auto c0 = builder.create<ConstantIndexOp>(loc, 0);
    builder.create<mlir::memref::StoreOp>(loc, toStore, val,
                                          std::vector<mlir::Value>({c0}));
    return;
  }
  llvm_unreachable("type must be LLVMPointer or MemRef");
}

ValueCategory ValueCategory::dereference(mlir::OpBuilder &builder) const {
  assert(val && "val must be not-null");

  auto loc = builder.getUnknownLoc();
  if (val.getType().isa<mlir::LLVM::LLVMPointerType>()) {
    if (!isReference)
      return ValueCategory(val, /*isReference*/ true);
    else
      return ValueCategory(builder.create<mlir::LLVM::LoadOp>(loc, val),
                           /*isReference*/ true);
  }

  if (auto mt = val.getType().cast<mlir::MemRefType>()) {
    auto c0 = builder.create<ConstantIndexOp>(loc, 0);
    auto shape = std::vector<int64_t>(mt.getShape());

    if (isReference) {
      if (shape.size() > 1) {
        shape.erase(shape.begin());
        auto mt0 = mlir::MemRefType::get(shape, mt.getElementType(),
                                         mt.getLayout(), mt.getMemorySpace());
        return ValueCategory(
            builder.create<polygeist::SubIndexOp>(loc, mt0, val, c0),
            /*isReference*/ true);
      } else {
        // shape[0] = -1;
        return ValueCategory(builder.create<mlir::memref::LoadOp>(
                                 loc, val, std::vector<mlir::Value>({c0})),
                             /*isReference*/ true);
      }
    }
    return ValueCategory(val, /*isReference*/ true);
  }
  llvm_unreachable("type must be LLVMPointer or MemRef");
}

// TODO: too long and difficult to understand.
void ValueCategory::store(mlir::OpBuilder &builder, ValueCategory toStore,
                          bool isArray) const {
  assert(toStore.val);
  if (isArray) {
    if (!toStore.isReference) {
      llvm::errs() << " toStore.val: " << toStore.val << " isref "
                   << toStore.isReference << " isar" << isArray << "\n";
    }
    assert(toStore.isReference);
    auto loc = builder.getUnknownLoc();
    auto zeroIndex = builder.create<ConstantIndexOp>(loc, 0);

    if (auto smt = toStore.val.getType().dyn_cast<mlir::MemRefType>()) {
      assert(smt.getShape().size() <= 2);

      if (auto mt = val.getType().dyn_cast<mlir::MemRefType>()) {
        assert(smt.getElementType() == mt.getElementType());
        if (mt.getShape().size() != smt.getShape().size()) {
          llvm::errs() << " val: " << val << " tsv: " << toStore.val << "\n";
          llvm::errs() << " mt: " << mt << " smt: " << smt << "\n";
        }
        assert(mt.getShape().size() == smt.getShape().size());
        assert(smt.getShape().back() == mt.getShape().back());

        for (ssize_t i = 0; i < smt.getShape().back(); i++) {
          SmallVector<mlir::Value, 2> idx;
          if (smt.getShape().size() == 2)
            idx.push_back(zeroIndex);
          idx.push_back(builder.create<ConstantIndexOp>(loc, i));
          builder.create<mlir::memref::StoreOp>(
              loc, builder.create<mlir::memref::LoadOp>(loc, toStore.val, idx),
              val, idx);
        }
      } else {
        auto pt = val.getType().cast<mlir::LLVM::LLVMPointerType>();
        mlir::Type elty;
        if (auto at = pt.getElementType().dyn_cast<LLVM::LLVMArrayType>()) {
          elty = at.getElementType();
          if (smt.getShape().back() != at.getNumElements()) {
            llvm::errs() << " pt: " << pt << " smt: " << smt << "\n";
            llvm::errs() << " val: " << val << " val.isRef: " << isReference
                         << " ts: " << toStore.val
                         << " ts.isRef: " << toStore.isReference
                         << " isArray: " << isArray << "\n";
          }
          assert(smt.getShape().back() == at.getNumElements());
        } else {
          auto st = pt.getElementType().dyn_cast<LLVM::LLVMStructType>();
          elty = st.getBody()[0];
          assert(smt.getShape().back() == (ssize_t)st.getBody().size());
        }
        if (elty != smt.getElementType()) {
          llvm::errs() << " pt: " << pt << " smt: " << smt << "\n";
          llvm::errs() << " elty: " << elty
                       << " smt.getElementType(): " << smt.getElementType()
                       << "\n";
          llvm::errs() << " val: " << val << " val.isRef: " << isReference
                       << " ts: " << toStore.val
                       << " ts.isRef: " << toStore.isReference
                       << " isArray: " << isArray << "\n";
        }
        assert(elty == smt.getElementType());
        elty = LLVM::LLVMPointerType::get(elty, pt.getAddressSpace());

        auto zero32 = builder.create<ConstantIntOp>(loc, 0, 32);
        for (ssize_t i = 0; i < smt.getShape().back(); i++) {
          SmallVector<mlir::Value, 2> idx;
          if (smt.getShape().size() == 2)
            idx.push_back(zeroIndex);
          idx.push_back(builder.create<ConstantIndexOp>(loc, i));
          mlir::Value lidx[] = {zero32,
                                builder.create<ConstantIntOp>(loc, i, 32)};
          builder.create<mlir::LLVM::StoreOp>(
              loc, builder.create<mlir::memref::LoadOp>(loc, toStore.val, idx),
              builder.create<mlir::LLVM::GEPOp>(loc, elty, val, lidx));
        }
      }
    } else if (auto smt = val.getType().dyn_cast<mlir::MemRefType>()) {
      assert(smt.getShape().size() <= 2);

      auto pt = toStore.val.getType().cast<LLVM::LLVMPointerType>();
      mlir::Type elty;
      if (auto at = pt.getElementType().dyn_cast<LLVM::LLVMArrayType>()) {
        elty = at.getElementType();
        assert(smt.getShape().back() == at.getNumElements());
      } else {
        auto st = pt.getElementType().dyn_cast<LLVM::LLVMStructType>();
        elty = st.getBody()[0];
        assert(smt.getShape().back() == (ssize_t)st.getBody().size());
      }
      assert(elty == smt.getElementType());
      elty = LLVM::LLVMPointerType::get(elty, pt.getAddressSpace());

      auto zero32 = builder.create<ConstantIntOp>(loc, 0, 32);
      for (ssize_t i = 0; i < smt.getShape().back(); i++) {
        SmallVector<mlir::Value, 2> idx;
        if (smt.getShape().size() == 2)
          idx.push_back(zeroIndex);
        idx.push_back(builder.create<ConstantIndexOp>(loc, i));
        mlir::Value lidx[] = {zero32,
                              builder.create<ConstantIntOp>(loc, i, 32)};
        builder.create<mlir::memref::StoreOp>(
            loc,
            builder.create<mlir::LLVM::LoadOp>(
                loc, builder.create<mlir::LLVM::GEPOp>(loc, elty, toStore.val,
                                                       lidx)),
            val, idx);
      }
    } else
      store(builder, toStore.getValue(builder));
  } else {
    store(builder, toStore.getValue(builder));
  }
}

template <typename OpTy> inline void warnUnconstrainedOp() {
  mlirclang::warning() << "Creating unconstrained " << OpTy::getOperationName()
                       << "\n";
}

ValueCategory ValueCategory::FPTrunc(OpBuilder &Builder, Location Loc,
                                     Type PromotionType) const {
  assert(val.getType().isa<FloatType>() &&
         "Expecting floating point source type");
  assert(PromotionType.isa<FloatType>() &&
         "Expecting floating point promotion type");
  assert(val.getType().getIntOrFloatBitWidth() >=
             PromotionType.getIntOrFloatBitWidth() &&
         "Source type must be wider than promotion type");

  warnUnconstrainedOp<arith::TruncFOp>();
  return Cast<arith::TruncFOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::FPExt(OpBuilder &Builder, Location Loc,
                                   Type PromotionType) const {
  assert(val.getType().isa<FloatType>() &&
         "Expecting floating point source type");
  assert(PromotionType.isa<FloatType>() &&
         "Expecting floating point promotion type");
  assert(val.getType().getIntOrFloatBitWidth() <=
             PromotionType.getIntOrFloatBitWidth() &&
         "Source type must be narrower than promotion type");

  warnUnconstrainedOp<arith::ExtFOp>();
  return Cast<arith::ExtFOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::SIToFP(OpBuilder &Builder, Location Loc,
                                    Type PromotionType) const {
  assert(val.getType().isa<IntegerType>() && "Expecting int source type");
  assert(PromotionType.isa<FloatType>() &&
         "Expecting floating point promotion type");

  warnUnconstrainedOp<arith::SIToFPOp>();
  return Cast<arith::SIToFPOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::UIToFP(OpBuilder &Builder, Location Loc,
                                    Type PromotionType) const {
  assert(val.getType().isa<IntegerType>() && "Expecting int source type");
  assert(PromotionType.isa<FloatType>() &&
         "Expecting floating point promotion type");

  warnUnconstrainedOp<arith::UIToFPOp>();
  return Cast<arith::UIToFPOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::FPToUI(OpBuilder &Builder, Location Loc,
                                    Type PromotionType) const {
  assert(val.getType().isa<FloatType>() &&
         "Expecting floating point source type");
  assert(PromotionType.isa<IntegerType>() &&
         "Expecting integer promotion type");

  warnUnconstrainedOp<arith::FPToUIOp>();
  return Cast<arith::FPToUIOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::FPToSI(OpBuilder &Builder, Location Loc,
                                    Type PromotionType) const {
  assert(val.getType().isa<FloatType>() &&
         "Expecting floating point source type");
  assert(PromotionType.isa<IntegerType>() &&
         "Expecting integer promotion type");

  warnUnconstrainedOp<arith::FPToSIOp>();
  return Cast<arith::FPToSIOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::IntCast(OpBuilder &Builder, Location Loc,
                                     Type PromotionType, bool IsSigned) const {
  assert((val.getType().isa<IntegerType, IndexType>()) &&
         "Expecting integer or index source type");
  assert((PromotionType.isa<IntegerType, IndexType>()) &&
         "Expecting integer or index promotion type");

  if (val.getType() == PromotionType)
    return *this;

  auto Res = [&]() -> Value {
    if (val.getType().isa<IndexType>() || PromotionType.isa<IndexType>()) {
      // Special indexcast case
      if (IsSigned)
        return Builder.createOrFold<arith::IndexCastOp>(Loc, PromotionType,
                                                        val);
      return Builder.createOrFold<arith::IndexCastUIOp>(Loc, PromotionType,
                                                        val);
    }

    auto SrcIntTy = val.getType().cast<IntegerType>();
    auto DstIntTy = PromotionType.cast<IntegerType>();

    const unsigned SrcBits = SrcIntTy.getWidth();
    const unsigned DstBits = DstIntTy.getWidth();
    if (SrcBits == DstBits)
      return Builder.createOrFold<arith::BitcastOp>(Loc, PromotionType, val);
    if (SrcBits > DstBits)
      return Builder.createOrFold<arith::TruncIOp>(Loc, PromotionType, val);
    if (IsSigned)
      return Builder.createOrFold<arith::ExtSIOp>(Loc, PromotionType, val);
    return Builder.createOrFold<arith::ExtUIOp>(Loc, PromotionType, val);
  }();

  return {Res, /*IsReference*/ false};
}

ValueCategory ValueCategory::PtrToInt(OpBuilder &Builder, Location Loc,
                                      Type DestTy) const {
  assert(val.getType().isa<LLVM::LLVMPointerType>() &&
         "Expecting pointer source type");
  assert(DestTy.isa<IntegerType>() &&
         "Expecting floating point promotion type");

  return Cast<LLVM::PtrToIntOp>(Builder, Loc, DestTy);
}

ValueCategory ValueCategory::IntToPtr(OpBuilder &Builder, Location Loc,
                                      Type DestTy) const {
  assert(val.getType().isa<IntegerType>() && "Expecting pointer source type");
  assert(DestTy.isa<LLVM::LLVMPointerType>() &&
         "Expecting floating point promotion type");

  return Cast<LLVM::IntToPtrOp>(Builder, Loc, DestTy);
}

ValueCategory ValueCategory::BitCast(OpBuilder &Builder, Location Loc,
                                     Type DestTy) const {
  assert(mlirclang::isFirstClassType(val.getType()) &&
         "Expecting first class type");
  assert(mlirclang::isFirstClassType(DestTy) && "Expecting first class type");
  assert(!mlirclang::isAggregateType(val.getType()) &&
         "Not expecting aggregate type");
  assert(!mlirclang::isAggregateType(DestTy) && "Not expecting aggregate type");
  assert((!mlirclang::isPointerOrMemRefTy(val.getType()) ||
          mlirclang::isPointerOrMemRefTy(DestTy)) &&
         "Cannot cast pointers to anything but pointers");
  assert((mlirclang::isPointerOrMemRefTy(val.getType()) ||
          mlirclang::getPrimitiveSizeInBits(val.getType()) ==
              mlirclang::getPrimitiveSizeInBits(DestTy)) &&
         "Expecting equal bitwidth");
  assert((!mlirclang::isPointerOrMemRefTy(val.getType()) ||
          mlirclang::getAddressSpace(val.getType()) ==
              mlirclang::getAddressSpace(DestTy)) &&
         "Expecting equal address spaces");
  assert((!(val.getType().isa<mlir::VectorType>() &&
            DestTy.isa<mlir::VectorType>()) ||
          val.getType().cast<mlir::VectorType>().getNumElements() ==
              DestTy.cast<mlir::VectorType>().getNumElements()) &&
         "Expecting same number of elements");
  assert((!val.getType().isa<mlir::VectorType>() ||
          val.getType().cast<mlir::VectorType>().getNumElements() == 1) &&
         "Expecting single-element vector");
  assert((!DestTy.isa<mlir::VectorType>() ||
          DestTy.cast<mlir::VectorType>().getNumElements() == 1) &&
         "Expecting single-element vector");

  return Cast<LLVM::BitcastOp>(Builder, Loc, DestTy);
}

ValueCategory ValueCategory::MemRef2Ptr(OpBuilder &Builder,
                                        Location Loc) const {
  const auto Ty = val.getType().dyn_cast<MemRefType>();
  if (!Ty) {
    assert(val.getType().isa<LLVM::LLVMPointerType>() &&
           "Expecting pointer type");
    return *this;
  }

  auto DestTy =
      LLVM::LLVMPointerType::get(Ty.getElementType(), Ty.getMemorySpaceAsInt());
  return {Builder.createOrFold<polygeist::Memref2PointerOp>(Loc, DestTy, val),
          isReference};
}

ValueCategory ValueCategory::Splat(OpBuilder &Builder, Location Loc,
                                   mlir::Type VecTy) const {
  assert(VecTy.isa<mlir::VectorType>() && "Expecting vector type for cast");
  assert(VecTy.cast<mlir::VectorType>().getElementType() == val.getType() &&
         "Cannot splat to a vector of different element type");
  return {Builder.createOrFold<vector::SplatOp>(Loc, val, VecTy), false};
}

ValueCategory ValueCategory::ICmpNE(mlir::OpBuilder &builder, Location Loc,
                                    mlir::Value RHS) const {
  return ICmp(builder, Loc, arith::CmpIPredicate::ne, RHS);
}

ValueCategory ValueCategory::FCmpUNE(mlir::OpBuilder &builder, Location Loc,
                                     mlir::Value RHS) const {
  return FCmp(builder, Loc, arith::CmpFPredicate::UNE, RHS);
}

ValueCategory ValueCategory::ICmp(OpBuilder &builder, Location Loc,
                                  arith::CmpIPredicate predicate,
                                  mlir::Value RHS) const {
  assert(val.getType() == RHS.getType() &&
         "Cannot compare values of different types");
  assert(val.getType().isa<IntegerType>() && "Expecting integer inputs");
  return {builder.createOrFold<arith::CmpIOp>(Loc, predicate, val, RHS), false};
}

ValueCategory ValueCategory::FCmp(OpBuilder &builder, Location Loc,
                                  arith::CmpFPredicate predicate,
                                  mlir::Value RHS) const {
  assert(val.getType() == RHS.getType() &&
         "Cannot compare values of different types");
  assert(val.getType().isa<FloatType>() && "Expecting floating point inputs");
  return {builder.createOrFold<arith::CmpFOp>(Loc, predicate, val, RHS), false};
}

template <typename OpTy>
static ValueCategory IntBinOp(mlir::OpBuilder &Builder, mlir::Location Loc,
                              mlir::Value LHS, mlir::Value RHS) {
  assert(LHS.getType() == RHS.getType() &&
         "Cannot operate on values of different types");
  assert(mlirclang::isIntOrIntVectorTy(LHS.getType()) &&
         "Expecting integers or integer vectors as inputs");
  return {Builder.createOrFold<OpTy>(Loc, LHS, RHS), false};
}

template <typename OpTy>
static ValueCategory FPBinOp(mlir::OpBuilder &Builder, mlir::Location Loc,
                             mlir::Value LHS, mlir::Value RHS) {
  assert(LHS.getType() == RHS.getType() &&
         "Cannot operate on values of different types");
  assert(mlirclang::isFPOrFPVectorTy(LHS.getType()) &&
         "Expecting integers or integer vectors as inputs");

  warnUnconstrainedOp<arith::DivFOp>();

  return {Builder.createOrFold<OpTy>(Loc, LHS, RHS), false};
}

template <typename OpTy>
static ValueCategory NUWNSWBinOp(mlir::OpBuilder &Builder, mlir::Location Loc,
                                 mlir::Value LHS, mlir::Value RHS, bool HasNUW,
                                 bool HasNSW) {
  // No way of adding these flags to MLIR.
  if (HasNUW)
    mlirclang::warning() << "Not adding NUW flag.\n";
  if (HasNSW)
    mlirclang::warning() << "Not adding NSW flag.\n";
  return IntBinOp<OpTy>(Builder, Loc, LHS, RHS);
}

ValueCategory ValueCategory::SDiv(OpBuilder &Builder, Location Loc, Value RHS,
                                  bool IsExact) const {
  if (IsExact)
    mlirclang::warning() << "Creating exact division is not supported\n";
  return IntBinOp<arith::DivSIOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::ExactSDiv(OpBuilder &Builder, Location Loc,
                                       Value RHS) const {
  return SDiv(Builder, Loc, RHS, /*IsExact*/ true);
}

ValueCategory ValueCategory::Neg(OpBuilder &Builder, Location Loc, bool HasNUW,
                                 bool HasNSW) const {
  ValueCategory Zero(Builder.createOrFold<ConstantIntOp>(Loc, 0, val.getType()),
                     /*IsReference*/ false);
  return Zero.Sub(Builder, Loc, val, HasNUW, HasNSW);
}

ValueCategory ValueCategory::Add(OpBuilder &Builder, Location Loc, Value RHS,
                                 bool HasNUW, bool HasNSW) const {
  return NUWNSWBinOp<arith::AddIOp>(Builder, Loc, val, RHS, HasNUW, HasNSW);
}

ValueCategory ValueCategory::FAdd(OpBuilder &Builder, Location Loc,
                                  Value RHS) const {
  return FPBinOp<arith::AddFOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::Sub(OpBuilder &Builder, Location Loc, Value RHS,
                                 bool HasNUW, bool HasNSW) const {
  return NUWNSWBinOp<arith::SubIOp>(Builder, Loc, val, RHS, HasNUW, HasNSW);
}

ValueCategory ValueCategory::FSub(OpBuilder &Builder, Location Loc,
                                  Value RHS) const {
  return FPBinOp<arith::SubFOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::SubIndex(OpBuilder &Builder, Location Loc,
                                      Type Type, Value Index,
                                      bool IsInBounds) const {
  assert(val.getType().isa<MemRefType>() && "Expecting a pointer as operand");
  assert(Index.getType().isa<IndexType>() && "Expecting an index type index");

  if (IsInBounds)
    mlirclang::warning() << "Cannot create an inbounds SubIndex operation\n";

  auto PtrTy = mlirclang::getPtrTyWithNewType(val.getType(), Type);
  return {Builder.createOrFold<polygeist::SubIndexOp>(Loc, PtrTy, val, Index),
          isReference};
}

ValueCategory ValueCategory::InBoundsSubIndex(OpBuilder &Builder, Location Loc,
                                              Type Type, Value Index) const {
  return SubIndex(Builder, Loc, Type, Index, /*IsInBounds*/ true);
}

ValueCategory ValueCategory::GEP(OpBuilder &Builder, Location Loc, Type Type,
                                 ValueRange IdxList, bool IsInBounds) const {
  assert(val.getType().isa<LLVM::LLVMPointerType>() &&
         "Expecting a pointer as operand");
  assert(std::all_of(IdxList.getType().begin(), IdxList.getType().end(),
                     [](mlir::Type Ty) { return Ty.isa<IntegerType>(); }) &&
         "Expecting integer indices");

  if (IsInBounds)
    mlirclang::warning() << "Cannot create an inbounds GEP operation\n";

  auto PtrTy = mlirclang::getPtrTyWithNewType(val.getType(), Type);
  return {Builder.createOrFold<LLVM::GEPOp>(Loc, PtrTy, val, IdxList),
          isReference};
}

ValueCategory ValueCategory::InBoundsGEP(OpBuilder &Builder, Location Loc,
                                         Type Type, ValueRange IdxList) const {
  return GEP(Builder, Loc, Type, IdxList, /*IsInBounds*/ true);
}

ValueCategory ValueCategory::GEPOrSubIndex(OpBuilder &Builder, Location Loc,
                                           Type Type, ValueRange IdxList,
                                           bool IsInBounds) const {
  const auto ValType = val.getType();
  assert((ValType.isa<LLVM::LLVMPointerType, MemRefType>()) &&
         "Expecting an LLVMPointer or MemRefType input");

  return llvm::TypeSwitch<mlir::Type, ValueCategory>(ValType)
      .Case<MemRefType>([&](auto) {
        assert(IdxList.size() == 1 && "SubIndexOp expects a single index");
        return SubIndex(Builder, Loc, Type, IdxList[0], IsInBounds);
      })
      .Case<LLVM::LLVMPointerType>(
          [&](auto) { return GEP(Builder, Loc, Type, IdxList, IsInBounds); })
      .Default([](auto) -> ValueCategory { llvm_unreachable("Invalid type"); });
}

ValueCategory ValueCategory::InBoundsGEPOrSubIndex(OpBuilder &Builder,
                                                   Location Loc, Type Type,
                                                   ValueRange IdxList) const {
  return GEPOrSubIndex(Builder, Loc, Type, IdxList, /*IsInBounds*/ true);
}
