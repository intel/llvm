//===- ValueCategory.cc ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValueCategory.h"
#include "Lib/TypeUtils.h"
#include "mlir/Dialect/Polygeist/IR/Ops.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/WithColor.h"

using namespace mlir;
using namespace mlir::arith;

ValueCategory::ValueCategory(mlir::Value val, bool isReference)
    : val(val), isReference(isReference) {
  assert(val && "null value");
  if (isReference) {
    if (!(isa<MemRefType>(val.getType()) ||
          isa<LLVM::LLVMPointerType>(val.getType()))) {
      llvm::errs() << "val: " << val << "\n";
    }
    assert((isa<MemRefType>(val.getType()) ||
            isa<LLVM::LLVMPointerType>(val.getType())) &&
           "Reference value must have pointer/memref type");
  }
}

ValueCategory::ValueCategory(mlir::Value Val, mlir::Value Index)
    : val{Val}, isReference{true}, Index{Index} {
  assert(((isa<MemRefType>(val.getType()) &&
           isa<VectorType>(cast<MemRefType>(val.getType()).getElementType())) ||
          (isa<LLVM::LLVMPointerType>(val.getType()) &&
           isa<VectorType>(
               cast<LLVM::LLVMPointerType>(val.getType()).getElementType()))) &&
         "Expecting memref/pointer of vector");
  assert(isa<IntegerType>(Index.getType()) && "Expecting integer index");
}

mlir::Value ValueCategory::getValue(mlir::OpBuilder &builder) const {
  assert(val && "must be not-null");
  if (!isReference)
    return val;
  auto loc = builder.getUnknownLoc();
  if (isa<mlir::LLVM::LLVMPointerType>(val.getType())) {
    return builder.create<mlir::LLVM::LoadOp>(loc, val);
  }
  if (auto mt = dyn_cast<mlir::MemRefType>(val.getType())) {
    assert(mt.getShape().size() == 1 && "must have shape 1");
    auto c0 = builder.create<ConstantIndexOp>(loc, 0);
    Value Loaded = builder.create<memref::LoadOp>(
        loc, val, std::vector<mlir::Value>({c0}));
    if (Index)
      Loaded =
          ValueCategory{Loaded, false}.ExtractElement(builder, loc, *Index).val;
    return Loaded;
  }
  llvm_unreachable("type must be LLVMPointer or MemRef");
}

ValueCategory ValueCategory::getNullValue(OpBuilder &Builder, Location Loc,
                                          Type Type) {
  const auto ZeroVal = static_cast<mlir::Value>(
      llvm::TypeSwitch<mlir::Type, mlir::Value>(Type)
          .Case<mlir::IntegerType>([&](auto Ty) {
            return Builder.createOrFold<arith::ConstantIntOp>(Loc, 0, Ty);
          })
          .Case<mlir::IndexType>([&](auto) {
            return Builder.createOrFold<arith::ConstantIndexOp>(Loc, 0);
          })
          .Case<mlir::FloatType>([&](auto Ty) {
            return Builder.createOrFold<arith::ConstantFloatOp>(
                Loc, llvm::APFloat::getZero(Ty.getFloatSemantics()), Ty);
          })
          .Case<mlir::VectorType>([&](auto VecTy) {
            const auto Element = ValueCategory::getNullValue(
                                     Builder, Loc, VecTy.getElementType())
                                     .val;
            return Builder.createOrFold<vector::SplatOp>(Loc, Element, Type);
          }));
  return {ZeroVal, false};
}

ValueCategory ValueCategory::getUndefValue(OpBuilder &Builder, Location Loc,
                                           Type Type) {
  // TODO: Replace with higher-level undef operation when defined.
  return {Builder.createOrFold<LLVM::UndefOp>(Loc, Type), false};
}

void ValueCategory::store(mlir::OpBuilder &builder, mlir::Value toStore) const {
  assert(isReference && "must be a reference");
  assert(val && "expect not-null");
  if (toStore.getType().isInteger(1)) {
    // Ad-hoc extension of booleans
    auto ElementType = static_cast<Type>(
        llvm::TypeSwitch<Type, Type>{val.getType()}
            .Case<MemRefType, LLVM::LLVMPointerType>(
                [](auto Ty) -> Type { return Ty.getElementType(); }));
    toStore = builder.createOrFold<arith::ExtUIOp>(builder.getUnknownLoc(),
                                                   ElementType, toStore);
  }
  auto loc = builder.getUnknownLoc();
  if (auto pt = dyn_cast<mlir::LLVM::LLVMPointerType>(val.getType())) {
    if (auto p2m = toStore.getDefiningOp<polygeist::Pointer2MemrefOp>()) {
      if (pt.getElementType() == p2m.getSource().getType())
        toStore = p2m.getSource();
      else if (auto nt = p2m.getSource().getDefiningOp<LLVM::NullOp>()) {
        if (isa<LLVM::LLVMPointerType>(pt.getElementType()))
          toStore =
              builder.create<LLVM::NullOp>(nt.getLoc(), pt.getElementType());
      }
    }

    if (Index) {
      auto ElemTy =
          cast<mlir::VectorType>(pt.getElementType()).getElementType();
      assert(ElemTy == toStore.getType() &&
             "Vector insertion element mismatch");
      ValueCategory Vec{builder.create<mlir::LLVM::LoadOp>(loc, val), false};
      Vec = Vec.InsertElement(builder, loc, toStore, *Index);
      toStore = Vec.val;
    }

    if (toStore.getType() != pt.getElementType()) {
      if (auto mt = dyn_cast<MemRefType>(toStore.getType())) {
        if (auto spt =
                dyn_cast<mlir::LLVM::LLVMPointerType>(pt.getElementType())) {
          CGEIST_WARNING({
            if (mt.getElementType() != spt.getElementType()) {
              // llvm::errs() << " func: " <<
              // val.getDefiningOp()->getParentOfType<FuncOp>() << "\n";
              llvm::WithColor::warning()
                  << "potential store type mismatch:\n"
                  << "val: " << val << " tosval: " << toStore << "\n"
                  << "mt: " << mt << "spt: " << spt << "\n";
            }
          });
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
  if (auto mt = dyn_cast<MemRefType>(val.getType())) {
    assert(mt.getShape().size() == 1 && "must have size 1");

    if (Index) {
      auto ElemTy =
          cast<mlir::VectorType>(mt.getElementType()).getElementType();
      assert(ElemTy == toStore.getType() &&
             "Vector insertion element mismatch");
      const auto C0 = builder.createOrFold<arith::ConstantIntOp>(
          loc, 0, builder.getI64Type());
      ValueCategory Vec{builder.createOrFold<memref::LoadOp>(loc, val, C0),
                        false};
      Vec = Vec.InsertElement(builder, loc, toStore, *Index);
      toStore = Vec.val;
    }

    if (auto PT = dyn_cast<mlir::LLVM::LLVMPointerType>(toStore.getType())) {
      if (auto MT = dyn_cast<mlir::MemRefType>(
              cast<MemRefType>(val.getType()).getElementType())) {
        assert(MT.getShape().size() == 1);
        assert(MT.getShape()[0] == ShapedType::kDynamic);
        assert(MT.getElementType() == PT.getElementType());
        toStore = builder.create<polygeist::Pointer2MemrefOp>(loc, MT, toStore);
      }
    }
    if (auto RHS = dyn_cast<mlir::MemRefType>(toStore.getType())) {
      if (auto LHS =
              dyn_cast<mlir::LLVM::LLVMPointerType>(mt.getElementType())) {
        assert(LHS.getElementType() == RHS.getElementType() &&
               "Store types mismatch");
        assert(LHS.getAddressSpace() == RHS.getMemorySpaceAsInt() &&
               "Store address spaces mismatch");
        toStore =
            builder.create<polygeist::Memref2PointerOp>(loc, LHS, toStore);
      }
    }
    assert(toStore.getType() ==
               cast<MemRefType>(val.getType()).getElementType() &&
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
  if (isa<mlir::LLVM::LLVMPointerType>(val.getType())) {
    if (!isReference)
      return ValueCategory(val, /*isReference*/ true);
    else
      return ValueCategory(builder.create<mlir::LLVM::LoadOp>(loc, val),
                           /*isReference*/ true);
  }

  if (auto mt = cast<mlir::MemRefType>(val.getType())) {
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
        // shape[0] = ShapedType::kDynamic;
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

    if (auto smt = dyn_cast<mlir::MemRefType>(toStore.val.getType())) {
      assert(smt.getShape().size() <= 2);

      if (auto mt = dyn_cast<mlir::MemRefType>(val.getType())) {
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
        auto pt = cast<mlir::LLVM::LLVMPointerType>(val.getType());
        mlir::Type elty;
        if (auto at = dyn_cast<LLVM::LLVMArrayType>(pt.getElementType())) {
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
          auto st = dyn_cast<LLVM::LLVMStructType>(pt.getElementType());
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
              builder.create<mlir::LLVM::GEPOp>(loc, elty, val, lidx,
                                                /* inbounds */ true));
        }
      }
    } else if (auto smt = dyn_cast<mlir::MemRefType>(val.getType())) {
      assert(smt.getShape().size() <= 2);

      auto pt = cast<LLVM::LLVMPointerType>(toStore.val.getType());
      mlir::Type elty;
      if (auto at = dyn_cast<LLVM::LLVMArrayType>(pt.getElementType())) {
        elty = at.getElementType();
        assert(smt.getShape().back() == at.getNumElements());
      } else {
        auto st = dyn_cast<LLVM::LLVMStructType>(pt.getElementType());
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
                loc, builder.create<mlir::LLVM::GEPOp>(
                         loc, elty, toStore.val, lidx, /* inbounds */ true)),
            val, idx);
      }
    } else
      store(builder, toStore.getValue(builder));
  } else {
    store(builder, toStore.getValue(builder));
  }
}

template <typename OpTy> inline void warnUnconstrainedOp() {
  llvm::WithColor::warning()
      << "Creating unconstrained " << OpTy::getOperationName() << "\n";
}

template <typename OpTy> inline void warnNonExactOp(bool IsExact) {
  if (!IsExact)
    return;
  llvm::WithColor::warning()
      << "Creating exact " << OpTy::getOperationName() << " is not suported.\n";
}

ValueCategory ValueCategory::FPTrunc(OpBuilder &Builder, Location Loc,
                                     Type PromotionType) const {
  assert(isa<FloatType>(val.getType()) &&
         "Expecting floating point source type");
  assert(isa<FloatType>(PromotionType) &&
         "Expecting floating point promotion type");
  assert(val.getType().getIntOrFloatBitWidth() >=
             PromotionType.getIntOrFloatBitWidth() &&
         "Source type must be wider than promotion type");

  CGEIST_WARNING(warnUnconstrainedOp<arith::TruncFOp>());
  return Cast<arith::TruncFOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::FPExt(OpBuilder &Builder, Location Loc,
                                   Type PromotionType) const {
  assert(isa<FloatType>(val.getType()) &&
         "Expecting floating point source type");
  assert(isa<FloatType>(PromotionType) &&
         "Expecting floating point promotion type");
  assert(val.getType().getIntOrFloatBitWidth() <=
             PromotionType.getIntOrFloatBitWidth() &&
         "Source type must be narrower than promotion type");

  CGEIST_WARNING(warnUnconstrainedOp<arith::ExtFOp>());
  return Cast<arith::ExtFOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::SIToFP(OpBuilder &Builder, Location Loc,
                                    Type PromotionType) const {
  assert(isa<IntegerType>(val.getType()) && "Expecting int source type");
  assert(isa<FloatType>(PromotionType) &&
         "Expecting floating point promotion type");

  CGEIST_WARNING(warnUnconstrainedOp<arith::SIToFPOp>());
  return Cast<arith::SIToFPOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::UIToFP(OpBuilder &Builder, Location Loc,
                                    Type PromotionType) const {
  assert(isa<IntegerType>(val.getType()) && "Expecting int source type");
  assert(isa<FloatType>(PromotionType) &&
         "Expecting floating point promotion type");

  CGEIST_WARNING(warnUnconstrainedOp<arith::UIToFPOp>());
  return Cast<arith::UIToFPOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::FPToUI(OpBuilder &Builder, Location Loc,
                                    Type PromotionType) const {
  assert(isa<FloatType>(val.getType()) &&
         "Expecting floating point source type");
  assert(isa<IntegerType>(PromotionType) && "Expecting integer promotion type");

  CGEIST_WARNING(warnUnconstrainedOp<arith::FPToUIOp>());
  return Cast<arith::FPToUIOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::FPToSI(OpBuilder &Builder, Location Loc,
                                    Type PromotionType) const {
  assert(isa<FloatType>(val.getType()) &&
         "Expecting floating point source type");
  assert(isa<IntegerType>(PromotionType) && "Expecting integer promotion type");

  CGEIST_WARNING(warnUnconstrainedOp<arith::FPToSIOp>());
  return Cast<arith::FPToSIOp>(Builder, Loc, PromotionType);
}

ValueCategory ValueCategory::IntCast(OpBuilder &Builder, Location Loc,
                                     Type PromotionType, bool IsSigned) const {
  assert((isa<IntegerType, IndexType>(val.getType())) &&
         "Expecting integer or index source type");
  assert((isa<IntegerType, IndexType>(PromotionType)) &&
         "Expecting integer or index promotion type");

  if (val.getType() == PromotionType)
    return *this;

  auto Res = [&]() -> Value {
    if (isa<IndexType>(val.getType()) || isa<IndexType>(PromotionType)) {
      // Special indexcast case
      if (IsSigned)
        return Builder.createOrFold<arith::IndexCastOp>(Loc, PromotionType,
                                                        val);
      return Builder.createOrFold<arith::IndexCastUIOp>(Loc, PromotionType,
                                                        val);
    }

    auto SrcIntTy = cast<IntegerType>(val.getType());
    auto DstIntTy = cast<IntegerType>(PromotionType);

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
  assert(isa<LLVM::LLVMPointerType>(val.getType()) &&
         "Expecting pointer source type");
  assert(isa<IntegerType>(DestTy) && "Expecting floating point promotion type");

  return Cast<LLVM::PtrToIntOp>(Builder, Loc, DestTy);
}

ValueCategory ValueCategory::IntToPtr(OpBuilder &Builder, Location Loc,
                                      Type DestTy) const {
  assert(isa<IntegerType>(val.getType()) && "Expecting pointer source type");
  assert(isa<LLVM::LLVMPointerType>(DestTy) &&
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
  assert((!(isa<mlir::VectorType>(val.getType()) &&
            isa<mlir::VectorType>(DestTy)) ||
          cast<mlir::VectorType>(val.getType()).getNumElements() ==
              cast<mlir::VectorType>(DestTy).getNumElements()) &&
         "Expecting same number of elements");
  assert((!isa<mlir::VectorType>(val.getType()) ||
          cast<mlir::VectorType>(val.getType()).getNumElements() == 1) &&
         "Expecting single-element vector");
  assert((!isa<mlir::VectorType>(DestTy) ||
          cast<mlir::VectorType>(DestTy).getNumElements() == 1) &&
         "Expecting single-element vector");

  return Cast<LLVM::BitcastOp>(Builder, Loc, DestTy);
}

ValueCategory ValueCategory::MemRef2Ptr(OpBuilder &Builder,
                                        Location Loc) const {
  const auto Ty = dyn_cast<MemRefType>(val.getType());
  if (!Ty) {
    assert(isa<LLVM::LLVMPointerType>(val.getType()) &&
           "Expecting pointer type");
    return *this;
  }

  auto DestTy =
      LLVM::LLVMPointerType::get(Ty.getElementType(), Ty.getMemorySpaceAsInt());
  return {Builder.createOrFold<polygeist::Memref2PointerOp>(Loc, DestTy, val),
          isReference};
}

ValueCategory
ValueCategory::Ptr2MemRef(OpBuilder &Builder, Location Loc,
                          llvm::ArrayRef<int64_t> Shape,
                          MemRefLayoutAttrInterface Layout) const {
  const auto Ty = dyn_cast<LLVM::LLVMPointerType>(val.getType());
  if (!Ty) {
    assert(isa<MemRefType>(val.getType()) && "Expecting MemRef type");
    return *this;
  }

  auto DestTy =
      MemRefType::get(Shape, Ty.getElementType(), Layout,
                      Builder.getI32IntegerAttr(Ty.getAddressSpace()));
  return {Builder.createOrFold<polygeist::Pointer2MemrefOp>(Loc, DestTy, val),
          isReference};
}

ValueCategory ValueCategory::Splat(OpBuilder &Builder, Location Loc,
                                   mlir::Type VecTy) const {
  assert(isa<mlir::VectorType>(VecTy) && "Expecting vector type for cast");
  assert(cast<mlir::VectorType>(VecTy).getElementType() == val.getType() &&
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
  assert(isa<IntegerType>(val.getType()) && "Expecting integer inputs");
  return {builder.createOrFold<arith::CmpIOp>(Loc, predicate, val, RHS), false};
}

ValueCategory ValueCategory::FCmp(OpBuilder &builder, Location Loc,
                                  arith::CmpFPredicate predicate,
                                  mlir::Value RHS) const {
  assert(val.getType() == RHS.getType() &&
         "Cannot compare values of different types");
  assert(isa<FloatType>(val.getType()) && "Expecting floating point inputs");
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

  CGEIST_WARNING(warnUnconstrainedOp<arith::DivFOp>());

  return {Builder.createOrFold<OpTy>(Loc, LHS, RHS), false};
}

template <typename OpTy>
static ValueCategory NUWNSWBinOp(mlir::OpBuilder &Builder, mlir::Location Loc,
                                 mlir::Value LHS, mlir::Value RHS, bool HasNUW,
                                 bool HasNSW) {
  // No way of adding these flags to MLIR.
  CGEIST_WARNING({
    if (HasNUW)
      llvm::WithColor::warning() << "Not adding NUW flag.\n";
    if (HasNSW)
      llvm::WithColor::warning() << "Not adding NSW flag.\n";
  })
  return IntBinOp<OpTy>(Builder, Loc, LHS, RHS);
}

ValueCategory ValueCategory::Mul(OpBuilder &Builder, Location Loc, Value RHS,
                                 bool HasNUW, bool HasNSW) const {
  return NUWNSWBinOp<arith::MulIOp>(Builder, Loc, val, RHS, HasNUW, HasNSW);
}

ValueCategory ValueCategory::FMul(OpBuilder &Builder, Location Loc,
                                  Value RHS) const {
  CGEIST_WARNING(warnUnconstrainedOp<arith::DivFOp>());
  return FPBinOp<arith::MulFOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::FDiv(OpBuilder &Builder, Location Loc,
                                  Value RHS) const {
  CGEIST_WARNING(warnUnconstrainedOp<arith::DivFOp>());
  return FPBinOp<arith::DivFOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::UDiv(OpBuilder &Builder, Location Loc, Value RHS,
                                  bool IsExact) const {
  CGEIST_WARNING(warnNonExactOp<arith::DivUIOp>(IsExact));
  return IntBinOp<arith::DivUIOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::ExactUDiv(OpBuilder &Builder, Location Loc,
                                       Value RHS) const {
  return UDiv(Builder, Loc, RHS, /*IsExact*/ true);
}

ValueCategory ValueCategory::SDiv(OpBuilder &Builder, Location Loc, Value RHS,
                                  bool IsExact) const {
  CGEIST_WARNING(warnNonExactOp<arith::DivSIOp>(IsExact));
  return IntBinOp<arith::DivSIOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::ExactSDiv(OpBuilder &Builder, Location Loc,
                                       Value RHS) const {
  return SDiv(Builder, Loc, RHS, /*IsExact*/ true);
}

ValueCategory ValueCategory::URem(OpBuilder &Builder, Location Loc,
                                  Value RHS) const {
  return IntBinOp<arith::RemUIOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::SRem(OpBuilder &Builder, Location Loc,
                                  Value RHS) const {
  return IntBinOp<arith::RemSIOp>(Builder, Loc, val, RHS);
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
  assert(isa<MemRefType>(val.getType()) && "Expecting a pointer as operand");
  assert(isa<IndexType>(Index.getType()) && "Expecting an index type index");

  CGEIST_WARNING({
    if (IsInBounds)
      llvm::WithColor::warning()
          << "Cannot create an inbounds SubIndex operation\n";
  });

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
  assert(isa<LLVM::LLVMPointerType>(val.getType()) &&
         "Expecting a pointer as operand");
  assert(std::all_of(IdxList.getType().begin(), IdxList.getType().end(),
                     [](mlir::Type Ty) { return isa<IntegerType>(Ty); }) &&
         "Expecting integer indices");

  CGEIST_WARNING({
    if (IsInBounds)
      llvm::WithColor::warning() << "Cannot create an inbounds GEP operation\n";
  });

  auto PtrTy = mlirclang::getPtrTyWithNewType(val.getType(), Type);
  return {
      Builder.createOrFold<LLVM::GEPOp>(Loc, PtrTy, val, IdxList, IsInBounds),
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
  assert((isa<LLVM::LLVMPointerType, MemRefType>(ValType)) &&
         "Expecting an LLVMPointer or MemRefType input");

  return llvm::TypeSwitch<mlir::Type, ValueCategory>(ValType)
      .Case<MemRefType>([&](auto) {
        assert(IdxList.size() == 1 && "SubIndexOp expects a single index");
        return SubIndex(Builder, Loc, Type, IdxList[0], IsInBounds);
      })
      .Case<LLVM::LLVMPointerType>(
          [&](auto) { return GEP(Builder, Loc, Type, IdxList, IsInBounds); });
}

ValueCategory ValueCategory::InBoundsGEPOrSubIndex(OpBuilder &Builder,
                                                   Location Loc, Type Type,
                                                   ValueRange IdxList) const {
  return GEPOrSubIndex(Builder, Loc, Type, IdxList, /*IsInBounds*/ true);
}

template <typename OpTy>
ValueCategory FPUnaryOp(OpBuilder &Builder, Location Loc, Value Val) {
  assert(mlirclang::isFPOrFPVectorTy(Val.getType()) &&
         "Expecting FP or FP vector operand type");
  CGEIST_WARNING(warnUnconstrainedOp<arith::NegFOp>());
  return {Builder.createOrFold<OpTy>(Loc, Val), false};
}

ValueCategory ValueCategory::FNeg(OpBuilder &Builder, Location Loc) const {
  return FPUnaryOp<arith::NegFOp>(Builder, Loc, val);
}

ValueCategory ValueCategory::Shl(OpBuilder &Builder, Location Loc, Value RHS,
                                 bool HasNUW, bool HasNSW) const {
  return NUWNSWBinOp<arith::ShLIOp>(Builder, Loc, val, RHS, HasNUW, HasNSW);
}

ValueCategory ValueCategory::AShr(OpBuilder &Builder, Location Loc, Value RHS,
                                  bool IsExact) const {
  CGEIST_WARNING(warnNonExactOp<arith::ShRSIOp>(IsExact));
  return IntBinOp<arith::ShRSIOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::LShr(OpBuilder &Builder, Location Loc, Value RHS,
                                  bool IsExact) const {
  CGEIST_WARNING(warnNonExactOp<arith::ShRUIOp>(IsExact));
  return IntBinOp<arith::ShRUIOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::And(mlir::OpBuilder &Builder, mlir::Location Loc,
                                 mlir::Value RHS) const {
  return IntBinOp<arith::AndIOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::Or(mlir::OpBuilder &Builder, mlir::Location Loc,
                                mlir::Value RHS) const {
  return IntBinOp<arith::OrIOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::Xor(mlir::OpBuilder &Builder, mlir::Location Loc,
                                 mlir::Value RHS) const {
  return IntBinOp<arith::XOrIOp>(Builder, Loc, val, RHS);
}

ValueCategory ValueCategory::Insert(OpBuilder &Builder, Location Loc, Value V,
                                    llvm::ArrayRef<int64_t> Indices) const {
  assert(Indices.size() == 1 && "Only supporting 1-D vectors for now");
  assert(isa<VectorType>(val.getType()) && "Expecting vector type");
  assert(cast<VectorType>(val.getType()).getElementType() == V.getType() &&
         "Cannot insert value in vector of different type");
  assert(cast<VectorType>(val.getType()).getNumElements() > Indices[0] &&
         "Invalid index");

  return {Builder.createOrFold<vector::InsertOp>(Loc, V, val, Indices), false};
}

ValueCategory ValueCategory::InsertElement(OpBuilder &Builder, Location Loc,
                                           Value V, Value Idx) const {
  assert(isa<VectorType>(val.getType()) && "Expecting vector type");
  assert(cast<VectorType>(val.getType()).getElementType() == V.getType() &&
         "Cannot insert value in vector of different type");
  assert(isa<IntegerType>(Idx.getType()) && "Index must be an integer");

  return {Builder.createOrFold<vector::InsertElementOp>(Loc, V, val, Idx),
          false};
}

ValueCategory ValueCategory::Extract(OpBuilder &Builder, Location Loc,
                                     llvm::ArrayRef<int64_t> Indices) const {
  assert(Indices.size() == 1 && "Only supporting 1-D vectors for now");
  assert(isa<VectorType>(val.getType()) && "Expecting vector type");
  assert(cast<VectorType>(val.getType()).getNumElements() > Indices[0] &&
         "Invalid index");

  return {Builder.createOrFold<vector::ExtractOp>(Loc, val, Indices), false};
}

ValueCategory ValueCategory::ExtractElement(OpBuilder &Builder, Location Loc,
                                            Value Idx) const {
  assert(isa<IntegerType>(Idx.getType()) && "Index must be an integer");
  assert(isa<VectorType>(val.getType()) && "Expecting vector type");

  return {Builder.createOrFold<vector::ExtractElementOp>(Loc, val, Idx), false};
}

ValueCategory ValueCategory::Shuffle(OpBuilder &Builder, Location Loc, Value V2,
                                     llvm::ArrayRef<int64_t> Indices) const {
  assert(isa<VectorType>(val.getType()) && "Expecting vector type");
  assert(isa<VectorType>(V2.getType()) && "Expecting vector type");
  assert(val.getType() == V2.getType() && "Expecting vectors of equal types");

  return {Builder.createOrFold<vector::ShuffleOp>(Loc, val, V2, Indices),
          false};
}

ValueCategory ValueCategory::Reshape(OpBuilder &Builder, Location Loc,
                                     llvm::ArrayRef<int64_t> Shape) const {
  assert(isa<VectorType>(val.getType()) && "Expecting input vector");
  assert(Shape.size() == 1 && "We only support 1-D vectors for now");
  const auto CurrTy = cast<VectorType>(val.getType());
  assert(CurrTy.getNumScalableDims() == 0 && "Scalable vectors not supported");
  const auto NewTy = VectorType::get(Shape, CurrTy.getElementType());
  if (CurrTy == NewTy)
    return *this;
  return {Builder.createOrFold<vector::ReshapeOp>(
              Loc, NewTy, val,
              Builder.createOrFold<arith::ConstantIndexOp>(
                  Loc, CurrTy.getShape()[0]),
              Builder.createOrFold<arith::ConstantIndexOp>(Loc, Shape[0]),
              Builder.getArrayAttr({})),
          false};
}
