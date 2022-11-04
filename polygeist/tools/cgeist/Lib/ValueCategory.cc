// Copyright (C) Codeplay Software Limited

//===- ValueCategory.cc ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValueCategory.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "polygeist/Ops.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/WithColor.h"

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
    const auto GetElementType = [](auto Ty) -> Type {
      return Ty.getElementType();
    };
    auto ElementType =
        llvm::TypeSwitch<Type, Type>(val.getType())
            .Case<LLVM::LLVMPointerType>(GetElementType)
            .Case<MemRefType>(GetElementType)
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
            llvm::errs() << "warning potential store type mismatch:\n";
            llvm::errs() << "val: " << val << " tosval: " << toStore << "\n";
            llvm::errs() << "mt: " << mt << "spt: " << spt << "\n";
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

template <typename OpTy> inline void warnUnconstrainedCast() {
  llvm::WithColor::warning()
      << "Creating unconstrained " << OpTy::getOperationName() << "\n";
}

ValueCategory ValueCategory::FPTrunc(OpBuilder &Builder,
                                     Type PromotionType) const {
  assert(val.getType().isa<FloatType>() &&
         "Expecting floating point source type");
  assert(PromotionType.isa<FloatType>() &&
         "Expecting floating point promotion type");
  assert(val.getType().getIntOrFloatBitWidth() >=
             PromotionType.getIntOrFloatBitWidth() &&
         "Source type must be wider than promotion type");

  warnUnconstrainedCast<arith::TruncFOp>();
  return Cast<arith::TruncFOp>(Builder, PromotionType);
}

ValueCategory ValueCategory::FPExt(OpBuilder &Builder,
                                   Type PromotionType) const {
  assert(val.getType().isa<FloatType>() &&
         "Expecting floating point source type");
  assert(PromotionType.isa<FloatType>() &&
         "Expecting floating point promotion type");
  assert(val.getType().getIntOrFloatBitWidth() <=
             PromotionType.getIntOrFloatBitWidth() &&
         "Source type must be narrower than promotion type");

  warnUnconstrainedCast<arith::ExtFOp>();
  return Cast<arith::ExtFOp>(Builder, PromotionType);
}

ValueCategory ValueCategory::SIToFP(OpBuilder &Builder,
                                    Type PromotionType) const {
  assert(val.getType().isa<IntegerType>() && "Expecting int source type");
  assert(PromotionType.isa<FloatType>() &&
         "Expecting floating point promotion type");

  warnUnconstrainedCast<arith::SIToFPOp>();
  return Cast<arith::SIToFPOp>(Builder, PromotionType);
}

ValueCategory ValueCategory::UIToFP(OpBuilder &Builder,
                                    Type PromotionType) const {
  assert(val.getType().isa<IntegerType>() && "Expecting int source type");
  assert(PromotionType.isa<FloatType>() &&
         "Expecting floating point promotion type");

  warnUnconstrainedCast<arith::UIToFPOp>();
  return Cast<arith::UIToFPOp>(Builder, PromotionType);
}

ValueCategory ValueCategory::FPToUI(OpBuilder &Builder,
                                    Type PromotionType) const {
  assert(val.getType().isa<FloatType>() &&
         "Expecting floating point source type");
  assert(PromotionType.isa<IntegerType>() &&
         "Expecting integer promotion type");

  warnUnconstrainedCast<arith::FPToUIOp>();
  return Cast<arith::FPToUIOp>(Builder, PromotionType);
}

ValueCategory ValueCategory::FPToSI(OpBuilder &Builder,
                                    Type PromotionType) const {
  assert(val.getType().isa<FloatType>() &&
         "Expecting floating point source type");
  assert(PromotionType.isa<IntegerType>() &&
         "Expecting integer promotion type");

  warnUnconstrainedCast<arith::FPToSIOp>();
  return Cast<arith::FPToSIOp>(Builder, PromotionType);
}

ValueCategory ValueCategory::IntCast(OpBuilder &Builder, Type PromotionType,
                                     bool IsSigned) const {
  assert(val.getType().isa<IntegerType>() && "Expecting integer source type");
  assert(PromotionType.isa<IntegerType>() &&
         "Expecting integer promotion type");

  if (val.getType() == PromotionType)
    return *this;

  auto SrcIntTy = val.getType().cast<IntegerType>();
  auto DstIntTy = PromotionType.cast<IntegerType>();

  const unsigned SrcBits = SrcIntTy.getWidth();
  const unsigned DstBits = DstIntTy.getWidth();

  auto Res = [&]() -> Value {
    if (SrcBits == DstBits)
      return Builder.createOrFold<arith::BitcastOp>(Builder.getUnknownLoc(),
                                                    PromotionType, val);
    if (SrcBits > DstBits)
      return Builder.createOrFold<arith::TruncIOp>(Builder.getUnknownLoc(),
                                                   PromotionType, val);
    if (IsSigned)
      return Builder.createOrFold<arith::ExtSIOp>(Builder.getUnknownLoc(),
                                                  PromotionType, val);
    return Builder.createOrFold<arith::ExtUIOp>(Builder.getUnknownLoc(),
                                                PromotionType, val);
  }();

  return {Res, /*IsReference*/ false};
}

ValueCategory ValueCategory::ICmpNE(mlir::OpBuilder &builder,
                                    ValueCategory RHS) const {
  return ICmp(builder, arith::CmpIPredicate::ne, RHS);
}

ValueCategory ValueCategory::FCmpUNE(mlir::OpBuilder &builder,
                                     ValueCategory RHS) const {
  return FCmp(builder, arith::CmpFPredicate::UNE, RHS);
}

ValueCategory ValueCategory::ICmp(OpBuilder &builder,
                                  arith::CmpIPredicate predicate,
                                  ValueCategory RHS) const {
  assert(val.getType() == RHS.val.getType() &&
         "Cannot compare values of different types");
  assert(val.getType().isa<IntegerType>() && "Expecting integer inputs");
  return {builder.createOrFold<arith::CmpIOp>(builder.getUnknownLoc(),
                                              predicate, val, RHS.val),
          false};
}

ValueCategory ValueCategory::FCmp(OpBuilder &builder,
                                  arith::CmpFPredicate predicate,
                                  ValueCategory RHS) const {
  assert(val.getType() == RHS.val.getType() &&
         "Cannot compare values of different types");
  assert(val.getType().isa<FloatType>() && "Expecting floatint point inputs");
  return {builder.createOrFold<arith::CmpFOp>(builder.getUnknownLoc(),
                                              predicate, val, RHS.val),
          false};
}
