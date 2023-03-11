//===- BareMemRefToLLVM.cpp - MemRef to LLVM with bare ptr call conv ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Polygeist/Utils/Utils.h"

#include <numeric>

using namespace mlir;
using namespace mlir::polygeist;

namespace {
/// Conversion similar to the canonical one, but not inserting the obtained
/// pointer in a struct.
struct GetGlobalMemrefOpLowering
    : public ConvertOpToLLVMPattern<memref::GetGlobalOp> {
  using ConvertOpToLLVMPattern<memref::GetGlobalOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::GetGlobalOp getGlobalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto memrefTy = getGlobalOp.getType();
    if (!canBeLoweredToBarePtr(memrefTy))
      return failure();

    const auto arrayTy =
        convertGlobalMemrefTypeToLLVM(memrefTy, *typeConverter);
    if (!arrayTy)
      return failure();
    const auto addressOf =
        static_cast<Value>(rewriter.create<LLVM::AddressOfOp>(
            getGlobalOp.getLoc(),
            LLVM::LLVMPointerType::get(arrayTy, memrefTy.getMemorySpaceAsInt()),
            adaptor.getName()));

    // Get the address of the first element in the array by creating a GEP with
    // the address of the GV as the base, and (rank + 1) number of 0 indices.
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        getGlobalOp, typeConverter->convertType(memrefTy), addressOf,
        SmallVector<LLVM::GEPArg>(memrefTy.getRank() + 1, 0),
        /* inbounds */ true);

    return success();
  }

private:
  /// Returns the LLVM type of the global variable given the memref type `type`.
  static Type convertGlobalMemrefTypeToLLVM(MemRefType type,
                                            TypeConverter &typeConverter) {
    // LLVM type for a global memref will be a multi-dimension array. For
    // declarations or uninitialized global memrefs, we can potentially flatten
    // this to a 1D array. However, for memref.global's with an initial value,
    // we do not intend to flatten the ElementsAttribute when going from std ->
    // LLVM dialect, so the LLVM type needs to me a multi-dimension array.
    const auto convElemTy = typeConverter.convertType(type.getElementType());
    if (!convElemTy)
      return {};
    // Shape has the outermost dim at index 0, so need to walk it backwards
    const auto shape = type.getShape();
    return std::accumulate(
        shape.rbegin(), shape.rend(), convElemTy,
        [](auto ty, auto dim) { return LLVM::LLVMArrayType::get(ty, dim); });
  }
};

/// Simply replace by the source, as we don't care about the shape.
struct ReshapeMemrefOpLowering
    : public ConvertOpToLLVMPattern<memref::ReshapeOp> {
  using ConvertOpToLLVMPattern<memref::ReshapeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::ReshapeOp reshape, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!canBeLoweredToBarePtr(reshape.getType()) ||
        !canBeLoweredToBarePtr(
            reshape.getSource().getType().cast<MemRefType>()))
      return failure();

    rewriter.replaceOp(reshape, adaptor.getSource());
    return success();
  }
};

/// Conversion similar to the canonical one, but not inserting the obtained
/// pointer in a struct.
struct AllocaMemrefOpLowering
    : public ConvertOpToLLVMPattern<memref::AllocaOp> {
  using ConvertOpToLLVMPattern<memref::AllocaOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::AllocaOp allocaOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto memrefType = allocaOp.getType();
    if (!memrefType.hasStaticShape() || !memrefType.getLayout().isIdentity())
      return failure();

    const auto ptrType = typeConverter->convertType(allocaOp.getType());
    if (!ptrType)
      return failure();
    const auto loc = allocaOp.getLoc();
    auto nullPtr = rewriter.create<LLVM::NullOp>(loc, ptrType);
    auto gepPtr = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, nullPtr,
        createIndexConstant(rewriter, loc,
                            allocaOp.getType().getNumElements()));
    auto sizeBytes =
        rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
        allocaOp, ptrType, sizeBytes, allocaOp.getAlignment().value_or(0));
    return success();
  }
};

static Value createAligned(ConversionPatternRewriter &rewriter, Location loc,
                           Value input, Value alignment) {
  auto one = rewriter.create<LLVM::ConstantOp>(loc, alignment.getType(), 1);
  auto bump = rewriter.create<LLVM::SubOp>(loc, alignment, one);
  auto bumped = rewriter.create<LLVM::AddOp>(loc, input, bump);
  auto mod = rewriter.create<LLVM::URemOp>(loc, bumped, alignment);
  return rewriter.create<LLVM::SubOp>(loc, bumped, mod);
}

/// Conversion similar to the canonical one, but not inserting the obtained
/// pointer in a struct.
struct AllocMemrefOpLowering : public ConvertOpToLLVMPattern<memref::AllocOp> {
  using ConvertOpToLLVMPattern<memref::AllocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp allocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto memrefType = allocOp.getType();
    const auto elementPtrType = typeConverter->convertType(memrefType);
    if (!elementPtrType || !memrefType.hasStaticShape() ||
        !memrefType.getLayout().isIdentity())
      return failure();

    const auto loc = allocOp.getLoc();
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memrefType, adaptor.getOperands(), rewriter,
                             sizes, strides, sizeBytes);

    const auto alignment =
        llvm::transformOptional(allocOp.getAlignment(), [&](auto val) {
          return createIndexConstant(rewriter, loc, val);
        });
    if (alignment) {
      // Adjust the allocation size to consider alignment.
      sizeBytes = rewriter.create<LLVM::AddOp>(loc, sizeBytes, *alignment);
    }

    auto module = allocOp->getParentOfType<ModuleOp>();
    const auto allocFuncOp =
        getAllocFn(*getTypeConverter(), module, getIndexType());

    const auto results =
        rewriter.create<LLVM::CallOp>(loc, allocFuncOp, sizeBytes).getResults();
    auto alignedPtr = static_cast<Value>(
        rewriter.create<LLVM::BitcastOp>(loc, elementPtrType, results));
    if (alignment) {
      // Compute the aligned pointer.
      const auto allocatedInt = static_cast<Value>(
          rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), alignedPtr));
      const auto alignmentInt =
          createAligned(rewriter, loc, allocatedInt, *alignment);
      alignedPtr =
          rewriter.create<LLVM::IntToPtrOp>(loc, elementPtrType, alignmentInt);
    }
    rewriter.replaceOp(allocOp, {alignedPtr});
    return success();
  }
};

/// Conversion similar to the canonical one, but not extracting the allocated
/// pointer from a struct.
struct DeallocOpLowering : public ConvertOpToLLVMPattern<memref::DeallocOp> {
  using ConvertOpToLLVMPattern<memref::DeallocOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp deallocOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!canBeLoweredToBarePtr(
            deallocOp.getMemref().getType().cast<MemRefType>()))
      return failure();
    // Insert the `free` declaration if it is not already present.
    const auto freeFunc =
        getFreeFn(*getTypeConverter(), deallocOp->getParentOfType<ModuleOp>());
    const auto casted =
        rewriter
            .create<LLVM::BitcastOp>(deallocOp.getLoc(), getVoidPtrType(),
                                     adaptor.getMemref())
            .getRes();
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(deallocOp, freeFunc, casted);
    return success();
  }
};

/// Lowers to an identity operation.
struct CastMemrefOpLowering : public ConvertOpToLLVMPattern<memref::CastOp> {
  using ConvertOpToLLVMPattern<memref::CastOp>::ConvertOpToLLVMPattern;

  LogicalResult match(memref::CastOp castOp) const override {
    const auto srcType = castOp.getOperand().getType().cast<MemRefType>();
    const auto dstType = castOp.getType().cast<MemRefType>();

    // This will be replaced by an identity function, so we need input and
    // output types to match.
    return success(canBeLoweredToBarePtr(dstType) &&
                   canBeLoweredToBarePtr(srcType) &&
                   typeConverter->convertType(srcType) ==
                       typeConverter->convertType(dstType));
  }

  void rewrite(memref::CastOp castOp, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(castOp, adaptor.getSource());
  }
};

struct MemorySpaceCastMemRefOpLowering
    : public ConvertOpToLLVMPattern<memref::MemorySpaceCastOp> {
  using ConvertOpToLLVMPattern<
      memref::MemorySpaceCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::MemorySpaceCastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto newTy = getTypeConverter()->convertType(castOp.getType());
    rewriter.replaceOpWithNewOp<LLVM::AddrSpaceCastOp>(castOp, newTy,
                                                       adaptor.getSource());
    return success();
  }
};

/// Base class for lowering operations implementing memory accesses.
struct MemAccessLowering : public ConvertToLLVMPattern {
  using ConvertToLLVMPattern::ConvertToLLVMPattern;

  /// Obtains offset from a memory access indices
  Value getStridedElementBarePtr(Location loc, MemRefType type, Value base,
                                 ValueRange indices,
                                 ConversionPatternRewriter &rewriter) const {
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    LogicalResult successStrides = getStridesAndOffset(type, strides, offset);
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    (void)successStrides;

    auto index =
        offset == 0 ? Value{} : createIndexConstant(rewriter, loc, offset);

    for (const auto &iter : llvm::enumerate(llvm::zip(indices, strides))) {
      auto increment = std::get<0>(iter.value());
      const auto stride = std::get<1>(iter.value());
      if (stride != 1) { // Skip if stride is 1.
        increment = rewriter.create<LLVM::MulOp>(
            loc, increment, createIndexConstant(rewriter, loc, stride));
      }
      index = index ? rewriter.create<LLVM::AddOp>(loc, index, increment)
                    : increment;
    }
    const auto elementPtrType = getTypeConverter()->convertType(type);
    if (!elementPtrType)
      return {};
    return index
               ? rewriter.create<LLVM::GEPOp>(loc, elementPtrType, base, index)
               : base;
  }
};

struct LoadMemRefOpLowering : public MemAccessLowering {
  LoadMemRefOpLowering(LLVMTypeConverter &typeConverter,
                       PatternBenefit benefit = 1)
      : MemAccessLowering{memref::LoadOp::getOperationName(),
                          &typeConverter.getContext(), typeConverter, benefit} {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> args,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<memref::LoadOp>(op);
    if (!canBeLoweredToBarePtr(loadOp.getMemRefType()))
      return failure();
    memref::LoadOp::Adaptor adaptor{args};
    const Value DataPtr = getStridedElementBarePtr(
        loadOp.getLoc(), loadOp.getMemRefType(), adaptor.getMemref(),
        adaptor.getIndices(), rewriter);
    if (!DataPtr)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, DataPtr);
    return success();
  }
};

struct StoreMemRefOpLowering : public MemAccessLowering {
  StoreMemRefOpLowering(LLVMTypeConverter &typeConverter,
                        PatternBenefit benefit = 1)
      : MemAccessLowering{memref::StoreOp::getOperationName(),
                          &typeConverter.getContext(), typeConverter, benefit} {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> args,
                  ConversionPatternRewriter &rewriter) const override {
    auto storeOp = cast<memref::StoreOp>(op);
    if (!canBeLoweredToBarePtr(storeOp.getMemRefType()))
      return failure();
    memref::StoreOp::Adaptor adaptor{args};
    const Value DataPtr = getStridedElementBarePtr(
        storeOp.getLoc(), storeOp.getMemRefType(), adaptor.getMemref(),
        adaptor.getIndices(), rewriter);
    if (!DataPtr)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(), DataPtr);
    return success();
  }
};
} // namespace

void mlir::polygeist::populateBareMemRefToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  assert(converter.getOptions().useBarePtrCallConv &&
         "Expecting \"bare pointer\" calling convention");
  patterns.add<GetGlobalMemrefOpLowering, ReshapeMemrefOpLowering,
               AllocMemrefOpLowering, AllocaMemrefOpLowering,
               CastMemrefOpLowering, DeallocOpLowering, LoadMemRefOpLowering,
               MemorySpaceCastMemRefOpLowering, StoreMemRefOpLowering>(
      converter, 2);

  // Patterns are tried in reverse add order, so this is tried before the
  // one added by default.
  converter.addConversion([&](MemRefType type) -> Optional<Type> {
    if (!canBeLoweredToBarePtr(type))
      return std::nullopt;
    const auto elemType = converter.convertType(type.getElementType());
    if (!elemType)
      return Type{};
    return LLVM::LLVMPointerType::get(elemType, type.getMemorySpaceAsInt());
  });
}
