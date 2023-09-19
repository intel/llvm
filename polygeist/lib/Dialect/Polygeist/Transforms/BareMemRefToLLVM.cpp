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
/// Returns the LLVM type of the global variable given the memref type `type`.
///
/// Copied from `mlir/lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp`
static Type
convertGlobalMemrefTypeToLLVM(MemRefType type,
                              const LLVMTypeConverter &typeConverter) {
  // LLVM type for a global memref will be a multi-dimension array. For
  // declarations or uninitialized global memrefs, we can potentially flatten
  // this to a 1D array. However, for memref.global's with an initial value,
  // we do not intend to flatten the ElementsAttribute when going from std ->
  // LLVM dialect, so the LLVM type needs to me a multi-dimension array.
  Type elementType = typeConverter.convertType(type.getElementType());
  Type arrayTy = elementType;
  // Shape has the outermost dim at index 0, so need to walk it backwards
  for (int64_t dim : llvm::reverse(type.getShape()))
    arrayTy = LLVM::LLVMArrayType::get(arrayTy, dim);
  return arrayTy;
}

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

    // LLVM type for a global memref will be a multi-dimension array. For
    // declarations or uninitialized global memrefs, we can potentially flatten
    // this to a 1D array. However, for memref.global's with an initial value,
    // we do not intend to flatten the ElementsAttribute when going from std ->
    // LLVM dialect, so the LLVM type needs to me a multi-dimension array.
    const auto convElemType =
        typeConverter->convertType(memrefTy.getElementType());
    if (!convElemType)
      return failure();
    FailureOr<unsigned> memSpace =
        getTypeConverter()->getMemRefAddressSpace(memrefTy);
    if (failed(memSpace))
      return getGlobalOp.emitOpError(
          "memory space cannot be converted to an integer address space");
    const auto addressOf =
        static_cast<Value>(rewriter.create<LLVM::AddressOfOp>(
            getGlobalOp.getLoc(),
            LLVM::LLVMPointerType::get(memrefTy.getContext(), *memSpace),
            adaptor.getName()));

    // Get the address of the first element in the array by creating a GEP with
    // the address of the GV as the base, and (rank + 1) number of 0 indices.
    Type arrayTy = convertGlobalMemrefTypeToLLVM(memrefTy, *getTypeConverter());
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(
        getGlobalOp, typeConverter->convertType(memrefTy), arrayTy, addressOf,
        SmallVector<LLVM::GEPArg>(memrefTy.getRank() + 1, 0),
        /* inbounds */ true);

    return success();
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
        !canBeLoweredToBarePtr(cast<MemRefType>(reshape.getSource().getType())))
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
    const auto convElemType =
        typeConverter->convertType(memrefType.getElementType());
    const auto loc = allocaOp.getLoc();
    auto nullPtr = rewriter.create<LLVM::NullOp>(loc, ptrType);
    auto gepPtr = rewriter.create<LLVM::GEPOp>(
        loc, ptrType, convElemType, nullPtr,
        createIndexAttrConstant(rewriter, loc, getIndexType(),
                                memrefType.getNumElements()));
    auto sizeBytes =
        rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(), gepPtr);

    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
        allocaOp, ptrType, convElemType, sizeBytes,
        allocaOp.getAlignment().value_or(0));
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
          return createIndexAttrConstant(rewriter, loc, getIndexType(), val);
        });
    if (alignment) {
      // Adjust the allocation size to consider alignment.
      sizeBytes = rewriter.create<LLVM::AddOp>(loc, sizeBytes, *alignment);
    }

    auto module = allocOp->getParentOfType<ModuleOp>();
    const auto allocFuncOp =
        getAllocFn(*getTypeConverter(), module, getIndexType());

    auto alignedPtr = static_cast<Value>(
        rewriter.create<LLVM::CallOp>(loc, allocFuncOp, sizeBytes)
            .getResults()
            .front());
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
            cast<MemRefType>(deallocOp.getMemref().getType())))
      return failure();
    // Insert the `free` declaration if it is not already present.
    const auto freeFunc =
        getFreeFn(*getTypeConverter(), deallocOp->getParentOfType<ModuleOp>());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(deallocOp, freeFunc,
                                              adaptor.getMemref());
    return success();
  }
};

/// Lowers to an identity operation.
struct CastMemrefOpLowering : public ConvertOpToLLVMPattern<memref::CastOp> {
  using ConvertOpToLLVMPattern<memref::CastOp>::ConvertOpToLLVMPattern;

  LogicalResult match(memref::CastOp castOp) const override {
    const auto srcType = cast<MemRefType>(castOp.getOperand().getType());
    const auto dstType = cast<MemRefType>(castOp.getType());

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

    auto index = offset == 0 ? Value{}
                             : createIndexAttrConstant(rewriter, loc,
                                                       getIndexType(), offset);

    for (const auto &iter : llvm::enumerate(llvm::zip(indices, strides))) {
      auto increment = std::get<0>(iter.value());
      const auto stride = std::get<1>(iter.value());
      if (stride != 1) { // Skip if stride is 1.
        increment = rewriter.create<LLVM::MulOp>(
            loc, increment,
            createIndexAttrConstant(rewriter, loc, getIndexType(), stride));
      }
      index = index ? rewriter.create<LLVM::AddOp>(loc, index, increment)
                    : increment;
    }
    const auto elementPtrType = getTypeConverter()->convertType(type);
    if (!elementPtrType)
      return {};
    const auto convElemType =
        getTypeConverter()->convertType(type.getElementType());
    return index ? rewriter.create<LLVM::GEPOp>(loc, elementPtrType,
                                                convElemType, base, index)
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
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        op, typeConverter->convertType(loadOp.getType()), DataPtr);
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

struct ViewMemRefOpLowering : public ConvertOpToLLVMPattern<memref::ViewOp> {
  using ConvertOpToLLVMPattern<memref::ViewOp>::ConvertOpToLLVMPattern;

  // Build and return the value for the idx^th shape dimension, either by
  // returning the constant shape dimension or counting the proper dynamic size.
  Value getSize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<int64_t> shape, ValueRange dynamicSizes,
                unsigned idx) const {
    assert(idx < shape.size());
    if (!ShapedType::isDynamic(shape[idx]))
      return createIndexAttrConstant(rewriter, loc, getIndexType(), shape[idx]);
    // Count the number of dynamic dims in range [0, idx]
    unsigned nDynamic =
        llvm::count_if(shape.take_front(idx), ShapedType::isDynamic);
    return dynamicSizes[nDynamic];
  }

  // Build and return the idx^th stride, either by returning the constant stride
  // or by computing the dynamic stride from the current `runningStride` and
  // `nextSize`. The caller should keep a running stride and update it with the
  // result returned by this function.
  Value getStride(ConversionPatternRewriter &rewriter, Location loc,
                  ArrayRef<int64_t> strides, Value nextSize,
                  Value runningStride, unsigned idx) const {
    assert(idx < strides.size());
    if (!ShapedType::isDynamic(strides[idx]))
      return createIndexAttrConstant(rewriter, loc, getIndexType(),
                                     strides[idx]);
    if (nextSize)
      return runningStride
                 ? rewriter.create<LLVM::MulOp>(loc, runningStride, nextSize)
                 : nextSize;
    assert(!runningStride);
    return createIndexAttrConstant(rewriter, loc, getIndexType(), 1);
  }

  LogicalResult
  matchAndRewrite(memref::ViewOp viewOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!canBeLoweredToBarePtr(viewOp.getSource().getType()))
      return failure();

    const auto sourceTy = dyn_cast_or_null<LLVM::LLVMPointerType>(
        getTypeConverter()->convertType(viewOp.getSource().getType()));
    const Type sourceElementTy = getTypeConverter()->convertType(
        viewOp.getSource().getType().getElementType());
    if (!sourceTy || !sourceElementTy)
      return viewOp.emitWarning("Source descriptor type not converted to LLVM"),
             failure();

    const auto loc = viewOp.getLoc();
    auto alignedPtr = rewriter.create<LLVM::GEPOp>(
        loc, sourceTy, sourceElementTy, adaptor.getSource(),
        adaptor.getByteShift());
    if (canBeLoweredToBarePtr(viewOp.getType())) {
      rewriter.replaceOp(viewOp, alignedPtr);
      return success();
    }

    // The code below handles the case when source type can be lowered to bare
    // pointer, but target type cannot.
    const Type targetTy = getTypeConverter()->convertType(viewOp.getType());
    const Type targetElementTy =
        typeConverter->convertType(viewOp.getType().getElementType());
    if (!targetTy || !targetElementTy)
      return viewOp.emitWarning("Target descriptor type not converted to LLVM"),
             failure();

    int64_t offset = -1;
    SmallVector<int64_t, 4> strides;
    auto successStrides =
        getStridesAndOffset(viewOp.getType(), strides, offset);
    if (failed(successStrides))
      return viewOp.emitWarning("cannot cast to non-strided shape"), failure();
    assert(offset == 0 && "expected offset to be 0");

    // Target memref must be contiguous in memory (innermost stride is 1), or
    // empty (special case when at least one of the memref dimensions is 0).
    if (!strides.empty() && (strides.back() != 1 && strides.back() != 0))
      return viewOp.emitWarning("cannot cast to non-contiguous shape"),
             failure();

    auto targetMemRef = MemRefDescriptor::undef(rewriter, loc, targetTy);

    targetMemRef.setAllocatedPtr(rewriter, loc, viewOp.getSource());

    targetMemRef.setAlignedPtr(rewriter, loc, alignedPtr);

    // The offset in the resulting type must be 0. This is because of
    // the type change: an offset on srcType* may not be expressible as an
    // offset on dstType*.
    targetMemRef.setOffset(
        rewriter, loc,
        createIndexAttrConstant(rewriter, loc, getIndexType(), offset));

    // Early exit for 0-D corner case.
    if (viewOp.getType().getRank() == 0)
      return rewriter.replaceOp(viewOp, {targetMemRef}), success();

    Value stride = nullptr, nextSize = nullptr;
    for (int i = viewOp.getType().getRank() - 1; i >= 0; --i) {
      // Update size.
      Value size = getSize(rewriter, loc, viewOp.getType().getShape(),
                           adaptor.getSizes(), i);
      targetMemRef.setSize(rewriter, loc, i, size);
      // Update stride.
      stride = getStride(rewriter, loc, strides, nextSize, stride, i);
      targetMemRef.setStride(rewriter, loc, i, stride);
      nextSize = size;
    }

    return rewriter.replaceOp(viewOp, {targetMemRef}), success();
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
               MemorySpaceCastMemRefOpLowering, StoreMemRefOpLowering,
               ViewMemRefOpLowering>(converter, 2);

  // Patterns are tried in reverse add order, so this is tried before the
  // one added by default.
  converter.addConversion([&](MemRefType type) -> Optional<Type> {
    if (!canBeLoweredToBarePtr(type)) {
      emitError(UnknownLoc::get(type.getContext()))
          << "'" << type << "' cannot be converted to a bare pointer";
      return Type{};
    }

    FailureOr<unsigned> addrSpace = converter.getMemRefAddressSpace(type);
    if (failed(addrSpace)) {
      emitError(UnknownLoc::get(type.getContext()),
                "conversion of memref memory space ")
          << type.getMemorySpace()
          << " to integer address space failed. Consider adding memory "
             "space conversions.";
      return Type{};
    }

    return LLVM::LLVMPointerType::get(type.getContext(), *addrSpace);
  });
}
