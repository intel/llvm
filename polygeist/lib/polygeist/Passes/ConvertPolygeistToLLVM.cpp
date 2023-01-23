//===- TrivialUse.cpp - Remove trivial use instruction ---------------- -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//
#include "PassDetails.h"

#include <numeric>

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SYCLToLLVM/SYCLToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"
#include "polygeist/Ops.h"
#include "polygeist/Passes/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-polygeist-to-llvm"

using namespace mlir;
using namespace polygeist;

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
        SmallVector<LLVM::GEPArg>(memrefTy.getRank() + 1, 0));

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

static LLVM::LLVMFuncOp getFreeFn(LLVMTypeConverter &typeConverter,
                                  ModuleOp module) {
  return typeConverter.getOptions().useGenericFunctions
             ? LLVM::lookupOrCreateGenericFreeFn(module)
             : LLVM::lookupOrCreateFreeFn(module);
}

static LLVM::LLVMFuncOp getAllocFn(LLVMTypeConverter &typeConverter,
                                   ModuleOp module, Type indexType) {
  return typeConverter.getOptions().useGenericFunctions
             ? LLVM::lookupOrCreateGenericAllocFn(module, indexType)
             : LLVM::lookupOrCreateMallocFn(module, indexType);
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

struct SignatureConversionPattern : public ConversionPattern {
  using ConversionPattern::ConversionPattern;

protected:
  /// States whether a signature should be converted
  ///
  /// A signature should be converted if it contains at least a memref type and
  /// all of the memref types in it can be lowered to a bare ptr.
  static bool shouldConvertSignature(TypeRange types) {
    bool hasMT{false};
    for (auto type : types) {
      if (auto mt = type.dyn_cast<MemRefType>()) {
        if (!canBeLoweredToBarePtr(mt))
          return false;
        hasMT = true;
      }
    }
    return hasMT;
  }
};

/// See mlir/lib/Dialect/Func/Transforms/FuncConversions.cpp
///
/// Copied here to be able to pass custom benefit.
struct ReturnOpTypeConversionPattern : public SignatureConversionPattern {
  ReturnOpTypeConversionPattern(MLIRContext *context,
                                PatternBenefit benefit = 1)
      : SignatureConversionPattern(func::ReturnOp::getOperationName(), benefit,
                                   context) {}
  ReturnOpTypeConversionPattern(TypeConverter &typeConverter,
                                MLIRContext *context,
                                PatternBenefit benefit = 1)
      : SignatureConversionPattern(typeConverter,
                                   func::ReturnOp::getOperationName(), benefit,
                                   context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> args,
                  ConversionPatternRewriter &rewriter) const override {
    auto returnOp = cast<func::ReturnOp>(op);
    if (!shouldConvertSignature(returnOp.getOperandTypes()))
      return failure();
    func::ReturnOp::Adaptor adaptor{args};

    rewriter.updateRootInPlace(
        returnOp, [&] { returnOp->setOperands(adaptor.getOperands()); });
    return success();
  }
};

/// See mlir/lib/Dialect/Func/Transforms/FuncConversions.cpp
///
/// Copied here to be able to pass custom benefit.
struct CallOpSignatureConversion : public SignatureConversionPattern {
  CallOpSignatureConversion(MLIRContext *context, PatternBenefit benefit = 1)
      : SignatureConversionPattern(func::CallOp::getOperationName(), benefit,
                                   context) {}
  CallOpSignatureConversion(TypeConverter &typeConverter, MLIRContext *context,
                            PatternBenefit benefit = 1)
      : SignatureConversionPattern(
            typeConverter, func::CallOp::getOperationName(), benefit, context) {
  }

  /// Hook for derived classes to implement combined matching and rewriting.
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> args,
                  ConversionPatternRewriter &rewriter) const override {
    auto callOp = cast<func::CallOp>(op);
    if (!shouldConvertSignature(callOp.getOperandTypes()))
      return failure();
    func::CallOp::Adaptor adaptor{args};
    // Convert the original function results.
    SmallVector<Type, 1> convertedResults;
    if (failed(typeConverter->convertTypes(callOp.getResultTypes(),
                                           convertedResults)))
      return failure();

    // Substitute with the new result types from the corresponding FuncType
    // conversion.
    rewriter.replaceOpWithNewOp<func::CallOp>(
        callOp, callOp.getCallee(), convertedResults, adaptor.getOperands());
    return success();
  }
};

/// See mlir/lib/Dialect/Func/Transforms/FuncConversions.cpp
///
/// Copied here to be able to pass custom benefit.
struct AnyFunctionOpInterfaceSignatureConversion
    : public SignatureConversionPattern {
  AnyFunctionOpInterfaceSignatureConversion(MLIRContext *context,
                                            PatternBenefit benefit = 1)
      : SignatureConversionPattern(Pattern::MatchInterfaceOpTypeTag(),
                                   FunctionOpInterface::getInterfaceID(),
                                   benefit, context) {}
  AnyFunctionOpInterfaceSignatureConversion(TypeConverter &typeConverter,
                                            MLIRContext *context,
                                            PatternBenefit benefit = 1)
      : SignatureConversionPattern(
            typeConverter, Pattern::MatchInterfaceOpTypeTag(),
            FunctionOpInterface::getInterfaceID(), benefit, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FunctionOpInterface>(op);
    const auto type = funcOp.getFunctionType().cast<FunctionType>();
    const auto operandTypes = type.getInputs();
    const auto resultTypes = type.getResults();
    if (!shouldConvertSignature(operandTypes) &&
        !shouldConvertSignature(resultTypes))
      return failure();

    // Convert the original function types.
    TypeConverter::SignatureConversion result(type.getNumInputs());
    SmallVector<Type, 1> newResults;
    if (failed(typeConverter->convertSignatureArgs(operandTypes, result)) ||
        failed(typeConverter->convertTypes(resultTypes, newResults)) ||
        failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                           *typeConverter, &result)))
      return failure();

    // Update the function signature in-place.
    auto newType = FunctionType::get(rewriter.getContext(),
                                     result.getConvertedTypes(), newResults);

    rewriter.updateRootInPlace(funcOp, [&] { funcOp.setType(newType); });

    return success();
  }
};

struct BaseSubIndexOpLowering : public ConvertOpToLLVMPattern<SubIndexOp> {
  using ConvertOpToLLVMPattern<SubIndexOp>::ConvertOpToLLVMPattern;

protected:
  // Compute the indices of the GEP operation we lower the SubIndexOp to.
  // The indices are computed based on:
  //   a) the (converted) source element type, and
  //   b) the (converted) result element type that is requested
  // Examples:
  //  - src ty: ptr<struct<array<1xi64>>>, res ty: ptr<i64>
  //      -> idxs = [0, 0, SubIndexOp's index]
  //  - src ty: ptr<struct<array<1xi64>>>, res ty: ptr<array<1xi64>>
  //      -> idxs = [0, SubIndexOp's index]
  //
  // Note: when the source element type is a struct with more than one member
  // type, the result type that is requested is deemed illegal unless it is one
  // of the source member types. For example assume:
  //   - src ty: ptr<struct<array<1xi64>,i32>>
  //   - res ty: ptr<i64>
  // This is illegal because res ty can only be either ptr<i32> or
  // ptr<array<1xi64>>
  static void computeIndices(const LLVM::LLVMStructType &srcElemType,
                             const Type &resElemType,
                             SmallVectorImpl<Value> &indices, SubIndexOp op,
                             OpAdaptor transformed,
                             ConversionPatternRewriter &rewriter) {
    assert(indices.empty() && "Expecting an empty vector");

    ArrayRef<Type> memTypes = srcElemType.getBody();
    unsigned numMembers = memTypes.size();
    assert((numMembers == 1 ||
            any_of(memTypes, [=](Type t) { return resElemType == t; })) &&
           "The requested result memref element type is illegal");

    Type indexType = transformed.getIndex().getType();
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), indexType, rewriter.getIntegerAttr(indexType, 0));
    indices.push_back(zero);

    if (numMembers == 1) {
      Type currType = srcElemType.getBody()[0];
      while (currType != resElemType) {
        indices.push_back(zero);

        TypeSwitch<Type>(currType)
            .Case<LLVM::LLVMStructType>([&](LLVM::LLVMStructType t) {
              assert(t.getBody().size() == 1 && "Expecting single member type");
              currType = t.getBody()[0];
            })
            .Case<LLVM::LLVMArrayType, LLVM::LLVMPointerType>(
                [&](auto t) { currType = t.getElementType(); })
            .Default([&](Type t) {
              currType = t;
              assert(currType == resElemType &&
                     "requested result type is illegal");
            });
      }
    }

    indices.push_back(transformed.getIndex());
  }
};

/// Conversion pattern that transforms a subview op into:
///   1. An `llvm.mlir.undef` operation to create a memref descriptor
///   2. Updates to the descriptor to introduce the data ptr, offset, size
///      and stride.
/// The subview op is replaced by the descriptor.
struct SubIndexOpLowering : public BaseSubIndexOpLowering {
  using BaseSubIndexOpLowering::BaseSubIndexOpLowering;

  LogicalResult
  matchAndRewrite(SubIndexOp subViewOp, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    assert(subViewOp.getSource().getType().isa<MemRefType>() &&
           "Source operand should be a memref type");
    assert(subViewOp.getType().isa<MemRefType>() &&
           "Result should be a memref type");

    auto sourceMemRefType = subViewOp.getSource().getType().cast<MemRefType>();
    auto viewMemRefType = subViewOp.getType().cast<MemRefType>();

    auto loc = subViewOp.getLoc();
    MemRefDescriptor targetMemRef(transformed.getSource());
    Value prev = targetMemRef.alignedPtr(rewriter, loc);
    Value idxs[] = {transformed.getIndex()};

    SmallVector<Value, 4> sizes, strides;
    if (sourceMemRefType.getRank() != viewMemRefType.getRank()) {
      if (sourceMemRefType.getRank() != viewMemRefType.getRank() + 1)
        return failure();

      size_t sz = 1;
      for (int64_t i = 1; i < sourceMemRefType.getRank(); i++) {
        if (sourceMemRefType.getShape()[i] == ShapedType::kDynamic)
          return failure();
        sz *= sourceMemRefType.getShape()[i];
      }
      Value cop = rewriter.create<LLVM::ConstantOp>(
          loc, idxs[0].getType(),
          rewriter.getIntegerAttr(idxs[0].getType(), sz));
      idxs[0] = rewriter.create<LLVM::MulOp>(loc, idxs[0], cop);
      for (int64_t i = 1; i < sourceMemRefType.getRank(); i++) {
        sizes.push_back(targetMemRef.size(rewriter, loc, i));
        strides.push_back(targetMemRef.stride(rewriter, loc, i));
      }
    } else {
      for (int64_t i = 0; i < sourceMemRefType.getRank(); i++) {
        sizes.push_back(targetMemRef.size(rewriter, loc, i));
        strides.push_back(targetMemRef.stride(rewriter, loc, i));
      }
    }

    Type sourceElemType = sourceMemRefType.getElementType();
    Type convSourceElemType = getTypeConverter()->convertType(sourceElemType);
    Type viewElemType = viewMemRefType.getElementType();
    Type convViewElemType = getTypeConverter()->convertType(viewElemType);

    // Handle the general (non-SYCL) case first.
    if (convViewElemType ==
        prev.getType().cast<LLVM::LLVMPointerType>().getElementType()) {
      auto memRefDesc = createMemRefDescriptor(
          loc, viewMemRefType, targetMemRef.allocatedPtr(rewriter, loc),
          rewriter.create<LLVM::GEPOp>(loc, prev.getType(), prev, idxs), sizes,
          strides, rewriter);

      rewriter.replaceOp(subViewOp, {memRefDesc});
      return success();
    }
    assert(convSourceElemType.isa<LLVM::LLVMStructType>() &&
           "Expecting struct type");

    // SYCL case
    assert(sourceMemRefType.getRank() == viewMemRefType.getRank() &&
           "Expecting the input and output MemRef ranks to be the same");

    SmallVector<Value, 4> indices;
    computeIndices(convSourceElemType.cast<LLVM::LLVMStructType>(),
                   convViewElemType, indices, subViewOp, transformed, rewriter);
    assert(!indices.empty() && "Expecting a least one index");

    // Note: MLIRScanner::InitializeValueByInitListExpr() in clang-mlir.cc, when
    // a memref element type is a struct type, the return type of a
    // polygeist.subindex operation should be a memref of the element type of
    // the struct.
    auto elemPtrTy = LLVM::LLVMPointerType::get(
        convViewElemType, viewMemRefType.getMemorySpaceAsInt());
    auto gep = rewriter.create<LLVM::GEPOp>(loc, elemPtrTy, prev, indices);
    auto memRefDesc = createMemRefDescriptor(loc, viewMemRefType, gep, gep,
                                             sizes, strides, rewriter);
    LLVM_DEBUG(llvm::dbgs() << "SubIndexOpLowering: gep: " << *gep << "\n");

    rewriter.replaceOp(subViewOp, {memRefDesc});
    return success();
  }
};

struct SubIndexBarePtrOpLowering : public BaseSubIndexOpLowering {
  using BaseSubIndexOpLowering::BaseSubIndexOpLowering;

  LogicalResult
  matchAndRewrite(SubIndexOp subViewOp, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    assert(subViewOp.getSource().getType().isa<MemRefType>() &&
           "Source operand should be a memref type");
    assert(subViewOp.getType().isa<MemRefType>() &&
           "Result should be a memref type");

    auto sourceMemRefType = subViewOp.getSource().getType().cast<MemRefType>();
    auto viewMemRefType = subViewOp.getType().cast<MemRefType>();
    if (!canBeLoweredToBarePtr(sourceMemRefType) ||
        !canBeLoweredToBarePtr(viewMemRefType))
      return failure();

    const auto loc = subViewOp.getLoc();
    const auto target = transformed.getSource();
    auto idx = transformed.getIndex();

    if (sourceMemRefType.getRank() != viewMemRefType.getRank()) {
      if (sourceMemRefType.getRank() != viewMemRefType.getRank() + 1)
        return failure();

      size_t sz = 1;
      for (int64_t i = 1; i < sourceMemRefType.getRank(); i++) {
        if (sourceMemRefType.getShape()[i] == ShapedType::kDynamic)
          return failure();
        sz *= sourceMemRefType.getShape()[i];
      }
      Value cop = rewriter.create<LLVM::ConstantOp>(
          loc, idx.getType(), rewriter.getIntegerAttr(idx.getType(), sz));
      idx = rewriter.create<LLVM::MulOp>(loc, idx, cop);
    }

    Type sourceElemType = sourceMemRefType.getElementType();
    Type convSourceElemType = getTypeConverter()->convertType(sourceElemType);
    if (!convSourceElemType)
      return failure();
    Type viewElemType = viewMemRefType.getElementType();
    Type convViewElemType = getTypeConverter()->convertType(viewElemType);
    Type resType = getTypeConverter()->convertType(subViewOp.getType());

    // Handle the general (non-SYCL) case first.
    if (convViewElemType ==
        target.getType().cast<LLVM::LLVMPointerType>().getElementType()) {
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(subViewOp, resType, target, idx);
      return success();
    }
    assert(convSourceElemType.isa<LLVM::LLVMStructType>() &&
           "Expecting struct type");

    // SYCL case
    assert(sourceMemRefType.getRank() == viewMemRefType.getRank() &&
           "Expecting the input and output MemRef ranks to be the same");

    SmallVector<Value> indices;
    computeIndices(convSourceElemType.cast<LLVM::LLVMStructType>(),
                   convViewElemType, indices, subViewOp, transformed, rewriter);
    assert(!indices.empty() && "Expecting a least one index");

    // Note: MLIRScanner::InitializeValueByInitListExpr() in clang-mlir.cc, when
    // a memref element type is a struct type, the return type of a
    // polygeist.subindex operation should be a memref of the element type of
    // the struct.

    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(subViewOp, resType, target,
                                             indices);

    return success();
  }
};

struct Memref2PointerOpLowering
    : public ConvertOpToLLVMPattern<Memref2PointerOp> {
  using ConvertOpToLLVMPattern<Memref2PointerOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Memref2PointerOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // MemRefDescriptor sourceMemRef(operands.front());
    MemRefDescriptor targetMemRef(
        transformed.getSource()); // MemRefDescriptor::undef(rewriter, loc,
                                  // targetDescTy);

    // Offset.
    Value baseOffset = targetMemRef.offset(rewriter, loc);
    Value ptr = targetMemRef.alignedPtr(rewriter, loc);
    Value idxs[] = {baseOffset};
    ptr = rewriter.create<LLVM::GEPOp>(loc, ptr.getType(), ptr, idxs);
    assert(ptr.getType().cast<LLVM::LLVMPointerType>().getAddressSpace() ==
               op.getType().getAddressSpace() &&
           "Expecting Memref2PointerOp source and result types to have the "
           "same address space");
    ptr = rewriter.create<LLVM::BitcastOp>(loc, op.getType(), ptr);

    rewriter.replaceOp(op, {ptr});
    return success();
  }
};

struct Pointer2MemrefOpLowering
    : public ConvertOpToLLVMPattern<Pointer2MemrefOp> {
  using ConvertOpToLLVMPattern<Pointer2MemrefOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Pointer2MemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // MemRefDescriptor sourceMemRef(operands.front());
    auto convertedType = getTypeConverter()->convertType(op.getType());
    assert(convertedType && "unexpected failure in memref type conversion");
    auto descr = MemRefDescriptor::undef(rewriter, loc, convertedType);
    assert(adaptor.getSource()
                   .getType()
                   .cast<LLVM::LLVMPointerType>()
                   .getAddressSpace() ==
               op.getType().cast<MemRefType>().getMemorySpaceAsInt() &&
           "Expecting Pointer2MemrefOp source and result types to have the "
           "same address space");
    auto ptr = rewriter.create<LLVM::BitcastOp>(
        op.getLoc(), descr.getElementPtrType(), adaptor.getSource());

    // Extract all strides and offsets and verify they are static.
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto result = getStridesAndOffset(op.getType(), strides, offset);
    (void)result;
    assert(succeeded(result) && "unexpected failure in stride computation");
    assert(offset != ShapedType::kDynamic && "expected static offset");

    bool first = true;
    assert(!llvm::any_of(strides, [&](int64_t stride) {
      if (first) {
        first = false;
        return false;
      }
      return stride == ShapedType::kDynamic;
    }) && "expected static strides except first element");

    descr.setAllocatedPtr(rewriter, loc, ptr);
    descr.setAlignedPtr(rewriter, loc, ptr);
    descr.setConstantOffset(rewriter, loc, offset);

    // Fill in sizes and strides
    for (unsigned i = 0, e = op.getType().getRank(); i != e; ++i) {
      descr.setConstantSize(rewriter, loc, i, op.getType().getDimSize(i));
      descr.setConstantStride(rewriter, loc, i, strides[i]);
    }

    rewriter.replaceOp(op, {descr});
    return success();
  }
};

struct StreamToTokenOpLowering
    : public ConvertOpToLLVMPattern<StreamToTokenOp> {
  using ConvertOpToLLVMPattern<StreamToTokenOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StreamToTokenOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {

    Value v[] = {transformed.getSource()};
    rewriter.replaceOp(op, v);
    return success();
  }
};

/// Lowers to a bitcast operation
struct BareMemref2PointerOpLowering
    : public ConvertOpToLLVMPattern<Memref2PointerOp> {
  using ConvertOpToLLVMPattern<Memref2PointerOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Memref2PointerOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {
    if (!canBeLoweredToBarePtr(op.getSource().getType()))
      return failure();

    const auto target = transformed.getSource();
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getType(), target);

    return success();
  }
};

/// Lowers to a bitcast operation
struct BarePointer2MemrefOpLowering
    : public ConvertOpToLLVMPattern<Pointer2MemrefOp> {
  using ConvertOpToLLVMPattern<Pointer2MemrefOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Pointer2MemrefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!canBeLoweredToBarePtr(op.getType()))
      return failure();

    const auto convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, convertedType,
                                                 adaptor.getSource());
    return success();
  }
};

struct TypeSizeOpLowering : public ConvertOpToLLVMPattern<TypeSizeOp> {
  using ConvertOpToLLVMPattern<TypeSizeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TypeSizeOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {

    Type NT = op.getSourceAttr().getValue();
    if (auto T = getTypeConverter()->convertType(NT)) {
      NT = T;
    }
    assert(NT);

    auto type = getTypeConverter()->convertType(op.getType());

    if (NT.isa<IntegerType, FloatType>() || LLVM::isCompatibleType(NT)) {
      DataLayout DLI(op->getParentOfType<ModuleOp>());
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, type, rewriter.getIntegerAttr(type, DLI.getTypeSize(NT)));
      return success();
    }

    if (NT != op.getSourceAttr().getValue() || type != op.getType()) {
      rewriter.replaceOpWithNewOp<TypeSizeOp>(op, type, NT);
      return success();
    }
    return failure();
  }
};

struct TypeAlignOpLowering : public ConvertOpToLLVMPattern<TypeAlignOp> {
  using ConvertOpToLLVMPattern<TypeAlignOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TypeAlignOp op, OpAdaptor transformed,
                  ConversionPatternRewriter &rewriter) const override {

    Type NT = op.getSourceAttr().getValue();
    if (auto T = getTypeConverter()->convertType(NT)) {
      NT = T;
    }
    assert(NT);

    auto type = getTypeConverter()->convertType(op.getType());

    if (NT.isa<IntegerType, FloatType>() || LLVM::isCompatibleType(NT)) {
      DataLayout DLI(op->getParentOfType<ModuleOp>());
      rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
          op, type, rewriter.getIntegerAttr(type, DLI.getTypeABIAlignment(NT)));
      return success();
    }

    if (NT != op.getSourceAttr().getValue() || type != op.getType()) {
      rewriter.replaceOpWithNewOp<TypeAlignOp>(op, type, NT);
      return success();
    }
    return failure();
  }
};

void populatePolygeistToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns) {
  patterns.add<TypeSizeOpLowering, TypeAlignOpLowering, SubIndexOpLowering,
               Memref2PointerOpLowering, Pointer2MemrefOpLowering>(converter);
  if (converter.getOptions().useBarePtrCallConv) {
    // When adding these patterns (and other patterns changing the default
    // conversion of operations on MemRef values), a higher benefit is passed
    // (2), so that these patterns have a higher priority than the ones
    // performing the default conversion, which should only run if the "bare
    // pointer" ones fail.
    patterns.add<SubIndexBarePtrOpLowering, BareMemref2PointerOpLowering,
                 BarePointer2MemrefOpLowering>(converter,
                                               /*benefit*/ 2);
  }
}

namespace {
struct LLVMOpLowering : public ConversionPattern {
  explicit LLVMOpLowering(LLVMTypeConverter &converter)
      : ConversionPattern(converter, Pattern::MatchAnyOpTypeTag(), 1,
                          &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = getTypeConverter();
    SmallVector<Type> convertedResultTypes;
    if (failed(converter->convertTypes(op->getResultTypes(),
                                       convertedResultTypes))) {
      return failure();
    }
    SmallVector<Type> convertedOperandTypes;
    if (failed(converter->convertTypes(op->getOperandTypes(),
                                       convertedOperandTypes))) {
      return failure();
    }
    if (convertedResultTypes == op->getResultTypes() &&
        convertedOperandTypes == op->getOperandTypes()) {
      return failure();
    }
    if (isa<UnrealizedConversionCastOp>(op))
      return failure();

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(convertedResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      state.addRegion();

    Operation *rewritten = rewriter.create(state);
    rewriter.replaceOp(op, rewritten->getResults());

    for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i)
      rewriter.inlineRegionBefore(op->getRegion(i), rewritten->getRegion(i),
                                  rewritten->getRegion(i).begin());

    return success();
  }
};

struct URLLVMOpLowering
    : public ConvertOpToLLVMPattern<UnrealizedConversionCastOp> {
  using ConvertOpToLLVMPattern<
      UnrealizedConversionCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    if (op->getResult(0).getType() != op->getOperand(0).getType())
      return failure();

    rewriter.replaceOp(op, op->getOperands());
    return success();
  }
};

// TODO lock this wrt module
static LLVM::LLVMFuncOp addMocCUDAFunction(ModuleOp module, Type streamTy) {
  const char fname[] = "fake_cuda_dispatch";

  MLIRContext *ctx = module.getContext();
  auto loc = module.getLoc();
  auto moduleBuilder = ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

  for (auto fn : module.getBody()->getOps<LLVM::LLVMFuncOp>()) {
    if (fn.getName() == fname)
      return fn;
  }

  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  auto i8Ptr = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));

  auto resumeOp = moduleBuilder.create<LLVM::LLVMFuncOp>(
      fname, LLVM::LLVMFunctionType::get(
                 voidTy, {i8Ptr,
                          LLVM::LLVMPointerType::get(
                              LLVM::LLVMFunctionType::get(voidTy, {i8Ptr})),
                          streamTy}));
  resumeOp.setPrivate();

  return resumeOp;
}

struct AsyncOpLowering : public ConvertOpToLLVMPattern<async::ExecuteOp> {
  using ConvertOpToLLVMPattern<async::ExecuteOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(async::ExecuteOp execute, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = execute->getParentOfType<ModuleOp>();

    MLIRContext *ctx = module.getContext();
    Location loc = execute.getLoc();

    auto voidTy = LLVM::LLVMVoidType::get(ctx);
    Type voidPtr = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));

    // Make sure that all constants will be inside the outlined async function
    // to reduce the number of function arguments.
    Region &funcReg = execute.getRegion();

    // Collect all outlined function inputs.
    SetVector<mlir::Value> functionInputs;

    getUsedValuesDefinedAbove(execute.getRegion(), funcReg, functionInputs);
    SmallVector<Value> toErase;
    for (auto a : functionInputs) {
      Operation *op = a.getDefiningOp();
      if (op && op->hasTrait<OpTrait::ConstantLike>())
        toErase.push_back(a);
    }
    for (auto a : toErase) {
      functionInputs.remove(a);
    }

    // Collect types for the outlined function inputs and outputs.
    TypeConverter *converter = getTypeConverter();
    auto typesRange = llvm::map_range(functionInputs, [&](Value value) {
      return converter->convertType(value.getType());
    });
    SmallVector<Type, 4> inputTypes(typesRange.begin(), typesRange.end());

    Type ftypes[] = {voidPtr};
    auto funcType = LLVM::LLVMFunctionType::get(voidTy, ftypes);

    // TODO: Derive outlined function name from the parent FuncOp (support
    // multiple nested async.execute operations).
    auto moduleBuilder =
        ImplicitLocOpBuilder::atBlockEnd(loc, module.getBody());

    static int off = 0;
    off++;
    auto func = moduleBuilder.create<LLVM::LLVMFuncOp>(
        execute.getLoc(),
        "kernelbody." + std::to_string((long long int)&execute) + "." +
            std::to_string(off),
        funcType);

    rewriter.setInsertionPointToStart(func.addEntryBlock());
    BlockAndValueMapping valueMapping;
    for (Value capture : toErase) {
      Operation *op = capture.getDefiningOp();
      for (auto r :
           llvm::zip(op->getResults(),
                     rewriter.clone(*op, valueMapping)->getResults())) {
        valueMapping.map(rewriter.getRemappedValue(std::get<0>(r)),
                         std::get<1>(r));
      }
    }
    // Prepare for coroutine conversion by creating the body of the function.
    {
      // Map from function inputs defined above the execute op to the function
      // arguments.
      auto arg = func.getArgument(0);

      if (functionInputs.size() == 0) {
      } else if (functionInputs.size() == 1 &&
                 converter->convertType(functionInputs[0].getType())
                     .isa<LLVM::LLVMPointerType>()) {
        valueMapping.map(
            functionInputs[0],
            rewriter.create<LLVM::BitcastOp>(
                execute.getLoc(),
                converter->convertType(functionInputs[0].getType()), arg));
      } else if (functionInputs.size() == 1 &&
                 converter->convertType(functionInputs[0].getType())
                     .isa<IntegerType>()) {
        valueMapping.map(
            functionInputs[0],
            rewriter.create<LLVM::PtrToIntOp>(
                execute.getLoc(),
                converter->convertType(functionInputs[0].getType()), arg));
      } else {
        SmallVector<Type> types;
        for (auto v : functionInputs)
          types.push_back(converter->convertType(v.getType()));
        auto ST = LLVM::LLVMStructType::getLiteral(ctx, types);
        auto alloc = rewriter.create<LLVM::BitcastOp>(
            execute.getLoc(), LLVM::LLVMPointerType::get(ST), arg);
        for (auto idx : llvm::enumerate(functionInputs)) {

          mlir::Value idxs[] = {
              rewriter.create<arith::ConstantIntOp>(loc, 0, 32),
              rewriter.create<arith::ConstantIntOp>(loc, idx.index(), 32),
          };
          Value next = rewriter.create<LLVM::GEPOp>(
              loc, LLVM::LLVMPointerType::get(idx.value().getType()), alloc,
              idxs);
          valueMapping.map(idx.value(),
                           rewriter.create<LLVM::LoadOp>(loc, next));
        }
        auto freef = getFreeFn(*getTypeConverter(), module);
        Value args[] = {arg};
        rewriter.create<LLVM::CallOp>(loc, freef, args);
      }

      // Clone all operations from the execute operation body into the outlined
      // function body.
      for (Operation &op : execute.getBody()->without_terminator())
        rewriter.clone(op, valueMapping);

      rewriter.create<LLVM::ReturnOp>(execute.getLoc(), ValueRange());
    }

    // Replace the original `async.execute` with a call to outlined function.
    {
      rewriter.setInsertionPoint(execute);
      SmallVector<Value> crossing;
      for (auto tup : llvm::zip(functionInputs, inputTypes)) {
        Value val = std::get<0>(tup);
        crossing.push_back(val);
      }

      SmallVector<Value> vals;
      if (crossing.size() == 0) {
        vals.push_back(
            rewriter.create<LLVM::NullOp>(execute.getLoc(), voidPtr));
      } else if (crossing.size() == 1 &&
                 converter->convertType(crossing[0].getType())
                     .isa<LLVM::LLVMPointerType>()) {
        vals.push_back(rewriter.create<LLVM::BitcastOp>(execute.getLoc(),
                                                        voidPtr, crossing[0]));
      } else if (crossing.size() == 1 &&
                 converter->convertType(crossing[0].getType())
                     .isa<IntegerType>()) {
        vals.push_back(rewriter.create<LLVM::IntToPtrOp>(execute.getLoc(),
                                                         voidPtr, crossing[0]));
      } else {
        SmallVector<Type> types;
        for (auto v : crossing)
          types.push_back(v.getType());
        auto ST = LLVM::LLVMStructType::getLiteral(ctx, types);

        auto mallocf = getAllocFn(*getTypeConverter(), module, getIndexType());

        Value args[] = {rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getI64Type(),
            rewriter.create<polygeist::TypeSizeOp>(loc, rewriter.getIndexType(),
                                                   ST))};
        mlir::Value alloc = rewriter.create<LLVM::BitcastOp>(
            loc, LLVM::LLVMPointerType::get(ST),
            rewriter.create<mlir::LLVM::CallOp>(loc, mallocf, args)
                .getResult());
        rewriter.setInsertionPoint(execute);
        for (auto idx : llvm::enumerate(crossing)) {

          mlir::Value idxs[] = {
              rewriter.create<arith::ConstantIntOp>(loc, 0, 32),
              rewriter.create<arith::ConstantIntOp>(loc, idx.index(), 32),
          };
          Value next = rewriter.create<LLVM::GEPOp>(
              loc, LLVM::LLVMPointerType::get(idx.value().getType()), alloc,
              idxs);
          rewriter.create<LLVM::StoreOp>(loc, idx.value(), next);
        }
        vals.push_back(
            rewriter.create<LLVM::BitcastOp>(execute.getLoc(), voidPtr, alloc));
      }
      vals.push_back(
          rewriter.create<LLVM::AddressOfOp>(execute.getLoc(), func));
      for (auto dep : execute.getDependencies()) {
        auto ctx = dep.getDefiningOp<polygeist::StreamToTokenOp>();
        vals.push_back(ctx.getSource());
      }
      assert(vals.size() == 3);

      auto f = addMocCUDAFunction(execute->getParentOfType<ModuleOp>(),
                                  vals.back().getType());

      rewriter.create<LLVM::CallOp>(execute.getLoc(), f, vals);
      rewriter.eraseOp(execute);
    }

    return success();
  }
};

struct GlobalOpTypeConversion : public OpConversionPattern<LLVM::GlobalOp> {
  explicit GlobalOpTypeConversion(LLVMTypeConverter &converter)
      : OpConversionPattern<LLVM::GlobalOp>(converter,
                                            &converter.getContext()) {}

  LogicalResult
  matchAndRewrite(LLVM::GlobalOp op, LLVM::GlobalOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter *converter = getTypeConverter();
    Type globalType = adaptor.getGlobalType();
    Type convertedType = converter->convertType(globalType);
    if (!convertedType)
      return failure();
    if (convertedType == globalType)
      return failure();

    rewriter.updateRootInPlace(
        op, [&]() { op.setGlobalTypeAttr(TypeAttr::get(convertedType)); });
    return success();
  }
};

struct ReturnOpTypeConversion : public ConvertOpToLLVMPattern<LLVM::ReturnOp> {
  using ConvertOpToLLVMPattern<LLVM::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LLVM::ReturnOp op, LLVM::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement =
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getArg());
    replacement->setAttrs(adaptor.getAttributes());
    return success();
  }
};

struct GPUModuleOpToModuleOpConversion
    : public ConvertOpToLLVMPattern<gpu::GPUModuleOp> {
  using ConvertOpToLLVMPattern<gpu::GPUModuleOp>::ConvertOpToLLVMPattern;

  /// Remove operations already defined in the parent module.
  static void
  removeAlreadyDefinedFunctions(gpu::GPUModuleOp deviceModule,
                                ConversionPatternRewriter &rewriter) {
    auto Module = deviceModule->getParentOfType<ModuleOp>();
    assert(Module && "Module not found");
    const auto Operations = deviceModule.getOps();
    SmallVector<std::reference_wrapper<Operation>> AlreadyDefined;
    std::copy_if(
        Operations.begin(), Operations.end(),
        std::back_inserter(AlreadyDefined), [&](Operation &Op) -> bool {
          if (isa<gpu::ModuleEndOp>(Op)) {
            // Erase GPUEndOp.
            return true;
          }
          // Erase operations already defined in the parent module.
          auto *Other = Module.lookupSymbol(SymbolTable::getSymbolName(&Op));
          return Other && Other->getParentOp() == Module;
        });
    for (auto Op : AlreadyDefined) {
      rewriter.eraseOp(&Op.get());
    }
  }

  LogicalResult
  matchAndRewrite(gpu::GPUModuleOp deviceModule, gpu::GPUModuleOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Erase functions already present in the parent module.
    removeAlreadyDefinedFunctions(deviceModule, rewriter);
    // Copy contents to the parent module and erase the operation.
    auto module = deviceModule->getParentOfType<ModuleOp>();
    assert(module && "Module not found");
    rewriter.mergeBlocks(deviceModule.getBody(), module.getBody(), {});
    rewriter.eraseOp(deviceModule);
    return success();
  }
};

struct GPUFuncOpToFuncOpConversion
    : public ConvertOpToLLVMPattern<gpu::GPUFuncOp> {
  using ConvertOpToLLVMPattern<gpu::GPUFuncOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp gpuFuncOp, gpu::GPUFuncOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = gpuFuncOp->getParentOfType<ModuleOp>();

    rewriter.setInsertionPointToEnd(module.getBody());
    auto NewFuncOp = rewriter.create<func::FuncOp>(
        gpuFuncOp.getLoc(), gpuFuncOp.getName(), gpuFuncOp.getFunctionType());
    NewFuncOp->setAttrs(gpuFuncOp->getAttrs());
    rewriter.notifyOperationInserted(NewFuncOp);

    rewriter.inlineRegionBefore(gpuFuncOp.getBody(), NewFuncOp.getBody(),
                                NewFuncOp.getBody().end());

    rewriter.eraseOp(gpuFuncOp);

    return success();
  }
};

struct GPUModuleEndOpLowering
    : public ConvertOpToLLVMPattern<gpu::ModuleEndOp> {
  using ConvertOpToLLVMPattern<gpu::ModuleEndOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ModuleEndOp op, gpu::ModuleEndOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct GPUReturnOpLowering : public ConvertOpToLLVMPattern<gpu::ReturnOp> {
  using ConvertOpToLLVMPattern<gpu::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(gpu::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct ConvertPolygeistToLLVMPass
    : public ConvertPolygeistToLLVMBase<ConvertPolygeistToLLVMPass> {
  ConvertPolygeistToLLVMPass() = default;
  ConvertPolygeistToLLVMPass(bool useBarePtrCallConv, bool emitCWrappers,
                             unsigned indexBitwidth, bool useAlignedAlloc,
                             const llvm::DataLayout &dataLayout) {
    this->useBarePtrCallConv = useBarePtrCallConv;
    this->emitCWrappers = emitCWrappers;
    this->indexBitwidth = indexBitwidth;
    this->dataLayout = dataLayout.getStringRepresentation();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LowerToLLVMOptions options(&getContext(),
                               dataLayoutAnalysis.getAtOrAbove(m));
    options.useBarePtrCallConv = useBarePtrCallConv;
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    options.dataLayout = llvm::DataLayout(this->dataLayout);

    for (int i = 0; i < 2; i++) {

      LLVMTypeConverter converter(&getContext(), options, &dataLayoutAnalysis);
      RewritePatternSet patterns(&getContext());
      sycl::populateSYCLToLLVMConversionPatterns(converter, patterns);
      populatePolygeistToLLVMConversionPatterns(converter, patterns);
      populateSCFToControlFlowConversionPatterns(patterns);
      cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
      populateMemRefToLLVMConversionPatterns(converter, patterns);
      populateFuncToLLVMConversionPatterns(converter, patterns);
      populateMathToLLVMConversionPatterns(converter, patterns);
      populateOpenMPToLLVMConversionPatterns(converter, patterns);
      arith::populateArithToLLVMConversionPatterns(converter, patterns);
      populateVectorToLLVMConversionPatterns(converter, patterns);

      converter.addConversion([&](async::TokenType type) { return type; });
      // This overrides the default
      if (useBarePtrCallConv)
        // Patterns are tried in reverse add order, so this is tried before the
        // one added by default.
        converter.addConversion([&](MemRefType type) -> Optional<Type> {
          if (!canBeLoweredToBarePtr(type))
            return std::nullopt;
          const auto elemType = converter.convertType(type.getElementType());
          if (!elemType)
            return Type{};
          return LLVM::LLVMPointerType::get(elemType,
                                            type.getMemorySpaceAsInt());
        });

      patterns
          .add<LLVMOpLowering, GlobalOpTypeConversion, ReturnOpTypeConversion>(
              converter);

      // TODO: This is a temporary solution. In the future, we might want to
      // handle GPUDialect lowering by extending the GpuToLLVMConversionPass.
      patterns.add<GPUFuncOpToFuncOpConversion, GPUModuleOpToModuleOpConversion,
                   GPUReturnOpLowering, GPUModuleEndOpLowering>(converter);

      patterns.add<URLLVMOpLowering>(converter);

      // Run these instead of the ones provided by the dialect to avoid lowering
      // memrefs to a struct.
      if (useBarePtrCallConv) {
        patterns.add<GetGlobalMemrefOpLowering, ReshapeMemrefOpLowering,
                     AllocMemrefOpLowering, AllocaMemrefOpLowering,
                     CastMemrefOpLowering, DeallocOpLowering,
                     LoadMemRefOpLowering, StoreMemRefOpLowering>(converter, 2);

        // These should be run before lowering to the LLVM dialect to avoid
        // lowering memrefs to a struct.
        patterns.add<AnyFunctionOpInterfaceSignatureConversion,
                     CallOpSignatureConversion, ReturnOpTypeConversionPattern>(
            converter, &getContext(), /*benefit*/ 2);
      }

      // Legality callback for operations that checks whether their operand and
      // results types are converted.
      auto areAllTypesConverted = [&](Operation *op) -> Optional<bool> {
        SmallVector<Type> convertedResultTypes;
        if (failed(converter.convertTypes(op->getResultTypes(),
                                          convertedResultTypes)))
          return std::nullopt;
        SmallVector<Type> convertedOperandTypes;
        if (failed(converter.convertTypes(op->getOperandTypes(),
                                          convertedOperandTypes)))
          return std::nullopt;
        return convertedResultTypes == op->getResultTypes() &&
               convertedOperandTypes == op->getOperandTypes();
      };

      LLVMConversionTarget target(getContext());
      target.addDynamicallyLegalOp<omp::ParallelOp, omp::WsLoopOp>(
          [&](Operation *op) { return converter.isLegal(&op->getRegion(0)); });
      target.addIllegalDialect<gpu::GPUDialect>();
      target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp, scf::WhileOp,
                          scf::ExecuteRegionOp, func::FuncOp>();
      target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                        omp::YieldOp, omp::BarrierOp, omp::TaskwaitOp>();
      target.addDynamicallyLegalDialect<LLVM::LLVMDialect>(
          areAllTypesConverted);
      target.addDynamicallyLegalOp<LLVM::GlobalOp>(
          [&](LLVM::GlobalOp op) -> Optional<bool> {
            if (converter.convertType(op.getGlobalType()) == op.getGlobalType())
              return true;
            return std::nullopt;
          });
      target.addDynamicallyLegalOp<LLVM::ReturnOp>(
          [&](LLVM::ReturnOp op) -> Optional<bool> {
            // Outside global ops, defer to the normal type-based check. Note
            // that the infrastructure will not do it automatically because
            // per-op checks override dialect-level checks unconditionally.
            if (!isa<LLVM::GlobalOp>(op->getParentOp()))
              return areAllTypesConverted(op);

            SmallVector<Type> convertedOperandTypes;
            if (failed(converter.convertTypes(op->getOperandTypes(),
                                              convertedOperandTypes)))
              return std::nullopt;
            return convertedOperandTypes == op->getOperandTypes();
          });
      /*
      target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
          [&](Operation *op) { return op->getOperand(0).getType() !=
      op->getResult(0).getType(); });
          */

      if (i == 1) {
        target.addIllegalOp<UnrealizedConversionCastOp>();
        patterns.add<AsyncOpLowering>(converter);
        patterns.add<StreamToTokenOpLowering>(converter);
      }
      if (failed(applyPartialConversion(m, target, std::move(patterns))))
        signalPassFailure();

      LLVM_DEBUG(llvm::dbgs() << "ConvertPolygeistToLLVMPass: Module after:\n";
                 m->dump(); llvm::dbgs() << "\n";);
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::polygeist::createConvertPolygeistToLLVMPass(
    const LowerToLLVMOptions &options) {
  auto allocLowering = options.allocLowering;
  // There is no way to provide additional patterns for pass, so
  // AllocLowering::None will always fail.
  assert(allocLowering != LowerToLLVMOptions::AllocLowering::None &&
         "LLVMLoweringPass doesn't support AllocLowering::None");
  bool useAlignedAlloc =
      (allocLowering == LowerToLLVMOptions::AllocLowering::AlignedAlloc);
  return std::make_unique<ConvertPolygeistToLLVMPass>(
      options.useBarePtrCallConv, false, options.getIndexBitwidth(),
      useAlignedAlloc, options.dataLayout);
}

std::unique_ptr<Pass> mlir::polygeist::createConvertPolygeistToLLVMPass() {
  // TODO: meaningful arguments to this pass should be specified as
  // Option<...>'s to the pass in Passes.td. For now, we'll provide some dummy
  // default values to allow for pass creation.
  auto dl = llvm::DataLayout("");
  return std::make_unique<ConvertPolygeistToLLVMPass>(false, false, 64u, false,
                                                      dl);
}
