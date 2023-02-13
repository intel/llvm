//===- SYCLToLLVM.cpp - SYCL to LLVM Patterns -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert SYCL dialect to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToLLVM/SYCLToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/SYCLToLLVM/DialectBuilder.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "polygeist/Passes/Utils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sycl-to-llvm"

using namespace mlir;
using namespace mlir::sycl;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Returns true if the given type is 'memref<?xSYCLType>', and false otherwise.
template <typename SYCLType> static bool isMemRefOf(const Type &type) {
  if (!type.isa<MemRefType>())
    return false;

  MemRefType memRefTy = type.cast<MemRefType>();
  ArrayRef<int64_t> shape = memRefTy.getShape();
  if (shape.size() != 1 || shape[0] != -1)
    return false;

  return memRefTy.getElementType().isa<SYCLType>();
}

// Returns the element type of 'memref<?xSYCLType>'.
template <typename SYCLType> static SYCLType getElementType(const Type &type) {
  assert(isMemRefOf<SYCLType>(type) && "Expecting memref<?xsycl::<type>>");
  Type elemType = type.cast<MemRefType>().getElementType();
  return elemType.cast<SYCLType>();
}

// Get LLVM struct type with i8 as the body with name \p name.
static Optional<Type> getI8Struct(StringRef name,
                                  LLVMTypeConverter &converter) {
  auto convertedTy =
      LLVM::LLVMStructType::getIdentified(&converter.getContext(), name);
  if (!convertedTy.isInitialized())
    if (failed(convertedTy.setBody(IntegerType::get(&converter.getContext(), 8),
                                   /*isPacked=*/false)))
      return std::nullopt;
  return convertedTy;
}

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

/// Create a LLVM struct type with name \p name, and the converted \p body as
/// the body.
static Optional<Type> convertBodyType(StringRef name,
                                      llvm::ArrayRef<mlir::Type> body,
                                      LLVMTypeConverter &converter) {
  SmallVector<Type> convertedElemTypes;
  convertedElemTypes.reserve(body.size());
  if (failed(converter.convertTypes(body, convertedElemTypes)))
    return std::nullopt;
  auto convertedTy =
      LLVM::LLVMStructType::getIdentified(&converter.getContext(), name);
  if (!convertedTy.isInitialized()) {
    if (failed(convertedTy.setBody(convertedElemTypes, /*isPacked=*/false)))
      return std::nullopt;
  } else if (convertedElemTypes != convertedTy.getBody()) {
    // If the name is already in use, create a new type.
    convertedTy = LLVM::LLVMStructType::getNewIdentified(
        &converter.getContext(), name, convertedElemTypes, /*isPacked=*/false);
  }

  return convertedTy;
}

/// Converts SYCL accessor common type to LLVM type.
static Optional<Type> convertAccessorCommonType(sycl::AccessorCommonType type,
                                                LLVMTypeConverter &converter) {
  return getI8Struct("class.sycl::_V1::detail::accessor_common", converter);
}

/// Converts SYCL accessor implement device type to LLVM type.
static Optional<Type>
convertAccessorImplDeviceType(sycl::AccessorImplDeviceType type,
                              LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::detail::AccessorImplDevice." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL accessor type to LLVM type.
static Optional<Type> convertAccessorType(sycl::AccessorType type,
                                          LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::accessor." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL accessor subscript type to LLVM type.
static Optional<Type>
convertAccessorSubscriptType(sycl::AccessorSubscriptType type,
                             LLVMTypeConverter &converter) {
  return convertBodyType(
      "class.sycl::_V1::detail::accessor_common.AccessorSubscript." +
          std::to_string(type.getCurrentDimension()),
      type.getBody(), converter);
}

/// Converts SYCL array type to LLVM type.
static Optional<Type> convertArrayType(sycl::ArrayType type,
                                       LLVMTypeConverter &converter) {
  assert(type.getBody().size() == 1 &&
         "Expecting SYCL array body to have size 1");
  assert(type.getBody()[0].isa<MemRefType>() &&
         "Expecting SYCL array body entry to be MemRefType");
  assert(type.getBody()[0].cast<MemRefType>().getElementType() ==
             converter.getIndexType() &&
         "Expecting SYCL array body entry element type to be the index type");
  auto structTy = LLVM::LLVMStructType::getIdentified(
      &converter.getContext(),
      "class.sycl::_V1::detail::array." + std::to_string(type.getDimension()));
  if (!structTy.isInitialized()) {
    auto arrayTy =
        LLVM::LLVMArrayType::get(converter.getIndexType(), type.getDimension());
    if (failed(structTy.setBody(arrayTy, /*isPacked=*/false)))
      return std::nullopt;
  }
  return structTy;
}

/// Converts SYCL AssertHappened type to LLVM type.
static Optional<Type> convertAssertHappenedType(sycl::AssertHappenedType type,
                                                LLVMTypeConverter &converter) {
  return convertBodyType("struct.sycl::_V1::detail::AssertHappened",
                         type.getBody(), converter);
}

/// Converts SYCL atomic type to LLVM type.
static Optional<Type> convertAtomicType(sycl::AtomicType type,
                                        LLVMTypeConverter &converter) {
  // FIXME: Make sure that we have llvm.ptr as the body, not memref, through
  // the conversion done in ConvertTOLLVMABI pass
  return convertBodyType("class.sycl::_V1::atomic", type.getBody(), converter);
}

/// Converts SYCL bfloat16 type to LLVM type.
static Optional<Type> convertBFloat16Type(sycl::BFloat16Type type,
                                          LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::ext::oneapi::bfloat16",
                         type.getBody(), converter);
}

/// Converts SYCL GetOp type to LLVM type.
static Optional<Type> convertGetOpType(sycl::GetOpType type,
                                       LLVMTypeConverter &converter) {
  return getI8Struct("class.sycl::_V1::detail::GetOp", converter);
}

/// Converts SYCL GetScalarOp type to LLVM type.
static Optional<Type> convertGetScalarOpType(sycl::GetScalarOpType type,
                                             LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::detail::GetScalarOp", type.getBody(),
                         converter);
}

/// Converts SYCL group type to LLVM type.
static Optional<Type> convertGroupType(sycl::GroupType type,
                                       LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::group." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL h_item type to LLVM type.
static Optional<Type> convertHItemType(sycl::HItemType type,
                                       LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::h_item", type.getBody(), converter);
}

/// Converts SYCL id type to LLVM type.
static Optional<Type> convertIDType(sycl::IDType type,
                                    LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::id." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL item base type to LLVM type.
static Optional<Type> convertItemBaseType(sycl::ItemBaseType type,
                                          LLVMTypeConverter &converter) {
  return convertBodyType("struct.sycl::_V1::detail::ItemBase." +
                             std::to_string(type.getDimension()) +
                             (type.getWithOffset() ? ".true" : ".false"),
                         type.getBody(), converter);
}

/// Converts SYCL item type to LLVM type.
static Optional<Type> convertItemType(sycl::ItemType type,
                                      LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::item." +
                             std::to_string(type.getDimension()) +
                             (type.getWithOffset() ? ".true" : ".false"),
                         type.getBody(), converter);
}

/// Converts SYCL kernel_handler type to LLVM type.
static Optional<Type> convertKernelHandlerType(sycl::KernelHandlerType type,
                                               LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::kernel_handler", type.getBody(),
                         converter);
}

/// Converts SYCL local accessor base device type to LLVM type.
static Optional<Type>
convertLocalAccessorBaseDeviceType(sycl::LocalAccessorBaseDeviceType type,
                                   LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::detail::LocalAccessorBaseDevice." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL local accessor base type to LLVM type.
static Optional<Type>
convertLocalAccessorBaseType(sycl::LocalAccessorBaseType type,
                             LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::local_accessor_base." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL local accessor type to LLVM type.
static Optional<Type> convertLocalAccessorType(sycl::LocalAccessorType type,
                                               LLVMTypeConverter &converter) {
  // FIXME: Make sure that we have llvm.ptr as the body, not memref, through
  // the conversion done in ConvertTOLLVMABI pass
  return convertBodyType("class.sycl::_V1::local_accessor." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL maximum type to LLVM type.
static Optional<Type> convertMaximumType(sycl::MaximumType type,
                                         LLVMTypeConverter &converter) {
  return getI8Struct("struct.sycl::_V1::maximum", converter);
}

/// Converts SYCL minimum type to LLVM type.
static Optional<Type> convertMinimumType(sycl::MinimumType type,
                                         LLVMTypeConverter &converter) {
  return getI8Struct("struct.sycl::_V1::minimum", converter);
}

/// Converts SYCL multi_ptr type to LLVM type.
static Optional<Type> convertMultiPtrType(sycl::MultiPtrType type,
                                          LLVMTypeConverter &converter) {
  // FIXME: Make sure that we have llvm.ptr as the body, not memref, through
  // the conversion done in ConvertTOLLVMABI pass
  return convertBodyType("class.sycl::_V1::multi_ptr", type.getBody(),
                         converter);
}

/// Converts SYCL nd item type to LLVM type.
static Optional<Type> convertNdItemType(sycl::NdItemType type,
                                        LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::nd_item." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL nd_range type to LLVM type.
static Optional<Type> convertNdRangeType(sycl::NdRangeType type,
                                         LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::nd_range." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL owner less base type to LLVM type.
static Optional<Type> convertOwnerLessBaseType(sycl::OwnerLessBaseType type,
                                               LLVMTypeConverter &converter) {
  return getI8Struct("class.sycl::_V1::detail::OwnerLessBase", converter);
}

/// Converts SYCL range type to LLVM type.
static Optional<Type> convertRangeType(sycl::RangeType type,
                                       LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::range." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL stream type to LLVM type.
static Optional<Type> convertStreamType(sycl::StreamType type,
                                        LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::stream", type.getBody(), converter);
}

/// Converts SYCL sub_group type to LLVM type.
static Optional<Type> convertSubGroupType(sycl::SubGroupType type,
                                          LLVMTypeConverter &converter) {
  return getI8Struct("struct.sycl::_V1::ext::oneapi::sub_group", converter);
}

/// Converts SYCL vec type to LLVM type.
static Optional<Type> convertSwizzledVecType(sycl::SwizzledVecType type,
                                             LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::detail::SwizzleOp", type.getBody(),
                         converter);
}

/// Converts SYCL TupleCopyAssignableValueHolder type to LLVM type.
static Optional<Type> convertTupleCopyAssignableValueHolderType(
    sycl::TupleCopyAssignableValueHolderType type,
    LLVMTypeConverter &converter) {
  return convertBodyType(
      "struct.sycl::_V1::detail::TupleCopyAssignableValueHolder",
      type.getBody(), converter);
}

/// Converts SYCL TupleValueHolder type to LLVM type.
static Optional<Type>
convertTupleValueHolderType(sycl::TupleValueHolderType type,
                            LLVMTypeConverter &converter) {
  return convertBodyType("struct.sycl::_V1::detail::TupleValueHolder",
                         type.getBody(), converter);
}

/// Converts SYCL vec type to LLVM type.
static Optional<Type> convertVecType(sycl::VecType type,
                                     LLVMTypeConverter &converter) {
  return convertBodyType("class.sycl::_V1::vec", type.getBody(), converter);
}

//===----------------------------------------------------------------------===//
// CallPattern - Converts `sycl.call` to LLVM.
//===----------------------------------------------------------------------===//

class CallPattern final : public ConvertOpToLLVMPattern<sycl::SYCLCallOp> {
public:
  using ConvertOpToLLVMPattern<sycl::SYCLCallOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(sycl::SYCLCallOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    return rewriteCall(op, opAdaptor, rewriter);
  }

private:
  /// Rewrite sycl.call to a func call to the appropriate member function.
  LogicalResult rewriteCall(SYCLCallOp op, OpAdaptor opAdaptor,
                            ConversionPatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "CallPattern: Rewriting op: "; op.dump();
               llvm::dbgs() << "\n");
    assert(op.getNumResults() <= 1 && "Call should produce at most one result");

    ModuleOp module = op.getOperation()->getParentOfType<ModuleOp>();
    FuncBuilder builder(rewriter, op.getLoc());

    bool producesResult = op.getNumResults() == 1;
    func::CallOp funcCall = builder.genCall(
        op.getMangledFunctionName(),
        producesResult ? TypeRange(op.getResult().getType()) : TypeRange(),
        op.getOperands(), module);

    rewriter.replaceOp(op.getOperation(),
                       producesResult ? funcCall->getResult(0) : ValueRange());

    LLVM_DEBUG({
      Operation *func = funcCall->getParentOfType<LLVM::LLVMFuncOp>();
      assert(func && "Could not find parent function");
      llvm::dbgs() << "CallPattern: Function after rewrite:\n" << *func << "\n";
    });

    return success();
  }
};

//===----------------------------------------------------------------------===//
// CastPattern - Converts `sycl.cast` to LLVM.
//===----------------------------------------------------------------------===//

class CastPattern final : public ConvertOpToLLVMPattern<sycl::SYCLCastOp> {
public:
  using ConvertOpToLLVMPattern<sycl::SYCLCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLCastOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteCast(op, opAdaptor, rewriter);
  }

private:
  /// Rewrite sycl.cast() to a LLVM bitcast operation.
  LogicalResult rewriteCast(SYCLCastOp op, OpAdaptor opAdaptor,
                            ConversionPatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "CastPattern: Rewriting op: "; op.dump();
               llvm::dbgs() << "\n");

    assert(op.getSource().getType().isa<MemRefType>() &&
           "The cast source type should be a memref type");
    assert(op.getResult().getType().isa<MemRefType>() &&
           "The result source type should be a memref type");

    // Ensure the input and result types are legal.
    auto srcType = op.getSource().getType().cast<MemRefType>();
    auto resType = op.getResult().getType().cast<MemRefType>();

    if (!isConvertibleAndHasIdentityMaps(srcType) ||
        !isConvertibleAndHasIdentityMaps(resType))
      return failure();

    // Cast the source memref descriptor's allocate & aligned pointers to the
    // type of those pointers in the results memref.
    Location loc = op.getLoc();
    LLVMBuilder builder(rewriter, loc);
    MemRefDescriptor srcMemRefDesc(opAdaptor.getSource());
    Value allocatedPtr = builder.genBitcast(
        getElementPtrType(resType), srcMemRefDesc.allocatedPtr(rewriter, loc));
    Value alignedPtr = builder.genBitcast(
        getElementPtrType(resType), srcMemRefDesc.alignedPtr(rewriter, loc));

    // Create the result memref descriptor.
    SmallVector<Value, 4> sizes, strides;
    for (int pos = 0; pos < resType.getRank(); ++pos) {
      sizes.push_back(srcMemRefDesc.size(rewriter, loc, pos));
      strides.push_back(srcMemRefDesc.stride(rewriter, loc, pos));
    }

    MemRefDescriptor resMemRefDesc = createMemRefDescriptor(
        loc, resType, allocatedPtr, alignedPtr, sizes, strides, rewriter);
    resMemRefDesc.setOffset(rewriter, loc, srcMemRefDesc.offset(rewriter, loc));

    rewriter.replaceOp(op.getOperation(), {resMemRefDesc});

    LLVM_DEBUG({
      Operation *func = op->getParentOfType<LLVM::LLVMFuncOp>();
      assert(func && "Could not find parent function");
      llvm::dbgs() << "CastPattern: Function after rewrite:\n" << *func << "\n";
    });

    return success();
  }
};

class BarePtrCastPattern final : public ConvertOpToLLVMPattern<SYCLCastOp> {
public:
  using ConvertOpToLLVMPattern<SYCLCastOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLCastOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto srcType = op.getSource().getType().cast<MemRefType>();
    const auto resType = op.getResult().getType().cast<MemRefType>();
    const auto convSrcType = typeConverter->convertType(srcType);
    const auto convResType = typeConverter->convertType(resType);

    // Ensure the input and result types are legal.
    if (!canBeLoweredToBarePtr(srcType) || !canBeLoweredToBarePtr(resType) ||
        !convSrcType || !convResType)
      return failure();

    Location loc = op.getLoc();
    LLVMBuilder builder(rewriter, loc);
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, convResType,
                                                 opAdaptor.getSource());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConstructorPattern - Converts `sycl.constructor` to LLVM.
//===----------------------------------------------------------------------===//
class ConstructorPattern final
    : public ConvertOpToLLVMPattern<sycl::SYCLConstructorOp> {
public:
  using ConvertOpToLLVMPattern<sycl::SYCLConstructorOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLConstructorOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return rewriteConstructor(op, opAdaptor, rewriter);
  }

private:
  /// Rewrite sycl.constructor to a func call to the appropriate constructor
  /// function.
  LogicalResult rewriteConstructor(SYCLConstructorOp op, OpAdaptor opAdaptor,
                                   ConversionPatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "ConstructorPattern: Rewriting op: "; op.dump();
               llvm::dbgs() << "\n");

    ModuleOp module = op.getOperation()->getParentOfType<ModuleOp>();
    FuncBuilder builder(rewriter, op.getLoc());
    func::CallOp funcCall = builder.genCall(
        op.getMangledFunctionName(), TypeRange(), op.getOperands(), module);
    rewriter.eraseOp(op);
    (void)funcCall;

    LLVM_DEBUG({
      Operation *func = funcCall->getParentOfType<LLVM::LLVMFuncOp>();
      assert(func && "Could not find parent function");
      llvm::dbgs() << "ConstructorPattern: Function after rewrite:\n"
                   << *func << "\n";
    });

    return success();
  }
};

//===----------------------------------------------------------------------===//
// SYCLRangeGetPattern - Convert `sycl.range.get` to LLVM.
//===----------------------------------------------------------------------===//

Value rangeGetRef(OpBuilder &builder, Location loc, Value range,
                  LLVM::GEPArg i) {
  const auto ty = builder.getI64Type();
  const auto addressSpace =
      range.getType().cast<LLVM::LLVMPointerType>().getAddressSpace();
  return builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(ty, addressSpace), range,
      ArrayRef<LLVM::GEPArg>{0, 0, 0, i}, /*inbounds*/ true);
}

Value rangeGet(OpBuilder &builder, Location loc, Value range, LLVM::GEPArg i) {
  return builder.create<LLVM::LoadOp>(loc, rangeGetRef(builder, loc, range, i));
}

class RangeGetPattern : public ConvertOpToLLVMPattern<SYCLRangeGetOp> {
public:
  using ConvertOpToLLVMPattern<SYCLRangeGetOp>::ConvertOpToLLVMPattern;

  LogicalResult match(SYCLRangeGetOp op) const final {
    return success(op.getType().isa<IntegerType>());
  }

  void rewrite(SYCLRangeGetOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, rangeGet(rewriter, op.getLoc(), opAdaptor.getRange(),
                                    opAdaptor.getIndex()));
  }
};

class RangeGetRefPattern : public ConvertOpToLLVMPattern<SYCLRangeGetOp> {
public:
  using ConvertOpToLLVMPattern<SYCLRangeGetOp>::ConvertOpToLLVMPattern;

  LogicalResult match(SYCLRangeGetOp op) const final {
    return success(op.getType().isa<MemRefType>());
  }

  void rewrite(SYCLRangeGetOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op,
                       rangeGetRef(rewriter, op.getLoc(), opAdaptor.getRange(),
                                   opAdaptor.getIndex()));
  }
};

//===----------------------------------------------------------------------===//
// SYCLRangeSizePattern - Convert `sycl.range.size` to LLVM.
//===----------------------------------------------------------------------===//

class RangeSizePattern : public ConvertOpToLLVMPattern<SYCLRangeSizeOp> {
public:
  using ConvertOpToLLVMPattern<SYCLRangeSizeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLRangeSizeOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    Value size =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1);
    const auto range = opAdaptor.getRange();
    for (unsigned i = 0, dim = getDimensions(op.getRange().getType()); i < dim;
         ++i) {
      const auto val = rangeGet(rewriter, loc, range, static_cast<int32_t>(i));
      size = rewriter.create<LLVM::MulOp>(loc, size, val);
    }
    rewriter.replaceOp(op, size);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NDRangeGetGlobalRangePattern - Converts `sycl.nd_range.get_global_range` to
// LLVM.
//===----------------------------------------------------------------------===//

/// Extract the global range out of an ND-range
Value getGlobalRange(OpBuilder &builder, Location loc, Value nd, RangeType ty) {
  const auto addressSpace =
      nd.getType().cast<LLVM::LLVMPointerType>().getAddressSpace();
  const Value gep = builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(ty, addressSpace), nd,
      ArrayRef<LLVM::GEPArg>{0, 0}, /*inbounds*/ true);
  return builder.create<LLVM::LoadOp>(loc, gep);
}

/// Convert SYCLNdRangeGetGlobalRange to LLVM
///
/// For this pattern, we have to load the global range.
class NDRangeGetGlobalRangePattern
    : public ConvertOpToLLVMPattern<SYCLNdRangeGetGlobalRange> {
public:
  using ConvertOpToLLVMPattern<
      SYCLNdRangeGetGlobalRange>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLNdRangeGetGlobalRange op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, getGlobalRange(rewriter, op.getLoc(),
                                          opAdaptor.getND(), op.getType()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NDRangeGetLocalRangePattern - Converts `sycl.nd_range.get_local_range` to
// LLVM.
//===----------------------------------------------------------------------===//

/// Extract the local range out of an ND-range
Value getLocalRange(OpBuilder &builder, Location loc, Value nd, RangeType ty) {
  const auto addressSpace =
      nd.getType().cast<LLVM::LLVMPointerType>().getAddressSpace();
  const Value gep = builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(ty, addressSpace), nd,
      ArrayRef<LLVM::GEPArg>{0, 1}, /*inbounds*/ true);
  return builder.create<LLVM::LoadOp>(loc, gep);
}

/// Convert SYCLNdRangeGetLocalRange to LLVM
///
/// For this pattern, we have to load the local range.
class NDRangeGetLocalRangePattern
    : public ConvertOpToLLVMPattern<SYCLNdRangeGetLocalRange> {
public:
  using ConvertOpToLLVMPattern<
      SYCLNdRangeGetLocalRange>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLNdRangeGetLocalRange op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, getLocalRange(rewriter, op.getLoc(),
                                         opAdaptor.getND(), op.getType()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NDRangeGetGroupRangePattern - Converts `sycl.nd_range.get_group_range` to
// LLVM.
//===----------------------------------------------------------------------===//

Value ndRangeGetGlobalDim(OpBuilder &builder, Location loc, Value nd,
                          int32_t dim) {
  const auto ty = builder.getI64Type();
  const auto addressSpace =
      nd.getType().cast<LLVM::LLVMPointerType>().getAddressSpace();
  const Value gep = builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(ty, addressSpace), nd,
      ArrayRef<LLVM::GEPArg>{0, 0, 0, 0, dim}, /*inbounds*/ true);
  return builder.create<LLVM::LoadOp>(loc, gep);
}

Value ndRangeGetLocalDim(OpBuilder &builder, Location loc, Value nd,
                         int32_t dim) {
  const auto ty = builder.getI64Type();
  const auto addressSpace =
      nd.getType().cast<LLVM::LLVMPointerType>().getAddressSpace();
  const Value gep = builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(ty, addressSpace), nd,
      ArrayRef<LLVM::GEPArg>{0, 1, 0, 0, dim}, /*inbounds*/ true);
  return builder.create<LLVM::LoadOp>(loc, gep);
}

/// Convert SYCLNdRangeGetGroupRange to LLVM
///
/// For this pattern, we have to load both the global and local range and
/// perform an element-wise division.
class NDRangeGetGroupRangePattern
    : public ConvertOpToLLVMPattern<SYCLNdRangeGetGroupRange> {
public:
  using ConvertOpToLLVMPattern<
      SYCLNdRangeGetGroupRange>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SYCLNdRangeGetGroupRange op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    const auto nd = opAdaptor.getND();
    const auto rangeTy = op.getType();
    Value alloca = rewriter.create<LLVM::AllocaOp>(
        loc, LLVM::LLVMPointerType::get(typeConverter->convertType(rangeTy)),
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 1),
        /*alignment*/ 0);
    const auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getI64Type());
    for (int32_t i = 0, dim = rangeTy.getDimension(); i < dim; ++i) {
      const auto lhs = ndRangeGetGlobalDim(rewriter, loc, nd, i);
      const auto rhs = ndRangeGetLocalDim(rewriter, loc, nd, i);
      const Value val = rewriter.create<LLVM::UDivOp>(loc, lhs, rhs);
      const Value ptr = rewriter.create<LLVM::GEPOp>(
          loc, ptrTy, alloca, ArrayRef<LLVM::GEPArg>{0, 0, 0, i},
          /*inbounds*/ true);
      rewriter.create<LLVM::StoreOp>(loc, val, ptr);
    }
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, alloca);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AccessorSubscriptPattern - Convert `sycl.accessor.subscript` to LLVM.
//===----------------------------------------------------------------------===//

static Value getAccessorMemoryRange(OpBuilder &builder, Location loc, Value acc,
                                    int32_t index) {
  const auto addressSpace =
      acc.getType().cast<LLVM::LLVMPointerType>().getAddressSpace();
  const auto gep = static_cast<Value>(builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getI64Type(), addressSpace), acc,
      ArrayRef<LLVM::GEPArg>{0, 0, 1, 0, 0, index}, /*inbounds*/ true));
  return builder.create<LLVM::LoadOp>(loc, gep);
}

static Value getIDComponent(OpBuilder &builder, Location loc, Value id,
                            int32_t index) {
  const auto addressSpace =
      id.getType().cast<LLVM::LLVMPointerType>().getAddressSpace();
  const auto gep = static_cast<Value>(builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(builder.getI64Type(), addressSpace), id,
      ArrayRef<LLVM::GEPArg>{0, 0, 0, index}, /*inbounds*/ true));
  return builder.create<LLVM::LoadOp>(loc, gep);
}

/// Base class for other patterns converting `sycl.accessor.subscript` to LLVM.
class AccessorSubscriptPattern
    : public ConvertOpToLLVMPattern<SYCLAccessorSubscriptOp> {
public:
  using ConvertOpToLLVMPattern<SYCLAccessorSubscriptOp>::ConvertOpToLLVMPattern;

protected:
  /// Whether the input accessor has atomic access mode.
  static bool hasAtomicAccessor(SYCLAccessorSubscriptOp op) {
    return op.getAcc()
               .getType()
               .getElementType()
               .cast<AccessorType>()
               .getAccessMode() == MemoryAccessMode::Atomic;
  }

  /// Whether the input accessor is 1-dimensional.
  static bool has1DAccessor(SYCLAccessorSubscriptOp op) {
    return op.getAcc()
               .getType()
               .getElementType()
               .cast<AccessorType>()
               .getDimension() == 1;
  }

  /// Whether the input offset is an id.
  static bool hasIDOffsetType(SYCLAccessorSubscriptOp op) {
    return op.getIndex().getType().isa<MemRefType>();
  }

  /// Calculates the linear index out of an id.
  static Value getLinearIndex(OpBuilder &builder, Location loc,
                              AccessorType accTy, OpAdaptor opAdaptor) {
    const auto id = opAdaptor.getIndex();
    const auto mem = opAdaptor.getAcc();
    // int64_t Res{0};
    Value res = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(), 0);
    for (unsigned i = 0, dim = accTy.getDimension(); i < dim; ++i) {
      // Res = Res * Mem[I] + Id[I]
      const auto memI = getAccessorMemoryRange(builder, loc, mem, i);
      const auto idI = getIDComponent(builder, loc, id, i);
      res = builder.create<LLVM::AddOp>(
          loc, builder.create<LLVM::MulOp>(loc, res, memI), idI);
    }
    return res;
  }

  static Value rewriteSubscript(SYCLAccessorSubscriptOp op, Value acc,
                                Value offset, Type retTy, OpBuilder &builder) {
    return builder.create<LLVM::GEPOp>(op.getLoc(), retTy, acc,
                                       ArrayRef<LLVM::GEPArg>{0, 1, 0, offset},
                                       /*inbounds*/ true);
  }

  static Value rewriteSubscriptScalarOffset(SYCLAccessorSubscriptOp op,
                                            OpAdaptor opAdaptor, Type retTy,
                                            OpBuilder &builder) {
    return rewriteSubscript(op, opAdaptor.getAcc(), opAdaptor.getIndex(), retTy,
                            builder);
  }

  static Value rewriteSubscriptIDOffset(SYCLAccessorSubscriptOp op,
                                        OpAdaptor opAdaptor, Type retTy,
                                        OpBuilder &builder) {
    const auto loc = op.getLoc();
    return rewriteSubscript(
        op, opAdaptor.getAcc(),
        getLinearIndex(
            builder, loc,
            op.getAcc().getType().getElementType().cast<AccessorType>(),
            opAdaptor),
        retTy, builder);
  }
};

/// Conversion pattern with non-atomic access mode and id offset type.
class SubscriptIDOffset : public AccessorSubscriptPattern {
public:
  using AccessorSubscriptPattern::AccessorSubscriptPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(!AccessorSubscriptPattern::hasAtomicAccessor(op) &&
                   AccessorSubscriptPattern::hasIDOffsetType(op));
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, rewriteSubscriptIDOffset(
                               op, opAdaptor,
                               getTypeConverter()->convertType(op.getType()),
                               rewriter));
  }
};

/// Conversion pattern with non-atomic access mode, scalar offset type and
/// 1-dimensional accessor.
class SubscriptScalarOffset1D : public AccessorSubscriptPattern {
public:
  using AccessorSubscriptPattern::AccessorSubscriptPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(!AccessorSubscriptPattern::hasAtomicAccessor(op) &&
                   !AccessorSubscriptPattern::hasIDOffsetType(op) &&
                   AccessorSubscriptPattern::has1DAccessor(op));
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOp(op, rewriteSubscriptScalarOffset(
                               op, opAdaptor,
                               getTypeConverter()->convertType(op.getType()),
                               rewriter));
  }
};

/// Conversion pattern with non-atomic access mode, scalar offset type and
/// N-dimensional accessor.
///
/// Return type is implementation specific. Handling DPC++ case here: struct
/// with two fields:
/// - id<Dim - 1>: Current offset;
/// - accessor<Dim>: Original accessor.
class SubscriptScalarOffsetND : public AccessorSubscriptPattern {
public:
  using AccessorSubscriptPattern::AccessorSubscriptPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(!AccessorSubscriptPattern::hasAtomicAccessor(op) &&
                   !AccessorSubscriptPattern::hasIDOffsetType(op) &&
                   !AccessorSubscriptPattern::has1DAccessor(op));
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto loc = op.getLoc();
    Value subscript = rewriter.create<LLVM::UndefOp>(
        loc, getTypeConverter()->convertType(op.getType()));
    // Insert initial offset in the first position
    subscript = rewriter.create<LLVM::InsertValueOp>(
        loc, subscript, opAdaptor.getIndex(), ArrayRef<int64_t>{0, 0, 0, 0});
    // Zero-initialize rest of the offset id<Dim - 1>
    const Value zero =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
    for (unsigned i = 1, dim = getDimensions(op.getAcc().getType()) - 1;
         i < dim; ++i) {
      subscript = rewriter.create<LLVM::InsertValueOp>(
          loc, subscript, zero, ArrayRef<int64_t>{0, 0, 0, i});
    }
    // Insert original accessor
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        op, subscript, rewriter.create<LLVM::LoadOp>(loc, opAdaptor.getAcc()),
        1);
  }
};

/// Conversion pattern with atomic access mode and id offset type.
class AtomicSubscriptIDOffset : public AccessorSubscriptPattern {
public:
  using AccessorSubscriptPattern::AccessorSubscriptPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(AccessorSubscriptPattern::hasAtomicAccessor(op) &&
                   AccessorSubscriptPattern::hasIDOffsetType(op));
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto atomicTy = op.getType().cast<AtomicType>();
    const auto mt =
        MemRefType::get(ShapedType::kDynamic, atomicTy.getDataType(), {},
                        static_cast<unsigned>(atomicTy.getAddrSpace()));
    auto *typeConverter = getTypeConverter();
    const Value undef = rewriter.create<LLVM::UndefOp>(
        op.getLoc(), typeConverter->convertType(atomicTy));
    const Value ptr = rewriteSubscriptIDOffset(
        op, opAdaptor, typeConverter->convertType(mt), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(op, undef, ptr, 0);
  }
};

/// Conversion pattern with atomic access mode and scalar offset type.
class AtomicSubscriptScalarOffset : public AccessorSubscriptPattern {
public:
  using AccessorSubscriptPattern::AccessorSubscriptPattern;

  LogicalResult match(SYCLAccessorSubscriptOp op) const final {
    return success(AccessorSubscriptPattern::hasAtomicAccessor(op) &&
                   !AccessorSubscriptPattern::hasIDOffsetType(op));
  }

  void rewrite(SYCLAccessorSubscriptOp op, OpAdaptor opAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    const auto atomicTy = op.getType().cast<AtomicType>();
    const auto mt =
        MemRefType::get(ShapedType::kDynamic, atomicTy.getDataType(), {},
                        static_cast<unsigned>(atomicTy.getAddrSpace()));
    auto *typeConverter = getTypeConverter();
    const Value undef = rewriter.create<LLVM::UndefOp>(
        op.getLoc(), typeConverter->convertType(atomicTy));
    const Value ptr = rewriteSubscriptScalarOffset(
        op, opAdaptor, typeConverter->convertType(mt), rewriter);
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(op, undef, ptr, 0);
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::sycl::populateSYCLToLLVMTypeConversion(
    LLVMTypeConverter &typeConverter) {
  // Same order as in SYCLOps.td
  typeConverter.addConversion([&](sycl::AccessorCommonType type) {
    return convertAccessorCommonType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AccessorImplDeviceType type) {
    return convertAccessorImplDeviceType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AccessorType type) {
    return convertAccessorType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AccessorSubscriptType type) {
    return convertAccessorSubscriptType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::ArrayType type) {
    return convertArrayType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AssertHappenedType type) {
    return convertAssertHappenedType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AtomicType type) {
    return convertAtomicType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::BFloat16Type type) {
    return convertBFloat16Type(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::GetOpType type) {
    return convertGetOpType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::GetScalarOpType type) {
    return convertGetScalarOpType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::GroupType type) {
    return convertGroupType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::HItemType type) {
    return convertHItemType(type, typeConverter);
  });
  typeConverter.addConversion(
      [&](sycl::IDType type) { return convertIDType(type, typeConverter); });
  typeConverter.addConversion([&](sycl::ItemBaseType type) {
    return convertItemBaseType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::ItemType type) {
    return convertItemType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::KernelHandlerType type) {
    return convertKernelHandlerType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::LocalAccessorBaseDeviceType type) {
    return convertLocalAccessorBaseDeviceType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::LocalAccessorBaseType type) {
    return convertLocalAccessorBaseType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::LocalAccessorType type) {
    return convertLocalAccessorType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::MaximumType type) {
    return convertMaximumType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::MinimumType type) {
    return convertMinimumType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::MultiPtrType type) {
    return convertMultiPtrType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::NdItemType type) {
    return convertNdItemType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::NdRangeType type) {
    return convertNdRangeType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::OwnerLessBaseType type) {
    return convertOwnerLessBaseType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::RangeType type) {
    return convertRangeType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::StreamType type) {
    return convertStreamType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::SubGroupType type) {
    return convertSubGroupType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::SwizzledVecType type) {
    return convertSwizzledVecType(type, typeConverter);
  });
  typeConverter.addConversion(
      [&](sycl::TupleCopyAssignableValueHolderType type) {
        return convertTupleCopyAssignableValueHolderType(type, typeConverter);
      });
  typeConverter.addConversion([&](sycl::TupleValueHolderType type) {
    return convertTupleValueHolderType(type, typeConverter);
  });
  typeConverter.addConversion(
      [&](sycl::VecType type) { return convertVecType(type, typeConverter); });
}

void mlir::sycl::populateSYCLToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateSYCLToLLVMTypeConversion(typeConverter);

  patterns.add<CallPattern>(typeConverter);
  patterns.add<CastPattern>(typeConverter);
  if (typeConverter.getOptions().useBarePtrCallConv)
    patterns.add<BarePtrCastPattern>(typeConverter, /*benefit*/ 2);
  patterns.add<ConstructorPattern>(typeConverter);
  if (typeConverter.getOptions().useBarePtrCallConv)
    patterns.add<AtomicSubscriptIDOffset, AtomicSubscriptScalarOffset,
                 NDRangeGetGlobalRangePattern, NDRangeGetGroupRangePattern,
                 NDRangeGetLocalRangePattern, RangeGetPattern,
                 RangeGetRefPattern, RangeSizePattern, SubscriptIDOffset,
                 SubscriptScalarOffset1D, SubscriptScalarOffsetND>(
        typeConverter);
}
