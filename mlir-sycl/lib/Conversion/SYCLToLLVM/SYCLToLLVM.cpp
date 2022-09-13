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
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
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

// Get LLVM type of "class.cl::sycl::detail::array" with \p dimNum number of
// dimensions and element type \p type.
static Optional<Type> getArrayTy(MLIRContext &context, unsigned dimNum,
                                 Type type) {
  assert((dimNum == 1 || dimNum == 2 || dimNum == 3) &&
         "Expecting number of dimensions to be 1, 2, or 3.");
  auto structTy = LLVM::LLVMStructType::getIdentified(
      &context, "class.cl::sycl::detail::array." + std::to_string(dimNum));
  if (!structTy.isInitialized()) {
    auto arrayTy = LLVM::LLVMArrayType::get(type, dimNum);
    if (failed(structTy.setBody(arrayTy, /*isPacked=*/false)))
      return llvm::None;
  }
  return structTy;
}

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

/// Create a LLVM struct type with name \p name, and the converted \p body as
/// the body.
static Optional<Type> convertBodyType(StringRef name,
                                      llvm::ArrayRef<mlir::Type> body,
                                      LLVMTypeConverter &converter) {
  auto convertedTy =
      LLVM::LLVMStructType::getIdentified(&converter.getContext(), name);
  if (!convertedTy.isInitialized()) {
    SmallVector<Type> convertedElemTypes;
    convertedElemTypes.reserve(body.size());
    if (failed(converter.convertTypes(body, convertedElemTypes)))
      return llvm::None;
    if (failed(convertedTy.setBody(convertedElemTypes, /*isPacked=*/false)))
      return llvm::None;
  }

  return convertedTy;
}

/// Converts SYCL accessor implement device type to LLVM type.
static Optional<Type>
convertAccessorImplDeviceType(sycl::AccessorImplDeviceType type,
                              LLVMTypeConverter &converter) {
  return convertBodyType("class.cl::sycl::detail::AccessorImplDevice." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL accessor type to LLVM type.
static Optional<Type> convertAccessorType(sycl::AccessorType type,
                                          LLVMTypeConverter &converter) {
  auto convertedTy = LLVM::LLVMStructType::getIdentified(
      &converter.getContext(),
      "class.cl::sycl::accessor." + std::to_string(type.getDimension()));
  if (!convertedTy.isInitialized()) {
    SmallVector<Type> convertedElemTypes;
    convertedElemTypes.reserve(type.getBody().size());
    if (failed(converter.convertTypes(type.getBody(), convertedElemTypes)))
      return llvm::None;

    auto ptrTy = LLVM::LLVMPointerType::get(type.getType(), /*addressSpace=*/1);
    auto structTy =
        LLVM::LLVMStructType::getLiteral(&converter.getContext(), ptrTy);
    convertedElemTypes.push_back(structTy);

    if (failed(convertedTy.setBody(convertedElemTypes, /*isPacked=*/false)))
      return llvm::None;
  }

  return convertedTy;
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
  return getArrayTy(converter.getContext(), type.getDimension(),
                    converter.getIndexType());
}

/// Converts SYCL group type to LLVM type.
static Optional<Type> convertGroupType(sycl::GroupType type,
                                       LLVMTypeConverter &converter) {
  return convertBodyType("class.cl::sycl::group." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL range or id type to LLVM type, given \p dimNum - number of
/// dimensions, \p name - the expected LLVM type name, \p converter - LLVM type
/// converter.
static Optional<Type> convertRangeOrIDTy(unsigned dimNum, StringRef name,
                                         LLVMTypeConverter &converter) {
  auto convertedTy = LLVM::LLVMStructType::getIdentified(
      &converter.getContext(), name.str() + "." + std::to_string(dimNum));
  if (!convertedTy.isInitialized()) {
    auto arrayTy =
        getArrayTy(converter.getContext(), dimNum, converter.getIndexType());
    if (!arrayTy.hasValue())
      return llvm::None;
    if (failed(convertedTy.setBody(arrayTy.getValue(), /*isPacked=*/false)))
      return llvm::None;
  }
  return convertedTy;
}

/// Converts SYCL id type to LLVM type.
static Optional<Type> convertIDType(sycl::IDType type,
                                    LLVMTypeConverter &converter) {
  return convertRangeOrIDTy(type.getDimension(), "class.cl::sycl::id",
                            converter);
}

/// Converts SYCL item base type to LLVM type.
static Optional<Type> convertItemBaseType(sycl::ItemBaseType type,
                                          LLVMTypeConverter &converter) {
  return convertBodyType("class.cl::sycl::detail::ItemBase." +
                             std::to_string(type.getDimension()) +
                             (type.getWithOffset() ? ".true" : ".false"),
                         type.getBody(), converter);
}

/// Converts SYCL item type to LLVM type.
static Optional<Type> convertItemType(sycl::ItemType type,
                                      LLVMTypeConverter &converter) {
  return convertBodyType("class.cl::sycl::item." +
                             std::to_string(type.getDimension()) +
                             (type.getWithOffset() ? ".true" : ".false"),
                         type.getBody(), converter);
}

/// Converts SYCL nd item type to LLVM type.
static Optional<Type> convertNdItemType(sycl::NdItemType type,
                                        LLVMTypeConverter &converter) {
  return convertBodyType("class.cl::sycl::nd_item." +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL range type to LLVM type.
static Optional<Type> convertRangeType(sycl::RangeType type,
                                       LLVMTypeConverter &converter) {
  return convertRangeOrIDTy(type.getDimension(), "class.cl::sycl::range",
                            converter);
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
    assert(op.Type().has_value() &&
           "Expecting op.Type() to have a valid value");
    return rewriteCall(op, opAdaptor, rewriter);
  }

private:
  /// Rewrite sycl.call() {Function = *, Type = *} to a LLVM call to the
  /// appropriate member function.
  LogicalResult rewriteCall(SYCLCallOp op, OpAdaptor opAdaptor,
                            ConversionPatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "CallPattern: Rewriting op: "; op.dump();
               llvm::dbgs() << "\n");

    ModuleOp module = op.getOperation()->getParentOfType<ModuleOp>();
    Type retType = op.getODSResults(0).empty()
                       ? LLVM::LLVMVoidType::get(module.getContext())
                       : typeConverter->convertType(op.result().getType());

    LLVMBuilder builder(rewriter, op.getLoc());
    SmallVector<Type> operandTypes(opAdaptor.Args().getTypes());
    FlatSymbolRefAttr funcRef = builder.getOrInsertFuncDecl(
        opAdaptor.MangledName(), retType, operandTypes, module);
    auto newOp = rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op.getOperation(), op.getNumResults() == 0 ? TypeRange() : retType,
        funcRef, opAdaptor.getOperands());
    (void) newOp;

    LLVM_DEBUG({
      Operation *func = newOp->getParentOfType<LLVM::LLVMFuncOp>();
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

    assert(op.source().getType().isa<MemRefType>() &&
           "The cast source type should be a memref type");
    assert(op.result().getType().isa<MemRefType>() &&
           "The result source type should be a memref type");

    // Ensure the input and result types are legal.
    auto srcType = op.source().getType().cast<MemRefType>();
    auto resType = op.result().getType().cast<MemRefType>();

    if (!isConvertibleAndHasIdentityMaps(srcType) ||
        !isConvertibleAndHasIdentityMaps(resType))
      return failure();

    // Cast the source memref descriptor's allocate & aligned pointers to the
    // type of those pointers in the results memref.
    Location loc = op.getLoc();
    LLVMBuilder builder(rewriter, loc);
    MemRefDescriptor srcMemRefDesc(opAdaptor.source());
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
  /// Rewrite sycl.constructor() { type = * } to a LLVM call to the appropriate
  /// constructor function.
  LogicalResult rewriteConstructor(SYCLConstructorOp op, OpAdaptor opAdaptor,
                                   ConversionPatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "ConstructorPattern: Rewriting op: "; op.dump();
               llvm::dbgs() << "\n");

    ModuleOp module = op.getOperation()->getParentOfType<ModuleOp>();
    LLVMBuilder builder(rewriter, op.getLoc());
    SmallVector<Type> operandTypes(opAdaptor.Args().getTypes());
    FlatSymbolRefAttr funcRef = builder.getOrInsertFuncDecl(
        opAdaptor.MangledName(), LLVM::LLVMVoidType::get(module.getContext()),
        operandTypes, module);
    builder.genCall(funcRef, {}, opAdaptor.getOperands());

    LLVM_DEBUG({
      Operation *func = op->getParentOfType<LLVM::LLVMFuncOp>();
      assert(func && "Could not find parent function");
      llvm::dbgs() << "ConstructorPattern: Function after rewrite:\n"
                   << *func << "\n";
    });

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::sycl::populateSYCLToLLVMTypeConversion(
    LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](sycl::AccessorImplDeviceType type) {
    return convertAccessorImplDeviceType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::AccessorType type) {
    return convertAccessorType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::ArrayType type) {
    return convertArrayType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::GroupType type) {
    return convertGroupType(type, typeConverter);
  });
  typeConverter.addConversion(
      [&](sycl::IDType type) { return convertIDType(type, typeConverter); });
  typeConverter.addConversion([&](sycl::ItemBaseType type) {
    return convertItemBaseType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::ItemType type) {
    return convertItemType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::NdItemType type) {
    return convertNdItemType(type, typeConverter);
  });
  typeConverter.addConversion([&](sycl::RangeType type) {
    return convertRangeType(type, typeConverter);
  });
}

void mlir::sycl::populateSYCLToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateSYCLToLLVMTypeConversion(typeConverter);

  patterns.add<CallPattern>(typeConverter);
  patterns.add<CastPattern>(typeConverter);
  patterns.add<ConstructorPattern>(typeConverter);
}
