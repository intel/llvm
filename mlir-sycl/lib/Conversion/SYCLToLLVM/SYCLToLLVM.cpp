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
#include "mlir/Conversion/SYCLToLLVM/SYCLFuncRegistry.h"
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

/// Converts SYCL range type to LLVM type.
static Optional<Type> convertRangeType(sycl::RangeType type,
                                       LLVMTypeConverter &converter) {
  return convertRangeOrIDTy(type.getDimension(), "class.cl::sycl::range",
                            converter);
}

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
  return convertBodyType("class.cl::sycl::detail::AccessorImplDevice" +
                             std::to_string(type.getDimension()),
                         type.getBody(), converter);
}

/// Converts SYCL accessor type to LLVM type.
static Optional<Type> convertAccessorType(sycl::AccessorType type,
                                          LLVMTypeConverter &converter) {
  auto convertedTy = LLVM::LLVMStructType::getIdentified(
      &converter.getContext(),
      "class.cl::sycl::accessor" + std::to_string(type.getDimension()));
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

//===----------------------------------------------------------------------===//
// ConstructorPattern - Converts `sycl.constructor` to LLVM.
//===----------------------------------------------------------------------===//

class ConstructorPattern final
    : public SYCLToLLVMConversion<sycl::SYCLConstructorOp> {
public:
  using SYCLToLLVMConversion<sycl::SYCLConstructorOp>::SYCLToLLVMConversion;

  LogicalResult
  matchAndRewrite(sycl::SYCLConstructorOp op, OpAdaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef typeStr = op.Type();
    if (typeStr == "id")
      return rewriteIdConstructor(op, opAdaptor, rewriter);

    LLVM_DEBUG(llvm::dbgs() << "op: "; op.dump(); llvm::dbgs() << "\n");
    llvm_unreachable("Unhandled sycl.constructor type");

    return failure();
  }

  /// Rewrite sycl.constructor() { type = @id } to a LLVM call to the
  /// appropriate constructor function for sycl::id.
  LogicalResult
  rewriteIdConstructor(SYCLConstructorOp op, OpAdaptor opAdaptor,
                       ConversionPatternRewriter &rewriter) const {
    assert(op.Type() == "id" && "Unexpected sycl.constructor type");
    LLVM_DEBUG(llvm::dbgs() << "ConstructorPattern: Rewriting op: "; op.dump();
               llvm::dbgs() << "\n");

    ModuleOp module = op.getOperation()->getParentOfType<ModuleOp>();
    MLIRContext *context = module.getContext();

    // Lookup the ctor function to use.
    const auto &registry = SYCLFuncRegistry::create(module, rewriter);
    auto voidTy = LLVM::LLVMVoidType::get(context);
    SYCLFuncDescriptor::FuncId funcId =
        registry.getFuncId(SYCLFuncDescriptor::FuncIdKind::IdCtor, voidTy,
                           opAdaptor.Args().getTypes());

    // Generate an LLVM call to the appropriate ctor.
    SYCLFuncDescriptor::call(funcId, opAdaptor.getOperands(), registry,
                             rewriter, op.getLoc());

    LLVM_DEBUG({
      Operation *func = op->getParentOfType<LLVM::LLVMFuncOp>();
      if (!func)
        func = op->getParentOfType<func::FuncOp>();

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
    llvm_unreachable("SYCLToLLVM - sycl::GroupType not handle (yet)");
    return llvm::None;
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
    llvm_unreachable("SYCLToLLVM - sycl::NdItemType not handle (yet)");
    return llvm::None;
  });
  typeConverter.addConversion([&](sycl::RangeType type) {
    return convertRangeType(type, typeConverter);
  });
}

void mlir::sycl::populateSYCLToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateSYCLToLLVMTypeConversion(typeConverter);

  patterns.add<ConstructorPattern>(patterns.getContext(), typeConverter);
}
