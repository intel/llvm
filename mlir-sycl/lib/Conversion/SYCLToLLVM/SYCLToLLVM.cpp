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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sycl-to-llvm-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Get the LLVM type of "class.cl::sycl::detail::array" with number of
// dimentions \p dimNum and element type \p type.
static Type getSYCLArrayTy(MLIRContext &context, unsigned dimNum, Type type) {
  assert((dimNum == 1 || dimNum == 2 || dimNum == 3) &&
         "Expecting number of dimensions to be 1, 2, or 3.");
  auto structTy = LLVM::LLVMStructType::getIdentified(
      &context, "class.cl::sycl::detail::array." + std::to_string(dimNum));
  if (!structTy.isInitialized()) {
    auto arrayTy = LLVM::LLVMArrayType::get(type, dimNum);
    auto res = structTy.setBody({arrayTy}, /*isPacked=*/false);
    assert(succeeded(res) &&
           "Unexpected failure from LLVMStructType::setBody.");
  }
  return structTy;
}

// Get the LLVM type of a SYCL range or id type, given \p type - the type in
// SYCL, \p name - the expected LLVM type name, \p converter - LLVM type
// converter.
template <typename T>
static Type getSYCLRangeOrIDTy(T type, StringRef name,
                               LLVMTypeConverter &converter) {
  unsigned dimNum = type.getDimension();
  auto structTy = LLVM::LLVMStructType::getIdentified(
      &converter.getContext(), name.str() + "." + std::to_string(dimNum));
  if (!structTy.isInitialized()) {
    auto res = structTy.setBody(getSYCLArrayTy(converter.getContext(), dimNum,
                                               converter.getIndexType()),
                                /*isPacked=*/false);
    assert(succeeded(res) &&
           "Unexpected failure from LLVMStructType::setBody.");
  }
  return LLVM::LLVMPointerType::get(structTy);
}

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::sycl::populateSYCLToLLVMTypeConversion(
    LLVMTypeConverter &typeConverter) {
  typeConverter.addConversion([&](mlir::sycl::IDType type) {
    return getSYCLRangeOrIDTy<mlir::sycl::IDType>(type, "class.cl::sycl::id",
                                                  typeConverter);
  });
  typeConverter.addConversion([&](mlir::sycl::RangeType type) {
    return getSYCLRangeOrIDTy<mlir::sycl::RangeType>(
        type, "class.cl::sycl::range", typeConverter);
  });
}

void mlir::sycl::populateSYCLToLLVMConversionPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns) {
  populateSYCLToLLVMTypeConversion(typeConverter);      
}
