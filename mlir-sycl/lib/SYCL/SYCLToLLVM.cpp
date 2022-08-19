// Copyright (C) Intel

//===--- SYCLToLLVM.cpp ---------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert SYCL dialects types to their
// corresponding LLVM dialect types.
//
//===----------------------------------------------------------------------===//

#include "SYCL/SYCLToLLVM.h"
#include "SYCL/SYCLOpsTypes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;

namespace {
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
} // namespace

void mlir::sycl::populateSYCLToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  converter.addConversion([&](mlir::sycl::IDType type) {
    return getSYCLRangeOrIDTy<mlir::sycl::IDType>(type, "class.cl::sycl::id",
                                                  converter);
  });
  converter.addConversion([&](mlir::sycl::RangeType type) {
    return getSYCLRangeOrIDTy<mlir::sycl::RangeType>(
        type, "class.cl::sycl::range", converter);
  });
}
