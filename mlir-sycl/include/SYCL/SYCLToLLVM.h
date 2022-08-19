// Copyright (C) Intel

//===--- SYCLToLLVM.h -----------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SYCL_SYCLTOLLVM_H_
#define SYCL_SYCLTOLLVM_H_

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;

namespace sycl {
/// Collect the patterns to convert from the SYCL dialect to LLVM dialect. The
/// conversion patterns capture the LLVMTypeConverter by reference meaning the
/// references have to remain alive during the entire pattern lifetime.
void populateSYCLToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);
} // namespace sycl
} // namespace mlir

#endif // SYCL_SYCLTOLLVM_H_
