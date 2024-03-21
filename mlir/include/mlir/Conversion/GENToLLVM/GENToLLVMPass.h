//===- GENToLLVMPass.h - GEN to LLVM dialect conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GENTOLLVM_GENTOLLVMPASS_H
#define MLIR_CONVERSION_GENTOLLVM_GENTOLLVMPASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTGENTOLLVM
#include "mlir/Conversion/Passes.h.inc"

namespace GEN {
void populateGENToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertGENToLLVM();
} // namespace GEN
} // namespace mlir

#endif // MLIR_CONVERSION_GENTOLLVM_GENTOLLVMPASS_H
