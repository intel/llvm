//===- GPUToGEN.h - GPU to GEN Passes ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides passes to convert GPU dialect to GEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOGEN_GPUTOGEN_H
#define MLIR_CONVERSION_GPUTOGEN_GPUTOGEN_H

#include <memory>

namespace mlir {

class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTGPUOPSTOGENOPS
#include "mlir/Conversion/Passes.h.inc"

void populateGPUToGENPatterns(RewritePatternSet &patterns);

} // namespace mlir
#endif // MLIR_CONVERSION_GPUTOGEN_GPUTOGEN_H
