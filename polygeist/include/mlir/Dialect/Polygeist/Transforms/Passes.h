//===- Passes.h - Transform Pass Construction and Registration --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_POLYGEIST_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class PatternRewriter;
class DominanceInfo;
class LLVMTypeConverter;
namespace polygeist {
//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Collect a set of patterns to convert memory-related operations from the
/// MemRef dialect to the LLVM dialect forcing a "bare pointer" calling
/// convention.
void populateBareMemRefToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns);

#define GEN_PASS_DECL
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createArgumentPromotionPass();
std::unique_ptr<Pass> createMem2RegPass();
std::unique_ptr<Pass> createLICMPass();
std::unique_ptr<Pass> createLICMPass(const LICMOptions &options);
std::unique_ptr<Pass> createLoopRestructurePass();
std::unique_ptr<Pass> createInnerSerializationPass();
std::unique_ptr<Pass> replaceAffineCFGPass();
std::unique_ptr<Pass> createOpenMPOptPass();
std::unique_ptr<Pass> createCanonicalizeForPass();
std::unique_ptr<Pass> createRaiseSCFToAffinePass();
std::unique_ptr<Pass> createCPUifyPass();
std::unique_ptr<Pass> createCPUifyPass(const SCFCPUifyOptions &options);
std::unique_ptr<Pass> createBarrierRemovalContinuation();
std::unique_ptr<Pass> detectReductionPass();
std::unique_ptr<Pass> createRemoveTrivialUsePass();
std::unique_ptr<Pass> createParallelLowerPass();
std::unique_ptr<Pass> createLegalizeForSPIRVPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"

} // namespace polygeist
} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_TRANSFORMS_PASSES_H
