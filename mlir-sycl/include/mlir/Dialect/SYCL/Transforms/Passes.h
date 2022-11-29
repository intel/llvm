//===- Passes.h - SYCL Patterns and Passes ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares patterns and passes on SYCL operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_SYCL_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace sycl {

/// Inline mode to attempt.
enum InlineMode { AlwaysInline, Simple, Aggressive };

#define GEN_PASS_DECL
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createInlinePass();
std::unique_ptr<Pass> createInlinePass(enum InlineMode InlineMode,
                                       bool RemoveDeadCallees);

std::unique_ptr<Pass> createSYCLMethodToSYCLCallPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"

} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_TRANSFORMS_PASSES_H
