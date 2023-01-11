//===- MethodUtils.h - Utilities for SYCL method operations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares helper functions to generate SYCL method operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_METHODUTILS_H_
#define MLIR_DIALECT_SYCL_METHODUTILS_H_

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Location;
class OpBuilder;
class ValueRange;
namespace sycl {
class SYCLMethodOpInterface;
/// Generates a list of values to be used as arguments to a
/// SYCLMethodOpInterface instance from \p Original.
SmallVector<Value> adaptSYCLMethodOpArguments(OpBuilder &Builder, Location Loc,
                                              ValueRange Original);
/// Abstracts different cast operations from which \p Original may have
/// originated.
Value abstractCasts(Value Original);
} // namespace sycl
} // namespace mlir

#endif // MLIR_DIALECT_SYCL_METHODUTILS_H_
