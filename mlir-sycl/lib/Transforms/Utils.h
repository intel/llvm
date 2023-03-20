//===- Utils.h - SYCL dialect transform utils ---------------*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_LIB_TRANSFORMS_UTILS_H
#define MLIR_SYCL_LIB_TRANSFORMS_UTILS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class OpBuilder;
namespace sycl {
class SYCLMethodOpInterface;
SmallVector<Value> adaptArgumentsForSYCLCall(OpBuilder &builder,
                                             SYCLMethodOpInterface method);
} // namespace sycl
} // namespace mlir

#endif // MLIR_SYCL_LIB_TRANSFORMS_UTILS_H
