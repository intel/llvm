//===--- SYCLAttributes.h -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SYCL_IR_SYCLATTRIBUTES_H
#define MLIR_DIALECT_SYCL_IR_SYCLATTRIBUTES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Dialect/SYCL/IR/SYCLEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h.inc"
#undef GET_ATTRDEF_CLASSES

#endif // MLIR_DIALECT_SYCL_IR_SYCLATTRIBUTES_H
