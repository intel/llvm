//===--- PolygeistTypes.h -------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_IR_POLYGEISTTYPES_H
#define MLIR_DIALECT_POLYGEIST_IR_POLYGEISTTYPES_H

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistDialect.h"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Polygeist/IR/PolygeistOpsTypes.h.inc"

#endif // MLIR_DIALECT_POLYGEIST_IR_POLYGEISTTYPES_H
