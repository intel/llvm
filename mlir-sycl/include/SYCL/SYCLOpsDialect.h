// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsDialect.h -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SYCL_OPS_DIALECT_H_
#define MLIR_SYCL_OPS_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the
/// sycl dialect.
#include "SYCL/SYCLOpsDialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// sycl operations.
#define GET_OP_CLASSES
#include "SYCL/SYCLOps.h.inc"

#endif // MLIR_SYCL_OPS_DIALECT_H_
