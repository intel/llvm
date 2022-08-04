//===- PolygeistDialect.cpp - Polygeist dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "polygeist/Ops.h"

using namespace mlir;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// Polygeist dialect.
//===----------------------------------------------------------------------===//

void PolygeistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "polygeist/PolygeistOps.cpp.inc"
      >();
}

#include "polygeist/PolygeistOpsDialect.cpp.inc"
