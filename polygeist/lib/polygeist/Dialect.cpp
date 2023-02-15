//===- PolygeistDialect.cpp - Polygeist dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "polygeist/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "polygeist/Ops.h"

using namespace mlir;
using namespace mlir::polygeist;

//===----------------------------------------------------------------------===//
// Polygeist Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining Polygeist
/// operations.
struct PolygeistInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Operations in the Polygeist dialect are legal to inline.
  bool isLegalToInline(Operation *Op, Region *Dest, bool WouldBeCloned,
                       BlockAndValueMapping &ValueMapping) const final {
    return true;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Polygeist dialect.
//===----------------------------------------------------------------------===//

void PolygeistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "polygeist/PolygeistOps.cpp.inc"
      >();
  addInterfaces<PolygeistInlinerInterface>();
}

#include "polygeist/PolygeistOpsDialect.cpp.inc"
