//===- Dialect.cpp - Polygeist dialect --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/IR/PolygeistDialect.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.h"
#include "mlir/Dialect/Polygeist/IR/PolygeistTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"

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
                       IRMapping &ValueMapping) const final {
    return true;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Polygeist dialect.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Polygeist/IR/PolygeistOpsTypes.cpp.inc"

void PolygeistDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Polygeist/IR/PolygeistOps.cpp.inc"
      >();

  mlir::Dialect::addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Polygeist/IR/PolygeistOpsTypes.cpp.inc"
      >();

  addInterfaces<PolygeistInlinerInterface>();
}

#include "mlir/Dialect/Polygeist/IR/PolygeistOpsDialect.cpp.inc"
