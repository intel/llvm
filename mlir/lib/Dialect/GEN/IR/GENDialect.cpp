//===- GENDialect.cpp - MLIR GEN Dialect implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GEN/IR/GENDialect.h"
#include "mlir/Dialect/GEN/IR/GENOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::GEN;

#include "mlir/Dialect/GEN/IR/GENOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// GEN dialect.
//===----------------------------------------------------------------------===//

void GENDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/GEN/IR/GENOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/GEN/IR/GENOpsAttrDefs.cpp.inc"
      >();
}
