//===- GENOps.cpp - GEN dialect operations --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GEN/IR/GENOps.h"
#include "mlir/Dialect/GEN/IR/GENDialect.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/GEN/IR/GENOpsEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/GEN/IR/GENOpsAttrDefs.cpp.inc"
#define GET_OP_CLASSES
#include "mlir/Dialect/GEN/IR/GENOps.cpp.inc"
