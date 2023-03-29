//===- AliasAnalysis.cpp - Alias Analysis for the SYCL MLIR dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "llvm/IR/Attributes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "alias-analysis"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Return true if the value \p val is a function argument, and false otherwise.
static bool isFuncArg(Value val) {
  auto blockArg = dyn_cast<BlockArgument>(val);
  if (!blockArg)
    return false;

  return isa_and_nonnull<FunctionOpInterface>(
      blockArg.getOwner()->getParentOp());
}

// Return true if the value \p val is a function argument that has the
// 'llvm.noalias' attribute, and false otherwise.
static bool isRestrict(Value val) {
  if (!isFuncArg(val))
    return false;

  auto blockArg = cast<BlockArgument>(val);
  auto func = cast<FunctionOpInterface>(blockArg.getOwner()->getParentOp());
  auto noAliasAttr = StringAttr::get(
      val.getContext(),
      LLVM::LLVMDialect::getDialectNamespace() + "." +
          llvm::Attribute::getNameFromAttrKind(llvm::Attribute::NoAlias));
  return !!func.getArgAttr(blockArg.getArgNumber(), noAliasAttr);
}

// Return true is the given type \p ty is a MemRef type with a SYCL element
// type.
static bool isMemRefOfSYCLType(Type ty) {
  if (auto mt = dyn_cast<MemRefType>(ty))
    return sycl::isSYCLType(mt.getElementType());
  return false;
}

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

AliasResult sycl::AliasAnalysis::aliasImpl(Value lhs, Value rhs) {
  if (lhs == rhs)
    return AliasResult::MustAlias;

  if (AliasResult aliasResult = handleRestrictAlias(lhs, rhs);
      !aliasResult.isMay())
    return aliasResult;

  return handleSYCLAlias(lhs, rhs);
}

AliasResult sycl::AliasAnalysis::handleRestrictAlias(Value lhs, Value rhs) {
  // Function arguments do not alias if any of them are 'restrict' qualified.
  if (isFuncArg(lhs) && isFuncArg(rhs))
    if (isRestrict(lhs) || isRestrict(rhs))
      return AliasResult::NoAlias;

  return AliasResult::MayAlias;
}

AliasResult sycl::AliasAnalysis::handleSYCLAlias(Value lhs, Value rhs) {
  Operation *lhsOp = lhs.getDefiningOp(), *rhsOp = rhs.getDefiningOp();
  if (!isSYCLOperation(lhsOp) && !isSYCLOperation(rhsOp))
    return AliasResult::MayAlias;

  // Handle accessor.subscript operations.
  if (AliasResult aliasResult = handleAccessorSubscriptAlias(lhs, rhs);
      !aliasResult.isMay())
    return aliasResult;

  // TODO: handle the other SYCL operations.

  return AliasResult::MayAlias;
}

AliasResult sycl::AliasAnalysis::handleAccessorSubscriptAlias(Value lhs,
                                                              Value rhs) {
  Operation *lhsOp = lhs.getDefiningOp(), *rhsOp = rhs.getDefiningOp();
  auto lhsSubOp = dyn_cast_or_null<sycl::SYCLAccessorSubscriptOp>(lhsOp);
  auto rhsSubOp = dyn_cast_or_null<sycl::SYCLAccessorSubscriptOp>(rhsOp);

  auto typesDoNotAlias = [](Type lhsTy, Type rhsTy) {
    return (lhsTy != rhsTy && isMemRefOfSYCLType(rhsTy));
  };

  // The value produced by an accessor subscript operation does not alias
  // with values that have a different SYCL type.
  // Example:
  //   %1 = affine.load %alloca[0] : memref<1x!sycl_id_1>
  //   %2 = sycl.accessor.subscript %arg0[%alloca_0] : memref<?xf32, 4>
  // here 'alias(%2, %alloca) = NoAlias' because:
  //   - the types of the value '%alloca' and '%2' types are different, and
  //   - %2 is the result of an accessor.subscript operation, and
  //   - %alloca type (memref<1x!sycl_id_1>) is a MemRef of a SYCL type

  if (!relaxedAliasing) {
    Type lhsTy = lhs.getType(), rhsTy = rhs.getType();
    if ((lhsSubOp && typesDoNotAlias(lhsTy, rhsTy)) ||
        (rhsSubOp && typesDoNotAlias(rhsTy, lhsTy)))
      return AliasResult::NoAlias;
  }

  return AliasResult::MayAlias;
}
