//===- AliasAnalysis.cpp - Alias Analysis for the SYCL MLIR dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
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

/// Return true if \p val is a function argument that has \p attr as attribute.
static bool isArgumentWithAttribute(Value val, const Twine &attr) {
  if (!isFuncArg(val))
    return false;

  auto blockArg = cast<BlockArgument>(val);
  auto func = cast<FunctionOpInterface>(blockArg.getOwner()->getParentOp());
  auto stringAttr = StringAttr::get(val.getContext(), attr);
  return (func.getArgAttr(blockArg.getArgNumber(), stringAttr) != nullptr);
}

// Return true if the value \p val is a function argument that has the
// 'llvm.noalias' attribute, and false otherwise.
static bool isNoAliasArgument(Value val) {
  return isArgumentWithAttribute(
      val, Twine(LLVM::LLVMDialect::getDialectNamespace())
               .concat(".")
               .concat(llvm::Attribute::getNameFromAttrKind(
                   llvm::Attribute::NoAlias)));
}

// Return true if the value \p val is a function argument that has the
// 'sycl.inner.disjoint' attribute, and false otherwise.
static bool isSYCLInnerDisjointArgument(Value val) {
  return isArgumentWithAttribute(val,
                                 sycl::SYCLDialect::getInnerDisjointAttrName());
}

// Return true is the given type \p ty is a MemRef type with a SYCL element
// type.
static bool isMemRefOfSYCLType(Type ty) {
  if (auto mt = dyn_cast<MemRefType>(ty))
    return isa<sycl::SYCLType>(mt.getElementType());
  return false;
}

//===----------------------------------------------------------------------===//
// AliasAnalysis
//===----------------------------------------------------------------------===//

AliasResult sycl::AliasAnalysis::aliasImpl(Value lhs, Value rhs) {
  if (lhs == rhs)
    return AliasResult::MustAlias;

  if (AliasResult aliasResult = handleNoAliasArguments(lhs, rhs);
      !aliasResult.isMay())
    return aliasResult;

  return handleSYCLAlias(lhs, rhs);
}

AliasResult sycl::AliasAnalysis::handleNoAliasArguments(Value lhs, Value rhs) {
  if (isNoAliasArgument(lhs) || isNoAliasArgument(rhs))
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

  // Buffers in SYCL accessors with attribute 'sycl.inner.disjoint' are
  // considered not aliased.
  if (lhsSubOp && rhsSubOp && isSYCLInnerDisjointArgument(lhsSubOp.getAcc()) &&
      isSYCLInnerDisjointArgument(rhsSubOp.getAcc()))
    return AliasResult::NoAlias;

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
