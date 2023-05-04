//===--- SYCLTraits.cpp ---------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLTraits.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sycl;

static LogicalResult verifyEqualDimensions(Operation *op) {
  const auto retTy = op->getResult(0).getType();
  if (retTy.isInteger(64)) {
    return success();
  }
  const unsigned thisDimensions = getDimensions(op->getOperandTypes()[0]);
  const unsigned retDimensions = getDimensions(retTy);
  if (thisDimensions != retDimensions) {
    return op->emitOpError("base type and return type dimensions mismatch: ")
           << thisDimensions << " vs " << retDimensions;
  }
  return success();
}

LogicalResult mlir::sycl::verifySYCLGetComponentTrait(Operation *op) {
  // size_t get(int dimension) const;
  // size_t &operator[](int dimension);
  // size_t operator[](int dimension) const;
  // only available if Dimensions == 1
  // size_t operator size_t() const;
  const unsigned numOperands = op->getNumOperands();
  assert((numOperands == 1 || numOperands == 2) &&
         "This operation can only accept one or two operands");
  const auto resultTypes = op->getResultTypes();
  const auto operandTypes = op->getOperandTypes();
  if (numOperands == 1) {
    Type type = resultTypes[0];
    if (!type.isIntOrIndex())
      return op->emitOpError(
          "must return a scalar type when a single argument is provided");
    if (getDimensions(operandTypes[0]) != 1)
      return op->emitOpError("operand 0 must have a single dimension to be "
                             "passed as the single argument to this operation");
  }
  return success();
}

static LogicalResult
verifyGetSYCLTyOperation(Operation *op, llvm::StringRef expectedRetTyName) {
  // SYCLTy *() const;
  // size_t *(int dimension) const;
  const unsigned numOperands = op->getNumOperands();
  assert((numOperands == 1 || numOperands == 2) &&
         "This operation can only accept one or two operands");
  const Type retTy = op->getResult(0).getType();
  const bool isI64RetTy = retTy.isInteger(64);
  switch (op->getNumOperands()) {
  case 1:
    if (isI64RetTy) {
      return op->emitOpError("expecting ")
             << expectedRetTyName << " result type. Got " << retTy;
    }
    return verifyEqualDimensions(op);
  case 2:
    if (!isI64RetTy) {
      return op->emitOpError("expecting an I64 result type. Got ") << retTy;
    }
    return success();
  default:
    llvm_unreachable("Invalid number of operands");
  }
}

LogicalResult mlir::sycl::verifySYCLGetIDTrait(Operation *op) {
  // id<Dimensions> *() const;
  // size_t *(int dimension) const;
  // size_t operator[](int dimension) const;
  // only available if Dimensions == 1
  // operator size_t() const;
  const unsigned numOperands = op->getNumOperands();
  assert((numOperands == 1 || numOperands == 2) &&
         "This operation can only accept one or two operands");
  Type retTy = op->getResultTypes()[0];
  auto operandTypes = op->getOperandTypes();
  if (retTy.isIntOrIndex()) {
    if (numOperands == 1 && getDimensions(operandTypes[0]) != 1)
      return op->emitOpError(
          "operand 0 must have a single dimension to be passed as the single "
          "argument to this operation");
  } else if (isa<IDType>(retTy)) {
    if (numOperands != 1)
      return op->emitOpError(
          "must be passed a single argument in order to define an id value");
    return verifyEqualDimensions(op);
  }
  return success();
}

LogicalResult mlir::sycl::verifySYCLGetRangeTrait(Operation *op) {
  return verifyGetSYCLTyOperation(op, "range");
}

LogicalResult mlir::sycl::verifySYCLGetGroupTrait(Operation *op) {
  return verifyGetSYCLTyOperation(op, "group");
}
