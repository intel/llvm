// Copyright (C) Codeplay Software Limited

//===--- SYCLTraits.cpp ---------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOpTraits.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sycl;

static mlir::LogicalResult
verifyEqualDimensions(mlir::sycl::SYCLMethodOpInterface Op) {
  const auto RetTy = Op->getResult(0).getType();
  if (RetTy.isInteger(64)) {
    return mlir::success();
  }
  const unsigned ThisDimensions = getDimensions(Op.getBaseType());
  const unsigned RetDimensions = getDimensions(RetTy);
  if (ThisDimensions != RetDimensions) {
    return Op->emitOpError("Base type and return type dimensions mismatch: ")
           << ThisDimensions << " vs " << RetDimensions;
  }
  return mlir::success();
}

mlir::LogicalResult mlir::sycl::verifySYCLGetComponentTrait(Operation *OpPtr) {
  // size_t get(int dimension) const;
  // size_t &operator[](int dimension);
  // size_t operator[](int dimension) const;
  // only available if Dimensions == 1
  // size_t operator size_t() const;
  auto Op = cast<SYCLMethodOpInterface>(OpPtr);
  const llvm::StringRef FunctionName = Op.getFunctionName();
  const bool IsSizeTCast = Op.getFunctionName() == "operator unsigned long";
  const mlir::Type RetTy = Op->getResult(0).getType();
  const bool IsScalarReturn = RetTy.isInteger(64);
  switch (Op->getNumOperands()) {
  case 1: {
    if (!IsSizeTCast) {
      return Op->emitOpError("The ")
             << FunctionName << " function expects an index argument";
    }
    if (!IsScalarReturn) {
      return Op->emitOpError(
                 "A cast to size_t must return a size_t value. Got ")
             << RetTy;
    }
    const unsigned Dimensions = getDimensions(Op.getBaseType());
    if (Dimensions != 1) {
      return Op->emitOpError("A cast to size_t can only be performed when the "
                             "number of dimensions is one. Got ")
             << Dimensions;
    }
    break;
  }
  case 2: {
    if (IsSizeTCast) {
      return Op->emitOpError(
          "A cast operation cannot recieve more than one argument");
    }
    if (FunctionName == "get" && !IsScalarReturn) {
      return Op.emitOpError(
          "The get method cannot return a reference, just a value");
    }
    break;
  }
  default:
    llvm_unreachable("Invalid number of operands");
  }
  return mlir::success();
}

static mlir::LogicalResult
verifyGetSYCLTyOperation(mlir::sycl::SYCLMethodOpInterface Op,
                         llvm::StringRef ExpectedRetTyName) {
  // SYCLTy *() const;
  // size_t *(int dimension) const;
  const mlir::Type RetTy = Op->getResult(0).getType();
  const bool IsI64RetTy = RetTy.isInteger(64);
  switch (Op->getNumOperands()) {
  case 1:
    if (IsI64RetTy) {
      return Op->emitError("Expecting ")
             << ExpectedRetTyName << " result type. Got " << RetTy;
    }
    return verifyEqualDimensions(Op);
  case 2:
    if (!IsI64RetTy) {
      return Op->emitError("Expecting an I64 result type. Got ") << RetTy;
    }
    return mlir::success();
  default:
    llvm_unreachable("Invalid number of operands");
  }
}

mlir::LogicalResult mlir::sycl::verifySYCLGetIDTrait(Operation *OpPtr) {
  // id<Dimensions> *() const;
  // size_t *(int dimension) const;
  // size_t operator[](int dimension) const;
  // only available if Dimensions == 1
  // operator size_t() const;
  auto Op = cast<SYCLMethodOpInterface>(OpPtr);
  const llvm::StringRef FuncName = Op.getFunctionName();
  const bool IsSizeTCast = FuncName == "operator unsigned long";
  const bool IsSubscript = FuncName == "operator[]";
  const mlir::Type RetTy = Op->getResult(0).getType();
  const bool IsRetScalar = RetTy.isa<mlir::sycl::IDType>();
  // operator size_t cannot be checked the generic way.
  if (FuncName != "operator unsigned long") {
    const LogicalResult GenericVerification =
        verifyGetSYCLTyOperation(Op, "ID");
    if (GenericVerification.failed()) {
      return GenericVerification;
    }
  }
  switch (Op->getNumOperands()) {
  case 1: {
    if (IsSubscript) {
      return Op->emitOpError("operator[] expects an index argument");
    }
    if (IsSizeTCast) {
      if (IsRetScalar) {
        return Op->emitOpError(
                   "A cast to size_t must return a size_t value. Got ")
               << RetTy;
      }
      const unsigned Dimensions = getDimensions(Op.getBaseType());
      if (Dimensions != 1) {
        return Op->emitOpError(
                   "A cast to size_t can only be performed when the "
                   "number of dimensions is one. Got ")
               << Dimensions;
      }
    }
    break;
  }
  case 2: {
    if (IsSizeTCast) {
      return Op->emitOpError(
          "A cast operation cannot recieve more than one argument");
    }
    break;
  }
  default:
    llvm_unreachable("Invalid number of operands");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::sycl::verifySYCLGetRangeTrait(Operation *Op) {
  return verifyGetSYCLTyOperation(cast<mlir::sycl::SYCLMethodOpInterface>(Op),
                                  "range");
}

mlir::LogicalResult mlir::sycl::verifySYCLGetGroupTrait(Operation *Op) {
  return verifyGetSYCLTyOperation(cast<mlir::sycl::SYCLMethodOpInterface>(Op),
                                  "group");
}

static LogicalResult verifyIndexSpaceTrait(Operation *Op) {
  const auto Ty = Op->getResultTypes();
  assert(Ty.size() == 1 && "Expecting a single return value");
  const auto IsIndex = Ty[0].isa<IndexType>();
  switch (Op->getNumOperands()) {
  case 0:
    return !IsIndex ? success()
                    : Op->emitOpError("Not expecting an index return value for "
                                      "this cardinality");
  case 1:
    if (auto C = Op->getOperand(0).getDefiningOp<arith::ConstantOp>()) {
      const auto Value = static_cast<arith::ConstantIntOp>(C).value();
      if (!(0 <= Value && Value < 3)) {
        return Op->emitOpError(
            "The SYCL index space can only be 1, 2, or 3 dimensional");
      }
    }
    return IsIndex
               ? success()
               : Op->emitOpError(
                     "Expecting an index return value for this cardinality");
  default:
    llvm_unreachable("Invalid cardinality");
  }
}

LogicalResult mlir::sycl::verifySYCLIndexSpaceGetIDTrait(Operation *Op) {
  return verifyIndexSpaceTrait(Op);
}

LogicalResult mlir::sycl::verifySYCLIndexSpaceGetRangeTrait(Operation *Op) {
  return verifyIndexSpaceTrait(Op);
}
