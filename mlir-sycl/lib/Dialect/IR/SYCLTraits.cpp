// Copyright (C) Codeplay Software Limited

//===--- SYCLTraits.cpp ---------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOpTraits.h"

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"

#include "llvm/ADT/TypeSwitch.h"

static unsigned getDimensions(mlir::Type Type) {
  if (auto MemRefTy = Type.dyn_cast<mlir::MemRefType>()) {
    Type = MemRefTy.getElementType();
  }
  return llvm::TypeSwitch<mlir::Type, unsigned>(Type)
      .Case<mlir::sycl::AccessorType, mlir::sycl::ItemType,
            mlir::sycl::NdRangeType, mlir::sycl::GroupType, mlir::sycl::IDType,
            mlir::sycl::NdItemType, mlir::sycl::RangeType>(
          [](auto Ty) { return Ty.getDimension(); });
}

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
