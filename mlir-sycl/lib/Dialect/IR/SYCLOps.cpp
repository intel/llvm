//===--- SYCLOps.cpp ------------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"

#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"

bool mlir::sycl::SYCLCastOp::areCastCompatible(::mlir::TypeRange Inputs,
                                               ::mlir::TypeRange Outputs) {
  if (Inputs.size() != 1 || Outputs.size() != 1) {
    return false;
  }

  const auto Input = Inputs.front().dyn_cast<MemRefType>();
  const auto Output = Outputs.front().dyn_cast<MemRefType>();
  if (!Input || !Output) {
    return false;
  }

  /// This is a hack - Since the sycl's CastOp takes as input/output MemRef, we
  /// want to ensure that the cast is valid within MemRef's world.
  /// In order to do that, we create a temporary Output that have the same
  /// MemRef characteristic to check the MemRef cast without having the
  /// ElementType triggering a condition like
  /// (Input.getElementType() != Output.getElementType()).
  /// This ElementType condition is checked later in this function.
  const auto TempOutput =
      mlir::MemRefType::get(Output.getShape(), Input.getElementType(),
                            Output.getLayout(), Output.getMemorySpace());
  if (!mlir::memref::CastOp::areCastCompatible(Input, TempOutput)) {
    return false;
  }

  const bool HasArrayTrait =
      Input.getElementType()
          .hasTrait<mlir::sycl::SYCLInheritanceTypeInterface<
              mlir::sycl::ArrayType>::Trait>();
  const bool IsArray = Output.getElementType().isa<mlir::sycl::ArrayType>();
  if (HasArrayTrait && IsArray)
    return true;

  const bool HasAccessorCommonTrait =
      Input.getElementType()
          .hasTrait<mlir::sycl::SYCLInheritanceTypeInterface<
              mlir::sycl::AccessorCommonType>::Trait>();
  const bool IsAccessorCommon =
      Output.getElementType().isa<mlir::sycl::AccessorCommonType>();
  if (HasAccessorCommonTrait && IsAccessorCommon)
    return true;

  return false;
}

mlir::LogicalResult mlir::sycl::SYCLConstructorOp::verify() {
  if (getOperand(0).getType().dyn_cast<mlir::MemRefType>() && isSYCLType(getOperand(0).getType().cast<mlir::MemRefType>().getElementType()))
    return success();
    
  return emitOpError(
        "The first argument of a sycl::constructor op has to be a MemRef to a SYCL type");
}

mlir::LogicalResult mlir::sycl::SYCLAccessorSubscriptOp::verify() {
  // Available only when: (Dimensions > 0)
  // reference operator[](id<Dimensions> index) const;

  // Available only when: (Dimensions > 1)
  // __unspecified__ operator[](size_t index) const;

  // Available only when: (AccessMode != access_mode::atomic && Dimensions == 1)
  // reference operator[](size_t index) const;
  const auto AccessorTy = getOperand(0)
                              .getType()
                              .cast<mlir::MemRefType>()
                              .getElementType()
                              .cast<mlir::sycl::AccessorType>();

  const unsigned Dimensions = AccessorTy.getDimension();
  if (Dimensions == 0)
    return emitOpError("Dimensions cannot be zero");

  const auto verifyResultType = [&]() -> mlir::LogicalResult {
    const auto resultType = getResult().getType().dyn_cast<mlir::MemRefType>();

    if (!resultType) {
      return emitOpError("Expecting memref return type. Got ") << resultType;
    }

    if (resultType.getElementType() != AccessorTy.getType()) {
      return emitOpError(
                 "Expecting a reference to this accessor's value type (")
             << AccessorTy.getType() << "). Got " << resultType;
    }

    return success();
  };

  return mlir::TypeSwitch<mlir::Type, mlir::LogicalResult>(
             getOperand(1).getType())
      .Case<mlir::MemRefType>([&](auto MemRefTy) -> mlir::LogicalResult {
        Type ElemTy = MemRefTy.getElementType();
        auto IDTy = ElemTy.dyn_cast<mlir::sycl::IDType>();
        assert(IDTy && "Unhandled input memref type");
        if (IDTy.getDimension() != Dimensions) {
          return emitOpError(
                     "Both the index and the accessor must have the same "
                     "number of dimensions, but the accessor has ")
                 << Dimensions << "dimensions and the index, "
                 << IDTy.getDimension();
        }
        return verifyResultType();
      })
      .Case<mlir::IntegerType>([&](auto) -> mlir::LogicalResult {
        if (Dimensions != 1) {
          // Implementation defined result type.
          return success();
        }
        if (AccessorTy.getAccessMode() ==
            mlir::sycl::MemoryAccessMode::Atomic) {
          return emitOpError(
              "Cannot use this signature when the atomic access mode is used");
        }
        return verifyResultType();
      })
      .Default([&](auto) -> mlir::LogicalResult {
        llvm_unreachable("Unhandled input type");
      });
}

#include "mlir/Dialect/SYCL/IR/SYCLOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
