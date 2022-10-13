// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsDialect.cpp -----------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsAlias.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

void mlir::sycl::SYCLDialect::initialize() {
  mlir::sycl::SYCLDialect::addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
      >();

  mlir::Dialect::addTypes<mlir::sycl::IDType, mlir::sycl::AccessorCommonType,
                          mlir::sycl::AccessorType, mlir::sycl::RangeType,
                          mlir::sycl::AccessorImplDeviceType,
                          mlir::sycl::ArrayType, mlir::sycl::ItemType,
                          mlir::sycl::ItemBaseType, mlir::sycl::NdItemType,
                          mlir::sycl::GroupType>();

  mlir::Dialect::addInterfaces<SYCLOpAsmInterface>();
}

mlir::Type
mlir::sycl::SYCLDialect::parseType(mlir::DialectAsmParser &Parser) const {
  mlir::StringRef Keyword;
  if (mlir::failed(Parser.parseKeyword(&Keyword))) {
    return nullptr;
  }

  if (Keyword == "id") {
    return mlir::sycl::IDType::parseType(Parser);
  }
  if (Keyword == "accessor_common") {
    return mlir::sycl::AccessorCommonType::parseType(Parser);
  }
  if (Keyword == "accessor") {
    return mlir::sycl::AccessorType::parseType(Parser);
  }
  if (Keyword == "range") {
    return mlir::sycl::RangeType::parseType(Parser);
  }
  if (Keyword == "accessor_impl_device") {
    return mlir::sycl::AccessorImplDeviceType::parseType(Parser);
  }
  if (Keyword == "array") {
    return mlir::sycl::ArrayType::parseType(Parser);
  }
  if (Keyword == "item") {
    return mlir::sycl::ItemType::parseType(Parser);
  }
  if (Keyword == "item_base") {
    return mlir::sycl::ItemBaseType::parseType(Parser);
  }
  if (Keyword == "nd_item") {
    return mlir::sycl::NdItemType::parseType(Parser);
  }
  if (Keyword == "group") {
    return mlir::sycl::GroupType::parseType(Parser);
  }

  Parser.emitError(Parser.getCurrentLocation(), "unknown SYCL type: ")
      << Keyword;
  return nullptr;
}

void mlir::sycl::SYCLDialect::printType(
    mlir::Type Type, mlir::DialectAsmPrinter &Printer) const {
  if (const auto ID = Type.dyn_cast<mlir::sycl::IDType>()) {
    Printer << "id<" << ID.getDimension() << ">";
  } else if (const auto AccCommon =
                 Type.dyn_cast<mlir::sycl::AccessorCommonType>()) {
    Printer << "accessor_common";
  } else if (const auto Acc = Type.dyn_cast<mlir::sycl::AccessorType>()) {
    Printer << "accessor<[" << Acc.getDimension() << ", " << Acc.getType()
            << ", " << Acc.getAccessModeAsString() << ", "
            << Acc.getTargetModeAsString() << "], (";
    llvm::interleaveComma(Acc.getBody(), Printer);
    Printer << ")>";
  } else if (const auto Range = Type.dyn_cast<mlir::sycl::RangeType>()) {
    Printer << "range<" << Range.getDimension() << ">";
  } else if (const auto AccDev =
                 Type.dyn_cast<mlir::sycl::AccessorImplDeviceType>()) {
    Printer << "accessor_impl_device<[" << AccDev.getDimension() << "], (";
    llvm::interleaveComma(AccDev.getBody(), Printer);
    Printer << ")>";
  } else if (const auto Arr = Type.dyn_cast<mlir::sycl::ArrayType>()) {
    Printer << "array<[" << Arr.getDimension() << "], (";
    llvm::interleaveComma(Arr.getBody(), Printer);
    Printer << ")>";
  } else if (const auto Item = Type.dyn_cast<mlir::sycl::ItemType>()) {
    Printer << "item<[" << Item.getDimension() << ", "
            << static_cast<bool>(Item.getWithOffset()) << "], (";
    llvm::interleaveComma(Item.getBody(), Printer);
    Printer << ")>";
  } else if (const auto ItemBase = Type.dyn_cast<mlir::sycl::ItemBaseType>()) {
    Printer << "item_base<[" << ItemBase.getDimension() << ", "
            << static_cast<bool>(ItemBase.getWithOffset()) << "], (";
    llvm::interleaveComma(ItemBase.getBody(), Printer);
    Printer << ")>";
  } else if (const auto NDItem = Type.dyn_cast<mlir::sycl::NdItemType>()) {
    Printer << "nd_item<[" << NDItem.getDimension() << "], (";
    llvm::interleaveComma(NDItem.getBody(), Printer);
    Printer << ")>";
  } else if (const auto Group = Type.dyn_cast<mlir::sycl::GroupType>()) {
    Printer << "group<[" << Group.getDimension() << "], (";
    llvm::interleaveComma(Group.getBody(), Printer);
    Printer << ")>";
  } else {
    assert(false && "The given type is not handled by the SYCL printer");
  }
}

llvm::Optional<llvm::StringRef>
mlir::sycl::SYCLDialect::findMethod(mlir::TypeID BaseType,
                                    llvm::StringRef MethodName) const {
  return methods.lookupMethod(BaseType, MethodName);
}

llvm::Optional<llvm::StringRef>
mlir::sycl::MethodRegistry::lookupMethod(mlir::TypeID BaseType,
                                         llvm::StringRef MethodName) const {
  const auto Iter = methods.find({BaseType, MethodName});
  return Iter == methods.end() ? llvm::None
                               : llvm::Optional<llvm::StringRef>{Iter->second};
}

bool mlir::sycl::MethodRegistry::registerMethod(mlir::TypeID TypeID,
                                                llvm::StringRef MethodName,
                                                llvm::StringRef OpName) {
  return methods.try_emplace({TypeID, MethodName}, OpName).second;
}

// If the operation is a SYCL method, register it.
template <typename T>
static typename std::enable_if_t<mlir::sycl::isSYCLMethod<T>::value>
addSYCLMethod(mlir::sycl::MethodRegistry &methods) {
  const auto TypeID = T::getTypeID();
  const llvm::StringRef OpName = T::getOperationName();
  for (llvm::StringRef Name : T::getMethodNames())
    assert(methods.registerMethod(TypeID, Name, OpName) && "Duplicated method");
}

// If the operation is not a SYCL method, do nothing.
template <typename T>
static typename std::enable_if_t<!mlir::sycl::isSYCLMethod<T>::value>
addSYCLMethod(mlir::sycl::MethodRegistry &) {}

template <typename... Args> void mlir::sycl::SYCLDialect::addOperations() {
  mlir::Dialect::addOperations<Args...>();
  (void)std::initializer_list<int>{0, (::addSYCLMethod<Args>(methods), 0)...};
}

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

  const auto HasArrayTrait =
      Input.getElementType()
          .hasTrait<mlir::sycl::SYCLInheritanceTypeInterface<
              mlir::sycl::ArrayType>::Trait>();
  const auto IsArray = Output.getElementType().isa<mlir::sycl::ArrayType>();
  return HasArrayTrait && IsArray;
}

mlir::LogicalResult mlir::sycl::SYCLAccessorSubscriptOp::verify() {
  // /* Available only when: (Dimensions > 0) */
  // reference operator[](id<Dimensions> index) const;

  // /* Available only when: (Dimensions > 1) */
  // __unspecified__ operator[](size_t index) const;

  // /* Available only when: (AccessMode != access_mode::atomic && Dimensions ==
  // 1) */
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
      .Case<mlir::sycl::IDType>([&](auto IDTy) -> mlir::LogicalResult {
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

static mlir::LogicalResult
verifyGetOperation(mlir::sycl::SYCLMethodOpInterface Op) {
  // size_t get(int dimension) const;
  // size_t &operator[](int dimension);
  // size_t operator[](int dimension) const;
  if (Op.getFunctionName() == "get" &&
      Op->getResult(0).getType().isa<mlir::MemRefType>()) {
    return Op.emitOpError(
        "The get method cannot return a reference, just a value");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::sycl::SYCLRangeGetOp::verify() {
  // size_t get(int dimension) const;
  // size_t &operator[](int dimension);
  // size_t operator[](int dimension) const;
  return verifyGetOperation(*this);
}

#include "mlir/Dialect/SYCL/IR/SYCLOpInterfaces.cpp.inc"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
