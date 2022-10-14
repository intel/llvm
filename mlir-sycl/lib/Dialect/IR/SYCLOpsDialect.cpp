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

static unsigned getDimensions(mlir::Type Type) {
  const auto GetDimension = [](auto Ty) -> unsigned {
    return Ty.getDimension();
  };
  if (auto MemRefTy = Type.dyn_cast<mlir::MemRefType>()) {
    Type = MemRefTy.getElementType();
  }
  return llvm::TypeSwitch<mlir::Type, unsigned>(Type)
      .Case<mlir::sycl::AccessorType>(GetDimension)
      .Case<mlir::sycl::RangeType>(GetDimension)
      .Case<mlir::sycl::IDType>(GetDimension)
      .Case<mlir::sycl::ItemType>(GetDimension)
      .Case<mlir::sycl::NdItemType>(GetDimension)
      .Case<mlir::sycl::GroupType>(GetDimension)
      .Default(
          [](auto) -> unsigned { llvm_unreachable("Invalid input type"); });
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

static mlir::LogicalResult
verifyGetOperation(mlir::sycl::SYCLMethodOpInterface Op) {
  // size_t get(int dimension) const;
  // size_t &operator[](int dimension);
  // size_t operator[](int dimension) const;
  // only available if Dimensions == 1
  // size_t operator size_t() const;
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

static mlir::LogicalResult
verifyGetIDOperation(mlir::sycl::SYCLMethodOpInterface Op) {
  // id<Dimensions> *() const;
  // size_t *(int dimension) const;
  // size_t operator[](int dimension) const;
  // only available if Dimensions == 1
  // operator size_t() const;
  const llvm::StringRef FuncName = Op.getFunctionName();
  const bool IsSizeTCast = FuncName == "operator unsigned long";
  const bool IsSubscript = FuncName == "operator[]";
  const mlir::Type RetTy = Op->getResult(0).getType();
  const bool IsRetScalar = RetTy.isa<mlir::sycl::IDType>();
  // operator size_t cannot be checked the generic way.
  if (FuncName != "operator unsigned long") {
    const auto GenericVerification = verifyGetSYCLTyOperation(Op, "ID");
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

static mlir::LogicalResult
verifyGetRangeOperation(mlir::sycl::SYCLMethodOpInterface Op) {
  return verifyGetSYCLTyOperation(Op, "range");
}

static mlir::LogicalResult
verifyGetGroupOperation(mlir::sycl::SYCLMethodOpInterface Op) {
  return verifyGetSYCLTyOperation(Op, "group");
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

mlir::LogicalResult mlir::sycl::SYCLRangeGetOp::verify() {
  // size_t get(int dimension) const;
  // size_t &operator[](int dimension);
  // size_t operator[](int dimension) const;
  // only available if Dimensions == 1
  // size_t operator size_t() const;
  return verifyGetOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLIDGetOp::verify() {
  // size_t get(int dimension) const;
  // size_t &operator[](int dimension);
  // size_t operator[](int dimension) const;
  // only available if Dimensions == 1
  // operator size_t() const;
  return verifyGetOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLItemGetIDOp::verify() {
  // id<Dimensions> get_id() const;
  // size_t get_id(int dimension) const;
  // size_t operator[](int dimension) const;
  // only available if Dimensions == 1
  // operator size_t() const;
  return verifyGetIDOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLItemGetRangeOp::verify() {
  // range<Dimensions> get_range() const;
  // size_t get_range(int dimension) const;
  return verifyGetRangeOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLNDItemGetGlobalIDOp::verify() {
  // id<Dimensions> get_global_id() const;
  // size_t get_global_id(int dimension) const;
  return verifyGetIDOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLNDItemGetLocalIDOp::verify() {
  // id<Dimensions> get_local_id() const;
  // size_t get_local_id(int dimension) const;
  return verifyGetIDOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLNDItemGetGroupOp::verify() {
  // group<Dimensions> get_group() const;
  // size_t get_group(int dimension) const;
  return verifyGetGroupOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLNDItemGetGroupRangeOp::verify() {
  // range<Dimensions> get_group_range() const;
  // size_t get_group_range(int dimension) const;
  return verifyGetRangeOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLNDItemGetGlobalRangeOp::verify() {
  // range<Dimensions> get_global_range() const;
  // size_t get_global_range(int dimension) const;
  return verifyGetRangeOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLNDItemGetLocalRangeOp::verify() {
  // range<Dimensions> get_local_range() const;
  // size_t get_local_range(int dimension) const;
  return verifyGetRangeOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLGroupGetGroupIDOp::verify() {
  // id<Dimensions> get_group_id() const;
  // size_t get_group_id(int dimension) const;
  return verifyGetIDOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLGroupGetLocalIDOp::verify() {
  // id<Dimensions> get_local_id() const;
  // size_t get_local_id(int dimension) const;
  return verifyGetIDOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLGroupGetLocalRangeOp::verify() {
  // range<Dimensions> get_local_range() const;
  // size_t get_local_range(int dimension) const;
  return verifyGetRangeOperation(*this);
}

mlir::LogicalResult mlir::sycl::SYCLGroupGetGroupRangeOp::verify() {
  // range<Dimensions> get_group_range() const;
  // size_t get_group_range(int dimension) const;
  return verifyGetRangeOperation(*this);
}

#include "mlir/Dialect/SYCL/IR/SYCLOpInterfaces.cpp.inc"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
