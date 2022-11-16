// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsDialect.cpp -----------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"

#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsAlias.h"

#include "mlir/IR/DialectImplementation.h"

void mlir::sycl::SYCLDialect::initialize() {
  mlir::sycl::SYCLDialect::addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
      >();

  mlir::Dialect::addTypes<
      mlir::sycl::IDType, mlir::sycl::AccessorCommonType,
      mlir::sycl::AccessorType, mlir::sycl::RangeType, mlir::sycl::NdRangeType,
      mlir::sycl::AccessorImplDeviceType, mlir::sycl::ArrayType,
      mlir::sycl::ItemType, mlir::sycl::ItemBaseType, mlir::sycl::NdItemType,
      mlir::sycl::GroupType, mlir::sycl::AtomicType, mlir::sycl::MultiPtrType>();

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
  if (Keyword == "nd_range") {
    return mlir::sycl::NdRangeType::parseType(Parser);
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
  if (Keyword == "atomic") {
    return mlir::sycl::AtomicType::parseType(Parser);
  }
  if (Keyword == "multi_ptr") {
    return mlir::sycl::MultiPtrType::parseType(Parser);
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
  } else if (const auto NdRange = Type.dyn_cast<mlir::sycl::NdRangeType>()) {
    Printer << "nd_range<[" << NdRange.getDimension() << "], (";
    llvm::interleaveComma(NdRange.getBody(), Printer);
    Printer << ")>";
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
  } else if (const auto Atomic = Type.dyn_cast<mlir::sycl::AtomicType>()) {
    Printer << "atomic<(";
    llvm::interleaveComma(Atomic.getBody(), Printer);
    Printer << ")>";
  }  else if (const auto MultiPtr = Type.dyn_cast<mlir::sycl::MultiPtrType>()) {
    Printer << "multi_ptr<(";
    llvm::interleaveComma(MultiPtr.getBody(), Printer);
    Printer << ")>";
  }  
  else {
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

#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.cpp.inc"
