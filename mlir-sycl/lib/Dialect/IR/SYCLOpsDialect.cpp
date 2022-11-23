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
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/WithColor.h"

#define DEBUG_TYPE "SYCLOpsDialect"

void mlir::sycl::SYCLDialect::initialize() {
  methods.init(*getContext());

  mlir::sycl::SYCLDialect::addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
      >();

  mlir::Dialect::addTypes<
      mlir::sycl::IDType, mlir::sycl::AccessorCommonType,
      mlir::sycl::AccessorType, mlir::sycl::RangeType, mlir::sycl::NdRangeType,
      mlir::sycl::AccessorImplDeviceType, mlir::sycl::AccessorSubscriptType,
      mlir::sycl::ArrayType, mlir::sycl::ItemType, mlir::sycl::ItemBaseType,
      mlir::sycl::NdItemType, mlir::sycl::GroupType, mlir::sycl::VecType,
      mlir::sycl::AtomicType>();

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
  if (Keyword == "accessor_subscript") {
    return mlir::sycl::AccessorSubscriptType::parseType(Parser);
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
  if (Keyword == "vec") {
    return mlir::sycl::VecType::parseType(Parser);
  }
  if (Keyword == "atomic") {
    return mlir::sycl::AtomicType::parseType(Parser);
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
  } else if (const auto AccSub =
                 Type.dyn_cast<mlir::sycl::AccessorSubscriptType>()) {
    Printer << "accessor_subscript<[" << AccSub.getCurrentDimension() << "], (";
    llvm::interleaveComma(AccSub.getBody(), Printer);
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
  } else if (const auto Vec = Type.dyn_cast<mlir::sycl::VecType>()) {
    Printer << "vec<[" << Vec.getDataType() << ", " << Vec.getNumElements()
            << "], (";
    llvm::interleaveComma(Vec.getBody(), Printer);
    Printer << ")>";
  } else if (const auto Atomic = Type.dyn_cast<mlir::sycl::AtomicType>()) {
    Printer << "atomic<[" << Atomic.getDataType() << ", "
            << Atomic.getAddressSpaceAsString() << "]>";
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
mlir::sycl::SYCLDialect::findMethodFromBaseClass(
    mlir::TypeID BaseType, llvm::StringRef MethodName) const {
  if (auto Method = findMethod(BaseType, MethodName))
    return Method;
  for (auto DerivedTy : getDerivedTypes(BaseType)) {
    if (auto Method = findMethod(DerivedTy, MethodName))
      return Method;
  }
  return llvm::None;
}

void mlir::sycl::SYCLDialect::registerMethodDefinition(
    llvm::StringRef Name, mlir::func::FuncOp Func) {
  methods.registerDefinition(Name, Func);
}

llvm::Optional<mlir::func::FuncOp>
mlir::sycl::SYCLDialect::lookupMethodDefinition(llvm::StringRef Name,
                                                mlir::FunctionType Type) const {
  return methods.lookupDefinition(Name, Type);
}

void mlir::sycl::MethodRegistry::init(mlir::MLIRContext &Ctx) {
  assert(!Module && "Registry already initialized");
  Module = ModuleOp::create(mlir::UnknownLoc::get(&Ctx), ModuleName);
}

llvm::Optional<llvm::StringRef>
mlir::sycl::MethodRegistry::lookupMethod(mlir::TypeID BaseType,
                                         llvm::StringRef MethodName) const {
  const auto Iter = Methods.find({BaseType, MethodName});
  return Iter == Methods.end() ? llvm::None
                               : llvm::Optional<llvm::StringRef>{Iter->second};
}

bool mlir::sycl::MethodRegistry::registerMethod(mlir::TypeID TypeID,
                                                llvm::StringRef MethodName,
                                                llvm::StringRef OpName) {
  return Methods.try_emplace({TypeID, MethodName}, OpName).second;
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

namespace llvm {
template <> struct DenseMapInfo<llvm::SmallString<0>> {
  using SmallString = llvm::SmallString<0>;

  static unsigned getHashValue(const SmallString &S) {
    return llvm::hash_value(S);
  }

  static bool isEqual(const SmallString &LHS, const SmallString &RHS) {
    return LHS == RHS;
  }

  static SmallString getEmptyKey() { return SmallString(""); }

  static SmallString getTombstoneKey() { return SmallString("~"); }
};
} // namespace llvm

void mlir::sycl::MethodRegistry::registerDefinition(llvm::StringRef Name,
                                                    mlir::func::FuncOp Func) {
  LLVM_DEBUG(llvm::dbgs() << "Registering function \"" << Name << "\": " << Func
                          << "\n");
  auto Clone = Func.clone();
  const auto FuncType = Clone.getFunctionType();
  auto Iter =
      Definitions.insert_as<std::pair<llvm::StringRef, mlir::FunctionType>>(
          {{Name, FuncType}, Clone}, {Name, FuncType});
  if (!Iter.second) {
    // Override current function.
    auto &ToOverride = Iter.first->second;
    assert(ToOverride.isDeclaration() && "Only a declaration can be overriden");
    assert(!Func.isDeclaration() &&
           "A declaration cannot be used to override another declaration");
    assert(ToOverride.getName() == Func.getName() &&
           "Functions must have the same mangled name");
    ToOverride.erase();
    ToOverride = Clone;
  }
  Module.push_back(Clone);
}

llvm::Optional<mlir::func::FuncOp> mlir::sycl::MethodRegistry::lookupDefinition(
    llvm::StringRef Name, mlir::FunctionType FuncType) const {
  LLVM_DEBUG(llvm::dbgs() << "Fetching function \"" << Name
                          << "\" with type: " << FuncType << "\n");

  const auto Iter =
      Definitions.find_as<std::pair<llvm::StringRef, mlir::FunctionType>>(
          {Name, FuncType});
  if (Iter == Definitions.end()) {
    llvm::WithColor::warning() << "Could not find function \"" << Name
                               << "\" with type " << FuncType << "\n";
    return llvm::None;
  }
  LLVM_DEBUG(llvm::dbgs() << "Function found: " << Iter->second << "\n");
  return Iter->second;
}

#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.cpp.inc"
