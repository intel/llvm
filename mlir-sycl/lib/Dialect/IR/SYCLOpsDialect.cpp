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

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/WithColor.h"

#define DEBUG_TYPE "SYCLOpsDialect"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.cpp.inc"

void mlir::sycl::SYCLDialect::initialize() {
  methods.init(*getContext());

  mlir::sycl::SYCLDialect::addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
      >();

  mlir::Dialect::addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.cpp.inc"
      >();

  mlir::Dialect::addInterfaces<SYCLOpAsmInterface>();
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

template <typename T>
static void addSYCLMethod(mlir::sycl::MethodRegistry &methods) {
  if constexpr (mlir::sycl::isSYCLMethod<T>::value) {
    // If the operation is a SYCL method, register it.
    const auto TypeID = T::getTypeID();
    const llvm::StringRef OpName = T::getOperationName();
    for (llvm::StringRef Name : T::getMethodNames())
      assert(methods.registerMethod(TypeID, Name, OpName) &&
             "Duplicated method");
  }
}

template <typename... Args> void mlir::sycl::SYCLDialect::addOperations() {
  mlir::Dialect::addOperations<Args...>();
  (addSYCLMethod<Args>(methods), ...);
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
