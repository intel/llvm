//===--- SYCLOpsDialect.cpp - SYCL Dialect registration in MLIR -----------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the dialect for the SYCL IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsAlias.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

//===----------------------------------------------------------------------===//
// SYCL Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

/// This class defines the interface for inlining SYCL operations.
class SYCLInlinerInterface : public mlir::DialectInlinerInterface {
public:
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  /// This hook checks whether is legal to inline the \p Callable operation and
  /// replace the \p Call operation with it. For the SYCL dialect we want to
  /// allow inlining only SYCLCallOp operations.
  bool isLegalToInline(mlir::Operation *Call, mlir::Operation *Callable,
                       bool WouldBeCloned) const final {
    return mlir::isa<mlir::sycl::SYCLCallOp>(Call);
  }

  /// This hook checks whether is legal to inline the \p Op operation into the
  /// \p Dest region. All operations in the SYCL dialect are legal to inline.
  bool isLegalToInline(mlir::Operation *Op, mlir::Region *Dest,
                       bool WouldBeCloned,
                       mlir::BlockAndValueMapping &ValueMapping) const final {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from the SYCL dialect, and a callable region. This method should generate
  /// an operation that takes \p Input as the only operand, and produces a
  /// single result of \p ResultType. If a conversion can not be generated,
  /// nullptr should be returned.
  mlir::Operation *
  materializeCallConversion(mlir::OpBuilder &Builder, mlir::Value Input,
                            mlir::Type ResultType,
                            mlir::Location ConversionLoc) const final {
    return Builder.create<mlir::sycl::SYCLCastOp>(ConversionLoc, ResultType,
                                                  Input);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// SYCL Dialect
//===----------------------------------------------------------------------===//

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
  mlir::Dialect::addInterfaces<SYCLInlinerInterface>();
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
