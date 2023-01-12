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
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/WithColor.h"

#define DEBUG_TYPE "SYCLOpsDialect"

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
  /// replace the \p Call operation with it.
  bool isLegalToInline(mlir::Operation *Call, mlir::Operation *Callable,
                       bool WouldBeCloned) const final {
    return true;
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
  /// single result of \p ResultType. If a conversion cannot be generated,
  /// nullptr should be returned.
  mlir::Operation *
  materializeCallConversion(mlir::OpBuilder &Builder, mlir::Value Input,
                            mlir::Type ResultType,
                            mlir::Location ConversionLoc) const final {
    return Builder.create<mlir::sycl::SYCLCastOp>(ConversionLoc, ResultType,
                                                  Input);
  }
};

class SYCLOpAsmInterface final : public mlir::OpAsmDialectInterface {
public:
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Type Type, llvm::raw_ostream &OS) const final;

private:
  static constexpr llvm::StringRef
  getAlias(mlir::sycl::MemoryAccessMode MemAccessMode) {
    switch (MemAccessMode) {
    case mlir::sycl::MemoryAccessMode::Read:
      return "r";
    case mlir::sycl::MemoryAccessMode::Write:
      return "w";
    case mlir::sycl::MemoryAccessMode::ReadWrite:
      return "rw";
    case mlir::sycl::MemoryAccessMode::DiscardWrite:
      return "dw";
    case mlir::sycl::MemoryAccessMode::DiscardReadWrite:
      return "drw";
    case mlir::sycl::MemoryAccessMode::Atomic:
      return "ato";
    }

    llvm_unreachable("Unhandled kind");
  }

  static constexpr llvm::StringRef
  getAlias(mlir::sycl::MemoryTargetMode MemTargetMode) {
    switch (MemTargetMode) {
    case mlir::sycl::MemoryTargetMode::GlobalBuffer:
      return "gb";
    case mlir::sycl::MemoryTargetMode::ConstantBuffer:
      return "cb";
    case mlir::sycl::MemoryTargetMode::Local:
      return "l";
    case mlir::sycl::MemoryTargetMode::Image:
      return "i";
    case mlir::sycl::MemoryTargetMode::HostBuffer:
      return "hb";
    case mlir::sycl::MemoryTargetMode::HostImage:
      return "hi";
    case mlir::sycl::MemoryTargetMode::ImageArray:
      return "ia";
    }

    llvm_unreachable("Unhandled kind");
  }
};

mlir::OpAsmDialectInterface::AliasResult
SYCLOpAsmInterface::getAlias(mlir::Type Type, llvm::raw_ostream &OS) const {
  return llvm::TypeSwitch<mlir::Type, AliasResult>(Type)
      // Keep approx. in the same order as in SYCLOps.td
      .Case<mlir::sycl::AccessorType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDimension()
           << "_" << Ty.getType() << "_" << getAlias(Ty.getAccessMode()) << "_"
           << getAlias(Ty.getTargetMode());
        return AliasResult::FinalAlias;
      })
      .Case<mlir::sycl::AccessorSubscriptType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_"
           << Ty.getCurrentDimension() << "_";
        return AliasResult::OverridableAlias;
      })
      .Case<mlir::sycl::AssertHappenedType, mlir::sycl::BFloat16Type,
            mlir::sycl::KernelHandlerType, mlir::sycl::StreamType>(
          [&](auto Ty) {
            OS << "sycl_" << decltype(Ty)::getMnemonic() << "_";
            return AliasResult::FinalAlias;
          })
      .Case<mlir::sycl::AtomicType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDataType()
           << "_" << mlir::sycl::accessAddressSpaceAsString(Ty.getAddrSpace())
           << "_";
        return AliasResult::FinalAlias;
      })
      .Case<mlir::sycl::AccessorImplDeviceType, mlir::sycl::ArrayType,
            mlir::sycl::GroupType, mlir::sycl::HItemType, mlir::sycl::IDType,
            mlir::sycl::LocalAccessorBaseDeviceType, mlir::sycl::NdItemType,
            mlir::sycl::NdRangeType, mlir::sycl::RangeType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDimension()
           << "_";
        return AliasResult::FinalAlias;
      })
      .Case<mlir::sycl::ItemType, mlir::sycl::ItemBaseType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDimension()
           << "_";
        return AliasResult::OverridableAlias;
      })
      .Case<mlir::sycl::LocalAccessorBaseType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDimension()
           << "_" << Ty.getType() << "_" << getAlias(Ty.getAccessMode());
        return AliasResult::FinalAlias;
      })
      .Case<mlir::sycl::LocalAccessorType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDimension()
           << "_" << Ty.getType();
        return AliasResult::FinalAlias;
      })
      .Case<mlir::sycl::MultiPtrType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDataType()
           << "_" << mlir::sycl::accessAddressSpaceAsString(Ty.getAddrSpace())
           << "_";
        return AliasResult::OverridableAlias;
      })
      .Case<mlir::sycl::GetScalarOpType, mlir::sycl::MinimumType,
            mlir::sycl::MaximumType, mlir::sycl::TupleValueHolderType>(
          [&](auto Ty) {
            OS << "sycl_" << decltype(Ty)::getMnemonic() << "_"
               << Ty.getDataType() << "_";
            return AliasResult::FinalAlias;
          })
      .Case<mlir::sycl::SwizzledVecType>([&](auto Ty) {
        const auto VecTy = Ty.getVecType();
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_"
           << VecTy.getDataType() << "_" << VecTy.getNumElements() << "_";
        return AliasResult::OverridableAlias;
      })
      .Case<mlir::sycl::TupleCopyAssignableValueHolderType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDataType()
           << "_";
        return AliasResult::OverridableAlias;
      })
      .Case<mlir::sycl::VecType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDataType()
           << "_" << Ty.getNumElements() << "_";
        return AliasResult::FinalAlias;
      })
      .Default(AliasResult::NoAlias);
}
} // namespace

//===----------------------------------------------------------------------===//
// SYCL Dialect
//===----------------------------------------------------------------------===//

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
  return std::nullopt;
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
  return Iter == Methods.end() ? std::nullopt
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
    return std::nullopt;
  }
  LLVM_DEBUG(llvm::dbgs() << "Function found: " << Iter->second << "\n");
  return Iter->second;
}

#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.cpp.inc"
