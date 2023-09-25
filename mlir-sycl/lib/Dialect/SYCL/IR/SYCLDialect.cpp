//===--- SYCLDialect.cpp - SYCL Dialect registration in MLIR --------------===//
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

#include "mlir/Dialect/SYCL/IR/SYCLDialect.h"

#include "mlir/Dialect/SYCL/IR/SYCLAttributes.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/IR/SYCLTypes.h"
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
                       mlir::IRMapping &ValueMapping) const final {
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
  static llvm::StringRef getAlias(mlir::sycl::AccessMode Attr) {
    switch (Attr) {
    case mlir::sycl::AccessMode::Read:
      return "r";
    case mlir::sycl::AccessMode::Write:
      return "w";
    case mlir::sycl::AccessMode::ReadWrite:
      return "rw";
    case mlir::sycl::AccessMode::DiscardWrite:
      return "dw";
    case mlir::sycl::AccessMode::DiscardReadWrite:
      return "drw";
    case mlir::sycl::AccessMode::Atomic:
      return "ato";
    }

    llvm_unreachable("Unhandled kind");
  }

  static llvm::StringRef getAlias(mlir::sycl::Target Target) {
    switch (Target) {
    case mlir::sycl::Target::Device:
      return "dev";
    case mlir::sycl::Target::ConstantBuffer:
      return "cb";
    case mlir::sycl::Target::Local:
      return "l";
    case mlir::sycl::Target::Image:
      return "i";
    case mlir::sycl::Target::HostBuffer:
      return "hb";
    case mlir::sycl::Target::HostImage:
      return "hi";
    case mlir::sycl::Target::ImageArray:
      return "ia";
    }

    llvm_unreachable("Unhandled kind");
  }

  static llvm::StringRef getAlias(mlir::sycl::AccessAddrSpace AddrSpace) {
    switch (AddrSpace) {
    case mlir::sycl::AccessAddrSpace::GlobalAccess:
      return "glo";
    case mlir::sycl::AccessAddrSpace::PrivateAccess:
      return "pri";
    case mlir::sycl::AccessAddrSpace::LocalAccess:
      return "loc";
    case mlir::sycl::AccessAddrSpace::ConstantAccess:
      return "cons";
    case mlir::sycl::AccessAddrSpace::GenericAccess:
      return "gen";
    case mlir::sycl::AccessAddrSpace::ExtIntelGlobalDeviceAccess:
      return "ext_int_gda";
    case mlir::sycl::AccessAddrSpace::ExtIntelHostAccess:
      return "ext_int_ha";
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
      .Case<mlir::sycl::HalfType, mlir::sycl::KernelHandlerType,
            mlir::sycl::StreamType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic();
        return AliasResult::FinalAlias;
      })
      .Case<mlir::sycl::AtomicType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDataType()
           << "_" << getAlias(Ty.getAddrSpace());
        return AliasResult::FinalAlias;
      })
      .Case<mlir::sycl::AccessorImplDeviceType, mlir::sycl::ArrayType,
            mlir::sycl::GroupType, mlir::sycl::IDType,
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
      .Case<mlir::sycl::MaximumType, mlir::sycl::MinimumType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDataType()
           << "_";
        return AliasResult::FinalAlias;
      })
      .Case<mlir::sycl::MultiPtrType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_" << Ty.getDataType()
           << "_" << getAlias(Ty.getAddrSpace());
        return AliasResult::OverridableAlias;
      })
      .Case<mlir::sycl::SwizzledVecType>([&](auto Ty) {
        const auto VecTy = Ty.getVecType();
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_"
           << VecTy.getDataType() << "_" << VecTy.getNumElements() << "_";
        return AliasResult::OverridableAlias;
      })
      .Case<mlir::sycl::VecType>([&](auto Ty) {
        OS << "sycl_" << decltype(Ty)::getMnemonic() << "_";
        mlir::Type DataTy = Ty.getDataType();
        if (getAlias(DataTy, OS) == AliasResult::NoAlias)
          OS << DataTy;
        OS << "_" << Ty.getNumElements() << "_";
        return AliasResult::FinalAlias;
      })
      .Default(AliasResult::NoAlias);
}
} // namespace

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLAttributes.cpp.inc"
#undef GET_ATTRDEF_CLASSES

//===----------------------------------------------------------------------===//
// SYCL Dialect
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.cpp.inc"

void mlir::sycl::SYCLDialect::initialize() {
  mlir::sycl::SYCLDialect::addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
      >();

  mlir::Dialect::addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.cpp.inc"
      >();

  mlir::Dialect::addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/SYCL/IR/SYCLAttributes.cpp.inc"
#undef GET_ATTRDEF_LIST
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

#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.cpp.inc"
