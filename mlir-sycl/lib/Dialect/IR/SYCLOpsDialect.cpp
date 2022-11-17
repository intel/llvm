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
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// SYCL Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

/// This class defines the interface for handling inlining SYCL operations.
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
    bool IsSYCLCall = mlir::isa<mlir::sycl::SYCLCallOp>(Call);
    DEBUG_WITH_TYPE("inlining", {
      if (!IsSYCLCall)
        llvm::dbgs() << "Cannot yet inline " << *Call << "\n";
    });

    return IsSYCLCall;
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

#if 0
  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from the SYCL dialect, and a callable region. This method should generate
  /// an operation that takes \p Input as the only operand, and produces a
  /// single result of \p ResultType. If a conversion can not be generated,
  /// nullptr should be returned.
  Operation *materializeCallConversion(OpBuilder &Builder, Value Input,
                                       Type ResultType,
                                       Location ConversionLoc) const final {
    return Builder.create<SYCLCastOp>(ConversionLoc, ResultType, Input);
  }
#endif
};

} // namespace

//===----------------------------------------------------------------------===//
// SYCL Dialect
//===----------------------------------------------------------------------===//

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
      mlir::sycl::GroupType>();

  mlir::Dialect::addInterfaces<SYCLOpAsmInterface>();
  mlir::Dialect::addInterfaces<SYCLInlinerInterface>();
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

#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.cpp.inc"
