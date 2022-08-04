// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsDialect.cpp -----------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SYCL/SYCLOpsDialect.h"
#include "SYCL/SYCLOpsAlias.h"
#include "SYCL/SYCLOpsTypes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

void mlir::sycl::SYCLDialect::initialize() {
  mlir::Dialect::addOperations<
#define GET_OP_LIST
#include "SYCL/SYCLOps.cpp.inc"
      >();

  mlir::Dialect::addTypes<
      mlir::sycl::IDType, mlir::sycl::AccessorType, mlir::sycl::RangeType,
      mlir::sycl::AccessorImplDeviceType, mlir::sycl::ArrayType,
      mlir::sycl::ItemType, mlir::sycl::ItemBaseType, mlir::sycl::NdItemType,
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

#include "SYCL/SYCLOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "SYCL/SYCLOps.cpp.inc"
