// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsTypes.cpp -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "llvm/ADT/TypeSwitch.h"

llvm::StringRef mlir::sycl::memoryAccessModeAsString(
    mlir::sycl::MemoryAccessMode MemAccessMode) {
  switch (MemAccessMode) {
  case MemoryAccessMode::Read:
    return "read";
  case MemoryAccessMode::Write:
    return "write";
  case MemoryAccessMode::ReadWrite:
    return "read_write";
  case MemoryAccessMode::DiscardWrite:
    return "discard_write";
  case MemoryAccessMode::DiscardReadWrite:
    return "discard_read_write";
  case MemoryAccessMode::Atomic:
    return "atomic";
  }
}

mlir::LogicalResult mlir::sycl::parseMemoryAccessMode(
    mlir::AsmParser &Parser,
    mlir::FailureOr<mlir::sycl::MemoryAccessMode> &MemAccessMode) {
  mlir::StringRef Keyword;
  if (Parser.parseKeyword(&Keyword)) {
    return mlir::ParseResult::failure();
  }

  if (Keyword == "read") {
    MemAccessMode.emplace(mlir::sycl::MemoryAccessMode::Read);
  } else if (Keyword == "write") {
    MemAccessMode.emplace(mlir::sycl::MemoryAccessMode::Write);
  } else if (Keyword == "read_write") {
    MemAccessMode.emplace(mlir::sycl::MemoryAccessMode::ReadWrite);
  } else if (Keyword == "discard_write") {
    MemAccessMode.emplace(mlir::sycl::MemoryAccessMode::DiscardWrite);
  } else if (Keyword == "discard_read_write") {
    MemAccessMode.emplace(mlir::sycl::MemoryAccessMode::DiscardReadWrite);
  } else if (Keyword == "atomic") {
    MemAccessMode.emplace(mlir::sycl::MemoryAccessMode::Atomic);
  } else {
    return Parser.emitError(Parser.getCurrentLocation(),
                            "expected valid MemoryAccessMode keyword");
  }

  return mlir::ParseResult::success();
}

void mlir::sycl::printMemoryAccessMode(AsmPrinter &Printer,
                                       MemoryAccessMode MemAccessMode) {
  Printer << memoryAccessModeAsString(MemAccessMode);
}

llvm::StringRef mlir::sycl::memoryTargetModeAsString(
    mlir::sycl::MemoryTargetMode MemTargetMode) {
  switch (MemTargetMode) {
  case MemoryTargetMode::GlobalBuffer:
    return "global_buffer";
  case MemoryTargetMode::ConstantBuffer:
    return "constant_buffer";
  case MemoryTargetMode::Local:
    return "local";
  case MemoryTargetMode::Image:
    return "image";
  case MemoryTargetMode::HostBuffer:
    return "host_buffer";
  case MemoryTargetMode::HostImage:
    return "host_image";
  case MemoryTargetMode::ImageArray:
    return "image_array";
  }
}

mlir::LogicalResult mlir::sycl::parseMemoryTargetMode(
    mlir::AsmParser &Parser,
    mlir::FailureOr<mlir::sycl::MemoryTargetMode> &MemTargetMode) {
  mlir::StringRef Keyword;
  if (Parser.parseKeyword(&Keyword)) {
    return mlir::ParseResult::failure();
  }

  if (Keyword == "global_buffer") {
    MemTargetMode.emplace(mlir::sycl::MemoryTargetMode::GlobalBuffer);
  } else if (Keyword == "constant_buffer") {
    MemTargetMode.emplace(mlir::sycl::MemoryTargetMode::ConstantBuffer);
  } else if (Keyword == "local") {
    MemTargetMode.emplace(mlir::sycl::MemoryTargetMode::Local);
  } else if (Keyword == "image") {
    MemTargetMode.emplace(mlir::sycl::MemoryTargetMode::Image);
  } else if (Keyword == "host_buffer") {
    MemTargetMode.emplace(mlir::sycl::MemoryTargetMode::HostBuffer);
  } else if (Keyword == "host_image") {
    MemTargetMode.emplace(mlir::sycl::MemoryTargetMode::HostImage);
  } else if (Keyword == "image_array") {
    MemTargetMode.emplace(mlir::sycl::MemoryTargetMode::ImageArray);
  } else {
    return Parser.emitError(Parser.getCurrentLocation(),
                            "expected valid MemoryTargetMode keyword");
  }

  return mlir::ParseResult::success();
}

void mlir::sycl::printMemoryTargetMode(AsmPrinter &Printer,
                                       MemoryTargetMode MemTargetMode) {
  Printer << memoryTargetModeAsString(MemTargetMode);
}

std::string
mlir::sycl::accessAddressSpaceAsString(mlir::sycl::AccessAddrSpace AccAddress) {
  return std::to_string(static_cast<int>(AccAddress));
}

mlir::LogicalResult mlir::sycl::parseAccessAddrSpace(
    mlir::AsmParser &Parser,
    mlir::FailureOr<mlir::sycl::AccessAddrSpace> &AccAddress) {

  int AddSpaceInt;
  if (Parser.parseInteger<int>(AddSpaceInt)) {
    return mlir::ParseResult::failure();
  }
  // FIXME: The current implementation of AccessAddrSpace only works for SPIRV
  // target.
  assert(0 <= AddSpaceInt && AddSpaceInt <= 6 &&
         "Expecting address space value between 0 and 6 (inclusive)");

  AccAddress.emplace(static_cast<mlir::sycl::AccessAddrSpace>(AddSpaceInt));
  return mlir::ParseResult::success();
}

void mlir::sycl::printAccessAddrSpace(AsmPrinter &Printer,
                                      AccessAddrSpace AccAddress) {
  Printer << accessAddressSpaceAsString(AccAddress);
}

std::string
mlir::sycl::decoratedAccessAsString(mlir::sycl::DecoratedAccess DecAccess) {
  return std::to_string(static_cast<int>(DecAccess));
}

mlir::LogicalResult mlir::sycl::parseDecoratedAccess(
    mlir::AsmParser &Parser,
    mlir::FailureOr<mlir::sycl::DecoratedAccess> &DecAccess) {

  int DecAccessInt;
  if (Parser.parseInteger<int>(DecAccessInt)) {
    return mlir::ParseResult::failure();
  }

  assert(0 <= DecAccessInt && DecAccessInt <= 2 &&
         "Expecting Decorated Access value between 0 and 2 (inclusive)");

  DecAccess.emplace(static_cast<mlir::sycl::DecoratedAccess>(DecAccessInt));
  return mlir::ParseResult::success();
}

void mlir::sycl::printDecoratedAccess(AsmPrinter &Printer,
                                      DecoratedAccess DecAccess) {
  Printer << decoratedAccessAsString(DecAccess);
}

llvm::SmallVector<mlir::TypeID>
mlir::sycl::getDerivedTypes(mlir::TypeID TypeID) {
  if (TypeID == mlir::sycl::AccessorCommonType::getTypeID())
    return {mlir::sycl::AccessorType::getTypeID()};
  if (TypeID == mlir::sycl::ArrayType::getTypeID())
    return {mlir::sycl::IDType::getTypeID(),
            mlir::sycl::RangeType::getTypeID()};
  return {};
}

mlir::LogicalResult
mlir::sycl::VecType::verify(llvm::function_ref<InFlightDiagnostic()> EmitError,
                            mlir::Type DataT, int NumElements,
                            llvm::ArrayRef<mlir::Type>) {
  if (!(NumElements == 1 || NumElements == 2 || NumElements == 3 ||
        NumElements == 4 || NumElements == 8 || NumElements == 16)) {
    return EmitError() << "SYCL vector types can only hold 1, 2, 3, 4, 8 or 16 "
                          "elements. Got "
                       << NumElements << ".";
  }

  if (!DataT.isa<IntegerType, FloatType>()) {
    return EmitError()
           << "SYCL vector types can only hold basic scalar types. Got "
           << DataT << ".";
  }

  if (auto IntTy = DataT.dyn_cast<IntegerType>()) {
    const auto Width = IntTy.getWidth();
    if (!(Width == 1 || Width == 8 || Width == 16 || Width == 32 ||
          Width == 64)) {
      return EmitError() << "Integer SYCL vector element types can only be i1, "
                            "i8, i16, i32 or i64. Got "
                         << Width << ".";
    }
  } else {
    const auto Width = DataT.cast<FloatType>().getWidth();
    if (!(Width == 16 || Width == 32 || Width == 64)) {
      return EmitError()
             << "FP SYCL vector element types can only be f16, f32 or f64. Got "
             << Width << ".";
    }
  }

  return success();
}

unsigned mlir::sycl::getDimensions(mlir::Type Type) {
  if (auto MemRefTy = Type.dyn_cast<mlir::MemRefType>())
    Type = MemRefTy.getElementType();
  return TypeSwitch<mlir::Type, unsigned>(Type)
      .Case<AccessorType, GroupType, IDType, ItemType, NdItemType, NdRangeType,
            RangeType>([](auto Ty) { return Ty.getDimension(); });
}
