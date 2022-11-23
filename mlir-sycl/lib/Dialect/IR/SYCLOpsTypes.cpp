// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsTypes.cpp -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"

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
  default:
    llvm_unreachable("Invalid memory access mode");
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
  default:
    llvm_unreachable("Invalid memory target mode");
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

llvm::SmallVector<mlir::TypeID>
mlir::sycl::getDerivedTypes(mlir::TypeID TypeID) {
  if (TypeID == mlir::sycl::AccessorCommonType::getTypeID())
    return {mlir::sycl::AccessorType::getTypeID()};
  if (TypeID == mlir::sycl::ArrayType::getTypeID())
    return {mlir::sycl::IDType::getTypeID(),
            mlir::sycl::RangeType::getTypeID()};
  return {};
}
