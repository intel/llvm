// Copyright (C) Codeplay Software Limited

//===--- SYCLOpsTypes.cpp -------------------------------------------------===//
//
// MLIR-SYCL is under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/IR/SYCLOpsTypes.h"
#include "mlir/IR/OpDefinition.h"

static mlir::ParseResult
parseMemoryAccessMode(mlir::DialectAsmParser &Parser,
                      mlir::sycl::MemoryAccessMode *MemAccessMode) {
  mlir::StringRef Keyword;
  if (Parser.parseKeyword(&Keyword)) {
    return mlir::ParseResult::failure();
  }

  if (Keyword == "read") {
    *MemAccessMode = mlir::sycl::MemoryAccessMode::Read;
  } else if (Keyword == "write") {
    *MemAccessMode = mlir::sycl::MemoryAccessMode::Write;
  } else if (Keyword == "read_write") {
    *MemAccessMode = mlir::sycl::MemoryAccessMode::ReadWrite;
  } else if (Keyword == "discard_write") {
    *MemAccessMode = mlir::sycl::MemoryAccessMode::DiscardWrite;
  } else if (Keyword == "discard_read_write") {
    *MemAccessMode = mlir::sycl::MemoryAccessMode::DiscardReadWrite;
  } else if (Keyword == "atomic") {
    *MemAccessMode = mlir::sycl::MemoryAccessMode::Atomic;
  } else {
    return Parser.emitError(Parser.getCurrentLocation(),
                            "expected valid MemoryAccessMode keyword");
  }

  return mlir::ParseResult::success();
}

static mlir::ParseResult
parseMemoryTargetMode(mlir::DialectAsmParser &Parser,
                      mlir::sycl::MemoryTargetMode *MemTargetMode) {
  mlir::StringRef Keyword;
  if (Parser.parseKeyword(&Keyword)) {
    return mlir::ParseResult::failure();
  }

  if (Keyword == "global_buffer") {
    *MemTargetMode = mlir::sycl::MemoryTargetMode::GlobalBuffer;
  } else if (Keyword == "constant_buffer") {
    *MemTargetMode = mlir::sycl::MemoryTargetMode::ConstantBuffer;
  } else if (Keyword == "local") {
    *MemTargetMode = mlir::sycl::MemoryTargetMode::Local;
  } else if (Keyword == "image") {
    *MemTargetMode = mlir::sycl::MemoryTargetMode::Image;
  } else if (Keyword == "host_buffer") {
    *MemTargetMode = mlir::sycl::MemoryTargetMode::HostBuffer;
  } else if (Keyword == "host_image") {
    *MemTargetMode = mlir::sycl::MemoryTargetMode::HostImage;
  } else if (Keyword == "image_array") {
    *MemTargetMode = mlir::sycl::MemoryTargetMode::ImageArray;
  } else {
    return Parser.emitError(Parser.getCurrentLocation(),
                            "expected valid MemoryTargetMode keyword");
  }

  return mlir::ParseResult::success();
}

////////////////////////////////////////////////////////////////////////////////
// IDType Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::IDType mlir::sycl::IDType::get(MLIRContext *Context,
                                           unsigned int Dimension) {
  return Base::get(Context, Dimension);
}

mlir::Type mlir::sycl::IDType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  unsigned int Dim;
  if (Parser.parseInteger<unsigned int>(Dim)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::IDType::get(Parser.getContext(), Dim);
}

unsigned int mlir::sycl::IDType::getDimension() const {
  return getImpl()->Dimension;
}

////////////////////////////////////////////////////////////////////////////////
// AccessorCommonType Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::AccessorCommonType
mlir::sycl::AccessorCommonType::get(MLIRContext *Context) {
  return Base::get(Context);
}

mlir::Type
mlir::sycl::AccessorCommonType::parseType(mlir::DialectAsmParser &Parser) {
  return mlir::sycl::AccessorCommonType::get(Parser.getContext());
}

////////////////////////////////////////////////////////////////////////////////
// AccessorType Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::AccessorType mlir::sycl::AccessorType::get(
    MLIRContext *Context, mlir::Type Type, unsigned int Dimension,
    MemoryAccessMode AccessMode, MemoryTargetMode TargetMode,
    llvm::SmallVector<mlir::Type, 4> Body) {
  return Base::get(Context, Type, Dimension, AccessMode, TargetMode, Body);
}

mlir::Type mlir::sycl::AccessorType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLSquare())) {
    return nullptr;
  }

  unsigned int Dim;
  if (Parser.parseInteger<unsigned int>(Dim)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  mlir::Type ET;
  if (Parser.parseType(ET)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  mlir::sycl::MemoryAccessMode MemAccessMode;
  if (parseMemoryAccessMode(Parser, &MemAccessMode)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  mlir::sycl::MemoryTargetMode MemTargetMode;
  if (parseMemoryTargetMode(Parser, &MemTargetMode)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseRSquare())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLParen())) {
    return nullptr;
  }

  mlir::SmallVector<Type, 4> Subtypes;
  do {
    mlir::Type Type;
    if (mlir::failed(Parser.parseType(Type))) {
      return nullptr;
    }
    Subtypes.push_back(Type);
  } while (succeeded(Parser.parseOptionalComma()));

  if (mlir::failed(Parser.parseRParen())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::AccessorType::get(Parser.getContext(), ET, Dim,
                                       MemAccessMode, MemTargetMode, Subtypes);
}

unsigned int mlir::sycl::AccessorType::getDimension() const {
  return getImpl()->Dimension;
}

mlir::Type mlir::sycl::AccessorType::getType() const { return getImpl()->Type; }

mlir::sycl::MemoryAccessMode mlir::sycl::AccessorType::getAccessMode() const {
  return getImpl()->AccessMode;
}

mlir::sycl::MemoryTargetMode mlir::sycl::AccessorType::getTargetMode() const {
  return getImpl()->TargetMode;
}

mlir::StringRef mlir::sycl::AccessorType::getAccessModeAsString() const {
  switch (getImpl()->AccessMode) {
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
  assert(false && "unreachable");
}

mlir::StringRef mlir::sycl::AccessorType::getTargetModeAsString() const {
  switch (getImpl()->TargetMode) {
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
  assert(false && "unreachable");
}

llvm::ArrayRef<mlir::Type> mlir::sycl::AccessorType::getBody() const {
  return getImpl()->Body;
}

////////////////////////////////////////////////////////////////////////////////
// Range Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::RangeType mlir::sycl::RangeType::get(MLIRContext *Context,
                                                 unsigned int Dimension) {
  return Base::get(Context, Dimension);
}

mlir::Type mlir::sycl::RangeType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  unsigned int Dim;
  if (Parser.parseInteger<unsigned int>(Dim)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::RangeType::get(Parser.getContext(), Dim);
}

unsigned int mlir::sycl::RangeType::getDimension() const {
  return getImpl()->Dimension;
}

////////////////////////////////////////////////////////////////////////////////
// NDRange Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::NdRangeType
mlir::sycl::NdRangeType::get(MLIRContext *Context, unsigned int Dimension,
                             llvm::SmallVector<mlir::Type, 4> Body) {
  return Base::get(Context, Dimension, Body);
}

mlir::Type mlir::sycl::NdRangeType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLSquare())) {
    return nullptr;
  }

  unsigned int Dim;
  if (Parser.parseInteger<unsigned int>(Dim)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseRSquare())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLParen())) {
    return nullptr;
  }

  mlir::SmallVector<Type, 4> Subtypes;
  do {
    mlir::Type Type;
    if (mlir::failed(Parser.parseType(Type))) {
      return nullptr;
    }
    Subtypes.push_back(Type);
  } while (succeeded(Parser.parseOptionalComma()));

  if (mlir::failed(Parser.parseRParen())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::NdRangeType::get(Parser.getContext(), Dim, Subtypes);
}

unsigned int mlir::sycl::NdRangeType::getDimension() const {
  return getImpl()->Dimension;
}

llvm::ArrayRef<mlir::Type> mlir::sycl::NdRangeType::getBody() const {
  return getImpl()->Body;
}

////////////////////////////////////////////////////////////////////////////////
// AccessorImplDeviceType Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::AccessorImplDeviceType
mlir::sycl::AccessorImplDeviceType::get(MLIRContext *Context,
                                        unsigned int Dimension,
                                        llvm::SmallVector<mlir::Type, 4> Body) {
  return Base::get(Context, Dimension, Body);
}

mlir::Type
mlir::sycl::AccessorImplDeviceType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLSquare())) {
    return nullptr;
  }

  unsigned int Dim;
  if (Parser.parseInteger<unsigned int>(Dim)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseRSquare())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLParen())) {
    return nullptr;
  }

  mlir::SmallVector<Type, 4> Subtypes;
  do {
    mlir::Type Type;
    if (mlir::failed(Parser.parseType(Type))) {
      return nullptr;
    }
    Subtypes.push_back(Type);
  } while (succeeded(Parser.parseOptionalComma()));

  if (mlir::failed(Parser.parseRParen())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::AccessorImplDeviceType::get(Parser.getContext(), Dim,
                                                 Subtypes);
}

unsigned int mlir::sycl::AccessorImplDeviceType::getDimension() const {
  return getImpl()->Dimension;
}

llvm::ArrayRef<mlir::Type> mlir::sycl::AccessorImplDeviceType::getBody() const {
  return getImpl()->Body;
}

////////////////////////////////////////////////////////////////////////////////
// Array Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::ArrayType
mlir::sycl::ArrayType::get(MLIRContext *Context, unsigned int Dimension,
                           llvm::SmallVector<mlir::Type, 4> Body) {
  return Base::get(Context, Dimension, Body);
}

mlir::Type mlir::sycl::ArrayType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLSquare())) {
    return nullptr;
  }

  unsigned int Dim;
  if (Parser.parseInteger<unsigned int>(Dim)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseRSquare())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLParen())) {
    return nullptr;
  }

  mlir::SmallVector<Type, 4> Subtypes;
  do {
    mlir::Type Type;
    if (mlir::failed(Parser.parseType(Type))) {
      return nullptr;
    }
    Subtypes.push_back(Type);
  } while (succeeded(Parser.parseOptionalComma()));

  if (mlir::failed(Parser.parseRParen())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::ArrayType::get(Parser.getContext(), Dim, Subtypes);
}

unsigned int mlir::sycl::ArrayType::getDimension() const {
  return getImpl()->Dimension;
}

llvm::ArrayRef<mlir::Type> mlir::sycl::ArrayType::getBody() const {
  return getImpl()->Body;
}

////////////////////////////////////////////////////////////////////////////////
// Item Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::ItemType
mlir::sycl::ItemType::get(MLIRContext *Context, unsigned int Dimension,
                          bool WithOffset,
                          llvm::SmallVector<mlir::Type, 4> Body) {
  return Base::get(Context, Dimension, WithOffset, Body);
}

mlir::Type mlir::sycl::ItemType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLSquare())) {
    return nullptr;
  }

  unsigned int Dim;
  if (Parser.parseInteger<unsigned int>(Dim)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  mlir::StringRef Keyword;
  bool Offset;
  if (Parser.parseKeyword(&Keyword)) {
    return nullptr;
  }
  if (Keyword == "true") {
    Offset = true;
  } else if (Keyword == "false") {
    Offset = false;
  } else {
    Parser.emitError(Parser.getCurrentLocation(), "expected boolean value");
    return nullptr;
  }

  if (mlir::failed(Parser.parseRSquare())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLParen())) {
    return nullptr;
  }

  mlir::SmallVector<Type, 4> Subtypes;
  do {
    mlir::Type Type;
    if (mlir::failed(Parser.parseType(Type))) {
      return nullptr;
    }
    Subtypes.push_back(Type);
  } while (succeeded(Parser.parseOptionalComma()));

  if (mlir::failed(Parser.parseRParen())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::ItemType::get(Parser.getContext(), Dim, Offset, Subtypes);
}

unsigned int mlir::sycl::ItemType::getDimension() const {
  return getImpl()->Dimension;
}

bool mlir::sycl::ItemType::getWithOffset() const {
  return getImpl()->WithOffset;
}

llvm::ArrayRef<mlir::Type> mlir::sycl::ItemType::getBody() const {
  return getImpl()->Body;
}

////////////////////////////////////////////////////////////////////////////////
// ItemBase Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::ItemBaseType
mlir::sycl::ItemBaseType::get(MLIRContext *Context, unsigned int Dimension,
                              bool WithOffset,
                              llvm::SmallVector<mlir::Type, 4> Body) {
  return Base::get(Context, Dimension, WithOffset, Body);
}

mlir::Type mlir::sycl::ItemBaseType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLSquare())) {
    return nullptr;
  }

  unsigned int Dim;
  if (Parser.parseInteger<unsigned int>(Dim)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  mlir::StringRef Keyword;
  bool Offset;
  if (Parser.parseKeyword(&Keyword)) {
    return nullptr;
  }
  if (Keyword == "true") {
    Offset = true;
  } else if (Keyword == "false") {
    Offset = false;
  } else {
    Parser.emitError(Parser.getCurrentLocation(), "expected boolean value");
    return nullptr;
  }

  if (mlir::failed(Parser.parseRSquare())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLParen())) {
    return nullptr;
  }

  mlir::SmallVector<Type, 4> Subtypes;
  do {
    mlir::Type Type;
    if (mlir::failed(Parser.parseType(Type))) {
      return nullptr;
    }
    Subtypes.push_back(Type);
  } while (succeeded(Parser.parseOptionalComma()));

  if (mlir::failed(Parser.parseRParen())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::ItemBaseType::get(Parser.getContext(), Dim, Offset,
                                       Subtypes);
}

unsigned int mlir::sycl::ItemBaseType::getDimension() const {
  return getImpl()->Dimension;
}

bool mlir::sycl::ItemBaseType::getWithOffset() const {
  return getImpl()->WithOffset;
}

llvm::ArrayRef<mlir::Type> mlir::sycl::ItemBaseType::getBody() const {
  return getImpl()->Body;
}

////////////////////////////////////////////////////////////////////////////////
// NdItem Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::NdItemType
mlir::sycl::NdItemType::get(MLIRContext *Context, unsigned int Dimension,
                            llvm::SmallVector<mlir::Type, 4> Body) {
  return Base::get(Context, Dimension, Body);
}

mlir::Type mlir::sycl::NdItemType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLSquare())) {
    return nullptr;
  }

  unsigned int Dim;
  if (Parser.parseInteger<unsigned int>(Dim)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseRSquare())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLParen())) {
    return nullptr;
  }

  mlir::SmallVector<Type, 4> Subtypes;
  do {
    mlir::Type Type;
    if (mlir::failed(Parser.parseType(Type))) {
      return nullptr;
    }
    Subtypes.push_back(Type);
  } while (succeeded(Parser.parseOptionalComma()));

  if (mlir::failed(Parser.parseRParen())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::NdItemType::get(Parser.getContext(), Dim, Subtypes);
}

unsigned int mlir::sycl::NdItemType::getDimension() const {
  return getImpl()->Dimension;
}

llvm::ArrayRef<mlir::Type> mlir::sycl::NdItemType::getBody() const {
  return getImpl()->Body;
}

////////////////////////////////////////////////////////////////////////////////
// GroupType Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::GroupType
mlir::sycl::GroupType::get(MLIRContext *Context, unsigned int Dimension,
                           llvm::SmallVector<mlir::Type, 4> Body) {
  return Base::get(Context, Dimension, Body);
}

mlir::Type mlir::sycl::GroupType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLSquare())) {
    return nullptr;
  }

  unsigned int Dim;
  if (Parser.parseInteger<unsigned int>(Dim)) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseRSquare())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseComma())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLParen())) {
    return nullptr;
  }

  mlir::SmallVector<Type, 4> Subtypes;
  do {
    mlir::Type Type;
    if (mlir::failed(Parser.parseType(Type))) {
      return nullptr;
    }
    Subtypes.push_back(Type);
  } while (succeeded(Parser.parseOptionalComma()));

  if (mlir::failed(Parser.parseRParen())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::GroupType::get(Parser.getContext(), Dim, Subtypes);
}

unsigned int mlir::sycl::GroupType::getDimension() const {
  return getImpl()->Dimension;
}

llvm::ArrayRef<mlir::Type> mlir::sycl::GroupType::getBody() const {
  return getImpl()->Body;
}

////////////////////////////////////////////////////////////////////////////////
// AtomicType Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::AtomicType
mlir::sycl::AtomicType::get(MLIRContext *Context,
                            llvm::SmallVector<mlir::Type, 4> Body) {
  return Base::get(Context, Body);
}

mlir::Type mlir::sycl::AtomicType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseLParen())) {
    return nullptr;
  }

  mlir::SmallVector<Type, 4> Subtypes;
  do {
    mlir::Type Type;
    if (mlir::failed(Parser.parseType(Type))) {
      return nullptr;
    }
    Subtypes.push_back(Type);
  } while (succeeded(Parser.parseOptionalComma()));

  if (mlir::failed(Parser.parseRParen())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::AtomicType::get(Parser.getContext(), Subtypes);
}

llvm::ArrayRef<mlir::Type> mlir::sycl::AtomicType::getBody() const {
  return getImpl()->Body;
}

////////////////////////////////////////////////////////////////////////////////
// MultiPtrType Operations
////////////////////////////////////////////////////////////////////////////////

mlir::sycl::MultiPtrType
mlir::sycl::MultiPtrType::get(MLIRContext *Context,
                              llvm::SmallVector<mlir::Type, 4> Body) {
  return Base::get(Context, Body);
}

mlir::Type mlir::sycl::MultiPtrType::parseType(mlir::DialectAsmParser &Parser) {
  if (mlir::failed(Parser.parseLess())) {
    return nullptr;
  }

  // parse the body
  if (mlir::failed(Parser.parseLParen())) {
    return nullptr;
  }

  mlir::SmallVector<Type, 4> Subtypes;
  do {
    mlir::Type Type;
    if (mlir::failed(Parser.parseType(Type))) {
      return nullptr;
    }
    Subtypes.push_back(Type);
  } while (succeeded(Parser.parseOptionalComma()));

  if (mlir::failed(Parser.parseRParen())) {
    return nullptr;
  }

  if (mlir::failed(Parser.parseGreater())) {
    return nullptr;
  }

  return mlir::sycl::MultiPtrType::get(Parser.getContext(), Subtypes);
}

llvm::ArrayRef<mlir::Type> mlir::sycl::MultiPtrType::getBody() const {
  return getImpl()->Body;
}
