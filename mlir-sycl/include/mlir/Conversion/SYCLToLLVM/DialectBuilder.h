//===- DialectBuilder.h - Dialect Builder ------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares builders to construct several dialect operations.
// Builders are derived from an abstract 'DialectBuilder' base class.
// Several builders can be added to this file.
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SYCLTOLLVM_DIALECTBUILDER_H
#define MLIR_CONVERSION_SYCLTOLLVM_DIALECTBUILDER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace sycl {

/// \class DialectBuilder
/// Abstract base class for all dialect builders.
class DialectBuilder {
public:
  DialectBuilder(OpBuilder &b, Location loc) : builder(b), loc(loc) {}
  virtual ~DialectBuilder() = 0;

  /// Inject a function declaration into the given module.
  FlatSymbolRefAttr getOrInsertFuncDecl(StringRef funcName, Type resultType,
                                        ArrayRef<Type> argsTypes,
                                        ModuleOp &module,
                                        bool isVarArg = false) const;

protected:
  /// Create a operation of type 'OP' given the argument list \p args, for the
  /// operation.
  template <typename OP, typename... Types> OP create(Types... args) const;

  BoolAttr getBoolAttr(bool val) const;
  IntegerAttr getIntegerAttr(Type type, int64_t val) const;
  IntegerAttr getIntegerAttr(Type type, APInt val) const;
  FloatAttr getF16FloatAttr(float val) const;
  FloatAttr getF32FloatAttr(float val) const;
  FloatAttr getF64FloatAttr(double val) const;
  ArrayAttr getI64ArrayAttr(ArrayRef<int64_t>) const;

protected:
  OpBuilder &builder;
  Location loc;
};

/// \class FuncBuilder
/// Construct operations in the Func dialect.
class FuncBuilder : public DialectBuilder {
public:
  FuncBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}

  func::CallOp genCall(FlatSymbolRefAttr funcSym, TypeRange resTypes,
                       ValueRange operands) const;
  func::CallOp genCall(StringRef funcName, TypeRange resTypes,
                       ValueRange operands, ModuleOp &module) const;
};

/// \class LLVMBuilder
/// Construct operations in the LLVM dialect.
class LLVMBuilder : public DialectBuilder {
public:
  LLVMBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}

  LLVM::AllocaOp genAlloca(Type type, Type elemType, Value size,
                           int64_t align) const;
  LLVM::BitcastOp genBitcast(Type type, Value val) const;
  LLVM::ExtractValueOp genExtractValue(Type type, Value container,
                                       ArrayRef<int64_t> pos) const;
  LLVM::CallOp genCall(FlatSymbolRefAttr funcSym, TypeRange resTypes,
                       ValueRange operands) const;
  LLVM::CallOp genCall(StringRef funcName, TypeRange resTypes,
                       ValueRange operands, ModuleOp &module) const;
  LLVM::ConstantOp genConstant(Type type, double val) const;
  LLVM::SExtOp genSignExtend(Type type, Value val) const;
};

/// \class MemRefBuilder
/// Construct operations in the MemRef dialect.
class MemRefBuilder : public DialectBuilder {
public:
  MemRefBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}

  memref::AllocOp genAlloc(MemRefType type) const;
  memref::AllocaOp genAlloca(MemRefType type) const;
  memref::CastOp genCast(Value input, MemRefType outputType) const;
  memref::DeallocOp genDealloc(Value val) const;
};

} // namespace sycl
} // namespace mlir

#endif // MLIR_CONVERSION_SYCLTOLLVM_DIALECTBUILDER_H
