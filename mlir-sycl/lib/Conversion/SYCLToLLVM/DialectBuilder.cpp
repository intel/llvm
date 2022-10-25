//===- DialectBuilder.cpp - Dialect Builder -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements builders for several dialects operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToLLVM/DialectBuilder.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <cassert>

using namespace mlir;
using namespace mlir::sycl;

#define DEBUG_TYPE "dialect-builder"

//===----------------------------------------------------------------------===//
// DialectBuilder
//===----------------------------------------------------------------------===//

DialectBuilder::~DialectBuilder() {}

template <typename OP, typename... Types>
OP DialectBuilder::create(Types... args) const {
  return builder.create<OP>(loc, args...);
}

FlatSymbolRefAttr DialectBuilder::getOrInsertFuncDecl(StringRef funcName,
                                                      Type resultType,
                                                      ArrayRef<Type> argsTypes,
                                                      ModuleOp &module,
                                                      bool isVarArg) const {
  assert(!funcName.contains('@') && "funcName should not contain '@'");

  if (module.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {
    auto funcDecl = SymbolRefAttr::get(builder.getContext(), funcName);
    LLVM_DEBUG(llvm::dbgs() << "Found declaration: " << funcDecl << "\n");
    return funcDecl;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = LLVM::LLVMFunctionType::get(resultType, argsTypes, isVarArg);
  builder.create<LLVM::LLVMFuncOp>(module.getLoc(), funcName, funcType);
  auto funcDecl = SymbolRefAttr::get(builder.getContext(), funcName);
  LLVM_DEBUG(llvm::dbgs() << "Created declaration: " << funcDecl << "\n");

  return funcDecl;
}

BoolAttr DialectBuilder::getBoolAttr(bool val) const {
  return builder.getBoolAttr(val);
}

IntegerAttr DialectBuilder::getIntegerAttr(Type type, int64_t val) const {
  return builder.getIntegerAttr(type, val);
}

IntegerAttr DialectBuilder::getIntegerAttr(Type type, APInt val) const {
  return builder.getIntegerAttr(type, val);
}

FloatAttr DialectBuilder::getF16FloatAttr(float val) const {
  return builder.getF16FloatAttr(val);
}

FloatAttr DialectBuilder::getF32FloatAttr(float val) const {
  return builder.getF32FloatAttr(val);
}

FloatAttr DialectBuilder::getF64FloatAttr(double val) const {
  return builder.getF64FloatAttr(val);
}

ArrayAttr DialectBuilder::getI64ArrayAttr(ArrayRef<int64_t> vals) const {
  return builder.getI64ArrayAttr(vals);
}

//===----------------------------------------------------------------------===//
// FuncBuilder
//===----------------------------------------------------------------------===//

func::CallOp FuncBuilder::genCall(FlatSymbolRefAttr funcSym, TypeRange resTypes,
                                  ValueRange operands) const {
  assert(funcSym && "Expecting a valid function symbol");
  return create<func::CallOp>(resTypes, funcSym, operands);
}

func::CallOp FuncBuilder::genCall(StringRef funcName, TypeRange resTypes,
                                  ValueRange operands, ModuleOp &module) const {
  assert(!funcName.contains('@') && "funcName should not contain '@'");
  return create<func::CallOp>(funcName, resTypes, operands);
}

//===----------------------------------------------------------------------===//
// LLVMBuilder
//===----------------------------------------------------------------------===//

LLVM::AllocaOp LLVMBuilder::genAlloca(Type type, Value size,
                                      int64_t align) const {
  return create<LLVM::AllocaOp>(type, size, align);
}

LLVM::BitcastOp LLVMBuilder::genBitcast(Type type, Value val) const {
  return create<LLVM::BitcastOp>(type, val);
}

LLVM::ExtractValueOp
LLVMBuilder::genExtractValue(Type type, Value container,
                             ArrayRef<int64_t> position) const {
  return create<LLVM::ExtractValueOp>(type, container, position);
}

LLVM::CallOp LLVMBuilder::genCall(FlatSymbolRefAttr funcSym, TypeRange resTypes,
                                  ValueRange operands) const {
  assert(funcSym && "Expecting a valid function symbol");
  return create<LLVM::CallOp>(resTypes, funcSym, operands);
}

LLVM::CallOp LLVMBuilder::genCall(StringRef funcName, TypeRange resTypes,
                                  ValueRange operands, ModuleOp &module) const {
  assert(!funcName.contains('@') && "funcName should not contain '@'");
  return create<LLVM::CallOp>(resTypes, funcName, operands);
}

LLVM::ConstantOp LLVMBuilder::genConstant(Type type, double val) const {
  return llvm::TypeSwitch<Type, LLVM::ConstantOp>(type)
      .Case<IndexType>([&](IndexType type) {
        return create<LLVM::ConstantOp>(type,
                                        getIntegerAttr(type, (int64_t)val));
      })
      .Case<IntegerType>([&](IntegerType type) {
        bool isBool = (type.getWidth() == 1);
        return (isBool) ? create<LLVM::ConstantOp>(type, getBoolAttr(val != 0))
                        : create<LLVM::ConstantOp>(
                              type, getIntegerAttr(type, APInt(type.getWidth(),
                                                               (int64_t)val)));
      })
      .Case<Float16Type>([&](Type) {
        return create<LLVM::ConstantOp>(type, getF16FloatAttr(val));
      })
      .Case<Float32Type>([&](Type) {
        return create<LLVM::ConstantOp>(type, getF32FloatAttr(val));
      })
      .Case<Float64Type>([&](Type) {
        return create<LLVM::ConstantOp>(type, getF64FloatAttr(val));
      })
      .Default([&](Type) {
        llvm_unreachable("Missing support for type");
        return LLVM::ConstantOp();
      });
}

LLVM::SExtOp LLVMBuilder::genSignExtend(Type type, Value val) const {
  return create<LLVM::SExtOp>(type, val);
}

//===----------------------------------------------------------------------===//
// MemRefBuilder
//===----------------------------------------------------------------------===//

memref::AllocOp MemRefBuilder::genAlloc(MemRefType type) const {
  return create<memref::AllocOp>(type);
}

memref::AllocaOp MemRefBuilder::genAlloca(MemRefType type) const {
  return create<memref::AllocaOp>(type);
}

memref::CastOp MemRefBuilder::genCast(Value input,
                                      MemRefType outputType) const {
  return create<memref::CastOp>(outputType, input);
}

memref::DeallocOp MemRefBuilder::genDealloc(Value val) const {
  return create<memref::DeallocOp>(val);
}
