//===- Utils.h - Utilities for Polygeist transformations ---------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_POLYGEIST_UTILS_UTILS_H_
#define MLIR_DIALECT_POLYGEIST_UTILS_UTILS_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir {
static inline mlir::scf::IfOp cloneWithoutResults(mlir::scf::IfOp op,
                                                  mlir::OpBuilder &rewriter,
                                                  mlir::IRMapping mapping = {},
                                                  mlir::TypeRange types = {}) {
  using namespace mlir;
  return rewriter.create<scf::IfOp>(
      op.getLoc(), types, mapping.lookupOrDefault(op.getCondition()), true);
}
static inline mlir::AffineIfOp cloneWithoutResults(mlir::AffineIfOp op,
                                                   mlir::OpBuilder &rewriter,
                                                   mlir::IRMapping mapping = {},
                                                   mlir::TypeRange types = {}) {
  using namespace mlir;
  SmallVector<mlir::Value> lower;
  for (auto o : op.getOperands())
    lower.push_back(mapping.lookupOrDefault(o));
  return rewriter.create<AffineIfOp>(op.getLoc(), types, op.getIntegerSet(),
                                     lower, true);
}

static inline mlir::scf::ForOp
cloneWithoutResults(mlir::scf::ForOp op, mlir::PatternRewriter &rewriter,
                    mlir::IRMapping mapping = {}) {
  using namespace mlir;
  return rewriter.create<scf::ForOp>(
      op.getLoc(), mapping.lookupOrDefault(op.getLowerBound()),
      mapping.lookupOrDefault(op.getUpperBound()),
      mapping.lookupOrDefault(op.getStep()));
}
static inline mlir::AffineForOp
cloneWithoutResults(mlir::AffineForOp op, mlir::PatternRewriter &rewriter,
                    mlir::IRMapping mapping = {}) {
  using namespace mlir;
  SmallVector<Value> lower;
  for (auto o : op.getLowerBoundOperands())
    lower.push_back(mapping.lookupOrDefault(o));
  SmallVector<Value> upper;
  for (auto o : op.getUpperBoundOperands())
    upper.push_back(mapping.lookupOrDefault(o));
  return rewriter.create<AffineForOp>(op.getLoc(), lower, op.getLowerBoundMap(),
                                      upper, op.getUpperBoundMap(),
                                      op.getStep());
}

static inline mlir::Block *getThenBlock(mlir::scf::IfOp op) {
  return op.thenBlock();
}
static inline mlir::Block *getThenBlock(mlir::AffineIfOp op) {
  return op.getThenBlock();
}
static inline mlir::Block *getElseBlock(mlir::scf::IfOp op) {
  return op.elseBlock();
}
static inline mlir::Block *getElseBlock(mlir::AffineIfOp op) {
  return op.getElseBlock();
}

static inline mlir::Region &getThenRegion(mlir::scf::IfOp op) {
  return op.getThenRegion();
}
static inline mlir::Region &getThenRegion(mlir::AffineIfOp op) {
  return op.getThenRegion();
}
static inline mlir::Region &getElseRegion(mlir::scf::IfOp op) {
  return op.getElseRegion();
}
static inline mlir::Region &getElseRegion(mlir::AffineIfOp op) {
  return op.getElseRegion();
}

static inline mlir::scf::YieldOp getThenYield(mlir::scf::IfOp op) {
  return op.thenYield();
}
static inline mlir::AffineYieldOp getThenYield(mlir::AffineIfOp op) {
  return llvm::cast<mlir::AffineYieldOp>(op.getThenBlock()->getTerminator());
}
static inline mlir::scf::YieldOp getElseYield(mlir::scf::IfOp op) {
  return op.elseYield();
}
static inline mlir::AffineYieldOp getElseYield(mlir::AffineIfOp op) {
  return llvm::cast<mlir::AffineYieldOp>(op.getElseBlock()->getTerminator());
}

static inline bool inBound(mlir::scf::IfOp op, mlir::Value v) {
  return op.getCondition() == v;
}
static inline bool inBound(mlir::AffineIfOp op, mlir::Value v) {
  return llvm::any_of(op.getOperands(), [&](mlir::Value e) { return e == v; });
}
static inline bool inBound(mlir::scf::ForOp op, mlir::Value v) {
  return op.getUpperBound() == v;
}
static inline bool inBound(mlir::AffineForOp op, mlir::Value v) {
  return llvm::any_of(op.getUpperBoundOperands(),
                      [&](mlir::Value e) { return e == v; });
}
static inline bool hasElse(mlir::scf::IfOp op) {
  return op.getElseRegion().getBlocks().size() > 0;
}
static inline bool hasElse(mlir::AffineIfOp op) {
  return op.getElseRegion().getBlocks().size() > 0;
}

/// States whether a MemRefType can be lowered to a bare pointer
///
/// Only ranked MemRefTypes with identity map and non-dynamic dimensions in the
/// range [1, rank) can be lowered to a bare pointer.
inline bool canBeLoweredToBarePtr(mlir::MemRefType memRefType) {
  if (!memRefType.getLayout().isIdentity() || !memRefType.hasRank())
    return false;
  const auto shape = memRefType.getShape();
  return std::none_of(shape.begin() + 1, shape.end(),
                      mlir::ShapedType::isDynamic);
}

inline LLVM::LLVMFuncOp getFreeFn(LLVMTypeConverter &typeConverter,
                                  ModuleOp module) {
  return typeConverter.getOptions().useGenericFunctions
             ? LLVM::lookupOrCreateGenericFreeFn(module)
             : LLVM::lookupOrCreateFreeFn(module);
}

inline LLVM::LLVMFuncOp getAllocFn(LLVMTypeConverter &typeConverter,
                                   ModuleOp module, Type indexType) {
  return typeConverter.getOptions().useGenericFunctions
             ? LLVM::lookupOrCreateGenericAllocFn(module, indexType)
             : LLVM::lookupOrCreateMallocFn(module, indexType);
}

} // namespace mlir

#endif // MLIR_DIALECT_POLYGEIST_UTILS_UTILS_H_
