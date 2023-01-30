//===- SYCLToGPU.cpp - SYCL to GPU Patterns -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert SYCL dialect to GPU dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToGPU/SYCLToGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::sycl;

namespace {
/// Returns the result of creating an operation to get a reference to an element
/// of a sycl::id or sycl::range.
Value createGetOp(OpBuilder &builder, Location loc, Type dimMtTy, Value res,
                  Value index, ArrayAttr argumentTypes,
                  FlatSymbolRefAttr functionName) {
  return TypeSwitch<Type, Value>(res.getType())
      .Case<IDType, RangeType>([&](auto arg) {
        // `this` type
        using ArgTy = decltype(arg);
        // Operation type depending on ArgTy
        using OpTy = std::conditional_t<std::is_same_v<ArgTy, IDType>,
                                        SYCLIDGetOp, SYCLRangeGetOp>;
        return builder.create<OpTy>(
            loc, dimMtTy, res, index, argumentTypes, functionName, functionName,
            builder.getAttr<FlatSymbolRefAttr>(ArgTy::getMnemonic()));
      });
}

/// Creates an operation of with name \p opName from the GPU dialect using the
/// dimension \p i as an attribute.
///
/// \return The result of such operation.
Value getDimension(OpBuilder &builder, Location loc, StringRef opName,
                   StringRef dimensionAttrName, int64_t i) {
  assert(0 <= i && i < gpu::GPUDialect::getNumWorkgroupDimensions() &&
         "Invalid index value");
  // The GPU dialect lacks a proper way to obtain this name.
  auto *op = builder.create(
      loc, builder.getStringAttr(opName), ValueRange{}, builder.getIndexType(),
      builder.getNamedAttr(
          dimensionAttrName,
          builder.getAttr<gpu::DimensionAttr>(static_cast<gpu::Dimension>(i))));
  assert(op->getNumResults() == 1 && "Invalid number of results");
  return op->getResult(0);
}

/// Replace \p op with a single operation with name \p opName from the GPU
/// dialect thanks to the known constant index \p index.
void convertWithConstIndex(ConversionPatternRewriter &rewriter,
                           StringRef opName, StringRef dimensionAttrname,
                           Operation *op, arith::ConstantOp index) {
  const auto i = static_cast<arith::ConstantIntOp>(index).value();
  rewriter.replaceOp(
      op, getDimension(rewriter, op->getLoc(), opName, dimensionAttrname, i));
}

/// Replace \p op with a sequence of operations that:
/// 1. Allocate a new array of the result type in the stack
/// 2. Initialize the dimensions of the array with the expected results using
/// the operation with name \p opName from the GPU dialect
/// 3. Load the value in position \p index (assumed to be inbounds)
void convertWithVarIndex(ConversionPatternRewriter &rewriter, StringRef opName,
                         StringRef dimensionAttrname, Operation *op,
                         int64_t dimensions, Value index) {
  assert(dimensions <= gpu::GPUDialect::getNumWorkgroupDimensions() &&
         "Invalid number of dimensions");
  const auto loc = op->getLoc();
  const auto indexTy = rewriter.getIndexType();
  const auto alloca = static_cast<Value>(rewriter.create<memref::AllocaOp>(
      loc, MemRefType::get({dimensions}, indexTy)));
  for (int64_t i = 0; i < dimensions; ++i) {
    const auto c =
        static_cast<Value>(rewriter.create<arith::ConstantIndexOp>(loc, i));
    const auto val = getDimension(rewriter, loc, opName, dimensionAttrname, i);
    rewriter.create<memref::StoreOp>(loc, val, alloca, c);
  }
  index = rewriter.create<arith::IndexCastOp>(loc, indexTy, index);
  rewriter.replaceOpWithNewOp<memref::LoadOp>(op, alloca, index);
}

/// Replace \p op with a sequence of operations that:
/// 1. Allocate a new object of the result type in the stack
/// 2. Load the object
/// 3. Initialize the dimensions of the object with the expected results using
/// the operation with name \p opName from the GPU dialect
void convertToFullObject(ConversionPatternRewriter &rewriter, StringRef opName,
                         StringRef dimensionAttrname, Operation *op,
                         int64_t dimensions) {
  // This conversion is platform dependent
  assert(dimensions <= gpu::GPUDialect::getNumWorkgroupDimensions() &&
         "Invalid number of dimensions");
  const auto loc = op->getLoc();
  const auto targetIndexTy = rewriter.getIntegerType(64);
  const auto getIndexTy = rewriter.getIntegerType(32);
  const auto dimMtTy = MemRefType::get(dimensions, targetIndexTy, {}, 4);
  // Allocate
  const auto resTy = op->getResultTypes()[0];
  const auto alloca = static_cast<Value>(
      rewriter.create<memref::AllocaOp>(loc, MemRefType::get(1, resTy)));
  // Load
  const auto zero =
      static_cast<Value>(rewriter.create<arith::ConstantIndexOp>(loc, 0));
  const auto res = static_cast<Value>(
      rewriter.replaceOpWithNewOp<memref::LoadOp>(op, alloca, zero));
  const auto argumentTypes =
      rewriter.getTypeArrayAttr({MemRefType::get(1, resTy, {}, 4), getIndexTy});
  const auto functionName = rewriter.getAttr<FlatSymbolRefAttr>("operator[]");
  // Initialize
  for (int64_t i = 0; i < dimensions; ++i) {
    const auto index = static_cast<Value>(
        rewriter.create<arith::ConstantIntOp>(loc, i, getIndexTy));
    const auto val = static_cast<Value>(rewriter.create<arith::IndexCastOp>(
        loc, targetIndexTy,
        getDimension(rewriter, loc, opName, dimensionAttrname, i)));
    const auto ptr = createGetOp(rewriter, loc, dimMtTy, res, index,
                                 argumentTypes, functionName);
    rewriter.create<memref::StoreOp>(loc, val, ptr, zero);
  }
}

/// Converts n-dimensional operations to operations of name \p opName, from the
/// GPU dialect.
///
/// There are three possible cases here:
/// 1. No argument is passed (see convertToFullObject())
/// 2. A constant argument is passed (see converWithConstIndex())
/// 3. A non-constant argument is passed (see convertWithVarIndex())
void rewrite(StringRef opName, StringRef dimensionAttrName, Operation *op,
             ValueRange operands, ConversionPatternRewriter &rewriter) {
  switch (op->getNumOperands()) {
  case 0: {
    const auto dimensions = getDimensions(op->getResultTypes()[0]);
    convertToFullObject(rewriter, opName, dimensionAttrName, op, dimensions);
    break;
  }
  case 1: {
    const auto dimension = operands[0];
    if (const auto definingOp = dimension.getDefiningOp<arith::ConstantOp>()) {
      convertWithConstIndex(rewriter, opName, dimensionAttrName, op,
                            definingOp);
    } else {
      // TODO: Do not rely in a default number of dimensions; take into
      // account kernel dimensionality. This should be available in the parent
      // function.
      constexpr int64_t defaultDimensions{3};
      convertWithVarIndex(rewriter, opName, dimensionAttrName, op,
                          defaultDimensions, dimension);
    }
    break;
  }
  default:
    llvm_unreachable("Invalid cardinality");
  }
}

/// Converts n-dimensional operations of type \tparam OpTy to operations of type
/// \tparam GPUOpTy.
///
/// Work is offloaded to the rewrite function above.
template <typename OpTy, typename GPUOpTy>
class GridOpPattern final : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto opName = GPUOpTy::getOperationName();
    ::rewrite(opName,
              GPUOpTy::getDimensionAttrName({opName, rewriter.getContext()}),
              op, opAdaptor.getOperands(), rewriter);
    return success();
  }
};

/// Converts one-dimensional operations of type \tparam OpTy to operations of
/// type \tparam GPUOpTy.
///
/// Due to the different output type, as casting is needed before returning.
template <typename OpTy, typename GPUOpTy>
class SingleDimGridOpPattern final : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the new operation
    const auto res = static_cast<Value>(rewriter.create<GPUOpTy>(op->getLoc()));
    // And cast
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, op->getResultTypes()[0],
                                                    res);
    return success();
  }
};
} // namespace

void mlir::sycl::populateSYCLToGPUConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<GridOpPattern<SYCLWorkGroupIDOp, gpu::BlockIdOp>,
               GridOpPattern<SYCLNumWorkItemsOp, gpu::GridDimOp>,
               GridOpPattern<SYCLWorkGroupSizeOp, gpu::BlockDimOp>,
               GridOpPattern<SYCLLocalIDOp, gpu::ThreadIdOp>,
               GridOpPattern<SYCLGlobalIDOp, gpu::GlobalIdOp>,
               SingleDimGridOpPattern<SYCLSubGroupIDOp, gpu::SubgroupIdOp>,
               SingleDimGridOpPattern<SYCLNumSubGroupsOp, gpu::NumSubgroupsOp>,
               SingleDimGridOpPattern<SYCLSubGroupSizeOp, gpu::SubgroupSizeOp>>(
      patterns.getContext());
}
