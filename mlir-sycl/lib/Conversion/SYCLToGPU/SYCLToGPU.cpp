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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSYCLTOGPU
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DEF_CONVERTSYCLTOGPU
} // namespace mlir

using namespace mlir;
using namespace mlir::sycl;

namespace {
template <typename SYCLOpTy> struct gpu_counterpart_operation {};

template <> struct gpu_counterpart_operation<SYCLWorkGroupIDOp> {
  using type = gpu::BlockIdOp;
};

template <> struct gpu_counterpart_operation<SYCLWorkGroupSizeOp> {
  using type = gpu::BlockDimOp;
};

template <> struct gpu_counterpart_operation<SYCLLocalIDOp> {
  using type = gpu::ThreadIdOp;
};

template <> struct gpu_counterpart_operation<SYCLGlobalIDOp> {
  using type = gpu::GlobalIdOp;
};

template <> struct gpu_counterpart_operation<SYCLSubGroupIDOp> {
  using type = gpu::SubgroupIdOp;
};
template <> struct gpu_counterpart_operation<SYCLNumSubGroupsOp> {
  using type = gpu::NumSubgroupsOp;
};
template <> struct gpu_counterpart_operation<SYCLSubGroupSizeOp> {
  using type = gpu::SubgroupSizeOp;
};

template <typename SYCLOpTy>
using gpu_counterpart_operation_t =
    typename gpu_counterpart_operation<SYCLOpTy>::type;

/// Returns the result of creating an operation to get a reference to an element
/// of a sycl::id or sycl::range.
Value createGetOp(OpBuilder &builder, Location loc, Type underlyingArrTy,
                  Value res, Value index) {
  return TypeSwitch<Type, Value>(
             cast<MemRefType>(res.getType()).getElementType())
      .Case<IDType, RangeType>([&](auto arg) {
        // `this` type
        using ArgTy = decltype(arg);
        // Operation type depending on ArgTy
        using OpTy = std::conditional_t<std::is_same_v<ArgTy, IDType>,
                                        SYCLIDGetOp, SYCLRangeGetOp>;
        return builder.create<OpTy>(loc, underlyingArrTy, res, index);
      });
}

/// Creates an operation of with name \p opName from the GPU dialect using the
/// dimension \p dim as an attribute.
///
/// \return The result of such operation.
Value getDimension(OpBuilder &builder, Location loc, StringRef opName,
                   StringRef dimensionAttrName, int64_t dim) {
  assert(0 <= dim && dim < gpu::GPUDialect::getNumWorkgroupDimensions() &&
         "Invalid dimension value");
  // The GPU dialect lacks a proper way to obtain this name.
  auto *op = builder.create(
      loc, builder.getStringAttr(opName), ValueRange{}, builder.getIndexType(),
      builder.getNamedAttr(dimensionAttrName,
                           builder.getAttr<gpu::DimensionAttr>(
                               static_cast<gpu::Dimension>(dim))));
  assert(op->getNumResults() == 1 && "Invalid number of results");
  return op->getResult(0);
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
  const auto underlyingArrTy = MemRefType::get(dimensions, targetIndexTy);
  // Allocate
  const auto resTy = op->getResultTypes()[0];
  const Value res =
      rewriter.create<memref::AllocaOp>(loc, MemRefType::get(1, resTy));
  // Load
  const auto zero =
      static_cast<Value>(rewriter.create<arith::ConstantIndexOp>(loc, 0));
  // Initialize
  for (int64_t i = 0; i < dimensions; ++i) {
    const auto index = static_cast<Value>(
        rewriter.create<arith::ConstantIntOp>(loc, i, getIndexTy));
    const auto val = static_cast<Value>(rewriter.create<arith::IndexCastOp>(
        loc, targetIndexTy,
        getDimension(rewriter, loc, opName, dimensionAttrname, i)));
    const auto ptr = createGetOp(rewriter, loc, underlyingArrTy, res, index);
    rewriter.create<memref::StoreOp>(loc, val, ptr, zero);
  }
  rewriter.replaceOpWithNewOp<memref::LoadOp>(op, res, zero);
}

template <typename OpTy, typename GPUOpTy = gpu_counterpart_operation_t<OpTy>>
class GridOpPattern : public OpConversionPattern<OpTy> {
public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

protected:
  using GPUOpType = GPUOpTy;
};

/// Converts n-dimensional operations of type \tparam OpTy not being passed an
/// argument to operations of type \tparam GPUOpTy.
template <typename OpTy>
class NDGridOpPattern final : public GridOpPattern<OpTy> {
public:
  using GridOpPattern<OpTy>::GridOpPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor opAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr auto opName = GridOpPattern<OpTy>::GPUOpType::getOperationName();
    const auto dimensions = getDimensions(op->getResultTypes()[0]);
    const auto dimensionAttrName =
        GridOpPattern<OpTy>::GPUOpType::getDimensionAttrName(
            {opName, rewriter.getContext()});
    convertToFullObject(rewriter, opName, dimensionAttrName, op, dimensions);
    return success();
  }
};

/// Converts one-dimensional operations of type \tparam OpTy to operations of
/// type \tparam GPUOpTy.
///
/// Due to the different output type, as casting is needed before returning.
template <typename OpTy>
class SingleDimGridOpPattern final : public GridOpPattern<OpTy> {
public:
  using GridOpPattern<OpTy>::GridOpPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create the new operation
    const auto res = static_cast<Value>(
        rewriter.create<typename GridOpPattern<OpTy>::GPUOpType>(op->getLoc()));
    // And cast
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, op->getResultTypes()[0],
                                                    res);
    return success();
  }
};
} // namespace

void mlir::populateSYCLToGPUConversionPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.add<NDGridOpPattern<SYCLWorkGroupIDOp>,
               NDGridOpPattern<SYCLWorkGroupSizeOp>,
               NDGridOpPattern<SYCLLocalIDOp>, NDGridOpPattern<SYCLGlobalIDOp>,
               SingleDimGridOpPattern<SYCLSubGroupIDOp>,
               SingleDimGridOpPattern<SYCLNumSubGroupsOp>,
               SingleDimGridOpPattern<SYCLSubGroupSizeOp>>(context);
}

namespace {
/// A pass converting MLIR SYCL operations into LLVM dialect.
class ConvertSYCLToGPUPass
    : public impl::ConvertSYCLToGPUBase<ConvertSYCLToGPUPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertSYCLToGPUPass::runOnOperation() {
  auto &context = getContext();

  RewritePatternSet patterns(&context);
  ConversionTarget target(context);

  populateSYCLToGPUConversionPatterns(patterns);

  target.addLegalDialect<arith::ArithDialect, gpu::GPUDialect,
                         memref::MemRefDialect, SYCLDialect,
                         vector::VectorDialect>();

  target.addIllegalOp<SYCLWorkGroupIDOp, SYCLWorkGroupSizeOp, SYCLLocalIDOp,
                      SYCLGlobalIDOp, SYCLSubGroupIDOp, SYCLNumSubGroupsOp,
                      SYCLSubGroupSizeOp>();

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
