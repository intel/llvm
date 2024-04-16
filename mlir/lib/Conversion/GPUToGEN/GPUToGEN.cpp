//===- GPUToGEN.cpp - GPU to GEN Patterns ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert GPU dialect to GEN dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToGEN/GPUToGEN.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GEN/IR/GENDialect.h"
#include "mlir/Dialect/GEN/IR/GENOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUOPSTOGENOPS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

template <typename GPUOp, typename GENOp>
class GPUIndexOpToGENLowering : public OpConversionPattern<GPUOp> {
public:
  using OpConversionPattern<GPUOp>::OpConversionPattern;
  using OpAdaptor = typename GPUOp::Adaptor;

  LogicalResult
  matchAndRewrite(GPUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    auto dim = static_cast<std::uint32_t>(adaptor.getDimension());
    Value idxDim = rewriter.create<arith::ConstantIntOp>(op->getLoc(), dim, 32);
    rewriter.replaceOpWithNewOp<GENOp>(op, rewriter.getIndexType(), idxDim);
    return success();
  }
};

class GPUBarrierToGENLowering : public OpConversionPattern<gpu::BarrierOp> {
public:
  using OpConversionPattern<gpu::BarrierOp>::OpConversionPattern;
  using OpAdaptor = typename gpu::BarrierOp::Adaptor;

  LogicalResult match(gpu::BarrierOp op) const final { return success(); }

  void rewrite(gpu::BarrierOp op, OpAdaptor,
               ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<GEN::BarrierOp>(op);
  }
};

class GPUShuffleToGENLowering : public OpConversionPattern<gpu::ShuffleOp> {
public:
  using OpConversionPattern<gpu::ShuffleOp>::OpConversionPattern;
  using OpAdaptor = typename gpu::ShuffleOp::Adaptor;

  LogicalResult
  matchAndRewrite(gpu::ShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {

    auto gpuMode = adaptor.getMode();
    const auto genMode = [](gpu::ShuffleMode mode) {
      switch (mode) {
      case gpu::ShuffleMode::XOR:
        return GEN::ShflKind::XOR;
      case gpu::ShuffleMode::DOWN:
        return GEN::ShflKind::DOWN;
      case gpu::ShuffleMode::UP:
        return GEN::ShflKind::UP;
      case gpu::ShuffleMode::IDX:
        return GEN::ShflKind::IDX;
      }
      llvm_unreachable("expected a matching shuffle mode");
    }(gpuMode);

    // TODO unable to validate gpu width parameter, potential for producing
    // invalid code
    IntegerAttr widthAttr;
    if (!matchPattern(adaptor.getWidth(), m_Constant(&widthAttr))) {
      return rewriter.notifyMatchFailure(
          op, "shuffle width must be a constant value");
    }

    Value trueValue = rewriter.create<arith::ConstantOp>(
        op->getLoc(), rewriter.getBoolAttr(true));
    auto result = rewriter.create<GEN::SubGroupShuffleOp>(
        op->getLoc(), op->getResult(0).getType(), adaptor.getValue(),
        adaptor.getOffset(), genMode);

    rewriter.replaceOp(op, {result, trueValue});
    return success();
  }
};

void mlir::populateGPUToGENPatterns(RewritePatternSet &patterns) {
  patterns.add<GPUIndexOpToGENLowering<gpu::ThreadIdOp, GEN::LocalIdOp>,
               GPUIndexOpToGENLowering<gpu::BlockIdOp, GEN::WorkGroupIdOp>,
               GPUIndexOpToGENLowering<gpu::BlockDimOp, GEN::WorkGroupSizeOp>,
               GPUIndexOpToGENLowering<gpu::GridDimOp, GEN::NumWorkGroupsOp>,
               GPUBarrierToGENLowering, GPUShuffleToGENLowering>(
      patterns.getContext());
}

namespace {
struct ConvertGpuOpsToGENOpsPass
    : public impl::ConvertGpuOpsToGENOpsBase<ConvertGpuOpsToGENOpsPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalOp<arith::ConstantOp>();
    target.addLegalDialect<GEN::GENDialect>();
    // The ops of gpu dialect that can currently be mapped to GEN
    target.addIllegalOp<gpu::ThreadIdOp, gpu::BlockIdOp, gpu::BlockDimOp,
                        gpu::GridDimOp, gpu::BarrierOp, gpu::ShuffleOp>();

    mlir::RewritePatternSet patterns(&getContext());
    populateGPUToGENPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<>> mlir::createConvertGpuOpsToGENOps() {
  return std::make_unique<ConvertGpuOpsToGENOpsPass>();
}
