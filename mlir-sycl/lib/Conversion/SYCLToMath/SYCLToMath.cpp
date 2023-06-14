//===- SYCLToMath>.cpp - SYCL to Math Patterns ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert SYCL dialect to Math dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SYCLToMath/SYCLToMath.h"

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSYCLTOMATH
#include "mlir/Conversion/SYCLPasses.h.inc"
#undef GEN_PASS_DEF_CONVERTSYCLTOMATH
} // namespace mlir

using namespace mlir;
using namespace mlir::sycl;

namespace {

template <typename SYCLOpT, typename MathOpT>
struct FloatUnaryOpPattern : public OpConversionPattern<SYCLOpT> {
  using OpConversionPattern<SYCLOpT>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SYCLOpT op, typename SYCLOpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MathOpT>(op, adaptor.getX());
    return success();
  }
};

template <typename SYCLOpT, typename MathOpT>
struct FloatBinaryOpPattern : public OpConversionPattern<SYCLOpT> {
  using OpConversionPattern<SYCLOpT>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SYCLOpT op, typename SYCLOpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MathOpT>(op, adaptor.getX(), adaptor.getY());
    return success();
  }
};

template <typename SYCLOpT, typename MathOpT>
struct FloatTernaryOpPattern : public OpConversionPattern<SYCLOpT> {
  using OpConversionPattern<SYCLOpT>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SYCLOpT op, typename SYCLOpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<MathOpT>(op, adaptor.getA(), adaptor.getB(),
                                         adaptor.getC());
    return success();
  }
};

} // anonymous namespace

void mlir::populateSYCLToMathConversionPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.insert<FloatUnaryOpPattern<SYCLCeilOp, math::CeilOp>,
                  FloatBinaryOpPattern<SYCLCopySignOp, math::CopySignOp>,
                  FloatUnaryOpPattern<SYCLCosOp, math::CosOp>,
                  FloatUnaryOpPattern<SYCLExpOp, math::ExpOp>,
                  FloatUnaryOpPattern<SYCLExp2Op, math::Exp2Op>,
                  FloatUnaryOpPattern<SYCLExpM1Op, math::ExpM1Op>,
                  FloatUnaryOpPattern<SYCLFabsOp, math::AbsFOp>,
                  FloatUnaryOpPattern<SYCLFloorOp, math::FloorOp>,
                  FloatTernaryOpPattern<SYCLFmaOp, math::FmaOp>,
                  FloatUnaryOpPattern<SYCLLogOp, math::LogOp>,
                  FloatUnaryOpPattern<SYCLLog10Op, math::Log10Op>,
                  FloatUnaryOpPattern<SYCLLog2Op, math::Log2Op>,
                  FloatBinaryOpPattern<SYCLPowOp, math::PowFOp>,
                  FloatUnaryOpPattern<SYCLRoundOp, math::RoundOp>,
                  FloatUnaryOpPattern<SYCLRsqrtOp, math::RsqrtOp>,
                  FloatUnaryOpPattern<SYCLSinOp, math::SinOp>,
                  FloatUnaryOpPattern<SYCLSqrtOp, math::SqrtOp>,
                  FloatUnaryOpPattern<SYCLTruncOp, math::TruncOp>>(context);
}

namespace {
/// A pass converting MLIR SYCL operations into Math dialect.
class ConvertSYCLToMathPass
    : public impl::ConvertSYCLToMathBase<ConvertSYCLToMathPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertSYCLToMathPass::runOnOperation() {
  auto &context = getContext();

  RewritePatternSet patterns(&context);
  ConversionTarget target(context);

  target.addLegalDialect<math::MathDialect>();
  target.addDynamicallyLegalDialect<sycl::SYCLDialect>([](Operation *op) {
    return !op->getName().getStringRef().starts_with("sycl.math.");
  });

  populateSYCLToMathConversionPatterns(patterns);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
