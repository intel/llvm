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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

template <typename SYCLOpT, typename TargetOpT>
struct FloatUnaryOpPattern : public OpConversionPattern<SYCLOpT> {
  using OpConversionPattern<SYCLOpT>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SYCLOpT op, typename SYCLOpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOpT>(op, adaptor.getOperand());
    return success();
  }
};

} // anonymous namespace

void mlir::populateSYCLToMathConversionPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.insert<FloatUnaryOpPattern<SYCLSqrtOp, math::SqrtOp>>(context);
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
