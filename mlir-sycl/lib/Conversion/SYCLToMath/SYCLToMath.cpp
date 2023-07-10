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
#include "mlir/Dialect/SYCL/IR/SYCLTraits.h"
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
struct OneToOneMappingPattern : public OpConversionPattern<SYCLOpT> {
  using OpConversionPattern<SYCLOpT>::OpConversionPattern;

  LogicalResult match(SYCLOpT op) const override {
    Type type = op.getType();
    return success(type.isF32() || type.isF64());
  }

  void rewrite(SYCLOpT op, typename SYCLOpT::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    // `op` has a native MLIR type, hence we can just replace it with its
    // counterpart in the `math` dialect.
    rewriter.replaceOpWithNewOp<MathOpT>(op, adaptor.getOperands());
  }
};

template <typename SYCLOpT, typename MathOpT>
struct UnwrapOperandsWrapResultPattern : public OpConversionPattern<SYCLOpT> {
  using OpConversionPattern<SYCLOpT>::OpConversionPattern;

  LogicalResult match(SYCLOpT op) const override {
    return success(isa<HalfType, VecType>(op.getType()));
  }

  void rewrite(SYCLOpT op, typename SYCLOpT::Adaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    // `op` is using a SYCL-specific type, hence we need to unwrap the operands
    // and wrap the result.
    auto loc = op.getLoc();
    SmallVector<Value, 3> unwrappedOperands;
    for (auto operand : adaptor.getOperands())
      unwrappedOperands.push_back(rewriter.create<SYCLUnwrapOp>(
          loc, getBodyType(op.getType()), operand));
    auto math = rewriter.create<MathOpT>(loc, unwrappedOperands);
    rewriter.replaceOpWithNewOp<SYCLWrapOp>(op, op.getType(), math.getResult());
  }

  Type getBodyType(Type type) const {
    auto *ctx = type.getContext();
    if (isa<HalfType>(type))
      return Float16Type::get(ctx);

    auto vecTy = cast<VecType>(type);
    if (isa<HalfType>(vecTy.getDataType()))
      return VectorType::get({vecTy.getNumElements()}, Float16Type::get(ctx));

    return VectorType::get({vecTy.getNumElements()}, vecTy.getDataType());
  }
};

} // anonymous namespace

void mlir::populateSYCLToMathConversionPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
#define MAP_OP(from, to)                                                       \
  OneToOneMappingPattern<from, to>, UnwrapOperandsWrapResultPattern<from, to>
  // clang-format off
  patterns.insert<
      MAP_OP(SYCLCeilOp, math::CeilOp),
      MAP_OP(SYCLCopySignOp, math::CopySignOp),
      MAP_OP(SYCLCosOp, math::CosOp),
      MAP_OP(SYCLExpOp, math::ExpOp),
      MAP_OP(SYCLExp2Op, math::Exp2Op),
      MAP_OP(SYCLExpM1Op, math::ExpM1Op),
      MAP_OP(SYCLFabsOp, math::AbsFOp),
      MAP_OP(SYCLFloorOp, math::FloorOp),
      MAP_OP(SYCLFmaOp, math::FmaOp),
      MAP_OP(SYCLLogOp, math::LogOp),
      MAP_OP(SYCLLog10Op, math::Log10Op),
      MAP_OP(SYCLLog2Op, math::Log2Op),
      MAP_OP(SYCLPowOp, math::PowFOp),
      MAP_OP(SYCLRoundOp, math::RoundOp),
      MAP_OP(SYCLRsqrtOp, math::RsqrtOp),
      MAP_OP(SYCLSinOp, math::SinOp),
      MAP_OP(SYCLSqrtOp, math::SqrtOp),
      MAP_OP(SYCLTruncOp, math::TruncOp)>(context);
  // clang-format on
#undef MAP_MATH_OP
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
  target.addDynamicallyLegalDialect<sycl::SYCLDialect>(
      [](Operation *op) { return !op->hasTrait<sycl::SYCLMathFunc>(); });

  populateSYCLToMathConversionPatterns(patterns);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
