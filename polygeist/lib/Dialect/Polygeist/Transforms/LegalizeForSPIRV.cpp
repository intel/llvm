//===- LegalizeForSPIRV.cpp - Prepare for translation to SPIRV IR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Polygeist/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace polygeist {
#define GEN_PASS_DEF_LLVMLEGALIZEFORSPIRV
#include "mlir/Dialect/Polygeist/Transforms/Passes.h.inc"
} // namespace polygeist
} // namespace mlir

using namespace mlir;

namespace {

/// Transform the following pattern:
///    %null = llvm.mlir.null : !llvm.ptr<i64>
///    %gep = llvm.getelementptr %null[1] : (!llvm.ptr<i64>) -> !llvm.ptr<i64>
///    %size = llvm.ptrtoint %gep : !llvm.ptr<i64> to i64
///    %ptr = llvm.alloca %size x i64 : (i64) -> !llvm.ptr<i64>
/// To:
///    %size = llvm.mlir.constant(1 : i64) : i64
///    %ptr = llvm.alloca %size x i64 : (i64) -> !llvm.ptr<i64>
///
class AllocaOpOfPtrToIntFolder final : public OpRewritePattern<LLVM::AllocaOp> {
public:
  using OpRewritePattern<LLVM::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AllocaOp allocaOp,
                                PatternRewriter &rewriter) const override {
    auto ptrToIntOp = allocaOp.getArraySize().getDefiningOp<LLVM::PtrToIntOp>();
    if (!ptrToIntOp)
      return failure();

    auto gepOp = ptrToIntOp.getArg().getDefiningOp<LLVM::GEPOp>();
    if (!gepOp)
      return failure();

    Value nullOp = gepOp.getBase().getDefiningOp<LLVM::ZeroOp>();
    if (!nullOp)
      return failure();

    if (!gepOp.getDynamicIndices().empty())
      return failure();

    llvm::ArrayRef<int32_t> constIndices = gepOp.getRawConstantIndices();
    if (constIndices.size() != 1)
      return failure();

    Value size = rewriter.create<LLVM::ConstantOp>(
        allocaOp.getLoc(), IntegerType::get(rewriter.getContext(), 32),
        rewriter.getIntegerAttr(rewriter.getI32Type(), constIndices.front()));

    if (std::optional<Type> optElemType = allocaOp.getElemType())
      rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
          allocaOp, allocaOp.getRes().getType(), *optElemType, size);
    else
      rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
          allocaOp, allocaOp.getRes().getType(), size);

    return success();
  }
};

struct LegalizeForSPIRVPass final
    : public mlir::polygeist::impl::LLVMLegalizeForSPIRVBase<
          LegalizeForSPIRVPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AllocaOpOfPtrToIntFolder>(patterns.getContext());
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::polygeist::createLegalizeForSPIRVPass() {
  return std::make_unique<LegalizeForSPIRVPass>();
}
