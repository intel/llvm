//===- SYCLMethodToSYCLCall.cpp - SYCLMethodOpInterface to SYCLCallOp -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate SYCLCallOp operations from
// SYCLMethodOpInterface instances.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SYCL/Transforms/Passes.h"

#include "PassDetail.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpInterfaces.h"
#include "mlir/Dialect/SYCL/IR/SYCLOpsDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "sycl-method-to-sycl-call"

using namespace mlir;
using namespace sycl;

static mlir::Value castToBaseType(PatternRewriter &Rewriter, mlir::Location Loc,
                                  mlir::Value Original,
                                  mlir::MemRefType BaseType) {
  const auto ThisType = Original.getType().cast<MemRefType>();
  const llvm::ArrayRef<int64_t> TargetShape = BaseType.getShape();
  const mlir::Type TargetElementType = BaseType.getElementType();
  const unsigned TargetMemSpace = BaseType.getMemorySpaceAsInt();

  assert(TargetShape == ThisType.getShape() &&
         "Shape should not change when casting to base class for a member "
         "function call.");
  assert(TargetMemSpace == ThisType.getMemorySpaceAsInt() &&
         "Memory space of the `this` argument should be preserved when "
         "creating a SYCLMethodOp instance.");

  // The element type will always change here.
  mlir::Value Cast = Rewriter.create<sycl::SYCLCastOp>(
      Loc,
      MemRefType::get(TargetShape, TargetElementType, {},
                      ThisType.getMemorySpaceAsInt()),
      Original);

  LLVM_DEBUG(llvm::dbgs() << "  Cast inserted: " << Cast << "\n");

  return Cast;
}

static LogicalResult convertMethod(SYCLMethodOpInterface method,
                                   PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "ConvertToLLVMABIPass: SYCLMethodOpLowering: ";
             method.dump(); llvm::dbgs() << "\n");

  SmallVector<mlir::Value> Args(method->getOperands());

  const auto BaseType = method.getBaseType().cast<MemRefType>();
  if (BaseType != Args[0].getType().cast<MemRefType>())
    Args[0] = castToBaseType(rewriter, method->getLoc(), Args[0], BaseType);

  const auto ResTyOrNone = [=]() -> llvm::Optional<mlir::Type> {
    const auto ResTys = method->getResultTypes();
    if (ResTys.empty())
      return llvm::None;
    assert(ResTys.size() == 1 && "Returning multiple values is not allowed in "
                                 "SYCLMethodOpInterface instances");
    return ResTys[0];
  };

  auto CallOp = rewriter.replaceOpWithNewOp<sycl::SYCLCallOp>(
      method, ResTyOrNone(), method.getTypeName(), method.getFunctionName(),
      method.getMangledFunctionName(), Args);

  LLVM_DEBUG(llvm::dbgs() << "  Converted to: " << CallOp << "\n");

  return success();
}

template <typename Op,
          typename = std::enable_if_t<mlir::sycl::isSYCLMethod<Op>::value>>
class SYCLMethodToSYCLCallPattern : public OpRewritePattern<Op> {
public:
  explicit SYCLMethodToSYCLCallPattern(MLIRContext *context)
      : OpRewritePattern<Op>(context, /*benefit*/ 1) {}

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final {
    return convertMethod(static_cast<SYCLMethodOpInterface>(op), rewriter);
  }
};

// If the operation is a method, add the SYCLMethod pattern for that
// operation.
template <typename T>
static typename std::enable_if_t<mlir::sycl::isSYCLMethod<T>::value>
addSYCLMethodPattern(RewritePatternSet &patterns) {
  patterns.add<SYCLMethodToSYCLCallPattern<T>>(patterns.getContext());
}

// If the operation is not a method, do nothing.
template <typename T>
static typename std::enable_if_t<!mlir::sycl::isSYCLMethod<T>::value>
addSYCLMethodPattern(RewritePatternSet &) {}

template <typename... Args>
static void addSYCLMethodPatterns(RewritePatternSet &patterns) {
  (void)std::initializer_list<int>{
      0, (addSYCLMethodPattern<Args>(patterns), 0)...};
}

static void addSYCLMethodPatterns(RewritePatternSet &patterns) {
  addSYCLMethodPatterns<
#define GET_OP_LIST
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
      >(patterns);
}

struct SYCLMethodToSYCLCall : SYCLMethodToSYCLCallBase<SYCLMethodToSYCLCall> {
  LogicalResult initialize(MLIRContext *context) final {
    RewritePatternSet owningPatterns(context);
    addSYCLMethodPatterns(owningPatterns);
    patterns = FrozenRewritePatternSet(std::move(owningPatterns));
    return success();
  }

  void runOnOperation() final {
    (void)applyPatternsAndFoldGreedily(getOperation(), patterns);
  }

  FrozenRewritePatternSet patterns;
};

/// Creates a pass that lowers SYCLMethodOpInterface instances to SYCLCallOps.
std::unique_ptr<Pass> mlir::sycl::createSYCLMethodToSYCLCallPass() {
  return std::make_unique<SYCLMethodToSYCLCall>();
}
