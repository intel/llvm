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

#include "Utils.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Dialect/SYCL/MethodUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sycl-method-to-sycl-call"

namespace mlir {
#define GEN_PASS_DEF_SYCLMETHODTOSYCLCALL
#include "mlir/Dialect/SYCL/Transforms/Passes.h.inc"
#undef GEN_PASS_DEF_SYCLMETHODTOSYCLCALL
} // namespace mlir

using namespace mlir;
using namespace sycl;

static LogicalResult convertMethod(SYCLMethodOpInterface method,
                                   PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "SYCLMethodToSYCLCallPass: SYCLMethodOpLowering: ";
             method.dump(); llvm::dbgs() << "\n");

  const auto Transformed = adaptArgumentsForSYCLCall(rewriter, method);

  const auto ResTyOrNone = [=]() -> llvm::Optional<mlir::Type> {
    const auto ResTys = method->getResultTypes();
    if (ResTys.empty())
      return std::nullopt;
    assert(ResTys.size() == 1 && "Returning multiple values is not allowed in "
                                 "SYCLMethodOpInterface instances");
    return ResTys[0];
  };

  llvm::Optional<llvm::StringRef> MangledFunctionName =
      method.getMangledFunctionName();
  if (!MangledFunctionName) {
    // If the optional MangledFunctionName attribute is not present, we try to
    // obtain the name of the function to call from the dialect's register.
    const auto *Dialect =
        method.getContext()->getLoadedDialect<sycl::SYCLDialect>();
    assert(Dialect && "SYCLDialect not loaded");

    auto Func = Dialect->lookupMethodDefinition(
        method.getFunctionName(),
        mlir::FunctionType::get(Dialect->getContext(),
                                ValueRange{Transformed}.getTypes(),
                                method->getResultTypes()));
    if (!Func) {
      return method->emitError(
                 "Could not obtain a valid definition for operation ")
             << method
             << " provide a definition using "
                "mlir::sycl::SYCLDialect::registerMethodDefinition() or using "
                "the MangledFunctionName field of this operation.";
    }

    SymbolTable Module(method->getParentWithTrait<OpTrait::SymbolTable>());
    if (auto *Op = Module.lookup(Func->getName())) {
      // If the function has already been cloned to this module, use that.
      Func = cast<func::FuncOp>(Op);
    } else {
      // If the function has not been inserted yet, do it now.
      Func = Func->clone();
      Module.insert(*Func);
    }
    MangledFunctionName = Func->getName();
  }

  auto CallOp = rewriter.replaceOpWithNewOp<sycl::SYCLCallOp>(
      method, ResTyOrNone(), method.getTypeName(), method.getFunctionName(),
      *MangledFunctionName, Transformed);

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

template <typename T>
static void addSYCLMethodPattern(RewritePatternSet &patterns) {
  if constexpr (mlir::sycl::isSYCLMethod<T>::value) {
    // If the operation is a method, add the SYCLMethod pattern for that
    // operation.
    patterns.add<SYCLMethodToSYCLCallPattern<T>>(patterns.getContext());
  }
}

template <typename... Args>
static void addSYCLMethodPatterns(RewritePatternSet &patterns) {
  (addSYCLMethodPattern<Args>(patterns), ...);
}

static void addSYCLMethodPatterns(RewritePatternSet &patterns) {
  addSYCLMethodPatterns<
#define GET_OP_LIST
#include "mlir/Dialect/SYCL/IR/SYCLOps.cpp.inc"
      >(patterns);
}

struct SYCLMethodToSYCLCall
    : impl::SYCLMethodToSYCLCallBase<SYCLMethodToSYCLCall> {
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
