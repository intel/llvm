//===- ConvertToLLVMABI.cpp - Convert to LLVM Pointer ABI -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts all arguments with MemRef type to LLVM pointer type,
// and replace all uses of the original argument with a
// `polygeist.pointer2memref` of the new argument.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SYCL/IR/SYCLOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-to-llvm-abi"

using namespace mlir;

namespace {

template <typename T> class FuncLowering : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "ConvertToLLVMABIPass: FuncLowering: "
                            << op.getName() << "\n");

    // Notify MLIR we're updating the function in place
    rewriter.startRootUpdate(op);

    bool isDeclaration = op.getBody().empty();
    if (!isDeclaration) {
      rewriter.setInsertionPointToStart(&op.getBody().front());

      Region &funcBody = op.getBody();
      SmallVector<BlockArgument, 8> Args(funcBody.getArguments().begin(),
                                         funcBody.getArguments().end());
      for (const auto &en : llvm::enumerate(Args)) {
        Value arg = en.value();
        auto MT = arg.getType().dyn_cast<MemRefType>();
        if (!MT)
          continue;

        LLVM_DEBUG(llvm::dbgs() << "  Replace argument " << en.index() << ": \""
                                << arg << "\"");
        Type PT = LLVM::LLVMPointerType::get(MT.getElementType(),
                                             MT.getMemorySpaceAsInt());
        Value newArg = funcBody.insertArgument(en.index(), PT, arg.getLoc());
        LLVM_DEBUG(llvm::dbgs() << " -> \"" << newArg << "\"\n");

        auto Ptr2Memref = rewriter.create<polygeist::Pointer2MemrefOp>(
            arg.getLoc(), MT, newArg);

        arg.replaceAllUsesWith(Ptr2Memref);
        funcBody.eraseArgument(en.index() + 1);
      }
    }

    FunctionType funcTy = op.getFunctionType();
    TypeConverter::SignatureConversion conversion(funcTy.getNumInputs());
    for (const auto &en : llvm::enumerate(funcTy.getInputs())) {
      if (auto MT = en.value().dyn_cast<MemRefType>())
        conversion.addInputs(
            en.index(), {LLVM::LLVMPointerType::get(MT.getElementType(),
                                                    MT.getMemorySpaceAsInt())});
      else
        conversion.addInputs(en.index(), {en.value()});
    }
    SmallVector<Type, 1> convertedResultTys;
    for (Type ty : funcTy.getResults()) {
      if (auto MT = ty.dyn_cast<MemRefType>())
        convertedResultTys.push_back(LLVM::LLVMPointerType::get(
            MT.getElementType(), MT.getMemorySpaceAsInt()));
      else
        convertedResultTys.push_back(ty);
    }

    auto newFuncTy = FunctionType::get(
        op.getContext(), conversion.getConvertedTypes(), convertedResultTys);
    op->setAttr(FunctionOpInterface::getTypeAttrName(),
                TypeAttr::get(newFuncTy));
    LLVM_DEBUG(llvm::dbgs() << "  New FunctionType: " << newFuncTy << "\n");

    // Notify MLIR in place updates are done
    rewriter.finalizeRootUpdate(op);

    return success();
  }
};

static LogicalResult convertOperation(Operation &op,
                                      PatternRewriter &rewriter) {
  LLVM_DEBUG(llvm::dbgs() << "ConvertToLLVMABIPass: OpLowering: " << op
                          << "\n");

  // Notify MLIR we're updating the function in place
  rewriter.startRootUpdate(&op);

  bool changed = false;
  for (const auto &en : llvm::enumerate(op.getOperands())) {
    auto MT = en.value().getType().dyn_cast<MemRefType>();
    if (!MT)
      continue;

    Type PT = LLVM::LLVMPointerType::get(MT.getElementType(),
                                         MT.getMemorySpaceAsInt());
    rewriter.setInsertionPoint(&op);
    auto memref2Ptr = rewriter.create<polygeist::Memref2PointerOp>(
        en.value().getLoc(), PT, en.value());
    op.setOperand(en.index(), memref2Ptr);
    changed = true;
  }
  for (Value result : op.getOpResults()) {
    auto MT = result.getType().dyn_cast<MemRefType>();
    if (!MT)
      continue;

    result.setType(LLVM::LLVMPointerType::get(MT.getElementType(),
                                              MT.getMemorySpaceAsInt()));
    rewriter.setInsertionPointAfter(&op);
    auto Ptr2Memref = rewriter.create<polygeist::Pointer2MemrefOp>(
        result.getLoc(), MT, result);
    result.replaceAllUsesExcept(Ptr2Memref, Ptr2Memref);
    changed = true;
  }
  LLVM_DEBUG({
    if (changed)
      llvm::dbgs() << "  New Op: " << op << "\n";
    else
      llvm::dbgs() << "  unchanged\n";
  });

  // Notify MLIR in place updates are done
  rewriter.finalizeRootUpdate(&op);

  return changed ? success() : failure();
}

template <typename T> class OpLowering : public OpRewritePattern<T> {
public:
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op,
                                PatternRewriter &rewriter) const override {
    return convertOperation(*op.getOperation(), rewriter);
  }
};

class SYCLConstructorLowering
    : public OpRewritePattern<sycl::SYCLConstructorOp> {
public:
  using OpRewritePattern<sycl::SYCLConstructorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(sycl::SYCLConstructorOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "ConvertToLLVMABIPass: SYCLConstructorLowering: "
                            << op << "\n");

    auto funcCallOp =
        rewriter.create<func::CallOp>(op.getLoc(), op.getMangledFunctionName(),
                                      TypeRange(), op.getOperands());
    rewriter.replaceOp(op.getOperation(), funcCallOp.getResults());
    LLVM_DEBUG(llvm::dbgs() << "  Converted to: " << funcCallOp << "\n");

    return success();
  }
};

struct ConvertToLLVMABIPass final
    : public mlir::polygeist::ConvertToLLVMABIBase<ConvertToLLVMABIPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<FuncLowering<gpu::GPUFuncOp>>(&getContext());
    patterns.add<FuncLowering<func::FuncOp>>(&getContext());
    patterns.add<OpLowering<gpu::ReturnOp>>(&getContext());
    patterns.add<OpLowering<func::ReturnOp>>(&getContext());
    patterns.add<OpLowering<sycl::SYCLCallOp>>(&getContext());
    patterns.add<OpLowering<func::CallOp>>(&getContext());
    patterns.add<SYCLConstructorLowering>(&getContext());

    ConversionTarget target(getContext());

    auto checkFuncTy = [](FunctionType funcTy) {
      bool hasMemRef = llvm::any_of(
          funcTy.getResults(), [](Type ty) { return ty.isa<MemRefType>(); });
      hasMemRef |= llvm::any_of(funcTy.getInputs(),
                                [](Type ty) { return ty.isa<MemRefType>(); });
      return !hasMemRef;
    };
    auto checkOperation = [](Operation &op) {
      bool hasMemRef = llvm::any_of(
          op.getResultTypes(), [](Type ty) { return ty.isa<MemRefType>(); });
      hasMemRef |= llvm::any_of(op.getOperands(), [](Value v) {
        return v.getType().isa<MemRefType>();
      });
      return !hasMemRef;
    };
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>(
        [&checkFuncTy](gpu::GPUFuncOp op) {
          return checkFuncTy(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::FuncOp>([&checkFuncTy](func::FuncOp op) {
      return checkFuncTy(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<gpu::ReturnOp>(
        [&checkOperation](gpu::ReturnOp op) {
          return checkOperation(*op.getOperation());
        });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&checkOperation](func::ReturnOp op) {
          return checkOperation(*op.getOperation());
        });
    target.addDynamicallyLegalOp<sycl::SYCLCallOp>(
        [&checkOperation](sycl::SYCLCallOp op) {
          return checkOperation(*op.getOperation());
        });
    target.addDynamicallyLegalOp<func::CallOp>(
        [&checkOperation](func::CallOp op) {
          return checkOperation(*op.getOperation());
        });
    target.addLegalOp<polygeist::Memref2PointerOp>();
    target.addLegalOp<polygeist::Pointer2MemrefOp>();
    target.addLegalOp<sycl::SYCLCastOp>();
    target.addIllegalOp<sycl::SYCLConstructorOp>();

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();

    LLVM_DEBUG({
      llvm::dbgs() << "ConvertToLLVMABIPass: Module after:\n";
      m->dump();
      llvm::dbgs() << "\n";
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::polygeist::createConvertToLLVMABIPass() {
  return std::make_unique<ConvertToLLVMABIPass>();
}
