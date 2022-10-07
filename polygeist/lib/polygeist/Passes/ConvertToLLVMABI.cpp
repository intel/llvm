//===- ConvertToLLVMABI.cpp - Convert to LLVM Pointer ABI -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass currently only handles GPU functions.
// This pass converts all arguments with MemRef type to LLVM pointer type,
// and replace all uses of the original argument with a
// `polygeist.pointer2memref` of the new argument.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-to-llvm-abi"

using namespace mlir;

namespace {

class GPUFuncLowering : public OpRewritePattern<gpu::GPUFuncOp> {
public:
  using OpRewritePattern<gpu::GPUFuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::GPUFuncOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "ConvertToLLVMABIPass: GPUFuncLowering: "
                            << op.getName() << "\n");

    // Notify MLIR we're updating the function in place
    rewriter.startRootUpdate(op);

    FunctionType funcTy = op.getFunctionType();
    TypeConverter::SignatureConversion conversion(funcTy.getNumInputs());

    rewriter.setInsertionPointToStart(&op.getBody().front());
    Region &funcBody = op.getBody();
    SmallVector<BlockArgument, 8> Args(funcBody.getArguments().begin(),
                                       funcBody.getArguments().end());
    for (const auto &en : llvm::enumerate(Args)) {
      Value arg = en.value();
      auto MT = arg.getType().dyn_cast<MemRefType>();
      if (!MT) {
        conversion.addInputs(en.index(), {arg.getType()});
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "  Replace argument " << en.index() << ": \""
                              << arg << "\"");
      Type PT = LLVM::LLVMPointerType::get(MT.getElementType(),
                                           MT.getMemorySpaceAsInt());
      Value newArg = funcBody.insertArgument(en.index(), PT, arg.getLoc());
      LLVM_DEBUG(llvm::dbgs() << " -> \"" << newArg << "\"\n");
      conversion.addInputs(en.index(), {PT});

      auto Ptr2Memref = rewriter.create<polygeist::Pointer2MemrefOp>(
          arg.getLoc(), MT, newArg);

      arg.replaceAllUsesWith(Ptr2Memref);
      funcBody.eraseArgument(en.index() + 1);
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

class GPUReturnLowering : public OpRewritePattern<gpu::ReturnOp> {
public:
  using OpRewritePattern<gpu::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(gpu::ReturnOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "ConvertToLLVMABIPass: GPUReturnLowering: " << op << "\n");

    // Notify MLIR we're updating the function in place
    rewriter.startRootUpdate(op);

    bool Changed = false;
    for (const auto &en : llvm::enumerate(op.getOperands())) {
      auto MT = en.value().getType().dyn_cast<MemRefType>();
      if (!MT)
        continue;

      Type PT = LLVM::LLVMPointerType::get(MT.getElementType(),
                                           MT.getMemorySpaceAsInt());
      auto memref2Ptr = rewriter.create<polygeist::Memref2PointerOp>(
          en.value().getLoc(), PT, en.value());
      op.setOperand(en.index(), memref2Ptr);
      Changed = true;
    }
    LLVM_DEBUG({
      if (Changed)
        llvm::dbgs() << "  New ReturnOp: " << op << "\n";
      else
        llvm::dbgs() << " unchanged\n";
    });

    // Notify MLIR in place updates are done
    rewriter.finalizeRootUpdate(op);

    return Changed ? success() : failure();
  }
};

struct ConvertToLLVMABIPass final
    : public mlir::polygeist::ConvertToLLVMABIBase<ConvertToLLVMABIPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<GPUFuncLowering>(&getContext());
    patterns.add<GPUReturnLowering>(&getContext());

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>([](gpu::GPUFuncOp op) {
      FunctionType funcTy = op.getFunctionType();
      bool hasMemRef = llvm::any_of(
          funcTy.getResults(), [](Type ty) { return ty.isa<MemRefType>(); });
      hasMemRef |= llvm::any_of(funcTy.getInputs(),
                                [](Type ty) { return ty.isa<MemRefType>(); });
      return !hasMemRef;
    });
    target.addDynamicallyLegalOp<gpu::ReturnOp>([](gpu::ReturnOp op) {
      return llvm::none_of(op.getOperands(), [](Value v) {
        return v.getType().isa<MemRefType>();
      });
    });
    target.addLegalOp<polygeist::Memref2PointerOp>();
    target.addLegalOp<polygeist::Pointer2MemrefOp>();

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
