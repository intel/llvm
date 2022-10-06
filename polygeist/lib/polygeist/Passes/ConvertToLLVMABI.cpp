//===- ConvertToLLVMABI.cpp - Convert to LLVM Pointer ABI -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass currently only handles SYCL Kernel functions.
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
    assert(op->getAttr("llvm.cconv") ==
               mlir::LLVM::CConvAttr::get(op.getContext(),
                                          LLVM::cconv::CConv::SPIR_KERNEL) &&
           "Expecting SYCL Kernel");
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
    for (auto &en : llvm::enumerate(Args)) {
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

    assert(funcTy.getNumResults() == 0 &&
           "Expecting SYCL kernel to return void");

    auto newFuncTy = FunctionType::get(
        op.getContext(), conversion.getConvertedTypes(), funcTy.getResults());
    op->setAttr(FunctionOpInterface::getTypeAttrName(),
                TypeAttr::get(newFuncTy));
    LLVM_DEBUG(llvm::dbgs() << "  New FunctionType: " << newFuncTy << "\n");

    // Notify MLIR in place updates are done
    rewriter.finalizeRootUpdate(op);

    return success();
  }
};

struct ConvertToLLVMABIPass final
    : public mlir::polygeist::ConvertToLLVMABIBase<ConvertToLLVMABIPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<GPUFuncLowering>(&getContext());

    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>([](gpu::GPUFuncOp op) {
      return llvm::none_of(op.getFunctionType().getInputs(),
                           [](Type ty) { return ty.isa<MemRefType>(); });
    });
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
