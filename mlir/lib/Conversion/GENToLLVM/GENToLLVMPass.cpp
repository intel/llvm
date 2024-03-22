//===- GENToLLVMPass.cpp - MLIR GEN to LLVM dialect conversion ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GENToLLVM/GENToLLVMPass.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GEN/IR/GENDialect.h"
#include "mlir/Dialect/GEN/IR/GENOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGENTOLLVM
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static LLVM::CallOp createDeviceFunctionCall(
    ConversionPatternRewriter &rewriter, StringRef funcName, Type retType,
    ArrayRef<Type> argTypes, ArrayRef<Value> args, bool convergent = false) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  MLIRContext *context = rewriter.getContext();
  Location loc = UnknownLoc::get(context);
  auto convergentAttr =
      rewriter.getArrayAttr(StringAttr::get(context, "convergent"));

  auto getOrCreateFunction = [&](StringRef funcName) {
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(funcOp);

    auto funcType = LLVM::LLVMFunctionType::get(retType, argTypes);
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto func = rewriter.create<LLVM::LLVMFuncOp>(loc, funcName, funcType);
    func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    if (convergent)
      func.setPassthroughAttr(convergentAttr);

    return func;
  };

  LLVM::LLVMFuncOp funcOp = getOrCreateFunction(funcName);
  auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, args);
  if (convergent)
    callOp->setAttr("passthrough", convergentAttr);

  return callOp;
}

static LLVM::CallOp createSubGroupShuffle(ConversionPatternRewriter &rewriter,
                                          Value value, Value mask,
                                          GEN::ShflKind kind) {
  assert(isa<IntegerType>(mask.getType()) &&
         cast<IntegerType>(mask.getType()).isInteger(32) &&
         "Expecting mask type to be i32");

  std::string fnName = "";
  switch (kind) {
  case GEN::ShflKind::XOR:
    fnName = "_Z21sub_group_shuffle_xor";
    break;
  case GEN::ShflKind::UP:
    fnName = "_Z20sub_group_shuffle_up";
    break;
  case GEN::ShflKind::DOWN:
    fnName = "_Z22sub_group_shuffle_down";
    break;
  case GEN::ShflKind::IDX:
    fnName = "_Z17sub_group_shuffle";
    break;
  }

  TypeSwitch<Type>(value.getType())
      .Case<Float16Type>([&](auto) { fnName += "Dh"; })
      .Case<Float32Type>([&](auto) { fnName += "f"; })
      .Case<Float64Type>([&](auto) { fnName += "d"; })
      .Case<IntegerType>([&](auto ty) {
        switch (ty.getWidth()) {
        case 8:
          fnName += "c";
          break;
        case 16:
          fnName += "s";
          break;
        case 32:
          fnName += "i";
          break;
        case 64:
          fnName += "l";
          break;
        default:
          llvm_unreachable("unhandled integer type");
        }
      });

  fnName += "j";

  return createDeviceFunctionCall(rewriter, fnName, value.getType(),
                                  {value.getType(), mask.getType()},
                                  {value, mask}, true /*convergent*/);
}

static Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<LLVM::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

namespace {

struct FuncCallLowering {
protected:
  Value rewrite(Operation *op, StringRef funcName, unsigned dim,
                ConversionPatternRewriter &rewriter) const {
    auto retType = rewriter.getIntegerType(64);
    auto argType = rewriter.getIntegerType(32);
    auto arg = createConstantI32(op->getLoc(), rewriter, dim);
    LLVM::CallOp callOp =
        createDeviceFunctionCall(rewriter, funcName, retType, {argType}, {arg});

    Type resType = op->getResult(0).getType();
    if (resType == callOp.getResult().getType())
      return callOp.getResult();

    return rewriter.create<LLVM::TruncOp>(op->getLoc(), resType,
                                          callOp.getResult());
  }
};

//===----------------------------------------------------------------------===//
// ThreadId Ops Lowerings
//===----------------------------------------------------------------------===//

template <typename SourceOp>
struct GENThreadIdLowering : public ConvertOpToLLVMPattern<SourceOp>,
                             public FuncCallLowering {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value res;
    if (isa<GEN::ThreadIdXOp>(op))
      res = rewrite(op, "_Z12get_local_idj", 0, rewriter);
    else if (isa<GEN::ThreadIdYOp>(op))
      res = rewrite(op, "_Z12get_local_idj", 1, rewriter);
    else if (isa<GEN::ThreadIdZOp>(op))
      res = rewrite(op, "_Z12get_local_idj", 2, rewriter);
    else
      llvm_unreachable("Unexpected operation");

    rewriter.replaceOp(op, res);
    return success();
  }
};

using GENThreadIdXLowering = GENThreadIdLowering<GEN::ThreadIdXOp>;
using GENThreadIdYLowering = GENThreadIdLowering<GEN::ThreadIdYOp>;
using GENThreadIdZLowering = GENThreadIdLowering<GEN::ThreadIdZOp>;

//===----------------------------------------------------------------------===//
// BlockId Ops Lowerings
//===----------------------------------------------------------------------===//

template <typename SourceOp>
struct GENBlockIdLowering : public ConvertOpToLLVMPattern<SourceOp>,
                            public FuncCallLowering {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value res;
    if (isa<GEN::BlockIdXOp>(op))
      res = rewrite(op, "_Z12get_group_idj", 0, rewriter);
    else if (isa<GEN::BlockIdYOp>(op))
      res = rewrite(op, "_Z12get_group_idj", 1, rewriter);
    else if (isa<GEN::BlockIdZOp>(op))
      res = rewrite(op, "_Z12get_group_idj", 2, rewriter);
    else
      llvm_unreachable("Unexpected operation");

    rewriter.replaceOp(op, res);
    return success();
  }
};

using GENBlockIdXLowering = GENBlockIdLowering<GEN::BlockIdXOp>;
using GENBlockIdYLowering = GENBlockIdLowering<GEN::BlockIdYOp>;
using GENBlockIdZLowering = GENBlockIdLowering<GEN::BlockIdZOp>;

//===----------------------------------------------------------------------===//
// BlockDim Ops Lowerings
//===----------------------------------------------------------------------===//

template <typename SourceOp>
struct GENBlockDimLowering : public ConvertOpToLLVMPattern<SourceOp>,
                             public FuncCallLowering {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value res;
    if (isa<GEN::BlockDimXOp>(op))
      res = rewrite(op, "_Z14get_local_sizej", 0, rewriter);
    else if (isa<GEN::BlockDimYOp>(op))
      res = rewrite(op, "_Z14get_local_sizej", 1, rewriter);
    else if (isa<GEN::BlockDimZOp>(op))
      res = rewrite(op, "_Z14get_local_sizej", 2, rewriter);
    else
      llvm_unreachable("Unexpected operation");

    rewriter.replaceOp(op, res);
    return success();
  }
};

using GENBlockDimXLowering = GENBlockDimLowering<GEN::BlockDimXOp>;
using GENBlockDimYLowering = GENBlockDimLowering<GEN::BlockDimYOp>;
using GENBlockDimZLowering = GENBlockDimLowering<GEN::BlockDimZOp>;

//===----------------------------------------------------------------------===//
// GridDim Ops Lowerings
//===----------------------------------------------------------------------===//

template <typename SourceOp>
struct GENGridDimLowering : public ConvertOpToLLVMPattern<SourceOp>,
                            public FuncCallLowering {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename SourceOp::Adaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value res;
    if (isa<GEN::GridDimXOp>(op))
      res = rewrite(op, "_Z14get_num_groupsj", 0, rewriter);
    else if (isa<GEN::GridDimYOp>(op))
      res = rewrite(op, "_Z14get_num_groupsj", 1, rewriter);
    else if (isa<GEN::GridDimZOp>(op))
      res = rewrite(op, "_Z14get_num_groupsj", 2, rewriter);
    else
      llvm_unreachable("Unexpected operation");

    rewriter.replaceOp(op, res);
    return success();
  }
};

using GENGridDimXLowering = GENGridDimLowering<GEN::GridDimXOp>;
using GENGridDimYLowering = GENGridDimLowering<GEN::GridDimYOp>;
using GENGridDimZLowering = GENGridDimLowering<GEN::GridDimZOp>;

//===----------------------------------------------------------------------===//
// Synchronization Ops Lowerings
//===----------------------------------------------------------------------===//

struct GENBarrierLowering : public ConvertOpToLLVMPattern<GEN::BarrierOp> {
  using ConvertOpToLLVMPattern<GEN::BarrierOp>::ConvertOpToLLVMPattern;

  enum MemFence {
    Local = 0x01,
    Global = 0x02,
  };

  LogicalResult
  matchAndRewrite(GEN::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto retType = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto argType = rewriter.getIntegerType(32);
    auto arg = createConstantI32(op->getLoc(), rewriter, MemFence::Local);
    LLVM::CallOp callOp =
        createDeviceFunctionCall(rewriter, "_Z7barrierj", {retType}, {argType},
                                 {arg}, true /*convergent*/);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

struct SubGroupShuffleLowering
    : public ConvertOpToLLVMPattern<GEN::SubGroupShuffleOp> {
  using ConvertOpToLLVMPattern<GEN::SubGroupShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GEN::SubGroupShuffleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value val = op.getValue();
    Value mask = op.getMask();
    GEN::ShflKind kind = op.getKind();
    LLVM::CallOp callOp = createSubGroupShuffle(rewriter, val, mask, kind);
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertGENToLLVM final
    : public impl::ConvertGENToLLVMBase<ConvertGENToLLVM> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet pattern(context);
    LowerToLLVMOptions options(context);
    LLVMTypeConverter converter(context, options);
    LLVMConversionTarget target(*context);

    GEN::populateGENToLLVMConversionPatterns(converter, pattern);

    if (failed(
            applyPartialConversion(getOperation(), target, std::move(pattern))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population and Registration
//===----------------------------------------------------------------------===//

void mlir::GEN::populateGENToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<GENThreadIdXLowering, GENThreadIdYLowering, GENThreadIdZLowering,
               GENBlockIdXLowering, GENBlockIdYLowering, GENBlockIdZLowering,
               GENBlockDimXLowering, GENBlockDimYLowering, GENBlockDimZLowering,
               GENGridDimXLowering, GENGridDimYLowering, GENGridDimZLowering>(
      converter);
  // clang-format on
  patterns.add<GENBarrierLowering, SubGroupShuffleLowering>(converter);
}

std::unique_ptr<Pass> mlir::GEN::createConvertGENToLLVM() {
  return std::make_unique<ConvertGENToLLVM>();
}
