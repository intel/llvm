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

//===----------------------------------------------------------------------===//
// ND-range Ops Lowerings
//===----------------------------------------------------------------------===//

class GEN3DNDRangeLoweringBase : public ConvertToLLVMPattern {
public:
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(op->getNumOperands() == 1 && "Expecting a single operand");
    Type resType = typeConverter->convertType(op->getResult(0).getType());
    LLVM::CallOp callOp = createDeviceFunctionCall(
        rewriter, builtinName, resType, rewriter.getI32Type(), operands[0]);
    rewriter.replaceOp(op, callOp);
    return success();
  }

protected:
  GEN3DNDRangeLoweringBase(StringRef builtinName, StringRef rootOpName,
                           const LLVMTypeConverter &typeConverter,
                           PatternBenefit benefit)
      : ConvertToLLVMPattern(rootOpName, &typeConverter.getContext(),
                             typeConverter, benefit),
        builtinName(builtinName) {}

private:
  StringRef builtinName;
};

template <typename SourceOp>
constexpr StringRef getBuiltinName();

template <>
StringRef getBuiltinName<GEN::LocalIdOp>() {
  return "_Z12get_local_idj";
}

template <>
StringRef getBuiltinName<GEN::WorkGroupIdOp>() {
  return "_Z12get_group_idj";
}

template <>
StringRef getBuiltinName<GEN::WorkGroupSizeOp>() {
  return "_Z14get_local_sizej";
}

template <>
StringRef getBuiltinName<GEN::NumWorkGroupsOp>() {
  return "_Z14get_num_groupsj";
}

template <typename SourceOp>
struct GEN3DNDRangeLowering : public GEN3DNDRangeLoweringBase {
  GEN3DNDRangeLowering(const LLVMTypeConverter &typeConverter,
                       PatternBenefit benefit = 1)
      : GEN3DNDRangeLoweringBase(getBuiltinName<SourceOp>(),
                                 SourceOp::getOperationName(), typeConverter,
                                 benefit) {}
};

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
  patterns.add<GEN3DNDRangeLowering<GEN::LocalIdOp>,
               GEN3DNDRangeLowering<GEN::WorkGroupIdOp>,
               GEN3DNDRangeLowering<GEN::WorkGroupSizeOp>,
               GEN3DNDRangeLowering<GEN::NumWorkGroupsOp>, GENBarrierLowering,
               SubGroupShuffleLowering>(converter);
}

std::unique_ptr<Pass> mlir::GEN::createConvertGENToLLVM() {
  return std::make_unique<ConvertGENToLLVM>();
}
