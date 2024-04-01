//===- GENToSPIRV.cpp - GEN to SPIRV dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GENToSPIRV/GENToSPIRV.h"

#include "mlir/Dialect/GEN/IR/GENDialect.h"
#include "mlir/Dialect/GEN/IR/GENOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGENTOSPIRV
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "gen-to-spirv-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ND-range Ops Lowerings
//===----------------------------------------------------------------------===//

/// Pattern to convert GEN3DNDRange operations to SPIR-V.
///
/// Convert:
/// ```mlir
/// %0 = gen.operation_name %dim
///  ```
/// To:
/// ```mlir
/// %__spirv_BuiltinName___addr = spirv.mlir.addressof
/// @__spirv_BuiltInBuiltinName : !spirv.ptr<vector<3xIndexType>, Input>
/// %__builtin_value = spirv.Load "Input" %__builtin__BuiltinName___addr :
/// vector<3xIndexType>
/// %0 = spirv.VectorExtractDynamic %__builtin_value[%dim] :
/// vector<3xIndexType>, i32
/// ```
/// With `BuiltinName` the name of a SPIR-V builtin, and `IndexType`, `i32` for
/// 32-bit targets and `i64` for 64-bit targets.
class GEN3DNDRangeLoweringBase : public ConversionPattern {
public:
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1 && "Expecting a single operand");
    // The builtin variable must be of type <3xi32> for 32-bit targets and
    // <3xi64> for 64-bit targets.
    Type builtinType =
        this->template getTypeConverter<SPIRVTypeConverter>()->getIndexType();
    constexpr StringLiteral spvBuiltinPrefix = "__spirv_BuiltIn";
    constexpr StringLiteral spvBuiltinSuffix = "";
    Value vector = spirv::getBuiltinVariableValue(
        op, builtin, builtinType, rewriter, spvBuiltinPrefix, spvBuiltinSuffix);
    rewriter.replaceOpWithNewOp<spirv::VectorExtractDynamicOp>(op, vector,
                                                               operands[0]);
    return success();
  }

protected:
  GEN3DNDRangeLoweringBase(spirv::BuiltIn builtin,
                           const TypeConverter &typeConverter, StringRef opName,
                           PatternBenefit benefit, MLIRContext *context)
      : ConversionPattern(typeConverter, opName, benefit, context),
        builtin(builtin) {}

private:
  spirv::BuiltIn builtin;
};

template <typename SourceOp, spirv::BuiltIn Builtin>
struct GEN3DNDRangeLowering : public GEN3DNDRangeLoweringBase {
  GEN3DNDRangeLowering(const TypeConverter &typeConverter, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : GEN3DNDRangeLoweringBase(Builtin, typeConverter,
                                 SourceOp::getOperationName(), benefit,
                                 context) {}
};

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::GEN::populateGENToSPIRVPatterns(SPIRVTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<
      GEN3DNDRangeLowering<GEN::LocalIdOp, spirv::BuiltIn::LocalInvocationId>,
      GEN3DNDRangeLowering<GEN::WorkGroupIdOp, spirv::BuiltIn::WorkgroupId>,
      GEN3DNDRangeLowering<GEN::WorkGroupSizeOp, spirv::BuiltIn::WorkgroupSize>,
      GEN3DNDRangeLowering<GEN::NumWorkGroupsOp,
                           spirv::BuiltIn::NumWorkgroups>>(
      converter, patterns.getContext());
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ConvertGENToSPIRVPass
    : public impl::ConvertGENToSPIRVBase<ConvertGENToSPIRVPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    spirv::TargetEnvAttr targetAttr = spirv::lookupTargetEnvOrDefault(op);
    std::unique_ptr<SPIRVConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);

    SPIRVTypeConverter typeConverter(targetAttr);

    // Fail hard when there are any remaining GEN ops.
    target->addIllegalDialect<GEN::GENDialect>();

    RewritePatternSet patterns(&getContext());
    GEN::populateGENToSPIRVPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(op, *target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<>> mlir::GEN::createConvertGENToSPIRVPass() {
  return std::make_unique<ConvertGENToSPIRVPass>();
}
