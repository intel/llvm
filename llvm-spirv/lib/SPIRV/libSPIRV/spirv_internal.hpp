// Copyright (c) 2020 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and/or associated documentation files (the "Materials"),
// to deal in the Materials without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Materials, and to permit persons to whom the
// Materials are furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Materials.
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM,OUT OF OR IN CONNECTION WITH THE MATERIALS OR THE USE OR OTHER DEALINGS
// IN THE MATERIALS.

// The header is for SPIR-V to LLVM IR internal definitions, that are not a part
// of Khronos SPIR-V specification.

#include "spirv/unified1/spirv.hpp"

#ifndef spirv_internal_HPP
#define spirv_internal_HPP

namespace spv {
namespace internal {

enum InternalLinkageType {
  ILTPrev = LinkageTypeMax - 2,
  ILTInternal
};

enum InternalOp {
  IOpAliasDomainDeclINTEL = 5911,
  IOpAliasScopeDeclINTEL = 5912,
  IOpAliasScopeListDeclINTEL = 5913,
  IOpTypeTokenINTEL = 6113,
  IOpConvertFToBF16INTEL = 6116,
  IOpConvertBF16ToFINTEL = 6117,
  IOpTypeJointMatrixINTEL = 6119,
  IOpJointMatrixLoadINTEL = 6120,
  IOpJointMatrixStoreINTEL = 6121,
  IOpJointMatrixMadINTEL = 6122,
  IOpArithmeticFenceINTEL = 6145,
  IOpJointMatrixWorkItemLengthINTEL = 6410,
  IOpPrev = OpMax - 2,
  IOpForward
};

enum InternalDecoration {
  IDecMathOpDSPModeINTEL = 5909,
  IDecAliasScopeINTEL = 5914,
  IDecNoAliasINTEL = 5915,
  IDecInitiationIntervalINTEL = 5917,
  IDecMaxConcurrencyINTEL = 5918,
  IDecPipelineEnableINTEL = 5919,
  IDecRuntimeAlignedINTEL = 5940,
  IDecCallableFunctionINTEL = 6087,
  IDecHostAccessINTEL = 6147,
  IDecInitModeINTEL = 6148,
  IDecImplementInCSRINTEL = 6149,
  IDecArgumentAttributeINTEL = 6409,
  IDecFuncParamKindINTEL = 9624,
  IDecFuncParamDescINTEL = 9625
};

enum InternalCapability {
  ICapFPGADSPControlINTEL = 5908,
  ICapMemoryAccessAliasingINTEL = 5910,
  ICapFPGAInvocationPipeliningAttributesINTEL = 5916,
  ICapRuntimeAlignedAttributeINTEL = 5939,
  ICapFastCompositeINTEL = 6093,
  ICapOptNoneINTEL = 6094,
  ICapTokenTypeINTEL = 6112,
  ICapBfloat16ConversionINTEL = 6115,
  ICapabilityJointMatrixINTEL = 6118,
  ICapabilityHWThreadQueryINTEL = 6134,
  ICapFPArithmeticFenceINTEL = 6144,
  ICapGlobalVariableDecorationsINTEL = 6146
};

enum InternalFunctionControlMask { IFunctionControlOptNoneINTELMask = 0x10000 };

enum InternalMemoryAccessMask {
  IMemAccessAliasScopeINTELMask = 0x10000,
  IMemAccessNoAliasINTELMask = 0x20000
};

enum InternalExecutionMode {
  IExecModeFastCompositeKernelINTEL = 6088,
  IExecModeStreamingInterfaceINTEL = 6154
};

enum InternalLoopControlMask { ILoopControlLoopCountINTELMask = 0x1000000 };

constexpr LinkageType LinkageTypeInternal =
    static_cast<LinkageType>(ILTInternal);

enum InternalJointMatrixLayout { RowMajor, ColumnMajor, PackedA, PackedB };

enum InternalBuiltIn {
  IBuiltInSubDeviceIDINTEL = 6135,
  IBuiltInGlobalHWThreadIDINTEL = 6136,
};

#define _SPIRV_OP(x, y) constexpr x x##y = static_cast<x>(I##x##y);
_SPIRV_OP(Capability, JointMatrixINTEL)
_SPIRV_OP(Op, TypeJointMatrixINTEL)
_SPIRV_OP(Op, JointMatrixLoadINTEL)
_SPIRV_OP(Op, JointMatrixStoreINTEL)
_SPIRV_OP(Op, JointMatrixMadINTEL)
_SPIRV_OP(Op, JointMatrixWorkItemLengthINTEL)
_SPIRV_OP(Capability, HWThreadQueryINTEL)
_SPIRV_OP(BuiltIn, SubDeviceIDINTEL)
_SPIRV_OP(BuiltIn, GlobalHWThreadIDINTEL)
#undef _SPIRV_OP

constexpr Op OpForward = static_cast<Op>(IOpForward);
constexpr Op OpAliasDomainDeclINTEL = static_cast<Op>(IOpAliasDomainDeclINTEL);
constexpr Op OpAliasScopeDeclINTEL = static_cast<Op>(IOpAliasScopeDeclINTEL);
constexpr Op OpAliasScopeListDeclINTEL =
    static_cast<Op>(IOpAliasScopeListDeclINTEL);
constexpr Op OpTypeTokenINTEL = static_cast<Op>(IOpTypeTokenINTEL);
constexpr Op OpArithmeticFenceINTEL = static_cast<Op>(IOpArithmeticFenceINTEL);
constexpr Op OpConvertFToBF16INTEL = static_cast<Op>(IOpConvertFToBF16INTEL);
constexpr Op OpConvertBF16ToFINTEL = static_cast<Op>(IOpConvertBF16ToFINTEL);

constexpr Decoration DecorationAliasScopeINTEL =
    static_cast<Decoration>(IDecAliasScopeINTEL );
constexpr Decoration DecorationNoAliasINTEL =
    static_cast<Decoration>(IDecNoAliasINTEL);
constexpr Decoration DecorationInitiationIntervalINTEL =
    static_cast<Decoration>(IDecInitiationIntervalINTEL);
constexpr Decoration DecorationMaxConcurrencyINTEL =
    static_cast<Decoration>(IDecMaxConcurrencyINTEL);
constexpr Decoration DecorationPipelineEnableINTEL =
    static_cast<Decoration>(IDecPipelineEnableINTEL);
constexpr Decoration DecorationCallableFunctionINTEL =
    static_cast<Decoration>(IDecCallableFunctionINTEL);
constexpr Decoration DecorationRuntimeAlignedINTEL =
    static_cast<Decoration>(IDecRuntimeAlignedINTEL);
constexpr Decoration DecorationHostAccessINTEL =
    static_cast<Decoration>(IDecHostAccessINTEL);
constexpr Decoration DecorationInitModeINTEL =
    static_cast<Decoration>(IDecInitModeINTEL);
constexpr Decoration DecorationImplementInCSRINTEL =
    static_cast<Decoration>(IDecImplementInCSRINTEL);
constexpr Decoration DecorationArgumentAttributeINTEL =
    static_cast<Decoration>(IDecArgumentAttributeINTEL);
constexpr Decoration DecorationFuncParamKindINTEL =
    static_cast<Decoration>(IDecFuncParamKindINTEL);
constexpr Decoration DecorationFuncParamDescINTEL =
    static_cast<Decoration>(IDecFuncParamDescINTEL);

constexpr Capability CapabilityFastCompositeINTEL =
    static_cast<Capability>(ICapFastCompositeINTEL);
constexpr Capability CapabilityOptNoneINTEL =
    static_cast<Capability>(ICapOptNoneINTEL);
constexpr Capability CapabilityFPGADSPControlINTEL =
    static_cast<Capability>(ICapFPGADSPControlINTEL);
constexpr Capability CapabilityMemoryAccessAliasingINTEL =
    static_cast<Capability>(ICapMemoryAccessAliasingINTEL);
constexpr Capability CapabilityFPGAInvocationPipeliningAttributesINTEL =
    static_cast<Capability>(ICapFPGAInvocationPipeliningAttributesINTEL);
constexpr Capability CapabilityTokenTypeINTEL =
    static_cast<Capability>(ICapTokenTypeINTEL);
constexpr Capability CapabilityRuntimeAlignedAttributeINTEL =
    static_cast<Capability>(ICapRuntimeAlignedAttributeINTEL);
constexpr Capability CapabilityFPArithmeticFenceINTEL =
    static_cast<Capability>(ICapFPArithmeticFenceINTEL);
constexpr Capability CapabilityBfloat16ConversionINTEL =
    static_cast<Capability>(ICapBfloat16ConversionINTEL);
constexpr Capability CapabilityGlobalVariableDecorationsINTEL =
    static_cast<Capability>(ICapGlobalVariableDecorationsINTEL);

constexpr FunctionControlMask FunctionControlOptNoneINTELMask =
    static_cast<FunctionControlMask>(IFunctionControlOptNoneINTELMask);

constexpr Decoration DecorationMathOpDSPModeINTEL =
    static_cast<Decoration>(IDecMathOpDSPModeINTEL);

constexpr MemoryAccessMask MemoryAccessAliasScopeINTELMask =
    static_cast<MemoryAccessMask>(IMemAccessAliasScopeINTELMask);
constexpr MemoryAccessMask MemoryAccessNoAliasINTELMask =
    static_cast<MemoryAccessMask>(IMemAccessNoAliasINTELMask);

constexpr ExecutionMode ExecutionModeFastCompositeKernelINTEL =
    static_cast<ExecutionMode>(IExecModeFastCompositeKernelINTEL);
constexpr ExecutionMode ExecutionModeStreamingInterfaceINTEL =
    static_cast<ExecutionMode>(IExecModeStreamingInterfaceINTEL);

constexpr LoopControlMask LoopControlLoopCountINTELMask =
    static_cast<LoopControlMask>(ILoopControlLoopCountINTELMask);

} // namespace internal
} // namespace spv

#endif // #ifndef spirv_internal_HPP
