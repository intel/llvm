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

#include "spirv.hpp"

#ifndef spirv_internal_HPP
#define spirv_internal_HPP

namespace spv {
namespace internal {

enum InternalLinkageType {
  ILTPrev = LinkageTypeMax - 2,
  ILTInternal
};

enum InternalOp {
  IOpAssumeTrueINTEL = 5630,
  IOpExpectINTEL = 5631,
  IOpAliasDomainDeclINTEL = 5911,
  IOpAliasScopeDeclINTEL = 5912,
  IOpAliasScopeListDeclINTEL = 5913,
  IOpTypeTokenINTEL = 6113,
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
  IDecCallableFunctionINTEL = 6087
};

enum InternalCapability {
  ICapOptimizationHintsINTEL = 5629,
  ICapFPGADSPControlINTEL = 5908,
  ICapMemoryAccessAliasingINTEL = 5910,
  ICapFPGAInvocationPipeliningAttributesINTEL = 5916,
  ICapFastCompositeINTEL = 6093,
  ICapOptNoneINTEL = 6094,
  ICapTokenTypeINTEL = 6112
};

enum InternalFunctionControlMask { IFunctionControlOptNoneINTELMask = 0x10000 };

enum InternalMemoryAccessMask {
  IMemAccessAliasScopeINTELMask = 0x10000,
  IMemAccessNoAliasINTELMask = 0x20000
};

enum InternalExecutionMode { IExecModeFastCompositeKernelINTEL = 6088 };

constexpr LinkageType LinkageTypeInternal =
    static_cast<LinkageType>(ILTInternal);

constexpr Op OpForward = static_cast<Op>(IOpForward);
constexpr Op OpAssumeTrueINTEL = static_cast<Op>(IOpAssumeTrueINTEL);
constexpr Op OpExpectINTEL = static_cast<Op>(IOpExpectINTEL);
constexpr Op OpAliasDomainDeclINTEL = static_cast<Op>(IOpAliasDomainDeclINTEL);
constexpr Op OpAliasScopeDeclINTEL = static_cast<Op>(IOpAliasScopeDeclINTEL);
constexpr Op OpAliasScopeListDeclINTEL =
    static_cast<Op>(IOpAliasScopeListDeclINTEL);
constexpr Op OpTypeTokenINTEL = static_cast<Op>(IOpTypeTokenINTEL);

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

constexpr Capability CapabilityOptimizationHintsINTEL =
    static_cast<Capability>(ICapOptimizationHintsINTEL);
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

} // namespace internal
} // namespace spv

#endif // #ifndef spirv_internal_HPP
