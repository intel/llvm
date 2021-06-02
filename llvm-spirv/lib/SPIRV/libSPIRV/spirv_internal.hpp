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
  IOpPrev = OpMax - 2,
  IOpForward
};

enum InternalDecoration {
  IDecAliasScopeINTEL = 5914,
  IDecNoAliasINTEL = 5915
};

enum InternalCapability {
  ICapOptimizationHintsINTEL = 5629,
  ICapMemoryAccessAliasingINTEL = 5910,
  ICapOptNoneINTEL = 6094
};

enum InternalFunctionControlMask { IFunctionControlOptNoneINTELMask = 0x10000 };

enum InternalMemoryAccessMask {
  IMemAccessAliasScopeINTELMask = 0x10000,
  IMemAccessNoAliasINTELMask = 0x20000
};

constexpr LinkageType LinkageTypeInternal =
    static_cast<LinkageType>(ILTInternal);

constexpr Op OpForward = static_cast<Op>(IOpForward);
constexpr Op OpAssumeTrueINTEL = static_cast<Op>(IOpAssumeTrueINTEL);
constexpr Op OpExpectINTEL = static_cast<Op>(IOpExpectINTEL);
constexpr Op OpAliasDomainDeclINTEL = static_cast<Op>(IOpAliasDomainDeclINTEL);
constexpr Op OpAliasScopeDeclINTEL = static_cast<Op>(IOpAliasScopeDeclINTEL);
constexpr Op OpAliasScopeListDeclINTEL =
    static_cast<Op>(IOpAliasScopeListDeclINTEL);

constexpr Decoration DecorationAliasScopeINTEL =
    static_cast<Decoration>(IDecAliasScopeINTEL );
constexpr Decoration DecorationNoAliasINTEL =
    static_cast<Decoration>(IDecNoAliasINTEL);

constexpr Capability CapabilityOptimizationHintsINTEL =
    static_cast<Capability>(ICapOptimizationHintsINTEL);
constexpr Capability CapabilityOptNoneINTEL =
    static_cast<Capability>(ICapOptNoneINTEL);
constexpr Capability CapabilityMemoryAccessAliasingINTEL =
    static_cast<Capability>(ICapMemoryAccessAliasingINTEL);

constexpr FunctionControlMask FunctionControlOptNoneINTELMask =
    static_cast<FunctionControlMask>(IFunctionControlOptNoneINTELMask);

constexpr MemoryAccessMask MemoryAccessAliasScopeINTELMask =
    static_cast<MemoryAccessMask>(IMemAccessAliasScopeINTELMask);
constexpr MemoryAccessMask MemoryAccessNoAliasINTELMask =
    static_cast<MemoryAccessMask>(IMemAccessNoAliasINTELMask);

} // namespace internal
} // namespace spv

#endif // #ifndef spirv_internal_HPP
