//===- SPIRVIsValidEnum.h - SPIR-V isValid enums ----------------*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines SPIR-V isValid enums.
///
//===----------------------------------------------------------------------===//
// WARNING:
//
// This file has been generated using `tools/spirv-tool/gen_spirv.bash` and
// should not be modified manually. If the file needs to be updated, edit the
// script and any other source file instead, before re-generating this file.
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVISVALIDENUM_H
#define SPIRV_LIBSPIRV_SPIRVISVALIDENUM_H

#include "SPIRVEnum.h"
#include "spirv.hpp"

using namespace spv;

namespace SPIRV {

inline bool isValid(spv::SourceLanguage V) {
  switch (V) {
  case SourceLanguageUnknown:
  case SourceLanguageESSL:
  case SourceLanguageGLSL:
  case SourceLanguageOpenCL_C:
  case SourceLanguageOpenCL_CPP:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::ExecutionModel V) {
  switch (V) {
  case ExecutionModelVertex:
  case ExecutionModelTessellationControl:
  case ExecutionModelTessellationEvaluation:
  case ExecutionModelGeometry:
  case ExecutionModelFragment:
  case ExecutionModelGLCompute:
  case ExecutionModelKernel:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::AddressingModel V) {
  switch (V) {
  case AddressingModelLogical:
  case AddressingModelPhysical32:
  case AddressingModelPhysical64:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::MemoryModel V) {
  switch (V) {
  case MemoryModelSimple:
  case MemoryModelGLSL450:
  case MemoryModelOpenCL:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::ExecutionMode V) {
  switch (V) {
  case ExecutionModeInvocations:
  case ExecutionModeSpacingEqual:
  case ExecutionModeSpacingFractionalEven:
  case ExecutionModeSpacingFractionalOdd:
  case ExecutionModeVertexOrderCw:
  case ExecutionModeVertexOrderCcw:
  case ExecutionModePixelCenterInteger:
  case ExecutionModeOriginUpperLeft:
  case ExecutionModeOriginLowerLeft:
  case ExecutionModeEarlyFragmentTests:
  case ExecutionModePointMode:
  case ExecutionModeXfb:
  case ExecutionModeDepthReplacing:
  case ExecutionModeDepthGreater:
  case ExecutionModeDepthLess:
  case ExecutionModeDepthUnchanged:
  case ExecutionModeLocalSize:
  case ExecutionModeLocalSizeHint:
  case ExecutionModeInputPoints:
  case ExecutionModeInputLines:
  case ExecutionModeInputLinesAdjacency:
  case ExecutionModeTriangles:
  case ExecutionModeInputTrianglesAdjacency:
  case ExecutionModeQuads:
  case ExecutionModeIsolines:
  case ExecutionModeOutputVertices:
  case ExecutionModeOutputPoints:
  case ExecutionModeOutputLineStrip:
  case ExecutionModeOutputTriangleStrip:
  case ExecutionModeVecTypeHint:
  case ExecutionModeContractionOff:
  case ExecutionModeInitializer:
  case ExecutionModeFinalizer:
  case ExecutionModeSubgroupSize:
  case ExecutionModeSubgroupsPerWorkgroup:
  case ExecutionModeMaxWorkgroupSizeINTEL:
  case ExecutionModeNoGlobalOffsetINTEL:
  case ExecutionModeMaxWorkDimINTEL:
  case ExecutionModeNumSIMDWorkitemsINTEL:
  case ExecutionModeDenormPreserve:
  case ExecutionModeDenormFlushToZero:
  case ExecutionModeSignedZeroInfNanPreserve:
  case ExecutionModeRoundingModeRTE:
  case ExecutionModeRoundingModeRTZ:
  case ExecutionModeRoundingModeRTPINTEL:
  case ExecutionModeRoundingModeRTNINTEL:
  case ExecutionModeFloatingPointModeALTINTEL:
  case ExecutionModeFloatingPointModeIEEEINTEL:
  case ExecutionModeSharedLocalMemorySizeINTEL:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::StorageClass V) {
  switch (V) {
  case StorageClassUniformConstant:
  case StorageClassInput:
  case StorageClassUniform:
  case StorageClassOutput:
  case StorageClassWorkgroup:
  case StorageClassCrossWorkgroup:
  case StorageClassPrivate:
  case StorageClassFunction:
  case StorageClassGeneric:
  case StorageClassPushConstant:
  case StorageClassAtomicCounter:
  case StorageClassImage:
  case StorageClassDeviceOnlyINTEL:
  case StorageClassHostOnlyINTEL:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::Dim V) {
  switch (V) {
  case Dim1D:
  case Dim2D:
  case Dim3D:
  case DimCube:
  case DimRect:
  case DimBuffer:
  case DimSubpassData:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::SamplerAddressingMode V) {
  switch (V) {
  case SamplerAddressingModeNone:
  case SamplerAddressingModeClampToEdge:
  case SamplerAddressingModeClamp:
  case SamplerAddressingModeRepeat:
  case SamplerAddressingModeRepeatMirrored:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::SamplerFilterMode V) {
  switch (V) {
  case SamplerFilterModeNearest:
  case SamplerFilterModeLinear:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::ImageFormat V) {
  switch (V) {
  case ImageFormatUnknown:
  case ImageFormatRgba32f:
  case ImageFormatRgba16f:
  case ImageFormatR32f:
  case ImageFormatRgba8:
  case ImageFormatRgba8Snorm:
  case ImageFormatRg32f:
  case ImageFormatRg16f:
  case ImageFormatR11fG11fB10f:
  case ImageFormatR16f:
  case ImageFormatRgba16:
  case ImageFormatRgb10A2:
  case ImageFormatRg16:
  case ImageFormatRg8:
  case ImageFormatR16:
  case ImageFormatR8:
  case ImageFormatRgba16Snorm:
  case ImageFormatRg16Snorm:
  case ImageFormatRg8Snorm:
  case ImageFormatR16Snorm:
  case ImageFormatR8Snorm:
  case ImageFormatRgba32i:
  case ImageFormatRgba16i:
  case ImageFormatRgba8i:
  case ImageFormatR32i:
  case ImageFormatRg32i:
  case ImageFormatRg16i:
  case ImageFormatRg8i:
  case ImageFormatR16i:
  case ImageFormatR8i:
  case ImageFormatRgba32ui:
  case ImageFormatRgba16ui:
  case ImageFormatRgba8ui:
  case ImageFormatR32ui:
  case ImageFormatRgb10a2ui:
  case ImageFormatRg32ui:
  case ImageFormatRg16ui:
  case ImageFormatRg8ui:
  case ImageFormatR16ui:
  case ImageFormatR8ui:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::ImageChannelOrder V) {
  switch (V) {
  case ImageChannelOrderR:
  case ImageChannelOrderA:
  case ImageChannelOrderRG:
  case ImageChannelOrderRA:
  case ImageChannelOrderRGB:
  case ImageChannelOrderRGBA:
  case ImageChannelOrderBGRA:
  case ImageChannelOrderARGB:
  case ImageChannelOrderIntensity:
  case ImageChannelOrderLuminance:
  case ImageChannelOrderRx:
  case ImageChannelOrderRGx:
  case ImageChannelOrderRGBx:
  case ImageChannelOrderDepth:
  case ImageChannelOrderDepthStencil:
  case ImageChannelOrderABGR:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::ImageChannelDataType V) {
  switch (V) {
  case ImageChannelDataTypeSnormInt8:
  case ImageChannelDataTypeSnormInt16:
  case ImageChannelDataTypeUnormInt8:
  case ImageChannelDataTypeUnormInt16:
  case ImageChannelDataTypeUnormShort565:
  case ImageChannelDataTypeUnormShort555:
  case ImageChannelDataTypeUnormInt101010:
  case ImageChannelDataTypeSignedInt8:
  case ImageChannelDataTypeSignedInt16:
  case ImageChannelDataTypeSignedInt32:
  case ImageChannelDataTypeUnsignedInt8:
  case ImageChannelDataTypeUnsignedInt16:
  case ImageChannelDataTypeUnsignedInt32:
  case ImageChannelDataTypeHalfFloat:
  case ImageChannelDataTypeFloat:
  case ImageChannelDataTypeUnormInt24:
  case ImageChannelDataTypeUnormInt101010_2:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::FPRoundingMode V) {
  switch (V) {
  case FPRoundingModeRTE:
  case FPRoundingModeRTZ:
  case FPRoundingModeRTP:
  case FPRoundingModeRTN:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::LinkageType V) {
  switch (V) {
  case LinkageTypeExport:
  case LinkageTypeImport:
  case LinkageTypeInternal:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::AccessQualifier V) {
  switch (V) {
  case AccessQualifierReadOnly:
  case AccessQualifierWriteOnly:
  case AccessQualifierReadWrite:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::FunctionParameterAttribute V) {
  switch (V) {
  case FunctionParameterAttributeZext:
  case FunctionParameterAttributeSext:
  case FunctionParameterAttributeByVal:
  case FunctionParameterAttributeSret:
  case FunctionParameterAttributeNoAlias:
  case FunctionParameterAttributeNoCapture:
  case FunctionParameterAttributeNoWrite:
  case FunctionParameterAttributeNoReadWrite:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::Decoration V) {
  switch (V) {
  case DecorationRelaxedPrecision:
  case DecorationSpecId:
  case DecorationBlock:
  case DecorationBufferBlock:
  case DecorationRowMajor:
  case DecorationColMajor:
  case DecorationArrayStride:
  case DecorationMatrixStride:
  case DecorationGLSLShared:
  case DecorationGLSLPacked:
  case DecorationCPacked:
  case DecorationBuiltIn:
  case DecorationNoPerspective:
  case DecorationFlat:
  case DecorationPatch:
  case DecorationCentroid:
  case DecorationSample:
  case DecorationInvariant:
  case DecorationRestrict:
  case DecorationAliased:
  case DecorationVolatile:
  case DecorationConstant:
  case DecorationCoherent:
  case DecorationNonWritable:
  case DecorationNonReadable:
  case DecorationUniform:
  case DecorationSaturatedConversion:
  case DecorationStream:
  case DecorationLocation:
  case DecorationComponent:
  case DecorationIndex:
  case DecorationBinding:
  case DecorationDescriptorSet:
  case DecorationOffset:
  case DecorationXfbBuffer:
  case DecorationXfbStride:
  case DecorationFuncParamAttr:
  case DecorationFPRoundingMode:
  case DecorationFPFastMathMode:
  case DecorationLinkageAttributes:
  case DecorationNoContraction:
  case DecorationInputAttachmentIndex:
  case DecorationAlignment:
  case DecorationMaxByteOffset:
  case DecorationUserSemantic:
  case DecorationRegisterINTEL:
  case DecorationMemoryINTEL:
  case DecorationNumbanksINTEL:
  case DecorationBankwidthINTEL:
  case DecorationMaxPrivateCopiesINTEL:
  case DecorationSinglepumpINTEL:
  case DecorationDoublepumpINTEL:
  case DecorationBankBitsINTEL:
  case DecorationForcePow2DepthINTEL:
  case DecorationBurstCoalesceINTEL:
  case DecorationCacheSizeINTEL:
  case DecorationDontStaticallyCoalesceINTEL:
  case DecorationPrefetchINTEL:
  case DecorationReferencedIndirectlyINTEL:
  case DecorationVectorComputeFunctionINTEL:
  case DecorationStackCallINTEL:
  case DecorationFuncParamKindINTEL:
  case DecorationFuncParamDescINTEL:
  case DecorationVectorComputeVariableINTEL:
  case DecorationGlobalVariableOffsetINTEL:
  case DecorationFuncParamIOKind:
  case DecorationSIMTCallINTEL:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::BuiltIn V) {
  switch (V) {
  case BuiltInPosition:
  case BuiltInPointSize:
  case BuiltInClipDistance:
  case BuiltInCullDistance:
  case BuiltInVertexId:
  case BuiltInInstanceId:
  case BuiltInPrimitiveId:
  case BuiltInInvocationId:
  case BuiltInLayer:
  case BuiltInViewportIndex:
  case BuiltInTessLevelOuter:
  case BuiltInTessLevelInner:
  case BuiltInTessCoord:
  case BuiltInPatchVertices:
  case BuiltInFragCoord:
  case BuiltInPointCoord:
  case BuiltInFrontFacing:
  case BuiltInSampleId:
  case BuiltInSamplePosition:
  case BuiltInSampleMask:
  case BuiltInFragDepth:
  case BuiltInHelperInvocation:
  case BuiltInNumWorkgroups:
  case BuiltInWorkgroupSize:
  case BuiltInWorkgroupId:
  case BuiltInLocalInvocationId:
  case BuiltInGlobalInvocationId:
  case BuiltInLocalInvocationIndex:
  case BuiltInWorkDim:
  case BuiltInGlobalSize:
  case BuiltInEnqueuedWorkgroupSize:
  case BuiltInGlobalOffset:
  case BuiltInGlobalLinearId:
  case BuiltInSubgroupSize:
  case BuiltInSubgroupMaxSize:
  case BuiltInNumSubgroups:
  case BuiltInNumEnqueuedSubgroups:
  case BuiltInSubgroupId:
  case BuiltInSubgroupLocalInvocationId:
  case BuiltInSubgroupEqMask:
  case BuiltInSubgroupGeMask:
  case BuiltInSubgroupGtMask:
  case BuiltInSubgroupLeMask:
  case BuiltInSubgroupLtMask:
  case BuiltInVertexIndex:
  case BuiltInInstanceIndex:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::Scope V) {
  switch (V) {
  case ScopeCrossDevice:
  case ScopeDevice:
  case ScopeWorkgroup:
  case ScopeSubgroup:
  case ScopeInvocation:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::GroupOperation V) {
  switch (V) {
  case GroupOperationReduce:
  case GroupOperationInclusiveScan:
  case GroupOperationExclusiveScan:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::KernelEnqueueFlags V) {
  switch (V) {
  case KernelEnqueueFlagsNoWait:
  case KernelEnqueueFlagsWaitKernel:
  case KernelEnqueueFlagsWaitWorkGroup:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::Capability V) {
  switch (V) {
  case CapabilityMatrix:
  case CapabilityShader:
  case CapabilityGeometry:
  case CapabilityTessellation:
  case CapabilityAddresses:
  case CapabilityLinkage:
  case CapabilityKernel:
  case CapabilityVector16:
  case CapabilityFloat16Buffer:
  case CapabilityFloat16:
  case CapabilityFloat64:
  case CapabilityInt64:
  case CapabilityInt64Atomics:
  case CapabilityImageBasic:
  case CapabilityImageReadWrite:
  case CapabilityImageMipmap:
  case CapabilityPipes:
  case CapabilityGroups:
  case CapabilityDeviceEnqueue:
  case CapabilityLiteralSampler:
  case CapabilityAtomicStorage:
  case CapabilityInt16:
  case CapabilityTessellationPointSize:
  case CapabilityGeometryPointSize:
  case CapabilityImageGatherExtended:
  case CapabilityStorageImageMultisample:
  case CapabilityUniformBufferArrayDynamicIndexing:
  case CapabilitySampledImageArrayDynamicIndexing:
  case CapabilityStorageBufferArrayDynamicIndexing:
  case CapabilityStorageImageArrayDynamicIndexing:
  case CapabilityClipDistance:
  case CapabilityCullDistance:
  case CapabilityImageCubeArray:
  case CapabilitySampleRateShading:
  case CapabilityImageRect:
  case CapabilitySampledRect:
  case CapabilityGenericPointer:
  case CapabilityInt8:
  case CapabilityInputAttachment:
  case CapabilitySparseResidency:
  case CapabilityMinLod:
  case CapabilitySampled1D:
  case CapabilityImage1D:
  case CapabilitySampledCubeArray:
  case CapabilitySampledBuffer:
  case CapabilityImageBuffer:
  case CapabilityImageMSArray:
  case CapabilityStorageImageExtendedFormats:
  case CapabilityImageQuery:
  case CapabilityDerivativeControl:
  case CapabilityInterpolationFunction:
  case CapabilityTransformFeedback:
  case CapabilityGeometryStreams:
  case CapabilityStorageImageReadWithoutFormat:
  case CapabilityStorageImageWriteWithoutFormat:
  case CapabilityMultiViewport:
  case CapabilitySubgroupDispatch:
  case CapabilityNamedBarrier:
  case CapabilityPipeStorage:
  case CapabilityGroupNonUniform:
  case CapabilityGroupNonUniformVote:
  case CapabilityGroupNonUniformArithmetic:
  case CapabilityGroupNonUniformBallot:
  case CapabilityGroupNonUniformShuffle:
  case CapabilityGroupNonUniformShuffleRelative:
  case CapabilityGroupNonUniformClustered:
  case CapabilityGroupNonUniformQuad:
  case CapabilityDenormPreserve:
  case CapabilityDenormFlushToZero:
  case CapabilitySignedZeroInfNanPreserve:
  case CapabilityRoundingModeRTE:
  case CapabilityRoundingModeRTZ:
  case CapabilityRoundToInfinityINTEL:
  case CapabilityFloatingPointModeINTEL:
  case CapabilityVectorComputeINTEL:
  case CapabilityVectorAnyINTEL:
  case CapabilityFPGAMemoryAttributesINTEL:
  case CapabilityFPGAMemoryAccessesINTEL:
  case CapabilityArbitraryPrecisionIntegersINTEL:
  case CapabilityArbitraryPrecisionFixedPointINTEL:
  case CapabilityArbitraryPrecisionFloatingPointINTEL:
  case CapabilityFPGALoopControlsINTEL:
  case CapabilityBlockingPipesINTEL:
  case CapabilityUnstructuredLoopControlsINTEL:
  case CapabilityKernelAttributesINTEL:
  case CapabilityFPGAKernelAttributesINTEL:
    return true;
  default:
    return false;
  }
}

inline bool isValidImageOperandsMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= ImageOperandsBiasMask;
  ValidMask |= ImageOperandsLodMask;
  ValidMask |= ImageOperandsGradMask;
  ValidMask |= ImageOperandsConstOffsetMask;
  ValidMask |= ImageOperandsOffsetMask;
  ValidMask |= ImageOperandsConstOffsetsMask;
  ValidMask |= ImageOperandsSampleMask;
  ValidMask |= ImageOperandsMinLodMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidFPFastMathModeMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= FPFastMathModeNotNaNMask;
  ValidMask |= FPFastMathModeNotInfMask;
  ValidMask |= FPFastMathModeNSZMask;
  ValidMask |= FPFastMathModeAllowRecipMask;
  ValidMask |= FPFastMathModeFastMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidSelectionControlMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= SelectionControlFlattenMask;
  ValidMask |= SelectionControlDontFlattenMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidLoopControlMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= LoopControlUnrollMask;
  ValidMask |= LoopControlDontUnrollMask;
  ValidMask |= LoopControlPartialCountMask;
  ValidMask |= LoopControlDependencyInfiniteMask;
  ValidMask |= LoopControlDependencyLengthMask;
  ValidMask |= LoopControlInitiationIntervalINTELMask;
  ValidMask |= LoopControlMaxConcurrencyINTELMask;
  ValidMask |= LoopControlDependencyArrayINTELMask;
  ValidMask |= LoopControlPipelineEnableINTELMask;
  ValidMask |= LoopControlLoopCoalesceINTELMask;
  ValidMask |= LoopControlMaxInterleavingINTELMask;
  ValidMask |= LoopControlSpeculatedIterationsINTELMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidFunctionControlMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= FunctionControlInlineMask;
  ValidMask |= FunctionControlDontInlineMask;
  ValidMask |= FunctionControlPureMask;
  ValidMask |= FunctionControlConstMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidMemorySemanticsMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= MemorySemanticsAcquireMask;
  ValidMask |= MemorySemanticsReleaseMask;
  ValidMask |= MemorySemanticsAcquireReleaseMask;
  ValidMask |= MemorySemanticsSequentiallyConsistentMask;
  ValidMask |= MemorySemanticsUniformMemoryMask;
  ValidMask |= MemorySemanticsSubgroupMemoryMask;
  ValidMask |= MemorySemanticsWorkgroupMemoryMask;
  ValidMask |= MemorySemanticsCrossWorkgroupMemoryMask;
  ValidMask |= MemorySemanticsAtomicCounterMemoryMask;
  ValidMask |= MemorySemanticsImageMemoryMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidMemoryAccessMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= MemoryAccessVolatileMask;
  ValidMask |= MemoryAccessAlignedMask;
  ValidMask |= MemoryAccessNontemporalMask;

  return (Mask & ~ValidMask) == 0;
}

inline bool isValidKernelProfilingInfoMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= KernelProfilingInfoCmdExecTimeMask;

  return (Mask & ~ValidMask) == 0;
}

} /* namespace SPIRV */

#endif // SPIRV_LIBSPIRV_SPIRVISVALIDENUM_H
