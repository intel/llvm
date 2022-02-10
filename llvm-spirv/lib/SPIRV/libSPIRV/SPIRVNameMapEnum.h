//===- SPIRVNameMapEnum.h - SPIR-V NameMap enums ----------------*- C++ -*-===//
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
/// This file defines SPIR-V NameMap enums.
///
//===----------------------------------------------------------------------===//
// WARNING:
//
// This file has been generated using `tools/spirv-tool/gen_spirv.bash` and
// should not be modified manually. If the file needs to be updated, edit the
// script and any other source file instead, before re-generating this file.
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVNAMEMAPENUM_H
#define SPIRV_LIBSPIRV_SPIRVNAMEMAPENUM_H

#include "SPIRVEnum.h"
#include "spirv/unified1/spirv.hpp"
#include "spirv_internal.hpp"

using namespace spv;

namespace SPIRV {

template <> inline void SPIRVMap<LinkageType, std::string>::init() {
  add(LinkageTypeExport, "Export");
  add(LinkageTypeImport, "Import");
  add(LinkageTypeLinkOnceODR, "LinkOnceODR");
  add(internal::LinkageTypeInternal, "Internal");
  add(LinkageTypeMax, "Max");
}
SPIRV_DEF_NAMEMAP(LinkageType, SPIRVLinkageTypeNameMap)

template <> inline void SPIRVMap<Decoration, std::string>::init() {
  add(DecorationRelaxedPrecision, "RelaxedPrecision");
  add(DecorationSpecId, "SpecId");
  add(DecorationBlock, "Block");
  add(DecorationBufferBlock, "BufferBlock");
  add(DecorationRowMajor, "RowMajor");
  add(DecorationColMajor, "ColMajor");
  add(DecorationArrayStride, "ArrayStride");
  add(DecorationMatrixStride, "MatrixStride");
  add(DecorationGLSLShared, "GLSLShared");
  add(DecorationGLSLPacked, "GLSLPacked");
  add(DecorationCPacked, "CPacked");
  add(DecorationBuiltIn, "BuiltIn");
  add(DecorationNoPerspective, "NoPerspective");
  add(DecorationFlat, "Flat");
  add(DecorationPatch, "Patch");
  add(DecorationCentroid, "Centroid");
  add(DecorationSample, "Sample");
  add(DecorationInvariant, "Invariant");
  add(DecorationRestrict, "Restrict");
  add(DecorationAliased, "Aliased");
  add(DecorationVolatile, "Volatile");
  add(DecorationConstant, "Constant");
  add(DecorationCoherent, "Coherent");
  add(DecorationNonWritable, "NonWritable");
  add(DecorationNonReadable, "NonReadable");
  add(DecorationUniform, "Uniform");
  add(DecorationUniformId, "UniformId");
  add(DecorationSaturatedConversion, "SaturatedConversion");
  add(DecorationStream, "Stream");
  add(DecorationLocation, "Location");
  add(DecorationComponent, "Component");
  add(DecorationIndex, "Index");
  add(DecorationBinding, "Binding");
  add(DecorationDescriptorSet, "DescriptorSet");
  add(DecorationOffset, "Offset");
  add(DecorationXfbBuffer, "XfbBuffer");
  add(DecorationXfbStride, "XfbStride");
  add(DecorationFuncParamAttr, "FuncParamAttr");
  add(DecorationFPRoundingMode, "FPRoundingMode");
  add(DecorationFPFastMathMode, "FPFastMathMode");
  add(DecorationLinkageAttributes, "LinkageAttributes");
  add(DecorationNoContraction, "NoContraction");
  add(DecorationInputAttachmentIndex, "InputAttachmentIndex");
  add(DecorationAlignment, "Alignment");
  add(DecorationMaxByteOffset, "MaxByteOffset");
  add(DecorationAlignmentId, "AlignmentId");
  add(DecorationMaxByteOffsetId, "MaxByteOffsetId");
  add(DecorationNoSignedWrap, "NoSignedWrap");
  add(DecorationNoUnsignedWrap, "NoUnsignedWrap");
  add(DecorationExplicitInterpAMD, "ExplicitInterpAMD");
  add(DecorationOverrideCoverageNV, "OverrideCoverageNV");
  add(DecorationPassthroughNV, "PassthroughNV");
  add(DecorationViewportRelativeNV, "ViewportRelativeNV");
  add(DecorationSecondaryViewportRelativeNV, "SecondaryViewportRelativeNV");
  add(DecorationPerPrimitiveNV, "PerPrimitiveNV");
  add(DecorationPerViewNV, "PerViewNV");
  add(DecorationPerTaskNV, "PerTaskNV");
  add(DecorationPerVertexNV, "PerVertexNV");
  add(DecorationNonUniform, "NonUniform");
  add(DecorationNonUniformEXT, "NonUniformEXT");
  add(DecorationRestrictPointer, "RestrictPointer");
  add(DecorationRestrictPointerEXT, "RestrictPointerEXT");
  add(DecorationAliasedPointer, "AliasedPointer");
  add(DecorationAliasedPointerEXT, "AliasedPointerEXT");
  add(DecorationSIMTCallINTEL, "SIMTCallINTEL");
  add(DecorationReferencedIndirectlyINTEL, "ReferencedIndirectlyINTEL");
  add(DecorationClobberINTEL, "ClobberINTEL");
  add(DecorationSideEffectsINTEL, "SideEffectsINTEL");
  add(DecorationVectorComputeVariableINTEL, "VectorComputeVariableINTEL");
  add(DecorationFuncParamIOKindINTEL, "FuncParamIOKind");
  add(DecorationVectorComputeFunctionINTEL, "VectorComputeFunctionINTEL");
  add(DecorationStackCallINTEL, "StackCallINTEL");
  add(DecorationGlobalVariableOffsetINTEL, "GlobalVariableOffsetINTEL");
  add(DecorationCounterBuffer, "CounterBuffer");
  add(DecorationHlslCounterBufferGOOGLE, "HlslCounterBufferGOOGLE");
  add(DecorationHlslSemanticGOOGLE, "HlslSemanticGOOGLE");
  add(DecorationUserSemantic, "UserSemantic");
  add(DecorationUserTypeGOOGLE, "UserTypeGOOGLE");
  add(DecorationFunctionRoundingModeINTEL, "FunctionRoundingModeINTEL");
  add(DecorationFunctionDenormModeINTEL, "FunctionDenormModeINTEL");
  add(DecorationRegisterINTEL, "RegisterINTEL");
  add(DecorationMemoryINTEL, "MemoryINTEL");
  add(DecorationNumbanksINTEL, "NumbanksINTEL");
  add(DecorationBankwidthINTEL, "BankwidthINTEL");
  add(DecorationMaxPrivateCopiesINTEL, "MaxPrivateCopiesINTEL");
  add(DecorationSinglepumpINTEL, "SinglepumpINTEL");
  add(DecorationDoublepumpINTEL, "DoublepumpINTEL");
  add(DecorationMaxReplicatesINTEL, "MaxReplicatesINTEL");
  add(DecorationSimpleDualPortINTEL, "SimpleDualPortINTEL");
  add(DecorationMergeINTEL, "MergeINTEL");
  add(DecorationBankBitsINTEL, "BankBitsINTEL");
  add(DecorationForcePow2DepthINTEL, "ForcePow2DepthINTEL");
  add(DecorationBurstCoalesceINTEL, "BurstCoalesceINTEL");
  add(DecorationCacheSizeINTEL, "CacheSizeINTEL");
  add(DecorationDontStaticallyCoalesceINTEL, "DontStaticallyCoalesceINTEL");
  add(DecorationPrefetchINTEL, "PrefetchINTEL");
  add(DecorationStallEnableINTEL, "StallEnableINTEL");
  add(DecorationFuseLoopsInFunctionINTEL, "FuseLoopsInFunctionINTEL");
  add(DecorationBufferLocationINTEL, "BufferLocationINTEL");
  add(DecorationIOPipeStorageINTEL, "IOPipeStorageINTEL");
  add(DecorationFunctionFloatingPointModeINTEL,
      "FunctionFloatingPointModeINTEL");
  add(DecorationSingleElementVectorINTEL, "SingleElementVectorINTEL");
  add(DecorationVectorComputeCallableFunctionINTEL,
      "VectorComputeCallableFunctionINTEL");
  add(DecorationMediaBlockIOINTEL, "MediaBlockIOINTEL");

  // From spirv_internal.hpp
  add(internal::DecorationFuncParamKindINTEL, "FuncParamKindINTEL");
  add(internal::DecorationFuncParamDescINTEL, "FuncParamDescINTEL");
  add(internal::DecorationCallableFunctionINTEL, "CallableFunctionINTEL");
  add(internal::DecorationMathOpDSPModeINTEL, "MathOpDSPModeINTEL");
  add(internal::DecorationAliasScopeINTEL, "AliasScopeINTEL");
  add(internal::DecorationNoAliasINTEL, "NoAliasINTEL");
  add(internal::DecorationInitiationIntervalINTEL, "InitiationIntervalINTEL");
  add(internal::DecorationMaxConcurrencyINTEL, "MaxConcurrencyINTEL");
  add(internal::DecorationPipelineEnableINTEL, "PipelineEnableINTEL");
  add(internal::DecorationRuntimeAlignedINTEL, "RuntimeAlignedINTEL");
  add(internal::DecorationArgumentAttributeINTEL, "ArgumentAttributeINTEL");

  add(DecorationMax, "Max");
}
SPIRV_DEF_NAMEMAP(Decoration, SPIRVDecorationNameMap)

template <> inline void SPIRVMap<BuiltIn, std::string>::init() {
  add(BuiltInPosition, "BuiltInPosition");
  add(BuiltInPointSize, "BuiltInPointSize");
  add(BuiltInClipDistance, "BuiltInClipDistance");
  add(BuiltInCullDistance, "BuiltInCullDistance");
  add(BuiltInVertexId, "BuiltInVertexId");
  add(BuiltInInstanceId, "BuiltInInstanceId");
  add(BuiltInPrimitiveId, "BuiltInPrimitiveId");
  add(BuiltInInvocationId, "BuiltInInvocationId");
  add(BuiltInLayer, "BuiltInLayer");
  add(BuiltInViewportIndex, "BuiltInViewportIndex");
  add(BuiltInTessLevelOuter, "BuiltInTessLevelOuter");
  add(BuiltInTessLevelInner, "BuiltInTessLevelInner");
  add(BuiltInTessCoord, "BuiltInTessCoord");
  add(BuiltInPatchVertices, "BuiltInPatchVertices");
  add(BuiltInFragCoord, "BuiltInFragCoord");
  add(BuiltInPointCoord, "BuiltInPointCoord");
  add(BuiltInFrontFacing, "BuiltInFrontFacing");
  add(BuiltInSampleId, "BuiltInSampleId");
  add(BuiltInSamplePosition, "BuiltInSamplePosition");
  add(BuiltInSampleMask, "BuiltInSampleMask");
  add(BuiltInFragDepth, "BuiltInFragDepth");
  add(BuiltInHelperInvocation, "BuiltInHelperInvocation");
  add(BuiltInNumWorkgroups, "BuiltInNumWorkgroups");
  add(BuiltInWorkgroupSize, "BuiltInWorkgroupSize");
  add(BuiltInWorkgroupId, "BuiltInWorkgroupId");
  add(BuiltInLocalInvocationId, "BuiltInLocalInvocationId");
  add(BuiltInGlobalInvocationId, "BuiltInGlobalInvocationId");
  add(BuiltInLocalInvocationIndex, "BuiltInLocalInvocationIndex");
  add(BuiltInWorkDim, "BuiltInWorkDim");
  add(BuiltInGlobalSize, "BuiltInGlobalSize");
  add(BuiltInEnqueuedWorkgroupSize, "BuiltInEnqueuedWorkgroupSize");
  add(BuiltInGlobalOffset, "BuiltInGlobalOffset");
  add(BuiltInGlobalLinearId, "BuiltInGlobalLinearId");
  add(BuiltInSubgroupSize, "BuiltInSubgroupSize");
  add(BuiltInSubgroupMaxSize, "BuiltInSubgroupMaxSize");
  add(BuiltInNumSubgroups, "BuiltInNumSubgroups");
  add(BuiltInNumEnqueuedSubgroups, "BuiltInNumEnqueuedSubgroups");
  add(BuiltInSubgroupId, "BuiltInSubgroupId");
  add(BuiltInSubgroupLocalInvocationId, "BuiltInSubgroupLocalInvocationId");
  add(BuiltInVertexIndex, "BuiltInVertexIndex");
  add(BuiltInInstanceIndex, "BuiltInInstanceIndex");
  add(BuiltInSubgroupEqMask, "BuiltInSubgroupEqMask");
  add(BuiltInSubgroupEqMaskKHR, "BuiltInSubgroupEqMaskKHR");
  add(BuiltInSubgroupGeMask, "BuiltInSubgroupGeMask");
  add(BuiltInSubgroupGeMaskKHR, "BuiltInSubgroupGeMaskKHR");
  add(BuiltInSubgroupGtMask, "BuiltInSubgroupGtMask");
  add(BuiltInSubgroupGtMaskKHR, "BuiltInSubgroupGtMaskKHR");
  add(BuiltInSubgroupLeMask, "BuiltInSubgroupLeMask");
  add(BuiltInSubgroupLeMaskKHR, "BuiltInSubgroupLeMaskKHR");
  add(BuiltInSubgroupLtMask, "BuiltInSubgroupLtMask");
  add(BuiltInSubgroupLtMaskKHR, "BuiltInSubgroupLtMaskKHR");
  add(BuiltInBaseVertex, "BuiltInBaseVertex");
  add(BuiltInBaseInstance, "BuiltInBaseInstance");
  add(BuiltInDrawIndex, "BuiltInDrawIndex");
  add(BuiltInPrimitiveShadingRateKHR, "BuiltInPrimitiveShadingRateKHR");
  add(BuiltInDeviceIndex, "BuiltInDeviceIndex");
  add(BuiltInViewIndex, "BuiltInViewIndex");
  add(BuiltInShadingRateKHR, "BuiltInShadingRateKHR");
  add(BuiltInBaryCoordNoPerspAMD, "BuiltInBaryCoordNoPerspAMD");
  add(BuiltInBaryCoordNoPerspCentroidAMD, "BuiltInBaryCoordNoPerspCentroidAMD");
  add(BuiltInBaryCoordNoPerspSampleAMD, "BuiltInBaryCoordNoPerspSampleAMD");
  add(BuiltInBaryCoordSmoothAMD, "BuiltInBaryCoordSmoothAMD");
  add(BuiltInBaryCoordSmoothCentroidAMD, "BuiltInBaryCoordSmoothCentroidAMD");
  add(BuiltInBaryCoordSmoothSampleAMD, "BuiltInBaryCoordSmoothSampleAMD");
  add(BuiltInBaryCoordPullModelAMD, "BuiltInBaryCoordPullModelAMD");
  add(BuiltInFragStencilRefEXT, "BuiltInFragStencilRefEXT");
  add(BuiltInViewportMaskNV, "BuiltInViewportMaskNV");
  add(BuiltInSecondaryPositionNV, "BuiltInSecondaryPositionNV");
  add(BuiltInSecondaryViewportMaskNV, "BuiltInSecondaryViewportMaskNV");
  add(BuiltInPositionPerViewNV, "BuiltInPositionPerViewNV");
  add(BuiltInViewportMaskPerViewNV, "BuiltInViewportMaskPerViewNV");
  add(BuiltInFullyCoveredEXT, "BuiltInFullyCoveredEXT");
  add(BuiltInTaskCountNV, "BuiltInTaskCountNV");
  add(BuiltInPrimitiveCountNV, "BuiltInPrimitiveCountNV");
  add(BuiltInPrimitiveIndicesNV, "BuiltInPrimitiveIndicesNV");
  add(BuiltInClipDistancePerViewNV, "BuiltInClipDistancePerViewNV");
  add(BuiltInCullDistancePerViewNV, "BuiltInCullDistancePerViewNV");
  add(BuiltInLayerPerViewNV, "BuiltInLayerPerViewNV");
  add(BuiltInMeshViewCountNV, "BuiltInMeshViewCountNV");
  add(BuiltInMeshViewIndicesNV, "BuiltInMeshViewIndicesNV");
  add(BuiltInBaryCoordNV, "BuiltInBaryCoordNV");
  add(BuiltInBaryCoordNoPerspNV, "BuiltInBaryCoordNoPerspNV");
  add(BuiltInFragSizeEXT, "BuiltInFragSizeEXT");
  add(BuiltInFragmentSizeNV, "BuiltInFragmentSizeNV");
  add(BuiltInFragInvocationCountEXT, "BuiltInFragInvocationCountEXT");
  add(BuiltInInvocationsPerPixelNV, "BuiltInInvocationsPerPixelNV");
  add(BuiltInLaunchIdKHR, "BuiltInLaunchIdKHR");
  add(BuiltInLaunchIdNV, "BuiltInLaunchIdNV");
  add(BuiltInLaunchSizeKHR, "BuiltInLaunchSizeKHR");
  add(BuiltInLaunchSizeNV, "BuiltInLaunchSizeNV");
  add(BuiltInWorldRayOriginKHR, "BuiltInWorldRayOriginKHR");
  add(BuiltInWorldRayOriginNV, "BuiltInWorldRayOriginNV");
  add(BuiltInWorldRayDirectionKHR, "BuiltInWorldRayDirectionKHR");
  add(BuiltInWorldRayDirectionNV, "BuiltInWorldRayDirectionNV");
  add(BuiltInObjectRayOriginKHR, "BuiltInObjectRayOriginKHR");
  add(BuiltInObjectRayOriginNV, "BuiltInObjectRayOriginNV");
  add(BuiltInObjectRayDirectionKHR, "BuiltInObjectRayDirectionKHR");
  add(BuiltInObjectRayDirectionNV, "BuiltInObjectRayDirectionNV");
  add(BuiltInRayTminKHR, "BuiltInRayTminKHR");
  add(BuiltInRayTminNV, "BuiltInRayTminNV");
  add(BuiltInRayTmaxKHR, "BuiltInRayTmaxKHR");
  add(BuiltInRayTmaxNV, "BuiltInRayTmaxNV");
  add(BuiltInInstanceCustomIndexKHR, "BuiltInInstanceCustomIndexKHR");
  add(BuiltInInstanceCustomIndexNV, "BuiltInInstanceCustomIndexNV");
  add(BuiltInObjectToWorldKHR, "BuiltInObjectToWorldKHR");
  add(BuiltInObjectToWorldNV, "BuiltInObjectToWorldNV");
  add(BuiltInWorldToObjectKHR, "BuiltInWorldToObjectKHR");
  add(BuiltInWorldToObjectNV, "BuiltInWorldToObjectNV");
  add(BuiltInHitTNV, "BuiltInHitTNV");
  add(BuiltInHitKindKHR, "BuiltInHitKindKHR");
  add(BuiltInHitKindNV, "BuiltInHitKindNV");
  add(BuiltInIncomingRayFlagsKHR, "BuiltInIncomingRayFlagsKHR");
  add(BuiltInIncomingRayFlagsNV, "BuiltInIncomingRayFlagsNV");
  add(BuiltInRayGeometryIndexKHR, "BuiltInRayGeometryIndexKHR");
  add(BuiltInWarpsPerSMNV, "BuiltInWarpsPerSMNV");
  add(BuiltInSMCountNV, "BuiltInSMCountNV");
  add(BuiltInWarpIDNV, "BuiltInWarpIDNV");
  add(BuiltInSMIDNV, "BuiltInSMIDNV");
  add(BuiltInMax, "BuiltInMax");
  add(internal::BuiltInSubDeviceIDINTEL, "BuiltInSubDeviceIDINTEL");
  add(internal::BuiltInGlobalHWThreadIDINTEL, "BuiltInGlobalHWThreadIDINTEL");
}
SPIRV_DEF_NAMEMAP(BuiltIn, SPIRVBuiltInNameMap)

template <> inline void SPIRVMap<Capability, std::string>::init() {
  add(CapabilityMatrix, "Matrix");
  add(CapabilityShader, "Shader");
  add(CapabilityGeometry, "Geometry");
  add(CapabilityTessellation, "Tessellation");
  add(CapabilityAddresses, "Addresses");
  add(CapabilityLinkage, "Linkage");
  add(CapabilityKernel, "Kernel");
  add(CapabilityVector16, "Vector16");
  add(CapabilityFloat16Buffer, "Float16Buffer");
  add(CapabilityFloat16, "Float16");
  add(CapabilityFloat64, "Float64");
  add(CapabilityInt64, "Int64");
  add(CapabilityInt64Atomics, "Int64Atomics");
  add(CapabilityImageBasic, "ImageBasic");
  add(CapabilityImageReadWrite, "ImageReadWrite");
  add(CapabilityImageMipmap, "ImageMipmap");
  add(CapabilityPipes, "Pipes");
  add(CapabilityGroups, "Groups");
  add(CapabilityDeviceEnqueue, "DeviceEnqueue");
  add(CapabilityLiteralSampler, "LiteralSampler");
  add(CapabilityAtomicStorage, "AtomicStorage");
  add(CapabilityInt16, "Int16");
  add(CapabilityTessellationPointSize, "TessellationPointSize");
  add(CapabilityGeometryPointSize, "GeometryPointSize");
  add(CapabilityImageGatherExtended, "ImageGatherExtended");
  add(CapabilityStorageImageMultisample, "StorageImageMultisample");
  add(CapabilityUniformBufferArrayDynamicIndexing,
      "UniformBufferArrayDynamicIndexing");
  add(CapabilitySampledImageArrayDynamicIndexing,
      "SampledImageArrayDynamicIndexing");
  add(CapabilityStorageBufferArrayDynamicIndexing,
      "StorageBufferArrayDynamicIndexing");
  add(CapabilityStorageImageArrayDynamicIndexing,
      "StorageImageArrayDynamicIndexing");
  add(CapabilityClipDistance, "ClipDistance");
  add(CapabilityCullDistance, "CullDistance");
  add(CapabilityImageCubeArray, "ImageCubeArray");
  add(CapabilitySampleRateShading, "SampleRateShading");
  add(CapabilityImageRect, "ImageRect");
  add(CapabilitySampledRect, "SampledRect");
  add(CapabilityGenericPointer, "GenericPointer");
  add(CapabilityInt8, "Int8");
  add(CapabilityInputAttachment, "InputAttachment");
  add(CapabilitySparseResidency, "SparseResidency");
  add(CapabilityMinLod, "MinLod");
  add(CapabilitySampled1D, "Sampled1D");
  add(CapabilityImage1D, "Image1D");
  add(CapabilitySampledCubeArray, "SampledCubeArray");
  add(CapabilitySampledBuffer, "SampledBuffer");
  add(CapabilityImageBuffer, "ImageBuffer");
  add(CapabilityImageMSArray, "ImageMSArray");
  add(CapabilityStorageImageExtendedFormats, "StorageImageExtendedFormats");
  add(CapabilityImageQuery, "ImageQuery");
  add(CapabilityDerivativeControl, "DerivativeControl");
  add(CapabilityInterpolationFunction, "InterpolationFunction");
  add(CapabilityTransformFeedback, "TransformFeedback");
  add(CapabilityGeometryStreams, "GeometryStreams");
  add(CapabilityStorageImageReadWithoutFormat, "StorageImageReadWithoutFormat");
  add(CapabilityStorageImageWriteWithoutFormat,
      "StorageImageWriteWithoutFormat");
  add(CapabilityMultiViewport, "MultiViewport");
  add(CapabilitySubgroupDispatch, "SubgroupDispatch");
  add(CapabilityNamedBarrier, "NamedBarrier");
  add(CapabilityPipeStorage, "PipeStorage");
  add(CapabilityGroupNonUniform, "GroupNonUniform");
  add(CapabilityGroupNonUniformVote, "GroupNonUniformVote");
  add(CapabilityGroupNonUniformArithmetic, "GroupNonUniformArithmetic");
  add(CapabilityGroupNonUniformBallot, "GroupNonUniformBallot");
  add(CapabilityGroupNonUniformShuffle, "GroupNonUniformShuffle");
  add(CapabilityGroupNonUniformShuffleRelative,
      "GroupNonUniformShuffleRelative");
  add(CapabilityGroupNonUniformClustered, "GroupNonUniformClustered");
  add(CapabilityGroupNonUniformQuad, "GroupNonUniformQuad");
  add(CapabilityShaderLayer, "ShaderLayer");
  add(CapabilityShaderViewportIndex, "ShaderViewportIndex");
  add(CapabilityFragmentShadingRateKHR, "FragmentShadingRateKHR");
  add(CapabilitySubgroupBallotKHR, "SubgroupBallotKHR");
  add(CapabilityDrawParameters, "DrawParameters");
  add(CapabilityWorkgroupMemoryExplicitLayoutKHR,
      "WorkgroupMemoryExplicitLayoutKHR");
  add(CapabilityWorkgroupMemoryExplicitLayout8BitAccessKHR,
      "WorkgroupMemoryExplicitLayout8BitAccessKHR");
  add(CapabilityWorkgroupMemoryExplicitLayout16BitAccessKHR,
      "WorkgroupMemoryExplicitLayout16BitAccessKHR");
  add(CapabilitySubgroupVoteKHR, "SubgroupVoteKHR");
  add(CapabilityStorageBuffer16BitAccess, "StorageBuffer16BitAccess");
  add(CapabilityStorageUniformBufferBlock16, "StorageUniformBufferBlock16");
  add(CapabilityStorageUniform16, "StorageUniform16");
  add(CapabilityUniformAndStorageBuffer16BitAccess,
      "UniformAndStorageBuffer16BitAccess");
  add(CapabilityStoragePushConstant16, "StoragePushConstant16");
  add(CapabilityStorageInputOutput16, "StorageInputOutput16");
  add(CapabilityDeviceGroup, "DeviceGroup");
  add(CapabilityMultiView, "MultiView");
  add(CapabilityVariablePointersStorageBuffer, "VariablePointersStorageBuffer");
  add(CapabilityVariablePointers, "VariablePointers");
  add(CapabilityAtomicStorageOps, "AtomicStorageOps");
  add(CapabilitySampleMaskPostDepthCoverage, "SampleMaskPostDepthCoverage");
  add(CapabilityStorageBuffer8BitAccess, "StorageBuffer8BitAccess");
  add(CapabilityUniformAndStorageBuffer8BitAccess,
      "UniformAndStorageBuffer8BitAccess");
  add(CapabilityStoragePushConstant8, "StoragePushConstant8");
  add(CapabilityDenormPreserve, "DenormPreserve");
  add(CapabilityDenormFlushToZero, "DenormFlushToZero");
  add(CapabilitySignedZeroInfNanPreserve, "SignedZeroInfNanPreserve");
  add(CapabilityRoundingModeRTE, "RoundingModeRTE");
  add(CapabilityRoundingModeRTZ, "RoundingModeRTZ");
  add(CapabilityRayQueryProvisionalKHR, "RayQueryProvisionalKHR");
  add(CapabilityRayQueryKHR, "RayQueryKHR");
  add(CapabilityRayTraversalPrimitiveCullingKHR,
      "RayTraversalPrimitiveCullingKHR");
  add(CapabilityRayTracingKHR, "RayTracingKHR");
  add(CapabilityFloat16ImageAMD, "Float16ImageAMD");
  add(CapabilityImageGatherBiasLodAMD, "ImageGatherBiasLodAMD");
  add(CapabilityFragmentMaskAMD, "FragmentMaskAMD");
  add(CapabilityStencilExportEXT, "StencilExportEXT");
  add(CapabilityImageReadWriteLodAMD, "ImageReadWriteLodAMD");
  add(CapabilityInt64ImageEXT, "Int64ImageEXT");
  add(CapabilityShaderClockKHR, "ShaderClockKHR");
  add(CapabilitySampleMaskOverrideCoverageNV, "SampleMaskOverrideCoverageNV");
  add(CapabilityGeometryShaderPassthroughNV, "GeometryShaderPassthroughNV");
  add(CapabilityShaderViewportIndexLayerEXT, "ShaderViewportIndexLayerEXT");
  add(CapabilityShaderViewportIndexLayerNV, "ShaderViewportIndexLayerNV");
  add(CapabilityShaderViewportMaskNV, "ShaderViewportMaskNV");
  add(CapabilityShaderStereoViewNV, "ShaderStereoViewNV");
  add(CapabilityPerViewAttributesNV, "PerViewAttributesNV");
  add(CapabilityFragmentFullyCoveredEXT, "FragmentFullyCoveredEXT");
  add(CapabilityMeshShadingNV, "MeshShadingNV");
  add(CapabilityImageFootprintNV, "ImageFootprintNV");
  add(CapabilityFragmentBarycentricNV, "FragmentBarycentricNV");
  add(CapabilityComputeDerivativeGroupQuadsNV, "ComputeDerivativeGroupQuadsNV");
  add(CapabilityFragmentDensityEXT, "FragmentDensityEXT");
  add(CapabilityShadingRateNV, "ShadingRateNV");
  add(CapabilityGroupNonUniformPartitionedNV, "GroupNonUniformPartitionedNV");
  add(CapabilityShaderNonUniform, "ShaderNonUniform");
  add(CapabilityShaderNonUniformEXT, "ShaderNonUniformEXT");
  add(CapabilityRuntimeDescriptorArray, "RuntimeDescriptorArray");
  add(CapabilityRuntimeDescriptorArrayEXT, "RuntimeDescriptorArrayEXT");
  add(CapabilityInputAttachmentArrayDynamicIndexing,
      "InputAttachmentArrayDynamicIndexing");
  add(CapabilityInputAttachmentArrayDynamicIndexingEXT,
      "InputAttachmentArrayDynamicIndexingEXT");
  add(CapabilityUniformTexelBufferArrayDynamicIndexing,
      "UniformTexelBufferArrayDynamicIndexing");
  add(CapabilityUniformTexelBufferArrayDynamicIndexingEXT,
      "UniformTexelBufferArrayDynamicIndexingEXT");
  add(CapabilityStorageTexelBufferArrayDynamicIndexing,
      "StorageTexelBufferArrayDynamicIndexing");
  add(CapabilityStorageTexelBufferArrayDynamicIndexingEXT,
      "StorageTexelBufferArrayDynamicIndexingEXT");
  add(CapabilityUniformBufferArrayNonUniformIndexing,
      "UniformBufferArrayNonUniformIndexing");
  add(CapabilityUniformBufferArrayNonUniformIndexingEXT,
      "UniformBufferArrayNonUniformIndexingEXT");
  add(CapabilitySampledImageArrayNonUniformIndexing,
      "SampledImageArrayNonUniformIndexing");
  add(CapabilitySampledImageArrayNonUniformIndexingEXT,
      "SampledImageArrayNonUniformIndexingEXT");
  add(CapabilityStorageBufferArrayNonUniformIndexing,
      "StorageBufferArrayNonUniformIndexing");
  add(CapabilityStorageBufferArrayNonUniformIndexingEXT,
      "StorageBufferArrayNonUniformIndexingEXT");
  add(CapabilityStorageImageArrayNonUniformIndexing,
      "StorageImageArrayNonUniformIndexing");
  add(CapabilityStorageImageArrayNonUniformIndexingEXT,
      "StorageImageArrayNonUniformIndexingEXT");
  add(CapabilityInputAttachmentArrayNonUniformIndexing,
      "InputAttachmentArrayNonUniformIndexing");
  add(CapabilityInputAttachmentArrayNonUniformIndexingEXT,
      "InputAttachmentArrayNonUniformIndexingEXT");
  add(CapabilityUniformTexelBufferArrayNonUniformIndexing,
      "UniformTexelBufferArrayNonUniformIndexing");
  add(CapabilityUniformTexelBufferArrayNonUniformIndexingEXT,
      "UniformTexelBufferArrayNonUniformIndexingEXT");
  add(CapabilityStorageTexelBufferArrayNonUniformIndexing,
      "StorageTexelBufferArrayNonUniformIndexing");
  add(CapabilityStorageTexelBufferArrayNonUniformIndexingEXT,
      "StorageTexelBufferArrayNonUniformIndexingEXT");
  add(CapabilityRayTracingNV, "RayTracingNV");
  add(CapabilityVulkanMemoryModel, "VulkanMemoryModel");
  add(CapabilityVulkanMemoryModelKHR, "VulkanMemoryModelKHR");
  add(CapabilityVulkanMemoryModelDeviceScope, "VulkanMemoryModelDeviceScope");
  add(CapabilityVulkanMemoryModelDeviceScopeKHR,
      "VulkanMemoryModelDeviceScopeKHR");
  add(CapabilityPhysicalStorageBufferAddresses,
      "PhysicalStorageBufferAddresses");
  add(CapabilityPhysicalStorageBufferAddressesEXT,
      "PhysicalStorageBufferAddressesEXT");
  add(CapabilityComputeDerivativeGroupLinearNV,
      "ComputeDerivativeGroupLinearNV");
  add(CapabilityRayTracingProvisionalKHR, "RayTracingProvisionalKHR");
  add(CapabilityCooperativeMatrixNV, "CooperativeMatrixNV");
  add(CapabilityFragmentShaderSampleInterlockEXT,
      "FragmentShaderSampleInterlockEXT");
  add(CapabilityFragmentShaderShadingRateInterlockEXT,
      "FragmentShaderShadingRateInterlockEXT");
  add(CapabilityShaderSMBuiltinsNV, "ShaderSMBuiltinsNV");
  add(CapabilityFragmentShaderPixelInterlockEXT,
      "FragmentShaderPixelInterlockEXT");
  add(CapabilityDemoteToHelperInvocationEXT, "DemoteToHelperInvocationEXT");
  add(CapabilitySubgroupShuffleINTEL, "SubgroupShuffleINTEL");
  add(CapabilitySubgroupBufferBlockIOINTEL, "SubgroupBufferBlockIOINTEL");
  add(CapabilitySubgroupImageBlockIOINTEL, "SubgroupImageBlockIOINTEL");
  add(CapabilitySubgroupImageMediaBlockIOINTEL,
      "SubgroupImageMediaBlockIOINTEL");
  add(CapabilityRoundToInfinityINTEL, "RoundToInfinityINTEL");
  add(CapabilityFloatingPointModeINTEL, "FloatingPointModeINTEL");
  add(CapabilityIntegerFunctions2INTEL, "IntegerFunctions2INTEL");
  add(CapabilityFunctionPointersINTEL, "FunctionPointersINTEL");
  add(CapabilityIndirectReferencesINTEL, "IndirectReferencesINTEL");
  add(CapabilityAsmINTEL, "AsmINTEL");
  add(CapabilityAtomicFloat32MinMaxEXT, "AtomicFloat32MinMaxEXT");
  add(CapabilityAtomicFloat64MinMaxEXT, "AtomicFloat64MinMaxEXT");
  add(CapabilityAtomicFloat16MinMaxEXT, "AtomicFloat16MinMaxEXT");
  add(CapabilityVectorComputeINTEL, "VectorComputeINTEL");
  add(CapabilityVectorAnyINTEL, "VectorAnyINTEL");
  add(CapabilityExpectAssumeKHR, "ExpectAssumeKHR");
  add(CapabilitySubgroupAvcMotionEstimationINTEL,
      "SubgroupAvcMotionEstimationINTEL");
  add(CapabilitySubgroupAvcMotionEstimationIntraINTEL,
      "SubgroupAvcMotionEstimationIntraINTEL");
  add(CapabilitySubgroupAvcMotionEstimationChromaINTEL,
      "SubgroupAvcMotionEstimationChromaINTEL");
  add(CapabilityVariableLengthArrayINTEL, "VariableLengthArrayINTEL");
  add(CapabilityFunctionFloatControlINTEL, "FunctionFloatControlINTEL");
  add(CapabilityFPGAMemoryAttributesINTEL, "FPGAMemoryAttributesINTEL");
  add(CapabilityFPFastMathModeINTEL, "FPFastMathModeINTEL");
  add(CapabilityArbitraryPrecisionIntegersINTEL,
      "ArbitraryPrecisionIntegersINTEL");
  add(CapabilityArbitraryPrecisionFloatingPointINTEL,
      "ArbitraryPrecisionFloatingPointINTEL");
  add(CapabilityUnstructuredLoopControlsINTEL, "UnstructuredLoopControlsINTEL");
  add(CapabilityFPGALoopControlsINTEL, "FPGALoopControlsINTEL");
  add(CapabilityKernelAttributesINTEL, "KernelAttributesINTEL");
  add(CapabilityFPGAKernelAttributesINTEL, "FPGAKernelAttributesINTEL");
  add(CapabilityFPGAMemoryAccessesINTEL, "FPGAMemoryAccessesINTEL");
  add(CapabilityFPGAClusterAttributesINTEL, "FPGAClusterAttributesINTEL");
  add(CapabilityLoopFuseINTEL, "LoopFuseINTEL");
  add(CapabilityFPGABufferLocationINTEL, "FPGABufferLocationINTEL");
  add(CapabilityArbitraryPrecisionFixedPointINTEL,
      "ArbitraryPrecisionFixedPointINTEL");
  add(CapabilityUSMStorageClassesINTEL, "USMStorageClassesINTEL");
  add(CapabilityIOPipesINTEL, "IOPipeINTEL");
  add(CapabilityBlockingPipesINTEL, "BlockingPipesINTEL");
  add(CapabilityFPGARegINTEL, "FPGARegINTEL");
  add(CapabilityDotProductInputAllKHR, "DotProductInputAllKHR");
  add(CapabilityDotProductInput4x8BitKHR, "DotProductInput4x8BitKHR");
  add(CapabilityDotProductInput4x8BitPackedKHR,
      "DotProductInput4x8BitPackedKHR");
  add(CapabilityDotProductKHR, "DotProductKHR");
  add(CapabilityBitInstructions, "BitInstructions");
  add(CapabilityAtomicFloat32AddEXT, "AtomicFloat32AddEXT");
  add(CapabilityAtomicFloat64AddEXT, "AtomicFloat64AddEXT");
  add(CapabilityLongConstantCompositeINTEL, "LongConstantCompositeINTEL");
  add(CapabilityAtomicFloat16AddEXT, "AtomicFloat16AddEXT");
  add(CapabilityDebugInfoModuleINTEL, "DebugInfoModuleINTEL");

  // From spirv_internal.hpp
  add(internal::CapabilityFPGADSPControlINTEL, "FPGADSPControlINTEL");
  add(internal::CapabilityFastCompositeINTEL, "FastCompositeINTEL");
  add(internal::CapabilityOptNoneINTEL, "OptNoneINTEL");
  add(internal::CapabilityMemoryAccessAliasingINTEL,
      "MemoryAccessAliasingINTEL");
  add(internal::CapabilityFPGAInvocationPipeliningAttributesINTEL,
      "FPGAInvocationPipeliningAttributesINTEL");
  add(internal::CapabilityTokenTypeINTEL, "TokenTypeINTEL");
  add(internal::CapabilityRuntimeAlignedAttributeINTEL,
      "RuntimeAlignedAttributeINTEL");
  add(CapabilityMax, "Max");
  add(internal::CapabilityFPArithmeticFenceINTEL, "FPArithmeticFenceINTEL");
  add(internal::CapabilityBfloat16ConversionINTEL, "Bfloat16ConversionINTEL");
  add(internal::CapabilityJointMatrixINTEL, "JointMatrixINTEL");
  add(internal::CapabilityHWThreadQueryINTEL, "HWThreadQueryINTEL");
}
SPIRV_DEF_NAMEMAP(Capability, SPIRVCapabilityNameMap)

} /* namespace SPIRV */

#endif // SPIRV_LIBSPIRV_SPIRVNAMEMAPENUM_H
