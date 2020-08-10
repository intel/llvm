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
#include "spirv.hpp"

using namespace spv;

namespace SPIRV {

template <> inline void SPIRVMap<SourceLanguage, std::string>::init() {
  add(SourceLanguageUnknown, "Unknown");
  add(SourceLanguageESSL, "ESSL");
  add(SourceLanguageGLSL, "GLSL");
  add(SourceLanguageOpenCL_C, "OpenCL_C");
  add(SourceLanguageOpenCL_CPP, "OpenCL_CPP");
  add(SourceLanguageHLSL, "HLSL");
  add(SourceLanguageMax, "Max");
}
SPIRV_DEF_NAMEMAP(SourceLanguage, SPIRVSourceLanguageNameMap)

template <> inline void SPIRVMap<ExecutionModel, std::string>::init() {
  add(ExecutionModelVertex, "Vertex");
  add(ExecutionModelTessellationControl, "TessellationControl");
  add(ExecutionModelTessellationEvaluation, "TessellationEvaluation");
  add(ExecutionModelGeometry, "Geometry");
  add(ExecutionModelFragment, "Fragment");
  add(ExecutionModelGLCompute, "GLCompute");
  add(ExecutionModelKernel, "Kernel");
  add(ExecutionModelTaskNV, "TaskNV");
  add(ExecutionModelMeshNV, "MeshNV");
  add(ExecutionModelRayGenerationNV, "RayGenerationNV");
  add(ExecutionModelIntersectionNV, "IntersectionNV");
  add(ExecutionModelAnyHitNV, "AnyHitNV");
  add(ExecutionModelClosestHitNV, "ClosestHitNV");
  add(ExecutionModelMissNV, "MissNV");
  add(ExecutionModelCallableNV, "CallableNV");
  add(ExecutionModelMax, "Max");
}
SPIRV_DEF_NAMEMAP(ExecutionModel, SPIRVExecutionModelNameMap)

template <> inline void SPIRVMap<AddressingModel, std::string>::init() {
  add(AddressingModelLogical, "Logical");
  add(AddressingModelPhysical32, "Physical32");
  add(AddressingModelPhysical64, "Physical64");
  add(AddressingModelPhysicalStorageBuffer64, "PhysicalStorageBuffer64");
  add(AddressingModelPhysicalStorageBuffer64EXT, "PhysicalStorageBuffer64EXT");
  add(AddressingModelMax, "Max");
}
SPIRV_DEF_NAMEMAP(AddressingModel, SPIRVAddressingModelNameMap)

template <> inline void SPIRVMap<MemoryModel, std::string>::init() {
  add(MemoryModelSimple, "Simple");
  add(MemoryModelGLSL450, "GLSL450");
  add(MemoryModelOpenCL, "OpenCL");
  add(MemoryModelVulkan, "Vulkan");
  add(MemoryModelVulkanKHR, "VulkanKHR");
  add(MemoryModelMax, "Max");
}
SPIRV_DEF_NAMEMAP(MemoryModel, SPIRVMemoryModelNameMap)

template <> inline void SPIRVMap<ExecutionMode, std::string>::init() {
  add(ExecutionModeInvocations, "Invocations");
  add(ExecutionModeSpacingEqual, "SpacingEqual");
  add(ExecutionModeSpacingFractionalEven, "SpacingFractionalEven");
  add(ExecutionModeSpacingFractionalOdd, "SpacingFractionalOdd");
  add(ExecutionModeVertexOrderCw, "VertexOrderCw");
  add(ExecutionModeVertexOrderCcw, "VertexOrderCcw");
  add(ExecutionModePixelCenterInteger, "PixelCenterInteger");
  add(ExecutionModeOriginUpperLeft, "OriginUpperLeft");
  add(ExecutionModeOriginLowerLeft, "OriginLowerLeft");
  add(ExecutionModeEarlyFragmentTests, "EarlyFragmentTests");
  add(ExecutionModePointMode, "PointMode");
  add(ExecutionModeXfb, "Xfb");
  add(ExecutionModeDepthReplacing, "DepthReplacing");
  add(ExecutionModeDepthGreater, "DepthGreater");
  add(ExecutionModeDepthLess, "DepthLess");
  add(ExecutionModeDepthUnchanged, "DepthUnchanged");
  add(ExecutionModeLocalSize, "LocalSize");
  add(ExecutionModeLocalSizeHint, "LocalSizeHint");
  add(ExecutionModeInputPoints, "InputPoints");
  add(ExecutionModeInputLines, "InputLines");
  add(ExecutionModeInputLinesAdjacency, "InputLinesAdjacency");
  add(ExecutionModeTriangles, "Triangles");
  add(ExecutionModeInputTrianglesAdjacency, "InputTrianglesAdjacency");
  add(ExecutionModeQuads, "Quads");
  add(ExecutionModeIsolines, "Isolines");
  add(ExecutionModeOutputVertices, "OutputVertices");
  add(ExecutionModeOutputPoints, "OutputPoints");
  add(ExecutionModeOutputLineStrip, "OutputLineStrip");
  add(ExecutionModeOutputTriangleStrip, "OutputTriangleStrip");
  add(ExecutionModeVecTypeHint, "VecTypeHint");
  add(ExecutionModeContractionOff, "ContractionOff");
  add(ExecutionModeInitializer, "Initializer");
  add(ExecutionModeFinalizer, "Finalizer");
  add(ExecutionModeSubgroupSize, "SubgroupSize");
  add(ExecutionModeSubgroupsPerWorkgroup, "SubgroupsPerWorkgroup");
  add(ExecutionModeSubgroupsPerWorkgroupId, "SubgroupsPerWorkgroupId");
  add(ExecutionModeLocalSizeId, "LocalSizeId");
  add(ExecutionModeLocalSizeHintId, "LocalSizeHintId");
  add(ExecutionModePostDepthCoverage, "PostDepthCoverage");
  add(ExecutionModeDenormPreserve, "DenormPreserve");
  add(ExecutionModeDenormFlushToZero, "DenormFlushToZero");
  add(ExecutionModeSignedZeroInfNanPreserve, "SignedZeroInfNanPreserve");
  add(ExecutionModeRoundingModeRTE, "RoundingModeRTE");
  add(ExecutionModeRoundingModeRTZ, "RoundingModeRTZ");
  add(ExecutionModeStencilRefReplacingEXT, "StencilRefReplacingEXT");
  add(ExecutionModeOutputLinesNV, "OutputLinesNV");
  add(ExecutionModeOutputPrimitivesNV, "OutputPrimitivesNV");
  add(ExecutionModeDerivativeGroupQuadsNV, "DerivativeGroupQuadsNV");
  add(ExecutionModeDerivativeGroupLinearNV, "DerivativeGroupLinearNV");
  add(ExecutionModeOutputTrianglesNV, "OutputTrianglesNV");
  add(ExecutionModePixelInterlockOrderedEXT, "PixelInterlockOrderedEXT");
  add(ExecutionModePixelInterlockUnorderedEXT, "PixelInterlockUnorderedEXT");
  add(ExecutionModeSampleInterlockOrderedEXT, "SampleInterlockOrderedEXT");
  add(ExecutionModeSampleInterlockUnorderedEXT, "SampleInterlockUnorderedEXT");
  add(ExecutionModeShadingRateInterlockOrderedEXT,
      "ShadingRateInterlockOrderedEXT");
  add(ExecutionModeShadingRateInterlockUnorderedEXT,
      "ShadingRateInterlockUnorderedEXT");
  add(ExecutionModeSharedLocalMemorySizeINTEL, "SharedLocalMemorySizeINTEL");
  add(ExecutionModeRoundingModeRTPINTEL, "RoundingModeRTPINTEL");
  add(ExecutionModeRoundingModeRTNINTEL, "RoundingModeRTNINTEL");
  add(ExecutionModeFloatingPointModeALTINTEL, "FloatingPointModeALTINTEL");
  add(ExecutionModeFloatingPointModeIEEEINTEL, "FloatingPointModeIEEEINTEL");
  add(ExecutionModeMaxWorkgroupSizeINTEL, "MaxWorkgroupSizeINTEL");
  add(ExecutionModeMaxWorkDimINTEL, "MaxWorkDimINTEL");
  add(ExecutionModeNoGlobalOffsetINTEL, "NoGlobalOffsetINTEL");
  add(ExecutionModeNumSIMDWorkitemsINTEL, "NumSIMDWorkitemsINTEL");
  add(ExecutionModeMax, "Max");
}
SPIRV_DEF_NAMEMAP(ExecutionMode, SPIRVExecutionModeNameMap)

template <> inline void SPIRVMap<StorageClass, std::string>::init() {
  add(StorageClassUniformConstant, "UniformConstant");
  add(StorageClassInput, "Input");
  add(StorageClassUniform, "Uniform");
  add(StorageClassOutput, "Output");
  add(StorageClassWorkgroup, "Workgroup");
  add(StorageClassCrossWorkgroup, "CrossWorkgroup");
  add(StorageClassPrivate, "Private");
  add(StorageClassFunction, "Function");
  add(StorageClassGeneric, "Generic");
  add(StorageClassPushConstant, "PushConstant");
  add(StorageClassAtomicCounter, "AtomicCounter");
  add(StorageClassImage, "Image");
  add(StorageClassStorageBuffer, "StorageBuffer");
  add(StorageClassCallableDataNV, "CallableDataNV");
  add(StorageClassIncomingCallableDataNV, "IncomingCallableDataNV");
  add(StorageClassRayPayloadNV, "RayPayloadNV");
  add(StorageClassHitAttributeNV, "HitAttributeNV");
  add(StorageClassIncomingRayPayloadNV, "IncomingRayPayloadNV");
  add(StorageClassShaderRecordBufferNV, "ShaderRecordBufferNV");
  add(StorageClassPhysicalStorageBuffer, "PhysicalStorageBuffer");
  add(StorageClassPhysicalStorageBufferEXT, "PhysicalStorageBufferEXT");
  add(StorageClassDeviceOnlyINTEL, "DeviceOnlyINTEL");
  add(StorageClassHostOnlyINTEL, "HostOnlyINTEL");
  add(StorageClassMax, "Max");
}
SPIRV_DEF_NAMEMAP(StorageClass, SPIRVStorageClassNameMap)

template <> inline void SPIRVMap<Dim, std::string>::init() {
  add(Dim1D, "1D");
  add(Dim2D, "2D");
  add(Dim3D, "3D");
  add(DimCube, "Cube");
  add(DimRect, "Rect");
  add(DimBuffer, "Buffer");
  add(DimSubpassData, "SubpassData");
  add(DimMax, "Max");
}
SPIRV_DEF_NAMEMAP(Dim, SPIRVDimNameMap)

template <> inline void SPIRVMap<SamplerAddressingMode, std::string>::init() {
  add(SamplerAddressingModeNone, "None");
  add(SamplerAddressingModeClampToEdge, "ClampToEdge");
  add(SamplerAddressingModeClamp, "Clamp");
  add(SamplerAddressingModeRepeat, "Repeat");
  add(SamplerAddressingModeRepeatMirrored, "RepeatMirrored");
  add(SamplerAddressingModeMax, "Max");
}
SPIRV_DEF_NAMEMAP(SamplerAddressingMode, SPIRVSamplerAddressingModeNameMap)

template <> inline void SPIRVMap<SamplerFilterMode, std::string>::init() {
  add(SamplerFilterModeNearest, "Nearest");
  add(SamplerFilterModeLinear, "Linear");
  add(SamplerFilterModeMax, "Max");
}
SPIRV_DEF_NAMEMAP(SamplerFilterMode, SPIRVSamplerFilterModeNameMap)

template <> inline void SPIRVMap<ImageFormat, std::string>::init() {
  add(ImageFormatUnknown, "Unknown");
  add(ImageFormatRgba32f, "Rgba32f");
  add(ImageFormatRgba16f, "Rgba16f");
  add(ImageFormatR32f, "R32f");
  add(ImageFormatRgba8, "Rgba8");
  add(ImageFormatRgba8Snorm, "Rgba8Snorm");
  add(ImageFormatRg32f, "Rg32f");
  add(ImageFormatRg16f, "Rg16f");
  add(ImageFormatR11fG11fB10f, "R11fG11fB10f");
  add(ImageFormatR16f, "R16f");
  add(ImageFormatRgba16, "Rgba16");
  add(ImageFormatRgb10A2, "Rgb10A2");
  add(ImageFormatRg16, "Rg16");
  add(ImageFormatRg8, "Rg8");
  add(ImageFormatR16, "R16");
  add(ImageFormatR8, "R8");
  add(ImageFormatRgba16Snorm, "Rgba16Snorm");
  add(ImageFormatRg16Snorm, "Rg16Snorm");
  add(ImageFormatRg8Snorm, "Rg8Snorm");
  add(ImageFormatR16Snorm, "R16Snorm");
  add(ImageFormatR8Snorm, "R8Snorm");
  add(ImageFormatRgba32i, "Rgba32i");
  add(ImageFormatRgba16i, "Rgba16i");
  add(ImageFormatRgba8i, "Rgba8i");
  add(ImageFormatR32i, "R32i");
  add(ImageFormatRg32i, "Rg32i");
  add(ImageFormatRg16i, "Rg16i");
  add(ImageFormatRg8i, "Rg8i");
  add(ImageFormatR16i, "R16i");
  add(ImageFormatR8i, "R8i");
  add(ImageFormatRgba32ui, "Rgba32ui");
  add(ImageFormatRgba16ui, "Rgba16ui");
  add(ImageFormatRgba8ui, "Rgba8ui");
  add(ImageFormatR32ui, "R32ui");
  add(ImageFormatRgb10a2ui, "Rgb10a2ui");
  add(ImageFormatRg32ui, "Rg32ui");
  add(ImageFormatRg16ui, "Rg16ui");
  add(ImageFormatRg8ui, "Rg8ui");
  add(ImageFormatR16ui, "R16ui");
  add(ImageFormatR8ui, "R8ui");
  add(ImageFormatMax, "Max");
}
SPIRV_DEF_NAMEMAP(ImageFormat, SPIRVImageFormatNameMap)

template <> inline void SPIRVMap<ImageChannelOrder, std::string>::init() {
  add(ImageChannelOrderR, "R");
  add(ImageChannelOrderA, "A");
  add(ImageChannelOrderRG, "RG");
  add(ImageChannelOrderRA, "RA");
  add(ImageChannelOrderRGB, "RGB");
  add(ImageChannelOrderRGBA, "RGBA");
  add(ImageChannelOrderBGRA, "BGRA");
  add(ImageChannelOrderARGB, "ARGB");
  add(ImageChannelOrderIntensity, "Intensity");
  add(ImageChannelOrderLuminance, "Luminance");
  add(ImageChannelOrderRx, "Rx");
  add(ImageChannelOrderRGx, "RGx");
  add(ImageChannelOrderRGBx, "RGBx");
  add(ImageChannelOrderDepth, "Depth");
  add(ImageChannelOrderDepthStencil, "DepthStencil");
  add(ImageChannelOrderABGR, "ABGR");
  add(ImageChannelOrderMax, "Max");
}
SPIRV_DEF_NAMEMAP(ImageChannelOrder, SPIRVImageChannelOrderNameMap)

template <> inline void SPIRVMap<ImageChannelDataType, std::string>::init() {
  add(ImageChannelDataTypeSnormInt8, "SnormInt8");
  add(ImageChannelDataTypeSnormInt16, "SnormInt16");
  add(ImageChannelDataTypeUnormInt8, "UnormInt8");
  add(ImageChannelDataTypeUnormInt16, "UnormInt16");
  add(ImageChannelDataTypeUnormShort565, "UnormShort565");
  add(ImageChannelDataTypeUnormShort555, "UnormShort555");
  add(ImageChannelDataTypeUnormInt101010, "UnormInt101010");
  add(ImageChannelDataTypeSignedInt8, "SignedInt8");
  add(ImageChannelDataTypeSignedInt16, "SignedInt16");
  add(ImageChannelDataTypeSignedInt32, "SignedInt32");
  add(ImageChannelDataTypeUnsignedInt8, "UnsignedInt8");
  add(ImageChannelDataTypeUnsignedInt16, "UnsignedInt16");
  add(ImageChannelDataTypeUnsignedInt32, "UnsignedInt32");
  add(ImageChannelDataTypeHalfFloat, "HalfFloat");
  add(ImageChannelDataTypeFloat, "Float");
  add(ImageChannelDataTypeUnormInt24, "UnormInt24");
  add(ImageChannelDataTypeUnormInt101010_2, "UnormInt101010_2");
  add(ImageChannelDataTypeMax, "Max");
}
SPIRV_DEF_NAMEMAP(ImageChannelDataType, SPIRVImageChannelDataTypeNameMap)

template <> inline void SPIRVMap<FPRoundingMode, std::string>::init() {
  add(FPRoundingModeRTE, "RTE");
  add(FPRoundingModeRTZ, "RTZ");
  add(FPRoundingModeRTP, "RTP");
  add(FPRoundingModeRTN, "RTN");
  add(FPRoundingModeMax, "Max");
}
SPIRV_DEF_NAMEMAP(FPRoundingMode, SPIRVFPRoundingModeNameMap)

template <> inline void SPIRVMap<LinkageType, std::string>::init() {
  add(LinkageTypeExport, "Export");
  add(LinkageTypeImport, "Import");
  add(LinkageTypeInternal, "Internal");
  add(LinkageTypeMax, "Max");
}
SPIRV_DEF_NAMEMAP(LinkageType, SPIRVLinkageTypeNameMap)

template <> inline void SPIRVMap<AccessQualifier, std::string>::init() {
  add(AccessQualifierReadOnly, "ReadOnly");
  add(AccessQualifierWriteOnly, "WriteOnly");
  add(AccessQualifierReadWrite, "ReadWrite");
  add(AccessQualifierMax, "Max");
}
SPIRV_DEF_NAMEMAP(AccessQualifier, SPIRVAccessQualifierNameMap)

template <>
inline void SPIRVMap<FunctionParameterAttribute, std::string>::init() {
  add(FunctionParameterAttributeZext, "Zext");
  add(FunctionParameterAttributeSext, "Sext");
  add(FunctionParameterAttributeByVal, "ByVal");
  add(FunctionParameterAttributeSret, "Sret");
  add(FunctionParameterAttributeNoAlias, "NoAlias");
  add(FunctionParameterAttributeNoCapture, "NoCapture");
  add(FunctionParameterAttributeNoWrite, "NoWrite");
  add(FunctionParameterAttributeNoReadWrite, "NoReadWrite");
  add(FunctionParameterAttributeMax, "Max");
}
SPIRV_DEF_NAMEMAP(FunctionParameterAttribute,
                  SPIRVFunctionParameterAttributeNameMap)

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
  add(DecorationSideEffectsINTEL, "SideEffectsINTEL");
  add(DecorationVectorComputeVariableINTEL, "VectorComputeVariableINTEL");
  add(DecorationFuncParamIOKind, "FuncParamIOKind");
  add(DecorationVectorComputeFunctionINTEL, "VectorComputeFunctionINTEL");
  add(DecorationStackCallINTEL, "StackCallINTEL");
  add(DecorationGlobalVariableOffsetINTEL, "GlobalVariableOffsetINTEL");
  add(DecorationCounterBuffer, "CounterBuffer");
  add(DecorationHlslCounterBufferGOOGLE, "HlslCounterBufferGOOGLE");
  add(DecorationHlslSemanticGOOGLE, "HlslSemanticGOOGLE");
  add(DecorationUserSemantic, "UserSemantic");
  add(DecorationUserTypeGOOGLE, "UserTypeGOOGLE");
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
  add(DecorationFuncParamKindINTEL, "FuncParamKindINTEL");
  add(DecorationFuncParamDescINTEL, "FuncParamDescINTEL");
  add(DecorationBufferLocationINTEL, "BufferLocationINTEL");
  add(DecorationIOPipeStorageINTEL, "IOPipeStorageINTEL");
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
  add(BuiltInDeviceIndex, "BuiltInDeviceIndex");
  add(BuiltInViewIndex, "BuiltInViewIndex");
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
  add(BuiltInLaunchIdNV, "BuiltInLaunchIdNV");
  add(BuiltInLaunchSizeNV, "BuiltInLaunchSizeNV");
  add(BuiltInWorldRayOriginNV, "BuiltInWorldRayOriginNV");
  add(BuiltInWorldRayDirectionNV, "BuiltInWorldRayDirectionNV");
  add(BuiltInObjectRayOriginNV, "BuiltInObjectRayOriginNV");
  add(BuiltInObjectRayDirectionNV, "BuiltInObjectRayDirectionNV");
  add(BuiltInRayTminNV, "BuiltInRayTminNV");
  add(BuiltInRayTmaxNV, "BuiltInRayTmaxNV");
  add(BuiltInInstanceCustomIndexNV, "BuiltInInstanceCustomIndexNV");
  add(BuiltInObjectToWorldNV, "BuiltInObjectToWorldNV");
  add(BuiltInWorldToObjectNV, "BuiltInWorldToObjectNV");
  add(BuiltInHitTNV, "BuiltInHitTNV");
  add(BuiltInHitKindNV, "BuiltInHitKindNV");
  add(BuiltInIncomingRayFlagsNV, "BuiltInIncomingRayFlagsNV");
  add(BuiltInWarpsPerSMNV, "BuiltInWarpsPerSMNV");
  add(BuiltInSMCountNV, "BuiltInSMCountNV");
  add(BuiltInWarpIDNV, "BuiltInWarpIDNV");
  add(BuiltInSMIDNV, "BuiltInSMIDNV");
  add(BuiltInMax, "BuiltInMax");
}
SPIRV_DEF_NAMEMAP(BuiltIn, SPIRVBuiltInNameMap)

template <> inline void SPIRVMap<Scope, std::string>::init() {
  add(ScopeCrossDevice, "CrossDevice");
  add(ScopeDevice, "Device");
  add(ScopeWorkgroup, "Workgroup");
  add(ScopeSubgroup, "Subgroup");
  add(ScopeInvocation, "Invocation");
  add(ScopeQueueFamily, "QueueFamily");
  add(ScopeQueueFamilyKHR, "QueueFamilyKHR");
  add(ScopeMax, "Max");
}
SPIRV_DEF_NAMEMAP(Scope, SPIRVScopeNameMap)

template <> inline void SPIRVMap<GroupOperation, std::string>::init() {
  add(GroupOperationReduce, "Reduce");
  add(GroupOperationInclusiveScan, "InclusiveScan");
  add(GroupOperationExclusiveScan, "ExclusiveScan");
  add(GroupOperationClusteredReduce, "ClusteredReduce");
  add(GroupOperationPartitionedReduceNV, "PartitionedReduceNV");
  add(GroupOperationPartitionedInclusiveScanNV, "PartitionedInclusiveScanNV");
  add(GroupOperationPartitionedExclusiveScanNV, "PartitionedExclusiveScanNV");
  add(GroupOperationMax, "Max");
}
SPIRV_DEF_NAMEMAP(GroupOperation, SPIRVGroupOperationNameMap)

template <> inline void SPIRVMap<KernelEnqueueFlags, std::string>::init() {
  add(KernelEnqueueFlagsNoWait, "NoWait");
  add(KernelEnqueueFlagsWaitKernel, "WaitKernel");
  add(KernelEnqueueFlagsWaitWorkGroup, "WaitWorkGroup");
  add(KernelEnqueueFlagsMax, "Max");
}
SPIRV_DEF_NAMEMAP(KernelEnqueueFlags, SPIRVKernelEnqueueFlagsNameMap)

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
  add(CapabilitySubgroupBallotKHR, "SubgroupBallotKHR");
  add(CapabilityDrawParameters, "DrawParameters");
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
  add(CapabilityFloat16ImageAMD, "Float16ImageAMD");
  add(CapabilityImageGatherBiasLodAMD, "ImageGatherBiasLodAMD");
  add(CapabilityFragmentMaskAMD, "FragmentMaskAMD");
  add(CapabilityStencilExportEXT, "StencilExportEXT");
  add(CapabilityImageReadWriteLodAMD, "ImageReadWriteLodAMD");
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
  add(CapabilityIntegerFunctions2INTEL, "IntegerFunctions2INTEL");
  add(CapabilityFunctionPointersINTEL, "FunctionPointersINTEL");
  add(CapabilityIndirectReferencesINTEL, "IndirectReferencesINTEL");
  add(CapabilityAsmINTEL, "AsmINTEL");
  add(CapabilityVectorComputeINTEL, "VectorComputeINTEL");
  add(CapabilityVectorAnyINTEL, "VectorAnyINTEL");
  add(CapabilityOptimizationHintsINTEL, "OptimizationHintsINTEL");
  add(CapabilitySubgroupAvcMotionEstimationINTEL,
      "SubgroupAvcMotionEstimationINTEL");
  add(CapabilitySubgroupAvcMotionEstimationIntraINTEL,
      "SubgroupAvcMotionEstimationIntraINTEL");
  add(CapabilitySubgroupAvcMotionEstimationChromaINTEL,
      "SubgroupAvcMotionEstimationChromaINTEL");
  add(CapabilityRoundToInfinityINTEL, "RoundToInfinityINTEL");
  add(CapabilityFloatingPointModeINTEL, "FloatingPointModeINTEL");
  add(CapabilityFPGAMemoryAttributesINTEL, "FPGAMemoryAttributesINTEL");
  add(CapabilityArbitraryPrecisionIntegersINTEL,
      "ArbitraryPrecisionIntegersINTEL");
  add(CapabilityArbitraryPrecisionFloatingPointINTEL,
      "ArbitraryPrecisionFloatingPointINTEL");
  add(CapabilityUnstructuredLoopControlsINTEL, "UnstructuredLoopControlsINTEL");
  add(CapabilityFPGALoopControlsINTEL, "FPGALoopControlsINTEL");
  add(CapabilityBlockingPipesINTEL, "BlockingPipesINTEL");
  add(CapabilityFPGARegINTEL, "FPGARegINTEL");
  add(CapabilityKernelAttributesINTEL, "KernelAttributesINTEL");
  add(CapabilityFPGAKernelAttributesINTEL, "FPGAKernelAttributesINTEL");
  add(CapabilityFPGABufferLocationINTEL, "FPGABufferLocationINTEL");
  add(CapabilityArbitraryPrecisionFixedPointINTEL,
      "ArbitraryPrecisionFixedPointINTEL");
  add(CapabilityUSMStorageClassesINTEL, "USMStorageClassesINTEL");
  add(CapabilityFPGAMemoryAccessesINTEL, "FPGAMemoryAccessesINTEL");
  add(CapabilityIOPipeINTEL, "IOPipeINTEL");
  add(CapabilityMax, "Max");
}
SPIRV_DEF_NAMEMAP(Capability, SPIRVCapabilityNameMap)

} /* namespace SPIRV */

#endif // SPIRV_LIBSPIRV_SPIRVNAMEMAPENUM_H
