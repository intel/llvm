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
#include "spirv/unified1/spirv.hpp"
#include "spirv_internal.hpp"

using namespace spv;

namespace SPIRV {

inline bool isValid(spv::ExecutionModel V) {
  switch (static_cast<uint32_t>(V)) {
  case ExecutionModelVertex:
  case ExecutionModelTessellationControl:
  case ExecutionModelTessellationEvaluation:
  case ExecutionModelGeometry:
  case ExecutionModelFragment:
  case ExecutionModelGLCompute:
  case ExecutionModelKernel:
  case ExecutionModelTaskNV:
  case ExecutionModelMeshNV:
  case ExecutionModelRayGenerationKHR:
  case ExecutionModelIntersectionKHR:
  case ExecutionModelAnyHitKHR:
  case ExecutionModelClosestHitKHR:
  case ExecutionModelMissKHR:
  case ExecutionModelCallableKHR:
  case ExecutionModeRegisterMapInterfaceINTEL:
  case ExecutionModeStreamingInterfaceINTEL:
  case ExecutionModeMaximumRegistersINTEL:
  case ExecutionModeMaximumRegistersIdINTEL:
  case ExecutionModeNamedMaximumRegistersINTEL:
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
  case AddressingModelPhysicalStorageBuffer64:
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
  case MemoryModelVulkan:
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
  case StorageClassStorageBuffer:
  case StorageClassCallableDataKHR:
  case StorageClassIncomingCallableDataKHR:
  case StorageClassRayPayloadKHR:
  case StorageClassHitAttributeKHR:
  case StorageClassIncomingRayPayloadKHR:
  case StorageClassShaderRecordBufferKHR:
  case StorageClassPhysicalStorageBuffer:
  case StorageClassCodeSectionINTEL:
  case StorageClassDeviceOnlyINTEL:
  case StorageClassHostOnlyINTEL:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::LinkageType V) {
  int LT = V;
  switch (LT) {
  case LinkageTypeExport:
  case LinkageTypeImport:
  case LinkageTypeLinkOnceODR:
  case internal::LinkageTypeInternal:
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
  case FunctionParameterAttributeRuntimeAlignedINTEL:
    return true;
  default:
    return false;
  }
}

inline bool isValid(spv::BuiltIn V) {
  switch (static_cast<uint32_t>(V)) {
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
  case BuiltInVertexIndex:
  case BuiltInInstanceIndex:
  case BuiltInSubgroupEqMask:
  case BuiltInSubgroupGeMask:
  case BuiltInSubgroupGtMask:
  case BuiltInSubgroupLeMask:
  case BuiltInSubgroupLtMask:
  case BuiltInBaseVertex:
  case BuiltInBaseInstance:
  case BuiltInDrawIndex:
  case BuiltInPrimitiveShadingRateKHR:
  case BuiltInDeviceIndex:
  case BuiltInViewIndex:
  case BuiltInShadingRateKHR:
  case BuiltInBaryCoordNoPerspAMD:
  case BuiltInBaryCoordNoPerspCentroidAMD:
  case BuiltInBaryCoordNoPerspSampleAMD:
  case BuiltInBaryCoordSmoothAMD:
  case BuiltInBaryCoordSmoothCentroidAMD:
  case BuiltInBaryCoordSmoothSampleAMD:
  case BuiltInBaryCoordPullModelAMD:
  case BuiltInFragStencilRefEXT:
  case BuiltInViewportMaskNV:
  case BuiltInSecondaryPositionNV:
  case BuiltInSecondaryViewportMaskNV:
  case BuiltInPositionPerViewNV:
  case BuiltInViewportMaskPerViewNV:
  case BuiltInFullyCoveredEXT:
  case BuiltInTaskCountNV:
  case BuiltInPrimitiveCountNV:
  case BuiltInPrimitiveIndicesNV:
  case BuiltInClipDistancePerViewNV:
  case BuiltInCullDistancePerViewNV:
  case BuiltInLayerPerViewNV:
  case BuiltInMeshViewCountNV:
  case BuiltInMeshViewIndicesNV:
  case BuiltInBaryCoordKHR:
  case BuiltInBaryCoordNoPerspKHR:
  case BuiltInFragSizeEXT:
  case BuiltInFragInvocationCountEXT:
  case BuiltInLaunchIdKHR:
  case BuiltInLaunchSizeKHR:
  case BuiltInWorldRayOriginKHR:
  case BuiltInWorldRayDirectionKHR:
  case BuiltInObjectRayOriginKHR:
  case BuiltInObjectRayDirectionKHR:
  case BuiltInRayTminKHR:
  case BuiltInRayTmaxKHR:
  case BuiltInInstanceCustomIndexKHR:
  case BuiltInObjectToWorldKHR:
  case BuiltInWorldToObjectKHR:
  case BuiltInHitTNV:
  case BuiltInHitKindKHR:
  case BuiltInCurrentRayTimeNV:
  case BuiltInIncomingRayFlagsKHR:
  case BuiltInRayGeometryIndexKHR:
  case BuiltInWarpsPerSMNV:
  case BuiltInSMCountNV:
  case BuiltInWarpIDNV:
  case BuiltInSMIDNV:
  case BuiltInCullMaskKHR:
  case internal::BuiltInSubDeviceIDINTEL:
  case internal::BuiltInGlobalHWThreadIDINTEL:
    return true;
  default:
    return false;
  }
}

inline bool isValidFunctionControlMask(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;
  ValidMask |= FunctionControlInlineMask;
  ValidMask |= FunctionControlDontInlineMask;
  ValidMask |= FunctionControlPureMask;
  ValidMask |= FunctionControlConstMask;
  ValidMask |= internal::FunctionControlOptNoneINTELMask;

  return (Mask & ~ValidMask) == 0;
}

} /* namespace SPIRV */

#endif // SPIRV_LIBSPIRV_SPIRVISVALIDENUM_H
