//===--------- program.cpp - Native CPU Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"

#include "common.hpp"
#include "common/ur_util.hpp"
#include "program.hpp"
#include <cstdint>
#include <memory>

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithIL(
    ur_context_handle_t /*hContext*/, const void * /*pIL*/, size_t /*length*/,
    const ur_program_properties_t * /*pProperties*/,
    ur_program_handle_t * /*phProgram*/) {

  DIE_NO_IMPLEMENTATION;
}

static ur_result_t
deserializeWGMetadata(const ur_program_metadata_t &MetadataElement,
                      native_cpu::WGSize_t &res, std::uint32_t DefaultVal) {
  size_t MDElemsSize = MetadataElement.size - sizeof(std::uint64_t);

  // Expect between 1 and 3 32-bit integer values.
  UR_ASSERT(MDElemsSize == sizeof(std::uint32_t) ||
                MDElemsSize == sizeof(std::uint32_t) * 2 ||
                MDElemsSize == sizeof(std::uint32_t) * 3,
            UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE);

  // Get pointer to data, skipping 64-bit size at the start of the data.
  const char *ValuePtr =
      reinterpret_cast<const char *>(MetadataElement.value.pData) +
      sizeof(std::uint64_t);
  // Read values and pad with a default value for missing elements.
  std::uint32_t WorkGroupElements[] = {DefaultVal, DefaultVal, DefaultVal};
  std::memcpy(WorkGroupElements, ValuePtr, MDElemsSize);
  std::get<0>(res) = WorkGroupElements[0];
  std::get<1>(res) = WorkGroupElements[1];
  std::get<2>(res) = WorkGroupElements[2];
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, uint32_t numDevices,
    ur_device_handle_t *phDevices, size_t * /*pLengths*/,
    const uint8_t **ppBinaries, const ur_program_properties_t *pProperties,
    ur_program_handle_t *phProgram) {
  if (numDevices > 1)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  auto hDevice = phDevices[0];
  auto pBinary = ppBinaries[0];

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phProgram, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pBinary != nullptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  auto hProgram = std::make_unique<ur_program_handle_t_>(
      hContext, reinterpret_cast<const unsigned char *>(pBinary));
  if (pProperties != nullptr) {
    for (uint32_t i = 0; i < pProperties->count; i++) {
      const auto &mdNode = pProperties->pMetadatas[i];
      std::string mdName(mdNode.pName);
      auto [Prefix, Tag] = splitMetadataName(mdName);
      if (Tag == __SYCL_UR_PROGRAM_METADATA_TAG_REQD_WORK_GROUP_SIZE ||
          Tag == __SYCL_UR_PROGRAM_METADATA_TAG_MAX_WORK_GROUP_SIZE) {
        bool isReqd =
            Tag == __SYCL_UR_PROGRAM_METADATA_TAG_REQD_WORK_GROUP_SIZE;
        native_cpu::WGSize_t wgSizeProp;
        auto res = deserializeWGMetadata(
            mdNode, wgSizeProp,
            isReqd ? 1 : std::numeric_limits<std::uint32_t>::max());
        if (res != UR_RESULT_SUCCESS) {
          return res;
        }
        (isReqd ? hProgram->KernelReqdWorkGroupSizeMD
                : hProgram->KernelMaxWorkGroupSizeMD)[Prefix] =
            std::move(wgSizeProp);
      } else if (Tag ==
                 __SYCL_UR_PROGRAM_METADATA_TAG_MAX_LINEAR_WORK_GROUP_SIZE) {
        hProgram->KernelMaxLinearWorkGroupSizeMD[Prefix] = mdNode.value.data64;
      }
    }
  }

  const nativecpu_entry *nativecpu_it =
      reinterpret_cast<const nativecpu_entry *>(pBinary);
  while (nativecpu_it->kernel_ptr != nullptr) {
    hProgram->_kernels.insert(
        std::make_pair(nativecpu_it->kernelname, nativecpu_it->kernel_ptr));
    nativecpu_it++;
  }

  *phProgram = hProgram.release();

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinaryExp(
    ur_context_handle_t, uint32_t, ur_device_handle_t *, size_t *,
    const uint8_t **, const ur_program_properties_t *, ur_program_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramBuild(ur_context_handle_t /*hContext*/,
               ur_program_handle_t /*hProgram*/, const char * /*pOptions*/) {

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile(ur_context_handle_t /*hContext*/,
                 ur_program_handle_t /*hProgram*/, const char * /*pOptions*/) {

  // Currently for Native CPU the program is offline compiled, so
  // urProgramCompile is a no-op.
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramLink(ur_context_handle_t /*hContext*/, uint32_t /*count*/,
              const ur_program_handle_t * /*phPrograms*/,
              const char * /*pOptions*/, ur_program_handle_t *phProgram) {
  if (nullptr != phProgram) {
    *phProgram = nullptr;
  }

  // Currently for Native CPU the program is already linked and all its
  // symbols are resolved, so this is a no-op.
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCompileExp(ur_program_handle_t,
                                                        uint32_t,
                                                        ur_device_handle_t *,
                                                        const char *) {
  // Currently for Native CPU the program is offline compiled, so
  // urProgramCompile is a no-op.
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuildExp(ur_program_handle_t,
                                                      uint32_t,
                                                      ur_device_handle_t *,
                                                      const char *) {
  // Currently for Native CPU the program is offline compiled and linked,
  // so urProgramBuild is a no-op.
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramLinkExp(
    ur_context_handle_t, uint32_t, ur_device_handle_t *, uint32_t,
    const ur_program_handle_t *, const char *, ur_program_handle_t *phProgram) {
  if (nullptr != phProgram) {
    *phProgram = nullptr;
  }
  // Currently for Native CPU the program is already linked and all its
  // symbols are resolved, so this is a no-op.
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRetain(ur_program_handle_t hProgram) {
  hProgram->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(ur_program_handle_t hProgram) {
  decrementOrDelete(hProgram);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetFunctionPointer(
    ur_device_handle_t /*hDevice*/, ur_program_handle_t /*hProgram*/,
    const char * /*pFunctionName*/, void ** /*ppFunctionPointer*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetGlobalVariablePointer(
    ur_device_handle_t, ur_program_handle_t /*hProgram*/,
    const char * /*pGlobalVariableName*/, size_t * /*pGlobalVariableSizeRet*/,
    void ** /*ppGlobalVariablePointerRet*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetInfo(ur_program_handle_t hProgram, ur_program_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper returnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_PROGRAM_INFO_REFERENCE_COUNT:
    return returnValue(hProgram->getReferenceCount());
  case UR_PROGRAM_INFO_CONTEXT:
    return returnValue(nullptr);
  case UR_PROGRAM_INFO_NUM_DEVICES:
    return returnValue(1u);
  case UR_PROGRAM_INFO_DEVICES:
    return returnValue(hProgram->_ctx->_device);
  case UR_PROGRAM_INFO_BINARY_SIZES:
    return returnValue("foo");
  case UR_PROGRAM_INFO_BINARIES:
    return returnValue("foo");
  case UR_PROGRAM_INFO_KERNEL_NAMES: {
    return returnValue("foo");
  }
  case UR_PROGRAM_INFO_IL:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  default:
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetBuildInfo(
    ur_program_handle_t /*hProgram*/, ur_device_handle_t /*hDevice*/,
    ur_program_build_info_t /*propName*/, size_t /*propSize*/,
    void * /*pPropValue*/, size_t * /*pPropSizeRet*/) {

  CONTINUE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t /*hProgram*/, uint32_t /*count*/,
    const ur_specialization_constant_info_t * /*pSpecConstants*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetNativeHandle(ur_program_handle_t /*hProgram*/,
                         ur_native_handle_t * /*phNativeProgram*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t /*hNativeProgram*/, ur_context_handle_t /*hContext*/,
    const ur_program_native_properties_t * /*pProperties*/,
    ur_program_handle_t * /*phProgram*/) {

  DIE_NO_IMPLEMENTATION;
}
