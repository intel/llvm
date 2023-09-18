//===--------- program.cpp - Native CPU Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"

#include "common.hpp"
#include "program.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urProgramCreateWithIL(ur_context_handle_t hContext, const void *pIL,
                      size_t length, const ur_program_properties_t *pProperties,
                      ur_program_handle_t *phProgram) {
  std::ignore = hContext;
  std::ignore = pIL;
  std::ignore = length;
  std::ignore = pProperties;
  std::ignore = phProgram;

  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithBinary(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    const uint8_t *pBinary, const ur_program_properties_t *pProperties,
    ur_program_handle_t *phProgram) {
  std::ignore = size;
  std::ignore = pProperties;

  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phProgram, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(pBinary != nullptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  auto hProgram = new ur_program_handle_t_(
      hContext, reinterpret_cast<const unsigned char *>(pBinary));

  const nativecpu_entry *nativecpu_it =
      reinterpret_cast<const nativecpu_entry *>(pBinary);
  while (nativecpu_it->kernel_ptr != nullptr) {
    hProgram->_kernels.insert(
        std::make_pair(nativecpu_it->kernelname, nativecpu_it->kernel_ptr));
    nativecpu_it++;
  }

  *phProgram = hProgram;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramBuild(ur_context_handle_t hContext,
                                                   ur_program_handle_t hProgram,
                                                   const char *pOptions) {
  std::ignore = hContext;
  std::ignore = hProgram;
  std::ignore = pOptions;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramCompile(ur_context_handle_t hContext, ur_program_handle_t hProgram,
                 const char *pOptions) {
  std::ignore = hContext;
  std::ignore = hProgram;
  std::ignore = pOptions;

  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramLink(ur_context_handle_t hContext, uint32_t count,
              const ur_program_handle_t *phPrograms, const char *pOptions,
              ur_program_handle_t *phProgram) {
  std::ignore = hContext;
  std::ignore = count;
  std::ignore = phPrograms;
  std::ignore = pOptions;
  std::ignore = phProgram;

  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRetain(ur_program_handle_t hProgram) {
  hProgram->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramRelease(ur_program_handle_t hProgram) {
  delete hProgram;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetFunctionPointer(
    ur_device_handle_t hDevice, ur_program_handle_t hProgram,
    const char *pFunctionName, void **ppFunctionPointer) {
  std::ignore = hDevice;
  std::ignore = hProgram;
  std::ignore = pFunctionName;
  std::ignore = ppFunctionPointer;

  DIE_NO_IMPLEMENTATION
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
  case UR_PROGRAM_INFO_SOURCE:
    return returnValue(nullptr);
  case UR_PROGRAM_INFO_BINARY_SIZES:
    return returnValue("foo");
  case UR_PROGRAM_INFO_BINARIES:
    return returnValue("foo");
  case UR_PROGRAM_INFO_KERNEL_NAMES: {
    return returnValue("foo");
  }
  default:
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urProgramGetBuildInfo(ur_program_handle_t hProgram, ur_device_handle_t hDevice,
                      ur_program_build_info_t propName, size_t propSize,
                      void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hProgram;
  std::ignore = hDevice;
  std::ignore = propName;
  std::ignore = propSize;
  std::ignore = pPropValue;
  std::ignore = pPropSizeRet;

  CONTINUE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramSetSpecializationConstants(
    ur_program_handle_t hProgram, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants) {
  std::ignore = hProgram;
  std::ignore = count;
  std::ignore = pSpecConstants;

  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramGetNativeHandle(
    ur_program_handle_t hProgram, ur_native_handle_t *phNativeProgram) {
  std::ignore = hProgram;
  std::ignore = phNativeProgram;

  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL urProgramCreateWithNativeHandle(
    ur_native_handle_t hNativeProgram, ur_context_handle_t hContext,
    const ur_program_native_properties_t *pProperties,
    ur_program_handle_t *phProgram) {
  std::ignore = hNativeProgram;
  std::ignore = hContext;
  std::ignore = pProperties;
  std::ignore = phProgram;

  DIE_NO_IMPLEMENTATION
}
