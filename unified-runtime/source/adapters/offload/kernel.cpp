//===----------- kernel.cpp - LLVM Offload Adapter  -----------------------===//
//
// Copyright (C) 2025 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.hpp"
#include "memory.hpp"
#include "program.hpp"
#include "ur2offload.hpp"
#include <OffloadAPI.h>
#include <ur/ur.hpp>
#include <ur_api.h>

UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(ur_program_handle_t hProgram, const char *pKernelName,
               ur_kernel_handle_t *phKernel) {
  ur_kernel_handle_t Kernel = new ur_kernel_handle_t_;

  auto Res = olGetSymbol(hProgram->OffloadProgram, pKernelName,
                         OL_SYMBOL_KIND_KERNEL, &Kernel->OffloadKernel);

  if (Res != OL_SUCCESS) {
    delete Kernel;
    return offloadResultToUR(Res);
  }

  Kernel->Name = pKernelName;
  Kernel->Program = hProgram;

  *phKernel = Kernel;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetInfo(ur_kernel_handle_t hKernel,
                                                    ur_kernel_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_KERNEL_INFO_REFERENCE_COUNT:
    return ReturnValue(hKernel->RefCount.load());
  case UR_KERNEL_INFO_FUNCTION_NAME:
    return ReturnValue(hKernel->Name.c_str());
  case UR_KERNEL_INFO_PROGRAM:
    return ReturnValue(hKernel->Program);
  case UR_KERNEL_INFO_CONTEXT:
    return ReturnValue(hKernel->Program->URContext);
  case UR_KERNEL_INFO_ATTRIBUTES:
    return ReturnValue("");
  case UR_KERNEL_INFO_NUM_ARGS:
    // This is unimplementable on liboffload (and AMD/Nvidia in general)
    [[fallthrough]];
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelRetain(ur_kernel_handle_t hKernel) {
  hKernel->RefCount++;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelRelease(ur_kernel_handle_t hKernel) {
  if (--hKernel->RefCount == 0) {
    delete hKernel;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetExecInfo(ur_kernel_handle_t, ur_kernel_exec_info_t, size_t,
                    const ur_kernel_exec_info_properties_t *, const void *) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, uint32_t argIndex,
    const ur_kernel_arg_pointer_properties_t *, const void *pArgValue) {
  hKernel->Args.addArg(argIndex, sizeof(pArgValue), &pArgValue);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_value_properties_t *, const void *pArgValue) {
  hKernel->Args.addArg(argIndex, argSize, pArgValue);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgMemObj(ur_kernel_handle_t hKernel, uint32_t argIndex,
                     const ur_kernel_arg_mem_obj_properties_t *Properties,
                     ur_mem_handle_t hArgValue) {
  // Handle zero-sized buffers
  if (hArgValue == nullptr) {
    hKernel->Args.addArg(argIndex, 0, nullptr);
    return UR_RESULT_SUCCESS;
  }

  ur_mem_flags_t MemAccess =
      Properties ? Properties->memoryAccess
                 : static_cast<ur_mem_flags_t>(UR_MEM_FLAG_READ_WRITE);
  hKernel->Args.addMemObjArg(argIndex, hArgValue, MemAccess);

  auto Ptr = std::get<BufferMem>(hArgValue->Mem).Ptr;
  hKernel->Args.addArg(argIndex, sizeof(void *), &Ptr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetGroupInfo(
    ur_kernel_handle_t, ur_device_handle_t, ur_kernel_group_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  if (propName == UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE) {
    size_t GroupSize[3] = {0, 0, 0};
    return ReturnValue(GroupSize, 3);
  }
  return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetSubGroupInfo(
    ur_kernel_handle_t, ur_device_handle_t, ur_kernel_sub_group_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  (void)propName;

  return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetNativeHandle(ur_kernel_handle_t, ur_native_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t, ur_context_handle_t, ur_program_handle_t,
    const ur_kernel_native_properties_t *, ur_kernel_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    ur_kernel_handle_t, uint32_t, const ur_specialization_constant_info_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetSuggestedLocalWorkSize(
    ur_kernel_handle_t, ur_queue_handle_t, uint32_t, const size_t *,
    const size_t *, size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
