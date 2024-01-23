//===--------------- kernel.cpp - Native CPU Adapter ----------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur_api.h"
#include "ur_util.hpp"

#include "common.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "program.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(ur_program_handle_t hProgram, const char *pKernelName,
               ur_kernel_handle_t *phKernel) {
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pKernelName, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  auto kernelEntry = hProgram->_kernels.find(pKernelName);
  if (kernelEntry == hProgram->_kernels.end())
    return UR_RESULT_ERROR_INVALID_KERNEL;

  auto f = reinterpret_cast<nativecpu_ptr_t>(
      const_cast<unsigned char *>(kernelEntry->second));
  auto kernel = new ur_kernel_handle_t_(pKernelName, *f);

  *phKernel = kernel;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_value_properties_t *pProperties,
    const void *pArgValue) {
  // Todo: error checking
  // Todo: I think that the opencl spec (and therefore the pi spec mandates that
  // arg is copied (this is why it is defined as const void*, I guess we should
  // do it
  // TODO: can args arrive out of order?
  std::ignore = argIndex;
  std::ignore = pProperties;

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(argSize, UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);

  hKernel->_args.emplace_back(const_cast<void *>(pArgValue));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgLocal(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_local_properties_t *pProperties) {
  std::ignore = pProperties;
  // emplace a placeholder kernel arg, gets replaced with a pointer to the
  // memory pool before enqueueing the kernel.
  hKernel->_args.emplace_back(nullptr);
  hKernel->_localArgInfo.emplace_back(argIndex, argSize);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetInfo(ur_kernel_handle_t hKernel,
                                                    ur_kernel_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {
  std::ignore = hKernel;
  std::ignore = propName;
  std::ignore = pPropValue;

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  // todo: check if we need this
  // std::shared_lock<ur_shared_mutex> Guard(hKernel->Mutex);
  switch (propName) {
    //  case UR_KERNEL_INFO_CONTEXT:
    //    return ReturnValue(ur_context_handle_t{ hKernel->Program->Context });
    //  case UR_KERNEL_INFO_PROGRAM:
    //    return ReturnValue(ur_program_handle_t{ Kernel->Program });
  case UR_KERNEL_INFO_FUNCTION_NAME:
    if (hKernel->_name) {
      return ReturnValue(hKernel->_name);
    }
    return UR_RESULT_ERROR_INVALID_FUNCTION_NAME;
    //  case UR_KERNEL_INFO_NUM_ARGS:
    //    return ReturnValue(uint32_t{ Kernel->ZeKernelProperties->numKernelArgs
    //    });
  case UR_KERNEL_INFO_REFERENCE_COUNT:
    return ReturnValue(uint32_t{hKernel->getReferenceCount()});
  case UR_KERNEL_INFO_ATTRIBUTES:
    return ReturnValue("");
  default:
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetGroupInfo(ur_kernel_handle_t hKernel, ur_device_handle_t hDevice,
                     ur_kernel_group_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hDevice;

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper returnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
    size_t global_work_size[3] = {0, 0, 0};
    return returnValue(global_work_size, 3);
  }
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    // todo: set proper values
    size_t max_threads = 128;
    return returnValue(max_threads);
  }
  case UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
    size_t group_size[3] = {1, 1, 1};
    return returnValue(group_size, 3);
  }
  case UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
    int bytes = 0;
    return returnValue(static_cast<uint64_t>(bytes));
  }
  case UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    // todo: set proper values
    int warpSize = 16;
    return returnValue(static_cast<size_t>(warpSize));
  }
  case UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
    int bytes = 0;
    return returnValue(static_cast<uint64_t>(bytes));
  }

  default:
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetSubGroupInfo(ur_kernel_handle_t hKernel, ur_device_handle_t hDevice,
                        ur_kernel_sub_group_info_t propName, size_t propSize,
                        void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hKernel;
  std::ignore = hDevice;

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE: {
    // todo: set proper values
    int WarpSize = 8;
    return ReturnValue(static_cast<uint32_t>(WarpSize));
  }
  case UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS: {
    // todo: set proper values
    int MaxWarps = 32;
    return ReturnValue(static_cast<uint32_t>(MaxWarps));
  }
  case UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS: {
    // todo: set proper values
    return ReturnValue(0);
  }
  case UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL: {
    // todo: set proper values
    return ReturnValue(0);
  }
  case UR_KERNEL_SUB_GROUP_INFO_FORCE_UINT32: {
    ur::unreachable();
  }
  }
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelRetain(ur_kernel_handle_t hKernel) {
  hKernel->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelRelease(ur_kernel_handle_t hKernel) {
  decrementOrDelete(hKernel);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgPointer(ur_kernel_handle_t hKernel, uint32_t argIndex,
                      const ur_kernel_arg_pointer_properties_t *pProperties,
                      const void *pArgValue) {
  // TODO: out_of_order args?
  std::ignore = argIndex;
  std::ignore = pProperties;

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pArgValue, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  auto ptrToPtr = reinterpret_cast<const intptr_t *>(pArgValue);
  auto derefPtr = reinterpret_cast<void *>(*ptrToPtr);
  hKernel->_args.push_back(derefPtr);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetExecInfo(
    ur_kernel_handle_t hKernel, ur_kernel_exec_info_t propName, size_t propSize,
    const ur_kernel_exec_info_properties_t *pProperties,
    const void *pPropValue) {
  std::ignore = hKernel;
  std::ignore = propName;
  std::ignore = propSize;
  std::ignore = pProperties;
  std::ignore = pPropValue;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgSampler(ur_kernel_handle_t hKernel, uint32_t argIndex,
                      const ur_kernel_arg_sampler_properties_t *pProperties,
                      ur_sampler_handle_t hArgValue) {
  std::ignore = hKernel;
  std::ignore = argIndex;
  std::ignore = pProperties;
  std::ignore = hArgValue;

  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgMemObj(ur_kernel_handle_t hKernel, uint32_t argIndex,
                     const ur_kernel_arg_mem_obj_properties_t *pProperties,
                     ur_mem_handle_t hArgValue) {
  // TODO: out_of_order args?
  std::ignore = argIndex;
  std::ignore = pProperties;

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // Taken from ur/adapters/cuda/kernel.cpp
  // zero-sized buffers are expected to be null.
  if (hArgValue == nullptr) {
    hKernel->_args.emplace_back(nullptr);
    return UR_RESULT_SUCCESS;
  }

  hKernel->_args.emplace_back(hArgValue->_mem);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants) {
  std::ignore = hKernel;
  std::ignore = count;
  std::ignore = pSpecConstants;

  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel, ur_native_handle_t *phNativeKernel) {
  std::ignore = hKernel;
  std::ignore = phNativeKernel;

  DIE_NO_IMPLEMENTATION
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t hNativeKernel, ur_context_handle_t hContext,
    ur_program_handle_t hProgram,
    const ur_kernel_native_properties_t *pProperties,
    ur_kernel_handle_t *phKernel) {
  std::ignore = hNativeKernel;
  std::ignore = hContext;
  std::ignore = hProgram;
  std::ignore = pProperties;
  std::ignore = phKernel;

  DIE_NO_IMPLEMENTATION
}
