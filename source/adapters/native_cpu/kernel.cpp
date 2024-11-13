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
  ur_kernel_handle_t_ *kernel;

  // Set reqd_work_group_size for kernel if needed
  std::optional<native_cpu::WGSize_t> ReqdWG;
  const auto &ReqdMap = hProgram->KernelReqdWorkGroupSizeMD;
  if (auto ReqdIt = ReqdMap.find(pKernelName); ReqdIt != ReqdMap.end()) {
    ReqdWG = ReqdIt->second;
  }

  std::optional<native_cpu::WGSize_t> MaxWG;
  const auto &MaxMap = hProgram->KernelMaxWorkGroupSizeMD;
  if (auto MaxIt = MaxMap.find(pKernelName); MaxIt != MaxMap.end()) {
    MaxWG = MaxIt->second;
  }
  std::optional<uint64_t> MaxLinearWG;
  const auto &MaxLinMap = hProgram->KernelMaxLinearWorkGroupSizeMD;
  if (auto MaxLIt = MaxLinMap.find(pKernelName); MaxLIt != MaxLinMap.end()) {
    MaxLinearWG = MaxLIt->second;
  }
  kernel = new ur_kernel_handle_t_(hProgram, pKernelName, *f, ReqdWG, MaxWG,
                                   MaxLinearWG);

  *phKernel = kernel;

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_value_properties_t *pProperties,
    const void *pArgValue) {
  // TODO: error checking
  std::ignore = argIndex;
  std::ignore = pProperties;

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(argSize, UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);

  hKernel->addArg(pArgValue, argIndex, argSize);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgLocal(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_local_properties_t *pProperties) {
  std::ignore = pProperties;
  // emplace a placeholder kernel arg, gets replaced with a pointer to the
  // memory pool before enqueueing the kernel.
  hKernel->addPtrArg(nullptr, argIndex);
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
    return ReturnValue(hKernel->_name);
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
    size_t GroupSize[3] = {0, 0, 0};
    const auto &ReqdWGSizeMDMap = hKernel->hProgram->KernelReqdWorkGroupSizeMD;
    const auto ReqdWGSizeMD = ReqdWGSizeMDMap.find(hKernel->_name);
    if (ReqdWGSizeMD != ReqdWGSizeMDMap.end()) {
      const auto ReqdWGSize = ReqdWGSizeMD->second;
      GroupSize[0] = std::get<0>(ReqdWGSize);
      GroupSize[1] = std::get<1>(ReqdWGSize);
      GroupSize[2] = std::get<2>(ReqdWGSize);
    }
    return returnValue(GroupSize, 3);
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
  case UR_KERNEL_GROUP_INFO_COMPILE_MAX_WORK_GROUP_SIZE:
  case UR_KERNEL_GROUP_INFO_COMPILE_MAX_LINEAR_WORK_GROUP_SIZE:
    // FIXME: could be added
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;

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
  std::ignore = argIndex;
  std::ignore = pProperties;

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pArgValue, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  hKernel->addPtrArg(const_cast<void *>(pArgValue), argIndex);

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
  std::ignore = argIndex;
  std::ignore = pProperties;

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // Taken from ur/adapters/cuda/kernel.cpp
  // zero-sized buffers are expected to be null.
  if (hArgValue == nullptr) {
    hKernel->addPtrArg(nullptr, argIndex);
    return UR_RESULT_SUCCESS;
  }

  hKernel->addPtrArg(hArgValue->_mem, argIndex);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    ur_kernel_handle_t hKernel, uint32_t count,
    const ur_specialization_constant_info_t *pSpecConstants) {
  std::ignore = hKernel;
  std::ignore = count;
  std::ignore = pSpecConstants;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetSuggestedLocalWorkSize(
    [[maybe_unused]] ur_kernel_handle_t hKernel,
    [[maybe_unused]] ur_queue_handle_t hQueue,
    [[maybe_unused]] uint32_t workDim,
    [[maybe_unused]] const size_t *pGlobalWorkOffset,
    [[maybe_unused]] const size_t *pGlobalWorkSize,
    [[maybe_unused]] size_t *pSuggestedLocalWorkSize) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
