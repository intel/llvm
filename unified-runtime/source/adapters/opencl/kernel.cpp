//===----------- kernel.cpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "kernel.hpp"
#include "adapter.hpp"
#include "common.hpp"
#include "device.hpp"
#include "memory.hpp"
#include "program.hpp"
#include "queue.hpp"
#include "sampler.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>

ur_result_t ur_kernel_handle_t_::makeWithNative(native_type NativeKernel,
                                                ur_program_handle_t Program,
                                                ur_context_handle_t Context,
                                                ur_kernel_handle_t &Kernel) {
  try {
    cl_context CLContext;
    CL_RETURN_ON_FAILURE(clGetKernelInfo(NativeKernel, CL_KERNEL_CONTEXT,
                                         sizeof(CLContext), &CLContext,
                                         nullptr));
    cl_program CLProgram;
    CL_RETURN_ON_FAILURE(clGetKernelInfo(NativeKernel, CL_KERNEL_PROGRAM,
                                         sizeof(CLProgram), &CLProgram,
                                         nullptr));

    if (Context->CLContext != CLContext) {
      return UR_RESULT_ERROR_INVALID_CONTEXT;
    }
    if (Program) {
      if (Program->CLProgram != CLProgram) {
        return UR_RESULT_ERROR_INVALID_PROGRAM;
      }
    } else {
      ur_native_handle_t hNativeHandle =
          reinterpret_cast<ur_native_handle_t>(CLProgram);
      UR_RETURN_ON_FAILURE(urProgramCreateWithNativeHandle(
          hNativeHandle, Context, nullptr, &Program));
    }

    auto URKernel =
        std::make_unique<ur_kernel_handle_t_>(NativeKernel, Program, Context);
    Kernel = URKernel.release();
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(ur_program_handle_t hProgram, const char *pKernelName,
               ur_kernel_handle_t *phKernel) {
  try {
    cl_int CLResult;
    cl_kernel Kernel =
        clCreateKernel(hProgram->CLProgram, pKernelName, &CLResult);

    if (CLResult == CL_INVALID_KERNEL_DEFINITION) {
      cl_adapter::setErrorMessage(
          "clCreateKernel failed with CL_INVALID_KERNEL_DEFINITION", CLResult);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
    }

    CL_RETURN_ON_FAILURE(CLResult);
    auto URKernel = std::make_unique<ur_kernel_handle_t_>(Kernel, hProgram,
                                                          hProgram->Context);
    *phKernel = URKernel.release();
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_value_properties_t *, const void *pArgValue) {

  CL_RETURN_ON_FAILURE(clSetKernelArg(
      hKernel->CLKernel, static_cast<cl_uint>(argIndex), argSize, pArgValue));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgLocal(ur_kernel_handle_t hKernel, uint32_t argIndex,
                    size_t argSize, const ur_kernel_arg_local_properties_t *) {

  CL_RETURN_ON_FAILURE(clSetKernelArg(
      hKernel->CLKernel, static_cast<cl_uint>(argIndex), argSize, nullptr));

  return UR_RESULT_SUCCESS;
}

// Querying the number of registers that a kernel uses is supported unofficially
// on some devices.
#ifndef CL_KERNEL_REGISTER_COUNT_INTEL
#define CL_KERNEL_REGISTER_COUNT_INTEL 0x425B
#endif

static cl_int mapURKernelInfoToCL(ur_kernel_info_t URPropName) {
  switch (static_cast<uint32_t>(URPropName)) {
  case UR_KERNEL_INFO_FUNCTION_NAME:
    return CL_KERNEL_FUNCTION_NAME;
  case UR_KERNEL_INFO_NUM_ARGS:
    return CL_KERNEL_NUM_ARGS;
  case UR_KERNEL_INFO_REFERENCE_COUNT:
    return CL_KERNEL_REFERENCE_COUNT;
  case UR_KERNEL_INFO_CONTEXT:
    return CL_KERNEL_CONTEXT;
  case UR_KERNEL_INFO_PROGRAM:
    return CL_KERNEL_PROGRAM;
  case UR_KERNEL_INFO_ATTRIBUTES:
    return CL_KERNEL_ATTRIBUTES;
  case UR_KERNEL_INFO_SPILL_MEM_SIZE:
    return CL_KERNEL_SPILL_MEM_SIZE_INTEL;
  case UR_KERNEL_INFO_NUM_REGS:
    return CL_KERNEL_REGISTER_COUNT_INTEL;
  default:
    return -1;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetInfo(ur_kernel_handle_t hKernel,
                                                    ur_kernel_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_KERNEL_INFO_PROGRAM: {
    return ReturnValue(hKernel->Program);
  }
  case UR_KERNEL_INFO_CONTEXT: {
    return ReturnValue(hKernel->Context);
  }
  case UR_KERNEL_INFO_REFERENCE_COUNT: {
    return ReturnValue(hKernel->RefCount.getCount());
  }
  default: {
    size_t CheckPropSize = 0;
    cl_int ClResult =
        clGetKernelInfo(hKernel->CLKernel, mapURKernelInfoToCL(propName),
                        propSize, pPropValue, &CheckPropSize);
    if (pPropValue && CheckPropSize != propSize) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }
    if (ClResult == CL_INVALID_VALUE) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
    CL_RETURN_ON_FAILURE(ClResult);
    if (pPropSizeRet) {
      *pPropSizeRet = CheckPropSize;
    }
  }
  }

  return UR_RESULT_SUCCESS;
}

static cl_int mapURKernelGroupInfoToCL(ur_kernel_group_info_t URPropName) {

  switch (static_cast<uint32_t>(URPropName)) {
  case UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE:
    return CL_KERNEL_GLOBAL_WORK_SIZE;
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE:
    return CL_KERNEL_WORK_GROUP_SIZE;
  case UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE:
    return CL_KERNEL_COMPILE_WORK_GROUP_SIZE;
  case UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE:
    return CL_KERNEL_LOCAL_MEM_SIZE;
  case UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
    return CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE;
  case UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE:
    return CL_KERNEL_PRIVATE_MEM_SIZE;
  default:
    return -1;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetGroupInfo(ur_kernel_handle_t hKernel, ur_device_handle_t hDevice,
                     ur_kernel_group_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {
  // From the CL spec for GROUP_INFO_GLOBAL: "If device is not a custom device
  // and kernel is not a built-in kernel, clGetKernelWorkGroupInfo returns the
  // error CL_INVALID_VALUE.". Unfortunately there doesn't seem to be a nice
  // way to query whether a kernel is a builtin kernel but this should suffice
  // to deter naive use of the query.
  if (propName == UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE) {
    cl_device_type ClDeviceType;
    CL_RETURN_ON_FAILURE(clGetDeviceInfo(hDevice->CLDevice, CL_DEVICE_TYPE,
                                         sizeof(ClDeviceType), &ClDeviceType,
                                         nullptr));
    if (ClDeviceType != CL_DEVICE_TYPE_CUSTOM) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
  }
  if (propName == UR_KERNEL_GROUP_INFO_COMPILE_MAX_WORK_GROUP_SIZE ||
      propName == UR_KERNEL_GROUP_INFO_COMPILE_MAX_LINEAR_WORK_GROUP_SIZE) {
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  CL_RETURN_ON_FAILURE(clGetKernelWorkGroupInfo(
      hKernel->CLKernel, hDevice->CLDevice, mapURKernelGroupInfoToCL(propName),
      propSize, pPropValue, pPropSizeRet));

  return UR_RESULT_SUCCESS;
}

static cl_int
mapURKernelSubGroupInfoToCL(ur_kernel_sub_group_info_t URPropName) {

  switch (static_cast<uint32_t>(URPropName)) {
  case UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE:
    return CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE;
  case UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS:
    return CL_KERNEL_MAX_NUM_SUB_GROUPS;
  case UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS:
    return CL_KERNEL_COMPILE_NUM_SUB_GROUPS;
  case UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL:
    return CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL;
  default:
    return -1;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetSubGroupInfo(ur_kernel_handle_t hKernel, ur_device_handle_t hDevice,
                        ur_kernel_sub_group_info_t propName, size_t,
                        void *pPropValue, size_t *pPropSizeRet) {

  std::shared_ptr<void> InputValue;
  size_t InputValueSize = 0;
  size_t RetVal;

  if (propName == UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE) {
    // OpenCL needs an input value for PI_KERNEL_MAX_SUB_GROUP_SIZE so if no
    // value is given we use the max work item size of the device in the first
    // dimension to avoid truncation of max sub-group size.
    uint32_t MaxDims = 0;
    ur_result_t URRet =
        urDeviceGetInfo(hDevice, UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
                        sizeof(uint32_t), &MaxDims, nullptr);
    if (URRet != UR_RESULT_SUCCESS)
      return URRet;
    std::shared_ptr<size_t[]> WgSizes{new size_t[MaxDims]};
    URRet = urDeviceGetInfo(hDevice, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
                            MaxDims * sizeof(size_t), WgSizes.get(), nullptr);
    if (URRet != UR_RESULT_SUCCESS)
      return URRet;
    for (size_t i = 1; i < MaxDims; ++i)
      WgSizes.get()[i] = 1;
    InputValue = std::move(WgSizes);
    InputValueSize = MaxDims * sizeof(size_t);
  }

  // We need to allow for the possibility that this device runs an older CL and
  // supports the original khr subgroup extension.
  cl_ext::clGetKernelSubGroupInfoKHR_fn GetKernelSubGroupInfo = nullptr;

  oclv::OpenCLVersion DevVer;
  CL_RETURN_ON_FAILURE(hDevice->getDeviceVersion(DevVer));

  if (DevVer < oclv::V2_1) {
    bool SubgroupExtSupported = false;

    UR_RETURN_ON_FAILURE(hDevice->checkDeviceExtensions({"cl_khr_subgroups"},
                                                        SubgroupExtSupported));
    if (!SubgroupExtSupported) {
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }
    cl_context Context = nullptr;
    CL_RETURN_ON_FAILURE(clGetKernelInfo(hKernel->CLKernel, CL_KERNEL_CONTEXT,
                                         sizeof(Context), &Context, nullptr));
    UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext(
        Context, ur::cl::getAdapter()->fnCache.clGetKernelSubGroupInfoKHRCache,
        cl_ext::GetKernelSubGroupInfoName, &GetKernelSubGroupInfo));
  } else {
    GetKernelSubGroupInfo = clGetKernelSubGroupInfo;
  }

  cl_int Ret = GetKernelSubGroupInfo(hKernel->CLKernel, hDevice->CLDevice,
                                     mapURKernelSubGroupInfoToCL(propName),
                                     InputValueSize, InputValue.get(),
                                     sizeof(size_t), &RetVal, pPropSizeRet);

  if (Ret == CL_INVALID_OPERATION) {
    // clGetKernelSubGroupInfo returns CL_INVALID_OPERATION if the device does
    // not support subgroups.
    if (propName == UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS) {
      RetVal = 1; // Minimum required by SYCL 2020 spec
      Ret = CL_SUCCESS;
    } else if (propName == UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS) {
      RetVal = 0; // Not specified by kernel
      Ret = CL_SUCCESS;
    } else if (propName == UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE) {
      // Return the maximum sub group size for the device
      size_t ResultSize = 0;
      // Two calls to urDeviceGetInfo are needed: the first determines the size
      // required to store the result, and the second returns the actual size
      // values.
      UR_RETURN_ON_FAILURE(urDeviceGetInfo(hDevice,
                                           UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL,
                                           0, nullptr, &ResultSize));
      assert(ResultSize % sizeof(uint32_t) == 0);
      std::vector<uint32_t> Result(ResultSize / sizeof(uint32_t));
      UR_RETURN_ON_FAILURE(urDeviceGetInfo(hDevice,
                                           UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL,
                                           ResultSize, Result.data(), nullptr));
      RetVal = *std::max_element(Result.begin(), Result.end());
      Ret = CL_SUCCESS;
    } else if (propName == UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL) {
      RetVal = 0; // Not specified by kernel
      Ret = CL_SUCCESS;
    }
  }

  if (pPropValue)
    *(static_cast<uint32_t *>(pPropValue)) = static_cast<uint32_t>(RetVal);
  if (pPropSizeRet)
    *pPropSizeRet = sizeof(uint32_t);

  CL_RETURN_ON_FAILURE(Ret);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelRetain(ur_kernel_handle_t hKernel) {
  hKernel->RefCount.retain();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelRelease(ur_kernel_handle_t hKernel) {
  if (hKernel->RefCount.release()) {
    delete hKernel;
  }
  return UR_RESULT_SUCCESS;
}

/**
 * Enables indirect access of pointers in kernels. Necessary to avoid telling CL
 * about every pointer that might be used.
 */
static ur_result_t usmSetIndirectAccess(ur_kernel_handle_t hKernel) {

  cl_bool TrueVal = CL_TRUE;
  clHostMemAllocINTEL_fn HFunc = nullptr;
  clSharedMemAllocINTEL_fn SFunc = nullptr;
  clDeviceMemAllocINTEL_fn DFunc = nullptr;
  cl_context CLContext;

  /* We test that each alloc type is supported before we actually try to set
   * KernelExecInfo. */
  CL_RETURN_ON_FAILURE(clGetKernelInfo(hKernel->CLKernel, CL_KERNEL_CONTEXT,
                                       sizeof(cl_context), &CLContext,
                                       nullptr));

  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clHostMemAllocINTEL_fn>(
      CLContext, ur::cl::getAdapter()->fnCache.clHostMemAllocINTELCache,
      cl_ext::HostMemAllocName, &HFunc));

  if (HFunc) {
    CL_RETURN_ON_FAILURE(clSetKernelExecInfo(
        hKernel->CLKernel, CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
        sizeof(cl_bool), &TrueVal));
  }

  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clDeviceMemAllocINTEL_fn>(
      CLContext, ur::cl::getAdapter()->fnCache.clDeviceMemAllocINTELCache,
      cl_ext::DeviceMemAllocName, &DFunc));

  if (DFunc) {
    CL_RETURN_ON_FAILURE(clSetKernelExecInfo(
        hKernel->CLKernel, CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
        sizeof(cl_bool), &TrueVal));
  }

  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clSharedMemAllocINTEL_fn>(
      CLContext, ur::cl::getAdapter()->fnCache.clSharedMemAllocINTELCache,
      cl_ext::SharedMemAllocName, &SFunc));

  if (SFunc) {
    CL_RETURN_ON_FAILURE(clSetKernelExecInfo(
        hKernel->CLKernel, CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
        sizeof(cl_bool), &TrueVal));
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetExecInfo(
    ur_kernel_handle_t hKernel, ur_kernel_exec_info_t propName, size_t propSize,
    const ur_kernel_exec_info_properties_t *, const void *pPropValue) {

  switch (propName) {
  case UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS: {
    if (*(static_cast<const ur_bool_t *>(pPropValue))) {
      UR_RETURN_ON_FAILURE(usmSetIndirectAccess(hKernel));
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_KERNEL_EXEC_INFO_CACHE_CONFIG: {
    // Setting the cache config is unsupported in OpenCL, but this is just a
    // hint.
    return UR_RESULT_SUCCESS;
  }
  case UR_KERNEL_EXEC_INFO_USM_PTRS: {
    CL_RETURN_ON_FAILURE(clSetKernelExecInfo(hKernel->CLKernel,
                                             CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
                                             propSize, pPropValue));
    return UR_RESULT_SUCCESS;
  }
  default: {
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, uint32_t argIndex,
    const ur_kernel_arg_pointer_properties_t *, const void *pArgValue) {

  if (hKernel->clSetKernelArgMemPointerINTEL == nullptr) {
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  CL_RETURN_ON_FAILURE(hKernel->clSetKernelArgMemPointerINTEL(
      hKernel->CLKernel, static_cast<cl_uint>(argIndex), pArgValue));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel, ur_native_handle_t *phNativeKernel) {

  *phNativeKernel = reinterpret_cast<ur_native_handle_t>(hKernel->CLKernel);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSuggestMaxCooperativeGroupCount(
    [[maybe_unused]] ur_kernel_handle_t hKernel,
    [[maybe_unused]] ur_device_handle_t hDevice,
    [[maybe_unused]] uint32_t workDim,
    [[maybe_unused]] const size_t *pLocalWorkSize,
    [[maybe_unused]] size_t dynamicSharedMemorySize,
    [[maybe_unused]] uint32_t *pGroupCountRet) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t hNativeKernel, ur_context_handle_t hContext,
    ur_program_handle_t hProgram,
    [[maybe_unused]] const ur_kernel_native_properties_t *pProperties,
    ur_kernel_handle_t *phKernel) {
  cl_kernel NativeHandle = reinterpret_cast<cl_kernel>(hNativeKernel);

  UR_RETURN_ON_FAILURE(ur_kernel_handle_t_::makeWithNative(
      NativeHandle, hProgram, hContext, *phKernel));

  (*phKernel)->IsNativeHandleOwned =
      pProperties ? pProperties->isNativeHandleOwned : false;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel, uint32_t argIndex,
    const ur_kernel_arg_mem_obj_properties_t *, ur_mem_handle_t hArgValue) {

  cl_mem CLArgValue = hArgValue ? hArgValue->CLMemory : nullptr;
  CL_RETURN_ON_FAILURE(clSetKernelArg(hKernel->CLKernel,
                                      static_cast<cl_uint>(argIndex),
                                      sizeof(CLArgValue), &CLArgValue));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgSampler(
    ur_kernel_handle_t hKernel, uint32_t argIndex,
    const ur_kernel_arg_sampler_properties_t *, ur_sampler_handle_t hArgValue) {

  cl_sampler CLArgSampler = hArgValue->CLSampler;
  cl_int RetErr =
      clSetKernelArg(hKernel->CLKernel, static_cast<cl_uint>(argIndex),
                     sizeof(CLArgSampler), &CLArgSampler);
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetSuggestedLocalWorkSize(
    ur_kernel_handle_t hKernel, ur_queue_handle_t hQueue, uint32_t workDim,
    const size_t *pGlobalWorkOffset, const size_t *pGlobalWorkSize,
    size_t *pSuggestedLocalWorkSize) {
  cl_device_id Device;
  cl_platform_id Platform;

  CL_RETURN_ON_FAILURE(clGetCommandQueueInfo(hQueue->CLQueue, CL_QUEUE_DEVICE,
                                             sizeof(cl_device_id), &Device,
                                             nullptr));

  CL_RETURN_ON_FAILURE(clGetDeviceInfo(
      Device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &Platform, nullptr));

  auto GetKernelSuggestedLocalWorkSizeFuncPtr =
      (clGetKernelSuggestedLocalWorkSizeKHR_fn)
          clGetExtensionFunctionAddressForPlatform(
              Platform, "clGetKernelSuggestedLocalWorkSizeKHR");
  if (!GetKernelSuggestedLocalWorkSizeFuncPtr)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  CL_RETURN_ON_FAILURE(GetKernelSuggestedLocalWorkSizeFuncPtr(
      hQueue->CLQueue, hKernel->CLKernel, workDim, pGlobalWorkOffset,
      pGlobalWorkSize, pSuggestedLocalWorkSize));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    ur_kernel_handle_t, uint32_t, const ur_specialization_constant_info_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
