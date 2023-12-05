//===----------- kernel.cpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "common.hpp"

#include <algorithm>
#include <memory>

UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(ur_program_handle_t hProgram, const char *pKernelName,
               ur_kernel_handle_t *phKernel) {

  cl_int CLResult;
  *phKernel = cl_adapter::cast<ur_kernel_handle_t>(clCreateKernel(
      cl_adapter::cast<cl_program>(hProgram), pKernelName, &CLResult));
  CL_RETURN_ON_FAILURE(CLResult);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_value_properties_t *, const void *pArgValue) {

  CL_RETURN_ON_FAILURE(clSetKernelArg(cl_adapter::cast<cl_kernel>(hKernel),
                                      cl_adapter::cast<cl_uint>(argIndex),
                                      argSize, pArgValue));

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgLocal(ur_kernel_handle_t hKernel, uint32_t argIndex,
                    size_t argSize, const ur_kernel_arg_local_properties_t *) {

  CL_RETURN_ON_FAILURE(clSetKernelArg(cl_adapter::cast<cl_kernel>(hKernel),
                                      cl_adapter::cast<cl_uint>(argIndex),
                                      argSize, nullptr));

  return UR_RESULT_SUCCESS;
}

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
  case UR_KERNEL_INFO_NUM_REGS:
    return CL_KERNEL_NUM_ARGS;
  default:
    return -1;
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetInfo(ur_kernel_handle_t hKernel,
                                                    ur_kernel_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {
  // We need this little bit of ugliness because the UR NUM_ARGS property is
  // size_t whereas the CL one is cl_uint. We should consider changing that see
  // #1038
  if (propName == UR_KERNEL_INFO_NUM_ARGS) {
    if (pPropSizeRet)
      *pPropSizeRet = sizeof(size_t);
    cl_uint NumArgs = 0;
    CL_RETURN_ON_FAILURE(clGetKernelInfo(cl_adapter::cast<cl_kernel>(hKernel),
                                         mapURKernelInfoToCL(propName),
                                         sizeof(NumArgs), &NumArgs, nullptr));
    if (pPropValue) {
      if (propSize != sizeof(size_t))
        return UR_RESULT_ERROR_INVALID_SIZE;
      *static_cast<size_t *>(pPropValue) = static_cast<size_t>(NumArgs);
    }
  } else {
    size_t CheckPropSize = 0;
    cl_int ClResult = clGetKernelInfo(cl_adapter::cast<cl_kernel>(hKernel),
                                      mapURKernelInfoToCL(propName), propSize,
                                      pPropValue, &CheckPropSize);
    if (pPropValue && CheckPropSize != propSize) {
      return UR_RESULT_ERROR_INVALID_SIZE;
    }
    CL_RETURN_ON_FAILURE(ClResult);
    if (pPropSizeRet) {
      *pPropSizeRet = CheckPropSize;
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
    CL_RETURN_ON_FAILURE(
        clGetDeviceInfo(cl_adapter::cast<cl_device_id>(hDevice), CL_DEVICE_TYPE,
                        sizeof(ClDeviceType), &ClDeviceType, nullptr));
    if (ClDeviceType != CL_DEVICE_TYPE_CUSTOM) {
      return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
    }
  }
  CL_RETURN_ON_FAILURE(clGetKernelWorkGroupInfo(
      cl_adapter::cast<cl_kernel>(hKernel),
      cl_adapter::cast<cl_device_id>(hDevice),
      mapURKernelGroupInfoToCL(propName), propSize, pPropValue, pPropSizeRet));

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

  cl_int Ret = clGetKernelSubGroupInfo(cl_adapter::cast<cl_kernel>(hKernel),
                                       cl_adapter::cast<cl_device_id>(hDevice),
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
      ur_result_t URRet =
          urDeviceGetInfo(hDevice, UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL, 0,
                          nullptr, &ResultSize);
      if (URRet != UR_RESULT_SUCCESS) {
        return URRet;
      }
      assert(ResultSize % sizeof(size_t) == 0);
      std::vector<size_t> Result(ResultSize / sizeof(size_t));
      URRet = urDeviceGetInfo(hDevice, UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL,
                              ResultSize, Result.data(), nullptr);
      if (URRet != UR_RESULT_SUCCESS) {
        return URRet;
      }
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
  CL_RETURN_ON_FAILURE(clRetainKernel(cl_adapter::cast<cl_kernel>(hKernel)));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelRelease(ur_kernel_handle_t hKernel) {
  CL_RETURN_ON_FAILURE(clReleaseKernel(cl_adapter::cast<cl_kernel>(hKernel)));
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
  CL_RETURN_ON_FAILURE(clGetKernelInfo(cl_adapter::cast<cl_kernel>(hKernel),
                                       CL_KERNEL_CONTEXT, sizeof(cl_context),
                                       &CLContext, nullptr));

  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clHostMemAllocINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clHostMemAllocINTELCache,
      cl_ext::HostMemAllocName, &HFunc));

  if (HFunc) {
    CL_RETURN_ON_FAILURE(
        clSetKernelExecInfo(cl_adapter::cast<cl_kernel>(hKernel),
                            CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
                            sizeof(cl_bool), &TrueVal));
  }

  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clDeviceMemAllocINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clDeviceMemAllocINTELCache,
      cl_ext::DeviceMemAllocName, &DFunc));

  if (DFunc) {
    CL_RETURN_ON_FAILURE(
        clSetKernelExecInfo(cl_adapter::cast<cl_kernel>(hKernel),
                            CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
                            sizeof(cl_bool), &TrueVal));
  }

  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clSharedMemAllocINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clSharedMemAllocINTELCache,
      cl_ext::SharedMemAllocName, &SFunc));

  if (SFunc) {
    CL_RETURN_ON_FAILURE(
        clSetKernelExecInfo(cl_adapter::cast<cl_kernel>(hKernel),
                            CL_KERNEL_EXEC_INFO_INDIRECT_SHARED_ACCESS_INTEL,
                            sizeof(cl_bool), &TrueVal));
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetExecInfo(
    ur_kernel_handle_t hKernel, ur_kernel_exec_info_t propName, size_t propSize,
    const ur_kernel_exec_info_properties_t *, const void *pPropValue) {

  switch (propName) {
  case UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS: {
    if (*(static_cast<const ur_bool_t *>(pPropValue)) == true) {
      CL_RETURN_ON_FAILURE(usmSetIndirectAccess(hKernel));
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_KERNEL_EXEC_INFO_CACHE_CONFIG: {
    // Setting the cache config is unsupported in OpenCL, but this is just a
    // hint.
    return UR_RESULT_SUCCESS;
  }
  case UR_KERNEL_EXEC_INFO_USM_PTRS: {
    CL_RETURN_ON_FAILURE(clSetKernelExecInfo(
        cl_adapter::cast<cl_kernel>(hKernel),
        CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL, propSize, pPropValue));
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

  cl_context CLContext;
  CL_RETURN_ON_FAILURE(clGetKernelInfo(cl_adapter::cast<cl_kernel>(hKernel),
                                       CL_KERNEL_CONTEXT, sizeof(cl_context),
                                       &CLContext, nullptr));

  clSetKernelArgMemPointerINTEL_fn FuncPtr = nullptr;
  UR_RETURN_ON_FAILURE(
      cl_ext::getExtFuncFromContext<clSetKernelArgMemPointerINTEL_fn>(
          CLContext,
          cl_ext::ExtFuncPtrCache->clSetKernelArgMemPointerINTELCache,
          cl_ext::SetKernelArgMemPointerName, &FuncPtr));

  if (FuncPtr) {
    /* OpenCL passes pointers by value not by reference. This means we need to
     * deref the arg to get the pointer value */
    auto PtrToPtr = reinterpret_cast<const intptr_t *>(pArgValue);
    auto DerefPtr = reinterpret_cast<void *>(*PtrToPtr);
    CL_RETURN_ON_FAILURE(FuncPtr(cl_adapter::cast<cl_kernel>(hKernel),
                                 cl_adapter::cast<cl_uint>(argIndex),
                                 DerefPtr));
  }

  return UR_RESULT_SUCCESS;
}
UR_APIEXPORT ur_result_t UR_APICALL urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel, ur_native_handle_t *phNativeKernel) {

  *phNativeKernel = reinterpret_cast<ur_native_handle_t>(hKernel);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t hNativeKernel, ur_context_handle_t, ur_program_handle_t,
    const ur_kernel_native_properties_t *pProperties,
    ur_kernel_handle_t *phKernel) {
  *phKernel = reinterpret_cast<ur_kernel_handle_t>(hNativeKernel);
  if (!pProperties || !pProperties->isNativeHandleOwned) {
    return urKernelRetain(*phKernel);
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel, uint32_t argIndex,
    const ur_kernel_arg_mem_obj_properties_t *, ur_mem_handle_t hArgValue) {

  cl_int RetErr = clSetKernelArg(
      cl_adapter::cast<cl_kernel>(hKernel), cl_adapter::cast<cl_uint>(argIndex),
      sizeof(hArgValue), cl_adapter::cast<const cl_mem *>(&hArgValue));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgSampler(
    ur_kernel_handle_t hKernel, uint32_t argIndex,
    const ur_kernel_arg_sampler_properties_t *, ur_sampler_handle_t hArgValue) {

  cl_int RetErr = clSetKernelArg(
      cl_adapter::cast<cl_kernel>(hKernel), cl_adapter::cast<cl_uint>(argIndex),
      sizeof(hArgValue), cl_adapter::cast<const cl_sampler *>(&hArgValue));
  CL_RETURN_ON_FAILURE(RetErr);
  return UR_RESULT_SUCCESS;
}
