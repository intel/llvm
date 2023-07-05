//===----------- kernel.cpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------===//
#include "common.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(ur_program_handle_t hProgram, const char *pKernelName,
               ur_kernel_handle_t *phKernel) {

  cl_int cl_result;
  *phKernel = cl_adapter::cast<ur_kernel_handle_t>(clCreateKernel(
      cl_adapter::cast<cl_program>(hProgram), pKernelName, &cl_result));
  CL_RETURN_ON_FAILURE(cl_result);
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

static cl_int map_ur_kernel_info_to_cl(ur_kernel_info_t urPropName) {

  cl_int cl_propName;
  switch (static_cast<uint32_t>(urPropName)) {
  case UR_KERNEL_INFO_FUNCTION_NAME:
    cl_propName = CL_KERNEL_FUNCTION_NAME;
    break;
  case UR_KERNEL_INFO_NUM_ARGS:
    cl_propName = CL_KERNEL_NUM_ARGS;
    break;
  case UR_KERNEL_INFO_REFERENCE_COUNT:
    cl_propName = CL_KERNEL_REFERENCE_COUNT;
    break;
  case UR_KERNEL_INFO_CONTEXT:
    cl_propName = CL_KERNEL_CONTEXT;
    break;
  case UR_KERNEL_INFO_PROGRAM:
    cl_propName = CL_KERNEL_PROGRAM;
    break;
  case UR_KERNEL_INFO_ATTRIBUTES:
    cl_propName = CL_KERNEL_ATTRIBUTES;
    break;
  case UR_KERNEL_INFO_NUM_REGS:
    cl_propName = CL_KERNEL_NUM_ARGS;
    break;
  default:
    cl_propName = -1;
  }

  return cl_propName;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetInfo(ur_kernel_handle_t hKernel,
                                                    ur_kernel_info_t propName,
                                                    size_t propSize,
                                                    void *pPropValue,
                                                    size_t *pPropSizeRet) {

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  CL_RETURN_ON_FAILURE(clGetKernelInfo(cl_adapter::cast<cl_kernel>(hKernel),
                                       map_ur_kernel_info_to_cl(propName),
                                       propSize, pPropValue, pPropSizeRet));

  return UR_RESULT_SUCCESS;
}

static cl_int
map_ur_kernel_group_info_to_cl(ur_kernel_group_info_t urPropName) {

  cl_int cl_propName;
  switch (static_cast<uint32_t>(urPropName)) {
  case UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE:
    cl_propName = CL_KERNEL_GLOBAL_WORK_SIZE;
    break;
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE:
    cl_propName = CL_KERNEL_WORK_GROUP_SIZE;
    break;
  case UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE:
    cl_propName = CL_KERNEL_COMPILE_WORK_GROUP_SIZE;
    break;
  case UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE:
    cl_propName = CL_KERNEL_LOCAL_MEM_SIZE;
    break;
  case UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE:
    cl_propName = CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE;
    break;
  case UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE:
    cl_propName = CL_KERNEL_PRIVATE_MEM_SIZE;
    break;
  default:
    cl_propName = -1;
  }

  return cl_propName;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetGroupInfo(ur_kernel_handle_t hKernel, ur_device_handle_t hDevice,
                     ur_kernel_group_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  CL_RETURN_ON_FAILURE(
      clGetKernelWorkGroupInfo(cl_adapter::cast<cl_kernel>(hKernel),
                               cl_adapter::cast<cl_device_id>(hDevice),
                               map_ur_kernel_group_info_to_cl(propName),
                               propSize, pPropValue, pPropSizeRet));

  return UR_RESULT_SUCCESS;
}

static cl_int
map_ur_kernel_sub_group_info_to_cl(ur_kernel_sub_group_info_t urPropName) {

  cl_int cl_propName;
  switch (static_cast<uint32_t>(urPropName)) {
  case UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE:
    cl_propName = CL_KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE;
    break;
  case UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS:
    cl_propName = CL_KERNEL_MAX_NUM_SUB_GROUPS;
    break;
  case UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS:
    cl_propName = CL_KERNEL_COMPILE_NUM_SUB_GROUPS;
    break;
  case UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL:
    cl_propName = CL_KERNEL_COMPILE_SUB_GROUP_SIZE_INTEL;
    break;
  default:
    cl_propName = -1;
  }

  return cl_propName;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetSubGroupInfo(ur_kernel_handle_t hKernel, ur_device_handle_t hDevice,
                        ur_kernel_sub_group_info_t propName, size_t propSize,
                        void *pPropValue, size_t *pPropSizeRet) {

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  std::shared_ptr<void> InputValue;
  size_t InputValueSize = 0;
  size_t RetVal;

  if (propName == UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE) {
    // OpenCL needs an input value for PI_KERNEL_MAX_SUB_GROUP_SIZE so if no
    // value is given we use the max work item size of the device in the first
    // dimention to avoid truncation of max sub-group size.
    uint32_t MaxDims = 0;
    ur_result_t UrRet =
        urDeviceGetInfo(hDevice, UR_DEVICE_INFO_MAX_WORK_ITEM_DIMENSIONS,
                        sizeof(uint32_t), &MaxDims, nullptr);
    if (UrRet != UR_RESULT_SUCCESS)
      return UrRet;
    std::shared_ptr<size_t[]> WGSizes{new size_t[MaxDims]};
    UrRet = urDeviceGetInfo(hDevice, UR_DEVICE_INFO_MAX_WORK_ITEM_SIZES,
                            MaxDims * sizeof(size_t), WGSizes.get(), nullptr);
    if (UrRet != UR_RESULT_SUCCESS)
      return UrRet;
    for (size_t i = 1; i < MaxDims; ++i)
      WGSizes.get()[i] = 1;
    InputValue = std::move(WGSizes);
    InputValueSize = MaxDims * sizeof(size_t);
  }

  cl_int Ret = clGetKernelSubGroupInfo(
      cl_adapter::cast<cl_kernel>(hKernel),
      cl_adapter::cast<cl_device_id>(hDevice),
      map_ur_kernel_sub_group_info_to_cl(propName), InputValueSize,
      InputValue.get(), sizeof(size_t), &RetVal, pPropSizeRet);

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
      size_t result_size = 0;
      // Two calls to urDeviceGetInfo are needed: the first determines the size
      // required to store the result, and the second returns the actual size
      // values.
      ur_result_t UrRet =
          urDeviceGetInfo(hDevice, UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL, 0,
                          nullptr, &result_size);
      if (UrRet != UR_RESULT_SUCCESS) {
        return UrRet;
      }
      assert(result_size % sizeof(size_t) == 0);
      std::vector<size_t> result(result_size / sizeof(size_t));
      UrRet = urDeviceGetInfo(hDevice, UR_DEVICE_INFO_SUB_GROUP_SIZES_INTEL,
                              result_size, result.data(), nullptr);
      if (UrRet != UR_RESULT_SUCCESS) {
        return UrRet;
      }
      RetVal = *std::max_element(result.begin(), result.end());
      Ret = CL_SUCCESS;
    } else if (propName == UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL) {
      RetVal = 0; // Not specified by kernel
      Ret = CL_SUCCESS;
    }
  }

  *(static_cast<uint32_t *>(pPropValue)) = static_cast<uint32_t>(RetVal);
  if (pPropSizeRet)
    *pPropSizeRet = sizeof(uint32_t);

  CL_RETURN_ON_FAILURE(Ret);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelRetain(ur_kernel_handle_t hKernel) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  CL_RETURN_ON_FAILURE(clRetainKernel(cl_adapter::cast<cl_kernel>(hKernel)));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelRelease(ur_kernel_handle_t hKernel) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  CL_RETURN_ON_FAILURE(clReleaseKernel(cl_adapter::cast<cl_kernel>(hKernel)));
  return UR_RESULT_SUCCESS;
}

/**
 * Enables indirect access of pointers in kernels. Necessary to avoid telling CL
 * about every pointer that might be used.
 */
static ur_result_t USMSetIndirectAccess(ur_kernel_handle_t hKernel) {

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
      cl_ext::clHostMemAllocName, &HFunc));

  if (HFunc) {
    CL_RETURN_ON_FAILURE(
        clSetKernelExecInfo(cl_adapter::cast<cl_kernel>(hKernel),
                            CL_KERNEL_EXEC_INFO_INDIRECT_HOST_ACCESS_INTEL,
                            sizeof(cl_bool), &TrueVal));
  }

  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clDeviceMemAllocINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clDeviceMemAllocINTELCache,
      cl_ext::clDeviceMemAllocName, &DFunc));

  if (DFunc) {
    CL_RETURN_ON_FAILURE(
        clSetKernelExecInfo(cl_adapter::cast<cl_kernel>(hKernel),
                            CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL,
                            sizeof(cl_bool), &TrueVal));
  }

  UR_RETURN_ON_FAILURE(cl_ext::getExtFuncFromContext<clSharedMemAllocINTEL_fn>(
      CLContext, cl_ext::ExtFuncPtrCache->clSharedMemAllocINTELCache,
      cl_ext::clSharedMemAllocName, &SFunc));

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
      CL_RETURN_ON_FAILURE(USMSetIndirectAccess(hKernel));
    }
    return UR_RESULT_SUCCESS;
  }
  case UR_KERNEL_EXEC_INFO_CACHE_CONFIG: {
    /* Setting the cache config is unsupported in OpenCL */
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
  case UR_KERNEL_EXEC_INFO_USM_PTRS: {
    CL_RETURN_ON_FAILURE(clSetKernelExecInfo(
        cl_adapter::cast<cl_kernel>(hKernel), propName, propSize, pPropValue));
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
          cl_ext::clSetKernelArgMemPointerName, &FuncPtr));

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

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phNativeKernel, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  *phNativeKernel = reinterpret_cast<ur_native_handle_t>(hKernel);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t hNativeKernel, ur_context_handle_t, ur_program_handle_t,
    const ur_kernel_native_properties_t *, ur_kernel_handle_t *phKernel) {
  UR_ASSERT(hNativeKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  *phKernel = reinterpret_cast<ur_kernel_handle_t>(hNativeKernel);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgMemObj(ur_kernel_handle_t hKernel, uint32_t argIndex,
                     const ur_kernel_arg_mem_obj_properties_t *pProperties,
                     ur_mem_handle_t hArgValue) {

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  cl_int ret_err = clSetKernelArg(
      cl_adapter::cast<cl_kernel>(hKernel), cl_adapter::cast<cl_uint>(argIndex),
      sizeof(hArgValue), cl_adapter::cast<const cl_mem *>(hArgValue));
  CL_RETURN_ON_FAILURE(ret_err);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgSampler(
    ur_kernel_handle_t hKernel, uint32_t argIndex,
    const ur_kernel_arg_sampler_properties_t *, ur_sampler_handle_t hArgValue) {

  cl_int ret_err = clSetKernelArg(
      cl_adapter::cast<cl_kernel>(hKernel), cl_adapter::cast<cl_uint>(argIndex),
      sizeof(hArgValue), cl_adapter::cast<const cl_sampler *>(&hArgValue));
  CL_RETURN_ON_FAILURE(ret_err);
  return UR_RESULT_SUCCESS;
}
