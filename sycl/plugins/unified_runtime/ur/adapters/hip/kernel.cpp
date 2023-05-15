//===--------- kernel.cpp - HIP Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "kernel.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(ur_program_handle_t hProgram, const char *pKernelName,
               ur_kernel_handle_t *phKernel) {
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pKernelName, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(phKernel, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ur_result_t retErr = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_kernel_handle_t_> retKernel{nullptr};

  try {
    ScopedContext active(hProgram->get_context());

    hipFunction_t hipFunc;
    retErr = UR_CHECK_ERROR(
        hipModuleGetFunction(&hipFunc, hProgram->get(), pKernelName));

    std::string kernel_name_woffset = std::string(pKernelName) + "_with_offset";
    hipFunction_t hipFuncWithOffsetParam;
    hipError_t offsetRes = hipModuleGetFunction(
        &hipFuncWithOffsetParam, hProgram->get(), kernel_name_woffset.c_str());

    // If there is no kernel with global offset parameter we mark it as missing
    if (offsetRes == hipErrorNotFound) {
      hipFuncWithOffsetParam = nullptr;
    } else {
      retErr = UR_CHECK_ERROR(offsetRes);
    }
    retKernel = std::unique_ptr<ur_kernel_handle_t_>(
        new ur_kernel_handle_t_{hipFunc, hipFuncWithOffsetParam, pKernelName,
                                hProgram, hProgram->get_context()});
  } catch (ur_result_t err) {
    retErr = err;
  } catch (...) {
    retErr = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  *phKernel = retKernel.release();
  return retErr;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetGroupInfo(ur_kernel_handle_t hKernel, ur_device_handle_t hDevice,
                     ur_kernel_group_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // Here we want to query about a kernel's cuda blocks!
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
    size_t global_work_size[3] = {0, 0, 0};

    int max_block_dimX{0}, max_block_dimY{0}, max_block_dimZ{0};
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_block_dimX, hipDeviceAttributeMaxBlockDimX,
                              hDevice->get()) == hipSuccess);
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_block_dimY, hipDeviceAttributeMaxBlockDimY,
                              hDevice->get()) == hipSuccess);
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_block_dimZ, hipDeviceAttributeMaxBlockDimZ,
                              hDevice->get()) == hipSuccess);

    int max_grid_dimX{0}, max_grid_dimY{0}, max_grid_dimZ{0};
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_grid_dimX, hipDeviceAttributeMaxGridDimX,
                              hDevice->get()) == hipSuccess);
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_grid_dimY, hipDeviceAttributeMaxGridDimY,
                              hDevice->get()) == hipSuccess);
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&max_grid_dimZ, hipDeviceAttributeMaxGridDimZ,
                              hDevice->get()) == hipSuccess);

    global_work_size[0] = max_block_dimX * max_grid_dimX;
    global_work_size[1] = max_block_dimY * max_grid_dimY;
    global_work_size[2] = max_block_dimZ * max_grid_dimZ;
    return ReturnValue(global_work_size, 3);
  }
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    int max_threads = 0;
    sycl::detail::ur::assertion(
        hipFuncGetAttribute(&max_threads,
                            HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                            hKernel->get()) == hipSuccess);
    return ReturnValue(size_t(max_threads));
  }
  case UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
    size_t group_size[3] = {0, 0, 0};
    // Returns the work-group size specified in the kernel source or IL.
    // If the work-group size is not specified in the kernel source or IL,
    // (0, 0, 0) is returned.
    // https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/clGetKernelWorkGroupInfo.html

    // TODO: can we extract the work group size from the PTX?
    return ReturnValue(group_size, 3);
  }
  case UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
    // OpenCL LOCAL == HIP SHARED
    int bytes = 0;
    sycl::detail::ur::assertion(
        hipFuncGetAttribute(&bytes, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                            hKernel->get()) == hipSuccess);
    return ReturnValue(uint64_t(bytes));
  }
  case UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    // Work groups should be multiples of the warp size
    int warpSize = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize,
                              hDevice->get()) == hipSuccess);
    return ReturnValue(static_cast<size_t>(warpSize));
  }
  case UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
    // OpenCL PRIVATE == HIP LOCAL
    int bytes = 0;
    sycl::detail::ur::assertion(
        hipFuncGetAttribute(&bytes, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                            hKernel->get()) == hipSuccess);
    return ReturnValue(uint64_t(bytes));
  }
  default:
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelRetain(ur_kernel_handle_t hKernel) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hKernel->get_reference_count() > 0u,
            UR_RESULT_ERROR_INVALID_KERNEL);

  hKernel->increment_reference_count();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelRelease(ur_kernel_handle_t hKernel) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  UR_ASSERT(hKernel->get_reference_count() != 0,
            UR_RESULT_ERROR_INVALID_KERNEL);

  // decrement ref count. If it is 0, delete the program.
  if (hKernel->decrement_reference_count() == 0) {
    // no internal cuda resources to clean up. Just delete it.
    delete hKernel;
    return UR_RESULT_SUCCESS;
  }

  return UR_RESULT_SUCCESS;
}

// TODO(ur): Not implemented on hip atm. Also, need to add tests for this
// feature.
UR_APIEXPORT ur_result_t UR_APICALL urKernelGetNativeHandle(
    ur_kernel_handle_t hKernel, ur_native_handle_t *phNativeKernel) {
  (void)hKernel;
  (void)phNativeKernel;

  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgValue(ur_kernel_handle_t hKernel, uint32_t argIndex,
                    size_t argSize, const void *pArgValue) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  ur_result_t retErr = UR_RESULT_SUCCESS;
  try {
    if (pArgValue) {
      hKernel->set_kernel_arg(argIndex, argSize, pArgValue);
    } else {
      hKernel->set_kernel_local_arg(argIndex, argSize);
    }
  } catch (ur_result_t err) {
    retErr = err;
  }
  return retErr;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetInfo(ur_kernel_handle_t hKernel,
                                                    ur_kernel_info_t propName,
                                                    size_t propSize,
                                                    void *pKernelInfo,
                                                    size_t *pPropSizeRet) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper ReturnValue(propSize, pKernelInfo, pPropSizeRet);

  switch (propName) {
  case UR_KERNEL_INFO_FUNCTION_NAME:
    return ReturnValue(hKernel->get_name());
  case UR_KERNEL_INFO_NUM_ARGS:
    return ReturnValue(hKernel->get_num_args());
  case UR_KERNEL_INFO_REFERENCE_COUNT:
    return ReturnValue(hKernel->get_reference_count());
  case UR_KERNEL_INFO_CONTEXT:
    return ReturnValue(hKernel->get_context());
  case UR_KERNEL_INFO_PROGRAM:
    return ReturnValue(hKernel->get_program());
  case UR_KERNEL_INFO_ATTRIBUTES:
    return ReturnValue("");
  default:
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetSubGroupInfo(ur_kernel_handle_t hKernel, ur_device_handle_t hDevice,
                        ur_kernel_sub_group_info_t propName, size_t propSize,
                        void *pPropValue, size_t *pPropSizeRet) {
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE: {
    // Sub-group size is equivalent to warp size
    int warpSize = 0;
    sycl::detail::ur::assertion(
        hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize,
                              hDevice->get()) == hipSuccess);
    return ReturnValue(static_cast<uint32_t>(warpSize));
  }
  case UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS: {
    // Number of sub-groups = max block size / warp size + possible remainder
    int max_threads = 0;
    sycl::detail::ur::assertion(
        hipFuncGetAttribute(&max_threads,
                            HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                            hKernel->get()) == hipSuccess);
    int warpSize = 0;
    urKernelGetSubGroupInfo(hKernel, hDevice,
                            UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE,
                            sizeof(uint32_t), &warpSize, nullptr);
    int maxWarps = (max_threads + warpSize - 1) / warpSize;
    return ReturnValue(static_cast<uint32_t>(maxWarps));
  }
  case UR_KERNEL_SUB_GROUP_INFO_COMPILE_NUM_SUB_GROUPS: {
    // Return value of 0 => not specified
    // TODO: Revisit if PTX is generated for compile-time work-group sizes
    return ReturnValue(0);
  }
  case UR_KERNEL_SUB_GROUP_INFO_SUB_GROUP_SIZE_INTEL: {
    // Return value of 0 => unspecified or "auto" sub-group size
    // Correct for now, since warp size may be read from special register
    // TODO: Return warp size once default is primary sub-group size
    // TODO: Revisit if we can recover [[sub_group_size]] attribute from PTX
    return ReturnValue(0);
  }
  default:
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, uint32_t argIndex, const void *pArgValue) {
  hKernel->set_kernel_arg(argIndex, sizeof(pArgValue), pArgValue);
  return UR_RESULT_SUCCESS;
}

// A NOP for the HIP backend
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetExecInfo(ur_kernel_handle_t hKernel, ur_kernel_exec_info_t propName,
                    size_t propSize, const void *pPropValue) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t hNativeKernel, ur_context_handle_t hContext,
    ur_program_handle_t hProgram,
    const ur_kernel_native_properties_t *pProperties,
    ur_kernel_handle_t *phKernel) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
