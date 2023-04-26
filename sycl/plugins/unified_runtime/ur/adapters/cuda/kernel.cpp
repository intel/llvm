//===--------- kernel.cpp - CUDA Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "kernel.hpp"
#include "memory.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(ur_program_handle_t hProgram, const char *pKernelName,
               ur_kernel_handle_t *phKernel) {
  UR_ASSERT(hProgram, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phKernel, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ur_result_t retErr = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_kernel_handle_t_> retKernel{nullptr};

  try {
    ScopedContext active(hProgram->get_context());

    CUfunction cuFunc;
    retErr = UR_CHECK_ERROR(
        cuModuleGetFunction(&cuFunc, hProgram->get(), pKernelName));

    std::string kernel_name_woffset = std::string(pKernelName) + "_with_offset";
    CUfunction cuFuncWithOffsetParam;
    CUresult offsetRes = cuModuleGetFunction(
        &cuFuncWithOffsetParam, hProgram->get(), kernel_name_woffset.c_str());

    // If there is no kernel with global offset parameter we mark it as missing
    if (offsetRes == CUDA_ERROR_NOT_FOUND) {
      cuFuncWithOffsetParam = nullptr;
    } else {
      retErr = UR_CHECK_ERROR(offsetRes);
    }
    retKernel = std::unique_ptr<ur_kernel_handle_t_>(
        new ur_kernel_handle_t_{cuFunc, cuFuncWithOffsetParam, pKernelName,
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
        cuDeviceGetAttribute(&max_block_dimX,
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                             hDevice->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_block_dimY,
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                             hDevice->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_block_dimZ,
                             CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                             hDevice->get()) == CUDA_SUCCESS);

    int max_grid_dimX{0}, max_grid_dimY{0}, max_grid_dimZ{0};
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_grid_dimX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                             hDevice->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_grid_dimY, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                             hDevice->get()) == CUDA_SUCCESS);
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&max_grid_dimZ, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                             hDevice->get()) == CUDA_SUCCESS);

    global_work_size[0] = max_block_dimX * max_grid_dimX;
    global_work_size[1] = max_block_dimY * max_grid_dimY;
    global_work_size[2] = max_block_dimZ * max_grid_dimZ;
    return ReturnValue(global_work_size, 3);
  }
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    int max_threads = 0;
    sycl::detail::ur::assertion(
        cuFuncGetAttribute(&max_threads,
                           CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                           hKernel->get()) == CUDA_SUCCESS);
    return ReturnValue(size_t(max_threads));
  }
  case UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
    size_t group_size[3] = {0, 0, 0};
    const auto &reqd_wg_size_md_map =
        hKernel->program_->kernelReqdWorkGroupSizeMD_;
    const auto reqd_wg_size_md = reqd_wg_size_md_map.find(hKernel->name_);
    if (reqd_wg_size_md != reqd_wg_size_md_map.end()) {
      const auto reqd_wg_size = reqd_wg_size_md->second;
      group_size[0] = std::get<0>(reqd_wg_size);
      group_size[1] = std::get<1>(reqd_wg_size);
      group_size[2] = std::get<2>(reqd_wg_size);
    }
    return ReturnValue(group_size, 3);
  }
  case UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
    // OpenCL LOCAL == CUDA SHARED
    int bytes = 0;
    sycl::detail::ur::assertion(
        cuFuncGetAttribute(&bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                           hKernel->get()) == CUDA_SUCCESS);
    return ReturnValue(uint64_t(bytes));
  }
  case UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    // Work groups should be multiples of the warp size
    int warpSize = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                             hDevice->get()) == CUDA_SUCCESS);
    return ReturnValue(static_cast<size_t>(warpSize));
  }
  case UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
    // OpenCL PRIVATE == CUDA LOCAL
    int bytes = 0;
    sycl::detail::ur::assertion(
        cuFuncGetAttribute(&bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                           hKernel->get()) == CUDA_SUCCESS);
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

// TODO(ur): Not implemented on cuda atm. Also, need to add tests for this
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
  case UR_KERNEL_INFO_NUM_REGS: {
    int numRegs = 0;
    sycl::detail::ur::assertion(
        cuFuncGetAttribute(&numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS,
                           hKernel->get()) == CUDA_SUCCESS);
    return ReturnValue(static_cast<uint32_t>(numRegs));
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
  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE: {
    // Sub-group size is equivalent to warp size
    int warpSize = 0;
    sycl::detail::ur::assertion(
        cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                             hDevice->get()) == CUDA_SUCCESS);
    return ReturnValue(static_cast<uint32_t>(warpSize));
  }
  case UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS: {
    // Number of sub-groups = max block size / warp size + possible remainder
    int max_threads = 0;
    sycl::detail::ur::assertion(
        cuFuncGetAttribute(&max_threads,
                           CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                           hKernel->get()) == CUDA_SUCCESS);
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

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgMemObj(
    ur_kernel_handle_t hKernel, uint32_t argIndex, ur_mem_handle_t hArgValue) {

  UR_ASSERT(hKernel, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hArgValue, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  ur_result_t retErr = UR_RESULT_SUCCESS;
  try {
    if (hArgValue->mem_type_ == ur_mem_handle_t_::mem_type::surface) {
      CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
      UR_CHECK_ERROR(cuArray3DGetDescriptor(
          &arrayDesc, hArgValue->mem_.surface_mem_.get_array()));
      if (arrayDesc.Format != CU_AD_FORMAT_UNSIGNED_INT32 &&
          arrayDesc.Format != CU_AD_FORMAT_SIGNED_INT32 &&
          arrayDesc.Format != CU_AD_FORMAT_HALF &&
          arrayDesc.Format != CU_AD_FORMAT_FLOAT) {
        setErrorMessage("PI CUDA kernels only support images with channel "
                        "types int32, uint32, float, and half.",
                        UR_RESULT_ERROR_ADAPTER_SPECIFIC);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }
      CUsurfObject cuSurf = hArgValue->mem_.surface_mem_.get_surface();
      hKernel->set_kernel_arg(argIndex, sizeof(cuSurf), (void *)&cuSurf);
    } else {
      CUdeviceptr cuPtr = hArgValue->mem_.buffer_mem_.get();
      hKernel->set_kernel_arg(argIndex, sizeof(CUdeviceptr), (void *)&cuPtr);
    }
  } catch (ur_result_t err) {
    retErr = err;
  }
  return retErr;
}

// A NOP for the CUDA backend
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
