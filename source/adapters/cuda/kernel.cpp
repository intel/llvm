//===--------- kernel.cpp - CUDA Adapter ----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernel.hpp"
#include "memory.hpp"
#include "sampler.hpp"

UR_APIEXPORT ur_result_t UR_APICALL
urKernelCreate(ur_program_handle_t hProgram, const char *pKernelName,
               ur_kernel_handle_t *phKernel) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  std::unique_ptr<ur_kernel_handle_t_> Kernel{nullptr};

  try {
    ScopedContext Active(hProgram->getContext());

    CUfunction CuFunc;
    CUresult FunctionResult =
        cuModuleGetFunction(&CuFunc, hProgram->get(), pKernelName);

    // We can't add this as a generic mapping in UR_CHECK_ERROR since cuda's
    // NOT_FOUND error applies to more than just functions.
    if (FunctionResult == CUDA_ERROR_NOT_FOUND) {
      throw UR_RESULT_ERROR_INVALID_KERNEL_NAME;
    } else {
      UR_CHECK_ERROR(FunctionResult);
    }

    std::string KernelNameWithOffset =
        std::string(pKernelName) + "_with_offset";
    CUfunction CuFuncWithOffsetParam;
    CUresult OffsetRes = cuModuleGetFunction(
        &CuFuncWithOffsetParam, hProgram->get(), KernelNameWithOffset.c_str());

    // If there is no kernel with global offset parameter we mark it as missing
    if (OffsetRes == CUDA_ERROR_NOT_FOUND) {
      CuFuncWithOffsetParam = nullptr;
    } else {
      UR_CHECK_ERROR(OffsetRes);
    }
    Kernel = std::unique_ptr<ur_kernel_handle_t_>(
        new ur_kernel_handle_t_{CuFunc, CuFuncWithOffsetParam, pKernelName,
                                hProgram, hProgram->getContext()});
  } catch (ur_result_t Err) {
    Result = Err;
  } catch (...) {
    Result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  *phKernel = Kernel.release();
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetGroupInfo(ur_kernel_handle_t hKernel, ur_device_handle_t hDevice,
                     ur_kernel_group_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_KERNEL_GROUP_INFO_GLOBAL_WORK_SIZE: {
    size_t GlobalWorkSize[3] = {0, 0, 0};

    int MaxBlockDimX{0}, MaxBlockDimY{0}, MaxBlockDimZ{0};
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxBlockDimX, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, hDevice->get()));
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxBlockDimY, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, hDevice->get()));
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxBlockDimZ, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, hDevice->get()));

    int MaxGridDimX{0}, MaxGridDimY{0}, MaxGridDimZ{0};
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxGridDimX, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, hDevice->get()));
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxGridDimY, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, hDevice->get()));
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &MaxGridDimZ, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, hDevice->get()));

    GlobalWorkSize[0] = MaxBlockDimX * MaxGridDimX;
    GlobalWorkSize[1] = MaxBlockDimY * MaxGridDimY;
    GlobalWorkSize[2] = MaxBlockDimZ * MaxGridDimZ;
    return ReturnValue(GlobalWorkSize, 3);
  }
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    int MaxThreads = 0;
    UR_CHECK_ERROR(cuFuncGetAttribute(
        &MaxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, hKernel->get()));
    return ReturnValue(size_t(MaxThreads));
  }
  case UR_KERNEL_GROUP_INFO_COMPILE_WORK_GROUP_SIZE: {
    size_t GroupSize[3] = {0, 0, 0};
    const auto &ReqdWGSizeMDMap =
        hKernel->get_program()->KernelReqdWorkGroupSizeMD;
    const auto ReqdWGSizeMD = ReqdWGSizeMDMap.find(hKernel->getName());
    if (ReqdWGSizeMD != ReqdWGSizeMDMap.end()) {
      const auto ReqdWGSize = ReqdWGSizeMD->second;
      GroupSize[0] = std::get<0>(ReqdWGSize);
      GroupSize[1] = std::get<1>(ReqdWGSize);
      GroupSize[2] = std::get<2>(ReqdWGSize);
    }
    return ReturnValue(GroupSize, 3);
  }
  case UR_KERNEL_GROUP_INFO_LOCAL_MEM_SIZE: {
    // OpenCL LOCAL == CUDA SHARED
    int Bytes = 0;
    UR_CHECK_ERROR(cuFuncGetAttribute(
        &Bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, hKernel->get()));
    return ReturnValue(uint64_t(Bytes));
  }
  case UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    // Work groups should be multiples of the warp size
    int WarpSize = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &WarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, hDevice->get()));
    return ReturnValue(static_cast<size_t>(WarpSize));
  }
  case UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
    // OpenCL PRIVATE == CUDA LOCAL
    int Bytes = 0;
    UR_CHECK_ERROR(cuFuncGetAttribute(
        &Bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, hKernel->get()));
    return ReturnValue(uint64_t(Bytes));
  }
  default:
    break;
  }

  return UR_RESULT_ERROR_INVALID_ENUMERATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelRetain(ur_kernel_handle_t hKernel) {
  UR_ASSERT(hKernel->getReferenceCount() > 0u, UR_RESULT_ERROR_INVALID_KERNEL);

  hKernel->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelRelease(ur_kernel_handle_t hKernel) {
  // double delete or someone is messing with the ref count.
  // either way, cannot safely proceed.
  UR_ASSERT(hKernel->getReferenceCount() != 0, UR_RESULT_ERROR_INVALID_KERNEL);

  // decrement ref count. If it is 0, delete the program.
  if (hKernel->decrementReferenceCount() == 0) {
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

UR_APIEXPORT ur_result_t UR_APICALL urKernelSuggestMaxCooperativeGroupCountExp(
    ur_kernel_handle_t hKernel, size_t localWorkSize,
    size_t dynamicSharedMemorySize, uint32_t *pGroupCountRet) {
  (void)hKernel;
  (void)localWorkSize;
  (void)dynamicSharedMemorySize;
  *pGroupCountRet = 1;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgValue(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_value_properties_t *pProperties,
    const void *pArgValue) {
  std::ignore = pProperties;
  UR_ASSERT(argSize, UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    hKernel->setKernelArg(argIndex, argSize, pArgValue);
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgLocal(
    ur_kernel_handle_t hKernel, uint32_t argIndex, size_t argSize,
    const ur_kernel_arg_local_properties_t *pProperties) {
  std::ignore = pProperties;
  UR_ASSERT(argSize, UR_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE);

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    hKernel->setKernelLocalArg(argIndex, argSize);
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelGetInfo(ur_kernel_handle_t hKernel,
                                                    ur_kernel_info_t propName,
                                                    size_t propSize,
                                                    void *pKernelInfo,
                                                    size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pKernelInfo, pPropSizeRet);

  switch (propName) {
  case UR_KERNEL_INFO_FUNCTION_NAME:
    return ReturnValue(hKernel->getName());
  case UR_KERNEL_INFO_NUM_ARGS:
    return ReturnValue(hKernel->getNumArgs());
  case UR_KERNEL_INFO_REFERENCE_COUNT:
    return ReturnValue(hKernel->getReferenceCount());
  case UR_KERNEL_INFO_CONTEXT:
    return ReturnValue(hKernel->getContext());
  case UR_KERNEL_INFO_PROGRAM:
    return ReturnValue(hKernel->get_program());
  case UR_KERNEL_INFO_ATTRIBUTES:
    return ReturnValue("");
  case UR_KERNEL_INFO_NUM_REGS: {
    int NumRegs = 0;
    UR_CHECK_ERROR(cuFuncGetAttribute(&NumRegs, CU_FUNC_ATTRIBUTE_NUM_REGS,
                                      hKernel->get()));
    return ReturnValue(static_cast<uint32_t>(NumRegs));
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
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE: {
    // Sub-group size is equivalent to warp size
    int WarpSize = 0;
    UR_CHECK_ERROR(cuDeviceGetAttribute(
        &WarpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, hDevice->get()));
    return ReturnValue(static_cast<uint32_t>(WarpSize));
  }
  case UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS: {
    // Number of sub-groups = max block size / warp size + possible remainder
    int MaxThreads = 0;
    UR_CHECK_ERROR(cuFuncGetAttribute(
        &MaxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, hKernel->get()));
    int WarpSize = 0;
    urKernelGetSubGroupInfo(hKernel, hDevice,
                            UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE,
                            sizeof(uint32_t), &WarpSize, nullptr);
    int MaxWarps = (MaxThreads + WarpSize - 1) / WarpSize;
    return ReturnValue(static_cast<uint32_t>(MaxWarps));
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

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgPointer(ur_kernel_handle_t hKernel, uint32_t argIndex,
                      const ur_kernel_arg_pointer_properties_t *pProperties,
                      const void *pArgValue) {
  std::ignore = pProperties;
  hKernel->setKernelArg(argIndex, sizeof(pArgValue), pArgValue);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgMemObj(ur_kernel_handle_t hKernel, uint32_t argIndex,
                     const ur_kernel_arg_mem_obj_properties_t *Properties,
                     ur_mem_handle_t hArgValue) {
  std::ignore = Properties;

  // Below sets kernel arg when zero-sized buffers are handled.
  // In such case the corresponding memory is null.
  if (hArgValue == nullptr) {
    hKernel->setKernelArg(argIndex, 0, nullptr);
    return UR_RESULT_SUCCESS;
  }

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    if (hArgValue->MemType == ur_mem_handle_t_::Type::Surface) {
      CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
      UR_CHECK_ERROR(cuArray3DGetDescriptor(
          &arrayDesc, std::get<SurfaceMem>(hArgValue->Mem).getArray()));
      if (arrayDesc.Format != CU_AD_FORMAT_UNSIGNED_INT32 &&
          arrayDesc.Format != CU_AD_FORMAT_SIGNED_INT32 &&
          arrayDesc.Format != CU_AD_FORMAT_HALF &&
          arrayDesc.Format != CU_AD_FORMAT_FLOAT) {
        setErrorMessage("PI CUDA kernels only support images with channel "
                        "types int32, uint32, float, and half.",
                        UR_RESULT_ERROR_ADAPTER_SPECIFIC);
        return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
      }
      CUsurfObject CuSurf = std::get<SurfaceMem>(hArgValue->Mem).getSurface();
      hKernel->setKernelArg(argIndex, sizeof(CuSurf), (void *)&CuSurf);
    } else {
      CUdeviceptr CuPtr = std::get<BufferMem>(hArgValue->Mem).get();
      hKernel->setKernelArg(argIndex, sizeof(CUdeviceptr), (void *)&CuPtr);
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

// A NOP for the CUDA backend
UR_APIEXPORT ur_result_t UR_APICALL urKernelSetExecInfo(
    ur_kernel_handle_t hKernel, ur_kernel_exec_info_t propName, size_t propSize,
    const ur_kernel_exec_info_properties_t *pProperties,
    const void *pPropValue) {
  std::ignore = hKernel;
  std::ignore = propSize;
  std::ignore = pPropValue;
  std::ignore = pProperties;

  switch (propName) {
  case UR_KERNEL_EXEC_INFO_USM_INDIRECT_ACCESS:
  case UR_KERNEL_EXEC_INFO_USM_PTRS:
  case UR_KERNEL_EXEC_INFO_CACHE_CONFIG:
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }
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
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgSampler(ur_kernel_handle_t hKernel, uint32_t argIndex,
                      const ur_kernel_arg_sampler_properties_t *pProperties,
                      ur_sampler_handle_t hArgValue) {
  std::ignore = pProperties;

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    uint32_t SamplerProps = hArgValue->Props;
    hKernel->setKernelArg(argIndex, sizeof(uint32_t), (void *)&SamplerProps);
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}
