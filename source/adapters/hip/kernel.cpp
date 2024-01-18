//===--------- kernel.cpp - HIP Adapter -----------------------------------===//
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
  std::unique_ptr<ur_kernel_handle_t_> RetKernel{nullptr};

  try {
    ScopedContext Active(hProgram->getDevice());

    hipFunction_t HIPFunc;
    hipError_t KernelError =
        hipModuleGetFunction(&HIPFunc, hProgram->get(), pKernelName);
    if (KernelError == hipErrorNotFound) {
      return UR_RESULT_ERROR_INVALID_KERNEL_NAME;
    }
    UR_CHECK_ERROR(KernelError);

    std::string KernelNameWoffset = std::string(pKernelName) + "_with_offset";
    hipFunction_t HIPFuncWithOffsetParam;
    hipError_t OffsetRes = hipModuleGetFunction(
        &HIPFuncWithOffsetParam, hProgram->get(), KernelNameWoffset.c_str());

    // If there is no kernel with global offset parameter we mark it as missing
    if (OffsetRes == hipErrorNotFound) {
      HIPFuncWithOffsetParam = nullptr;
    } else {
      UR_CHECK_ERROR(OffsetRes);
    }
    RetKernel = std::unique_ptr<ur_kernel_handle_t_>(
        new ur_kernel_handle_t_{HIPFunc, HIPFuncWithOffsetParam, pKernelName,
                                hProgram, hProgram->getContext()});
  } catch (ur_result_t Err) {
    Result = Err;
  } catch (...) {
    Result = UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  }

  *phKernel = RetKernel.release();
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
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &MaxBlockDimX, hipDeviceAttributeMaxBlockDimX, hDevice->get()));
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &MaxBlockDimY, hipDeviceAttributeMaxBlockDimY, hDevice->get()));
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &MaxBlockDimZ, hipDeviceAttributeMaxBlockDimZ, hDevice->get()));

    int max_grid_dimX{0}, max_grid_dimY{0}, max_grid_dimZ{0};
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &max_grid_dimX, hipDeviceAttributeMaxGridDimX, hDevice->get()));
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &max_grid_dimY, hipDeviceAttributeMaxGridDimY, hDevice->get()));
    UR_CHECK_ERROR(hipDeviceGetAttribute(
        &max_grid_dimZ, hipDeviceAttributeMaxGridDimZ, hDevice->get()));

    GlobalWorkSize[0] = MaxBlockDimX * max_grid_dimX;
    GlobalWorkSize[1] = MaxBlockDimY * max_grid_dimY;
    GlobalWorkSize[2] = MaxBlockDimZ * max_grid_dimZ;
    return ReturnValue(GlobalWorkSize, 3);
  }
  case UR_KERNEL_GROUP_INFO_WORK_GROUP_SIZE: {
    int MaxThreads = 0;
    UR_CHECK_ERROR(hipFuncGetAttribute(
        &MaxThreads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, hKernel->get()));
    return ReturnValue(size_t(MaxThreads));
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
    int Bytes = 0;
    UR_CHECK_ERROR(hipFuncGetAttribute(
        &Bytes, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, hKernel->get()));
    return ReturnValue(uint64_t(Bytes));
  }
  case UR_KERNEL_GROUP_INFO_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: {
    // Work groups should be multiples of the warp size
    int WarpSize = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(&WarpSize, hipDeviceAttributeWarpSize,
                                         hDevice->get()));
    return ReturnValue(static_cast<size_t>(WarpSize));
  }
  case UR_KERNEL_GROUP_INFO_PRIVATE_MEM_SIZE: {
    // OpenCL PRIVATE == HIP LOCAL
    int Bytes = 0;
    UR_CHECK_ERROR(hipFuncGetAttribute(
        &Bytes, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, hKernel->get()));
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

// TODO(ur): Not implemented on hip atm. Also, need to add tests for this
// feature.
UR_APIEXPORT ur_result_t UR_APICALL
urKernelGetNativeHandle(ur_kernel_handle_t, ur_native_handle_t *) {
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
    const ur_kernel_arg_value_properties_t *, const void *pArgValue) {
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
    return ReturnValue(hKernel->getProgram());
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
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  switch (propName) {
  case UR_KERNEL_SUB_GROUP_INFO_MAX_SUB_GROUP_SIZE: {
    // Sub-group size is equivalent to warp size
    int WarpSize = 0;
    UR_CHECK_ERROR(hipDeviceGetAttribute(&WarpSize, hipDeviceAttributeWarpSize,
                                         hDevice->get()));
    return ReturnValue(static_cast<uint32_t>(WarpSize));
  }
  case UR_KERNEL_SUB_GROUP_INFO_MAX_NUM_SUB_GROUPS: {
    // Number of sub-groups = max block size / warp size + possible remainder
    int MaxThreads = 0;
    UR_CHECK_ERROR(hipFuncGetAttribute(
        &MaxThreads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, hKernel->get()));
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

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgPointer(
    ur_kernel_handle_t hKernel, uint32_t argIndex,
    const ur_kernel_arg_pointer_properties_t *, const void *pArgValue) {
  hKernel->setKernelArg(argIndex, sizeof(pArgValue), pArgValue);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetArgMemObj(ur_kernel_handle_t hKernel, uint32_t argIndex,
                     const ur_kernel_arg_mem_obj_properties_t *Properties,
                     ur_mem_handle_t hArgValue) {
  // Below sets kernel arg when zero-sized buffers are handled.
  // In such case the corresponding memory is null.
  if (hArgValue == nullptr) {
    hKernel->setKernelArg(argIndex, 0, nullptr);
    return UR_RESULT_SUCCESS;
  }

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    auto Device = hKernel->getProgram()->getDevice();
    hKernel->Args.addMemObjArg(argIndex, hArgValue, Properties->memoryAccess);
    if (hArgValue->isImage()) {
      auto array = std::get<SurfaceMem>(hArgValue->Mem).getArray(Device);
      hipArray_Format Format;
      size_t NumChannels;
      UR_CHECK_ERROR(getArrayDesc(array, Format, NumChannels));
      if (Format != HIP_AD_FORMAT_UNSIGNED_INT32 &&
          Format != HIP_AD_FORMAT_SIGNED_INT32 &&
          Format != HIP_AD_FORMAT_HALF && Format != HIP_AD_FORMAT_FLOAT) {
        detail::ur::die(
            "UR HIP kernels only support images with channel types int32, "
            "uint32, float, and half.");
      }
      hipSurfaceObject_t hipSurf =
          std::get<SurfaceMem>(hArgValue->Mem).getSurface(Device);
      hKernel->setKernelArg(argIndex, sizeof(hipSurf), (void *)&hipSurf);
    } else {
      void *HIPPtr = std::get<BufferMem>(hArgValue->Mem).getVoid(Device);
      hKernel->setKernelArg(argIndex, sizeof(void *), (void *)&HIPPtr);
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetArgSampler(
    ur_kernel_handle_t hKernel, uint32_t argIndex,
    const ur_kernel_arg_sampler_properties_t *, ur_sampler_handle_t hArgValue) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    uint32_t SamplerProps = hArgValue->Props;
    hKernel->setKernelArg(argIndex, sizeof(uint32_t), (void *)&SamplerProps);
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

// A NOP for the HIP backend
UR_APIEXPORT ur_result_t UR_APICALL
urKernelSetExecInfo(ur_kernel_handle_t, ur_kernel_exec_info_t, size_t,
                    const ur_kernel_exec_info_properties_t *, const void *) {
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelCreateWithNativeHandle(
    ur_native_handle_t, ur_context_handle_t, ur_program_handle_t,
    const ur_kernel_native_properties_t *, ur_kernel_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urKernelSetSpecializationConstants(
    [[maybe_unused]] ur_kernel_handle_t hKernel,
    [[maybe_unused]] uint32_t count,
    [[maybe_unused]] const ur_specialization_constant_info_t *pSpecConstants) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
