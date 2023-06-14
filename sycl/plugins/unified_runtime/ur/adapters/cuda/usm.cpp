//===--------- usm.cpp - CUDA Adapter ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <cassert>

#include "common.hpp"
#include "context.hpp"
#include "device.hpp"
#include "event.hpp"
#include "platform.hpp"
#include "queue.hpp"

#include <cuda.h>

/// USM: Implements USM Host allocations using CUDA Pinned Memory
/// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#page-locked-host-memory
UR_APIEXPORT ur_result_t UR_APICALL urUSMHostAlloc(
    ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
    [[maybe_unused]] ur_usm_pool_handle_t pool, size_t size, void **ppMem) {
  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  size_t DeviceMaxMemAllocSize = 0;
  UR_ASSERT(urDeviceGetInfo(hContext->getDevice(),
                            UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, sizeof(size_t),
                            static_cast<void *>(&DeviceMaxMemAllocSize),
                            nullptr) == UR_RESULT_SUCCESS,
            UR_RESULT_ERROR_INVALID_DEVICE);
  UR_ASSERT(size > 0 && size <= DeviceMaxMemAllocSize,
            UR_RESULT_ERROR_INVALID_USM_SIZE);

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    ScopedContext Active(hContext);
    Result = UR_CHECK_ERROR(cuMemAllocHost(ppMem, size));
  } catch (ur_result_t Err) {
    Result = Err;
  }

  UR_ASSERT(!pUSMDesc || (pUSMDesc->align == 0 ||
                          ((pUSMDesc->align & (pUSMDesc->align - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  assert(Result == UR_RESULT_SUCCESS &&
         (!pUSMDesc || pUSMDesc->align == 0 ||
          reinterpret_cast<std::uintptr_t>(*ppMem) % pUSMDesc->align == 0));

  return Result;
}

/// USM: Implements USM device allocations using a normal CUDA device pointer
///
UR_APIEXPORT ur_result_t UR_APICALL urUSMDeviceAlloc(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_usm_desc_t *pUSMDesc, [[maybe_unused]] ur_usm_pool_handle_t pool,
    size_t size, void **ppMem) {
  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(!pUSMDesc || (pUSMDesc->align == 0 ||
                          ((pUSMDesc->align & (pUSMDesc->align - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  size_t DeviceMaxMemAllocSize = 0;
  UR_ASSERT(urDeviceGetInfo(hDevice, UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE,
                            sizeof(size_t),
                            static_cast<void *>(&DeviceMaxMemAllocSize),
                            nullptr) == UR_RESULT_SUCCESS,
            UR_RESULT_ERROR_INVALID_DEVICE);
  UR_ASSERT(size > 0 && size <= DeviceMaxMemAllocSize,
            UR_RESULT_ERROR_INVALID_USM_SIZE);

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    ScopedContext Active(hContext);
    Result = UR_CHECK_ERROR(cuMemAlloc((CUdeviceptr *)ppMem, size));
  } catch (ur_result_t Err) {
    return Err;
  }

  assert(Result == UR_RESULT_SUCCESS &&
         (!pUSMDesc || pUSMDesc->align == 0 ||
          reinterpret_cast<std::uintptr_t>(*ppMem) % pUSMDesc->align == 0));

  return Result;
}

/// USM: Implements USM Shared allocations using CUDA Managed Memory
///
UR_APIEXPORT ur_result_t UR_APICALL urUSMSharedAlloc(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_usm_desc_t *pUSMDesc, [[maybe_unused]] ur_usm_pool_handle_t pool,
    size_t size, void **ppMem) {
  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(!pUSMDesc || (pUSMDesc->align == 0 ||
                          ((pUSMDesc->align & (pUSMDesc->align - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  size_t DeviceMaxMemAllocSize = 0;
  UR_ASSERT(urDeviceGetInfo(hDevice, UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE,
                            sizeof(size_t),
                            static_cast<void *>(&DeviceMaxMemAllocSize),
                            nullptr) == UR_RESULT_SUCCESS,
            UR_RESULT_ERROR_INVALID_DEVICE);
  UR_ASSERT(size > 0 && size <= DeviceMaxMemAllocSize,
            UR_RESULT_ERROR_INVALID_USM_SIZE);

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    ScopedContext Active(hContext);
    Result = UR_CHECK_ERROR(
        cuMemAllocManaged((CUdeviceptr *)ppMem, size, CU_MEM_ATTACH_GLOBAL));
  } catch (ur_result_t Err) {
    return Err;
  }

  assert(Result == UR_RESULT_SUCCESS &&
         (!pUSMDesc || pUSMDesc->align == 0 ||
          reinterpret_cast<std::uintptr_t>(*ppMem) % pUSMDesc->align == 0));

  return Result;
}

/// USM: Frees the given USM pointer associated with the context.
///
UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    ScopedContext Active(hContext);
    bool IsManaged;
    unsigned int Type;
    void *AttributeValues[2] = {&IsManaged, &Type};
    CUpointer_attribute Attributes[2] = {CU_POINTER_ATTRIBUTE_IS_MANAGED,
                                         CU_POINTER_ATTRIBUTE_MEMORY_TYPE};
    Result = UR_CHECK_ERROR(cuPointerGetAttributes(
        2, Attributes, AttributeValues, (CUdeviceptr)pMem));
    UR_ASSERT(Type == CU_MEMORYTYPE_DEVICE || Type == CU_MEMORYTYPE_HOST,
              UR_RESULT_ERROR_INVALID_MEM_OBJECT);
    if (IsManaged || Type == CU_MEMORYTYPE_DEVICE) {
      // Memory allocated with cuMemAlloc and cuMemAllocManaged must be freed
      // with cuMemFree
      Result = UR_CHECK_ERROR(cuMemFree((CUdeviceptr)pMem));
    } else {
      // Memory allocated with cuMemAllocHost must be freed with cuMemFreeHost
      Result = UR_CHECK_ERROR(cuMemFreeHost(pMem));
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propValueSize,
                     void *pPropValue, size_t *pPropValueSizeRet) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ur_result_t Result = UR_RESULT_SUCCESS;

  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);

  try {
    ScopedContext Active(hContext);
    switch (propName) {
    case UR_USM_ALLOC_INFO_TYPE: {
      unsigned int Value;
      // do not throw if cuPointerGetAttribute returns CUDA_ERROR_INVALID_VALUE
      CUresult Ret = cuPointerGetAttribute(
          &Value, CU_POINTER_ATTRIBUTE_IS_MANAGED, (CUdeviceptr)pMem);
      if (Ret == CUDA_ERROR_INVALID_VALUE) {
        // pointer not known to the CUDA subsystem
        return ReturnValue(UR_USM_TYPE_UNKNOWN);
      }
      Result = checkErrorUR(Ret, __func__, __LINE__ - 5, __FILE__);
      if (Value) {
        // pointer to managed memory
        return ReturnValue(UR_USM_TYPE_SHARED);
      }
      Result = UR_CHECK_ERROR(cuPointerGetAttribute(
          &Value, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)pMem));
      UR_ASSERT(Value == CU_MEMORYTYPE_DEVICE || Value == CU_MEMORYTYPE_HOST,
                UR_RESULT_ERROR_INVALID_MEM_OBJECT);
      if (Value == CU_MEMORYTYPE_DEVICE) {
        // pointer to device memory
        return ReturnValue(UR_USM_TYPE_DEVICE);
      }
      if (Value == CU_MEMORYTYPE_HOST) {
        // pointer to host memory
        return ReturnValue(UR_USM_TYPE_HOST);
      }
      // should never get here
#ifdef _MSC_VER
      __assume(0);
#else
      __builtin_unreachable();
#endif
    }
    case UR_USM_ALLOC_INFO_BASE_PTR: {
#if __CUDA_API_VERSION >= 10020
      // CU_POINTER_ATTRIBUTE_RANGE_START_ADDR was introduced in CUDA 10.2
      unsigned int Value;
      result = UR_CHECK_ERROR(cuPointerGetAttribute(
          &Value, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)pMem));
      return ReturnValue(Value);
#else
      return UR_RESULT_ERROR_INVALID_VALUE;
#endif
    }
    case UR_USM_ALLOC_INFO_SIZE: {
#if __CUDA_API_VERSION >= 10020
      // CU_POINTER_ATTRIBUTE_RANGE_SIZE was introduced in CUDA 10.2
      unsigned int Value;
      result = UR_CHECK_ERROR(cuPointerGetAttribute(
          &Value, CU_POINTER_ATTRIBUTE_RANGE_SIZE, (CUdeviceptr)pMem));
      return ReturnValue(Value);
#else
      return UR_RESULT_ERROR_INVALID_VALUE;
#endif
    }
    case UR_USM_ALLOC_INFO_DEVICE: {
      // get device index associated with this pointer
      unsigned int DeviceIndex;
      Result = UR_CHECK_ERROR(cuPointerGetAttribute(
          &DeviceIndex, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
          (CUdeviceptr)pMem));

      // currently each device is in its own platform, so find the platform at
      // the same index
      std::vector<ur_platform_handle_t> Platforms;
      Platforms.resize(DeviceIndex + 1);
      Result = urPlatformGet(DeviceIndex + 1, Platforms.data(), nullptr);

      // get the device from the platform
      ur_device_handle_t Device = Platforms[DeviceIndex]->Devices[0].get();
      return ReturnValue(Device);
    }
    default:
      return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}
