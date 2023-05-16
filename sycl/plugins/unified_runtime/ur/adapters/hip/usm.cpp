//===--------- usm.cpp - HIP Adapter ------------------------------===//
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
#include "platform.hpp"

/// USM: Implements USM Host allocations using HIP Pinned Memory
///
UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
               ur_usm_pool_handle_t pool, size_t size, void **ppMem) {
  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  size_t device_max_mem_alloc_size = 0;
  UR_ASSERT(urDeviceGetInfo(hContext->get_device(),
                            UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, sizeof(size_t),
                            static_cast<void *>(&device_max_mem_alloc_size),
                            nullptr) == UR_RESULT_SUCCESS,
            UR_RESULT_ERROR_INVALID_DEVICE);
  UR_ASSERT(size > 0 && size <= device_max_mem_alloc_size,
            UR_RESULT_ERROR_INVALID_USM_SIZE);

  ur_result_t result = UR_RESULT_SUCCESS;
  try {
    ScopedContext active(hContext);
    result = UR_CHECK_ERROR(hipHostMalloc(ppMem, size));
  } catch (ur_result_t error) {
    result = error;
  }

  UR_ASSERT(!pUSMDesc || (pUSMDesc->align == 0 ||
                          ((pUSMDesc->align & (pUSMDesc->align - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  assert(result == UR_RESULT_SUCCESS &&
         (!pUSMDesc || pUSMDesc->align == 0 ||
          reinterpret_cast<std::uintptr_t>(*ppMem) % pUSMDesc->align == 0));

  return result;
}

/// USM: Implements USM device allocations using a normal HIP device pointer
///
UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t pool,
                 size_t size, void **ppMem) {
  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  size_t device_max_mem_alloc_size = 0;
  UR_ASSERT(urDeviceGetInfo(hDevice, UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE,
                            sizeof(size_t),
                            static_cast<void *>(&device_max_mem_alloc_size),
                            nullptr) == UR_RESULT_SUCCESS,
            UR_RESULT_ERROR_INVALID_DEVICE);
  UR_ASSERT(size > 0 && size <= device_max_mem_alloc_size,
            UR_RESULT_ERROR_INVALID_USM_SIZE);

  ur_result_t result = UR_RESULT_SUCCESS;
  try {
    ScopedContext active(hContext);
    result = UR_CHECK_ERROR(hipMalloc(ppMem, size));
  } catch (ur_result_t error) {
    result = error;
  }
  UR_ASSERT(!pUSMDesc || (pUSMDesc->align == 0 ||
                          ((pUSMDesc->align & (pUSMDesc->align - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  assert(result == UR_RESULT_SUCCESS &&
         (!pUSMDesc || pUSMDesc->align == 0 ||
          reinterpret_cast<std::uintptr_t>(*ppMem) % pUSMDesc->align == 0));

  return result;
}

/// USM: Implements USM Shared allocations using HIP Managed Memory
///
UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t pool,
                 size_t size, void **ppMem) {
  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  size_t device_max_mem_alloc_size = 0;
  UR_ASSERT(urDeviceGetInfo(hDevice, UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE,
                            sizeof(size_t),
                            static_cast<void *>(&device_max_mem_alloc_size),
                            nullptr) == UR_RESULT_SUCCESS,
            UR_RESULT_ERROR_INVALID_DEVICE);
  UR_ASSERT(size > 0 && size <= device_max_mem_alloc_size,
            UR_RESULT_ERROR_INVALID_USM_SIZE);

  ur_result_t result = UR_RESULT_SUCCESS;
  try {
    ScopedContext active(hContext);
    result = UR_CHECK_ERROR(hipMallocManaged(ppMem, size, hipMemAttachGlobal));
  } catch (ur_result_t error) {
    result = error;
  }
  UR_ASSERT(!pUSMDesc || (pUSMDesc->align == 0 ||
                          ((pUSMDesc->align & (pUSMDesc->align - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  assert(result == UR_RESULT_SUCCESS &&
         (!pUSMDesc || pUSMDesc->align == 0 ||
          reinterpret_cast<std::uintptr_t>(*ppMem) % pUSMDesc->align == 0));

  return result;
}

/// USM: Frees the given USM pointer associated with the context.
///
UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  ur_result_t result = UR_RESULT_SUCCESS;
  try {
    ScopedContext active(hContext);
    unsigned int type;
    hipPointerAttribute_t hipPointerAttributeType;
    result =
        UR_CHECK_ERROR(hipPointerGetAttributes(&hipPointerAttributeType, pMem));
    type = hipPointerAttributeType.memoryType;
    UR_ASSERT(type == hipMemoryTypeDevice || type == hipMemoryTypeHost,
              UR_RESULT_ERROR_INVALID_MEM_OBJECT);
    if (type == hipMemoryTypeDevice) {
      result = UR_CHECK_ERROR(hipFree(pMem));
    }
    if (type == hipMemoryTypeHost) {
      result = UR_CHECK_ERROR(hipFreeHost(pMem));
    }
  } catch (ur_result_t error) {
    result = error;
  }
  return result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propValueSize,
                     void *pPropValue, size_t *pPropValueSizeRet) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(pMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  ur_result_t result = UR_RESULT_SUCCESS;
  hipPointerAttribute_t hipPointerAttributeType;

  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);

  try {
    ScopedContext active(hContext);
    switch (propName) {
    case UR_USM_ALLOC_INFO_TYPE: {
      unsigned int value;
      // do not throw if hipPointerGetAttribute returns hipErrorInvalidValue
      hipError_t ret = hipPointerGetAttributes(&hipPointerAttributeType, pMem);
      if (ret == hipErrorInvalidValue) {
        // pointer not known to the HIP subsystem
        return ReturnValue(UR_USM_TYPE_UNKNOWN);
      }
      result = check_error_ur(ret, __func__, __LINE__ - 5, __FILE__);
      value = hipPointerAttributeType.isManaged;
      if (value) {
        // pointer to managed memory
        return ReturnValue(UR_USM_TYPE_SHARED);
      }
      result = UR_CHECK_ERROR(
          hipPointerGetAttributes(&hipPointerAttributeType, pMem));
      value = hipPointerAttributeType.memoryType;
      UR_ASSERT(value == hipMemoryTypeDevice || value == hipMemoryTypeHost,
                UR_RESULT_ERROR_INVALID_MEM_OBJECT);
      if (value == hipMemoryTypeDevice) {
        // pointer to device memory
        return ReturnValue(UR_USM_TYPE_DEVICE);
      }
      if (value == hipMemoryTypeHost) {
        // pointer to host memory
        return ReturnValue(UR_USM_TYPE_HOST);
      }
      // should never get here
#ifdef _MSC_VER
      __assume(0);
#else
      __builtin_unreachable();
#endif
      return ReturnValue(UR_USM_TYPE_UNKNOWN);
    }
    case UR_USM_ALLOC_INFO_BASE_PTR:
    case UR_USM_ALLOC_INFO_SIZE:
      return UR_RESULT_ERROR_INVALID_VALUE;
    case UR_USM_ALLOC_INFO_DEVICE: {
      // get device index associated with this pointer
      result = UR_CHECK_ERROR(
          hipPointerGetAttributes(&hipPointerAttributeType, pMem));

      int device_idx = hipPointerAttributeType.device;

      // currently each device is in its own platform, so find the platform at
      // the same index
      std::vector<ur_platform_handle_t> platforms;
      platforms.resize(device_idx + 1);
      result = urPlatformGet(device_idx + 1, platforms.data(), nullptr);

      // get the device from the platform
      ur_device_handle_t device = platforms[device_idx]->devices_[0].get();
      return ReturnValue(device);
    }
    default:
      return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
  } catch (ur_result_t error) {
    result = error;
  }
  return result;
}
