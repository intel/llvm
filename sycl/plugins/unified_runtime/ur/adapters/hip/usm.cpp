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
UR_APIEXPORT ur_result_t UR_APICALL urUSMHostAlloc(
    ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
    [[maybe_unused]] ur_usm_pool_handle_t pool, size_t size, void **ppMem) {

  UR_ASSERT(!pUSMDesc || (pUSMDesc->align == 0 ||
                          ((pUSMDesc->align & (pUSMDesc->align - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    ScopedContext Active(hContext);
    Result = UR_CHECK_ERROR(hipHostMalloc(ppMem, size));
  } catch (ur_result_t Error) {
    Result = Error;
  }

  if (Result == UR_RESULT_SUCCESS) {
    assert((!pUSMDesc || pUSMDesc->align == 0 ||
            reinterpret_cast<std::uintptr_t>(*ppMem) % pUSMDesc->align == 0));
  }

  return Result;
}

/// USM: Implements USM device allocations using a normal HIP device pointer
UR_APIEXPORT ur_result_t UR_APICALL urUSMDeviceAlloc(
    ur_context_handle_t hContext, ur_device_handle_t,
    const ur_usm_desc_t *pUSMDesc, [[maybe_unused]] ur_usm_pool_handle_t pool,
    size_t size, void **ppMem) {
  UR_ASSERT(!pUSMDesc || (pUSMDesc->align == 0 ||
                          ((pUSMDesc->align & (pUSMDesc->align - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    ScopedContext Active(hContext);
    Result = UR_CHECK_ERROR(hipMalloc(ppMem, size));
  } catch (ur_result_t Error) {
    Result = Error;
  }

  if (Result == UR_RESULT_SUCCESS) {
    assert((!pUSMDesc || pUSMDesc->align == 0 ||
            reinterpret_cast<std::uintptr_t>(*ppMem) % pUSMDesc->align == 0));
  }

  return Result;
}

/// USM: Implements USM Shared allocations using HIP Managed Memory
UR_APIEXPORT ur_result_t UR_APICALL urUSMSharedAlloc(
    ur_context_handle_t hContext, ur_device_handle_t,
    const ur_usm_desc_t *pUSMDesc, [[maybe_unused]] ur_usm_pool_handle_t pool,
    size_t size, void **ppMem) {
  UR_ASSERT(!pUSMDesc || (pUSMDesc->align == 0 ||
                          ((pUSMDesc->align & (pUSMDesc->align - 1)) == 0)),
            UR_RESULT_ERROR_INVALID_VALUE);

  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    ScopedContext Active(hContext);
    Result = UR_CHECK_ERROR(hipMallocManaged(ppMem, size, hipMemAttachGlobal));
  } catch (ur_result_t Error) {
    Result = Error;
  }

  if (Result == UR_RESULT_SUCCESS) {
    assert((!pUSMDesc || pUSMDesc->align == 0 ||
            reinterpret_cast<std::uintptr_t>(*ppMem) % pUSMDesc->align == 0));
  }

  return Result;
}

/// USM: Frees the given USM pointer associated with the context.
UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    ScopedContext Active(hContext);
    unsigned int Type;
    hipPointerAttribute_t hipPointerAttributeType;
    Result =
        UR_CHECK_ERROR(hipPointerGetAttributes(&hipPointerAttributeType, pMem));
    Type = hipPointerAttributeType.memoryType;
    UR_ASSERT(Type == hipMemoryTypeDevice || Type == hipMemoryTypeHost,
              UR_RESULT_ERROR_INVALID_MEM_OBJECT);
    if (Type == hipMemoryTypeDevice) {
      Result = UR_CHECK_ERROR(hipFree(pMem));
    }
    if (Type == hipMemoryTypeHost) {
      Result = UR_CHECK_ERROR(hipFreeHost(pMem));
    }
  } catch (ur_result_t Error) {
    Result = Error;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propValueSize,
                     void *pPropValue, size_t *pPropValueSizeRet) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  hipPointerAttribute_t hipPointerAttributeType;

  UrReturnHelper ReturnValue(propValueSize, pPropValue, pPropValueSizeRet);

  try {
    ScopedContext Active(hContext);
    switch (propName) {
    case UR_USM_ALLOC_INFO_TYPE: {
      unsigned int Value;
      // do not throw if hipPointerGetAttribute returns hipErrorInvalidValue
      hipError_t Ret = hipPointerGetAttributes(&hipPointerAttributeType, pMem);
      if (Ret == hipErrorInvalidValue) {
        // pointer not known to the HIP subsystem
        return ReturnValue(UR_USM_TYPE_UNKNOWN);
      }
      Result = checkErrorUR(Ret, __func__, __LINE__ - 5, __FILE__);
      Value = hipPointerAttributeType.isManaged;
      if (Value) {
        // pointer to managed memory
        return ReturnValue(UR_USM_TYPE_SHARED);
      }
      Result = UR_CHECK_ERROR(
          hipPointerGetAttributes(&hipPointerAttributeType, pMem));
      Value = hipPointerAttributeType.memoryType;
      UR_ASSERT(Value == hipMemoryTypeDevice || Value == hipMemoryTypeHost,
                UR_RESULT_ERROR_INVALID_MEM_OBJECT);
      if (Value == hipMemoryTypeDevice) {
        // pointer to device memory
        return ReturnValue(UR_USM_TYPE_DEVICE);
      }
      if (Value == hipMemoryTypeHost) {
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
      Result = UR_CHECK_ERROR(
          hipPointerGetAttributes(&hipPointerAttributeType, pMem));

      int DeviceIdx = hipPointerAttributeType.device;

      // currently each device is in its own platform, so find the platform at
      // the same index
      std::vector<ur_platform_handle_t> Platforms;
      Platforms.resize(DeviceIdx + 1);
      Result = urPlatformGet(DeviceIdx + 1, Platforms.data(), nullptr);

      // get the device from the platform
      ur_device_handle_t Device = Platforms[DeviceIdx]->Devices[0].get();
      return ReturnValue(Device);
    }
    default:
      return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
  } catch (ur_result_t Error) {
    Result = Error;
  }
  return Result;
}
