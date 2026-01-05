//===--------- usm.cpp - Level Zero Adapter -------------------------------===//
//
// Copyright (C) 2023-2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <climits>
#include <optional>
#include <string.h>
#include <tuple>

#include "context.hpp"
#include "event.hpp"
#include "logger/ur_logger.hpp"
#include "queue.hpp"
#include "ur_api.h"
#include "ur_interface_loader.hpp"
#include "ur_level_zero.hpp"
#include "ur_util.hpp"
#include "usm.hpp"

#include <umf/base.h>
#include <umf/experimental/ctl.h>
#include <umf/memory_pool.h>
#include <umf_helpers.hpp>

namespace umf {
ur_result_t getProviderNativeError(const char *providerName,
                                   int32_t nativeError) {
  if (strcmp(providerName, "Level Zero") == 0) {
    // L0 provider stores native errors of ur_result_t type
    return static_cast<ur_result_t>(nativeError);
  }

  return UR_RESULT_ERROR_UNKNOWN;
}
} // namespace umf

usm::DisjointPoolAllConfigs DisjointPoolConfigInstance =
    InitializeDisjointPoolConfig();

usm::DisjointPoolAllConfigs InitializeDisjointPoolConfig() {
  // Prefer the UR-specific env var, fall back to the PI-specific one.
  const char *PoolTraceVal = std::getenv("UR_L0_USM_ALLOCATOR_TRACE");
  if (!PoolTraceVal) {
    PoolTraceVal = std::getenv("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR_TRACE");
  }

  // Parse integer value, defaulting to 0 on missing/invalid input.
  int PoolTrace = 0;
  if (PoolTraceVal) {
    PoolTrace = std::atoi(PoolTraceVal);
  }

  const char *PoolConfigVal = std::getenv("UR_L0_USM_ALLOCATOR");
  if (!PoolConfigVal) {
    PoolConfigVal = std::getenv("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR");
  }
  if (!PoolConfigVal) {
    return usm::DisjointPoolAllConfigs(PoolTrace);
  }

  return usm::parseDisjointPoolConfig(PoolConfigVal, PoolTrace);
}

enum class USMAllocationForceResidencyType {
  // Do not force memory residency at allocation time.
  None = 0,
  // Force memory resident on the device of allocation at allocation time.
  // For host allocation force residency on all devices in a context.
  Device = 1,
  // Force memory resident on all devices in the context with P2P
  // access to the device of allocation.
  // For host allocation force residency on all devices in a context.
  P2PDevices = 2
};

// Input value is of the form 0xHSD, where:
//   4-bits of D control device allocations
//   4-bits of S control shared allocations
//   4-bits of H control host allocations
// Each 4-bit value is holding a USMAllocationForceResidencyType enum value.
// The default is 0x2, i.e. force full residency for device allocations only.
//
static uint32_t USMAllocationForceResidency = [] {
  const char *UrRet = std::getenv("UR_L0_USM_RESIDENT");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_USM_RESIDENT");
  const char *Str = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  try {
    if (Str) {
      // Auto-detect radix to allow more convinient hex base
      return std::stoi(Str, nullptr, 0);
    }
  } catch (...) {
  }
  return 0x2;
}();

// Convert from an integer value to USMAllocationForceResidencyType enum value
static USMAllocationForceResidencyType
USMAllocationForceResidencyConvert(uint32_t Val) {
  switch (Val) {
  case 1:
    return USMAllocationForceResidencyType::Device;
  case 2:
    return USMAllocationForceResidencyType::P2PDevices;
  default:
    return USMAllocationForceResidencyType::None;
  };
}

static USMAllocationForceResidencyType USMHostAllocationForceResidency = [] {
  return USMAllocationForceResidencyConvert(
      (USMAllocationForceResidency & 0xf00) >> 8);
}();
static USMAllocationForceResidencyType USMSharedAllocationForceResidency = [] {
  return USMAllocationForceResidencyConvert(
      (USMAllocationForceResidency & 0x0f0) >> 4);
}();
static USMAllocationForceResidencyType USMDeviceAllocationForceResidency = [] {
  return USMAllocationForceResidencyConvert(
      (USMAllocationForceResidency & 0x00f));
}();

// Make USM allocation resident as requested
static ur_result_t USMAllocationMakeResident(
    USMAllocationForceResidencyType ForceResidency, ur_context_handle_t Context,
    ur_device_handle_t Device, // nullptr for host allocation
    void *Ptr, size_t Size) {

  if (ForceResidency == USMAllocationForceResidencyType::None)
    return UR_RESULT_SUCCESS;

  std::list<ur_device_handle_t> Devices;
  if (!Device) {
    // Host allocation, make it resident on all devices in the context
    Devices.insert(Devices.end(), Context->Devices.begin(),
                   Context->Devices.end());
  } else {
    Devices.push_back(Device);
    if (ForceResidency == USMAllocationForceResidencyType::P2PDevices) {
      // Check if the P2P devices are already cached
      auto it = Context->P2PDeviceCache.find(Device);
      if (it != Context->P2PDeviceCache.end()) {
        // Use cached P2P devices
        Devices.insert(Devices.end(), it->second.begin(), it->second.end());
      } else {
        // Query for P2P devices and update the cache
        std::list<ur_device_handle_t> P2PDevices;
        ze_bool_t P2P;
        for (const auto &D : Context->Devices) {
          if (D == Device)
            continue;
          ZE2UR_CALL(zeDeviceCanAccessPeer,
                     (D->ZeDevice, Device->ZeDevice, &P2P));
          if (P2P)
            P2PDevices.push_back(D);
        }
        // Update the cache
        Context->P2PDeviceCache[Device] = P2PDevices;
        Devices.insert(Devices.end(), P2PDevices.begin(), P2PDevices.end());
      }
    }
  }
  for (const auto &D : Devices) {
    ZE2UR_CALL(zeContextMakeMemoryResident,
               (Context->ZeContext, D->ZeDevice, Ptr, Size));
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t USMDeviceAllocImpl(void **ResultPtr,
                                      ur_context_handle_t Context,
                                      ur_device_handle_t Device,
                                      ur_usm_device_mem_flags_t /*Flags*/,
                                      size_t Size, uint32_t Alignment) {
  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_device_mem_alloc_desc_t> ZeDesc;
  ZeDesc.flags = 0;
  ZeDesc.ordinal = 0;

  ZeStruct<ze_relaxed_allocation_limits_exp_desc_t> RelaxedDesc;
  if (Device->useRelaxedAllocationLimits() &&
      (Size > Device->ZeDeviceProperties->maxMemAllocSize)) {
    // Tell Level-Zero to accept Size > maxMemAllocSize if
    // large allocations are used.
    RelaxedDesc.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
    ZeDesc.pNext = &RelaxedDesc;
  }

  ze_result_t ZeResult = ZE_CALL_NOCHECK(
      zeMemAllocDevice, (Context->ZeContext, &ZeDesc, Size, Alignment,
                         Device->ZeDevice, ResultPtr));
  if (ZeResult != ZE_RESULT_SUCCESS) {
    if (ZeResult == ZE_RESULT_ERROR_UNSUPPORTED_SIZE) {
      return UR_RESULT_ERROR_INVALID_USM_SIZE;
    }
    return ze2urResult(ZeResult);
  }

  UR_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            UR_RESULT_ERROR_INVALID_VALUE);

  // TODO: Return any non-success result from USMAllocationMakeResident once
  // oneapi-src/level-zero-spec#240 is resolved.
  auto Result = USMAllocationMakeResident(USMDeviceAllocationForceResidency,
                                          Context, Device, *ResultPtr, Size);
  if (Result == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ||
      Result == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY) {
    return Result;
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t
USMSharedAllocImpl(void **ResultPtr, ur_context_handle_t Context,
                   ur_device_handle_t Device, ur_usm_host_mem_flags_t,
                   ur_usm_device_mem_flags_t, size_t Size, uint32_t Alignment) {

  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_host_mem_alloc_desc_t> ZeHostDesc;
  ZeHostDesc.flags = 0;
  ZeStruct<ze_device_mem_alloc_desc_t> ZeDevDesc;
  ZeDevDesc.flags = 0;
  ZeDevDesc.ordinal = 0;

  ZeStruct<ze_relaxed_allocation_limits_exp_desc_t> RelaxedDesc;
  if (Size > Device->ZeDeviceProperties->maxMemAllocSize) {
    // Tell Level-Zero to accept Size > maxMemAllocSize
    RelaxedDesc.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
    ZeDevDesc.pNext = &RelaxedDesc;
  }

  ze_result_t ZeResult = ZE_CALL_NOCHECK(
      zeMemAllocShared, (Context->ZeContext, &ZeDevDesc, &ZeHostDesc, Size,
                         Alignment, Device->ZeDevice, ResultPtr));
  if (ZeResult != ZE_RESULT_SUCCESS) {
    if (ZeResult == ZE_RESULT_ERROR_UNSUPPORTED_SIZE) {
      return UR_RESULT_ERROR_INVALID_USM_SIZE;
    }
    return ze2urResult(ZeResult);
  }

  UR_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            UR_RESULT_ERROR_INVALID_VALUE);

  // TODO: Return any non-success result from USMAllocationMakeResident once
  // oneapi-src/level-zero-spec#240 is resolved.
  auto Result = USMAllocationMakeResident(USMSharedAllocationForceResidency,
                                          Context, Device, *ResultPtr, Size);
  if (Result == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ||
      Result == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY) {
    return Result;
  }

  // TODO: Handle PI_MEM_ALLOC_DEVICE_READ_ONLY.
  return UR_RESULT_SUCCESS;
}

static ur_result_t USMHostAllocImpl(void **ResultPtr,
                                    ur_context_handle_t Context,
                                    ur_usm_host_mem_flags_t /*flags*/,
                                    size_t Size, uint32_t Alignment) {
  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_host_mem_alloc_desc_t> ZeHostDesc;
  ZeHostDesc.flags = 0;
  ze_result_t ZeResult =
      ZE_CALL_NOCHECK(zeMemAllocHost, (Context->ZeContext, &ZeHostDesc, Size,
                                       Alignment, ResultPtr));
  if (ZeResult != ZE_RESULT_SUCCESS) {
    if (ZeResult == ZE_RESULT_ERROR_UNSUPPORTED_SIZE) {
      return UR_RESULT_ERROR_INVALID_USM_SIZE;
    }
    return ze2urResult(ZeResult);
  }

  UR_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            UR_RESULT_ERROR_INVALID_VALUE);

  // TODO: Return any non-success result from USMAllocationMakeResident once
  // oneapi-src/level-zero-spec#240 is resolved.
  auto Result = USMAllocationMakeResident(USMHostAllocationForceResidency,
                                          Context, nullptr, *ResultPtr, Size);
  if (Result == UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY ||
      Result == UR_RESULT_ERROR_OUT_OF_HOST_MEMORY) {
    return Result;
  }
  return UR_RESULT_SUCCESS;
}

namespace ur::level_zero {

ur_result_t urUSMHostAlloc(
    /// [in] handle of the context object
    ur_context_handle_t Context,
    /// [in][optional] USM memory allocation descriptor
    const ur_usm_desc_t *USMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t Pool,
    /// [in] size in bytes of the USM memory object to be allocated
    size_t Size,
    /// [out] pointer to USM host memory object
    void **RetMem) {
  ur_usm_pool_handle_t UrPool = nullptr;
  if (Pool) {
    UrPool = Pool;
  } else {
    UrPool = &Context->DefaultPool;
  }

  return UrPool->allocate(Context, nullptr, USMDesc, UR_USM_TYPE_HOST, Size,
                          RetMem);
}

ur_result_t urUSMDeviceAlloc(
    /// [in] handle of the context object
    ur_context_handle_t Context,
    /// [in] handle of the device object
    ur_device_handle_t Device,
    /// [in][optional] USM memory allocation descriptor
    const ur_usm_desc_t *USMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t Pool,
    /// [in] size in bytes of the USM memory object to be allocated
    size_t Size,
    /// [out] pointer to USM device memory object
    void **RetMem) {

  ur_usm_pool_handle_t UrPool = nullptr;
  if (Pool) {
    UrPool = Pool;
  } else {
    UrPool = &Context->DefaultPool;
  }

  return UrPool->allocate(Context, Device, USMDesc, UR_USM_TYPE_DEVICE, Size,
                          RetMem);
}

ur_result_t urUSMSharedAlloc(
    /// [in] handle of the context object
    ur_context_handle_t Context,
    /// [in] handle of the device object
    ur_device_handle_t Device,
    /// [in][optional] USM memory allocation descriptor
    const ur_usm_desc_t *USMDesc,
    /// [in][optional] Pointer to a pool created using urUSMPoolCreate
    ur_usm_pool_handle_t Pool,
    /// [in] size in bytes of the USM memory object to be allocated
    size_t Size,
    /// [out] pointer to USM shared memory object
    void **RetMem) {
  ur_usm_pool_handle_t UrPool = nullptr;
  if (Pool) {
    UrPool = Pool;
  } else {
    UrPool = &Context->DefaultPool;
  }

  return UrPool->allocate(Context, Device, USMDesc, UR_USM_TYPE_SHARED, Size,
                          RetMem);
}

ur_result_t
/// [in] handle of the context object
urUSMFree(ur_context_handle_t Context,
          /// [in] pointer to USM memory object
          void *Mem) {
  ur_platform_handle_t Plt = Context->getPlatform();

  std::scoped_lock<ur_shared_mutex> Lock(
      IndirectAccessTrackingEnabled ? Plt->ContextsMutex : Context->Mutex);

  return USMFreeHelper(Context, Mem);
}

ur_result_t urUSMGetMemAllocInfo(
    /// [in] handle of the context object
    ur_context_handle_t Context,
    /// [in] pointer to USM memory object
    const void *Ptr,
    /// [in] the name of the USM allocation property to query
    ur_usm_alloc_info_t PropName,
    /// [in] size in bytes of the USM allocation property value
    size_t PropValueSize,
    /// [out][optional] value of the USM allocation property
    void *PropValue,
    /// [out][optional] bytes returned in USM allocation property
    size_t *PropValueSizeRet) {
  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  ZE2UR_CALL(zeMemGetAllocProperties,
             (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
              &ZeDeviceHandle));

  UrReturnHelper ReturnValue(PropValueSize, PropValue, PropValueSizeRet);
  switch (PropName) {
  case UR_USM_ALLOC_INFO_TYPE: {
    ur_usm_type_t MemAllocaType;
    switch (ZeMemoryAllocationProperties.type) {
    case ZE_MEMORY_TYPE_UNKNOWN:
      MemAllocaType = UR_USM_TYPE_UNKNOWN;
      break;
    case ZE_MEMORY_TYPE_HOST:
      MemAllocaType = UR_USM_TYPE_HOST;
      break;
    case ZE_MEMORY_TYPE_DEVICE:
      MemAllocaType = UR_USM_TYPE_DEVICE;
      break;
    case ZE_MEMORY_TYPE_SHARED:
      MemAllocaType = UR_USM_TYPE_SHARED;
      break;
    default:
      UR_LOG(ERR, "urUSMGetMemAllocInfo: unexpected usm memory type");
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
    return ReturnValue(MemAllocaType);
  }
  case UR_USM_ALLOC_INFO_DEVICE:
    if (ZeDeviceHandle) {
      auto Platform = Context->getPlatform();
      auto Device = Platform->getDeviceFromNativeHandle(ZeDeviceHandle);
      return Device ? ReturnValue(Device) : UR_RESULT_ERROR_INVALID_VALUE;
    } else {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  case UR_USM_ALLOC_INFO_BASE_PTR: {
    void *Base;
    ZE2UR_CALL(zeMemGetAddressRange, (Context->ZeContext, Ptr, &Base, nullptr));
    return ReturnValue(Base);
  }
  case UR_USM_ALLOC_INFO_SIZE: {
    size_t Size;
    ZE2UR_CALL(zeMemGetAddressRange, (Context->ZeContext, Ptr, nullptr, &Size));
    return ReturnValue(Size);
  }
  case UR_USM_ALLOC_INFO_POOL: {
    umf_memory_pool_handle_t UmfPool = nullptr;
    auto UmfRet = umfPoolByPtr(Ptr, &UmfPool);
    if (UmfRet != UMF_RESULT_SUCCESS || !UmfPool) {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    std::shared_lock<ur_shared_mutex> ContextLock(Context->Mutex);

    for (auto &Pool : Context->UsmPoolHandles) {
      if (Pool->hasPool(UmfPool)) {
        return ReturnValue(Pool);
      }
    }

    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  default:
    UR_LOG(ERR, "urUSMGetMemAllocInfo: unsupported ParamName");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urUSMPoolCreate(
    /// [in] handle of the context object
    ur_context_handle_t Context,
    /// [in] pointer to USM pool descriptor. Can be chained with
    /// ::ur_usm_pool_limits_desc_t
    ur_usm_pool_desc_t *PoolDesc,
    /// [out] pointer to USM memory pool
    ur_usm_pool_handle_t *Pool) {

  try {
    *Pool = reinterpret_cast<ur_usm_pool_handle_t>(
        new ur_usm_pool_handle_t_(Context, PoolDesc));

    std::scoped_lock<ur_shared_mutex> ContextLock(Context->Mutex);
    Context->UsmPoolHandles.insert(Context->UsmPoolHandles.cend(), *Pool);

  } catch (const UsmAllocationException &Ex) {
    return Ex.getError();
  } catch (umf_result_t e) {
    return umf::umf2urResult(e);
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t
/// [in] pointer to USM memory pool
urUSMPoolRetain(ur_usm_pool_handle_t Pool) {
  Pool->RefCount.retain();
  return UR_RESULT_SUCCESS;
}

ur_result_t
/// [in] pointer to USM memory pool
urUSMPoolRelease(ur_usm_pool_handle_t Pool) {
  if (Pool->RefCount.release()) {
    std::scoped_lock<ur_shared_mutex> ContextLock(Pool->Context->Mutex);
    Pool->Context->UsmPoolHandles.remove(Pool);
    delete Pool;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urUSMPoolGetInfo(
    /// [in] handle of the USM memory pool
    ur_usm_pool_handle_t Pool,
    /// [in] name of the pool property to query
    ur_usm_pool_info_t PropName,
    /// [in] size in bytes of the pool property value provided
    size_t PropSize,
    /// [out][typename(propName, propSize)] value of the pool property
    void *PropValue,
    /// [out] size in bytes returned in pool property value
    size_t *PropSizeRet) {
  UrReturnHelper ReturnValue(PropSize, PropValue, PropSizeRet);

  switch (PropName) {
  case UR_USM_POOL_INFO_REFERENCE_COUNT: {
    return ReturnValue(Pool->RefCount.getCount());
  }
  case UR_USM_POOL_INFO_CONTEXT: {
    return ReturnValue(Pool->Context);
  }
  default: {
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  }
}

ur_result_t urUSMImportExp(ur_context_handle_t Context, void *HostPtr,
                           size_t Size) {
  UR_ASSERT(Context, UR_RESULT_ERROR_INVALID_CONTEXT);

  // Promote the host ptr to USM host memory.
  if (ZeUSMImport.Supported && HostPtr != nullptr) {
    // Query memory type of the host pointer
    ze_device_handle_t ZeDeviceHandle;
    ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;
    ZE2UR_CALL(zeMemGetAllocProperties,
               (Context->ZeContext, HostPtr, &ZeMemoryAllocationProperties,
                &ZeDeviceHandle));

    // If not shared of any type, we can import the ptr
    if (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_UNKNOWN) {
      // Promote the host ptr to USM host memory
      ze_driver_handle_t driverHandle =
          Context->getPlatform()->ZeDriverHandleExpTranslated;
      ZeUSMImport.doZeUSMImport(driverHandle, HostPtr, Size);
    }
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urUSMReleaseExp(ur_context_handle_t Context, void *HostPtr) {
  UR_ASSERT(Context, UR_RESULT_ERROR_INVALID_CONTEXT);

  // Release the imported memory.
  if (ZeUSMImport.Supported && HostPtr != nullptr)
    ZeUSMImport.doZeUSMRelease(
        Context->getPlatform()->ZeDriverHandleExpTranslated, HostPtr);
  return UR_RESULT_SUCCESS;
}

ur_result_t UR_APICALL urUSMPoolCreateExp(
    ur_context_handle_t Context,  ///< [in] handle of the context object
    ur_device_handle_t Device,    ///< [in] handle of the device object
    ur_usm_pool_desc_t *PoolDesc, ///< [in] pointer to USM pool descriptor.
                                  ///< Can be chained with
                                  ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *Pool    ///< [out] pointer to USM memory pool
) {
  try {
    *Pool = reinterpret_cast<ur_usm_pool_handle_t>(
        new ur_usm_pool_handle_t_(Context, Device, PoolDesc));

    std::scoped_lock<ur_shared_mutex> ContextLock(Context->Mutex);
    Context->UsmPoolHandles.insert(Context->UsmPoolHandles.cend(), *Pool);

  } catch (const UsmAllocationException &Ex) {
    return Ex.getError();
  } catch (umf_result_t e) {
    return umf::umf2urResult(e);
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t UR_APICALL urUSMPoolDestroyExp(ur_context_handle_t /*Context*/,
                                           ur_device_handle_t /*Device*/,
                                           ur_usm_pool_handle_t Pool) {
  std::scoped_lock<ur_shared_mutex> ContextLock(Pool->Context->Mutex);
  Pool->Context->UsmPoolHandles.remove(Pool);
  delete Pool;

  return UR_RESULT_SUCCESS;
}

ur_result_t UR_APICALL urUSMPoolSetInfoExp(ur_usm_pool_handle_t /*Pool*/,
                                           ur_usm_pool_info_t PropName,
                                           void * /*PropValue*/,
                                           size_t PropSize) {
  if (PropSize < sizeof(size_t)) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }

  switch (PropName) {
  // TODO: Support for pool release threshold and maximum size hints.
  case UR_USM_POOL_INFO_RELEASE_THRESHOLD_EXP:
  case UR_USM_POOL_INFO_MAXIMUM_SIZE_EXP:
  // TODO: Allow user to overwrite pool peak statistics.
  case UR_USM_POOL_INFO_RESERVED_HIGH_EXP:
  case UR_USM_POOL_INFO_USED_HIGH_EXP:
    break;
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t UR_APICALL urUSMPoolGetDefaultDevicePoolExp(
    ur_context_handle_t Context, ur_device_handle_t /*Device*/,
    ur_usm_pool_handle_t *Pool) {

  // Default async pool should contain an internal pool for all detected
  // devices.
  *Pool = &Context->AsyncPool;

  return UR_RESULT_SUCCESS;
}

ur_result_t UR_APICALL urUSMPoolGetInfoExp(ur_usm_pool_handle_t hPool,
                                           ur_usm_pool_info_t PropName,
                                           void *PropValue,
                                           size_t *PropSizeRet) {
  size_t value = 0;
  switch (PropName) {
  case UR_USM_POOL_INFO_RELEASE_THRESHOLD_EXP:
    // Current pool implementation ignores threshold.
    value = 0;
    break;
  case UR_USM_POOL_INFO_RESERVED_CURRENT_EXP:
    value = hPool->getTotalReservedSize();
    break;
  case UR_USM_POOL_INFO_USED_CURRENT_EXP:
    value = hPool->getTotalUsedSize();
    break;
  case UR_USM_POOL_INFO_RESERVED_HIGH_EXP:
    value = hPool->getPeakReservedSize();
    break;
  case UR_USM_POOL_INFO_USED_HIGH_EXP:
    value = hPool->getPeakUsedSize();
    break;
  default:
    // Unknown enumerator
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  if (PropValue) {
    *(size_t *)PropValue = value;
  }

  if (PropSizeRet) {
    *(size_t *)PropSizeRet = sizeof(size_t);
  }

  return UR_RESULT_SUCCESS;
}

ur_result_t UR_APICALL urUSMPoolGetDevicePoolExp(ur_context_handle_t,
                                                 ur_device_handle_t,
                                                 ur_usm_pool_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolSetDevicePoolExp(ur_context_handle_t,
                                                 ur_device_handle_t,
                                                 ur_usm_pool_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMPoolTrimToExp(ur_context_handle_t,
                                          ur_device_handle_t,
                                          ur_usm_pool_handle_t, size_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t UR_APICALL urUSMContextMemcpyExp(ur_context_handle_t Context,
                                             void *pDst, const void *pSrc,
                                             size_t Size) {
  // zeCommandListAppendMemoryCopy must not be called from simultaneous
  // threads with the same command list handle, so we need exclusive lock.
  std::scoped_lock<ur_mutex> Lock(Context->ImmediateCommandListMutex);
  ZE2UR_CALL(zeCommandListAppendMemoryCopy, (Context->ZeCommandListInit, pDst,
                                             pSrc, Size, nullptr, 0, nullptr));
  return UR_RESULT_SUCCESS;
}
} // namespace ur::level_zero

static ur_result_t USMFreeImpl(ur_context_handle_t Context, void *Ptr) {
  ur_result_t Res = UR_RESULT_SUCCESS;
  if (checkL0LoaderTeardown()) {
    auto ZeResult = ZE_CALL_NOCHECK(zeMemFree, (Context->ZeContext, Ptr));
    // Handle When the driver is already released
    if (ZeResult == ZE_RESULT_ERROR_UNINITIALIZED) {
      Res = UR_RESULT_SUCCESS;
    } else {
      Res = ze2urResult(ZeResult);
    }
  }
  return Res;
}

static ur_result_t USMQueryPageSize(ur_context_handle_t Context, void *Ptr,
                                    size_t *PageSize) {
  ZeStruct<ze_memory_allocation_properties_t> AllocProperties = {};
  ZE2UR_CALL(zeMemGetAllocProperties,
             (Context->ZeContext, Ptr, &AllocProperties, nullptr));
  *PageSize = AllocProperties.pageSize;

  return UR_RESULT_SUCCESS;
}

umf_result_t L0MemoryProvider::initialize(ur_context_handle_t Ctx,
                                          ur_device_handle_t Dev) {
  Context = Ctx;
  Device = Dev;

  return UMF_RESULT_SUCCESS;
}

enum umf_result_t L0MemoryProvider::alloc(size_t Size, size_t Align,
                                          void **Ptr) {
  auto Res = allocateImpl(Ptr, Size, Align);
  if (Res != UR_RESULT_SUCCESS) {
    getLastStatusRef() = Res;
    return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
  }

  // TODO: Align the size? free doesn't accept alignment param.
  AllocStats.update(AllocationStats::INCREASE, Size);

  return UMF_RESULT_SUCCESS;
}

enum umf_result_t L0MemoryProvider::free(void *Ptr, size_t Size) {
  (void)Size;

  auto Res = USMFreeImpl(Context, Ptr);
  if (Res != UR_RESULT_SUCCESS) {
    getLastStatusRef() = Res;
    return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
  }

  AllocStats.update(AllocationStats::DECREASE, Size);

  return UMF_RESULT_SUCCESS;
}

umf_result_t L0MemoryProvider::GetL0MinPageSize(const void *Mem,
                                                size_t *PageSize) {
  ur_result_t Res = UR_RESULT_SUCCESS;
  void *Ptr = const_cast<void *>(Mem);

  if (!Mem) {
    Res = allocateImpl(&Ptr, 1, 0);
    if (Res != UR_RESULT_SUCCESS) {
      goto err_set_status;
    }
  }

  // Query L0 for the minimal page size.
  Res = USMQueryPageSize(Context, Ptr, PageSize);
  if (Res != UR_RESULT_SUCCESS) {
    goto err_dealloc;
  }

  if (!Mem) {
    Res = USMFreeImpl(Context, Ptr);
    if (Res != UR_RESULT_SUCCESS) {
      goto err_set_status;
    }
  }

  return UMF_RESULT_SUCCESS;

err_dealloc:
  if (!Mem) {
    USMFreeImpl(Context, Ptr);
  }
err_set_status:
  getLastStatusRef() = Res;
  return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
}

umf_result_t L0MemoryProvider::get_min_page_size(const void *Ptr,
                                                 size_t *PageSize) {

  // Query L0 for min page size. Use provided 'Ptr'.
  if (Ptr) {
    return GetL0MinPageSize(Ptr, PageSize);
  }

  // Return cached min page size.
  if (MinPageSizeCached) {
    *PageSize = MinPageSize;
    return UMF_RESULT_SUCCESS;
  }

  // Query L0 for min page size and cache it in 'MinPageSize'.
  auto Ret = GetL0MinPageSize(nullptr, &MinPageSize);
  if (Ret) {
    return Ret;
  }

  *PageSize = MinPageSize;
  MinPageSizeCached = true;

  return UMF_RESULT_SUCCESS;
}

typedef struct ze_ipc_data_t {
  int pid;
  ze_ipc_mem_handle_t zeHandle;
} ze_ipc_data_t;

umf_result_t L0MemoryProvider::ext_get_ipc_handle_size(size_t *Size) {
  UR_ASSERT(Size, UMF_RESULT_ERROR_INVALID_ARGUMENT);
  *Size = sizeof(ze_ipc_data_t);

  return UMF_RESULT_SUCCESS;
}

umf_result_t L0MemoryProvider::ext_get_ipc_handle(const void *Ptr,
                                                  size_t /*Size*/,
                                                  void *IpcData) {

  UR_ASSERT(Ptr && IpcData, UMF_RESULT_ERROR_INVALID_ARGUMENT);
  ze_ipc_data_t *zeIpcData = (ze_ipc_data_t *)IpcData;
  auto Ret = ZE_CALL_NOCHECK(zeMemGetIpcHandle,
                             (Context->ZeContext, Ptr, &zeIpcData->zeHandle));
  if (Ret != ZE_RESULT_SUCCESS) {
    return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
  }

  zeIpcData->pid = ur_getpid();

  return UMF_RESULT_SUCCESS;
}

umf_result_t L0MemoryProvider::ext_put_ipc_handle(void *IpcData) {
  UR_ASSERT(IpcData, UMF_RESULT_ERROR_INVALID_ARGUMENT);

  // zeMemPutIpcHandle was introduced in Level Zero 1.6. Before Level Zero 1.6,
  // IPC handle was released automatically when corresponding memory buffer
  // was freed.
#if (ZE_API_VERSION_CURRENT >= ZE_MAKE_VERSION(1, 6))
  ze_ipc_data_t *zeIpcData = (ze_ipc_data_t *)IpcData;
  auto Ret = ZE_CALL_NOCHECK(zeMemPutIpcHandle,
                             (Context->ZeContext, zeIpcData->zeHandle));
  if (Ret != ZE_RESULT_SUCCESS) {
    return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
  }
#endif

  return UMF_RESULT_SUCCESS;
}

umf_result_t L0MemoryProvider::ext_open_ipc_handle(void *IpcData, void **Ptr) {
  UR_ASSERT(IpcData && Ptr, UMF_RESULT_ERROR_INVALID_ARGUMENT);
  ze_ipc_data_t *zeIpcData = (ze_ipc_data_t *)IpcData;

  int fdLocal = -1;
  if (zeIpcData->pid != ur_getpid()) {
    int fdRemote = -1;
    memcpy(&fdRemote, &zeIpcData->zeHandle, sizeof(fdRemote));
    fdLocal = ur_duplicate_fd(zeIpcData->pid, fdRemote);
    if (fdLocal == -1) {
      UR_LOG(ERR, "duplicating file descriptor from IPC handle failed");
      return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
    }

    memcpy(&zeIpcData->zeHandle, &fdLocal, sizeof(fdLocal));
  }

  auto Ret =
      ZE_CALL_NOCHECK(zeMemOpenIpcHandle, (Context->ZeContext, Device->ZeDevice,
                                           zeIpcData->zeHandle, 0, Ptr));
  if (fdLocal != -1) {
    ur_close_fd(fdLocal);
  }

  if (Ret != ZE_RESULT_SUCCESS) {
    return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
  }

  return UMF_RESULT_SUCCESS;
}

umf_result_t L0MemoryProvider::ext_close_ipc_handle(void *Ptr,
                                                    size_t /*Size*/) {

  UR_ASSERT(Ptr, UMF_RESULT_ERROR_INVALID_ARGUMENT);
  auto Ret = ZE_CALL_NOCHECK(zeMemCloseIpcHandle, (Context->ZeContext, Ptr));
  if (Ret != ZE_RESULT_SUCCESS) {
    return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
  }

  return UMF_RESULT_SUCCESS;
}

umf_result_t L0MemoryProvider::ext_ctl(umf_ctl_query_source_t /*Source*/,
                                       const char *Name, void *Arg, size_t Size,
                                       umf_ctl_query_type_t /*QueryType*/,
                                       va_list /*Args*/) {
  if (!Arg || Size < sizeof(size_t)) {
    return UMF_RESULT_ERROR_INVALID_ARGUMENT;
  }

  if (std::string(Name) == "stats.allocated_memory") {
    *(reinterpret_cast<size_t *>(Arg)) = AllocStats.getCurrent();
    UR_LOG(DEBUG, "L0MemoryProvider::ext_ctl with name: {}, value: {}", Name,
           AllocStats.getCurrent());
  } else if (std::string(Name) == "stats.peak_memory") {
    *(reinterpret_cast<size_t *>(Arg)) = AllocStats.getPeak();
    UR_LOG(DEBUG, "L0MemoryProvider::ext_ctl with name: {}, value: {}", Name,
           AllocStats.getPeak());
  } else {
    UR_LOG(ERR, "L0MemoryProvider::ext_ctl with unknown name: {}", Name);
    return UMF_RESULT_ERROR_INVALID_ARGUMENT;
  }

  return UMF_RESULT_SUCCESS;
}

ur_result_t L0SharedMemoryProvider::allocateImpl(void **ResultPtr, size_t Size,
                                                 uint32_t Alignment) {
  return USMSharedAllocImpl(ResultPtr, Context, Device, /*host flags*/ 0,
                            /*device flags*/ 0, Size, Alignment);
}

ur_result_t L0SharedReadOnlyMemoryProvider::allocateImpl(void **ResultPtr,
                                                         size_t Size,
                                                         uint32_t Alignment) {
  ur_usm_device_desc_t UsmDeviceDesc{};
  UsmDeviceDesc.flags = UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY;
  return USMSharedAllocImpl(ResultPtr, Context, Device, UsmDeviceDesc.flags,
                            /*host flags*/ 0, Size, Alignment);
}

ur_result_t L0DeviceMemoryProvider::allocateImpl(void **ResultPtr, size_t Size,
                                                 uint32_t Alignment) {
  return USMDeviceAllocImpl(ResultPtr, Context, Device, /* flags */ 0, Size,
                            Alignment);
}

ur_result_t L0HostMemoryProvider::allocateImpl(void **ResultPtr, size_t Size,
                                               uint32_t Alignment) {
  return USMHostAllocImpl(ResultPtr, Context, /* flags */ 0, Size, Alignment);
}

static usm::DisjointPoolMemType
DescToDisjointPoolMemType(const usm::pool_descriptor &desc) {
  switch (desc.type) {
  case UR_USM_TYPE_DEVICE:
    return usm::DisjointPoolMemType::Device;
  case UR_USM_TYPE_SHARED:
    if (desc.deviceReadOnly)
      return usm::DisjointPoolMemType::SharedReadOnly;
    else
      return usm::DisjointPoolMemType::Shared;
  case UR_USM_TYPE_HOST:
    return usm::DisjointPoolMemType::Host;
  default:
    throw UR_RESULT_ERROR_INVALID_ARGUMENT;
  }
}

typedef usm::pool_descriptor l0_memory_provider_params_t;

template <typename ProviderParams = std::nullptr_t>
static umf::provider_unique_handle_t
MakeProvider(ProviderParams *Params = nullptr) {
  if constexpr (std::is_same_v<ProviderParams, l0_memory_provider_params_t>) {
    umf_result_t Ret = UMF_RESULT_SUCCESS;
    umf::provider_unique_handle_t &&L0Provider = nullptr;

    switch (Params->type) {
    case UR_USM_TYPE_HOST:
      std::tie(Ret, L0Provider) =
          umf::memoryProviderMakeUnique<L0HostMemoryProvider>(Params->hContext,
                                                              Params->hDevice);
      break;
    case UR_USM_TYPE_DEVICE:
      std::tie(Ret, L0Provider) =
          umf::memoryProviderMakeUnique<L0DeviceMemoryProvider>(
              Params->hContext, Params->hDevice);
      break;
    case UR_USM_TYPE_SHARED:
      if (Params->deviceReadOnly) {
        std::tie(Ret, L0Provider) =
            umf::memoryProviderMakeUnique<L0SharedReadOnlyMemoryProvider>(
                Params->hContext, Params->hDevice);
      } else {
        std::tie(Ret, L0Provider) =
            umf::memoryProviderMakeUnique<L0SharedMemoryProvider>(
                Params->hContext, Params->hDevice);
      }
      break;
    default:
      UR_LOG(ERR, "urUSMPoolCreate: invalid USM type found");
      Ret = UMF_RESULT_ERROR_INVALID_ARGUMENT;
    }

    if (Ret != UMF_RESULT_SUCCESS) {
      UR_LOG(ERR, "urUSMPoolCreate: failed to create UMF provider");
      throw UsmAllocationException(umf::umf2urResult(Ret));
    }

    return std::move(L0Provider);
  }

  return nullptr;
}

UsmPool::UsmPool(ur_usm_pool_handle_t UrPool, umf::pool_unique_handle_t UmfPool)
    : UrPool(UrPool), UmfPool(std::move(UmfPool)),
      AsyncPool(
          [](ur_event_handle_t Event) { return urEventReleaseInternal(Event); },
          [Context = UrPool->Context](void *Ptr) {
            return USMFreeHelper(Context, Ptr);
          }) {}

ur_usm_pool_handle_t_::ur_usm_pool_handle_t_(ur_context_handle_t Context,
                                             ur_usm_pool_desc_t *PoolDesc,
                                             bool IsProxy)
    : Context(Context) {
  // TODO: handle zero-init flag 'UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK'
  auto DisjointPoolConfigs = DisjointPoolConfigInstance;
  if (auto Limits = find_stype_node<ur_usm_pool_limits_desc_t>(PoolDesc)) {
    for (auto &Config : DisjointPoolConfigs.Configs) {
      Config.MaxPoolableSize = Limits->maxPoolableSize;
      Config.SlabMinSize = Limits->minDriverAllocSize;
    }
  }

  auto DevicesAndSubDevices =
      CollectDevicesForUsmPoolCreation(Context->Devices);
  auto Descriptors = usm::pool_descriptor::createFromDevices(
      this, Context, DevicesAndSubDevices);
  for (auto &Desc : Descriptors) {
    umf::pool_unique_handle_t Pool = nullptr;
    if (IsProxy) {
      Pool = usm::makeProxyPool(MakeProvider(&Desc));
    } else {
      auto &PoolConfig =
          DisjointPoolConfigs.Configs[DescToDisjointPoolMemType(Desc)];
      Pool = usm::makeDisjointPool(MakeProvider(&Desc), PoolConfig);
    }

    std::unique_ptr<UsmPool> usmPool =
        std::make_unique<UsmPool>(this, std::move(Pool));
    auto Ret = umf::umf2urResult(
        umfPoolSetTag(usmPool->UmfPool.get(), usmPool.get(), nullptr));
    if (Ret) {
      UR_LOG(ERR, "urUSMPoolCreate: failed to store USM pool tag");
      throw UsmAllocationException(Ret);
    }

    Ret = PoolManager.addPool(Desc, std::move(usmPool));
    if (Ret) {
      UR_LOG(ERR, "urUSMPoolCreate: failed to store UMF pool");
      throw UsmAllocationException(Ret);
    }
  }
}

ur_usm_pool_handle_t_::ur_usm_pool_handle_t_(ur_context_handle_t Context,
                                             ur_device_handle_t Device,
                                             ur_usm_pool_desc_t *PoolDesc)
    : Context(Context) {
  // TODO: handle zero-init flag 'UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK'
  auto DisjointPoolConfigs = DisjointPoolConfigInstance;
  if (auto Limits = find_stype_node<ur_usm_pool_limits_desc_t>(PoolDesc)) {
    for (auto &Config : DisjointPoolConfigs.Configs) {
      Config.MaxPoolableSize = Limits->maxPoolableSize;
      Config.SlabMinSize = Limits->minDriverAllocSize;
    }
  }

  // Create pool descriptor for single device provided
  std::vector<usm::pool_descriptor> Descriptors;
  {
    auto &Desc = Descriptors.emplace_back();
    Desc.poolHandle = this;
    Desc.hContext = Context;
    Desc.hDevice = Device;
    Desc.type = UR_USM_TYPE_DEVICE;
  }
  {
    auto &Desc = Descriptors.emplace_back();
    Desc.poolHandle = this;
    Desc.hContext = Context;
    Desc.hDevice = Device;
    Desc.type = UR_USM_TYPE_SHARED;
    Desc.deviceReadOnly = false;
  }
  {
    auto &Desc = Descriptors.emplace_back();
    Desc.poolHandle = this;
    Desc.hContext = Context;
    Desc.hDevice = Device;
    Desc.type = UR_USM_TYPE_SHARED;
    Desc.deviceReadOnly = true;
  }

  for (auto &Desc : Descriptors) {
    auto &PoolConfig =
        DisjointPoolConfigs.Configs[DescToDisjointPoolMemType(Desc)];

    std::unique_ptr<UsmPool> usmPool = std::make_unique<UsmPool>(
        this, usm::makeDisjointPool(MakeProvider(&Desc), PoolConfig));
    auto Ret = umf::umf2urResult(
        umfPoolSetTag(usmPool->UmfPool.get(), usmPool.get(), nullptr));
    if (Ret) {
      UR_LOG(ERR, "urUSMPoolCreate: failed to store USM pool tag");
      throw UsmAllocationException(Ret);
    }
    Ret = PoolManager.addPool(Desc, std::move(usmPool));
    if (Ret) {
      UR_LOG(ERR, "urUSMPoolCreate: failed to store UMF pool");
      throw UsmAllocationException(Ret);
    }
  }
}

UsmPool *ur_usm_pool_handle_t_::getPool(const usm::pool_descriptor &Desc) {
  auto PoolOpt = PoolManager.getPool(Desc);
  return PoolOpt.has_value() ? PoolOpt.value() : nullptr;
}

std::optional<std::pair<void *, ur_event_handle_t>>
ur_usm_pool_handle_t_::allocateEnqueued(ur_queue_handle_t Queue,
                                        ur_device_handle_t Device,
                                        const ur_usm_desc_t *USMDesc,
                                        ur_usm_type_t Type, size_t Size) {
  uint32_t Alignment = USMDesc ? USMDesc->align : 0;
  if (Alignment > 0) {
    if (Alignment > 65536 || (Alignment & (Alignment - 1)) != 0)
      return std::nullopt;
  }

  bool DeviceReadOnly = false;
  if (auto UsmDeviceDesc = find_stype_node<ur_usm_device_desc_t>(USMDesc)) {
    DeviceReadOnly =
        (Type == UR_USM_TYPE_SHARED) &&
        (UsmDeviceDesc->flags & UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY);
  }

  auto *Pool = getPool(
      usm::pool_descriptor{this, Queue->Context, Device, Type, DeviceReadOnly});
  if (!Pool) {
    return std::nullopt;
  }

  auto Allocation = Pool->AsyncPool.getBestFit(Size, Alignment, Queue);

  if (!Allocation) {
    return std::nullopt;
  }

  auto *Event = Allocation->Event;
  if (Event->Completed ||
      (Allocation->Queue == Queue && Queue->isInOrderQueue())) {
    urEventReleaseInternal(Event);
    Event = nullptr;
  }

  return std::make_pair(Allocation->Ptr, Event);
}

ur_result_t ur_usm_pool_handle_t_::allocate(ur_context_handle_t Context,
                                            ur_device_handle_t Device,
                                            const ur_usm_desc_t *USMDesc,
                                            ur_usm_type_t Type, size_t Size,
                                            void **RetMem) {
  uint32_t Alignment = USMDesc ? USMDesc->align : 0;
  // L0 supports alignment up to 64KB and silently ignores higher values.
  // We flag alignment > 64KB as an invalid value.
  // L0 spec says that alignment values that are not powers of 2 are invalid.
  // If alignment == 0, then we are allowing the L0 driver to choose the
  // alignment so no need to check.
  if (Alignment > 0) {
    if (Alignment > 65536 || (Alignment & (Alignment - 1)) != 0)
      return UR_RESULT_ERROR_INVALID_VALUE;
  }

  bool DeviceReadOnly = false;
  if (auto UsmDeviceDesc = find_stype_node<ur_usm_device_desc_t>(USMDesc)) {
    DeviceReadOnly =
        (Type == UR_USM_TYPE_SHARED) &&
        (UsmDeviceDesc->flags & UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY);
  }

  ur_platform_handle_t Plt =
      (Device) ? Device->Platform : Context->getPlatform();
  // If indirect access tracking is enabled then lock the mutex which is
  // guarding contexts container in the platform. This prevents new kernels from
  // being submitted in any context while we are in the process of allocating a
  // memory, this is needed to properly capture allocations by kernels with
  // indirect access. This lock also protects access to the context's data
  // structures. If indirect access tracking is not enabled then lock context
  // mutex to protect access to context's data structures.
  std::shared_lock<ur_shared_mutex> ContextLock(Context->Mutex,
                                                std::defer_lock);
  std::unique_lock<ur_shared_mutex> IndirectAccessTrackingLock(
      Plt->ContextsMutex, std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    IndirectAccessTrackingLock.lock();
    // We are going to defer memory release if there are kernels with indirect
    // access, that is why explicitly retain context to be sure that it is
    // released after all memory allocations in this context are released.
    UR_CALL(ur::level_zero::urContextRetain(Context));
  } else {
    ContextLock.lock();
  }

  auto *Pool = getPool(
      usm::pool_descriptor{this, Context, Device, Type, DeviceReadOnly});
  if (!Pool) {
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  }
  auto UmfPool = Pool->UmfPool.get();

  *RetMem = umfPoolAlignedMalloc(UmfPool, Size, Alignment);
  if (*RetMem == nullptr) {
    auto UmfRet = umfPoolGetLastAllocationError(UmfPool);
    UR_LOG(ERR, "enqueueUSMAllocHelper: allocation from the UMF pool {} failed",
           UmfPool);
    return umf::umf2urResult(UmfRet);
  }

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(*RetMem),
                               std::forward_as_tuple(Context));
  }

  size_t UsableSize = 0;
  auto UmfRet = umfPoolMallocUsableSize(UmfPool, *RetMem, &UsableSize);
  if (UmfRet != UMF_RESULT_SUCCESS &&
      UmfRet != UMF_RESULT_ERROR_NOT_SUPPORTED) {
    return umf::umf2urResult(UmfRet);
  }

  AllocStats.update(AllocationStats::UpdateType::INCREASE, UsableSize);

  return UR_RESULT_SUCCESS;
}

ur_result_t ur_usm_pool_handle_t_::free(void *Mem,
                                        umf_memory_pool_handle_t UmfPool) {
  size_t Size = 0;
  auto UmfRet = umfPoolMallocUsableSize(UmfPool, Mem, &Size);
  if (UmfRet != UMF_RESULT_SUCCESS &&
      UmfRet != UMF_RESULT_ERROR_NOT_SUPPORTED) {
    return umf::umf2urResult(UmfRet);
  }

  UmfRet = umfPoolFree(UmfPool, Mem);
  if (UmfRet != UMF_RESULT_SUCCESS) {
    return umf::umf2urResult(UmfRet);
  }

  AllocStats.update(AllocationStats::UpdateType::DECREASE, Size);

  return UR_RESULT_SUCCESS;
}

UsmPool *
ur_usm_pool_handle_t_::getPoolByHandle(const umf_memory_pool_handle_t UmfPool) {
  UsmPool *Pool = nullptr;
  PoolManager.forEachPool([&](UsmPool *p) {
    if (p->UmfPool.get() == UmfPool) {
      Pool = p;
      return false; /* stop iterating */
    }
    return true;
  });
  return Pool;
}

void ur_usm_pool_handle_t_::cleanupPools() {
  PoolManager.forEachPool([&](UsmPool *p) {
    p->AsyncPool.cleanup();
    return true;
  });
}

void ur_usm_pool_handle_t_::cleanupPoolsForQueue(ur_queue_handle_t Queue) {
  PoolManager.forEachPool([&](UsmPool *p) {
    p->AsyncPool.cleanupForQueue(Queue);
    return true;
  });
}

size_t ur_usm_pool_handle_t_::getTotalReservedSize() {
  size_t TotalAllocatedSize = 0;
  umf_result_t UmfRet = UMF_RESULT_SUCCESS;
  PoolManager.forEachPool([&](UsmPool *p) {
    umf_memory_provider_handle_t hProvider = nullptr;
    size_t AllocatedSize = 0;
    UmfRet = umfPoolGetMemoryProvider(p->UmfPool.get(), &hProvider);
    if (UmfRet != UMF_RESULT_SUCCESS) {
      return false;
    }

    UmfRet = umfCtlGet("umf.provider.by_handle.{}.stats.allocated_memory",
                       &AllocatedSize, sizeof(AllocatedSize), hProvider);
    if (UmfRet != UMF_RESULT_SUCCESS) {
      return false;
    }

    TotalAllocatedSize += AllocatedSize;
    return true;
  });

  return UmfRet == UMF_RESULT_SUCCESS ? TotalAllocatedSize : 0;
}

size_t ur_usm_pool_handle_t_::getPeakReservedSize() {
  size_t MaxPeakSize = 0;
  umf_result_t Ret = UMF_RESULT_SUCCESS;
  PoolManager.forEachPool([&](UsmPool *p) {
    umf_memory_provider_handle_t hProvider = nullptr;
    size_t PeakSize = 0;
    Ret = umfPoolGetMemoryProvider(p->UmfPool.get(), &hProvider);
    if (Ret != UMF_RESULT_SUCCESS) {
      return false;
    }

    Ret = umfCtlGet("umf.provider.by_handle.{}.stats.peak_memory", &PeakSize,
                    sizeof(PeakSize), hProvider);
    if (Ret != UMF_RESULT_SUCCESS) {
      return false;
    }

    MaxPeakSize = std::max(MaxPeakSize, PeakSize);
    return true;
  });

  return Ret == UMF_RESULT_SUCCESS ? MaxPeakSize : 0;
}

size_t ur_usm_pool_handle_t_::getTotalUsedSize() {
  return AllocStats.getCurrent();
}

size_t ur_usm_pool_handle_t_::getPeakUsedSize() { return AllocStats.getPeak(); }

bool ur_usm_pool_handle_t_::hasPool(const umf_memory_pool_handle_t Pool) {
  return getPoolByHandle(Pool) != nullptr;
}

// If indirect access tracking is not enabled then this functions just performs
// zeMemFree. If indirect access tracking is enabled then reference counting is
// performed.
ur_result_t ZeMemFreeHelper(ur_context_handle_t Context, void *Ptr) {
  ur_platform_handle_t Plt = Context->getPlatform();
  std::unique_lock<ur_shared_mutex> ContextsLock(Plt->ContextsMutex,
                                                 std::defer_lock);
  if (IndirectAccessTrackingEnabled) {
    ContextsLock.lock();
    auto It = Context->MemAllocs.find(Ptr);
    if (It == std::end(Context->MemAllocs)) {
      die("All memory allocations must be tracked!");
    }
    if (!It->second.RefCount.release()) {
      // Memory can't be deallocated yet.
      return UR_RESULT_SUCCESS;
    }

    // Reference count is zero, it is ok to free memory.
    // We don't need to track this allocation anymore.
    Context->MemAllocs.erase(It);
  }

  ZE2UR_CALL(zeMemFree, (Context->ZeContext, Ptr));

  if (IndirectAccessTrackingEnabled)
    UR_CALL(ContextReleaseHelper(Context));

  return UR_RESULT_SUCCESS;
}

static bool ShouldUseUSMAllocator() {
  // Enable allocator by default if it's not explicitly disabled
  const char *UrRet = std::getenv("UR_L0_DISABLE_USM_ALLOCATOR");
  const char *PiRet = std::getenv("SYCL_PI_LEVEL_ZERO_DISABLE_USM_ALLOCATOR");
  const char *Res = UrRet ? UrRet : (PiRet ? PiRet : nullptr);
  return Res == nullptr;
}

const bool UseUSMAllocator = ShouldUseUSMAllocator();

// Helper function to deallocate USM memory, if indirect access support is
// enabled then a caller must lock the platform-level mutex guarding the
// container with contexts because deallocating the memory can turn RefCount of
// a context to 0 and as a result the context being removed from the list of
// tracked contexts.
// If indirect access tracking is not enabled then caller must lock Context
// mutex.
ur_result_t USMFreeHelper(ur_context_handle_t Context, void *Ptr,
                          bool OwnZeMemHandle) {
  if (!OwnZeMemHandle) {
    // Memory should not be freed
    return UR_RESULT_SUCCESS;
  }

  if (IndirectAccessTrackingEnabled) {
    auto It = Context->MemAllocs.find(Ptr);
    if (It == std::end(Context->MemAllocs)) {
      die("All memory allocations must be tracked!");
    }
    if (!It->second.RefCount.release()) {
      // Memory can't be deallocated yet.
      return UR_RESULT_SUCCESS;
    }

    // Reference count is zero, it is ok to free memory.
    // We don't need to track this allocation anymore.
    Context->MemAllocs.erase(It);
  }

  umf_memory_pool_handle_t UmfPool = nullptr;
  auto UmfRet = umfPoolByPtr(Ptr, &UmfPool);
  if (UmfRet != UMF_RESULT_SUCCESS || !UmfPool) {
    if (IndirectAccessTrackingEnabled)
      UR_CALL(ContextReleaseHelper(Context));
    return UR_RESULT_ERROR_INVALID_MEM_OBJECT;
  }

  UsmPool *UsmPool = nullptr;
  UmfRet = umfPoolGetTag(UmfPool, (void **)&UsmPool);
  if (UmfRet != UMF_RESULT_SUCCESS || !UsmPool) {
    // This should never happen
    UR_LOG(ERR, "enqueueUSMFreeExp: invalid pool tag");
    return UR_RESULT_ERROR_UNKNOWN;
  }

  auto Ret = UsmPool->UrPool->free(Ptr, UmfPool);
  if (IndirectAccessTrackingEnabled) {
    UR_CALL(ContextReleaseHelper(Context));
  }
  return Ret;
}
