//===--------- usm.cpp - Level Zero Adapter --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include <algorithm>
#include <climits>
#include <string.h>

#include "context.hpp"
#include "event.hpp"
#include "usm.hpp"

#include "ur_level_zero.hpp"

#include <umf_helpers.hpp>

ur_result_t umf2urResult(umf_result_t umfResult) {
  if (umfResult == UMF_RESULT_SUCCESS)
    return UR_RESULT_SUCCESS;

  switch (umfResult) {
  case UMF_RESULT_ERROR_OUT_OF_HOST_MEMORY:
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  case UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC: {
    auto hProvider = umfGetLastFailedMemoryProvider();
    if (hProvider == nullptr)
      return UR_RESULT_ERROR_UNKNOWN;

    ur_result_t Err = UR_RESULT_ERROR_UNKNOWN;
    umfMemoryProviderGetLastNativeError(hProvider, nullptr,
                                        reinterpret_cast<int32_t *>(&Err));
    return Err;
  }
  case UMF_RESULT_ERROR_INVALID_ARGUMENT:
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  case UMF_RESULT_ERROR_INVALID_ALIGNMENT:
    return UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT;
  case UMF_RESULT_ERROR_NOT_SUPPORTED:
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  default:
    return UR_RESULT_ERROR_UNKNOWN;
  };
}

usm::DisjointPoolAllConfigs InitializeDisjointPoolConfig() {
  const char *PoolUrConfigVal = std::getenv("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR");
  const char *PoolPiConfigVal = std::getenv("UR_L0_USM_ALLOCATOR");
  const char *PoolConfigVal =
      PoolUrConfigVal ? PoolUrConfigVal : PoolPiConfigVal;
  if (PoolConfigVal == nullptr) {
    return usm::DisjointPoolAllConfigs();
  }

  const char *PoolUrTraceVal = std::getenv("UR_L0_USM_ALLOCATOR_TRACE");
  const char *PoolPiTraceVal =
      std::getenv("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR_TRACE");
  const char *PoolTraceVal = PoolUrTraceVal
                                 ? PoolUrTraceVal
                                 : (PoolPiTraceVal ? PoolPiTraceVal : nullptr);

  bool PoolTrace = false;
  if (PoolTraceVal != nullptr) {
    PoolTrace = std::atoi(PoolTraceVal);
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
      ze_bool_t P2P;
      for (const auto &D : Context->Devices) {
        if (D == Device)
          continue;
        // TODO: Cache P2P devices for a context
        ZE2UR_CALL(zeDeviceCanAccessPeer,
                   (D->ZeDevice, Device->ZeDevice, &P2P));
        if (P2P)
          Devices.push_back(D);
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
                                      ur_usm_device_mem_flags_t *Flags,
                                      size_t Size, uint32_t Alignment) {
  std::ignore = Flags;
  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_device_mem_alloc_desc_t> ZeDesc;
  ZeDesc.flags = 0;
  ZeDesc.ordinal = 0;

  ZeStruct<ze_relaxed_allocation_limits_exp_desc_t> RelaxedDesc;
  if (Size > Device->ZeDeviceProperties->maxMemAllocSize) {
    // Tell Level-Zero to accept Size > maxMemAllocSize
    RelaxedDesc.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
    ZeDesc.pNext = &RelaxedDesc;
  }

  ZE2UR_CALL(zeMemAllocDevice, (Context->ZeContext, &ZeDesc, Size, Alignment,
                                Device->ZeDevice, ResultPtr));

  UR_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            UR_RESULT_ERROR_INVALID_VALUE);

  USMAllocationMakeResident(USMDeviceAllocationForceResidency, Context, Device,
                            *ResultPtr, Size);
  return UR_RESULT_SUCCESS;
}

static ur_result_t USMSharedAllocImpl(void **ResultPtr,
                                      ur_context_handle_t Context,
                                      ur_device_handle_t Device,
                                      ur_usm_host_mem_flags_t *,
                                      ur_usm_device_mem_flags_t *, size_t Size,
                                      uint32_t Alignment) {

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

  ZE2UR_CALL(zeMemAllocShared, (Context->ZeContext, &ZeDevDesc, &ZeHostDesc,
                                Size, Alignment, Device->ZeDevice, ResultPtr));

  UR_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            UR_RESULT_ERROR_INVALID_VALUE);

  USMAllocationMakeResident(USMSharedAllocationForceResidency, Context, Device,
                            *ResultPtr, Size);

  // TODO: Handle PI_MEM_ALLOC_DEVICE_READ_ONLY.
  return UR_RESULT_SUCCESS;
}

static ur_result_t USMHostAllocImpl(void **ResultPtr,
                                    ur_context_handle_t Context,
                                    ur_usm_host_mem_flags_t *Flags, size_t Size,
                                    uint32_t Alignment) {
  std::ignore = Flags;
  // TODO: translate PI properties to Level Zero flags
  ZeStruct<ze_host_mem_alloc_desc_t> ZeHostDesc;
  ZeHostDesc.flags = 0;
  ZE2UR_CALL(zeMemAllocHost,
             (Context->ZeContext, &ZeHostDesc, Size, Alignment, ResultPtr));

  UR_ASSERT(Alignment == 0 ||
                reinterpret_cast<std::uintptr_t>(*ResultPtr) % Alignment == 0,
            UR_RESULT_ERROR_INVALID_VALUE);

  USMAllocationMakeResident(USMHostAllocationForceResidency, Context, nullptr,
                            *ResultPtr, Size);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMHostAlloc(
    ur_context_handle_t Context, ///< [in] handle of the context object
    const ur_usm_desc_t
        *USMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t Pool, ///< [in][optional] Pointer to a pool created
                               ///< using urUSMPoolCreate
    size_t
        Size, ///< [in] size in bytes of the USM memory object to be allocated
    void **RetMem ///< [out] pointer to USM host memory object
) {

  uint32_t Align = USMDesc ? USMDesc->align : 0;
  // L0 supports alignment up to 64KB and silently ignores higher values.
  // We flag alignment > 64KB as an invalid value.
  if (Align > 65536)
    return UR_RESULT_ERROR_INVALID_VALUE;

  ur_platform_handle_t Plt = Context->getPlatform();
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
    UR_CALL(urContextRetain(Context));
  } else {
    ContextLock.lock();
  }

  if (!UseUSMAllocator ||
      // L0 spec says that allocation fails if Alignment != 2^n, in order to
      // keep the same behavior for the allocator, just call L0 API directly and
      // return the error code.
      ((Align & (Align - 1)) != 0)) {
    ur_usm_host_mem_flags_t Flags{};
    ur_result_t Res = USMHostAllocImpl(RetMem, Context, &Flags, Size, Align);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*RetMem),
                                 std::forward_as_tuple(Context));
    }
    return Res;
  }

  // There is a single allocator for Host USM allocations, so we don't need to
  // find the allocator depending on context as we do for Shared and Device
  // allocations.
  umf_memory_pool_handle_t hPoolInternal = nullptr;
  if (Pool) {
    hPoolInternal = Pool->HostMemPool.get();
  } else {
    hPoolInternal = Context->HostMemPool.get();
  }

  *RetMem = umfPoolAlignedMalloc(hPoolInternal, Size, Align);
  if (*RetMem == nullptr) {
    auto umfRet = umfPoolGetLastAllocationError(hPoolInternal);
    return umf2urResult(umfRet);
  }

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(*RetMem),
                               std::forward_as_tuple(Context));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMDeviceAlloc(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_device_handle_t Device,   ///< [in] handle of the device object
    const ur_usm_desc_t
        *USMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t Pool, ///< [in][optional] Pointer to a pool created
                               ///< using urUSMPoolCreate
    size_t
        Size, ///< [in] size in bytes of the USM memory object to be allocated
    void **RetMem ///< [out] pointer to USM device memory object
) {

  uint32_t Alignment = USMDesc ? USMDesc->align : 0;

  // L0 supports alignment up to 64KB and silently ignores higher values.
  // We flag alignment > 64KB as an invalid value.
  if (Alignment > 65536)
    return UR_RESULT_ERROR_INVALID_VALUE;

  ur_platform_handle_t Plt = Device->Platform;

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
    UR_CALL(urContextRetain(Context));
  } else {
    ContextLock.lock();
  }

  if (!UseUSMAllocator ||
      // L0 spec says that allocation fails if Alignment != 2^n, in order to
      // keep the same behavior for the allocator, just call L0 API directly and
      // return the error code.
      ((Alignment & (Alignment - 1)) != 0)) {
    ur_result_t Res =
        USMDeviceAllocImpl(RetMem, Context, Device, nullptr, Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*RetMem),
                                 std::forward_as_tuple(Context));
    }
    return Res;
  }

  umf_memory_pool_handle_t hPoolInternal = nullptr;
  if (Pool) {
    hPoolInternal = Pool->DeviceMemPools[Device].get();
  } else {
    auto It = Context->DeviceMemPools.find(Device->ZeDevice);
    if (It == Context->DeviceMemPools.end())
      return UR_RESULT_ERROR_INVALID_VALUE;

    hPoolInternal = It->second.get();
  }

  *RetMem = umfPoolAlignedMalloc(hPoolInternal, Size, Alignment);
  if (*RetMem == nullptr) {
    auto umfRet = umfPoolGetLastAllocationError(hPoolInternal);
    return umf2urResult(umfRet);
  }

  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(*RetMem),
                               std::forward_as_tuple(Context));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMSharedAlloc(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_device_handle_t Device,   ///< [in] handle of the device object
    const ur_usm_desc_t
        *USMDesc, ///< [in][optional] USM memory allocation descriptor
    ur_usm_pool_handle_t Pool, ///< [in][optional] Pointer to a pool created
                               ///< using urUSMPoolCreate
    size_t
        Size, ///< [in] size in bytes of the USM memory object to be allocated
    void **RetMem ///< [out] pointer to USM shared memory object
) {

  uint32_t Alignment = USMDesc ? USMDesc->align : 0;

  ur_usm_host_mem_flags_t UsmHostFlags{};

  // See if the memory is going to be read-only on the device.
  bool DeviceReadOnly = false;
  ur_usm_device_mem_flags_t UsmDeviceFlags{};

  void *pNext = USMDesc ? const_cast<void *>(USMDesc->pNext) : nullptr;
  while (pNext != nullptr) {
    const ur_base_desc_t *BaseDesc =
        reinterpret_cast<const ur_base_desc_t *>(pNext);
    if (BaseDesc->stype == UR_STRUCTURE_TYPE_USM_DEVICE_DESC) {
      const ur_usm_device_desc_t *UsmDeviceDesc =
          reinterpret_cast<const ur_usm_device_desc_t *>(pNext);
      UsmDeviceFlags = UsmDeviceDesc->flags;
    }
    if (BaseDesc->stype == UR_STRUCTURE_TYPE_USM_HOST_DESC) {
      const ur_usm_host_desc_t *UsmHostDesc =
          reinterpret_cast<const ur_usm_host_desc_t *>(pNext);
      UsmHostFlags = UsmHostDesc->flags;
    }
    pNext = const_cast<void *>(BaseDesc->pNext);
  }
  DeviceReadOnly = UsmDeviceFlags & UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY;

  // L0 supports alignment up to 64KB and silently ignores higher values.
  // We flag alignment > 64KB as an invalid value.
  if (Alignment > 65536)
    return UR_RESULT_ERROR_INVALID_VALUE;

  ur_platform_handle_t Plt = Device->Platform;

  // If indirect access tracking is enabled then lock the mutex which is
  // guarding contexts container in the platform. This prevents new kernels from
  // being submitted in any context while we are in the process of allocating a
  // memory, this is needed to properly capture allocations by kernels with
  // indirect access. This lock also protects access to the context's data
  // structures. If indirect access tracking is not enabled then lock context
  // mutex to protect access to context's data structures.
  std::scoped_lock<ur_shared_mutex> Lock(
      IndirectAccessTrackingEnabled ? Plt->ContextsMutex : Context->Mutex);

  if (IndirectAccessTrackingEnabled) {
    // We are going to defer memory release if there are kernels with indirect
    // access, that is why explicitly retain context to be sure that it is
    // released after all memory allocations in this context are released.
    UR_CALL(urContextRetain(Context));
  }

  if (!UseUSMAllocator ||
      // L0 spec says that allocation fails if Alignment != 2^n, in order to
      // keep the same behavior for the allocator, just call L0 API directly and
      // return the error code.
      ((Alignment & (Alignment - 1)) != 0)) {
    ur_result_t Res = USMSharedAllocImpl(RetMem, Context, Device, &UsmHostFlags,
                                         &UsmDeviceFlags, Size, Alignment);
    if (IndirectAccessTrackingEnabled) {
      // Keep track of all memory allocations in the context
      Context->MemAllocs.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(*RetMem),
                                 std::forward_as_tuple(Context));
    }
    return Res;
  }

  umf_memory_pool_handle_t hPoolInternal = nullptr;
  if (Pool) {
    hPoolInternal = (DeviceReadOnly)
                        ? Pool->SharedReadOnlyMemPools[Device].get()
                        : Pool->SharedMemPools[Device].get();
  } else {
    auto &Allocator = (DeviceReadOnly ? Context->SharedReadOnlyMemPools
                                      : Context->SharedMemPools);
    auto It = Allocator.find(Device->ZeDevice);
    if (It == Allocator.end())
      return UR_RESULT_ERROR_INVALID_VALUE;

    hPoolInternal = It->second.get();
  }

  *RetMem = umfPoolAlignedMalloc(hPoolInternal, Size, Alignment);
  if (*RetMem == nullptr) {
    auto umfRet = umfPoolGetLastAllocationError(hPoolInternal);
    return umf2urResult(umfRet);
  }

  if (DeviceReadOnly) {
    Context->SharedReadOnlyAllocs.insert(*RetMem);
  }
  if (IndirectAccessTrackingEnabled) {
    // Keep track of all memory allocations in the context
    Context->MemAllocs.emplace(std::piecewise_construct,
                               std::forward_as_tuple(*RetMem),
                               std::forward_as_tuple(Context));
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(
    ur_context_handle_t Context, ///< [in] handle of the context object
    void *Mem                    ///< [in] pointer to USM memory object
) {
  ur_platform_handle_t Plt = Context->getPlatform();

  std::scoped_lock<ur_shared_mutex> Lock(
      IndirectAccessTrackingEnabled ? Plt->ContextsMutex : Context->Mutex);

  return USMFreeHelper(Context, Mem);
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMGetMemAllocInfo(
    ur_context_handle_t Context, ///< [in] handle of the context object
    const void *Ptr,             ///< [in] pointer to USM memory object
    ur_usm_alloc_info_t
        PropName, ///< [in] the name of the USM allocation property to query
    size_t PropValueSize, ///< [in] size in bytes of the USM allocation property
                          ///< value
    void *PropValue, ///< [out][optional] value of the USM allocation property
    size_t *PropValueSizeRet ///< [out][optional] bytes returned in USM
                             ///< allocation property
) {
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
      urPrint("urUSMGetMemAllocInfo: unexpected usm memory type\n");
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
  default:
    urPrint("urUSMGetMemAllocInfo: unsupported ParamName\n");
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
  return UR_RESULT_SUCCESS;
}

static ur_result_t USMFreeImpl(ur_context_handle_t Context, void *Ptr) {
  ZE2UR_CALL(zeMemFree, (Context->ZeContext, Ptr));
  return UR_RESULT_SUCCESS;
}

static ur_result_t USMQueryPageSize(ur_context_handle_t Context, void *Ptr,
                                    size_t *PageSize) {
  ZeStruct<ze_memory_allocation_properties_t> AllocProperties = {};
  ZE2UR_CALL(zeMemGetAllocProperties,
             (Context->ZeContext, Ptr, &AllocProperties, nullptr));
  *PageSize = AllocProperties.pageSize;

  return UR_RESULT_SUCCESS;
}

umf_result_t USMMemoryProvider::initialize(ur_context_handle_t Ctx,
                                           ur_device_handle_t Dev) {
  Context = Ctx;
  Device = Dev;

  // Query L0 for the minimal page size and cache it in 'MinPageSize'
  void *Ptr;
  auto Res = allocateImpl(&Ptr, 1, 0);
  if (Res != UR_RESULT_SUCCESS) {
    goto err_set_status;
  }

  Res = USMQueryPageSize(Context, Ptr, &MinPageSize);
  if (Res != UR_RESULT_SUCCESS) {
    goto err_set_status;
  }

  Res = USMFreeImpl(Context, Ptr);
  if (Res != UR_RESULT_SUCCESS) {
    goto err_set_status;
  }

  return UMF_RESULT_SUCCESS;

err_set_status:
  getLastStatusRef() = Res;
  return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
}

enum umf_result_t USMMemoryProvider::alloc(size_t Size, size_t Align,
                                           void **Ptr) {
  auto Res = allocateImpl(Ptr, Size, Align);
  if (Res != UR_RESULT_SUCCESS) {
    getLastStatusRef() = Res;
    return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
  }

  return UMF_RESULT_SUCCESS;
}

enum umf_result_t USMMemoryProvider::free(void *Ptr, size_t Size) {
  (void)Size;

  auto Res = USMFreeImpl(Context, Ptr);
  if (Res != UR_RESULT_SUCCESS) {
    getLastStatusRef() = Res;
    return UMF_RESULT_ERROR_MEMORY_PROVIDER_SPECIFIC;
  }

  return UMF_RESULT_SUCCESS;
}

void USMMemoryProvider::get_last_native_error(const char **ErrMsg,
                                              int32_t *ErrCode) {
  (void)ErrMsg;
  *ErrCode = static_cast<int32_t>(getLastStatusRef());
}

umf_result_t USMMemoryProvider::get_min_page_size(void *Ptr, size_t *PageSize) {
  (void)Ptr;
  *PageSize = MinPageSize;
  return UMF_RESULT_SUCCESS;
}

ur_result_t USMSharedMemoryProvider::allocateImpl(void **ResultPtr, size_t Size,
                                                  uint32_t Alignment) {
  return USMSharedAllocImpl(ResultPtr, Context, Device, nullptr, nullptr, Size,
                            Alignment);
}

ur_result_t USMSharedReadOnlyMemoryProvider::allocateImpl(void **ResultPtr,
                                                          size_t Size,
                                                          uint32_t Alignment) {
  ur_usm_device_desc_t UsmDeviceDesc{};
  UsmDeviceDesc.flags = UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY;
  ur_usm_host_desc_t UsmHostDesc{};
  return USMSharedAllocImpl(ResultPtr, Context, Device, &UsmDeviceDesc.flags,
                            &UsmHostDesc.flags, Size, Alignment);
}

ur_result_t USMDeviceMemoryProvider::allocateImpl(void **ResultPtr, size_t Size,
                                                  uint32_t Alignment) {
  return USMDeviceAllocImpl(ResultPtr, Context, Device, nullptr, Size,
                            Alignment);
}

ur_result_t USMHostMemoryProvider::allocateImpl(void **ResultPtr, size_t Size,
                                                uint32_t Alignment) {
  return USMHostAllocImpl(ResultPtr, Context, nullptr, Size, Alignment);
}

ur_usm_pool_handle_t_::ur_usm_pool_handle_t_(ur_context_handle_t Context,
                                             ur_usm_pool_desc_t *PoolDesc) {

  zeroInit = static_cast<uint32_t>(PoolDesc->flags &
                                   UR_USM_POOL_FLAG_ZERO_INITIALIZE_BLOCK);

  void *pNext = const_cast<void *>(PoolDesc->pNext);
  while (pNext != nullptr) {
    const ur_base_desc_t *BaseDesc =
        reinterpret_cast<const ur_base_desc_t *>(pNext);
    switch (BaseDesc->stype) {
    case UR_STRUCTURE_TYPE_USM_POOL_LIMITS_DESC: {
      const ur_usm_pool_limits_desc_t *Limits =
          reinterpret_cast<const ur_usm_pool_limits_desc_t *>(BaseDesc);
      for (auto &config : DisjointPoolConfigs.Configs) {
        config.MaxPoolableSize = Limits->maxPoolableSize;
        config.SlabMinSize = Limits->minDriverAllocSize;
      }
      break;
    }
    default: {
      urPrint("urUSMPoolCreate: unexpected chained stype\n");
      throw UsmAllocationException(UR_RESULT_ERROR_INVALID_ARGUMENT);
    }
    }
    pNext = const_cast<void *>(BaseDesc->pNext);
  }

  auto MemProvider =
      umf::memoryProviderMakeUnique<USMHostMemoryProvider>(Context, nullptr)
          .second;

  HostMemPool =
      umf::poolMakeUnique<usm::DisjointPool, 1>(
          {std::move(MemProvider)},
          this->DisjointPoolConfigs.Configs[usm::DisjointPoolMemType::Host])
          .second;

  for (auto device : Context->Devices) {
    MemProvider =
        umf::memoryProviderMakeUnique<USMDeviceMemoryProvider>(Context, device)
            .second;
    DeviceMemPools.emplace(
        std::piecewise_construct, std::make_tuple(device),
        std::make_tuple(umf::poolMakeUnique<usm::DisjointPool, 1>(
                            {std::move(MemProvider)},
                            this->DisjointPoolConfigs
                                .Configs[usm::DisjointPoolMemType::Device])
                            .second));

    MemProvider =
        umf::memoryProviderMakeUnique<USMSharedMemoryProvider>(Context, device)
            .second;
    SharedMemPools.emplace(
        std::piecewise_construct, std::make_tuple(device),
        std::make_tuple(umf::poolMakeUnique<usm::DisjointPool, 1>(
                            {std::move(MemProvider)},
                            this->DisjointPoolConfigs
                                .Configs[usm::DisjointPoolMemType::Shared])
                            .second));

    MemProvider =
        umf::memoryProviderMakeUnique<USMSharedReadOnlyMemoryProvider>(Context,
                                                                       device)
            .second;
    SharedReadOnlyMemPools.emplace(
        std::piecewise_construct, std::make_tuple(device),
        std::make_tuple(
            umf::poolMakeUnique<usm::DisjointPool, 1>(
                {std::move(MemProvider)},
                this->DisjointPoolConfigs
                    .Configs[usm::DisjointPoolMemType::SharedReadOnly])
                .second));
  }
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolCreate(
    ur_context_handle_t Context, ///< [in] handle of the context object
    ur_usm_pool_desc_t
        *PoolDesc, ///< [in] pointer to USM pool descriptor. Can be chained with
                   ///< ::ur_usm_pool_limits_desc_t
    ur_usm_pool_handle_t *Pool ///< [out] pointer to USM memory pool
) {

  try {
    *Pool = reinterpret_cast<ur_usm_pool_handle_t>(
        new ur_usm_pool_handle_t_(Context, PoolDesc));
  } catch (const UsmAllocationException &Ex) {
    return Ex.getError();
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t
urUSMPoolRetain(ur_usm_pool_handle_t Pool ///< [in] pointer to USM memory pool
) {
  Pool->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

ur_result_t
urUSMPoolRelease(ur_usm_pool_handle_t Pool ///< [in] pointer to USM memory pool
) {
  if (Pool->RefCount.decrementAndTest()) {
    delete Pool;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urUSMPoolGetInfo(
    ur_usm_pool_handle_t Pool,   ///< [in] handle of the USM memory pool
    ur_usm_pool_info_t PropName, ///< [in] name of the pool property to query
    size_t PropSize, ///< [in] size in bytes of the pool property value provided
    void *PropValue, ///< [out][typename(propName, propSize)] value of the pool
                     ///< property
    size_t *PropSizeRet ///< [out] size in bytes returned in pool property value
) {
  std::ignore = Pool;
  std::ignore = PropName;
  std::ignore = PropSize;
  std::ignore = PropValue;
  std::ignore = PropSizeRet;
  urPrint("[UR][L0] %s function not implemented!\n", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
    if (!It->second.RefCount.decrementAndTest()) {
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
    if (!It->second.RefCount.decrementAndTest()) {
      // Memory can't be deallocated yet.
      return UR_RESULT_SUCCESS;
    }

    // Reference count is zero, it is ok to free memory.
    // We don't need to track this allocation anymore.
    Context->MemAllocs.erase(It);
  }

  if (!UseUSMAllocator) {
    ur_result_t Res = USMFreeImpl(Context, Ptr);
    if (IndirectAccessTrackingEnabled)
      UR_CALL(ContextReleaseHelper(Context));
    return Res;
  }

  // Query the device of the allocation to determine the right allocator context
  ze_device_handle_t ZeDeviceHandle;
  ZeStruct<ze_memory_allocation_properties_t> ZeMemoryAllocationProperties;

  // Query memory type of the pointer we're freeing to determine the correct
  // way to do it(directly or via an allocator)
  auto ZeResult =
      ZE_CALL_NOCHECK(zeMemGetAllocProperties,
                      (Context->ZeContext, Ptr, &ZeMemoryAllocationProperties,
                       &ZeDeviceHandle));

  // Handle the case that L0 RT was already unloaded
  if (ZeResult == ZE_RESULT_ERROR_UNINITIALIZED) {
    if (IndirectAccessTrackingEnabled)
      UR_CALL(ContextReleaseHelper(Context));
    return UR_RESULT_SUCCESS;
  } else if (ZeResult) {
    return ze2urResult(ZeResult);
  }

  // If memory type is host release from host pool
  if (ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_HOST) {
    auto umfRet = umfPoolFree(Context->HostMemPool.get(), Ptr);

    if (IndirectAccessTrackingEnabled)
      UR_CALL(ContextReleaseHelper(Context));
    return umf2urResult(umfRet);
  }

  // Points out an allocation in SharedReadOnlyMemPools
  auto SharedReadOnlyAllocsIterator = Context->SharedReadOnlyAllocs.end();

  if (!ZeDeviceHandle) {
    // The only case where it is OK not have device identified is
    // if the memory is not known to the driver. We should not ever get
    // this either, probably.
    UR_ASSERT(ZeMemoryAllocationProperties.type == ZE_MEMORY_TYPE_UNKNOWN,
              UR_RESULT_ERROR_INVALID_DEVICE);
  } else {
    ur_device_handle_t Device;
    // All context member devices or their descendants are of the same platform.
    auto Platform = Context->getPlatform();
    Device = Platform->getDeviceFromNativeHandle(ZeDeviceHandle);
    UR_ASSERT(Device, UR_RESULT_ERROR_INVALID_DEVICE);

    auto DeallocationHelper =
        [Context, Device,
         Ptr](std::unordered_map<ze_device_handle_t, umf::pool_unique_handle_t>
                  &PoolMap) {
          auto It = PoolMap.find(Device->ZeDevice);
          if (It == PoolMap.end())
            return UR_RESULT_ERROR_INVALID_VALUE;

          // The right pool is found, deallocate the pointer
          auto umfRet = umfPoolFree(It->second.get(), Ptr);

          if (IndirectAccessTrackingEnabled)
            UR_CALL(ContextReleaseHelper(Context));
          return umf2urResult(umfRet);
        };

    switch (ZeMemoryAllocationProperties.type) {
    case ZE_MEMORY_TYPE_SHARED:
      // Distinguish device_read_only allocations since they have own pool.
      SharedReadOnlyAllocsIterator = Context->SharedReadOnlyAllocs.find(Ptr);
      return DeallocationHelper(SharedReadOnlyAllocsIterator !=
                                        Context->SharedReadOnlyAllocs.end()
                                    ? Context->SharedReadOnlyMemPools
                                    : Context->SharedMemPools);
    case ZE_MEMORY_TYPE_DEVICE:
      return DeallocationHelper(Context->DeviceMemPools);
    default:
      // Handled below
      break;
    }
  }

  ur_result_t Res = USMFreeImpl(Context, Ptr);
  if (SharedReadOnlyAllocsIterator != Context->SharedReadOnlyAllocs.end()) {
    Context->SharedReadOnlyAllocs.erase(SharedReadOnlyAllocsIterator);
  }
  if (IndirectAccessTrackingEnabled)
    UR_CALL(ContextReleaseHelper(Context));
  return Res;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMImportExp(ur_context_handle_t Context,
                                                   void *HostPtr, size_t Size) {
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
      ze_driver_handle_t driverHandle = Context->getPlatform()->ZeDriver;
      ZeUSMImport.doZeUSMImport(driverHandle, HostPtr, Size);
    }
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMReleaseExp(ur_context_handle_t Context,
                                                    void *HostPtr) {
  UR_ASSERT(Context, UR_RESULT_ERROR_INVALID_CONTEXT);

  // Release the imported memory.
  if (ZeUSMImport.Supported && HostPtr != nullptr)
    ZeUSMImport.doZeUSMRelease(Context->getPlatform()->ZeDriver, HostPtr);
  return UR_RESULT_SUCCESS;
}
