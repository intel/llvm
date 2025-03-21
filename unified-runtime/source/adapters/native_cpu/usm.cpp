//===------------- usm.cpp - NATIVE CPU Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ur/ur.hpp"
#include "ur_api.h"

#include "common.hpp"
#include "context.hpp"
#include <cstdlib>

#ifdef __unix__
#include <sys/mman.h>
#endif

namespace umf {
ur_result_t getProviderNativeError(const char *, int32_t) {
  return UR_RESULT_ERROR_UNKNOWN;
}
} // namespace umf

// Device flags are merged with host flags for native CPU since they are the
// same target. If flags conflict, e.g. WRITE_ONLY | READ_ONLY, the behaviour is
// unspecified.
static inline ur_usm_host_mem_flags_t
merge_flags(ur_usm_host_mem_flags_t HF, ur_usm_device_mem_flags_t DF) {
  HF |= UR_USM_HOST_MEM_FLAG_INITIAL_PLACEMENT *
        !!(UR_USM_DEVICE_MEM_FLAG_INITIAL_PLACEMENT & DF);
  HF |= UR_USM_HOST_MEM_FLAG_HOST_READ_ONLY *
        !!(UR_USM_DEVICE_MEM_FLAG_DEVICE_READ_ONLY & DF);
  HF |= UR_USM_HOST_MEM_FLAG_HOST_WRITE_ONLY *
        !!(UR_USM_DEVICE_MEM_FLAG_DEVICE_WRITE_ONLY & DF);
  HF |= UR_USM_HOST_MEM_FLAG_HOST_COHERENT *
        !!(UR_USM_DEVICE_MEM_FLAG_DEVICE_COHERENT & DF);
  HF |= UR_USM_HOST_MEM_FLAG_HOST_NON_COHERENT *
        !!(UR_USM_DEVICE_MEM_FLAG_DEVICE_NON_COHERENT & DF);
  HF |= UR_USM_HOST_MEM_FLAG_HOST_ACCESS_RANDOM *
        !!(UR_USM_DEVICE_MEM_FLAG_DEVICE_ACCESS_RANDOM & DF);
  HF |= UR_USM_HOST_MEM_FLAG_HOST_ACCESS_SEQUENTIAL *
        !!(UR_USM_DEVICE_MEM_FLAG_DEVICE_ACCESS_SEQUENTIAL & DF);
  HF |= UR_USM_HOST_MEM_FLAG_HOST_ACCESS_HOT *
        !!(UR_USM_DEVICE_MEM_FLAG_DEVICE_ACCESS_HOT & DF);
  HF |= UR_USM_HOST_MEM_FLAG_HOST_ACCESS_COLD *
        !!(UR_USM_DEVICE_MEM_FLAG_DEVICE_ACCESS_COLD & DF);
  HF |= UR_USM_HOST_MEM_FLAG_HOST_UNCACHED *
        !!(UR_USM_DEVICE_MEM_FLAG_DEVICE_UNCACHED & DF);
  HF |= UR_USM_HOST_MEM_FLAG_WRITE_COMBINE *
        !!(UR_USM_DEVICE_MEM_FLAG_WRITE_COMBINE & DF);
  return HF;
}

static void advise_host_flags(ur_usm_host_mem_flags_t URF, void *P,
                              size_t Size) {
  static_assert(std::is_unsigned_v<ur_usm_host_mem_flags_t>,
                "UB in left shift on signed type");
  for (ur_usm_host_mem_flags_t F = 1; F; F <<= 1) {
    switch (URF & F) {
    case 0:
      continue;
#ifndef _WIN32 // Assume POSIX if not Windows
    case UR_USM_HOST_MEM_FLAG_HOST_ACCESS_SEQUENTIAL:
      posix_madvise(P, Size, POSIX_MADV_SEQUENTIAL);
      continue;
    case UR_USM_HOST_MEM_FLAG_HOST_ACCESS_RANDOM:
      posix_madvise(P, Size, POSIX_MADV_RANDOM);
      continue;
#ifdef __linux__
    // These advice are Linux-specific and are not part of POSIX. Thus we
    // use `madvise` instead of the generic `posix_memadvise`
    case UR_USM_HOST_MEM_FLAG_HOST_ACCESS_HOT:
      madvise(P, Size, MADV_WILLNEED);
    case UR_USM_HOST_MEM_FLAG_HOST_ACCESS_COLD:
      madvise(P, Size, MADV_COLD);
    }
  }
#endif
#endif
}

static ur_result_t alloc_helper(ur_context_handle_t hContext,
                                const ur_usm_desc_t *pUSMDesc, size_t size,
                                void **ppMem, ur_usm_type_t type) {
  auto alignment = (pUSMDesc && pUSMDesc->align) ? pUSMDesc->align : 1u;
  UR_ASSERT(isPowerOf2(alignment), UR_RESULT_ERROR_UNSUPPORTED_ALIGNMENT);
  UR_ASSERT(ppMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  // TODO: Check Max size when UR_DEVICE_INFO_MAX_MEM_ALLOC_SIZE is implemented
  UR_ASSERT(size > 0, UR_RESULT_ERROR_INVALID_USM_SIZE);
  ur_usm_host_mem_flags_t HostFlags{};
  ur_usm_device_mem_flags_t DeviceFlags{};

  auto *ptr = hContext->add_alloc(alignment, type, size, nullptr);
  UR_ASSERT(ptr != nullptr, UR_RESULT_ERROR_OUT_OF_RESOURCES);

  if (const auto *HD = find_stype_node<ur_usm_host_desc_t>(pUSMDesc)) {
    HostFlags = HD->flags;
  }
  if (const auto *DD = find_stype_node<ur_usm_device_desc_t>(pUSMDesc)) {
    DeviceFlags = DD->flags;
  }
  if (auto Flags = merge_flags(HostFlags, DeviceFlags)) {
    advise_host_flags(Flags, ptr, size);
  }
  *ppMem = ptr;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMHostAlloc(ur_context_handle_t hContext, const ur_usm_desc_t *pUSMDesc,
               ur_usm_pool_handle_t pool, size_t size, void **ppMem) {
  std::ignore = pool;

  return alloc_helper(hContext, pUSMDesc, size, ppMem, UR_USM_TYPE_HOST);
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMDeviceAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t pool,
                 size_t size, void **ppMem) {
  std::ignore = hDevice;
  std::ignore = pool;

  return alloc_helper(hContext, pUSMDesc, size, ppMem, UR_USM_TYPE_DEVICE);
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMSharedAlloc(ur_context_handle_t hContext, ur_device_handle_t hDevice,
                 const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t pool,
                 size_t size, void **ppMem) {
  std::ignore = hDevice;
  std::ignore = pool;

  return alloc_helper(hContext, pUSMDesc, size, ppMem, UR_USM_TYPE_SHARED);
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMFree(ur_context_handle_t hContext,
                                              void *pMem) {

  UR_ASSERT(pMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  auto res = hContext->remove_alloc(pMem);

  return res;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMGetMemAllocInfo(ur_context_handle_t hContext, const void *pMem,
                     ur_usm_alloc_info_t propName, size_t propSize,
                     void *pPropValue, size_t *pPropSizeRet) {

  UR_ASSERT(pMem != nullptr, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  if (propName == UR_USM_ALLOC_INFO_BASE_PTR) {
    // TODO: logic to compute base ptr given ptr
    DIE_NO_IMPLEMENTATION;
  }

  const native_cpu::usm_alloc_info &alloc_info =
      hContext->get_alloc_info_entry(pMem);
  switch (propName) {
  case UR_USM_ALLOC_INFO_TYPE:
    return ReturnValue(alloc_info.type);
  case UR_USM_ALLOC_INFO_SIZE:
    return ReturnValue(alloc_info.size);
  case UR_USM_ALLOC_INFO_DEVICE:
    return ReturnValue(alloc_info.device);
  case UR_USM_ALLOC_INFO_POOL:
    return ReturnValue(alloc_info.pool);
  default:
    DIE_NO_IMPLEMENTATION;
  }
  return UR_RESULT_ERROR_INVALID_VALUE;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolCreate(ur_context_handle_t hContext, ur_usm_pool_desc_t *pPoolDesc,
                ur_usm_pool_handle_t *ppPool) {
  std::ignore = hContext;
  std::ignore = pPoolDesc;
  std::ignore = ppPool;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolRetain(ur_usm_pool_handle_t pPool) {
  std::ignore = pPool;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolRelease(ur_usm_pool_handle_t pPool) {
  std::ignore = pPool;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL
urUSMPoolGetInfo(ur_usm_pool_handle_t hPool, ur_usm_pool_info_t propName,
                 size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  std::ignore = hPool;
  std::ignore = propName;
  std::ignore = propSize;
  std::ignore = pPropValue;
  std::ignore = pPropSizeRet;

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMImportExp(ur_context_handle_t Context,
                                                   void *HostPtr, size_t Size) {
  std::ignore = Context;
  std::ignore = HostPtr;
  std::ignore = Size;
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMReleaseExp(ur_context_handle_t Context,
                                                    void *HostPtr) {
  std::ignore = Context;
  std::ignore = HostPtr;
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolCreateExp(ur_context_handle_t,
                                                       ur_device_handle_t,
                                                       ur_usm_pool_desc_t *,
                                                       ur_usm_pool_handle_t *) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolDestroyExp(ur_context_handle_t,
                                                        ur_device_handle_t,
                                                        ur_usm_pool_handle_t) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolSetThresholdExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t, size_t) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetDefaultDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t *) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetInfoExp(ur_usm_pool_handle_t,
                                                        ur_usm_pool_info_t,
                                                        void *, size_t *) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolGetDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t *) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolSetDevicePoolExp(
    ur_context_handle_t, ur_device_handle_t, ur_usm_pool_handle_t) {
  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPoolTrimToExp(ur_context_handle_t,
                                                       ur_device_handle_t,
                                                       ur_usm_pool_handle_t,
                                                       size_t) {
  DIE_NO_IMPLEMENTATION;
}
