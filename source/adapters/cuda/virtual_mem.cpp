//===--------- virtual_mem.cpp - CUDA Adapter -----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "physical_mem.hpp"

#include <cassert>
#include <cuda.h>

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGranularityGetInfo(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_virtual_mem_granularity_info_t propName, size_t propSize,
    void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  ScopedContext Active(hContext);
  switch (propName) {
  case UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM:
  case UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED: {
    CUmemAllocationGranularity_flags Flags =
        propName == UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM
            ? CU_MEM_ALLOC_GRANULARITY_MINIMUM
            : CU_MEM_ALLOC_GRANULARITY_RECOMMENDED;
    CUmemAllocationProp AllocProps = {};
    AllocProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    AllocProps.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    UR_ASSERT(GetDeviceOrdinal(hDevice, AllocProps.location.id),
              UR_RESULT_ERROR_INVALID_DEVICE);

    size_t Granularity;
    UR_CHECK_ERROR(
        cuMemGetAllocationGranularity(&Granularity, &AllocProps, Flags));
    return ReturnValue(Granularity);
  }
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemReserve(ur_context_handle_t hContext, const void *pStart,
                    size_t size, void **ppStart) {
  ScopedContext Active(hContext);
  return UR_CHECK_ERROR(cuMemAddressReserve((CUdeviceptr *)ppStart, size, 0,
                                            (CUdeviceptr)pStart, 0));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemFree(
    ur_context_handle_t hContext, const void *pStart, size_t size) {
  ScopedContext Active(hContext);
  UR_CHECK_ERROR(cuMemAddressFree((CUdeviceptr)pStart, size));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemSetAccess(ur_context_handle_t hContext, const void *pStart,
                      size_t size, ur_virtual_mem_access_flags_t flags) {
  CUmemAccessDesc AccessDesc;
  if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE)
    AccessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  else if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY)
    AccessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
  else
    AccessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
  AccessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  // TODO: When contexts support multiple devices, we should create a descriptor
  //       for each. We may also introduce a variant of this function with a
  //       specific device.
  UR_ASSERT(GetDeviceOrdinal(hContext->getDevice(), AccessDesc.location.id),
            UR_RESULT_ERROR_INVALID_DEVICE);

  ScopedContext Active(hContext);
  UR_CHECK_ERROR(cuMemSetAccess((CUdeviceptr)pStart, size, &AccessDesc, 1));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemMap(ur_context_handle_t hContext, const void *pStart, size_t size,
                ur_physical_mem_handle_t hPhysicalMem, size_t offset,
                ur_virtual_mem_access_flags_t flags) {
  ScopedContext Active(hContext);
  UR_CHECK_ERROR(
      cuMemMap((CUdeviceptr)pStart, size, offset, hPhysicalMem->get(), 0));
  if (flags)
    UR_ASSERT(urVirtualMemSetAccess(hContext, pStart, size, flags),
              UR_RESULT_ERROR_INVALID_ARGUMENT);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemUnmap(
    ur_context_handle_t hContext, const void *pStart, size_t size) {
  ScopedContext Active(hContext);
  UR_CHECK_ERROR(cuMemUnmap((CUdeviceptr)pStart, size));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGetInfo(
    ur_context_handle_t hContext, const void *pStart,
    [[maybe_unused]] size_t size, ur_virtual_mem_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  ScopedContext Active(hContext);
  switch (propName) {
  case UR_VIRTUAL_MEM_INFO_ACCESS_MODE: {
    CUmemLocation MemLocation = {};
    MemLocation.type = CU_MEM_LOCATION_TYPE_DEVICE;
    UR_ASSERT(GetDeviceOrdinal(hContext->getDevice(), MemLocation.id),
              UR_RESULT_ERROR_INVALID_DEVICE);

    unsigned long long CuAccessFlags;
    UR_CHECK_ERROR(
        cuMemGetAccess(&CuAccessFlags, &MemLocation, (CUdeviceptr)pStart));

    ur_virtual_mem_access_flags_t UrAccessFlags = 0;
    if (CuAccessFlags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
      UrAccessFlags = UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE;
    else if (CuAccessFlags == CU_MEM_ACCESS_FLAGS_PROT_READ)
      UrAccessFlags = UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY;
    return ReturnValue(UrAccessFlags);
  }
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  return UR_RESULT_SUCCESS;
}
