//===--------- virtual_mem.cpp - HIP Adapter ------------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "context.hpp"
#include "event.hpp"
#include "physical_mem.hpp"

#include <cassert>

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGranularityGetInfo(
    ur_context_handle_t, ur_device_handle_t hDevice,
    [[maybe_unused]] size_t allocationSize,
    ur_virtual_mem_granularity_info_t propName, size_t propSize,
    void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  ScopedDevice Active(hDevice);
  switch (propName) {
  case UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM:
  case UR_VIRTUAL_MEM_GRANULARITY_INFO_RECOMMENDED: {
    hipMemAllocationGranularity_flags Flags =
        propName == UR_VIRTUAL_MEM_GRANULARITY_INFO_MINIMUM
            ? hipMemAllocationGranularityMinimum
            : hipMemAllocationGranularityRecommended;
    hipMemAllocationProp AllocProps = {};
    AllocProps.location.type = hipMemLocationTypeDevice;
    AllocProps.type = hipMemAllocationTypePinned;
    AllocProps.location.id = hDevice->getIndex();

    size_t Granularity;
    UR_CHECK_ERROR(
        hipMemGetAllocationGranularity(&Granularity, &AllocProps, Flags));
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
  // Reserve the virtual mem. Only need to do once for arbitrary device
  ScopedDevice Active(hContext->getDevices()[0]);
  UR_CHECK_ERROR(hipMemAddressReserve(ppStart, size, 0,
                                      const_cast<void *>(pStart), 0));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemFree(ur_context_handle_t,
                                                     const void *pStart,
                                                     size_t size) {
  UR_CHECK_ERROR(hipMemAddressFree(const_cast<void *>(pStart), size));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemSetAccess(ur_context_handle_t hContext, const void *pStart,
                      size_t size, ur_virtual_mem_access_flags_t flags) {
  // Set access for every device in the context
  for (auto &Device : hContext->getDevices()) {
    hipMemAccessDesc AccessDesc = {};
    if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE)
      AccessDesc.flags = hipMemAccessFlagsProtReadWrite;
    else if (flags & UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY)
      AccessDesc.flags = hipMemAccessFlagsProtRead;
    else
      AccessDesc.flags = hipMemAccessFlagsProtNone;
    AccessDesc.location.type = hipMemLocationTypeDevice;
    AccessDesc.location.id = Device->getIndex();
    ScopedDevice Active(Device);
    UR_CHECK_ERROR(hipMemSetAccess(const_cast<void *>(pStart), size,
                                   &AccessDesc, 1));
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urVirtualMemMap(ur_context_handle_t hContext, const void *pStart, size_t size,
                ur_physical_mem_handle_t hPhysicalMem, size_t offset,
                ur_virtual_mem_access_flags_t flags) {
  // Map the virtual mem. Only need to do once for arbitrary device
  ScopedDevice Active(hContext->getDevices()[0]);
  UR_CHECK_ERROR(hipMemMap(const_cast<void *>(pStart), size, offset,
                           hPhysicalMem->get(), 0));
  if (flags)
    UR_CHECK_ERROR(urVirtualMemSetAccess(hContext, pStart, size, flags));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemUnmap(
    ur_context_handle_t hContext, const void *pStart, size_t size) {
  // Unmap the virtual mem. Only need to do once for arbitrary device
  ScopedDevice Active(hContext->getDevices()[0]);
  UR_CHECK_ERROR(hipMemUnmap(const_cast<void *>(pStart), size));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urVirtualMemGetInfo(
    ur_context_handle_t hContext, const void *pStart,
    [[maybe_unused]] size_t size, ur_virtual_mem_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  // Set arbitrary device
  ScopedDevice Active(hContext->getDevices()[0]);
  switch (propName) {
  case UR_VIRTUAL_MEM_INFO_ACCESS_MODE: {
    hipMemLocation MemLocation = {};
    MemLocation.type = hipMemLocationTypeDevice;
    MemLocation.id = hContext->getDevices()[0]->getIndex();

    unsigned long long HipAccessFlags;
    UR_CHECK_ERROR(hipMemGetAccess(&HipAccessFlags, &MemLocation,
                                   const_cast<void *>(pStart)));

    ur_virtual_mem_access_flags_t UrAccessFlags = 0;
    if (HipAccessFlags == hipMemAccessFlagsProtReadWrite)
      UrAccessFlags = UR_VIRTUAL_MEM_ACCESS_FLAG_READ_WRITE;
    else if (HipAccessFlags == hipMemAccessFlagsProtRead)
      UrAccessFlags = UR_VIRTUAL_MEM_ACCESS_FLAG_READ_ONLY;
    return ReturnValue(UrAccessFlags);
  }
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  return UR_RESULT_SUCCESS;
}
