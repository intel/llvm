//===---------------- physical_mem.cpp - Level Zero Adapter ---------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "physical_mem.hpp"
#include "common.hpp"
#include "context.hpp"
#include "device.hpp"

namespace ur::level_zero {

ur_result_t urPhysicalMemCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    [[maybe_unused]] const ur_physical_mem_properties_t *pProperties,
    ur_physical_mem_handle_t *phPhysicalMem) {
  ZeStruct<ze_physical_mem_desc_t> PhysicalMemDesc;
  PhysicalMemDesc.flags = 0;
  PhysicalMemDesc.size = size;

  ze_physical_mem_handle_t ZePhysicalMem;
  ZE2UR_CALL(zePhysicalMemCreate, (hContext->getZeHandle(), hDevice->ZeDevice,
                                   &PhysicalMemDesc, &ZePhysicalMem));
  try {
    *phPhysicalMem =
        new ur_physical_mem_handle_t_(ZePhysicalMem, hContext, hDevice, size,
                                      /*EnableIpc=*/false);
  } catch (const std::bad_alloc &) {
    zePhysicalMemDestroy(hContext->getZeHandle(), ZePhysicalMem);
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    zePhysicalMemDestroy(hContext->getZeHandle(), ZePhysicalMem);
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemRetain(ur_physical_mem_handle_t hPhysicalMem) {
  hPhysicalMem->RefCount.retain();
  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemRelease(ur_physical_mem_handle_t hPhysicalMem) {
  if (!hPhysicalMem->RefCount.release())
    return UR_RESULT_SUCCESS;

  if (checkL0LoaderTeardown()) {
    ZE2UR_CALL(zePhysicalMemDestroy, (hPhysicalMem->Context->getZeHandle(),
                                      hPhysicalMem->ZePhysicalMem));
  }
  delete hPhysicalMem;

  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemGetInfo(ur_physical_mem_handle_t hPhysicalMem,
                                 ur_physical_mem_info_t propName,
                                 size_t propSize, void *pPropValue,
                                 size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_PHYSICAL_MEM_INFO_CONTEXT:
    return ReturnValue(hPhysicalMem->Context);
  case UR_PHYSICAL_MEM_INFO_DEVICE:
    return ReturnValue(hPhysicalMem->Device);
  case UR_PHYSICAL_MEM_INFO_SIZE:
    return ReturnValue(hPhysicalMem->Size);
  case UR_PHYSICAL_MEM_INFO_PROPERTIES: {
    ur_physical_mem_flags_t Flags = static_cast<ur_physical_mem_flags_t>(0);
    ur_physical_mem_properties_t Props = {
        UR_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES, nullptr, Flags};
    return ReturnValue(Props);
  }
  case UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT:
    return ReturnValue(hPhysicalMem->RefCount.getCount());
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urIPCGetPhysMemHandleExp(ur_context_handle_t,
                                     ur_physical_mem_handle_t, void **,
                                     size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCPutPhysMemHandleExp(ur_context_handle_t, const void *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCOpenPhysMemHandleExp(ur_context_handle_t, ur_device_handle_t,
                                      const void *, size_t,
                                      ur_physical_mem_handle_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCClosePhysMemHandleExp(ur_context_handle_t,
                                       ur_physical_mem_handle_t) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero
