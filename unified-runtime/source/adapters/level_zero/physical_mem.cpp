//===--------- physical_mem.cpp - Level Zero Adapter ----------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"
#include "ur_interface_loader.hpp"
#include "ur_level_zero.hpp"

namespace ur::level_zero::v1 {

ur_result_t urPhysicalMemCreate(
    ::ur_context_handle_t hContextOpque, ::ur_device_handle_t hDeviceOpque,
    size_t size,
    [[maybe_unused]] const ur_physical_mem_properties_t *pProperties,
    ::ur_physical_mem_handle_t *phPhysicalMem) {
  auto hContext = common_cast(hContextOpque);
  auto hDevice = common_cast(hDeviceOpque);

  ZeStruct<ze_physical_mem_desc_t> PhysicalMemDesc;
  PhysicalMemDesc.flags = 0;
  PhysicalMemDesc.size = size;

  ze_physical_mem_handle_t ZePhysicalMem;
  ZE2UR_CALL(zePhysicalMemCreate, (hContext->getZeHandle(), hDevice->ZeDevice,
                                   &PhysicalMemDesc, &ZePhysicalMem));
  try {
    *phPhysicalMem = common_cast(new ur_physical_mem_handle_t_(
        ZePhysicalMem, hContextOpque, hDevice, size, /*EnableIpc=*/false));
  } catch (const std::bad_alloc &) {
    zePhysicalMemDestroy(hContext->getZeHandle(), ZePhysicalMem);
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    zePhysicalMemDestroy(hContext->getZeHandle(), ZePhysicalMem);
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemRetain(::ur_physical_mem_handle_t hPhysicalMemOpque) {
  common_cast(hPhysicalMemOpque)->RefCount.retain();
  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemRelease(::ur_physical_mem_handle_t hPhysicalMemOpque) {
  auto hPhysicalMem = common_cast(hPhysicalMemOpque);
  if (!hPhysicalMem->RefCount.release())
    return UR_RESULT_SUCCESS;

  if (checkL0LoaderTeardown()) {
    ZE2UR_CALL(zePhysicalMemDestroy,
               (common_cast(hPhysicalMem->Context)->getZeHandle(),
                hPhysicalMem->ZePhysicalMem));
  }
  delete hPhysicalMem;

  return UR_RESULT_SUCCESS;
}

ur_result_t urPhysicalMemGetInfo(::ur_physical_mem_handle_t hPhysicalMemOpque,
                                 ur_physical_mem_info_t propName,
                                 size_t propSize, void *pPropValue,
                                 size_t *pPropSizeRet) {
  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  auto hPhysicalMem = common_cast(hPhysicalMemOpque);

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

ur_result_t
urIPCGetPhysMemHandleExp(::ur_context_handle_t /* hContext */,
                         ::ur_physical_mem_handle_t /* hPhysMem */,
                         void ** /* ppIPCPhysMemHandleData */,
                         size_t * /* pIPCPhysMemHandleDataSizeRet */) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t urIPCPutPhysMemHandleExp(::ur_context_handle_t /* hContext */,
                                     const void * /* pIPCPhysMemHandleData */) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t
urIPCOpenPhysMemHandleExp(::ur_context_handle_t /* hContext */,
                          ::ur_device_handle_t /* hDevice */,
                          const void * /* pIPCPhysMemHandleData */,
                          size_t /* ipcPhysMemHandleDataSize */,
                          ::ur_physical_mem_handle_t * /* phPhysMem */) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

ur_result_t
urIPCClosePhysMemHandleExp(::ur_context_handle_t /* hContext */,
                           ::ur_physical_mem_handle_t /* hPhysMem */) {
  UR_LOG_LEGACY(ERR,
                logger::LegacyMessage("[UR][L0] {} function not implemented!"),
                "{} function not implemented!", __FUNCTION__);
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

} // namespace ur::level_zero::v1
