//===--------- address_range_handle.cpp - Level Zero Adapter --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "interfaces.hpp"
#include "platform.hpp"
#include "unified-runtime/ur_api.h"

namespace ur::level_zero {

ur_result_t urUSMGetHandleForAddressRangeExp(
    ::ur_context_handle_t hContextOpque, const void *pMem, size_t size,
    ur_exp_address_range_export_type_t exportType,
    const ur_exp_address_range_handle_properties_t *pProperties,
    ur_exp_address_range_export_t *pExport) {
  auto hContext = common_cast(hContextOpque);
  auto &Ext = hContext->getPlatform()->ZeMemGetHandleForAddressRangeExt;
  if (!Ext.Supported)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  // No flags are defined yet, and DMA-BUF is the only handle type the driver
  // can return.
  if ((pProperties && pProperties->flags != 0) ||
      exportType != UR_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD)
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;

  // The function comes straight from the driver, which expects driver handles.
  ze_context_handle_t ZeContext = nullptr;
  ZE2UR_CALL(zelLoaderTranslateHandle,
             (ZEL_HANDLE_CONTEXT, hContext->getZeHandle(),
              reinterpret_cast<void **>(&ZeContext)));

  int Fd = -1;
  ZexStruct<ur_ze_intel_mem_range_location_exp_properties_t> ZeLocation;
  ze_result_t ZeResult = ZE_CALL_NOCHECK_NAME(
      Ext.zeIntelMemGetHandleForAddressRange,
      (ZeContext, &Fd, pMem, size,
       UR_ZE_INTEL_MEM_RANGE_HANDLE_TYPE_EXP_DMA_BUF_FD, &ZeLocation),
      "zeIntelMemGetHandleForAddressRange");
  // The memory is not device memory, or its backing can't be exported.
  if (ZeResult == UR_ZE_RESULT_ERROR_INCOMPATIBLE_RESOURCE)
    return UR_RESULT_ERROR_INVALID_ARGUMENT;
  if (ZeResult != ZE_RESULT_SUCCESS)
    return ze2urResult(ZeResult);

  pExport->type = UR_EXP_ADDRESS_RANGE_EXPORT_TYPE_DMA_BUF_FD;
  pExport->handle.fd = Fd;
  pExport->offset = ZeLocation.offset;
  pExport->base = ZeLocation.base;
  pExport->size = ZeLocation.size;
  return UR_RESULT_SUCCESS;
}

} // namespace ur::level_zero
