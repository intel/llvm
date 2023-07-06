//===---------------- physical_mem.cpp - Level Zero Adapter ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "physical_mem.hpp"
#include "common.hpp"
#include "context.hpp"
#include "device.hpp"
#include "ur_level_zero.hpp"

UR_APIEXPORT ur_result_t UR_APICALL urPhysicalMemCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    [[maybe_unused]] const ur_physical_mem_properties_t *pProperties,
    ur_physical_mem_handle_t *phPhysicalMem) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phPhysicalMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(size > 0, UR_RESULT_ERROR_INVALID_VALUE);

  ZeStruct<ze_physical_mem_desc_t> PhysicalMemDesc;
  PhysicalMemDesc.flags = 0;
  PhysicalMemDesc.size = size;

  ze_physical_mem_handle_t ZePhysicalMem;
  ZE2UR_CALL(zePhysicalMemCreate, (hContext->ZeContext, hDevice->ZeDevice,
                                   &PhysicalMemDesc, &ZePhysicalMem));
  try {
    *phPhysicalMem = new ur_physical_mem_handle_t_(ZePhysicalMem, hContext);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRetain(ur_physical_mem_handle_t hPhysicalMem) {
  UR_ASSERT(hPhysicalMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  hPhysicalMem->RefCount.increment();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRelease(ur_physical_mem_handle_t hPhysicalMem) {
  UR_ASSERT(hPhysicalMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  if (!hPhysicalMem->RefCount.decrementAndTest())
    return UR_RESULT_SUCCESS;

  ZE2UR_CALL(zePhysicalMemDestroy,
             (hPhysicalMem->Context->ZeContext, hPhysicalMem->ZePhysicalMem));
  delete hPhysicalMem;

  return UR_RESULT_SUCCESS;
}
