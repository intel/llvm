//===--------- physical_mem.cpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "physical_mem.hpp"
#include "common.hpp"
#include "context.hpp"
#include "event.hpp"

#include <cassert>
#include <cuda.h>

UR_APIEXPORT ur_result_t UR_APICALL urPhysicalMemCreate(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    [[maybe_unused]] const ur_physical_mem_properties_t *pProperties,
    ur_physical_mem_handle_t *phPhysicalMem) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hDevice, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phPhysicalMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  UR_ASSERT(size > 0, UR_RESULT_ERROR_INVALID_VALUE);

  CUmemAllocationProp AllocProps = {};
  AllocProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  AllocProps.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  UR_ASSERT(GetDeviceOrdinal(hDevice, alloc_props.location.id),
            UR_RESULT_ERROR_INVALID_DEVICE);

  CUmemGenericAllocationHandle ResHandle;
  UR_CHECK_ERROR(cuMemCreate(&ResHandle, size, &AllocProps, 0));
  *phPhysicalMem = new ur_physical_mem_handle_t_(ResHandle, hContext);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRetain(ur_physical_mem_handle_t hPhysicalMem) {
  UR_ASSERT(hPhysicalMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  hPhysicalMem->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRelease(ur_physical_mem_handle_t hPhysicalMem) {
  UR_ASSERT(hPhysicalMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  if (hPhysicalMem->decrementReferenceCount() > 0)
    return UR_RESULT_SUCCESS;

  try {
    std::unique_ptr<ur_physical_mem_handle_t_> PhysicalMemGuard(hPhysicalMem);

    ScopedContext Active(hPhysicalMem->getContext());
    UR_CHECK_ERROR(cuMemRelease(hPhysicalMem->get()));
    return UR_RESULT_SUCCESS;
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
}
