//===--------- physical_mem.cpp - CUDA Adapter ----------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
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
  CUmemAllocationProp AllocProps = {};
  AllocProps.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  AllocProps.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  AllocProps.location.id = hDevice->getIndex();

  CUmemGenericAllocationHandle ResHandle;
  switch (auto Result = cuMemCreate(&ResHandle, size, &AllocProps, 0)) {
  case CUDA_ERROR_INVALID_VALUE:
    return UR_RESULT_ERROR_INVALID_SIZE;
  default:
    UR_CHECK_ERROR(Result);
  }
  try {
    *phPhysicalMem = new ur_physical_mem_handle_t_(
        ResHandle, hContext, hDevice, size,
        pProperties ? *pProperties : ur_physical_mem_properties_t{});
  } catch (std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRetain(ur_physical_mem_handle_t hPhysicalMem) {
  hPhysicalMem->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urPhysicalMemRelease(ur_physical_mem_handle_t hPhysicalMem) {
  if (hPhysicalMem->decrementReferenceCount() > 0)
    return UR_RESULT_SUCCESS;

  try {
    std::unique_ptr<ur_physical_mem_handle_t_> PhysicalMemGuard(hPhysicalMem);

    ScopedContext Active(hPhysicalMem->getDevice());
    UR_CHECK_ERROR(cuMemRelease(hPhysicalMem->get()));
  } catch (ur_result_t err) {
    return err;
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urPhysicalMemGetInfo(
    ur_physical_mem_handle_t hPhysicalMem, ur_physical_mem_info_t propName,
    size_t propSize, void *pPropValue, size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);

  switch (propName) {
  case UR_PHYSICAL_MEM_INFO_CONTEXT: {
    return ReturnValue(hPhysicalMem->getContext());
  }
  case UR_PHYSICAL_MEM_INFO_DEVICE: {
    return ReturnValue(hPhysicalMem->getDevice());
  }
  case UR_PHYSICAL_MEM_INFO_SIZE: {
    return ReturnValue(hPhysicalMem->getSize());
  }
  case UR_PHYSICAL_MEM_INFO_PROPERTIES: {
    return ReturnValue(hPhysicalMem->getProperties());
  }
  case UR_PHYSICAL_MEM_INFO_REFERENCE_COUNT: {
    return ReturnValue(hPhysicalMem->getReferenceCount());
  }
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_ENUMERATION;
  }
}
