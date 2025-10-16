//===-------------- memory.cpp - Native CPU Adapter -----------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memory.hpp"
#include "common.hpp"
#include "ur_api.h"

UR_APIEXPORT ur_result_t UR_APICALL
urMemImageCreate(ur_context_handle_t /*hContext*/, ur_mem_flags_t /*flags*/,
                 const ur_image_format_t * /*pImageFormat*/,
                 const ur_image_desc_t * /*pImageDesc*/, void * /*pHost*/,
                 ur_mem_handle_t * /*phMem*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags, size_t size,
    const ur_buffer_properties_t *pProperties, ur_mem_handle_t *phBuffer) {

  // TODO: add proper error checking and double check flag semantics
  // TODO: support UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER flag

  UR_ASSERT(phBuffer, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  UR_ASSERT((flags & UR_MEM_FLAGS_MASK) == 0,
            UR_RESULT_ERROR_INVALID_ENUMERATION);
  if (flags &
      (UR_MEM_FLAG_USE_HOST_POINTER | UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER)) {
    UR_ASSERT(pProperties && pProperties->pHost,
              UR_RESULT_ERROR_INVALID_HOST_PTR);
  }

  UR_ASSERT(size != 0, UR_RESULT_ERROR_INVALID_BUFFER_SIZE);

  const bool useHostPtr = flags & UR_MEM_FLAG_USE_HOST_POINTER;
  const bool copyHostPtr = flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER;

  ur_mem_handle_t_ *retMem;

  if (useHostPtr) {
    retMem = new ur_buffer(hContext, pProperties->pHost);
  } else if (copyHostPtr) {
    retMem = new ur_buffer(hContext, pProperties->pHost, size);
  } else {
    retMem = new ur_buffer(hContext, size);
  }

  *phBuffer = retMem;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(ur_mem_handle_t /*hMem*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRelease(ur_mem_handle_t hMem) {
  UR_ASSERT(hMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  decrementOrDelete(hMem);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
    ur_buffer_create_type_t /*bufferCreateType*/,
    const ur_buffer_region_t *pRegion, ur_mem_handle_t *phMem) {

  UR_ASSERT(hBuffer && !hBuffer->isImage() &&
                !(static_cast<ur_buffer *>(hBuffer))->isSubBuffer(),
            UR_RESULT_ERROR_INVALID_MEM_OBJECT);

  std::shared_lock<ur_shared_mutex> Guard(hBuffer->Mutex);

  if (flags != UR_MEM_FLAG_READ_WRITE) {
    die("urMemBufferPartition: NativeCPU implements only read-write buffer,"
        "no read-only or write-only yet.");
  }

  try {
    auto partitionedBuffer = new ur_buffer(static_cast<ur_buffer *>(hBuffer),
                                           pRegion->origin, pRegion->size);
    *phMem = reinterpret_cast<ur_mem_handle_t>(partitionedBuffer);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urMemGetNativeHandle(ur_mem_handle_t /*hMem*/, ur_device_handle_t /*hDevice*/,
                     ur_native_handle_t * /*phNativeMem*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreateWithNativeHandle(
    ur_native_handle_t /*hNativeMem*/, ur_context_handle_t /*hContext*/,
    const ur_mem_native_properties_t * /*pProperties*/,
    ur_mem_handle_t * /*phMem*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreateWithNativeHandle(
    ur_native_handle_t /*hNativeMem*/, ur_context_handle_t /*hContext*/,
    const ur_image_format_t * /*pImageFormat*/,
    const ur_image_desc_t * /*pImageDesc*/,
    const ur_mem_native_properties_t * /*pProperties*/,
    ur_mem_handle_t * /*phMem*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetInfo(ur_mem_handle_t /*hMemory*/,
                                                 ur_mem_info_t /*propName*/,
                                                 size_t /*propSize*/,
                                                 void * /*pPropValue*/,
                                                 size_t * /*pPropSizeRet*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageGetInfo(
    ur_mem_handle_t /*hMemory*/, ur_image_info_t /*propName*/,
    size_t /*propSize*/, void * /*pPropValue*/, size_t * /*pPropSizeRet*/) {

  DIE_NO_IMPLEMENTATION;
}

UR_APIEXPORT ur_result_t UR_APICALL urIPCGetMemHandleExp(ur_context_handle_t,
                                                         void *, void *,
                                                         size_t *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urIPCPutMemHandleExp(ur_context_handle_t,
                                                         void *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urIPCOpenMemHandleExp(ur_context_handle_t,
                                                          ur_device_handle_t,
                                                          void *, size_t,
                                                          void **) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urIPCCloseMemHandleExp(ur_context_handle_t,
                                                           void *) {
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}
