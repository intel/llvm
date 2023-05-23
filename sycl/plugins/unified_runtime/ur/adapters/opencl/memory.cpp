//===--------- memory.cpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

#include <sycl/detail/cl.h>

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags, size_t size,
    const ur_buffer_properties_t *pProperties, ur_mem_handle_t *phBuffer) {
  cl_int ret_err = CL_INVALID_OPERATION;
  if (pProperties) {
    // TODO: need to check if all properties are supported by OpenCL RT and
    // ignore unsupported
    clCreateBufferWithPropertiesINTEL_fn FuncPtr = nullptr;
    cl_context CLContext = cl::cast<cl_context>(hContext);
    // First we need to look up the function pointer
    ret_err =
        cl_ext::getExtFuncFromContext<clCreateBufferWithPropertiesINTEL_fn>(
            CLContext,
            cl_ext::ExtFuncPtrCache->clCreateBufferWithPropertiesINTELCache,
            cl_ext::clCreateBufferWithPropertiesName, &FuncPtr);
    if (FuncPtr) {
      std::vector<cl_mem_properties_intel> propertiesIntel;
      auto prop = static_cast<ur_base_properties_t *>(pProperties->pNext);
      while (prop) {
        switch (prop->stype) {
        case UR_STRUCTURE_TYPE_BUFFER_CHANNEL_PROPERTIES: {
          auto bufferChannelProperty =
              reinterpret_cast<ur_buffer_channel_properties_t *>(prop);
          propertiesIntel.push_back(CL_MEM_CHANNEL_INTEL);
          propertiesIntel.push_back(bufferChannelProperty->channel);
        } break;
        case UR_STRUCTURE_TYPE_BUFFER_ALLOC_LOCATION_PROPERTIES: {
          auto bufferLocationProperty =
              reinterpret_cast<ur_buffer_alloc_location_properties_t *>(prop);
          propertiesIntel.push_back(CL_MEM_ALLOC_FLAGS_INTEL);
          propertiesIntel.push_back(bufferLocationProperty->location);
        } break;
        default:
          break;
        }
        prop = static_cast<ur_base_properties_t *>(prop->pNext);
      }
      propertiesIntel.push_back(0);

      *phBuffer = reinterpret_cast<ur_mem_handle_t>(FuncPtr(
          CLContext, propertiesIntel.data(), static_cast<cl_mem_flags>(flags),
          size, pProperties->pHost, cl::cast<cl_int *>(&ret_err)));
      CL_RETURN_ON_FAILURE(ret_err);
    }
  }

  *phBuffer = reinterpret_cast<ur_mem_handle_t>(clCreateBuffer(
      cl::cast<cl_context>(hContext), static_cast<cl_mem_flags>(flags), size,
      pProperties->pHost, cl::cast<cl_int *>(&ret_err)));
  CL_RETURN_ON_FAILURE(ret_err);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    void *pHost, ur_mem_handle_t *phMem) {
  cl_int ret_err = CL_INVALID_OPERATION;
  *phMem = reinterpret_cast<ur_mem_handle_t>(clCreateImage(
      cl::cast<cl_context>(hContext), static_cast<cl_mem_flags>(flags),
      cl::cast<const cl_image_format *>(pImageFormat),
      cl::cast<const cl_image_desc *>(pImageDesc), pHost,
      cl::cast<cl_int *>(&ret_err)));
  CL_RETURN_ON_FAILURE(ret_err);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
    ur_buffer_create_type_t bufferCreateType, const ur_buffer_region_t *pRegion,
    ur_mem_handle_t *phMem) {
  cl_int ret_err = CL_INVALID_OPERATION;
  *phMem = reinterpret_cast<ur_mem_handle_t>(clCreateSubBuffer(
      cl::cast<cl_mem>(hBuffer), static_cast<cl_mem_flags>(flags),
      cl::cast<cl_buffer_create_type>(bufferCreateType), pRegion,
      cl::cast<cl_int *>(&ret_err)));
  CL_RETURN_ON_FAILURE(ret_err);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urMemGetNativeHandle(ur_mem_handle_t hMem, ur_native_handle_t *phNativeMem) {
  return urGetNativeHandle(hMem, phNativeMem);
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ur_context_handle_t hContext,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem) {
  (void)hContext;
  (void)pProperties;
  UR_ASSERT(phMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  *phMem = reinterpret_cast<ur_mem_handle_t>(hNativeMem);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreateWithNativeHandle(
    ur_native_handle_t hNativeMem, ur_context_handle_t hContext,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem) {
  (void)hContext;
  (void)pImageFormat;
  (void)pImageDesc;
  (void)pProperties;
  UR_ASSERT(phMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  *phMem = reinterpret_cast<ur_mem_handle_t>(hNativeMem);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetInfo(ur_mem_handle_t hMemory,
                                                 ur_mem_info_t propName,
                                                 size_t propSize,
                                                 void *pPropValue,
                                                 size_t *pPropSizeRet) {
  CL_RETURN_ON_FAILURE(clGetMemObjectInfo(cl::cast<cl_mem>(hMemory), propName, propSize,
                               pPropValue, pPropSizeRet));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageGetInfo(ur_mem_handle_t hMemory,
                                                      ur_image_info_t propName,
                                                      size_t propSize,
                                                      void *pPropValue,
                                                      size_t *pPropSizeRet) { 
  CL_RETURN_ON_FAILURE(clGetImageInfo(cl::cast<cl_mem>(hMemory), propName, propSize,
                           pPropValue, pPropSizeRet));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(ur_mem_handle_t hMem) {
  CL_RETURN_ON_FAILURE(clRetainMemObject(cl::cast<cl_mem>(hMem)));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRelease(ur_mem_handle_t hMem) {
  CL_RETURN_ON_FAILURE(clReleaseMemObject(cl::cast<cl_mem>(hMem)));
  return UR_RESULT_SUCCESS;
}
