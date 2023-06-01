//===--------- memory.cpp - OpenCL Adapter ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//

#include "common.hpp"

#include <sycl/detail/cl.h>

cl_image_format
map_ur_image_format_to_cl(const ur_image_format_t *pImageFormat) {
  cl_image_format clImageFormat;
  switch (pImageFormat->channelOrder) {
  case UR_IMAGE_CHANNEL_ORDER_A:
    clImageFormat.image_channel_order = CL_A;
    break;
  case UR_IMAGE_CHANNEL_ORDER_R:
    clImageFormat.image_channel_order = CL_R;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RG:
    clImageFormat.image_channel_order = CL_RG;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RA:
    clImageFormat.image_channel_order = CL_RA;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGB:
    clImageFormat.image_channel_order = CL_RGB;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGBA:
    clImageFormat.image_channel_order = CL_RGBA;
    break;
  case UR_IMAGE_CHANNEL_ORDER_BGRA:
    clImageFormat.image_channel_order = CL_BGRA;
    break;
  case UR_IMAGE_CHANNEL_ORDER_ARGB:
    clImageFormat.image_channel_order = CL_ARGB;
    break;
  case UR_IMAGE_CHANNEL_ORDER_ABGR:
    clImageFormat.image_channel_order = CL_ABGR;
    break;
  case UR_IMAGE_CHANNEL_ORDER_INTENSITY:
    clImageFormat.image_channel_order = CL_INTENSITY;
    break;
  case UR_IMAGE_CHANNEL_ORDER_LUMINANCE:
    clImageFormat.image_channel_order = CL_LUMINANCE;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RX:
    clImageFormat.image_channel_order = CL_Rx;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGX:
    clImageFormat.image_channel_order = CL_RGx;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGBX:
    clImageFormat.image_channel_order = CL_RGBx;
    break;
  case UR_IMAGE_CHANNEL_ORDER_SRGBA:
    clImageFormat.image_channel_order = CL_sRGBA;
    break;
  default:
    clImageFormat.image_channel_order = -1;
    break;
  }

  switch (pImageFormat->channelType) {
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT8:
    clImageFormat.image_channel_data_type = CL_SNORM_INT8;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT16:
    clImageFormat.image_channel_data_type = CL_SNORM_INT16;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT8:
    clImageFormat.image_channel_data_type = CL_UNORM_INT8;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16:
    clImageFormat.image_channel_data_type = CL_UNORM_INT16;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565:
    clImageFormat.image_channel_data_type = CL_UNORM_SHORT_565;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555:
    clImageFormat.image_channel_data_type = CL_UNORM_SHORT_555;
    break;
  case UR_IMAGE_CHANNEL_TYPE_INT_101010:
    clImageFormat.image_channel_data_type = CL_UNORM_INT_101010;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
    clImageFormat.image_channel_data_type = CL_SIGNED_INT8;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
    clImageFormat.image_channel_data_type = CL_SIGNED_INT16;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
    clImageFormat.image_channel_data_type = CL_SIGNED_INT32;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
    clImageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
    clImageFormat.image_channel_data_type = CL_UNSIGNED_INT16;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    clImageFormat.image_channel_data_type = CL_UNSIGNED_INT32;
    break;
  case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
    clImageFormat.image_channel_data_type = CL_HALF_FLOAT;
    break;
  case UR_IMAGE_CHANNEL_TYPE_FLOAT:
    clImageFormat.image_channel_data_type = CL_FLOAT;
    break;
  default:
    clImageFormat.image_channel_data_type = -1;
    break;
  }

  return clImageFormat;
}

cl_image_desc map_ur_image_desc_to_cl(const ur_image_desc_t *pImageDesc) {
  cl_image_desc clImageDesc;
  clImageDesc.image_type =
      cl_adapter::cast<cl_mem_object_type>(pImageDesc->type);

  switch (pImageDesc->type) {
  case UR_MEM_TYPE_BUFFER:
    clImageDesc.image_type = CL_MEM_OBJECT_BUFFER;
    break;
  case UR_MEM_TYPE_IMAGE2D:
    clImageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    break;
  case UR_MEM_TYPE_IMAGE3D:
    clImageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
    break;
  case UR_MEM_TYPE_IMAGE2D_ARRAY:
    clImageDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    break;
  case UR_MEM_TYPE_IMAGE1D:
    clImageDesc.image_type = CL_MEM_OBJECT_IMAGE1D;
    break;
  case UR_MEM_TYPE_IMAGE1D_ARRAY:
    clImageDesc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
    break;
  case UR_MEM_TYPE_IMAGE1D_BUFFER:
    clImageDesc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
    break;
  default:
    clImageDesc.image_type = -1;
    break;
  }

  clImageDesc.image_width = pImageDesc->width;
  clImageDesc.image_height = pImageDesc->height;
  clImageDesc.image_depth = pImageDesc->depth;
  clImageDesc.image_array_size = pImageDesc->arraySize;
  clImageDesc.image_row_pitch = pImageDesc->rowPitch;
  clImageDesc.image_slice_pitch = pImageDesc->slicePitch;
  clImageDesc.num_mip_levels = pImageDesc->numMipLevel;
  clImageDesc.num_samples = pImageDesc->numSamples;
  clImageDesc.buffer = nullptr;
  clImageDesc.mem_object = nullptr;

  return clImageDesc;
}

cl_int map_ur_mem_image_info_to_cl(ur_image_info_t urPropName) {
  cl_int clPropName;
  switch (urPropName) {
  case UR_IMAGE_INFO_FORMAT:
    clPropName = CL_IMAGE_FORMAT;
    break;
  case UR_IMAGE_INFO_ELEMENT_SIZE:
    clPropName = CL_IMAGE_ELEMENT_SIZE;
    break;
  case UR_IMAGE_INFO_ROW_PITCH:
    clPropName = CL_IMAGE_ROW_PITCH;
    break;
  case UR_IMAGE_INFO_SLICE_PITCH:
    clPropName = CL_IMAGE_SLICE_PITCH;
    break;
  case UR_IMAGE_INFO_WIDTH:
    clPropName = CL_IMAGE_WIDTH;
    break;
  case UR_IMAGE_INFO_HEIGHT:
    clPropName = CL_IMAGE_HEIGHT;
    break;
  case UR_IMAGE_INFO_DEPTH:
    clPropName = CL_IMAGE_DEPTH;
    break;
  default:
    clPropName = -1;
  }

  return clPropName;
}

cl_int map_ur_mem_info_to_cl(ur_mem_info_t urPropName) {
  cl_int clPropName;
  switch (urPropName) {
  case UR_MEM_INFO_SIZE:
    clPropName = CL_MEM_SIZE;
    break;
  case UR_MEM_INFO_CONTEXT:
    clPropName = CL_MEM_CONTEXT;
    break;
  default:
    clPropName = -1;
  }

  return clPropName;
}

cl_map_flags convert_ur_mem_flags_to_cl(ur_mem_flags_t ur_flags) {
  cl_map_flags cl_flags = 0;
  if (ur_flags & UR_MEM_FLAG_READ_WRITE) {
    cl_flags |= CL_MEM_READ_WRITE;
  }
  if (ur_flags & UR_MEM_FLAG_WRITE_ONLY) {
    cl_flags |= CL_MEM_WRITE_ONLY;
  }
  if (ur_flags & UR_MEM_FLAG_READ_ONLY) {
    cl_flags |= CL_MEM_READ_ONLY;
  }
  if (ur_flags & UR_MEM_FLAG_USE_HOST_POINTER) {
    cl_flags |= CL_MEM_USE_HOST_PTR;
  }
  if (ur_flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
    cl_flags |= CL_MEM_ALLOC_HOST_PTR;
  }
  if (ur_flags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) {
    cl_flags |= CL_MEM_COPY_HOST_PTR;
  }

  return cl_flags;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags, size_t size,
    const ur_buffer_properties_t *pProperties, ur_mem_handle_t *phBuffer) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phBuffer, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  cl_int ret_err = CL_INVALID_OPERATION;
  if (pProperties) {
    // TODO: need to check if all properties are supported by OpenCL RT and
    // ignore unsupported
    clCreateBufferWithPropertiesINTEL_fn FuncPtr = nullptr;
    cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
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
          size, pProperties->pHost, cl_adapter::cast<cl_int *>(&ret_err)));
      CL_RETURN_ON_FAILURE(ret_err);
    }
  }

  *phBuffer = reinterpret_cast<ur_mem_handle_t>(clCreateBuffer(
      cl_adapter::cast<cl_context>(hContext), static_cast<cl_mem_flags>(flags),
      size, pProperties->pHost, cl_adapter::cast<cl_int *>(&ret_err)));
  CL_RETURN_ON_FAILURE(ret_err);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    void *pHost, ur_mem_handle_t *phMem) {
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  cl_int ret_err = CL_INVALID_OPERATION;

  cl_image_format image_format = map_ur_image_format_to_cl(pImageFormat);
  cl_image_desc image_desc = map_ur_image_desc_to_cl(pImageDesc);
  cl_map_flags map_flags = convert_ur_mem_flags_to_cl(flags);

  *phMem = reinterpret_cast<ur_mem_handle_t>(clCreateImage(
      cl_adapter::cast<cl_context>(hContext), map_flags, &image_format,
      &image_desc, pHost, cl_adapter::cast<cl_int *>(&ret_err)));
  CL_RETURN_ON_FAILURE(ret_err);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
    ur_buffer_create_type_t bufferCreateType, const ur_buffer_region_t *pRegion,
    ur_mem_handle_t *phMem) {
  UR_ASSERT(hBuffer, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  cl_int ret_err = CL_INVALID_OPERATION;

  cl_buffer_create_type buffer_create_type;
  switch (bufferCreateType) {
  case UR_BUFFER_CREATE_TYPE_REGION:
    buffer_create_type = CL_BUFFER_CREATE_TYPE_REGION;
    break;
  default:
    break;
  }

  _cl_buffer_region buffer_region;
  buffer_region.origin = pRegion->origin;
  buffer_region.size = pRegion->size;

  *phMem = reinterpret_cast<ur_mem_handle_t>(
      clCreateSubBuffer(cl_adapter::cast<cl_mem>(hBuffer),
                        static_cast<cl_mem_flags>(flags), buffer_create_type,
                        &buffer_region, cl_adapter::cast<cl_int *>(&ret_err)));
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
  UR_ASSERT(hNativeMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
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
  UR_ASSERT(hNativeMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(hContext, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  UR_ASSERT(phMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);
  *phMem = reinterpret_cast<ur_mem_handle_t>(hNativeMem);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetInfo(ur_mem_handle_t hMemory,
                                                 ur_mem_info_t propName,
                                                 size_t propSize,
                                                 void *pPropValue,
                                                 size_t *pPropSizeRet) {
  UR_ASSERT(hMemory, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  const cl_int clPropName = map_ur_mem_info_to_cl(propName);

  CL_RETURN_ON_FAILURE(clGetMemObjectInfo(cl_adapter::cast<cl_mem>(hMemory),
                                          clPropName, propSize, pPropValue,
                                          pPropSizeRet));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageGetInfo(ur_mem_handle_t hMemory,
                                                      ur_image_info_t propName,
                                                      size_t propSize,
                                                      void *pPropValue,
                                                      size_t *pPropSizeRet) {
  UR_ASSERT(hMemory, UR_RESULT_ERROR_INVALID_NULL_HANDLE);

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  const cl_int clPropName = map_ur_mem_image_info_to_cl(propName);

  CL_RETURN_ON_FAILURE(clGetImageInfo(cl_adapter::cast<cl_mem>(hMemory),
                                      clPropName, propSize, pPropValue,
                                      pPropSizeRet));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(ur_mem_handle_t hMem) {
  UR_ASSERT(hMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  CL_RETURN_ON_FAILURE(clRetainMemObject(cl_adapter::cast<cl_mem>(hMem)));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRelease(ur_mem_handle_t hMem) {
  UR_ASSERT(hMem, UR_RESULT_ERROR_INVALID_NULL_HANDLE);
  CL_RETURN_ON_FAILURE(clReleaseMemObject(cl_adapter::cast<cl_mem>(hMem)));
  return UR_RESULT_SUCCESS;
}
