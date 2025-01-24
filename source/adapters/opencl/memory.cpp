//===--------- memory.cpp - OpenCL Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "common.hpp"

#include <unordered_map>

const std::unordered_map<ur_image_channel_order_t, cl_channel_order>
    ChannelOrderMap = {
        {UR_IMAGE_CHANNEL_ORDER_A, CL_A},
        {UR_IMAGE_CHANNEL_ORDER_R, CL_R},
        {UR_IMAGE_CHANNEL_ORDER_RG, CL_RG},
        {UR_IMAGE_CHANNEL_ORDER_RA, CL_RA},
        {UR_IMAGE_CHANNEL_ORDER_RGB, CL_RGB},
        {UR_IMAGE_CHANNEL_ORDER_RGBA, CL_RGBA},
        {UR_IMAGE_CHANNEL_ORDER_BGRA, CL_BGRA},
        {UR_IMAGE_CHANNEL_ORDER_ARGB, CL_ARGB},
        {UR_IMAGE_CHANNEL_ORDER_ABGR, CL_ABGR},
        {UR_IMAGE_CHANNEL_ORDER_INTENSITY, CL_INTENSITY},
        {UR_IMAGE_CHANNEL_ORDER_LUMINANCE, CL_LUMINANCE},
        {UR_IMAGE_CHANNEL_ORDER_RX, CL_Rx},
        {UR_IMAGE_CHANNEL_ORDER_RGX, CL_RGx},
        {UR_IMAGE_CHANNEL_ORDER_RGBX, CL_RGBx},
        {UR_IMAGE_CHANNEL_ORDER_SRGBA, CL_sRGBA},
};

const std::unordered_map<ur_image_channel_type_t, cl_channel_type>
    ChannelTypeMap = {
        {UR_IMAGE_CHANNEL_TYPE_SNORM_INT8, CL_SNORM_INT8},
        {UR_IMAGE_CHANNEL_TYPE_SNORM_INT16, CL_SNORM_INT16},
        {UR_IMAGE_CHANNEL_TYPE_UNORM_INT8, CL_UNORM_INT8},
        {UR_IMAGE_CHANNEL_TYPE_UNORM_INT16, CL_UNORM_INT16},
        {UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565, CL_UNORM_SHORT_565},
        {UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555, CL_UNORM_SHORT_555},
        {UR_IMAGE_CHANNEL_TYPE_INT_101010, CL_UNORM_INT_101010},
        {UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8, CL_SIGNED_INT8},
        {UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16, CL_SIGNED_INT16},
        {UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32, CL_SIGNED_INT32},
        {UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8, CL_UNSIGNED_INT8},
        {UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16, CL_UNSIGNED_INT16},
        {UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32, CL_UNSIGNED_INT32},
        {UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT, CL_HALF_FLOAT},
        {UR_IMAGE_CHANNEL_TYPE_FLOAT, CL_FLOAT},
};

cl_image_format mapURImageFormatToCL(const ur_image_format_t &PImageFormat) {
  cl_image_format CLImageFormat = {UR_IMAGE_CHANNEL_ORDER_FORCE_UINT32,
                                   UR_IMAGE_CHANNEL_TYPE_FORCE_UINT32};

  auto channelOrderIt = ChannelOrderMap.find(PImageFormat.channelOrder);
  if (channelOrderIt != ChannelOrderMap.end()) {
    CLImageFormat.image_channel_order = channelOrderIt->second;
  }

  auto channelTypeIt = ChannelTypeMap.find(PImageFormat.channelType);
  if (channelTypeIt != ChannelTypeMap.end()) {
    CLImageFormat.image_channel_data_type = channelTypeIt->second;
  }

  return CLImageFormat;
}

ur_image_format_t mapCLImageFormatToUR(const cl_image_format *PImageFormat) {
  ur_image_format_t URImageFormat = {UR_IMAGE_CHANNEL_ORDER_FORCE_UINT32,
                                     UR_IMAGE_CHANNEL_TYPE_FORCE_UINT32};

  auto reverseChannelOrderIt =
      std::find_if(ChannelOrderMap.begin(), ChannelOrderMap.end(),
                   [PImageFormat](const auto &pair) {
                     return pair.second == PImageFormat->image_channel_order;
                   });
  if (reverseChannelOrderIt != ChannelOrderMap.end()) {
    URImageFormat.channelOrder = reverseChannelOrderIt->first;
  }

  URImageFormat.channelOrder = (reverseChannelOrderIt != ChannelOrderMap.end())
                                   ? reverseChannelOrderIt->first
                                   : UR_IMAGE_CHANNEL_ORDER_FORCE_UINT32;

  auto reverseChannelTypeIt = std::find_if(
      ChannelTypeMap.begin(), ChannelTypeMap.end(),
      [PImageFormat](const auto &pair) {
        return pair.second == PImageFormat->image_channel_data_type;
      });
  if (reverseChannelTypeIt != ChannelTypeMap.end()) {
    URImageFormat.channelType = reverseChannelTypeIt->first;
  }

  return URImageFormat;
}

cl_image_format mapURImageFormatToCL(const ur_image_format_t *PImageFormat) {
  cl_image_format CLImageFormat;
  switch (PImageFormat->channelOrder) {
  case UR_IMAGE_CHANNEL_ORDER_A:
    CLImageFormat.image_channel_order = CL_A;
    break;
  case UR_IMAGE_CHANNEL_ORDER_R:
    CLImageFormat.image_channel_order = CL_R;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RG:
    CLImageFormat.image_channel_order = CL_RG;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RA:
    CLImageFormat.image_channel_order = CL_RA;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGB:
    CLImageFormat.image_channel_order = CL_RGB;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGBA:
    CLImageFormat.image_channel_order = CL_RGBA;
    break;
  case UR_IMAGE_CHANNEL_ORDER_BGRA:
    CLImageFormat.image_channel_order = CL_BGRA;
    break;
  case UR_IMAGE_CHANNEL_ORDER_ARGB:
    CLImageFormat.image_channel_order = CL_ARGB;
    break;
  case UR_IMAGE_CHANNEL_ORDER_ABGR:
    CLImageFormat.image_channel_order = CL_ABGR;
    break;
  case UR_IMAGE_CHANNEL_ORDER_INTENSITY:
    CLImageFormat.image_channel_order = CL_INTENSITY;
    break;
  case UR_IMAGE_CHANNEL_ORDER_LUMINANCE:
    CLImageFormat.image_channel_order = CL_LUMINANCE;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RX:
    CLImageFormat.image_channel_order = CL_Rx;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGX:
    CLImageFormat.image_channel_order = CL_RGx;
    break;
  case UR_IMAGE_CHANNEL_ORDER_RGBX:
    CLImageFormat.image_channel_order = CL_RGBx;
    break;
  case UR_IMAGE_CHANNEL_ORDER_SRGBA:
    CLImageFormat.image_channel_order = CL_sRGBA;
    break;
  default:
    CLImageFormat.image_channel_order = -1;
    break;
  }

  switch (PImageFormat->channelType) {
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT8:
    CLImageFormat.image_channel_data_type = CL_SNORM_INT8;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SNORM_INT16:
    CLImageFormat.image_channel_data_type = CL_SNORM_INT16;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT8:
    CLImageFormat.image_channel_data_type = CL_UNORM_INT8;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_INT16:
    CLImageFormat.image_channel_data_type = CL_UNORM_INT16;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565:
    CLImageFormat.image_channel_data_type = CL_UNORM_SHORT_565;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555:
    CLImageFormat.image_channel_data_type = CL_UNORM_SHORT_555;
    break;
  case UR_IMAGE_CHANNEL_TYPE_INT_101010:
    CLImageFormat.image_channel_data_type = CL_UNORM_INT_101010;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8:
    CLImageFormat.image_channel_data_type = CL_SIGNED_INT8;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16:
    CLImageFormat.image_channel_data_type = CL_SIGNED_INT16;
    break;
  case UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32:
    CLImageFormat.image_channel_data_type = CL_SIGNED_INT32;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8:
    CLImageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16:
    CLImageFormat.image_channel_data_type = CL_UNSIGNED_INT16;
    break;
  case UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32:
    CLImageFormat.image_channel_data_type = CL_UNSIGNED_INT32;
    break;
  case UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT:
    CLImageFormat.image_channel_data_type = CL_HALF_FLOAT;
    break;
  case UR_IMAGE_CHANNEL_TYPE_FLOAT:
    CLImageFormat.image_channel_data_type = CL_FLOAT;
    break;
  default:
    CLImageFormat.image_channel_data_type = -1;
    break;
  }

  return CLImageFormat;
}

cl_image_desc mapURImageDescToCL(const ur_image_desc_t *PImageDesc) {
  cl_image_desc CLImageDesc;
  CLImageDesc.image_type =
      cl_adapter::cast<cl_mem_object_type>(PImageDesc->type);

  switch (PImageDesc->type) {
  case UR_MEM_TYPE_IMAGE2D:
    CLImageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    break;
  case UR_MEM_TYPE_IMAGE3D:
    CLImageDesc.image_type = CL_MEM_OBJECT_IMAGE3D;
    break;
  case UR_MEM_TYPE_IMAGE2D_ARRAY:
    CLImageDesc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    break;
  case UR_MEM_TYPE_IMAGE1D:
    CLImageDesc.image_type = CL_MEM_OBJECT_IMAGE1D;
    break;
  case UR_MEM_TYPE_IMAGE1D_ARRAY:
    CLImageDesc.image_type = CL_MEM_OBJECT_IMAGE1D_ARRAY;
    break;
  default:
    CLImageDesc.image_type = -1;
    break;
  }

  CLImageDesc.image_width = PImageDesc->width;
  CLImageDesc.image_height = PImageDesc->height;
  CLImageDesc.image_depth = PImageDesc->depth;
  CLImageDesc.image_array_size = PImageDesc->arraySize;
  CLImageDesc.image_row_pitch = PImageDesc->rowPitch;
  CLImageDesc.image_slice_pitch = PImageDesc->slicePitch;
  CLImageDesc.num_mip_levels = PImageDesc->numMipLevel;
  CLImageDesc.num_samples = PImageDesc->numSamples;
  CLImageDesc.buffer = nullptr;
  CLImageDesc.mem_object = nullptr;

  return CLImageDesc;
}

cl_int mapURMemImageInfoToCL(ur_image_info_t URPropName) {

  switch (URPropName) {
  case UR_IMAGE_INFO_FORMAT:
    return CL_IMAGE_FORMAT;
  case UR_IMAGE_INFO_ELEMENT_SIZE:
    return CL_IMAGE_ELEMENT_SIZE;
  case UR_IMAGE_INFO_ROW_PITCH:
    return CL_IMAGE_ROW_PITCH;
  case UR_IMAGE_INFO_SLICE_PITCH:
    return CL_IMAGE_SLICE_PITCH;
  case UR_IMAGE_INFO_WIDTH:
    return CL_IMAGE_WIDTH;
  case UR_IMAGE_INFO_HEIGHT:
    return CL_IMAGE_HEIGHT;
  case UR_IMAGE_INFO_DEPTH:
    return CL_IMAGE_DEPTH;
  case UR_IMAGE_INFO_ARRAY_SIZE:
    return CL_IMAGE_ARRAY_SIZE;
  case UR_IMAGE_INFO_NUM_MIP_LEVELS:
    return CL_IMAGE_NUM_MIP_LEVELS;
  case UR_IMAGE_INFO_NUM_SAMPLES:
    return CL_IMAGE_NUM_SAMPLES;
  default:
    return -1;
  }
}

cl_int mapURMemInfoToCL(ur_mem_info_t URPropName) {

  switch (URPropName) {
  case UR_MEM_INFO_SIZE:
    return CL_MEM_SIZE;
  case UR_MEM_INFO_CONTEXT:
    return CL_MEM_CONTEXT;
  case UR_MEM_INFO_REFERENCE_COUNT:
    return CL_MEM_REFERENCE_COUNT;
  default:
    return -1;
  }
}

cl_map_flags convertURMemFlagsToCL(ur_mem_flags_t URFlags) {
  cl_map_flags CLFlags = 0;
  if (URFlags & UR_MEM_FLAG_READ_WRITE) {
    CLFlags |= CL_MEM_READ_WRITE;
  }
  if (URFlags & UR_MEM_FLAG_WRITE_ONLY) {
    CLFlags |= CL_MEM_WRITE_ONLY;
  }
  if (URFlags & UR_MEM_FLAG_READ_ONLY) {
    CLFlags |= CL_MEM_READ_ONLY;
  }
  if (URFlags & UR_MEM_FLAG_USE_HOST_POINTER) {
    CLFlags |= CL_MEM_USE_HOST_PTR;
  }
  if (URFlags & UR_MEM_FLAG_ALLOC_HOST_POINTER) {
    CLFlags |= CL_MEM_ALLOC_HOST_PTR;
  }
  if (URFlags & UR_MEM_FLAG_ALLOC_COPY_HOST_POINTER) {
    CLFlags |= CL_MEM_COPY_HOST_PTR;
  }

  return CLFlags;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags, size_t size,
    const ur_buffer_properties_t *pProperties, ur_mem_handle_t *phBuffer) {

  cl_int RetErr = CL_INVALID_OPERATION;
  if (pProperties) {
    // TODO: need to check if all properties are supported by OpenCL RT and
    // ignore unsupported
    clCreateBufferWithPropertiesINTEL_fn FuncPtr = nullptr;
    cl_context CLContext = cl_adapter::cast<cl_context>(hContext);
    // First we need to look up the function pointer
    RetErr =
        cl_ext::getExtFuncFromContext<clCreateBufferWithPropertiesINTEL_fn>(
            CLContext,
            cl_ext::ExtFuncPtrCache->clCreateBufferWithPropertiesINTELCache,
            cl_ext::CreateBufferWithPropertiesName, &FuncPtr);
    if (FuncPtr) {
      std::vector<cl_mem_properties_intel> PropertiesIntel;
      auto Prop = static_cast<ur_base_properties_t *>(pProperties->pNext);
      while (Prop) {
        switch (Prop->stype) {
        case UR_STRUCTURE_TYPE_BUFFER_CHANNEL_PROPERTIES: {
          auto BufferChannelProperty =
              reinterpret_cast<ur_buffer_channel_properties_t *>(Prop);
          PropertiesIntel.push_back(CL_MEM_CHANNEL_INTEL);
          PropertiesIntel.push_back(BufferChannelProperty->channel);
        } break;
        case UR_STRUCTURE_TYPE_BUFFER_ALLOC_LOCATION_PROPERTIES: {
          auto BufferLocationProperty =
              reinterpret_cast<ur_buffer_alloc_location_properties_t *>(Prop);
          PropertiesIntel.push_back(CL_MEM_ALLOC_BUFFER_LOCATION_INTEL);
          PropertiesIntel.push_back(BufferLocationProperty->location);
        } break;
        default:
          break;
        }
        Prop = static_cast<ur_base_properties_t *>(Prop->pNext);
      }
      PropertiesIntel.push_back(0);

      *phBuffer = reinterpret_cast<ur_mem_handle_t>(FuncPtr(
          CLContext, PropertiesIntel.data(), static_cast<cl_mem_flags>(flags),
          size, pProperties->pHost, cl_adapter::cast<cl_int *>(&RetErr)));
      return mapCLErrorToUR(RetErr);
    }
  }

  void *HostPtr = pProperties ? pProperties->pHost : nullptr;
  *phBuffer = reinterpret_cast<ur_mem_handle_t>(clCreateBuffer(
      cl_adapter::cast<cl_context>(hContext), static_cast<cl_mem_flags>(flags),
      size, HostPtr, cl_adapter::cast<cl_int *>(&RetErr)));
  CL_RETURN_ON_FAILURE(RetErr);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreate(
    ur_context_handle_t hContext, ur_mem_flags_t flags,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    void *pHost, ur_mem_handle_t *phMem) {

  cl_int RetErr = CL_INVALID_OPERATION;

  cl_image_format ImageFormat = mapURImageFormatToCL(pImageFormat);
  cl_image_desc ImageDesc = mapURImageDescToCL(pImageDesc);
  cl_map_flags MapFlags = convertURMemFlagsToCL(flags);

  *phMem = reinterpret_cast<ur_mem_handle_t>(clCreateImage(
      cl_adapter::cast<cl_context>(hContext), MapFlags, &ImageFormat,
      &ImageDesc, pHost, cl_adapter::cast<cl_int *>(&RetErr)));
  CL_RETURN_ON_FAILURE(RetErr);

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferPartition(
    ur_mem_handle_t hBuffer, ur_mem_flags_t flags,
    ur_buffer_create_type_t bufferCreateType, const ur_buffer_region_t *pRegion,
    ur_mem_handle_t *phMem) {

  cl_int RetErr = CL_INVALID_OPERATION;

  cl_buffer_create_type BufferCreateType;
  switch (bufferCreateType) {
  case UR_BUFFER_CREATE_TYPE_REGION:
    BufferCreateType = CL_BUFFER_CREATE_TYPE_REGION;
    break;
  default:
    return UR_RESULT_ERROR_INVALID_ENUMERATION;
  }

  _cl_buffer_region BufferRegion;
  BufferRegion.origin = pRegion->origin;
  BufferRegion.size = pRegion->size;

  *phMem = reinterpret_cast<ur_mem_handle_t>(clCreateSubBuffer(
      cl_adapter::cast<cl_mem>(hBuffer), static_cast<cl_mem_flags>(flags),
      BufferCreateType, &BufferRegion, cl_adapter::cast<cl_int *>(&RetErr)));

  if (RetErr == CL_INVALID_VALUE) {
    size_t BufferSize = 0;
    CL_RETURN_ON_FAILURE(clGetMemObjectInfo(cl_adapter::cast<cl_mem>(hBuffer),
                                            CL_MEM_SIZE, sizeof(BufferSize),
                                            &BufferSize, nullptr));
    if (BufferRegion.size + BufferRegion.origin > BufferSize)
      return UR_RESULT_ERROR_INVALID_BUFFER_SIZE;
  }
  return mapCLErrorToUR(RetErr);
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetNativeHandle(
    ur_mem_handle_t hMem, ur_device_handle_t, ur_native_handle_t *phNativeMem) {
  return getNativeHandle(hMem, phNativeMem);
}

UR_APIEXPORT ur_result_t UR_APICALL urMemBufferCreateWithNativeHandle(
    ur_native_handle_t hNativeMem,
    [[maybe_unused]] ur_context_handle_t hContext,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem) {
  *phMem = reinterpret_cast<ur_mem_handle_t>(hNativeMem);
  if (!pProperties || !pProperties->isNativeHandleOwned) {
    return urMemRetain(*phMem);
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageCreateWithNativeHandle(
    ur_native_handle_t hNativeMem,
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    const ur_mem_native_properties_t *pProperties, ur_mem_handle_t *phMem) {
  *phMem = reinterpret_cast<ur_mem_handle_t>(hNativeMem);
  if (!pProperties || !pProperties->isNativeHandleOwned) {
    return urMemRetain(*phMem);
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemGetInfo(ur_mem_handle_t hMemory,
                                                 ur_mem_info_t propName,
                                                 size_t propSize,
                                                 void *pPropValue,
                                                 size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  const cl_int CLPropName = mapURMemInfoToCL(propName);

  size_t CheckPropSize = 0;
  auto ClResult =
      clGetMemObjectInfo(cl_adapter::cast<cl_mem>(hMemory), CLPropName,
                         propSize, pPropValue, &CheckPropSize);
  if (pPropValue && CheckPropSize != propSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }
  CL_RETURN_ON_FAILURE(ClResult);
  if (pPropSizeRet) {
    *pPropSizeRet = CheckPropSize;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemImageGetInfo(ur_mem_handle_t hMemory,
                                                      ur_image_info_t propName,
                                                      size_t propSize,
                                                      void *pPropValue,
                                                      size_t *pPropSizeRet) {

  UrReturnHelper ReturnValue(propSize, pPropValue, pPropSizeRet);
  const cl_int CLPropName = mapURMemImageInfoToCL(propName);

  size_t CheckPropSize = 0;
  auto ClResult = clGetImageInfo(cl_adapter::cast<cl_mem>(hMemory), CLPropName,
                                 propSize, pPropValue, &CheckPropSize);
  if (pPropValue && CheckPropSize != propSize) {
    return UR_RESULT_ERROR_INVALID_SIZE;
  }
  CL_RETURN_ON_FAILURE(ClResult);
  if (pPropSizeRet) {
    *pPropSizeRet = CheckPropSize;
  } else {
    if (propName == UR_IMAGE_INFO_FORMAT) {
      ur_image_format_t format = mapCLImageFormatToUR(
          reinterpret_cast<const cl_image_format *>(pPropValue));
      return ReturnValue(format);
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRetain(ur_mem_handle_t hMem) {
  CL_RETURN_ON_FAILURE(clRetainMemObject(cl_adapter::cast<cl_mem>(hMem)));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urMemRelease(ur_mem_handle_t hMem) {
  CL_RETURN_ON_FAILURE(clReleaseMemObject(cl_adapter::cast<cl_mem>(hMem)));
  return UR_RESULT_SUCCESS;
}
