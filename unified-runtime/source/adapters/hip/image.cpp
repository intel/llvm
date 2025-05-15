//===--------- image.cpp - HIP Adapter -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>

#include "common.hpp"
#include "context.hpp"
#include "enqueue.hpp"
#include "event.hpp"
#include "image.hpp"
#include "logger/ur_logger.hpp"
#include "queue.hpp"
#include "sampler.hpp"
#include "ur/ur.hpp"
#include "ur_api.h"

ur_result_t urCalculateNumChannels(ur_image_channel_order_t order,
                                   unsigned int *NumChannels) {
  switch (order) {
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_A:
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_R:
    *NumChannels = 1;
    return UR_RESULT_SUCCESS;
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_RG:
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_RA:
    *NumChannels = 2;
    return UR_RESULT_SUCCESS;
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_RGB:
    return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_RGBA:
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_ARGB:
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_BGRA:
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_ABGR:
    *NumChannels = 4;
    return UR_RESULT_SUCCESS;
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_RX:
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_RGX:
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_RGBX:
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_SRGBA:
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_INTENSITY:
  case ur_image_channel_order_t::UR_IMAGE_CHANNEL_ORDER_LUMINANCE:
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
  }
}

/// Convert a UR image format to a HIP image format and
/// get the pixel size in bytes.
/// /param image_channel_type is the ur_image_channel_type_t.
/// /param image_channel_order is the ur_image_channel_order_t.
///        this is used for normalized channel formats, as HIP
///        combines the channel format and order for normalized
///        channel types.
/// /param return_hip_format will be set to the equivalent HIP
///        format if not nullptr.
/// /param return_pixel_size_bytes will be set to the pixel
///        byte size if not nullptr.
/// /param return_normalized_dtype_flag will be set if the
///        data type is normalized if not nullptr.
ur_result_t
urToHipImageChannelFormat(ur_image_channel_type_t image_channel_type,
                          ur_image_channel_order_t image_channel_order,
                          hipArray_Format *return_hip_format,
                          size_t *return_pixel_size_bytes,
                          unsigned int *return_normalized_dtype_flag) {

  hipArray_Format hip_format = HIP_AD_FORMAT_UNSIGNED_INT8;
  size_t pixel_size_bytes = 0;
  unsigned int num_channels = 0;
  unsigned int normalized_dtype_flag = 0;
  UR_CALL(urCalculateNumChannels(image_channel_order, &num_channels));

  switch (image_channel_type) {
#define CASE(FROM, TO, SIZE, NORM)                                             \
  case FROM: {                                                                 \
    hip_format = TO;                                                           \
    pixel_size_bytes = SIZE * num_channels;                                    \
    normalized_dtype_flag = NORM;                                              \
    break;                                                                     \
  }

    CASE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8, HIP_AD_FORMAT_UNSIGNED_INT8, 1, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8, HIP_AD_FORMAT_SIGNED_INT8, 1, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16, HIP_AD_FORMAT_UNSIGNED_INT16, 2,
         0)
    CASE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16, HIP_AD_FORMAT_SIGNED_INT16, 2, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT, HIP_AD_FORMAT_HALF, 2, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32, HIP_AD_FORMAT_UNSIGNED_INT32, 4,
         0)
    CASE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32, HIP_AD_FORMAT_SIGNED_INT32, 4, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_FLOAT, HIP_AD_FORMAT_FLOAT, 4, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_UNORM_INT8, HIP_AD_FORMAT_UNSIGNED_INT8, 1, 1)
    CASE(UR_IMAGE_CHANNEL_TYPE_SNORM_INT8, HIP_AD_FORMAT_SIGNED_INT8, 1, 1)
    CASE(UR_IMAGE_CHANNEL_TYPE_UNORM_INT16, HIP_AD_FORMAT_UNSIGNED_INT16, 2, 1)
    CASE(UR_IMAGE_CHANNEL_TYPE_SNORM_INT16, HIP_AD_FORMAT_SIGNED_INT16, 2, 1)

#undef CASE
  default:
    break;
  }

  if (return_hip_format) {
    *return_hip_format = hip_format;
  }
  if (return_pixel_size_bytes) {
    *return_pixel_size_bytes = pixel_size_bytes;
  }
  if (return_normalized_dtype_flag) {
    *return_normalized_dtype_flag = normalized_dtype_flag;
  }
  return UR_RESULT_SUCCESS;
}

ur_result_t
hipToUrImageChannelFormat(hipArray_Format hip_format,
                          ur_image_channel_type_t *return_image_channel_type) {

  switch (hip_format) {
#define HIP_TO_UR_IMAGE_CHANNEL_TYPE(FROM, TO)                                 \
  case FROM: {                                                                 \
    *return_image_channel_type = TO;                                           \
    return UR_RESULT_SUCCESS;                                                  \
  }

    HIP_TO_UR_IMAGE_CHANNEL_TYPE(HIP_AD_FORMAT_UNSIGNED_INT8,
                                 UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8);
    HIP_TO_UR_IMAGE_CHANNEL_TYPE(HIP_AD_FORMAT_UNSIGNED_INT16,
                                 UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16);
    HIP_TO_UR_IMAGE_CHANNEL_TYPE(HIP_AD_FORMAT_UNSIGNED_INT32,
                                 UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32);
    HIP_TO_UR_IMAGE_CHANNEL_TYPE(HIP_AD_FORMAT_SIGNED_INT8,
                                 UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8);
    HIP_TO_UR_IMAGE_CHANNEL_TYPE(HIP_AD_FORMAT_SIGNED_INT16,
                                 UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16);
    HIP_TO_UR_IMAGE_CHANNEL_TYPE(HIP_AD_FORMAT_SIGNED_INT32,
                                 UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32);
    HIP_TO_UR_IMAGE_CHANNEL_TYPE(HIP_AD_FORMAT_HALF,
                                 UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT);
    HIP_TO_UR_IMAGE_CHANNEL_TYPE(HIP_AD_FORMAT_FLOAT,
                                 UR_IMAGE_CHANNEL_TYPE_FLOAT);

#undef HIP_TO_UR_IMAGE_CHANNEL_TYPE
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
  }
}

ur_result_t urTextureCreate(ur_sampler_handle_t hSampler,
                            const ur_image_desc_t *pImageDesc,
                            const HIP_RESOURCE_DESC &ResourceDesc,
                            const unsigned int normalized_dtype_flag,
                            ur_exp_image_native_handle_t *phRetImage) {

  try {
    /// Layout of UR samplers for HIP
    ///
    /// Sampler property layout:
    /// |     <bits>     | <usage>
    /// -----------------------------------
    /// |  31 30 ... 13  | N/A
    /// |       12       | cubemap filter mode
    /// |       11       | mip filter mode
    /// |    10 9 8      | addressing mode 3
    /// |     7 6 5      | addressing mode 2
    /// |     4 3 2      | addressing mode 1
    /// |       1        | filter mode
    /// |       0        | normalize coords
    HIP_TEXTURE_DESC ImageTexDesc = {};
    HIPaddress_mode AddrMode[3] = {};
    for (size_t i = 0; i < 3; i++) {
      ur_sampler_addressing_mode_t AddrModeProp =
          hSampler->getAddressingModeDim(i);
      if (AddrModeProp == (UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE -
                           UR_SAMPLER_ADDRESSING_MODE_NONE)) {
        AddrMode[i] = HIP_TR_ADDRESS_MODE_CLAMP;
      } else if (AddrModeProp == (UR_SAMPLER_ADDRESSING_MODE_CLAMP -
                                  UR_SAMPLER_ADDRESSING_MODE_NONE)) {
        AddrMode[i] = HIP_TR_ADDRESS_MODE_BORDER;
      } else if (AddrModeProp == (UR_SAMPLER_ADDRESSING_MODE_REPEAT -
                                  UR_SAMPLER_ADDRESSING_MODE_NONE)) {
        AddrMode[i] = HIP_TR_ADDRESS_MODE_WRAP;
      } else if (AddrModeProp == (UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT -
                                  UR_SAMPLER_ADDRESSING_MODE_NONE)) {
        AddrMode[i] = HIP_TR_ADDRESS_MODE_MIRROR;
      }
    }

    HIPfilter_mode FilterMode;
    ur_sampler_filter_mode_t FilterModeProp = hSampler->getFilterMode();
    FilterMode =
        FilterModeProp ? HIP_TR_FILTER_MODE_LINEAR : HIP_TR_FILTER_MODE_POINT;
    ImageTexDesc.filterMode = FilterMode;

    // Mipmap attributes
    HIPfilter_mode MipFilterMode;
    ur_sampler_filter_mode_t MipFilterModeProp = hSampler->getMipFilterMode();
    MipFilterMode = MipFilterModeProp ? HIP_TR_FILTER_MODE_LINEAR
                                      : HIP_TR_FILTER_MODE_POINT;
    ImageTexDesc.mipmapFilterMode = MipFilterMode;
    ImageTexDesc.maxMipmapLevelClamp = hSampler->MaxMipmapLevelClamp;
    ImageTexDesc.minMipmapLevelClamp = hSampler->MinMipmapLevelClamp;
    ImageTexDesc.maxAnisotropy =
        static_cast<unsigned int>(hSampler->MaxAnisotropy);

    // The address modes can interfere with other dimensions
    // e.g. 1D texture sampling can be interfered with when setting other
    // dimension address modes despite their nonexistence.
    ImageTexDesc.addressMode[0] = AddrMode[0]; // 1D
    ImageTexDesc.addressMode[1] = pImageDesc->height > 0
                                      ? AddrMode[1]
                                      : ImageTexDesc.addressMode[1]; // 2D
    ImageTexDesc.addressMode[2] =
        pImageDesc->depth > 0 ? AddrMode[2] : ImageTexDesc.addressMode[2]; // 3D

    // flags takes the normalized coordinates setting -- unnormalized is default
    ImageTexDesc.flags = (hSampler->isNormalizedCoords())
                             ? HIP_TRSF_NORMALIZED_COORDINATES
                             : ImageTexDesc.flags;

    // HIP default promotes 8-bit and 16-bit integers to float between [0,1]
    // This flag prevents this behaviour.
    if (!normalized_dtype_flag) {
      ImageTexDesc.flags |= HIP_TRSF_READ_AS_INTEGER;
    }
    // Cubemap attributes
    ur_exp_sampler_cubemap_filter_mode_t CubemapFilterModeProp =
        hSampler->getCubemapFilterMode();
    if (CubemapFilterModeProp == UR_EXP_SAMPLER_CUBEMAP_FILTER_MODE_SEAMLESS) {
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
    }

    hipTextureObject_t Texture;
    UR_CHECK_ERROR(
        hipTexObjectCreate(&Texture, &ResourceDesc, &ImageTexDesc, nullptr));
    *phRetImage = reinterpret_cast<ur_exp_image_native_handle_t>(Texture);
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPitchedAllocExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_usm_desc_t * /*pUSMDesc*/, ur_usm_pool_handle_t /*pool*/,
    size_t widthInBytes, size_t height, size_t elementSizeBytes, void **ppMem,
    size_t *pResultPitch) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  UR_ASSERT((height > 0), UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT((elementSizeBytes > 0), UR_RESULT_ERROR_INVALID_VALUE);

  // elementSizeBytes can only take on values of 4, 8, or 16.
  // small data types need to be minimised to 4.
  if (elementSizeBytes < 4) {
    elementSizeBytes = 4;
  }
  UR_ASSERT((elementSizeBytes == 4 || elementSizeBytes == 8 ||
             elementSizeBytes == 16),
            UR_RESULT_ERROR_INVALID_VALUE);
  ur_result_t Result = UR_RESULT_SUCCESS;
  try {
    ScopedDevice Active(hDevice);
    UR_CHECK_ERROR(hipMemAllocPitch(static_cast<hipDeviceptr_t *>(ppMem),
                                    pResultPitch, widthInBytes, height,
                                    elementSizeBytes));
  } catch (ur_result_t error) {
    Result = error;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesUnsampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_native_handle_t hImage) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    UR_CHECK_ERROR(
        hipDestroySurfaceObject(reinterpret_cast<hipSurfaceObject_t>(hImage)));
  } catch (ur_result_t error) {
    return error;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesSampledImageHandleDestroyExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_native_handle_t hImage) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);
  try {
    UR_CHECK_ERROR(
        hipTexObjectDestroy(reinterpret_cast<hipTextureObject_t>(hImage)));
  } catch (ur_result_t error) {
    return error;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageAllocateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  // Populate descriptor
  HIP_ARRAY3D_DESCRIPTOR array_desc = {};

  UR_CALL(urCalculateNumChannels(pImageFormat->channelOrder,
                                 &array_desc.NumChannels));

  UR_CALL(urToHipImageChannelFormat(pImageFormat->channelType,
                                    pImageFormat->channelOrder,
                                    &array_desc.Format, nullptr, nullptr));

  array_desc.Flags = 0; // No flags required
  array_desc.Width = pImageDesc->width;
  switch (pImageDesc->type) {
  case UR_MEM_TYPE_IMAGE1D:
    array_desc.Height = 0;
    array_desc.Depth = 0;
    break;
  case UR_MEM_TYPE_IMAGE2D:
    array_desc.Height = pImageDesc->height;
    array_desc.Depth = 0;
    break;
  case UR_MEM_TYPE_IMAGE3D:
    array_desc.Height = pImageDesc->height;
    array_desc.Depth = pImageDesc->depth;
    break;
  case UR_MEM_TYPE_IMAGE1D_ARRAY:
    array_desc.Height = 0;
    array_desc.Depth = pImageDesc->arraySize;
    array_desc.Flags |= hipArrayLayered;
    break;
  case UR_MEM_TYPE_IMAGE2D_ARRAY:
    array_desc.Height = pImageDesc->height;
    array_desc.Depth = pImageDesc->arraySize;
    array_desc.Flags |= hipArrayLayered;
    break;
  case UR_MEM_TYPE_IMAGE_CUBEMAP_EXP:
    array_desc.Height = pImageDesc->height;
    array_desc.Depth = pImageDesc->arraySize; // Should be 6 ONLY
    array_desc.Flags |= hipArrayCubemap;
    break;
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ScopedDevice Active(hDevice);

  // Allocate a hipArray
  if (pImageDesc->numMipLevel == 1) {
    hipArray_t ImageArray{nullptr};

    try {
      UR_CHECK_ERROR(hipArray3DCreate(&ImageArray, &array_desc));
      *phImageMem =
          reinterpret_cast<ur_exp_image_mem_native_handle_t>(ImageArray);
    } catch (ur_result_t Err) {
      if (ImageArray) {
        (void)hipArrayDestroy(ImageArray);
      }
      return Err;
    } catch (...) {
      if (ImageArray) {
        (void)hipArrayDestroy(ImageArray);
      }
      return UR_RESULT_ERROR_UNKNOWN;
    }
  } else {
    // Allocate a hipMipmappedArray
    hipMipmappedArray_t mip_array{nullptr};
    array_desc.Flags = hipArraySurfaceLoadStore;

    try {
      UR_CHECK_ERROR(hipMipmappedArrayCreate(&mip_array, &array_desc,
                                             pImageDesc->numMipLevel));
      *phImageMem =
          reinterpret_cast<ur_exp_image_mem_native_handle_t>(mip_array);
    } catch (ur_result_t Err) {
      if (mip_array) {
        (void)hipMipmappedArrayDestroy(mip_array);
      }
      return Err;
    } catch (...) {
      if (mip_array) {
        (void)hipMipmappedArrayDestroy(mip_array);
      }
      return UR_RESULT_ERROR_UNKNOWN;
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  ScopedDevice Active(hDevice);
  try {
    hipArray_t ImageArray = reinterpret_cast<hipArray_t>(hImageMem);
    UR_CHECK_ERROR(hipArrayDestroy(ImageArray));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesUnsampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat,
    [[maybe_unused]] const ur_image_desc_t *pImageDesc,
    ur_exp_image_native_handle_t *phImage) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  unsigned int NumChannels = 0;
  UR_CALL(urCalculateNumChannels(pImageFormat->channelOrder, &NumChannels));

  hipArray_Format format;
  size_t PixelSizeBytes;
  UR_CALL(urToHipImageChannelFormat(pImageFormat->channelType,
                                    pImageFormat->channelOrder, &format,
                                    &PixelSizeBytes, nullptr));

  try {

    ScopedDevice Active(hDevice);

    hipResourceDesc image_res_desc = {};

    // We have a hipArray_t
    image_res_desc.resType = hipResourceTypeArray;
    image_res_desc.res.array.array = reinterpret_cast<hipArray_t>(hImageMem);

    // We create surfaces in the unsampled images case as it conforms to how
    // HIP deals with unsampled images.
    hipSurfaceObject_t surface;
    UR_CHECK_ERROR(hipCreateSurfaceObject(&surface, &image_res_desc));
    *phImage = reinterpret_cast<ur_exp_image_native_handle_t>(surface);

  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_sampler_handle_t hSampler, ur_exp_image_native_handle_t *phImage) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  ScopedDevice Active(hDevice);

  unsigned int NumChannels = 0;
  UR_CALL(urCalculateNumChannels(pImageFormat->channelOrder, &NumChannels));

  hipArray_Format format;
  size_t PixelSizeBytes;
  unsigned int normalized_dtype_flag;
  UR_CALL(urToHipImageChannelFormat(pImageFormat->channelType,
                                    pImageFormat->channelOrder, &format,
                                    &PixelSizeBytes, &normalized_dtype_flag));

  try {
    HIP_RESOURCE_DESC image_res_desc = {};

    unsigned int memType{};
    // hipPointerGetAttribute can detect HIP-registered memory of types:
    // hipMemoryTypeHost, hipMemoryTypeDevice, or hipMemoryTypeArray.
    UR_CHECK_ERROR(
        hipPointerGetAttribute(&memType, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                               reinterpret_cast<hipDeviceptr_t>(hImageMem)));
    UR_ASSERT(memType == hipMemoryTypeDevice || memType == hipMemoryTypeArray,
              UR_RESULT_ERROR_INVALID_VALUE);
    if (memType == hipMemoryTypeArray) {
      // We have a hipArray_t
      if (pImageDesc->numMipLevel == 1) {
        image_res_desc.resType = HIP_RESOURCE_TYPE_ARRAY;
        image_res_desc.res.array.hArray =
            reinterpret_cast<hipArray_t>(hImageMem);
      }
      // We have a hipMipmappedArray_t
      else {
        image_res_desc.resType = HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        image_res_desc.res.mipmap.hMipmappedArray =
            reinterpret_cast<hipMipmappedArray_t>(hImageMem);
      }
    } else if (memType == hipMemoryTypeDevice) {
      // We have a USM pointer
      if (pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        image_res_desc.resType = HIP_RESOURCE_TYPE_LINEAR;
        image_res_desc.res.linear.devPtr =
            reinterpret_cast<hipDeviceptr_t>(hImageMem);
        image_res_desc.res.linear.format = format;
        image_res_desc.res.linear.numChannels = NumChannels;
        image_res_desc.res.linear.sizeInBytes =
            pImageDesc->width * PixelSizeBytes;
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        image_res_desc.resType = HIP_RESOURCE_TYPE_PITCH2D;
        image_res_desc.res.pitch2D.devPtr =
            reinterpret_cast<hipDeviceptr_t>(hImageMem);
        image_res_desc.res.pitch2D.format = format;
        image_res_desc.res.pitch2D.numChannels = NumChannels;
        image_res_desc.res.pitch2D.width = pImageDesc->width;
        image_res_desc.res.pitch2D.height = pImageDesc->height;
        image_res_desc.res.pitch2D.pitchInBytes = pImageDesc->rowPitch;
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        // Cannot create 3D image from USM.
        return UR_RESULT_ERROR_INVALID_VALUE;
      }
    } else {
      // Unknown image memory type.
      return UR_RESULT_ERROR_INVALID_VALUE;
    }

    UR_CHECK_ERROR(urTextureCreate(hSampler, pImageDesc, image_res_desc,
                                   normalized_dtype_flag, phImage));

  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue, const void *pSrc, void *pDst,
    const ur_image_desc_t *pSrcImageDesc, const ur_image_desc_t *pDstImageDesc,
    const ur_image_format_t *pSrcImageFormat,
    const ur_image_format_t *pDstImageFormat,
    ur_exp_image_copy_region_t *pCopyRegion,
    ur_exp_image_copy_flags_t imageCopyFlags, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT((imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE ||
             imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST ||
             imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE),
            UR_RESULT_ERROR_INVALID_VALUE);
  UR_ASSERT(pSrcImageFormat->channelOrder == pDstImageFormat->channelOrder,
            UR_RESULT_ERROR_INVALID_ARGUMENT);

  unsigned int NumChannels = 0;
  size_t PixelSizeBytes = 0;

  UR_CALL(urCalculateNumChannels(pSrcImageFormat->channelOrder, &NumChannels));

  // We need to get this now in bytes for calculating the total image size
  // later.
  UR_CALL(urToHipImageChannelFormat(pSrcImageFormat->channelType,
                                    pSrcImageFormat->channelOrder, nullptr,
                                    &PixelSizeBytes, nullptr));

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t Stream = hQueue->getNextTransferStream();
    enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);

    // We have to use a different copy function for each image dimensionality.

    static constexpr uint64_t MinCopyHeight{1};
    if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE) {
      if (pDstImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        unsigned int memType{};
        // hipPointerGetAttribute can detect HIP-registered memory of types:
        // hipMemoryTypeHost, hipMemoryTypeDevice, or hipMemoryTypeArray.
        UR_CHECK_ERROR(
            hipPointerGetAttribute(&memType, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   reinterpret_cast<hipDeviceptr_t>(pDst)));
        UR_ASSERT(memType == hipMemoryTypeDevice ||
                      memType == hipMemoryTypeArray,
                  UR_RESULT_ERROR_INVALID_VALUE);

        size_t CopyExtentBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        const char *SrcWithOffset = static_cast<const char *>(pSrc) +
                                    (pCopyRegion->srcOffset.x * PixelSizeBytes);

        if (memType == hipMemoryTypeArray) {
          // HIP doesn not provide async copies between host and image arrays
          // memory in versions earlier than 6.2.
#if HIP_VERSION >= 60200000
          UR_CHECK_ERROR(
              hipMemcpyHtoAAsync(static_cast<hipArray_t>(pDst),
                                 pCopyRegion->dstOffset.x * PixelSizeBytes,
                                 static_cast<const void *>(SrcWithOffset),
                                 CopyExtentBytes, Stream));
#else
          UR_CHECK_ERROR(hipMemcpyHtoA(
              static_cast<hipArray_t>(pDst),
              pCopyRegion->dstOffset.x * PixelSizeBytes,
              static_cast<const void *>(SrcWithOffset), CopyExtentBytes));
#endif
        } else if (memType == hipMemoryTypeDevice) {
          void *DstWithOffset =
              static_cast<void *>(static_cast<char *>(pDst) +
                                  (PixelSizeBytes * pCopyRegion->dstOffset.x));
          UR_CHECK_ERROR(hipMemcpyHtoDAsync(
              static_cast<hipDeviceptr_t>(DstWithOffset),
              const_cast<void *>(static_cast<const void *>(SrcWithOffset)),
              CopyExtentBytes, Stream));
        } else {
          // This should be unreachable.
          return UR_RESULT_ERROR_INVALID_VALUE;
        }
      } else if (pDstImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        hip_Memcpy2D cpy_desc = {};
        cpy_desc.srcMemoryType = hipMemoryTypeHost;
        cpy_desc.srcHost = pSrc;
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.srcPitch = pSrcImageDesc->width * PixelSizeBytes;
        if (pDstImageDesc->rowPitch == 0) {
          cpy_desc.dstMemoryType = hipMemoryTypeArray;
          cpy_desc.dstArray = static_cast<hipArray_t>(pDst);
        } else {
          // Pitched memory
          cpy_desc.dstMemoryType = hipMemoryTypeDevice;
          cpy_desc.dstDevice = static_cast<hipDeviceptr_t>(pDst);
          cpy_desc.dstPitch = pDstImageDesc->rowPitch;
        }
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        UR_CHECK_ERROR(hipMemcpyParam2DAsync(&cpy_desc, Stream));
      } else if (pDstImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        HIP_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = hipMemoryTypeHost;
        cpy_desc.srcHost = pSrc;
        cpy_desc.srcPitch = pSrcImageDesc->width * PixelSizeBytes;
        cpy_desc.srcHeight = pSrcImageDesc->height;
        cpy_desc.dstMemoryType = hipMemoryTypeArray;
        cpy_desc.dstArray = static_cast<hipArray_t>(pDst);
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        // 'hipMemcpy3DAsync' requires us to correctly create 'hipMemcpy3DParms'
        // struct object which adds a little complexity (e.g. 'hipPitchedPtr').
        UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&cpy_desc, Stream));
      } else if (pDstImageDesc->type == UR_MEM_TYPE_IMAGE1D_ARRAY ||
                 pDstImageDesc->type == UR_MEM_TYPE_IMAGE2D_ARRAY ||
                 pDstImageDesc->type == UR_MEM_TYPE_IMAGE_CUBEMAP_EXP) {
        HIP_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = hipMemoryTypeHost;
        cpy_desc.srcHost = pSrc;
        cpy_desc.srcPitch = pSrcImageDesc->width * PixelSizeBytes;
        cpy_desc.srcHeight = std::max(MinCopyHeight, pSrcImageDesc->height);
        cpy_desc.dstMemoryType = hipMemoryTypeArray;
        cpy_desc.dstArray = static_cast<hipArray_t>(pDst);
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height =
            std::max(MinCopyHeight, pCopyRegion->copyExtent.height);
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        // 'hipMemcpy3DAsync' requires us to correctly create 'hipMemcpy3DParms'
        // struct object which adds a little complexity (e.g. 'hipPitchedPtr').
        UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&cpy_desc, Stream));
      }
    } else if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST) {
      if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        unsigned int memType{};
        // hipPointerGetAttribute can detect HIP-registered memory of types:
        // hipMemoryTypeHost, hipMemoryTypeDevice, or hipMemoryTypeArray.
        UR_CHECK_ERROR(
            hipPointerGetAttribute(&memType, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   reinterpret_cast<hipDeviceptr_t>(pDst)));
        UR_ASSERT(memType == hipMemoryTypeDevice ||
                      memType == hipMemoryTypeArray,
                  UR_RESULT_ERROR_INVALID_VALUE);

        size_t CopyExtentBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        void *DstWithOffset =
            static_cast<void *>(static_cast<char *>(pDst) +
                                (PixelSizeBytes * pCopyRegion->dstOffset.x));

        if (memType == hipMemoryTypeArray) {
          // HIP doesn not provide async copies between image arrays and host
          // memory in versions earlier than 6.2.
#if HIP_VERSION >= 60200000
          UR_CHECK_ERROR(hipMemcpyAtoHAsync(
              DstWithOffset, static_cast<hipArray_t>(const_cast<void *>(pSrc)),
              PixelSizeBytes * pCopyRegion->srcOffset.x, CopyExtentBytes,
              Stream));
#else
          UR_CHECK_ERROR(hipMemcpyAtoH(
              DstWithOffset, static_cast<hipArray_t>(const_cast<void *>(pSrc)),
              PixelSizeBytes * pCopyRegion->srcOffset.x, CopyExtentBytes));
#endif
        } else if (memType == hipMemoryTypeDevice) {
          const char *SrcWithOffset =
              static_cast<const char *>(pSrc) +
              (pCopyRegion->srcOffset.x * PixelSizeBytes);
          UR_CHECK_ERROR(hipMemcpyDtoHAsync(
              DstWithOffset,
              static_cast<hipDeviceptr_t>(const_cast<char *>(SrcWithOffset)),
              CopyExtentBytes, Stream));
        } else {
          // This should be unreachable.
          return UR_RESULT_ERROR_INVALID_VALUE;
        }
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        hip_Memcpy2D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstMemoryType = hipMemoryTypeHost;
        cpy_desc.dstHost = pDst;
        if (pSrcImageDesc->rowPitch == 0) {
          cpy_desc.srcMemoryType = hipMemoryTypeArray;
          cpy_desc.srcArray = static_cast<hipArray_t>(const_cast<void *>(pSrc));
        } else {
          // Pitched memory
          cpy_desc.srcMemoryType = hipMemoryTypeDevice;
          cpy_desc.srcPitch = pSrcImageDesc->rowPitch;
          cpy_desc.srcDevice =
              static_cast<hipDeviceptr_t>(const_cast<void *>(pSrc));
        }
        cpy_desc.dstMemoryType = hipMemoryTypeHost;
        cpy_desc.dstHost = pDst;
        cpy_desc.dstPitch = pDstImageDesc->width * PixelSizeBytes;
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        UR_CHECK_ERROR(hipMemcpyParam2DAsync(&cpy_desc, Stream));
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        HIP_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = hipMemoryTypeArray;
        cpy_desc.srcArray = static_cast<hipArray_t>(const_cast<void *>(pSrc));
        cpy_desc.dstMemoryType = hipMemoryTypeHost;
        cpy_desc.dstHost = pDst;
        cpy_desc.dstPitch = pDstImageDesc->width * PixelSizeBytes;
        cpy_desc.dstHeight = pDstImageDesc->height;
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        // 'hipMemcpy3DAsync' requires us to correctly create 'hipMemcpy3DParms'
        // struct object which adds a little complexity (e.g. 'hipPitchedPtr').
        UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&cpy_desc, Stream));
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE1D_ARRAY ||
                 pSrcImageDesc->type == UR_MEM_TYPE_IMAGE2D_ARRAY ||
                 pSrcImageDesc->type == UR_MEM_TYPE_IMAGE_CUBEMAP_EXP) {
        HIP_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = hipMemoryTypeArray;
        cpy_desc.srcArray = static_cast<hipArray_t>(const_cast<void *>(pSrc));
        cpy_desc.dstMemoryType = hipMemoryTypeHost;
        cpy_desc.dstHost = pDst;
        cpy_desc.dstPitch = pDstImageDesc->width * PixelSizeBytes;
        cpy_desc.dstHeight = std::max(MinCopyHeight, pDstImageDesc->height);
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height =
            std::max(MinCopyHeight, pCopyRegion->copyExtent.height);
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        // 'hipMemcpy3DAsync' requires us to correctly create 'hipMemcpy3DParms'
        // struct object which adds a little complexity (e.g. 'hipPitchedPtr').
        UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&cpy_desc, Stream));
      }
    } else {
      // imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE

      // we don't support copying between different image types.
      if (pSrcImageDesc->type != pDstImageDesc->type) {
        UR_LOG(ERR,
               "Unsupported copy operation between different type of images");
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
      }

      // All the following async copy function calls should be treated as
      // synchronous because of the explicit call to hipStreamSynchronize at
      // the end
      if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        hip_Memcpy2D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = 0;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = 0;
        cpy_desc.srcMemoryType = hipMemoryTypeArray;
        cpy_desc.srcArray = static_cast<hipArray_t>(const_cast<void *>(pSrc));
        cpy_desc.dstMemoryType = hipMemoryTypeArray;
        cpy_desc.dstArray = static_cast<hipArray_t>(pDst);
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = 1;
        UR_CHECK_ERROR(hipMemcpyParam2DAsync(&cpy_desc, Stream));
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        hip_Memcpy2D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.srcMemoryType = hipMemoryTypeArray;
        cpy_desc.srcArray = static_cast<hipArray_t>(const_cast<void *>(pSrc));
        cpy_desc.dstMemoryType = hipMemoryTypeArray;
        cpy_desc.dstArray = static_cast<hipArray_t>(pDst);
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        UR_CHECK_ERROR(hipMemcpyParam2DAsync(&cpy_desc, Stream));
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        HIP_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = hipMemoryTypeArray;
        cpy_desc.srcArray = static_cast<hipArray_t>(const_cast<void *>(pSrc));
        cpy_desc.dstMemoryType = hipMemoryTypeArray;
        cpy_desc.dstArray = static_cast<hipArray_t>(pDst);
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        // 'hipMemcpy3DAsync' requires us to correctly create 'hipMemcpy3DParms'
        // struct object which adds a little complexity (e.g. 'hipPitchedPtr').
        UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&cpy_desc, Stream));
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE1D_ARRAY ||
                 pSrcImageDesc->type == UR_MEM_TYPE_IMAGE2D_ARRAY ||
                 pSrcImageDesc->type == UR_MEM_TYPE_IMAGE_CUBEMAP_EXP) {
        HIP_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = hipMemoryTypeArray;
        cpy_desc.srcArray = static_cast<hipArray_t>(const_cast<void *>(pSrc));
        cpy_desc.dstMemoryType = hipMemoryTypeArray;
        cpy_desc.dstArray = static_cast<hipArray_t>(pDst);
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height =
            std::max(MinCopyHeight, pCopyRegion->copyExtent.height);
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        // 'hipMemcpy3DAsync' requires us to correctly create 'hipMemcpy3DParms'
        // struct object which adds a little complexity (e.g. 'hipPitchedPtr').
        UR_CHECK_ERROR(hipDrvMemcpy3DAsync(&cpy_desc, Stream));
      }
      // Synchronization is required here to handle the case of copying data
      // from host to device, then device to device and finally device to host.
      // Without it, there is a risk of the copies not being executed in the
      // intended order.
      UR_CHECK_ERROR(hipStreamSynchronize(Stream));
    }

    if (phEvent) {
      auto NewEvent = ur_event_handle_t_::makeNative(UR_COMMAND_MEM_IMAGE_COPY,
                                                     hQueue, Stream);
      NewEvent->record();
      *phEvent = NewEvent;
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageGetInfoExp(
    [[maybe_unused]] ur_context_handle_t hContext,
    [[maybe_unused]] ur_exp_image_mem_native_handle_t hImageMem,
    [[maybe_unused]] ur_image_info_t propName,
    [[maybe_unused]] void *pPropValue, [[maybe_unused]] size_t *pPropSizeRet) {
  // hipArrayGetDescriptor and hipArray3DGetDescriptor are supported only since
  // ROCm 5.6.0, so we can't query image array information for older versions.
#if HIP_VERSION >= 50600000
  unsigned int memType{};
  hipError_t Err =
      hipPointerGetAttribute(&memType, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                             reinterpret_cast<hipDeviceptr_t>(hImageMem));
  if (Err != hipSuccess) {
    return mapErrorUR(Err);
  }
  UR_ASSERT(memType == hipMemoryTypeArray, UR_RESULT_ERROR_INVALID_VALUE);

  hipArray_t ImageArray;
  // If hipMipmappedArrayGetLevel failed, hImageMem is already hipArray_t.
  Err = hipMipmappedArrayGetLevel(
      &ImageArray, reinterpret_cast<hipMipmappedArray_t>(hImageMem), 0);
  if (Err != hipSuccess) {
    ImageArray = reinterpret_cast<hipArray_t>(hImageMem);
  }

  HIP_ARRAY3D_DESCRIPTOR ArrayDesc;
  Err = hipArray3DGetDescriptor(&ArrayDesc, ImageArray);
  if (Err != hipSuccess) {
    return mapErrorUR(Err);
  }

  switch (propName) {
  case UR_IMAGE_INFO_WIDTH:
    if (pPropValue) {
      *static_cast<size_t *>(pPropValue) = ArrayDesc.Width;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(size_t);
    }
    return UR_RESULT_SUCCESS;
  case UR_IMAGE_INFO_HEIGHT:
    if (pPropValue) {
      *static_cast<size_t *>(pPropValue) = ArrayDesc.Height;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(size_t);
    }
    return UR_RESULT_SUCCESS;
  case UR_IMAGE_INFO_DEPTH:
    if (pPropValue) {
      *static_cast<size_t *>(pPropValue) = ArrayDesc.Depth;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(size_t);
    }
    return UR_RESULT_SUCCESS;
  case UR_IMAGE_INFO_FORMAT: {
    ur_image_channel_type_t ChannelType{};
    ur_image_channel_order_t ChannelOrder{};
    UR_CALL(hipToUrImageChannelFormat(ArrayDesc.Format, &ChannelType));
    // HIP does not have a notion of channel "order" in the same way that
    // SYCL 1.2.1 does.
    switch (ArrayDesc.NumChannels) {
    case 1:
      ChannelOrder = UR_IMAGE_CHANNEL_ORDER_R;
      break;
    case 2:
      ChannelOrder = UR_IMAGE_CHANNEL_ORDER_RG;
      break;
    case 4:
      ChannelOrder = UR_IMAGE_CHANNEL_ORDER_RGBA;
      break;
    default:
      setErrorMessage("Unexpected NumChannels returned by HIP",
                      UR_RESULT_ERROR_INVALID_VALUE);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
    }
    if (pPropValue) {
      (static_cast<ur_image_format_t *>(pPropValue))->channelType = ChannelType;
      (static_cast<ur_image_format_t *>(pPropValue))->channelOrder =
          ChannelOrder;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(ur_image_format_t);
    }
    return UR_RESULT_SUCCESS;
  }
  default:
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
#else
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif
}

bool verifyStandardImageSupport(const ur_device_handle_t hDevice,
                                const ur_image_desc_t *pImageDesc,
                                ur_exp_image_mem_type_t imageMemHandleType) {
  // Verify standard image dimensions are within device limits.
  size_t maxImageWidth, maxImageHeight, maxImageDepth;

  if (pImageDesc->depth != 0 && pImageDesc->type == UR_MEM_TYPE_IMAGE3D) {

    // Verify for standard 3D images.
    UR_CHECK_ERROR(urDeviceGetInfo(hDevice, UR_DEVICE_INFO_IMAGE3D_MAX_WIDTH,
                                   sizeof(size_t), &maxImageWidth, nullptr));
    UR_CHECK_ERROR(urDeviceGetInfo(hDevice, UR_DEVICE_INFO_IMAGE3D_MAX_HEIGHT,
                                   sizeof(size_t), &maxImageHeight, nullptr));
    UR_CHECK_ERROR(urDeviceGetInfo(hDevice, UR_DEVICE_INFO_IMAGE3D_MAX_DEPTH,
                                   sizeof(size_t), &maxImageDepth, nullptr));
    if ((pImageDesc->width > maxImageWidth) ||
        (pImageDesc->height > maxImageHeight) ||
        (pImageDesc->depth > maxImageDepth)) {
      return false;
    }
  } else if (pImageDesc->height != 0 &&
             pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {

    if (imageMemHandleType == UR_EXP_IMAGE_MEM_TYPE_USM_POINTER) {
      // Verify for standard 2D images backed by linear memory.
      UR_CHECK_ERROR(urDeviceGetInfo(hDevice,
                                     UR_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP,
                                     sizeof(size_t), &maxImageWidth, nullptr));
      UR_CHECK_ERROR(urDeviceGetInfo(hDevice,
                                     UR_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP,
                                     sizeof(size_t), &maxImageHeight, nullptr));

      size_t maxImageLinearPitch;
      UR_CHECK_ERROR(
          urDeviceGetInfo(hDevice, UR_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP,
                          sizeof(size_t), &maxImageLinearPitch, nullptr));
      if (pImageDesc->rowPitch > maxImageLinearPitch) {
        return false;
      }
    } else {
      // Verify for standard 2D images backed by opaque memory.
      UR_CHECK_ERROR(urDeviceGetInfo(hDevice, UR_DEVICE_INFO_IMAGE2D_MAX_WIDTH,
                                     sizeof(size_t), &maxImageWidth, nullptr));
      UR_CHECK_ERROR(urDeviceGetInfo(hDevice, UR_DEVICE_INFO_IMAGE2D_MAX_HEIGHT,
                                     sizeof(size_t), &maxImageHeight, nullptr));
    }

    if ((pImageDesc->width > maxImageWidth) ||
        (pImageDesc->height > maxImageHeight)) {
      return false;
    }
  } else if (pImageDesc->width != 0 &&
             pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {

    if (imageMemHandleType == UR_EXP_IMAGE_MEM_TYPE_USM_POINTER) {
      // Verify for standard 1D images backed by linear memory.
      //
      /// TODO: We have a query for `max_image_linear_width`, however, that
      /// query is for 2D textures (at least as far as the CUDA/HIP
      /// implementations go). We should split the `max_image_linear_width`
      /// query into 1D and 2D variants to ensure that 1D image dimensions
      /// can be properly verified and used to the fullest extent.
      int32_t maxImageLinearWidth;
      UR_CHECK_ERROR(hipDeviceGetAttribute(&maxImageLinearWidth,
                                           hipDeviceAttributeMaxTexture1DLinear,
                                           hDevice->get()));
      maxImageWidth = static_cast<size_t>(maxImageLinearWidth);
    } else {
      // Verify for standard 1D images backed by opaque memory.
      UR_CHECK_ERROR(urDeviceGetInfo(hDevice,
                                     UR_DEVICE_INFO_IMAGE_MAX_BUFFER_SIZE,
                                     sizeof(size_t), &maxImageWidth, nullptr));
    }
    if ((pImageDesc->width > maxImageWidth)) {
      return false;
    }
  }

  return true;
}

bool verifyMipmapImageSupport(
    [[maybe_unused]] const ur_device_handle_t hDevice,
    const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_image_mem_type_t imageMemHandleType) {
  // Verify mipmap image support.
  // Mimpaps are not currently supported for the AMD target.
  if (pImageDesc->numMipLevel > 1) {
    return false;
  }

  return true;
}

bool verifyCubemapImageSupport(
    [[maybe_unused]] const ur_device_handle_t hDevice,
    const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_image_mem_type_t imageMemHandleType) {
  // Verify cubemap image support.
  // Cubemaps are not currently supported for the AMD target.
  if (pImageDesc->type == UR_MEM_TYPE_IMAGE_CUBEMAP_EXP) {
    return false;
  }

  return true;
}

bool verifyLayeredImageSupport(
    [[maybe_unused]] const ur_device_handle_t hDevice,
    const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_image_mem_type_t imageMemHandleType) {
  // Verify layered image support.
  // Layered images are not currently supported for the AMD target.
  if ((pImageDesc->type == UR_MEM_TYPE_IMAGE1D_ARRAY) ||
      pImageDesc->type == UR_MEM_TYPE_IMAGE2D_ARRAY) {
    return false;
  }

  return true;
}

bool verifyGatherImageSupport(
    [[maybe_unused]] const ur_device_handle_t hDevice,
    const ur_image_desc_t *pImageDesc,
    [[maybe_unused]] ur_exp_image_mem_type_t imageMemHandleType) {
  // Verify gather image support.
  // Gather images are not currently supported for the AMD target.
  if (pImageDesc->type == UR_MEM_TYPE_IMAGE_GATHER_EXP) {
    return false;
  }

  return true;
}

bool verifyCommonImagePropertiesSupport(
    const ur_device_handle_t hDevice, const ur_image_desc_t *pImageDesc,
    const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType) {

  bool supported = true;

  supported &=
      verifyStandardImageSupport(hDevice, pImageDesc, imageMemHandleType);

  supported &=
      verifyMipmapImageSupport(hDevice, pImageDesc, imageMemHandleType);

  supported &=
      verifyLayeredImageSupport(hDevice, pImageDesc, imageMemHandleType);

  supported &=
      verifyCubemapImageSupport(hDevice, pImageDesc, imageMemHandleType);

  supported &=
      verifyGatherImageSupport(hDevice, pImageDesc, imageMemHandleType);

  // Verify 3-channel format support.
  // HIP does not allow 3-channel formats.
  if (pImageFormat->channelOrder == UR_IMAGE_CHANNEL_ORDER_RGB ||
      pImageFormat->channelOrder == UR_IMAGE_CHANNEL_ORDER_RGX) {
    return false;
  }

  return supported;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesGetImageMemoryHandleTypeSupportExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_desc_t *pImageDesc, const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType, ur_bool_t *pSupportedRet) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  // Verify support for common image properties (dims, channel types, image
  // types, etc.).
  *pSupportedRet = verifyCommonImagePropertiesSupport(
      hDevice, pImageDesc, pImageFormat, imageMemHandleType);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesGetImageUnsampledHandleSupportExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_desc_t *pImageDesc, const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType, ur_bool_t *pSupportedRet) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  // Currently Bindless Images do not allow creation of unsampled image handles
  // from non-opaque (USM) memory.
  if (imageMemHandleType == UR_EXP_IMAGE_MEM_TYPE_USM_POINTER) {
    *pSupportedRet = false;
    return UR_RESULT_SUCCESS;
  }

  // Bindless Images do not allow creation of `unsampled_image_handle`s for
  // mipmap images.
  if (pImageDesc->numMipLevel > 1) {
    *pSupportedRet = false;
    return UR_RESULT_SUCCESS;
  }

  // Verify support for common image properties (dims, channel types, image
  // types, etc.).
  *pSupportedRet = verifyCommonImagePropertiesSupport(
      hDevice, pImageDesc, pImageFormat, imageMemHandleType);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesGetImageSampledHandleSupportExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_desc_t *pImageDesc, const ur_image_format_t *pImageFormat,
    ur_exp_image_mem_type_t imageMemHandleType, ur_bool_t *pSupportedRet) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  // Verify support for common image properties (dims, channel types, image
  // types, etc.).
  *pSupportedRet = verifyCommonImagePropertiesSupport(
      hDevice, pImageDesc, pImageFormat, imageMemHandleType);
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMipmapGetLevelExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hImageMem, uint32_t mipmapLevel,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedDevice Active(hDevice);
    hipArray_t ImageArray;
    UR_CHECK_ERROR(hipMipmappedArrayGetLevel(
        &ImageArray, reinterpret_cast<hipMipmappedArray_t>(hImageMem),
        mipmapLevel));
    *phImageMem =
        reinterpret_cast<ur_exp_image_mem_native_handle_t>(ImageArray);
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_native_handle_t hMem) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  ScopedDevice Active(hDevice);
  try {
    UR_CHECK_ERROR(
        hipMipmappedArrayDestroy(reinterpret_cast<hipMipmappedArray_t>(hMem)));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImportExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    ur_exp_external_mem_type_t memHandleType,
    ur_exp_external_mem_desc_t *pExternalMemDesc,
    ur_exp_external_mem_handle_t *phExternalMem) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedDevice Active(hDevice);

    hipExternalMemoryHandleDesc extMemDesc = {};
    extMemDesc.size = size;

    void *pNext = const_cast<void *>(pExternalMemDesc->pNext);
    while (pNext != nullptr) {
      const ur_base_desc_t *BaseDesc =
          static_cast<const ur_base_desc_t *>(pNext);
      if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR) {
        auto FileDescriptor =
            static_cast<const ur_exp_file_descriptor_t *>(pNext);

        extMemDesc.handle.fd = FileDescriptor->fd;
        extMemDesc.type = hipExternalMemoryHandleTypeOpaqueFd;
      } else if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE) {
        auto Win32Handle = static_cast<const ur_exp_win32_handle_t *>(pNext);

        switch (memHandleType) {
        case UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT:
          extMemDesc.type = hipExternalMemoryHandleTypeOpaqueWin32;
          break;
        case UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT_DX12_RESOURCE:
          // Memory descriptor flag values such as hipExternalMemoryDedicated
          // are not available before HIP 5.6, so we safely fallback to marking
          // this as an unsupported.
#if HIP_VERSION >= 50600000
          extMemDesc.type = hipExternalMemoryHandleTypeD3D12Resource;
          extMemDesc.flags = hipExternalMemoryDedicated;
#else
          return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
#endif
          break;
        case UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD:
        default:
          return UR_RESULT_ERROR_INVALID_VALUE;
        }
        extMemDesc.handle.win32.handle = Win32Handle->handle;
      }
      pNext = const_cast<void *>(BaseDesc->pNext);
    }

    hipExternalMemory_t extMem;
    UR_CHECK_ERROR(hipImportExternalMemory(&extMem, &extMemDesc));
    *phExternalMem = reinterpret_cast<ur_exp_external_mem_handle_t>(extMem);

  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMapExternalArrayExp(
    ur_context_handle_t /*hContext*/, ur_device_handle_t /*hDevice*/,
    const ur_image_format_t * /*pImageFormat*/,
    const ur_image_desc_t * /*pImageDesc*/,
    [[maybe_unused]] ur_exp_external_mem_handle_t hExternalMem,
    ur_exp_image_mem_native_handle_t * /*phImageMem*/) {
  // hipExternalMemoryGetMappedMipmappedArray should be introduced from ROCm 6.
  // However, there is an issue at the moment with the required function symbol
  // missing from the libamdhip64.so library, despite being shown in the docs.
  // TODO: Update this with a link to a bug report filed on the ROCm github.
  return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMapExternalLinearMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint64_t offset,
    uint64_t size, ur_exp_external_mem_handle_t hExternalMem, void **ppRetMem) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedDevice Active(hDevice);

    hipExternalMemoryBufferDesc BufferDesc = {};
    BufferDesc.size = size;
    BufferDesc.offset = offset;
    BufferDesc.flags = 0;

    hipDeviceptr_t retMem;
    UR_CHECK_ERROR(hipExternalMemoryGetMappedBuffer(
        &retMem, reinterpret_cast<hipExternalMemory_t>(hExternalMem),
        &BufferDesc));

    *ppRetMem = static_cast<void *>(retMem);

  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesReleaseExternalMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_mem_handle_t hExternalMem) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedDevice Active(hDevice);
    UR_CHECK_ERROR(hipDestroyExternalMemory(
        reinterpret_cast<hipExternalMemory_t>(hExternalMem)));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesFreeMappedLinearMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, void *pMem) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);
  UR_ASSERT(pMem, UR_RESULT_ERROR_INVALID_NULL_POINTER);

  try {
    ScopedDevice Active(hDevice);
    UR_CHECK_ERROR(hipFree(static_cast<hipDeviceptr_t>(pMem)));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImportExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_type_t semHandleType,
    ur_exp_external_semaphore_desc_t *pExternalSemaphoreDesc,
    ur_exp_external_semaphore_handle_t *phExternalSemaphoreHandle) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedDevice Active(hDevice);

    hipExternalSemaphoreHandleDesc extSemDesc = {};

    void *pNext = const_cast<void *>(pExternalSemaphoreDesc->pNext);
    while (pNext != nullptr) {
      const ur_base_desc_t *BaseDesc =
          static_cast<const ur_base_desc_t *>(pNext);
      if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR) {
        auto FileDescriptor =
            static_cast<const ur_exp_file_descriptor_t *>(pNext);

        extSemDesc.handle.fd = FileDescriptor->fd;
        extSemDesc.type = hipExternalSemaphoreHandleTypeOpaqueFd;
      } else if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE) {
        auto Win32Handle = static_cast<const ur_exp_win32_handle_t *>(pNext);
        switch (semHandleType) {
        case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT:
          extSemDesc.type = hipExternalSemaphoreHandleTypeOpaqueWin32;
          break;
        case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT_DX12_FENCE:
          extSemDesc.type = hipExternalSemaphoreHandleTypeD3D12Fence;
          break;
        case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_OPAQUE_FD:
          [[fallthrough]];
        default:
          return UR_RESULT_ERROR_INVALID_VALUE;
        }
        extSemDesc.handle.win32.handle = Win32Handle->handle;
      }
      pNext = const_cast<void *>(BaseDesc->pNext);
    }

    hipExternalSemaphore_t semaphore;
    UR_CHECK_ERROR(hipImportExternalSemaphore(&semaphore, &extSemDesc));

    *phExternalSemaphoreHandle =
        reinterpret_cast<ur_exp_external_semaphore_handle_t>(semaphore);
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesReleaseExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_external_semaphore_handle_t hExternalSemaphore) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedDevice Active(hDevice);
    UR_CHECK_ERROR(hipDestroyExternalSemaphore(
        reinterpret_cast<hipExternalSemaphore_t>(hExternalSemaphore)));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasValue, uint64_t waitValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t Stream = hQueue->getNextTransferStream();

    enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);

    hipExternalSemaphoreWaitParams SemWaitParams = {};
    if (hasValue) {
      SemWaitParams.params.fence.value = waitValue;
    }

    // Wait for one external semaphore
    UR_CHECK_ERROR(hipWaitExternalSemaphoresAsync(
        reinterpret_cast<hipExternalSemaphore_t *>(&hSemaphore), &SemWaitParams,
        1 /* numExtSems */, Stream));

    if (phEvent) {
      auto NewEvent = ur_event_handle_t_::makeNative(
          UR_COMMAND_EXTERNAL_SEMAPHORE_WAIT_EXP, hQueue, Stream);
      NewEvent->record();
      *phEvent = NewEvent;
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSignalExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_external_semaphore_handle_t hSemaphore,
    bool hasValue, uint64_t signalValue, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {

  try {
    ScopedDevice Active(hQueue->getDevice());
    hipStream_t Stream = hQueue->getNextTransferStream();

    enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);

    hipExternalSemaphoreSignalParams SemSignalParams = {};
    if (hasValue) {
      SemSignalParams.params.fence.value = signalValue;
    }

    // Signal one external semaphore
    UR_CHECK_ERROR(hipSignalExternalSemaphoresAsync(
        reinterpret_cast<hipExternalSemaphore_t *>(&hSemaphore),
        &SemSignalParams, 1 /* numExtSems */, Stream));

    if (phEvent) {
      auto NewEvent = ur_event_handle_t_::makeNative(
          UR_COMMAND_EXTERNAL_SEMAPHORE_SIGNAL_EXP, hQueue, Stream);
      NewEvent->record();
      *phEvent = NewEvent;
    }
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}
