//===--------- image.cpp - CUDA Adapter -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda.h>
#include <map>
#include <utility>

#include "common.hpp"
#include "context.hpp"
#include "enqueue.hpp"
#include "event.hpp"
#include "image.hpp"
#include "logger/ur_logger.hpp"
#include "memory.hpp"
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

/// Convert a UR image format to a CUDA image format and
/// get the pixel size in bytes.
/// /param image_channel_type is the ur_image_channel_type_t.
/// /param image_channel_order is the ur_image_channel_order_t.
///        this is used for normalized channel formats, as CUDA
///        combines the channel format and order for normalized
///        channel types.
/// /param return_cuda_format will be set to the equivalent cuda
///        format if not nullptr.
/// /param return_pixel_size_bytes will be set to the pixel
///        byte size if not nullptr.
/// /param return_normalized_dtype_flag will be set if the
///        data type is normalized if not nullptr.
ur_result_t
urToCudaImageChannelFormat(ur_image_channel_type_t image_channel_type,
                           ur_image_channel_order_t image_channel_order,
                           CUarray_format *return_cuda_format,
                           size_t *return_pixel_size_bytes,
                           unsigned int *return_normalized_dtype_flag) {

  CUarray_format cuda_format = CU_AD_FORMAT_UNSIGNED_INT8;
  size_t pixel_size_bytes = 0;
  unsigned int num_channels = 0;
  unsigned int normalized_dtype_flag = 0;
  UR_CHECK_ERROR(urCalculateNumChannels(image_channel_order, &num_channels));

  switch (image_channel_type) {
#define CASE(FROM, TO, SIZE, NORM)                                             \
  case FROM: {                                                                 \
    cuda_format = TO;                                                          \
    pixel_size_bytes = SIZE * num_channels;                                    \
    normalized_dtype_flag = NORM;                                              \
    break;                                                                     \
  }

    CASE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8, CU_AD_FORMAT_UNSIGNED_INT8, 1, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8, CU_AD_FORMAT_SIGNED_INT8, 1, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16, CU_AD_FORMAT_UNSIGNED_INT16, 2,
         0)
    CASE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16, CU_AD_FORMAT_SIGNED_INT16, 2, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT, CU_AD_FORMAT_HALF, 2, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32, CU_AD_FORMAT_UNSIGNED_INT32, 4,
         0)
    CASE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32, CU_AD_FORMAT_SIGNED_INT32, 4, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_FLOAT, CU_AD_FORMAT_FLOAT, 4, 0)
    CASE(UR_IMAGE_CHANNEL_TYPE_UNORM_INT8, CU_AD_FORMAT_UNSIGNED_INT8, 1, 1)
    CASE(UR_IMAGE_CHANNEL_TYPE_SNORM_INT8, CU_AD_FORMAT_SIGNED_INT8, 1, 1)
    CASE(UR_IMAGE_CHANNEL_TYPE_UNORM_INT16, CU_AD_FORMAT_UNSIGNED_INT16, 2, 1)
    CASE(UR_IMAGE_CHANNEL_TYPE_SNORM_INT16, CU_AD_FORMAT_SIGNED_INT16, 2, 1)

#undef CASE
  default:
    break;
  }

  if (return_cuda_format) {
    *return_cuda_format = cuda_format;
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
cudaToUrImageChannelFormat(CUarray_format cuda_format,
                           ur_image_channel_type_t *return_image_channel_type) {

  switch (cuda_format) {
#define CUDA_TO_UR_IMAGE_CHANNEL_TYPE(FROM, TO)                                \
  case FROM: {                                                                 \
    *return_image_channel_type = TO;                                           \
    return UR_RESULT_SUCCESS;                                                  \
  }
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_UNSIGNED_INT8,
                                  UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_UNSIGNED_INT16,
                                  UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_UNSIGNED_INT32,
                                  UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_SIGNED_INT8,
                                  UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_SIGNED_INT16,
                                  UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_SIGNED_INT32,
                                  UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_HALF,
                                  UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_FLOAT,
                                  UR_IMAGE_CHANNEL_TYPE_FLOAT);
  default:
    // Default invalid enum
    *return_image_channel_type = UR_IMAGE_CHANNEL_TYPE_FORCE_UINT32;
    return UR_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT;
  }
}

ur_result_t urTextureCreate(ur_sampler_handle_t hSampler,
                            const ur_image_desc_t *pImageDesc,
                            const CUDA_RESOURCE_DESC &ResourceDesc,
                            const unsigned int normalized_dtype_flag,
                            ur_exp_image_native_handle_t *phRetImage) {

  try {
    /// pi_sampler_properties
    /// Layout of UR samplers for CUDA
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
    CUDA_TEXTURE_DESC ImageTexDesc = {};
    CUaddress_mode AddrMode[3] = {};
    for (size_t i = 0; i < 3; i++) {
      ur_sampler_addressing_mode_t AddrModeProp =
          hSampler->getAddressingModeDim(i);
      if (AddrModeProp == (UR_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE -
                           UR_SAMPLER_ADDRESSING_MODE_NONE)) {
        AddrMode[i] = CU_TR_ADDRESS_MODE_CLAMP;
      } else if (AddrModeProp == (UR_SAMPLER_ADDRESSING_MODE_CLAMP -
                                  UR_SAMPLER_ADDRESSING_MODE_NONE)) {
        AddrMode[i] = CU_TR_ADDRESS_MODE_BORDER;
      } else if (AddrModeProp == (UR_SAMPLER_ADDRESSING_MODE_REPEAT -
                                  UR_SAMPLER_ADDRESSING_MODE_NONE)) {
        AddrMode[i] = CU_TR_ADDRESS_MODE_WRAP;
      } else if (AddrModeProp == (UR_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT -
                                  UR_SAMPLER_ADDRESSING_MODE_NONE)) {
        AddrMode[i] = CU_TR_ADDRESS_MODE_MIRROR;
      }
    }

    CUfilter_mode FilterMode;
    ur_sampler_filter_mode_t FilterModeProp = hSampler->getFilterMode();
    FilterMode =
        FilterModeProp ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;
    ImageTexDesc.filterMode = FilterMode;

    // Mipmap attributes
    CUfilter_mode MipFilterMode;
    ur_sampler_filter_mode_t MipFilterModeProp = hSampler->getMipFilterMode();
    MipFilterMode =
        MipFilterModeProp ? CU_TR_FILTER_MODE_LINEAR : CU_TR_FILTER_MODE_POINT;
    ImageTexDesc.mipmapFilterMode = MipFilterMode;
    ImageTexDesc.maxMipmapLevelClamp = hSampler->MaxMipmapLevelClamp;
    ImageTexDesc.minMipmapLevelClamp = hSampler->MinMipmapLevelClamp;
    ImageTexDesc.maxAnisotropy = static_cast<unsigned>(hSampler->MaxAnisotropy);

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
                             ? CU_TRSF_NORMALIZED_COORDINATES
                             : ImageTexDesc.flags;

    // CUDA default promotes 8-bit and 16-bit integers to float between [0,1]
    // This flag prevents this behaviour.
    if (!normalized_dtype_flag) {
      ImageTexDesc.flags |= CU_TRSF_READ_AS_INTEGER;
    }
    // Cubemap attributes
    ur_exp_sampler_cubemap_filter_mode_t CubemapFilterModeProp =
        hSampler->getCubemapFilterMode();
    if (CubemapFilterModeProp == UR_EXP_SAMPLER_CUBEMAP_FILTER_MODE_SEAMLESS) {
#if CUDA_VERSION >= 11060
      ImageTexDesc.flags |= CU_TRSF_SEAMLESS_CUBEMAP;
#else
      setErrorMessage("The UR_EXP_SAMPLER_CUBEMAP_FILTER_MODE_SEAMLESS "
                      "feature requires cuda 11.6 or later.",
                      UR_RESULT_ERROR_ADAPTER_SPECIFIC);
      return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
#endif
    }

    CUtexObject Texture;
    UR_CHECK_ERROR(
        cuTexObjectCreate(&Texture, &ResourceDesc, &ImageTexDesc, nullptr));
    *phRetImage = (ur_exp_image_native_handle_t)Texture;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urUSMPitchedAllocExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_usm_desc_t *pUSMDesc, ur_usm_pool_handle_t pool,
    size_t widthInBytes, size_t height, size_t elementSizeBytes, void **ppMem,
    size_t *pResultPitch) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);
  std::ignore = pUSMDesc;
  std::ignore = pool;

  UR_ASSERT((widthInBytes > 0), UR_RESULT_ERROR_INVALID_VALUE);
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
    ScopedContext Active(hDevice);
    UR_CHECK_ERROR(cuMemAllocPitch((CUdeviceptr *)ppMem, pResultPitch,
                                   widthInBytes, height, elementSizeBytes));
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

  UR_CHECK_ERROR(cuSurfObjectDestroy((CUsurfObject)hImage));
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

  UR_CHECK_ERROR(cuTexObjectDestroy((CUtexObject)hImage));
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
  CUDA_ARRAY3D_DESCRIPTOR array_desc = {};

  UR_CHECK_ERROR(urCalculateNumChannels(pImageFormat->channelOrder,
                                        &array_desc.NumChannels));

  UR_CHECK_ERROR(urToCudaImageChannelFormat(
      pImageFormat->channelType, pImageFormat->channelOrder, &array_desc.Format,
      nullptr, nullptr));

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
    array_desc.Flags |= CUDA_ARRAY3D_LAYERED;
    break;
  case UR_MEM_TYPE_IMAGE2D_ARRAY:
    array_desc.Height = pImageDesc->height;
    array_desc.Depth = pImageDesc->arraySize;
    array_desc.Flags |= CUDA_ARRAY3D_LAYERED;
    break;
  case UR_MEM_TYPE_IMAGE_CUBEMAP_EXP:
    array_desc.Height = pImageDesc->height;
    array_desc.Depth = pImageDesc->arraySize; // Should be 6 ONLY
    array_desc.Flags |= CUDA_ARRAY3D_CUBEMAP;
    break;
  default:
    return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
  }

  ScopedContext Active(hDevice);

  // Allocate a cuArray
  if (pImageDesc->numMipLevel == 1) {
    CUarray ImageArray{};

    try {
      UR_CHECK_ERROR(cuArray3DCreate(&ImageArray, &array_desc));
      *phImageMem = (ur_exp_image_mem_native_handle_t)ImageArray;
    } catch (ur_result_t Err) {
      if (ImageArray != CUarray{}) {
        UR_CHECK_ERROR(cuArrayDestroy(ImageArray));
      }
      return Err;
    } catch (...) {
      if (ImageArray != CUarray{}) {
        UR_CHECK_ERROR(cuArrayDestroy(ImageArray));
      }
      return UR_RESULT_ERROR_UNKNOWN;
    }
  } else // Allocate a cuMipmappedArray
  {
    CUmipmappedArray mip_array{};
    array_desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;

    try {
      UR_CHECK_ERROR(cuMipmappedArrayCreate(&mip_array, &array_desc,
                                            pImageDesc->numMipLevel));
      *phImageMem = (ur_exp_image_mem_native_handle_t)mip_array;
    } catch (ur_result_t Err) {
      if (mip_array) {
        UR_CHECK_ERROR(cuMipmappedArrayDestroy(mip_array));
      }
      return Err;
    } catch (...) {
      if (mip_array) {
        UR_CHECK_ERROR(cuMipmappedArrayDestroy(mip_array));
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

  ScopedContext Active(hDevice);
  try {
    UR_CHECK_ERROR(cuArrayDestroy((CUarray)hImageMem));
    if (auto it = hDevice->ChildCuarrayFromMipmapMap.find((CUarray)hImageMem);
        it != hDevice->ChildCuarrayFromMipmapMap.end()) {
      UR_CHECK_ERROR(cuMipmappedArrayDestroy((CUmipmappedArray)it->second));
      hDevice->ChildCuarrayFromMipmapMap.erase(it);
    }
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
  UR_CHECK_ERROR(
      urCalculateNumChannels(pImageFormat->channelOrder, &NumChannels));

  CUarray_format format;
  size_t PixelSizeBytes;
  UR_CHECK_ERROR(urToCudaImageChannelFormat(pImageFormat->channelType,
                                            pImageFormat->channelOrder, &format,
                                            &PixelSizeBytes, nullptr));

  try {

    ScopedContext Active(hDevice);

    CUDA_RESOURCE_DESC image_res_desc = {};

    // We have a CUarray
    image_res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    image_res_desc.res.array.hArray = (CUarray)hImageMem;

    // We create surfaces in the unsampled images case as it conforms to how
    // CUDA deals with unsampled images.
    CUsurfObject surface;
    UR_CHECK_ERROR(cuSurfObjectCreate(&surface, &image_res_desc));
    *phImage = (ur_exp_image_native_handle_t)surface;

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

  ScopedContext Active(hDevice);

  unsigned int NumChannels = 0;
  UR_CHECK_ERROR(
      urCalculateNumChannels(pImageFormat->channelOrder, &NumChannels));

  CUarray_format format;
  size_t PixelSizeBytes;
  unsigned int normalized_dtype_flag;
  UR_CHECK_ERROR(urToCudaImageChannelFormat(
      pImageFormat->channelType, pImageFormat->channelOrder, &format,
      &PixelSizeBytes, &normalized_dtype_flag));

  try {
    CUDA_RESOURCE_DESC image_res_desc = {};

    unsigned int mem_type;
    // If this function doesn't return successfully, we assume that hImageMem is
    // a CUarray or CUmipmappedArray. If this function returns successfully, we
    // check whether hImageMem is device memory (even managed memory isn't
    // considered shared).
    CUresult Err = cuPointerGetAttribute(
        &mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)hImageMem);
    if (Err != CUDA_SUCCESS) {
      // We have a CUarray
      if (pImageDesc->numMipLevel == 1) {
        image_res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
        image_res_desc.res.array.hArray = (CUarray)hImageMem;
      }
      // We have a CUmipmappedArray
      else {
        image_res_desc.resType = CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
        image_res_desc.res.mipmap.hMipmappedArray = (CUmipmappedArray)hImageMem;
      }
    } else if (mem_type == CU_MEMORYTYPE_DEVICE) {
      // We have a USM pointer
      if (pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        image_res_desc.resType = CU_RESOURCE_TYPE_LINEAR;
        image_res_desc.res.linear.devPtr = (CUdeviceptr)hImageMem;
        image_res_desc.res.linear.format = format;
        image_res_desc.res.linear.numChannels = NumChannels;
        image_res_desc.res.linear.sizeInBytes =
            pImageDesc->width * PixelSizeBytes;
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        image_res_desc.resType = CU_RESOURCE_TYPE_PITCH2D;
        image_res_desc.res.pitch2D.devPtr = (CUdeviceptr)hImageMem;
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

  auto as_CUArray = [](const void *ptr) {
    return static_cast<CUarray>(const_cast<void *>(ptr));
  };

  unsigned int NumChannels = 0;
  size_t PixelSizeBytes = 0;

  UR_CHECK_ERROR(
      urCalculateNumChannels(pSrcImageFormat->channelOrder, &NumChannels));

  // We need to get this now in bytes for calculating the total image size
  // later.
  UR_CHECK_ERROR(urToCudaImageChannelFormat(pSrcImageFormat->channelType,
                                            pSrcImageFormat->channelOrder,
                                            nullptr, &PixelSizeBytes, nullptr));

  try {
    ScopedContext Active(hQueue->getDevice());
    CUstream Stream = hQueue->getNextTransferStream();
    enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);

    // We have to use a different copy function for each image dimensionality.

    if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE) {
      if (pDstImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        CUmemorytype memType;

        // Check what type of memory is pDst. If cuPointerGetAttribute returns
        // somthing different from CUDA_SUCCESS then we know that pDst memory
        // type is a CuArray. Otherwise, it's CU_MEMORYTYPE_DEVICE.
        bool isCudaArray =
            cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                  (CUdeviceptr)pDst) != CUDA_SUCCESS;

        size_t CopyExtentBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        const char *SrcWithOffset = static_cast<const char *>(pSrc) +
                                    (pCopyRegion->srcOffset.x * PixelSizeBytes);

        if (isCudaArray) {
          UR_CHECK_ERROR(cuMemcpyHtoAAsync(
              (CUarray)pDst, pCopyRegion->dstOffset.x * PixelSizeBytes,
              static_cast<const void *>(SrcWithOffset), CopyExtentBytes,
              Stream));
        } else if (memType == CU_MEMORYTYPE_DEVICE) {
          void *DstWithOffset =
              static_cast<void *>(static_cast<char *>(pDst) +
                                  (PixelSizeBytes * pCopyRegion->dstOffset.x));
          UR_CHECK_ERROR(
              cuMemcpyHtoDAsync((CUdeviceptr)DstWithOffset,
                                static_cast<const void *>(SrcWithOffset),
                                CopyExtentBytes, Stream));
        } else {
          // This should be unreachable.
          return UR_RESULT_ERROR_INVALID_VALUE;
        }
      } else if (pDstImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        CUDA_MEMCPY2D cpy_desc = {};
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.srcHost = pSrc;
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.srcPitch = pSrcImageDesc->width * PixelSizeBytes;
        if (pDstImageDesc->rowPitch == 0) {
          cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
          cpy_desc.dstArray = (CUarray)pDst;
        } else {
          // Pitched memory
          cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
          cpy_desc.dstDevice = (CUdeviceptr)pDst;
          cpy_desc.dstPitch = pDstImageDesc->rowPitch;
        }
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        UR_CHECK_ERROR(cuMemcpy2DAsync(&cpy_desc, Stream));
      } else if (pDstImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        CUDA_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.srcHost = pSrc;
        cpy_desc.srcPitch = pSrcImageDesc->width * PixelSizeBytes;
        cpy_desc.srcHeight = pSrcImageDesc->height;
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.dstArray = (CUarray)pDst;
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        UR_CHECK_ERROR(cuMemcpy3DAsync(&cpy_desc, Stream));
      } else if (pDstImageDesc->type == UR_MEM_TYPE_IMAGE1D_ARRAY ||
                 pDstImageDesc->type == UR_MEM_TYPE_IMAGE2D_ARRAY ||
                 pDstImageDesc->type == UR_MEM_TYPE_IMAGE_CUBEMAP_EXP) {
        CUDA_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.srcHost = pSrc;
        cpy_desc.srcPitch = pSrcImageDesc->width * PixelSizeBytes;
        cpy_desc.srcHeight = std::max(uint64_t{1}, pSrcImageDesc->height);
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.dstArray = (CUarray)pDst;
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = std::max(uint64_t{1}, pCopyRegion->copyExtent.height);
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        UR_CHECK_ERROR(cuMemcpy3DAsync(&cpy_desc, Stream));
      }
    } else if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST) {
      if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        CUmemorytype memType;
        // Check what type of memory is pSrc. If cuPointerGetAttribute returns
        // somthing different from CUDA_SUCCESS then we know that pSrc memory
        // type is a CuArray. Otherwise, it's CU_MEMORYTYPE_DEVICE.
        bool isCudaArray =
            cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                  (CUdeviceptr)pSrc) != CUDA_SUCCESS;

        size_t CopyExtentBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        void *DstWithOffset =
            static_cast<void *>(static_cast<char *>(pDst) +
                                (PixelSizeBytes * pCopyRegion->dstOffset.x));

        if (isCudaArray) {
          UR_CHECK_ERROR(
              cuMemcpyAtoHAsync(DstWithOffset, as_CUArray(pSrc),
                                PixelSizeBytes * pCopyRegion->srcOffset.x,
                                CopyExtentBytes, Stream));
        } else if (memType == CU_MEMORYTYPE_DEVICE) {
          const char *SrcWithOffset =
              static_cast<const char *>(pSrc) +
              (pCopyRegion->srcOffset.x * PixelSizeBytes);
          UR_CHECK_ERROR(cuMemcpyDtoHAsync(DstWithOffset,
                                           (CUdeviceptr)SrcWithOffset,
                                           CopyExtentBytes, Stream));
        } else {
          // This should be unreachable.
          return UR_RESULT_ERROR_INVALID_VALUE;
        }
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        CUDA_MEMCPY2D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.dstHost = pDst;
        if (pSrcImageDesc->rowPitch == 0) {
          cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
          cpy_desc.srcArray = as_CUArray(pSrc);
        } else {
          // Pitched memory
          cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
          cpy_desc.srcPitch = pSrcImageDesc->rowPitch;
          cpy_desc.srcDevice = (CUdeviceptr)pSrc;
        }
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.dstHost = pDst;
        cpy_desc.dstPitch = pDstImageDesc->width * PixelSizeBytes;
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        UR_CHECK_ERROR(cuMemcpy2DAsync(&cpy_desc, Stream));
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        CUDA_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.srcArray = as_CUArray(pSrc);
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.dstHost = pDst;
        cpy_desc.dstPitch = pDstImageDesc->width * PixelSizeBytes;
        cpy_desc.dstHeight = pDstImageDesc->height;
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        UR_CHECK_ERROR(cuMemcpy3DAsync(&cpy_desc, Stream));
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE1D_ARRAY ||
                 pSrcImageDesc->type == UR_MEM_TYPE_IMAGE2D_ARRAY ||
                 pSrcImageDesc->type == UR_MEM_TYPE_IMAGE_CUBEMAP_EXP) {
        CUDA_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.srcArray = as_CUArray(pSrc);
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.dstHost = pDst;
        cpy_desc.dstPitch = pDstImageDesc->width * PixelSizeBytes;
        cpy_desc.dstHeight = std::max(uint64_t{1}, pDstImageDesc->height);
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = std::max(uint64_t{1}, pCopyRegion->copyExtent.height);
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        UR_CHECK_ERROR(cuMemcpy3DAsync(&cpy_desc, Stream));
      }
    } else {
      // imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE

      // we don't support copying between different image types.
      if (pSrcImageDesc->type != pDstImageDesc->type) {
        logger::error(
            "Unsupported copy operation between different type of images");
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
      }

      // All the following async copy function calls should be treated as
      // synchronous because of the explicit call to cuStreamSynchronize at
      // the end
      if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        CUDA_MEMCPY2D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = 0;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = 0;
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.srcArray = as_CUArray(pSrc);
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.dstArray = (CUarray)pDst;
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = 1;
        UR_CHECK_ERROR(cuMemcpy2DAsync(&cpy_desc, Stream));
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        CUDA_MEMCPY2D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.srcArray = as_CUArray(pSrc);
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.dstArray = (CUarray)pDst;
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        UR_CHECK_ERROR(cuMemcpy2DAsync(&cpy_desc, Stream));
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        CUDA_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.srcArray = as_CUArray(pSrc);
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.dstArray = (CUarray)pDst;
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = pCopyRegion->copyExtent.height;
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        UR_CHECK_ERROR(cuMemcpy3DAsync(&cpy_desc, Stream));
      } else if (pSrcImageDesc->type == UR_MEM_TYPE_IMAGE1D_ARRAY ||
                 pSrcImageDesc->type == UR_MEM_TYPE_IMAGE2D_ARRAY ||
                 pSrcImageDesc->type == UR_MEM_TYPE_IMAGE_CUBEMAP_EXP) {
        CUDA_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = pCopyRegion->srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = pCopyRegion->srcOffset.y;
        cpy_desc.srcZ = pCopyRegion->srcOffset.z;
        cpy_desc.dstXInBytes = pCopyRegion->dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = pCopyRegion->dstOffset.y;
        cpy_desc.dstZ = pCopyRegion->dstOffset.z;
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.srcArray = as_CUArray(pSrc);
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.dstArray = (CUarray)pDst;
        cpy_desc.WidthInBytes = PixelSizeBytes * pCopyRegion->copyExtent.width;
        cpy_desc.Height = std::max(uint64_t{1}, pCopyRegion->copyExtent.height);
        cpy_desc.Depth = pCopyRegion->copyExtent.depth;
        UR_CHECK_ERROR(cuMemcpy3DAsync(&cpy_desc, Stream));
      }
      // Synchronization is required here to handle the case of copying data
      // from host to device, then device to device and finally device to host.
      // Without it, there is a risk of the copies not being executed in the
      // intended order.
      cuStreamSynchronize(Stream);
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
    ur_context_handle_t, ur_exp_image_mem_native_handle_t hImageMem,
    ur_image_info_t propName, void *pPropValue, size_t *pPropSizeRet) {

  CUarray hCUarray;
  CUresult Err = cuMipmappedArrayGetLevel(
      &hCUarray, reinterpret_cast<CUmipmappedArray>(hImageMem), 0);

  // If cuMipmappedArrayGetLevel failed, hImageMem is already CUarray.
  if (Err != CUDA_SUCCESS) {
    hCUarray = reinterpret_cast<CUarray>(hImageMem);
  }

  CUDA_ARRAY3D_DESCRIPTOR ArrayDesc;
  UR_CHECK_ERROR(cuArray3DGetDescriptor(&ArrayDesc, hCUarray));
  switch (propName) {
  case UR_IMAGE_INFO_WIDTH:
    if (pPropValue) {
      *(size_t *)pPropValue = ArrayDesc.Width;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(size_t);
    }
    return UR_RESULT_SUCCESS;
  case UR_IMAGE_INFO_HEIGHT:
    if (pPropValue) {
      *(size_t *)pPropValue = ArrayDesc.Height;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(size_t);
    }
    return UR_RESULT_SUCCESS;
  case UR_IMAGE_INFO_DEPTH:
    if (pPropValue) {
      *(size_t *)pPropValue = ArrayDesc.Depth;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(size_t);
    }
    return UR_RESULT_SUCCESS;
  case UR_IMAGE_INFO_FORMAT:
    ur_image_channel_type_t ChannelType;
    ur_image_channel_order_t ChannelOrder;
    UR_CHECK_ERROR(cudaToUrImageChannelFormat(ArrayDesc.Format, &ChannelType));
    // CUDA does not have a notion of channel "order" in the same way that
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
      die("Unexpected NumChannels returned by CUDA");
    }
    if (pPropValue) {
      ((ur_image_format_t *)pPropValue)->channelType = ChannelType;
      ((ur_image_format_t *)pPropValue)->channelOrder = ChannelOrder;
    }
    if (pPropSizeRet) {
      *pPropSizeRet = sizeof(ur_image_format_t);
    }
    return UR_RESULT_SUCCESS;
  default:
    return UR_RESULT_ERROR_INVALID_VALUE;
  }
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
    ScopedContext Active(hDevice);
    CUarray ImageArray;
    UR_CHECK_ERROR(cuMipmappedArrayGetLevel(
        &ImageArray, (CUmipmappedArray)hImageMem, mipmapLevel));
    *phImageMem = (ur_exp_image_mem_native_handle_t)ImageArray;
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

  ScopedContext Active(hDevice);
  try {
    UR_CHECK_ERROR(cuMipmappedArrayDestroy((CUmipmappedArray)hMem));
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
    ScopedContext Active(hDevice);

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC extMemDesc = {};
    extMemDesc.size = size;

    void *pNext = const_cast<void *>(pExternalMemDesc->pNext);
    while (pNext != nullptr) {
      const ur_base_desc_t *BaseDesc =
          static_cast<const ur_base_desc_t *>(pNext);
      if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR) {
        auto FileDescriptor =
            static_cast<const ur_exp_file_descriptor_t *>(pNext);

        extMemDesc.handle.fd = FileDescriptor->fd;
        extMemDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
      } else if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE) {
        auto Win32Handle = static_cast<const ur_exp_win32_handle_t *>(pNext);

        switch (memHandleType) {
        case UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT:
          extMemDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
          break;
        case UR_EXP_EXTERNAL_MEM_TYPE_WIN32_NT_DX12_RESOURCE:
          extMemDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE;
          extMemDesc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED;
          break;
        case UR_EXP_EXTERNAL_MEM_TYPE_OPAQUE_FD:
        default:
          return UR_RESULT_ERROR_INVALID_VALUE;
        }
        extMemDesc.handle.win32.handle = Win32Handle->handle;
      }
      pNext = const_cast<void *>(BaseDesc->pNext);
    }

    CUexternalMemory extMem;
    UR_CHECK_ERROR(cuImportExternalMemory(&extMem, &extMemDesc));
    *phExternalMem = (ur_exp_external_mem_handle_t)extMem;

  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMapExternalArrayExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_external_mem_handle_t hExternalMem,
    ur_exp_image_mem_native_handle_t *phImageMem) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  unsigned int NumChannels = 0;
  UR_CHECK_ERROR(
      urCalculateNumChannels(pImageFormat->channelOrder, &NumChannels));

  CUarray_format format;
  UR_CHECK_ERROR(urToCudaImageChannelFormat(pImageFormat->channelType,
                                            pImageFormat->channelOrder, &format,
                                            nullptr, nullptr));

  try {
    ScopedContext Active(hDevice);

    CUDA_ARRAY3D_DESCRIPTOR ArrayDesc = {};
    ArrayDesc.Width = pImageDesc->width;
    ArrayDesc.Height = pImageDesc->height;
    ArrayDesc.Depth = pImageDesc->depth;
    ArrayDesc.NumChannels = NumChannels;
    ArrayDesc.Format = format;

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapDesc = {};
    mipmapDesc.numLevels = pImageDesc->numMipLevel;
    mipmapDesc.arrayDesc = ArrayDesc;

    // External memory is mapped to a CUmipmappedArray
    // If desired, a CUarray is retrieved from the mipmaps 0th level
    CUmipmappedArray memMipMap;
    UR_CHECK_ERROR(cuExternalMemoryGetMappedMipmappedArray(
        &memMipMap, (CUexternalMemory)hExternalMem, &mipmapDesc));

    if (pImageDesc->numMipLevel > 1) {
      *phImageMem = (ur_exp_image_mem_native_handle_t)memMipMap;
    } else {
      CUarray memArray;
      UR_CHECK_ERROR(cuMipmappedArrayGetLevel(&memArray, memMipMap, 0));

      hDevice->ChildCuarrayFromMipmapMap.emplace(memArray, memMipMap);

      *phImageMem = (ur_exp_image_mem_native_handle_t)memArray;
    }

  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMapExternalLinearMemoryExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, uint64_t offset,
    uint64_t size, ur_exp_external_mem_handle_t hExternalMem, void **ppRetMem) {
  UR_ASSERT(std::find(hContext->getDevices().begin(),
                      hContext->getDevices().end(),
                      hDevice) != hContext->getDevices().end(),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedContext Active(hDevice);

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC BufferDesc = {};
    BufferDesc.size = size;
    BufferDesc.offset = offset;
    BufferDesc.flags = 0;

    CUdeviceptr retMem;
    UR_CHECK_ERROR(cuExternalMemoryGetMappedBuffer(
        &retMem, (CUexternalMemory)hExternalMem, &BufferDesc));

    *ppRetMem = (void *)retMem;

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
    ScopedContext Active(hDevice);
    UR_CHECK_ERROR(cuDestroyExternalMemory((CUexternalMemory)hExternalMem));
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
    ScopedContext Active(hDevice);

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC extSemDesc = {};

    void *pNext = const_cast<void *>(pExternalSemaphoreDesc->pNext);
    while (pNext != nullptr) {
      const ur_base_desc_t *BaseDesc =
          static_cast<const ur_base_desc_t *>(pNext);
      if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR) {
        auto FileDescriptor =
            static_cast<const ur_exp_file_descriptor_t *>(pNext);

        extSemDesc.handle.fd = FileDescriptor->fd;
        extSemDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
      } else if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE) {
        auto Win32Handle = static_cast<const ur_exp_win32_handle_t *>(pNext);
        switch (semHandleType) {
        case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT:
          extSemDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32;
          break;
        case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_WIN32_NT_DX12_FENCE:
          extSemDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE;
          break;
        case UR_EXP_EXTERNAL_SEMAPHORE_TYPE_OPAQUE_FD:
        default:
          return UR_RESULT_ERROR_INVALID_VALUE;
        }
        extSemDesc.handle.win32.handle = Win32Handle->handle;
      }
      pNext = const_cast<void *>(BaseDesc->pNext);
    }

    CUexternalSemaphore semaphore;
    UR_CHECK_ERROR(cuImportExternalSemaphore(&semaphore, &extSemDesc));

    *phExternalSemaphoreHandle = (ur_exp_external_semaphore_handle_t)semaphore;
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
    ScopedContext Active(hDevice);
    UR_CHECK_ERROR(
        cuDestroyExternalSemaphore((CUexternalSemaphore)hExternalSemaphore));
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
    ScopedContext Active(hQueue->getDevice());
    CUstream Stream = hQueue->getNextTransferStream();

    enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);

    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS SemWaitParams = {};
    if (hasValue) {
      SemWaitParams.params.fence.value = waitValue;
    }

    // Wait for one external semaphore
    UR_CHECK_ERROR(cuWaitExternalSemaphoresAsync(
        (CUexternalSemaphore *)&hSemaphore, &SemWaitParams, 1 /* numExtSems */,
        Stream));

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
    ScopedContext Active(hQueue->getDevice());
    CUstream Stream = hQueue->getNextTransferStream();

    enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);

    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS SemSignalParams = {};
    if (hasValue) {
      SemSignalParams.params.fence.value = signalValue;
    }

    // Signal one external semaphore
    UR_CHECK_ERROR(cuSignalExternalSemaphoresAsync(
        (CUexternalSemaphore *)&hSemaphore, &SemSignalParams,
        1 /* numExtSems */, Stream));

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
