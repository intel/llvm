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
    return UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED;
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
    return UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED;
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
ur_result_t
urToCudaImageChannelFormat(ur_image_channel_type_t image_channel_type,
                           ur_image_channel_order_t image_channel_order,
                           CUarray_format *return_cuda_format,
                           size_t *return_pixel_size_bytes) {

  CUarray_format cuda_format;
  size_t pixel_size_bytes = 0;
  unsigned int num_channels = 0;
  UR_CHECK_ERROR(urCalculateNumChannels(image_channel_order, &num_channels));

  switch (image_channel_type) {
#define CASE(FROM, TO, SIZE)                                                   \
  case FROM: {                                                                 \
    cuda_format = TO;                                                          \
    pixel_size_bytes = SIZE * num_channels;                                    \
    break;                                                                     \
  }

    CASE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8, CU_AD_FORMAT_UNSIGNED_INT8, 1)
    CASE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT8, CU_AD_FORMAT_SIGNED_INT8, 1)
    CASE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16, CU_AD_FORMAT_UNSIGNED_INT16, 2)
    CASE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT16, CU_AD_FORMAT_SIGNED_INT16, 2)
    CASE(UR_IMAGE_CHANNEL_TYPE_HALF_FLOAT, CU_AD_FORMAT_HALF, 2)
    CASE(UR_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32, CU_AD_FORMAT_UNSIGNED_INT32, 4)
    CASE(UR_IMAGE_CHANNEL_TYPE_SIGNED_INT32, CU_AD_FORMAT_SIGNED_INT32, 4)
    CASE(UR_IMAGE_CHANNEL_TYPE_FLOAT, CU_AD_FORMAT_FLOAT, 4)

#undef CASE
  default:
    break;
  }

  // These new formats were brought in in CUDA 11.5
#if CUDA_VERSION >= 11050

  // If none of the above channel types were passed, check those below
  if (pixel_size_bytes == 0) {

    // We can't use a switch statement here because these single
    // UR_IMAGE_CHANNEL_TYPEs can correspond to multiple [u/s]norm CU_AD_FORMATs
    // depending on the number of channels. We use a std::map instead to
    // retrieve the correct CUDA format

    // map < <channel type, num channels> , <CUDA format, data type byte size> >
    const std::map<std::pair<ur_image_channel_type_t, uint32_t>,
                   std::pair<CUarray_format, uint32_t>>
        norm_channel_type_map{
            {{UR_IMAGE_CHANNEL_TYPE_UNORM_INT8, 1},
             {CU_AD_FORMAT_UNORM_INT8X1, 1}},
            {{UR_IMAGE_CHANNEL_TYPE_UNORM_INT8, 2},
             {CU_AD_FORMAT_UNORM_INT8X2, 2}},
            {{UR_IMAGE_CHANNEL_TYPE_UNORM_INT8, 4},
             {CU_AD_FORMAT_UNORM_INT8X4, 4}},

            {{UR_IMAGE_CHANNEL_TYPE_SNORM_INT8, 1},
             {CU_AD_FORMAT_SNORM_INT8X1, 1}},
            {{UR_IMAGE_CHANNEL_TYPE_SNORM_INT8, 2},
             {CU_AD_FORMAT_SNORM_INT8X2, 2}},
            {{UR_IMAGE_CHANNEL_TYPE_SNORM_INT8, 4},
             {CU_AD_FORMAT_SNORM_INT8X4, 4}},

            {{UR_IMAGE_CHANNEL_TYPE_UNORM_INT16, 1},
             {CU_AD_FORMAT_UNORM_INT16X1, 2}},
            {{UR_IMAGE_CHANNEL_TYPE_UNORM_INT16, 2},
             {CU_AD_FORMAT_UNORM_INT16X2, 4}},
            {{UR_IMAGE_CHANNEL_TYPE_UNORM_INT16, 4},
             {CU_AD_FORMAT_UNORM_INT16X4, 8}},

            {{UR_IMAGE_CHANNEL_TYPE_SNORM_INT16, 1},
             {CU_AD_FORMAT_SNORM_INT16X1, 2}},
            {{UR_IMAGE_CHANNEL_TYPE_SNORM_INT16, 2},
             {CU_AD_FORMAT_SNORM_INT16X2, 4}},
            {{UR_IMAGE_CHANNEL_TYPE_SNORM_INT16, 4},
             {CU_AD_FORMAT_SNORM_INT16X4, 8}},
        };

    try {
      auto cuda_format_and_size = norm_channel_type_map.at(
          std::make_pair(image_channel_type, num_channels));
      cuda_format = cuda_format_and_size.first;
      pixel_size_bytes = cuda_format_and_size.second;
    } catch (const std::out_of_range &) {
      return UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED;
    }
  }

#endif

  if (return_cuda_format) {
    *return_cuda_format = cuda_format;
  }
  if (return_pixel_size_bytes) {
    *return_pixel_size_bytes = pixel_size_bytes;
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
#if CUDA_VERSION >= 11050

    // Note that the CUDA UNORM and SNORM formats also encode the number of
    // channels.
    // Since UR does not encode this, we map different CUDA formats to the same
    // UR channel type.
    // Since this function is only called from `urBindlessImagesImageGetInfoExp`
    // which has access to `CUDA_ARRAY3D_DESCRIPTOR`, we can determine the
    // number of channels in the calling function.

    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_UNORM_INT8X1,
                                  UR_IMAGE_CHANNEL_TYPE_UNORM_INT8);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_UNORM_INT8X2,
                                  UR_IMAGE_CHANNEL_TYPE_UNORM_INT8);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_UNORM_INT8X4,
                                  UR_IMAGE_CHANNEL_TYPE_UNORM_INT8);

    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_UNORM_INT16X1,
                                  UR_IMAGE_CHANNEL_TYPE_UNORM_INT16);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_UNORM_INT16X2,
                                  UR_IMAGE_CHANNEL_TYPE_UNORM_INT16);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_UNORM_INT16X4,
                                  UR_IMAGE_CHANNEL_TYPE_UNORM_INT16);

    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_SNORM_INT8X1,
                                  UR_IMAGE_CHANNEL_TYPE_SNORM_INT8);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_SNORM_INT8X2,
                                  UR_IMAGE_CHANNEL_TYPE_SNORM_INT8);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_SNORM_INT8X4,
                                  UR_IMAGE_CHANNEL_TYPE_SNORM_INT8);

    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_SNORM_INT16X1,
                                  UR_IMAGE_CHANNEL_TYPE_SNORM_INT16);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_SNORM_INT16X2,
                                  UR_IMAGE_CHANNEL_TYPE_SNORM_INT16);
    CUDA_TO_UR_IMAGE_CHANNEL_TYPE(CU_AD_FORMAT_SNORM_INT16X4,
                                  UR_IMAGE_CHANNEL_TYPE_SNORM_INT16);
#endif
#undef MAP
  default:
    return UR_RESULT_ERROR_IMAGE_FORMAT_NOT_SUPPORTED;
  }
}

ur_result_t urTextureCreate(ur_sampler_handle_t hSampler,
                            const ur_image_desc_t *pImageDesc,
                            const CUDA_RESOURCE_DESC &ResourceDesc,
                            ur_exp_image_handle_t *phRetImage) {

  try {
    /// pi_sampler_properties
    /// Layout of UR samplers for CUDA
    ///
    /// Sampler property layout:
    /// |     <bits>     | <usage>
    /// -----------------------------------
    /// |  31 30 ... 12  | N/A
    /// |       11       | mip filter mode
    /// |    10 9 8      | addressing mode 3
    /// |     7 6 5      | addressing mode 2
    /// |     4 3 2      | addressing mode 1
    /// |       1        | filter mode
    /// |       0        | normalize coords
    CUDA_TEXTURE_DESC ImageTexDesc = {};
    CUaddress_mode AddrMode[3];
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
    ImageTexDesc.flags |= CU_TRSF_READ_AS_INTEGER;

    CUtexObject Texture;
    UR_CHECK_ERROR(
        cuTexObjectCreate(&Texture, &ResourceDesc, &ImageTexDesc, nullptr));
    *phRetImage = (ur_exp_image_handle_t)Texture;
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
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
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
    ScopedContext Active(hDevice->getContext());
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
urBindlessImagesUnsampledImageHandleDestroyExp(ur_context_handle_t hContext,
                                               ur_device_handle_t hDevice,
                                               ur_exp_image_handle_t hImage) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  UR_CHECK_ERROR(cuSurfObjectDestroy((CUsurfObject)hImage));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesSampledImageHandleDestroyExp(ur_context_handle_t hContext,
                                             ur_device_handle_t hDevice,
                                             ur_exp_image_handle_t hImage) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  UR_CHECK_ERROR(cuTexObjectDestroy((CUtexObject)hImage));
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageAllocateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_mem_handle_t *phImageMem) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  // Populate descriptor
  CUDA_ARRAY3D_DESCRIPTOR array_desc = {};

  UR_CHECK_ERROR(urCalculateNumChannels(pImageFormat->channelOrder,
                                        &array_desc.NumChannels));

  UR_CHECK_ERROR(urToCudaImageChannelFormat(pImageFormat->channelType,
                                            pImageFormat->channelOrder,
                                            &array_desc.Format, nullptr));

  array_desc.Flags = 0; // No flags required
  array_desc.Width = pImageDesc->width;
  if (pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
    array_desc.Height = 0;
    array_desc.Depth = 0;
  } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
    array_desc.Height = pImageDesc->height;
    array_desc.Depth = 0;
  } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
    array_desc.Height = pImageDesc->height;
    array_desc.Depth = pImageDesc->depth;
  }

  ScopedContext Active(hDevice->getContext());

  // Allocate a cuArray
  if (pImageDesc->numMipLevel == 1) {
    CUarray ImageArray;

    try {
      UR_CHECK_ERROR(cuArray3DCreate(&ImageArray, &array_desc));
      *phImageMem = (ur_exp_image_mem_handle_t)ImageArray;
    } catch (ur_result_t Err) {
      cuArrayDestroy(ImageArray);
      return Err;
    } catch (...) {
      cuArrayDestroy(ImageArray);
      return UR_RESULT_ERROR_UNKNOWN;
    }
  } else // Allocate a cuMipmappedArray
  {
    CUmipmappedArray mip_array;
    array_desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;

    try {
      UR_CHECK_ERROR(cuMipmappedArrayCreate(&mip_array, &array_desc,
                                            pImageDesc->numMipLevel));
      *phImageMem = (ur_exp_image_mem_handle_t)mip_array;
    } catch (ur_result_t Err) {
      cuMipmappedArrayDestroy(mip_array);
      return Err;
    } catch (...) {
      cuMipmappedArrayDestroy(mip_array);
      return UR_RESULT_ERROR_UNKNOWN;
    }
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageFreeExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  ScopedContext Active(hDevice->getContext());
  try {
    UR_CHECK_ERROR(cuArrayDestroy((CUarray)hImageMem));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesUnsampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem, const ur_image_format_t *pImageFormat,
    const ur_image_desc_t *pImageDesc, ur_mem_handle_t *phMem,
    ur_exp_image_handle_t *phImage) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  unsigned int NumChannels = 0;
  UR_CHECK_ERROR(
      urCalculateNumChannels(pImageFormat->channelOrder, &NumChannels));

  CUarray_format format;
  size_t PixelSizeBytes;
  UR_CHECK_ERROR(urToCudaImageChannelFormat(pImageFormat->channelType,
                                            pImageFormat->channelOrder, &format,
                                            &PixelSizeBytes));

  try {

    ScopedContext Active(hDevice->getContext());

    CUDA_RESOURCE_DESC image_res_desc = {};

    // We have a CUarray
    image_res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    image_res_desc.res.array.hArray = (CUarray)hImageMem;

    // We create surfaces in the unsampled images case as it conforms to how
    // CUDA deals with unsampled images.
    CUsurfObject surface;
    UR_CHECK_ERROR(cuSurfObjectCreate(&surface, &image_res_desc));
    *phImage = (ur_exp_image_handle_t)surface;

    auto urMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
        hContext, (CUarray)hImageMem, surface, pImageDesc->type});

    if (urMemObj == nullptr) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    *phMem = urMemObj.release();

  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesSampledImageCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hImageMem, const ur_image_format_t *pImageFormat,
    const ur_image_desc_t *pImageDesc, ur_sampler_handle_t hSampler,
    ur_mem_handle_t *phMem, ur_exp_image_handle_t *phImage) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  ScopedContext Active(hDevice->getContext());

  unsigned int NumChannels = 0;
  UR_CHECK_ERROR(
      urCalculateNumChannels(pImageFormat->channelOrder, &NumChannels));

  CUarray_format format;
  size_t PixelSizeBytes;
  UR_CHECK_ERROR(urToCudaImageChannelFormat(pImageFormat->channelType,
                                            pImageFormat->channelOrder, &format,
                                            &PixelSizeBytes));

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

    UR_CHECK_ERROR(
        urTextureCreate(hSampler, pImageDesc, image_res_desc, phImage));

    auto urMemObj = std::unique_ptr<ur_mem_handle_t_>(new ur_mem_handle_t_{
        hContext, (CUarray)hImageMem, (CUtexObject)*phImage, hSampler,
        pImageDesc->type});

    if (urMemObj == nullptr) {
      return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
    }

    *phMem = urMemObj.release();
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImageCopyExp(
    ur_queue_handle_t hQueue, void *pDst, void *pSrc,
    const ur_image_format_t *pImageFormat, const ur_image_desc_t *pImageDesc,
    ur_exp_image_copy_flags_t imageCopyFlags, ur_rect_offset_t srcOffset,
    ur_rect_offset_t dstOffset, ur_rect_region_t copyExtent,
    ur_rect_region_t hostExtent, uint32_t numEventsInWaitList,
    const ur_event_handle_t *phEventWaitList, ur_event_handle_t *phEvent) {
  UR_ASSERT((imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE ||
             imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST ||
             imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE),
            UR_RESULT_ERROR_INVALID_VALUE);

  unsigned int NumChannels = 0;
  size_t PixelSizeBytes = 0;

  UR_CHECK_ERROR(
      urCalculateNumChannels(pImageFormat->channelOrder, &NumChannels));

  // We need to get this now in bytes for calculating the total image size
  // later.
  UR_CHECK_ERROR(urToCudaImageChannelFormat(pImageFormat->channelType,
                                            pImageFormat->channelOrder, nullptr,
                                            &PixelSizeBytes));

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream Stream = hQueue->getNextTransferStream();
    enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);
    // We have to use a different copy function for each image dimensionality.

    if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_HOST_TO_DEVICE) {
      if (pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        size_t CopyExtentBytes = PixelSizeBytes * copyExtent.width;
        char *SrcWithOffset = (char *)pSrc + (srcOffset.x * PixelSizeBytes);
        UR_CHECK_ERROR(
            cuMemcpyHtoAAsync((CUarray)pDst, dstOffset.x * PixelSizeBytes,
                              (void *)SrcWithOffset, CopyExtentBytes, Stream));
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        CUDA_MEMCPY2D cpy_desc = {};
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.srcHost = pSrc;
        cpy_desc.srcXInBytes = srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = srcOffset.y;
        cpy_desc.dstXInBytes = dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = dstOffset.y;
        cpy_desc.srcPitch = hostExtent.width * PixelSizeBytes;
        if (pImageDesc->rowPitch == 0) {
          cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
          cpy_desc.dstArray = (CUarray)pDst;
        } else {
          // Pitched memory
          cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
          cpy_desc.dstDevice = (CUdeviceptr)pDst;
          cpy_desc.dstPitch = pImageDesc->rowPitch;
        }
        cpy_desc.WidthInBytes = PixelSizeBytes * copyExtent.width;
        cpy_desc.Height = copyExtent.height;
        UR_CHECK_ERROR(cuMemcpy2DAsync(&cpy_desc, Stream));
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        CUDA_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = srcOffset.x * PixelSizeBytes;
        cpy_desc.srcY = srcOffset.y;
        cpy_desc.srcZ = srcOffset.z;
        cpy_desc.dstXInBytes = dstOffset.x * PixelSizeBytes;
        cpy_desc.dstY = dstOffset.y;
        cpy_desc.dstZ = dstOffset.z;
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.srcHost = pSrc;
        cpy_desc.srcPitch = hostExtent.width * PixelSizeBytes;
        cpy_desc.srcHeight = hostExtent.height;
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.dstArray = (CUarray)pDst;
        cpy_desc.WidthInBytes = PixelSizeBytes * copyExtent.width;
        cpy_desc.Height = copyExtent.height;
        cpy_desc.Depth = copyExtent.depth;
        UR_CHECK_ERROR(cuMemcpy3DAsync(&cpy_desc, Stream));
      }
    } else if (imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_HOST) {
      if (pImageDesc->type == UR_MEM_TYPE_IMAGE1D) {
        size_t CopyExtentBytes = PixelSizeBytes * copyExtent.width;
        size_t src_offset_bytes = PixelSizeBytes * srcOffset.x;
        void *dst_with_offset =
            (void *)((char *)pDst + (PixelSizeBytes * dstOffset.x));
        UR_CHECK_ERROR(cuMemcpyAtoHAsync(dst_with_offset, (CUarray)pSrc,
                                         src_offset_bytes, CopyExtentBytes,
                                         Stream));
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE2D) {
        CUDA_MEMCPY2D cpy_desc = {};
        cpy_desc.srcXInBytes = srcOffset.x;
        cpy_desc.srcY = srcOffset.y;
        cpy_desc.dstXInBytes = dstOffset.x;
        cpy_desc.dstY = dstOffset.y;
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.dstHost = pDst;
        if (pImageDesc->rowPitch == 0) {
          cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
          cpy_desc.srcArray = (CUarray)pSrc;
        } else {
          // Pitched memory
          cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_DEVICE;
          cpy_desc.srcPitch = pImageDesc->rowPitch;
          cpy_desc.srcDevice = (CUdeviceptr)pSrc;
        }
        cpy_desc.WidthInBytes = PixelSizeBytes * copyExtent.width;
        cpy_desc.Height = copyExtent.height;
        UR_CHECK_ERROR(cuMemcpy2DAsync(&cpy_desc, Stream));
      } else if (pImageDesc->type == UR_MEM_TYPE_IMAGE3D) {
        CUDA_MEMCPY3D cpy_desc = {};
        cpy_desc.srcXInBytes = srcOffset.x;
        cpy_desc.srcY = srcOffset.y;
        cpy_desc.srcZ = srcOffset.z;
        cpy_desc.dstXInBytes = dstOffset.x;
        cpy_desc.dstY = dstOffset.y;
        cpy_desc.dstZ = dstOffset.z;
        cpy_desc.srcMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_ARRAY;
        cpy_desc.srcArray = (CUarray)pSrc;
        cpy_desc.dstMemoryType = CUmemorytype_enum::CU_MEMORYTYPE_HOST;
        cpy_desc.dstHost = pDst;
        cpy_desc.WidthInBytes = PixelSizeBytes * copyExtent.width;
        cpy_desc.Height = copyExtent.height;
        cpy_desc.Depth = copyExtent.depth;
        UR_CHECK_ERROR(cuMemcpy3DAsync(&cpy_desc, Stream));
      }
    } else {
      /// imageCopyFlags == UR_EXP_IMAGE_COPY_FLAG_DEVICE_TO_DEVICE
      /// TODO: implemet device to device copy
      return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
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
    ur_exp_image_mem_handle_t hImageMem, ur_image_info_t propName,
    void *pPropValue, size_t *pPropSizeRet) {

  CUDA_ARRAY3D_DESCRIPTOR ArrayDesc;
  UR_CHECK_ERROR(cuArray3DGetDescriptor(&ArrayDesc, (CUarray)hImageMem));
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
    ur_exp_image_mem_handle_t hImageMem, uint32_t mipmapLevel,
    ur_exp_image_mem_handle_t *phImageMem) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedContext Active(hDevice->getContext());
    CUarray ImageArray;
    UR_CHECK_ERROR(cuMipmappedArrayGetLevel(
        &ImageArray, (CUmipmappedArray)hImageMem, mipmapLevel));
    *phImageMem = (ur_exp_image_mem_handle_t)ImageArray;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesMipmapFreeExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_image_mem_handle_t hMem) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  ScopedContext Active(hDevice->getContext());
  try {
    UR_CHECK_ERROR(cuMipmappedArrayDestroy((CUmipmappedArray)hMem));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesImportOpaqueFDExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice, size_t size,
    ur_exp_interop_mem_desc_t *pInteropMemDesc,
    ur_exp_interop_mem_handle_t *phInteropMem) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedContext Active(hDevice->getContext());

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC extMemDesc = {};
    extMemDesc.size = size;

    void *pNext = const_cast<void *>(pInteropMemDesc->pNext);
    while (pNext != nullptr) {
      const ur_base_desc_t *BaseDesc =
          reinterpret_cast<const ur_base_desc_t *>(pNext);
      if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR) {
        const ur_exp_file_descriptor_t *FileDescriptor =
            reinterpret_cast<const ur_exp_file_descriptor_t *>(pNext);

        extMemDesc.handle.fd = FileDescriptor->fd;
        extMemDesc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
      } else if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
      }
      pNext = const_cast<void *>(BaseDesc->pNext);
    }

    CUexternalMemory extMem;
    UR_CHECK_ERROR(cuImportExternalMemory(&extMem, &extMemDesc));
    *phInteropMem = (ur_exp_interop_mem_handle_t)extMem;

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
    ur_exp_interop_mem_handle_t hInteropMem,
    ur_exp_image_mem_handle_t *phImageMem) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  unsigned int NumChannels = 0;
  UR_CHECK_ERROR(
      urCalculateNumChannels(pImageFormat->channelOrder, &NumChannels));

  CUarray_format format;
  UR_CHECK_ERROR(urToCudaImageChannelFormat(
      pImageFormat->channelType, pImageFormat->channelOrder, &format, nullptr));

  try {
    ScopedContext Active(hDevice->getContext());

    CUDA_ARRAY3D_DESCRIPTOR ArrayDesc = {};
    ArrayDesc.Width = pImageDesc->width;
    ArrayDesc.Height = pImageDesc->height;
    ArrayDesc.Depth = pImageDesc->depth;
    ArrayDesc.NumChannels = NumChannels;
    ArrayDesc.Format = format;

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapDesc = {};
    mipmapDesc.numLevels = 1;
    mipmapDesc.arrayDesc = ArrayDesc;

    CUmipmappedArray memMipMap;
    UR_CHECK_ERROR(cuExternalMemoryGetMappedMipmappedArray(
        &memMipMap, (CUexternalMemory)hInteropMem, &mipmapDesc));

    CUarray memArray;
    UR_CHECK_ERROR(cuMipmappedArrayGetLevel(&memArray, memMipMap, 0));

    *phImageMem = (ur_exp_image_mem_handle_t)memArray;

  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesReleaseInteropExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_interop_mem_handle_t hInteropMem) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedContext Active(hDevice->getContext());
    UR_CHECK_ERROR(cuDestroyExternalMemory((CUexternalMemory)hInteropMem));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urBindlessImagesImportExternalSemaphoreOpaqueFDExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_interop_semaphore_desc_t *pInteropSemaphoreDesc,
    ur_exp_interop_semaphore_handle_t *phInteropSemaphoreHandle) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedContext Active(hDevice->getContext());

    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC extSemDesc = {};

    void *pNext = const_cast<void *>(pInteropSemaphoreDesc->pNext);
    while (pNext != nullptr) {
      const ur_base_desc_t *BaseDesc =
          reinterpret_cast<const ur_base_desc_t *>(pNext);
      if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_FILE_DESCRIPTOR) {
        const ur_exp_file_descriptor_t *FileDescriptor =
            reinterpret_cast<const ur_exp_file_descriptor_t *>(pNext);

        extSemDesc.handle.fd = FileDescriptor->fd;
        extSemDesc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
      } else if (BaseDesc->stype == UR_STRUCTURE_TYPE_EXP_WIN32_HANDLE) {
        return UR_RESULT_ERROR_UNSUPPORTED_FEATURE;
      }
      pNext = const_cast<void *>(BaseDesc->pNext);
    }

    CUexternalSemaphore semaphore;
    UR_CHECK_ERROR(cuImportExternalSemaphore(&semaphore, &extSemDesc));

    *phInteropSemaphoreHandle = (ur_exp_interop_semaphore_handle_t)semaphore;
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesDestroyExternalSemaphoreExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    ur_exp_interop_semaphore_handle_t hInteropSemaphore) {
  UR_ASSERT((hContext->getDevice()->get() == hDevice->get()),
            UR_RESULT_ERROR_INVALID_CONTEXT);

  try {
    ScopedContext Active(hDevice->getContext());
    UR_CHECK_ERROR(
        cuDestroyExternalSemaphore((CUexternalSemaphore)hInteropSemaphore));
  } catch (ur_result_t Err) {
    return Err;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urBindlessImagesWaitExternalSemaphoreExp(
    ur_queue_handle_t hQueue, ur_exp_interop_semaphore_handle_t hSemaphore,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream Stream = hQueue->getNextTransferStream();

    enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);

    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS SemWaitParams = {};

    // Wait for one external semaphore
    UR_CHECK_ERROR(cuWaitExternalSemaphoresAsync(
        (CUexternalSemaphore *)&hSemaphore, &SemWaitParams, 1 /* numExtSems */,
        Stream));

    if (phEvent) {
      auto NewEvent = ur_event_handle_t_::makeNative(
          UR_COMMAND_INTEROP_SEMAPHORE_WAIT_EXP, hQueue, Stream);
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
    ur_queue_handle_t hQueue, ur_exp_interop_semaphore_handle_t hSemaphore,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {

  try {
    ScopedContext Active(hQueue->getContext());
    CUstream Stream = hQueue->getNextTransferStream();

    enqueueEventsWait(hQueue, Stream, numEventsInWaitList, phEventWaitList);

    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS SemSignalParams = {};

    // Signal one external semaphore
    UR_CHECK_ERROR(cuSignalExternalSemaphoresAsync(
        (CUexternalSemaphore *)&hSemaphore, &SemSignalParams,
        1 /* numExtSems */, Stream));

    if (phEvent) {
      auto NewEvent = ur_event_handle_t_::makeNative(
          UR_COMMAND_INTEROP_SEMAPHORE_SIGNAL_EXP, hQueue, Stream);
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
