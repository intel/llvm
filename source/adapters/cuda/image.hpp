//===--------- image.hpp - CUDA Adapter -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <cuda.h>
#include <ur_api.h>

#include "common.hpp"
ur_result_t urCalculateNumChannels(ur_image_channel_order_t order,
                                   unsigned int *num_channels);

ur_result_t
urToCudaImageChannelFormat(ur_image_channel_type_t image_channel_type,
                           ur_image_channel_order_t image_channel_order,
                           CUarray_format *return_cuda_format,
                           size_t *return_pixel_types_size_bytes);

ur_result_t
cudaToUrImageChannelFormat(CUarray_format cuda_format,
                           ur_image_channel_type_t *return_image_channel_type);

ur_result_t urTextureCreate(ur_context_handle_t hContext,
                            ur_sampler_desc_t SamplerDesc,
                            const ur_image_desc_t *pImageDesc,
                            CUDA_RESOURCE_DESC ResourceDesc,
                            ur_exp_image_handle_t *phRetImage);
