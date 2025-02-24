//===--------- image.hpp - CUDA Adapter -----------------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
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
                           size_t *return_pixel_types_size_bytes,
                           unsigned int *return_normalized_dtype_flag);

ur_result_t
cudaToUrImageChannelFormat(CUarray_format cuda_format,
                           ur_image_channel_type_t *return_image_channel_type);

ur_result_t urTextureCreate(ur_sampler_handle_t hSampler,
                            const ur_image_desc_t *pImageDesc,
                            const CUDA_RESOURCE_DESC &ResourceDesc,
                            const unsigned int normalized_dtype_flag,
                            ur_exp_image_native_handle_t *phRetImage);
