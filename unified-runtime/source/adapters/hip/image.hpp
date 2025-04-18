//===--------- image.hpp - HIP Adapter -----------------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <hip/hip_runtime.h>
#include <ur_api.h>

#include "common.hpp"
ur_result_t urCalculateNumChannels(ur_image_channel_order_t order,
                                   unsigned int *num_channels);

ur_result_t
urToHipImageChannelFormat(ur_image_channel_type_t image_channel_type,
                          ur_image_channel_order_t image_channel_order,
                          hipArray_Format *return_hip_format,
                          size_t *return_pixel_types_size_bytes);

ur_result_t
hipToUrImageChannelFormat(hipArray_Format hip_format,
                          ur_image_channel_type_t *return_image_channel_type);

ur_result_t urTextureCreate(ur_sampler_handle_t hSampler,
                            const ur_image_desc_t *pImageDesc,
                            const HIP_RESOURCE_DESC &ResourceDesc,
                            const unsigned int normalized_dtype_flag,
                            ur_exp_image_native_handle_t *phRetImage);
