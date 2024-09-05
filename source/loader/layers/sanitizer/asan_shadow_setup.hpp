/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_shadow_setup.hpp
 *
 */

#pragma once

#include "common.hpp"

namespace ur_sanitizer_layer {

ur_result_t SetupShadowMemoryOnCPU(uptr &ShadowBegin, uptr &ShadowEnd);
ur_result_t DestroyShadowMemoryOnCPU();

ur_result_t SetupShadowMemoryOnPVC(ur_context_handle_t Context,
                                   uptr &ShadowBegin, uptr &ShadowEnd);
ur_result_t DestroyShadowMemoryOnPVC();

ur_result_t SetupShadowMemoryOnDG2(ur_context_handle_t Context,
                                   uptr &ShadowBegin, uptr &ShadowEnd);
ur_result_t DestroyShadowMemoryOnDG2();

} // namespace ur_sanitizer_layer
