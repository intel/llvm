/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_ddi.hpp
 *
 */

#include "ur_ddi.h"

namespace ur_sanitizer_layer {

void initAsanInterceptor();
void destroyAsanInterceptor();

} // namespace ur_sanitizer_layer
