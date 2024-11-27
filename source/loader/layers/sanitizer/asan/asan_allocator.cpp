/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_allocator.cpp
 *
 */

#include "asan_allocator.hpp"
#include "ur_sanitizer_layer.hpp"

namespace ur_sanitizer_layer {
namespace asan {

void AllocInfo::print() {
    getContext()->logger.info(
        "AllocInfo(Alloc=[{}-{}),  User=[{}-{}), AllocSize={}, Type={})",
        (void *)AllocBegin, (void *)(AllocBegin + AllocSize), (void *)UserBegin,
        (void *)(UserEnd), AllocSize, ToString(Type));
}

} // namespace asan
} // namespace ur_sanitizer_layer
