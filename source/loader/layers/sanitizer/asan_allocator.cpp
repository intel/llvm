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

#pragma once

#include "asan_allocator.hpp"

namespace ur_sanitizer_layer {

const char *getFormatString(MemoryType MemoryType) {
    switch (MemoryType) {
    case MemoryType::DEVICE_USM:
        return "USM Device Memory";
    case MemoryType::HOST_USM:
        return "USM Host Memory";
    case MemoryType::SHARED_USM:
        return "USM Shared Memory";
    case MemoryType::MEM_BUFFER:
        return "Memory Buffer";
    default:
        return "Unknown Memory";
    }
}

} // namespace ur_sanitizer_layer
