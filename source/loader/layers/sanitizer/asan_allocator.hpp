/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_allocator.hpp
 *
 */

#pragma once

#include "common.hpp"
#include "stacktrace.hpp"

namespace ur_sanitizer_layer {

enum MemoryType { DEVICE_USM, SHARED_USM, HOST_USM, MEM_BUFFER };

struct USMAllocInfo {
    uptr AllocBegin = 0;
    uptr UserBegin = 0;
    uptr UserEnd = 0;
    size_t AllocSize = 0;

    MemoryType Type = MemoryType::DEVICE_USM;
    bool IsReleased = false;

    ur_context_handle_t Context = nullptr;
    ur_device_handle_t Device = nullptr;

    StackTrace AllocStack;
    StackTrace ReleaseStack;
};

const char *getFormatString(MemoryType MemoryType);

} // namespace ur_sanitizer_layer
