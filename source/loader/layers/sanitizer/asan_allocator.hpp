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

enum class AllocType : uint32_t {
    DEVICE_USM,
    SHARED_USM,
    HOST_USM,
    MEM_BUFFER,
    DEVICE_GLOBAL
};

struct AllocInfo {
    uptr AllocBegin = 0;
    uptr UserBegin = 0;
    uptr UserEnd = 0;
    size_t AllocSize = 0;

    AllocType Type = AllocType::DEVICE_USM;
    bool IsReleased = false;

    ur_context_handle_t Context = nullptr;
    ur_device_handle_t Device = nullptr;

    StackTrace AllocStack;
    StackTrace ReleaseStack;
};

inline const char *ToString(AllocType Type) {
    switch (Type) {
    case AllocType::DEVICE_USM:
        return "Device USM";
    case AllocType::HOST_USM:
        return "Host USM";
    case AllocType::SHARED_USM:
        return "Shared USM";
    case AllocType::MEM_BUFFER:
        return "Memory Buffer";
    case AllocType::DEVICE_GLOBAL:
        return "Device Global";
    default:
        return "Unknown Type";
    }
}

} // namespace ur_sanitizer_layer
