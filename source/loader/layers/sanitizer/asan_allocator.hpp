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

#include <map>
#include <memory>

namespace ur_sanitizer_layer {

enum class AllocType : uint32_t {
    UNKNOWN,
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

    AllocType Type = AllocType::UNKNOWN;
    bool IsReleased = false;

    ur_context_handle_t Context = nullptr;
    ur_device_handle_t Device = nullptr;

    StackTrace AllocStack;
    StackTrace ReleaseStack;

    void print();
    size_t getRedzoneSize() { return AllocSize - (UserEnd - UserBegin); }
};

using AllocationMap = std::map<uptr, std::shared_ptr<AllocInfo>>;
using AllocationIterator = AllocationMap::iterator;

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
