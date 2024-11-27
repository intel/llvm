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

#include "sanitizer_common/sanitizer_allocator.hpp"
#include "sanitizer_common/sanitizer_common.hpp"
#include "sanitizer_common/sanitizer_stacktrace.hpp"

namespace ur_sanitizer_layer {
namespace asan {

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

} // namespace asan
} // namespace ur_sanitizer_layer
