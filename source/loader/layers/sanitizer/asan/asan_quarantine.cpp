/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_quarantine.cpp
 *
 */

#include "asan_quarantine.hpp"

namespace ur_sanitizer_layer {
namespace asan {

std::vector<AllocationIterator> Quarantine::put(ur_device_handle_t Device,
                                                AllocationIterator &It) {
    auto &AI = It->second;
    auto AllocSize = AI->AllocSize;
    auto &Cache = getCache(Device);

    std::vector<AllocationIterator> DequeueList;
    std::scoped_lock<ur_mutex> Guard(Cache.Mutex);
    while (Cache.size() + AllocSize > m_MaxQuarantineSize) {
        auto ElementOp = Cache.dequeue();
        if (!ElementOp) {
            break;
        }
        DequeueList.emplace_back(*ElementOp);
    }
    Cache.enqueue(It);
    return DequeueList;
}

} // namespace asan
} // namespace ur_sanitizer_layer
