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

std::vector<AllocationIterator> Quarantine::put(ur_device_handle_t Device,
                                                AllocationIterator &It) {
    auto &AI = It->second;
    auto AllocSize = AI->AllocSize;
    auto &Cache = m_Map[Device];
    std::vector<AllocationIterator> DequeueList;
    while (Cache.Size() + AllocSize > m_MaxQuarantineSize) {
        auto ItOp = Cache.Dequeue();
        if (!ItOp) {
            break;
        }
        DequeueList.emplace_back(*ItOp);
    }
    m_Map[Device].Enqueue(It);
    return DequeueList;
}

} // namespace ur_sanitizer_layer
