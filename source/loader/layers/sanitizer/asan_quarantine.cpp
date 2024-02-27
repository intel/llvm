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

std::vector<std::shared_ptr<AllocInfo>>
Quarantine::put(ur_device_handle_t Device, std::shared_ptr<AllocInfo> &Ptr) {
    auto AllocSize = Ptr->AllocSize;
    auto &Cache = m_Map[Device];
    std::vector<std::shared_ptr<AllocInfo>> DequeueList;
    while (Cache.Size() + AllocSize > m_MaxQuarantineSize) {
        DequeueList.emplace_back(Cache.Dequeue());
    }
    m_Map[Device].Enqueue(Ptr);
    return DequeueList;
}

} // namespace ur_sanitizer_layer
