/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_quarantine.hpp
 *
 */

#pragma once

#include "asan_allocator.hpp"

#include <atomic>
#include <queue>
#include <unordered_map>
#include <vector>

namespace ur_sanitizer_layer {
namespace asan {

class QuarantineCache {
  public:
    using Element = AllocationIterator;
    using List = std::queue<Element>;

    // The following methods are not thread safe, use this lock
    ur_mutex Mutex;

    // Total memory used, including internal accounting.
    uptr size() const { return m_Size; }

    void enqueue(Element &It) {
        m_List.push(It);
        m_Size += It->second->AllocSize;
    }

    std::optional<Element> dequeue() {
        if (m_List.empty()) {
            return std::optional<Element>{};
        }
        auto It = m_List.front();
        m_List.pop();
        m_Size -= It->second->AllocSize;
        return It;
    }

  private:
    List m_List;
    std::atomic_uintptr_t m_Size = 0;
};

class Quarantine {
  public:
    explicit Quarantine(size_t MaxQuarantineSize)
        : m_MaxQuarantineSize(MaxQuarantineSize) {}

    std::vector<AllocationIterator> put(ur_device_handle_t Device,
                                        AllocationIterator &Ptr);

  private:
    QuarantineCache &getCache(ur_device_handle_t Device) {
        std::scoped_lock<ur_mutex> Guard(m_Mutex);
        return m_Map[Device];
    }

    std::unordered_map<ur_device_handle_t, QuarantineCache> m_Map;
    ur_mutex m_Mutex;
    size_t m_MaxQuarantineSize;
};

} // namespace asan
} // namespace ur_sanitizer_layer
