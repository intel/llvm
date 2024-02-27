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
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

namespace ur_sanitizer_layer {

class QuarantineCache {
  public:
    explicit QuarantineCache() {}

    // Total memory used, including internal accounting.
    uptr Size() const { return m_Size; }

    // Memory used for internal accounting.
    // uptr OverheadSize() const { return m_List.size() * sizeof(QuarantineBatch); }

    void Enqueue(std::shared_ptr<AllocInfo> &ptr) {
        m_List.push(ptr);
        m_Size += ptr->AllocSize;
    }

    std::shared_ptr<AllocInfo> Dequeue() {
        if (m_List.empty()) {
            return nullptr;
        }
        auto b = m_List.front();
        m_List.pop();
        m_Size -= b->AllocSize;
        return b;
    }

    void PrintStats() const {}

  private:
    typedef std::queue<std::shared_ptr<AllocInfo>> List;

    List m_List;
    std::atomic_uintptr_t m_Size;
};

class Quarantine {
  public:
    explicit Quarantine(size_t MaxQuarantineSize)
        : m_MaxQuarantineSize(MaxQuarantineSize) {}

    std::vector<std::shared_ptr<AllocInfo>>
    put(ur_device_handle_t Device, std::shared_ptr<AllocInfo> &Ptr);

  private:
    std::unordered_map<ur_device_handle_t, QuarantineCache> m_Map;
    size_t m_MaxQuarantineSize;
};

} // namespace ur_sanitizer_layer
