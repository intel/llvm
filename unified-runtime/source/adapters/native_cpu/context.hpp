//===--------- context.hpp - Native CPU Adapter ---------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mutex>
#include <set>
#include <ur_api.h>

#include "common.hpp"
#include "device.hpp"
#include "ur/ur.hpp"

namespace native_cpu {
struct usm_alloc_info {
  ur_usm_type_t type;
  const void *base_ptr;
  size_t size;
  ur_device_handle_t device;
  ur_usm_pool_handle_t pool;

  // We store a pointer to the actual allocation because it is needed when
  // freeing memory.
  void *base_alloc_ptr;
  constexpr usm_alloc_info(ur_usm_type_t type, const void *base_ptr,
                           size_t size, ur_device_handle_t device,
                           ur_usm_pool_handle_t pool, void *base_alloc_ptr)
      : type(type), base_ptr(base_ptr), size(size), device(device), pool(pool),
        base_alloc_ptr(base_alloc_ptr) {}
};

constexpr usm_alloc_info usm_alloc_info_null_entry(UR_USM_TYPE_UNKNOWN, nullptr,
                                                   0, nullptr, nullptr,
                                                   nullptr);

constexpr size_t alloc_header_size = sizeof(usm_alloc_info);

// Computes the padding that we need to add to ensure the
// pointer returned by UR is aligned as the user requested.
static size_t get_padding(uint32_t alignment) {
  assert(alignment >= alignof(usm_alloc_info) &&
         "memory not aligned to usm_alloc_info");
  if (!alignment || alloc_header_size % alignment == 0)
    return 0;
  size_t padd = 0;
  if (alignment <= alloc_header_size) {
    padd = alignment - (alloc_header_size % alignment);
  } else {
    padd = alignment - alloc_header_size;
  }
  return padd;
}

// In order to satisfy the MemAllocInfo queries we allocate extra memory
// for the native_cpu::usm_alloc_info struct.
// To satisfy the alignment requirements we "pad" the memory
// allocation so that the pointer returned to the user
// always satisfies (ptr % align) == 0.
static inline void *malloc_impl(uint32_t alignment, size_t size) {
  assert(alignment >= alignof(usm_alloc_info) &&
         "memory not aligned to usm_alloc_info");
  void *ptr = native_cpu::aligned_malloc(
      alignment, alloc_header_size + get_padding(alignment) + size);
  return ptr;
}

// The info struct is retrieved by subtracting its size from the pointer
// returned to the user.
static inline uint8_t *get_alloc_info_addr(const void *ptr) {
  return (uint8_t *)const_cast<void *>(ptr) - alloc_header_size;
}

static usm_alloc_info get_alloc_info(void *ptr) {
  return *(usm_alloc_info *)get_alloc_info_addr(ptr);
}

} // namespace native_cpu

struct ur_context_handle_t_ : RefCounted {
  ur_context_handle_t_(ur_device_handle_t_ *phDevices) : _device{phDevices} {}

  ur_device_handle_t _device;

  ur_result_t remove_alloc(void *ptr) {
    std::lock_guard<std::mutex> lock(alloc_mutex);
    const native_cpu::usm_alloc_info &info = native_cpu::get_alloc_info(ptr);
    UR_ASSERT(info.type != UR_USM_TYPE_UNKNOWN,
              UR_RESULT_ERROR_INVALID_MEM_OBJECT);

    native_cpu::aligned_free(info.base_alloc_ptr);
    allocations.erase(ptr);
    return UR_RESULT_SUCCESS;
  }

  // Note this is made non-const to access the mutex
  const native_cpu::usm_alloc_info &get_alloc_info_entry(const void *ptr) {
    std::lock_guard<std::mutex> lock(alloc_mutex);
    auto it = allocations.find(ptr);
    if (it == allocations.end()) {
      return native_cpu::usm_alloc_info_null_entry;
    }

    return *(native_cpu::usm_alloc_info *)native_cpu::get_alloc_info_addr(ptr);
  }

  void *add_alloc(uint32_t alignment, ur_usm_type_t type, size_t size,
                  ur_usm_pool_handle_t pool) {
    std::lock_guard<std::mutex> lock(alloc_mutex);
    // We need to ensure that we align to at least alignof(usm_alloc_info),
    // otherwise its start address may be unaligned.
    alignment =
        std::max<size_t>(alignment, alignof(native_cpu::usm_alloc_info));
    void *alloc = native_cpu::malloc_impl(alignment, size);
    if (!alloc)
      return nullptr;
    // Compute the address of the pointer that we'll return to the user.
    void *ptr = native_cpu::alloc_header_size +
                native_cpu::get_padding(alignment) + (uint8_t *)alloc;
    uint8_t *info_addr = native_cpu::get_alloc_info_addr(ptr);
    if (!info_addr)
      return nullptr;
    // Do a placement new of the alloc_info to avoid allocation and copy
    auto info = new (info_addr)
        native_cpu::usm_alloc_info(type, ptr, size, this->_device, pool, alloc);
    if (!info)
      return nullptr;
    allocations.insert(ptr);
    return ptr;
  }

private:
  std::mutex alloc_mutex;
  std::set<const void *> allocations;
};
