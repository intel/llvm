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

#include <unordered_map>
#include <ur_api.h>

#include "common.hpp"
#include "device.hpp"
#include "ur/ur.hpp"

namespace native_cpu {
struct usm_alloc_info {
  const ur_usm_type_t type;
  const void *base_ptr;
  const size_t size;
  const ur_device_handle_t device;
  const ur_usm_pool_handle_t pool;
  usm_alloc_info(ur_usm_type_t type, const void *base_ptr, size_t size,
                 ur_device_handle_t device, ur_usm_pool_handle_t pool)
      : type(type), base_ptr(base_ptr), size(size), device(device), pool(pool) {
  }
};
} // namespace native_cpu

struct ur_context_handle_t_ : RefCounted {
  ur_context_handle_t_(ur_device_handle_t_ *phDevices) : _device{phDevices} {}

  ur_device_handle_t _device;

  void add_alloc_info_entry(const void *ptr, ur_usm_type_t type, size_t size,
                            ur_usm_pool_handle_t pool) {
    native_cpu::usm_alloc_info info(type, ptr, size, this->_device, pool);
    alloc_info.insert(std::make_pair(ptr, info));
  }

  native_cpu::usm_alloc_info get_alloc_info_entry(const void *ptr) const {
    auto it = alloc_info.find(ptr);
    if (it == alloc_info.end()) {
      return native_cpu::usm_alloc_info(UR_USM_TYPE_UNKNOWN, ptr, 0, nullptr,
                                        nullptr);
    }
    return it->second;
  }

  void remove_alloc_info_entry(void *ptr) { alloc_info.erase(ptr); }

private:
  std::unordered_map<const void *, native_cpu::usm_alloc_info> alloc_info;
};
