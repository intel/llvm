//===--------- ur_cuda.hpp - CUDA Adapter ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

// We need the PI header temporarily while the UR device impl still uses the
// PI context type
#include <sycl/detail/pi.h>
#include <ur/ur.hpp>

#include <cuda.h>

// struct _ur_platform_handle_t;
// using ur_platform_handle_t = _ur_platform_handle_t *;
// struct _ur_device_handle_t;
// using ur_device_handle_t = _ur_device_handle_t *;

struct ur_platform_handle_t_ : public _ur_platform {
  std::vector<std::unique_ptr<ur_device_handle_t_>> devices_;
};

struct ur_device_handle_t_ : public _pi_object {
private:
  using native_type = CUdevice;

  native_type cuDevice_;
  CUcontext cuContext_;
  CUevent evBase_; // CUDA event used as base counter
  std::atomic_uint32_t refCount_;
  ur_platform_handle_t platform_;

  static constexpr pi_uint32 max_work_item_dimensions = 3u;
  size_t max_work_item_sizes[max_work_item_dimensions];
  int max_work_group_size;

public:
  ur_device_handle_t_(native_type cuDevice, CUcontext cuContext, CUevent evBase,
                      ur_platform_handle_t platform)
      : cuDevice_(cuDevice), cuContext_(cuContext),
        evBase_(evBase), refCount_{1}, platform_(platform) {}

  ur_device_handle_t_() { cuDevicePrimaryCtxRelease(cuDevice_); }

  native_type get() const noexcept { return cuDevice_; };

  CUcontext get_context() const noexcept { return cuContext_; };

  uint32_t get_reference_count() const noexcept { return refCount_; }

  ur_platform_handle_t get_platform() const noexcept { return platform_; };

  uint64_t get_elapsed_time(CUevent) const;

  void save_max_work_item_sizes(size_t size,
                                size_t *save_max_work_item_sizes) noexcept {
    memcpy(max_work_item_sizes, save_max_work_item_sizes, size);
  };

  void save_max_work_group_size(int value) noexcept {
    max_work_group_size = value;
  };

  void get_max_work_item_sizes(size_t ret_size,
                               size_t *ret_max_work_item_sizes) const noexcept {
    memcpy(ret_max_work_item_sizes, max_work_item_sizes, ret_size);
  };

  int get_max_work_group_size() const noexcept { return max_work_group_size; };
};

struct ur_context_handle_t_ : _pi_object {

  struct deleter_data {
    pi_context_extended_deleter function;
    void *user_data;

    void operator()() { function(user_data); }
  };

  using native_type = CUcontext;

  native_type cuContext_;
  ur_device_handle_t deviceId_;
  std::atomic_uint32_t refCount_;

  ur_context_handle_t_(ur_device_handle_t_ *devId)
      : cuContext_{devId->get_context()}, deviceId_{devId}, refCount_{1} {
    urDeviceRetain(deviceId_);
  };

  ~ur_context_handle_t_() {
    urDeviceRelease(deviceId_);
  }

  void invoke_extended_deleters() {
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto &deleter : extended_deleters_) {
      deleter();
    }
  }

  void set_extended_deleter(pi_context_extended_deleter function,
                            void *user_data) {
    std::lock_guard<std::mutex> guard(mutex_);
    extended_deleters_.emplace_back(deleter_data{function, user_data});
  }

  ur_device_handle_t get_device() const noexcept { return deviceId_; }

  native_type get() const noexcept { return cuContext_; }

  pi_uint32 increment_reference_count() noexcept { return ++refCount_; }

  pi_uint32 decrement_reference_count() noexcept { return --refCount_; }

  pi_uint32 get_reference_count() const noexcept { return refCount_; }

private:
  std::mutex mutex_;
  std::vector<deleter_data> extended_deleters_;
};

// Make the Unified Runtime handles definition complete.
// This is used in various "create" API where new handles are allocated.
// struct _zer_platform_handle_t : public _ur_platform_handle_t {
//   using _ur_platform_handle_t::_ur_platform_handle_t;
// };
// struct _zer_device_handle_t : public _ur_device_handle_t {
//   using _ur_device_handle_t::_ur_device_handle_t;
// };
