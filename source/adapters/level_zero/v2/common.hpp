//===--------- common.hpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <exception>
#include <ze_api.h>

#include "../common.hpp"
#include "logger/ur_logger.hpp"

namespace v2 {

namespace raii {

template <typename ZeHandleT, ze_result_t (*destroy)(ZeHandleT)>
struct ze_handle_wrapper {
  ze_handle_wrapper(bool ownZeHandle = true)
      : handle(nullptr), ownZeHandle(ownZeHandle) {}

  ze_handle_wrapper(ZeHandleT handle, bool ownZeHandle = true)
      : handle(handle), ownZeHandle(ownZeHandle) {}

  ze_handle_wrapper(const ze_handle_wrapper &) = delete;
  ze_handle_wrapper &operator=(const ze_handle_wrapper &) = delete;

  ze_handle_wrapper(ze_handle_wrapper &&other)
      : handle(other.handle), ownZeHandle(other.ownZeHandle) {
    other.handle = nullptr;
  }

  ze_handle_wrapper &operator=(ze_handle_wrapper &&other) {
    if (this == &other) {
      return *this;
    }

    if (handle) {
      reset();
    }
    handle = other.handle;
    ownZeHandle = other.ownZeHandle;
    other.handle = nullptr;
    return *this;
  }

  ~ze_handle_wrapper() {
    try {
      reset();
    } catch (...) {
    }
  }

  void reset() {
    if (!handle) {
      return;
    }

    auto zeResult = ZE_CALL_NOCHECK(destroy, (handle));
    // Gracefully handle the case that L0 was already unloaded.
    if (zeResult && zeResult != ZE_RESULT_ERROR_UNINITIALIZED)
      throw ze2urResult(zeResult);

    handle = nullptr;
  }

  ZeHandleT release() {
    auto handle = this->handle;
    this->handle = nullptr;
    return handle;
  }

  ZeHandleT get() const { return handle; }

  ZeHandleT *ptr() { return &handle; }

private:
  ZeHandleT handle;
  bool ownZeHandle;
};

template <typename URHandle, ur_result_t (*retain)(URHandle),
          ur_result_t (*release)(URHandle)>
struct ur_shared_handle {
  using handle_t = URHandle;

  ur_shared_handle() : handle(nullptr) {}
  explicit ur_shared_handle(handle_t handle) : handle(handle) {}
  ~ur_shared_handle() {
    try {
      reset();
    } catch (...) {
    }
  }

  ur_shared_handle(const ur_shared_handle &other) : handle(other.handle) {
    retain(handle);
  }
  ur_shared_handle(ur_shared_handle &&other) : handle(other.handle) {
    other.handle = nullptr;
  }
  ur_shared_handle(std::nullptr_t) : handle(nullptr) {}

  void reset() {
    if (!handle) {
      return;
    }

    UR_CALL_THROWS(release(handle));
    handle = nullptr;
  }

  ur_shared_handle &operator=(const ur_shared_handle &other) {
    if (handle) {
      release(handle);
    }
    handle = other.handle;
    retain(handle);
    return *this;
  }
  ur_shared_handle &operator=(ur_shared_handle &&other) {
    if (handle) {
      release(handle);
    }
    handle = other.handle;
    other.handle = nullptr;
    return *this;
  }
  ur_shared_handle &operator=(std::nullptr_t) {
    if (handle) {
      release(handle);
    }
    new (this) ur_shared_handle(nullptr);
    return *this;
  }

  handle_t *ptr() { return &handle; }
  handle_t get() const { return handle; }
  handle_t operator->() { return handle; }
  operator handle_t() { return handle; }

private:
  handle_t handle;
};

using ze_kernel_handle_t =
    ze_handle_wrapper<::ze_kernel_handle_t, zeKernelDestroy>;

using ze_event_handle_t =
    ze_handle_wrapper<::ze_event_handle_t, zeEventDestroy>;

using ze_event_pool_handle_t =
    ze_handle_wrapper<::ze_event_pool_handle_t, zeEventPoolDestroy>;

using ur_queue_shared_handle_t =
    ur_shared_handle<ur_queue_handle_t, urQueueRetain, urQueueRelease>;

using ur_kernel_shared_handle_t =
    ur_shared_handle<ur_kernel_handle_t, urKernelRetain, urKernelRelease>;

} // namespace raii
} // namespace v2
