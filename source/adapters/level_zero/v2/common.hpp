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
namespace {
const char *desturctorNames[] = {"zeKernelDestroy", "zeEventDestroy",
                                 "zeEventPoolDestroy", "zeContextDestroy",
                                 "zeCommandListDestroy"};
}

namespace v2 {

namespace raii {

template <typename ZeHandleT, ze_result_t (*destroy)(ZeHandleT), size_t nameId>
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
      // TODO: add appropriate logging or pass the error
      // to the caller (make the dtor noexcept(false) or use tls?)
    }
  }

  void reset() {
    if (!handle) {
      return;
    }

    if (ownZeHandle) {
      auto zeResult =
          ZE_CALL_NOCHECK_NAME(destroy, (handle), desturctorNames[nameId]);
      // Gracefully handle the case that L0 was already unloaded.
      if (zeResult && zeResult != ZE_RESULT_ERROR_UNINITIALIZED)
        throw ze2urResult(zeResult);
    }

    handle = nullptr;
  }

  std::pair<ZeHandleT, bool> release() {
    auto handle = this->handle;
    this->handle = nullptr;
    return {handle, ownZeHandle};
  }

  ZeHandleT get() const { return handle; }

  ZeHandleT *ptr() { return &handle; }

private:
  ZeHandleT handle;
  bool ownZeHandle;
};

using ze_kernel_handle_t =
    ze_handle_wrapper<::ze_kernel_handle_t, zeKernelDestroy, 0>;

using ze_event_handle_t =
    ze_handle_wrapper<::ze_event_handle_t, zeEventDestroy, 1>;

using ze_event_pool_handle_t =
    ze_handle_wrapper<::ze_event_pool_handle_t, zeEventPoolDestroy, 2>;

using ze_context_handle_t =
    ze_handle_wrapper<::ze_context_handle_t, zeContextDestroy, 3>;

using ze_command_list_handle_t =
    ze_handle_wrapper<::ze_command_list_handle_t, zeCommandListDestroy, 4>;

} // namespace raii
} // namespace v2
