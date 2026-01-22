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

#include "../helpers/shared_helpers.hpp"
#include "logger/ur_logger.hpp"
#include "ur_interface_loader.hpp"

namespace v2 {
namespace raii {

// Base class to store common data
struct ur_object : ur::handle_base<ur::level_zero_v2::ddi_getter> {
  ur_object() : handle_base() {}

  // This mutex protects accesses to all the non-const member variables.
  // Exclusive access is required to modify any of these members.
  //
  // To get shared access to the object in a scope use std::shared_lock:
  //    std::shared_lock Lock(Obj->Mutex);
  // To get exclusive access to the object in a scope use std::scoped_lock:
  //    std::scoped_lock Lock(Obj->Mutex);
  //
  // If several UR objects are accessed in a scope then each object's mutex must
  // be locked. For example, to get write access to Obj1 and Obj2 and read
  // access to Obj3 in a scope use the following approach:
  //   std::shared_lock Obj3Lock(Obj3->Mutex, std::defer_lock);
  //   std::scoped_lock LockAll(Obj1->Mutex, Obj2->Mutex, Obj3Lock);
  ur_shared_mutex Mutex;

  // Indicates if we own the native handle or it came from interop that
  // asked to not transfer the ownership to SYCL RT.
  bool OwnNativeHandle = false;
};

template <typename ZeHandleT, ze_result_t (*destroy)(ZeHandleT),
          const char *destroyName>
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

    if (ownZeHandle && checkL0LoaderTeardown()) {
      ze_result_t zeResult =
          ZE_CALL_NOCHECK_NAME(destroy, (handle), destroyName);
      // Gracefully handle the case that L0 was already unloaded.
      if (zeResult && (zeResult != ZE_RESULT_ERROR_UNINITIALIZED &&
                       zeResult != ZE_RESULT_ERROR_UNKNOWN)) {
        UR_DFAILURE("destroy failed in L0 with" << zeResult);
        throw ze2urResult(zeResult);
      }
      if (zeResult == ZE_RESULT_ERROR_UNKNOWN) {
        zeResult = ZE_RESULT_ERROR_UNINITIALIZED;
      }
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

#define HANDLE_WRAPPER_TYPE(ZeHandleT, DestroyFunc)                            \
  inline constexpr char ZeHandleT##_destroyName[] = #DestroyFunc;              \
  using ZeHandleT =                                                            \
      ze_handle_wrapper<::ZeHandleT, DestroyFunc, ZeHandleT##_destroyName>;

HANDLE_WRAPPER_TYPE(ze_kernel_handle_t, zeKernelDestroy)
HANDLE_WRAPPER_TYPE(ze_event_handle_t, zeEventDestroy)
HANDLE_WRAPPER_TYPE(ze_event_pool_handle_t, zeEventPoolDestroy)
HANDLE_WRAPPER_TYPE(ze_context_handle_t, zeContextDestroy)
HANDLE_WRAPPER_TYPE(ze_command_list_handle_t, zeCommandListDestroy)
HANDLE_WRAPPER_TYPE(ze_image_handle_t, zeImageDestroy)

template <typename RawHandle, ur_result_t (*retain)(RawHandle),
          ur_result_t (*release)(RawHandle)>
struct ur_handle {
  ur_handle(RawHandle handle = nullptr) : handle(handle) {
    if (handle) {
      retain(handle);
    }
  }

  ur_handle(const ur_handle &) = delete;
  ur_handle &operator=(const ur_handle &) = delete;

  ur_handle(ur_handle &&rhs) {
    this->handle = rhs.handle;
    rhs.handle = nullptr;
  }

  ur_handle &operator=(ur_handle &&rhs) {
    if (this == &rhs) {
      return *this;
    }

    if (this->handle) {
      release(this->handle);
    }

    this->handle = rhs.handle;
    rhs.handle = nullptr;

    return *this;
  }

  ~ur_handle() {
    if (handle) {
      release(handle);
    }
  }

  RawHandle get() const { return handle; }

  RawHandle operator->() const { return get(); }

private:
  RawHandle handle;
};

using ur_context_handle_t =
    ur_handle<::ur_context_handle_t, ur::level_zero_v2::urContextRetain,
              ur::level_zero_v2::urContextRelease>;
using ur_device_handle_t =
    ur_handle<::ur_device_handle_t, ur::level_zero_v2::urDeviceRetain,
              ur::level_zero_v2::urDeviceRelease>;

} // namespace raii
} // namespace v2
