//===--------- common.hpp - Level Zero Adapter ---------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <exception>
#include <ze_api.h>

#include "../common/device.hpp"
#include "../common/helpers/shared_helpers.hpp"
#include "logger/ur_logger.hpp"
#include "ur_interface_loader.hpp"

// Transition alias: existing v2 code uses bare `v2::foo` throughout (e.g.
// `v2::raii::...`, `v2::EVENT_FLAGS_COUNTER`). The actual namespace is now
// `ur::level_zero::v2`; this alias keeps those references resolving.
namespace v2 = ur::level_zero::v2;

namespace ur::level_zero::v2 {

// V2-local plain-data payload shared by every concrete v2 handle. Does NOT
// carry `ddi_table`; that lives on `ur_handle_base_t` below (or, for
// handle types that implement a `ur_<noun>_interface_t` from common/, on
// the interface itself).
struct ur_object_t {
  ur_shared_mutex Mutex;
  bool OwnNativeHandle = false;
};

// Default base for v2 concrete handle types that don't implement a
// common/-level `_interface_t`. Auto-populates `ddi_table` with v2's DDI
// at construction. `ddi_table` is the first member so it sits at offset
// 0 of every concrete handle (where the loader's intercept layer reads
// it).
struct ur_handle_base_t {
  const ur_dditable_t *ddi_table = ur::level_zero::v2::ddi_getter::value();
  ur_shared_mutex Mutex;
  bool OwnNativeHandle = false;

  ur_handle_base_t() = default;
  ur_handle_base_t(const ur_handle_base_t &) = delete;
  ur_handle_base_t &operator=(const ur_handle_base_t &) = delete;
};

namespace raii {

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
    ur_handle<::ur_context_handle_t, ur::level_zero::v2::urContextRetain,
              ur::level_zero::v2::urContextRelease>;
using ur_device_handle_t =
    ur_handle<::ur_device_handle_t, ur::level_zero::common::urDeviceRetain,
              ur::level_zero::common::urDeviceRelease>;

} // namespace raii

struct ur_context_handle_t_;
struct ur_event_handle_t_;
struct ur_usm_pool_handle_t_;
struct ur_kernel_handle_t_;
struct ur_queue_handle_t_;
struct ur_mem_handle_t_;
struct ur_exp_command_buffer_handle_t_;
struct ur_exp_graph_handle_t_;
struct ur_exp_executable_graph_handle_t_;

template <typename V2Type, typename HandleType>
inline V2Type *v2_cast(HandleType h) {
  return reinterpret_cast<V2Type *>(h);
}

inline ur_context_handle_t_ *v2_cast(::ur_context_handle_t h) {
  return reinterpret_cast<ur_context_handle_t_ *>(h);
}

inline ur_event_handle_t_ *v2_cast(::ur_event_handle_t h) {
  return reinterpret_cast<ur_event_handle_t_ *>(h);
}

inline ur_usm_pool_handle_t_ *v2_cast(::ur_usm_pool_handle_t h) {
  return reinterpret_cast<ur_usm_pool_handle_t_ *>(h);
}

inline ur_kernel_handle_t_ *v2_cast(::ur_kernel_handle_t h) {
  return reinterpret_cast<ur_kernel_handle_t_ *>(h);
}

inline ur_queue_handle_t_ *v2_cast(::ur_queue_handle_t h) {
  return reinterpret_cast<ur_queue_handle_t_ *>(h);
}

inline ur_mem_handle_t_ *v2_cast(::ur_mem_handle_t h) {
  return reinterpret_cast<ur_mem_handle_t_ *>(h);
}

inline ur_exp_command_buffer_handle_t_ *
v2_cast(::ur_exp_command_buffer_handle_t h) {
  return reinterpret_cast<ur_exp_command_buffer_handle_t_ *>(h);
}

inline ur_exp_graph_handle_t_ *v2_cast(::ur_exp_graph_handle_t h) {
  return reinterpret_cast<ur_exp_graph_handle_t_ *>(h);
}

inline ur_exp_executable_graph_handle_t_ *
v2_cast(::ur_exp_executable_graph_handle_t h) {
  return reinterpret_cast<ur_exp_executable_graph_handle_t_ *>(h);
}

} // namespace ur::level_zero::v2
