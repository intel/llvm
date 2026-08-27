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
#include <type_traits>
#include <vector>
#include <ze_api.h>

#include "../common/device.hpp"
#include "../common/helpers/shared_helpers.hpp"
#include "logger/ur_logger.hpp"
#include "ur_interface_loader.hpp"
#include <ur/ur.hpp>

namespace ur::level_zero::v2 {

struct ur_object_t : ur::handle_base<ddi_getter> {
  ur_shared_mutex Mutex;
  bool OwnNativeHandle = false;
};

// Forward declarations for v2-only concrete handle types.
struct ur_context_handle_t_;
typedef struct ur_context_handle_t_ *ur_context_handle_t;
struct ur_event_handle_t_;
typedef struct ur_event_handle_t_ *ur_event_handle_t;
struct ur_usm_pool_handle_t_;
typedef struct ur_usm_pool_handle_t_ *ur_usm_pool_handle_t;
struct ur_kernel_handle_t_;
typedef struct ur_kernel_handle_t_ *ur_kernel_handle_t;
struct ur_queue_handle_t_;
typedef struct ur_queue_handle_t_ *ur_queue_handle_t;
struct ur_mem_handle_t_;
typedef struct ur_mem_handle_t_ *ur_mem_handle_t;
struct ur_exp_command_buffer_handle_t_;
typedef struct ur_exp_command_buffer_handle_t_ *ur_exp_command_buffer_handle_t;
struct ur_exp_graph_handle_t_;
typedef struct ur_exp_graph_handle_t_ *ur_exp_graph_handle_t;
struct ur_exp_executable_graph_handle_t_;
typedef struct ur_exp_executable_graph_handle_t_
    *ur_exp_executable_graph_handle_t;

// Cast from an opaque UR handle to the v2 concrete type. The loader only ever
// reads offset 0 (ddi_table).
namespace detail {
// Maps an opaque handle typedef to its corresponding v2 internal struct.
template <typename Opaque> struct v2_handle_traits;
template <> struct v2_handle_traits<::ur_context_handle_t> {
  using type = ur_context_handle_t_;
};
template <> struct v2_handle_traits<::ur_event_handle_t> {
  using type = ur_event_handle_t_;
};
template <> struct v2_handle_traits<::ur_usm_pool_handle_t> {
  using type = ur_usm_pool_handle_t_;
};
template <> struct v2_handle_traits<::ur_kernel_handle_t> {
  using type = ur_kernel_handle_t_;
};
template <> struct v2_handle_traits<::ur_queue_handle_t> {
  using type = ur_queue_handle_t_;
};
template <> struct v2_handle_traits<::ur_mem_handle_t> {
  using type = ur_mem_handle_t_;
};
template <> struct v2_handle_traits<::ur_exp_command_buffer_handle_t> {
  using type = ur_exp_command_buffer_handle_t_;
};
template <> struct v2_handle_traits<::ur_exp_graph_handle_t> {
  using type = ur_exp_graph_handle_t_;
};
template <> struct v2_handle_traits<::ur_exp_executable_graph_handle_t> {
  using type = ur_exp_executable_graph_handle_t_;
};
template <> struct v2_handle_traits<::ur_device_handle_t> {
  using type = ur::level_zero::ur_device_handle_t_;
};
template <> struct v2_handle_traits<::ur_program_handle_t> {
  using type = ur::level_zero::ur_program_handle_t_;
};

template <typename Opaque>
using v2_internal_t = typename v2_handle_traits<Opaque>::type;

// Reverse mapping: v2 internal struct -> opaque handle.
template <typename Internal> struct v2_opaque_handle_for;
template <> struct v2_opaque_handle_for<ur_context_handle_t_> {
  using type = ::ur_context_handle_t;
};
template <> struct v2_opaque_handle_for<ur_event_handle_t_> {
  using type = ::ur_event_handle_t;
};
template <> struct v2_opaque_handle_for<ur_usm_pool_handle_t_> {
  using type = ::ur_usm_pool_handle_t;
};
template <> struct v2_opaque_handle_for<ur_kernel_handle_t_> {
  using type = ::ur_kernel_handle_t;
};
template <> struct v2_opaque_handle_for<ur_queue_handle_t_> {
  using type = ::ur_queue_handle_t;
};
template <> struct v2_opaque_handle_for<ur_mem_handle_t_> {
  using type = ::ur_mem_handle_t;
};
template <> struct v2_opaque_handle_for<ur_exp_command_buffer_handle_t_> {
  using type = ::ur_exp_command_buffer_handle_t;
};
template <> struct v2_opaque_handle_for<ur_exp_graph_handle_t_> {
  using type = ::ur_exp_graph_handle_t;
};
template <> struct v2_opaque_handle_for<ur_exp_executable_graph_handle_t_> {
  using type = ::ur_exp_executable_graph_handle_t;
};
template <> struct v2_opaque_handle_for<ur::level_zero::ur_device_handle_t_> {
  using type = ::ur_device_handle_t;
};
template <> struct v2_opaque_handle_for<ur::level_zero::ur_program_handle_t_> {
  using type = ::ur_program_handle_t;
};
} // namespace detail

// Opaque handle -> v2 internal pointer.
template <typename Opaque>
inline detail::v2_internal_t<Opaque> *v2_cast(Opaque h) {
  return reinterpret_cast<detail::v2_internal_t<Opaque> *>(h);
}

// Opaque handle array -> v2 internal pointer array.
template <typename Opaque>
inline detail::v2_internal_t<Opaque> **v2_cast(Opaque *ph) {
  return reinterpret_cast<detail::v2_internal_t<Opaque> **>(ph);
}

// const Opaque handle array -> const v2 internal pointer array.
template <typename Opaque>
inline detail::v2_internal_t<Opaque> *const *v2_cast(const Opaque *ph) {
  return reinterpret_cast<detail::v2_internal_t<Opaque> *const *>(ph);
}

// V2 internal pointer -> opaque handle (reverse direction).
template <typename Internal>
inline typename detail::v2_opaque_handle_for<Internal>::type
v2_cast(Internal *p) {
  return reinterpret_cast<
      typename detail::v2_opaque_handle_for<Internal>::type>(p);
}

// V2 internal pointer array -> opaque handle array.
template <typename Internal>
inline typename detail::v2_opaque_handle_for<Internal>::type *
v2_cast(Internal **p) {
  return reinterpret_cast<
      typename detail::v2_opaque_handle_for<Internal>::type *>(p);
}

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

    if (ownZeHandle && ur::level_zero::checkL0LoaderTeardown()) {
      ze_result_t zeResult =
          ZE_CALL_NOCHECK_NAME(destroy, (handle), destroyName);
      // Gracefully handle the case that L0 was already unloaded.
      if (zeResult && (zeResult != ZE_RESULT_ERROR_UNINITIALIZED &&
                       zeResult != ZE_RESULT_ERROR_UNKNOWN)) {
        UR_DFAILURE("destroy failed in L0 with" << zeResult);
        throw ur::level_zero::ze2urResult(zeResult);
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

#define HANDLE_WRAPPER_TYPE_NAMED(Alias, ZeHandleT, DestroyFunc)               \
  inline constexpr char Alias##_destroyName[] = #DestroyFunc;                  \
  using Alias =                                                                \
      ze_handle_wrapper<::ZeHandleT, DestroyFunc, Alias##_destroyName>;

// Common case: the alias name matches the raw handle type name.
#define HANDLE_WRAPPER_TYPE(ZeHandleT, DestroyFunc)                            \
  HANDLE_WRAPPER_TYPE_NAMED(ZeHandleT, ZeHandleT, DestroyFunc)

HANDLE_WRAPPER_TYPE(ze_kernel_handle_t, zeKernelDestroy)
HANDLE_WRAPPER_TYPE(ze_event_handle_t, zeEventDestroy)
HANDLE_WRAPPER_TYPE(ze_event_pool_handle_t, zeEventPoolDestroy)
HANDLE_WRAPPER_TYPE(ze_context_handle_t, zeContextDestroy)
HANDLE_WRAPPER_TYPE(ze_command_list_handle_t, zeCommandListDestroy)
HANDLE_WRAPPER_TYPE(ze_image_handle_t, zeImageDestroy)

// Counter-based event IPC handle opened via zeEventCounterBasedOpenIpcHandle.
// Same underlying handle type as ze_event_handle_t but a different teardown
// function, hence a distinct alias.
HANDLE_WRAPPER_TYPE_NAMED(ipc_event_handle_t, ze_event_handle_t,
                          zeEventCounterBasedCloseIpcHandle)

template <typename Handle, typename HandleOpque,
          ur_result_t (*retain)(HandleOpque),
          ur_result_t (*release)(HandleOpque)>
struct ur_handle {
  ur_handle(Handle handle = nullptr) : handle(handle) {
    if (handle) {
      retain(v2_cast(handle));
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
      release(v2_cast(this->handle));
    }

    this->handle = rhs.handle;
    rhs.handle = nullptr;

    return *this;
  }

  ~ur_handle() {
    if (handle) {
      release(v2_cast(handle));
    }
  }

  Handle get() const { return handle; }

  Handle operator->() const { return get(); }

private:
  Handle handle;
};

using ur_context_handle_t =
    ur_handle<v2::ur_context_handle_t, ::ur_context_handle_t, urContextRetain,
              urContextRelease>;
using ur_device_handle_t =
    ur_handle<ur::level_zero::ur_device_handle_t, ::ur_device_handle_t,
              urDeviceRetain, urDeviceRelease>;
} // namespace raii

} // namespace ur::level_zero::v2
