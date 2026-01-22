//===--------- common.hpp - Level Zero Adapter ----------------------------===//
//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <list>
#include <map>
#include <mutex>
#include <stdarg.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "common/helpers/shared_helpers.hpp"
#include "ur_interface_loader.hpp"
#include <ur/ur.hpp>

namespace ur::level_zero::v1 {

struct ur_object_t : ur::handle_base<ddi_getter> {
  ur_shared_mutex Mutex;
  bool OwnNativeHandle = false;
};

// Forward declarations for v1-only concrete handle types.
struct ur_context_handle_t_;
typedef struct ur_context_handle_t_ *ur_context_handle_t;
struct ur_event_handle_t_;
typedef struct ur_event_handle_t_ *ur_event_handle_t;
struct ur_queue_handle_t_;
typedef struct ur_queue_handle_t_ *ur_queue_handle_t;
struct ur_kernel_handle_t_;
typedef struct ur_kernel_handle_t_ *ur_kernel_handle_t;
struct ur_mem_handle_t_;
typedef struct ur_mem_handle_t_ *ur_mem_handle_t;
struct ur_usm_pool_handle_t_;
typedef struct ur_usm_pool_handle_t_ *ur_usm_pool_handle_t;
struct ur_exp_command_buffer_handle_t_;
typedef struct ur_exp_command_buffer_handle_t_ *ur_exp_command_buffer_handle_t;

// Cast from an opaque UR handle to the v1 concrete type. The loader only ever
// reads offset 0 (ddi_table).
namespace detail {
// Maps an opaque handle typedef to its corresponding v1 internal struct.
template <typename Opaque> struct v1_handle_traits;
template <> struct v1_handle_traits<::ur_context_handle_t> {
  using type = ur_context_handle_t_;
};
template <> struct v1_handle_traits<::ur_event_handle_t> {
  using type = ur_event_handle_t_;
};
template <> struct v1_handle_traits<::ur_queue_handle_t> {
  using type = ur_queue_handle_t_;
};
template <> struct v1_handle_traits<::ur_kernel_handle_t> {
  using type = ur_kernel_handle_t_;
};
template <> struct v1_handle_traits<::ur_mem_handle_t> {
  using type = ur_mem_handle_t_;
};
template <> struct v1_handle_traits<::ur_usm_pool_handle_t> {
  using type = ur_usm_pool_handle_t_;
};
template <> struct v1_handle_traits<::ur_exp_command_buffer_handle_t> {
  using type = ur_exp_command_buffer_handle_t_;
};

template <typename Opaque>
using v1_internal_t = typename v1_handle_traits<Opaque>::type;

// Reverse mapping: v1 internal struct -> opaque handle.
template <typename Internal> struct v1_opaque_handle_for;
template <> struct v1_opaque_handle_for<ur_context_handle_t_> {
  using type = ::ur_context_handle_t;
};
template <> struct v1_opaque_handle_for<ur_event_handle_t_> {
  using type = ::ur_event_handle_t;
};
template <> struct v1_opaque_handle_for<ur_queue_handle_t_> {
  using type = ::ur_queue_handle_t;
};
template <> struct v1_opaque_handle_for<ur_kernel_handle_t_> {
  using type = ::ur_kernel_handle_t;
};
template <> struct v1_opaque_handle_for<ur_mem_handle_t_> {
  using type = ::ur_mem_handle_t;
};
template <> struct v1_opaque_handle_for<ur_usm_pool_handle_t_> {
  using type = ::ur_usm_pool_handle_t;
};
template <> struct v1_opaque_handle_for<ur_exp_command_buffer_handle_t_> {
  using type = ::ur_exp_command_buffer_handle_t;
};
} // namespace detail

// Opaque handle -> v1 internal pointer.
template <typename Opaque>
inline detail::v1_internal_t<Opaque> *v1_cast(Opaque h) {
  return reinterpret_cast<detail::v1_internal_t<Opaque> *>(h);
}

// Opaque handle array -> v1 internal pointer array.
template <typename Opaque>
inline detail::v1_internal_t<Opaque> **v1_cast(Opaque *ph) {
  return reinterpret_cast<detail::v1_internal_t<Opaque> **>(ph);
}

// const Opaque handle array -> const v1 internal pointer array.
template <typename Opaque>
inline detail::v1_internal_t<Opaque> *const *v1_cast(const Opaque *ph) {
  return reinterpret_cast<detail::v1_internal_t<Opaque> *const *>(ph);
}

// V1 internal pointer -> opaque handle (reverse direction).
template <typename Internal>
inline typename detail::v1_opaque_handle_for<Internal>::type
v1_cast(Internal *p) {
  return reinterpret_cast<
      typename detail::v1_opaque_handle_for<Internal>::type>(p);
}

// V1 internal pointer array -> opaque handle array.
template <typename Internal>
inline typename detail::v1_opaque_handle_for<Internal>::type *
v1_cast(Internal **p) {
  return reinterpret_cast<
      typename detail::v1_opaque_handle_for<Internal>::type *>(p);
}

// Record for a memory allocation. This structure is used to keep information
// for each memory allocation.
struct MemAllocRecord : ur_object_t {
  MemAllocRecord(ur_context_handle_t Context, bool OwnZeMemHandle = true)
      : Context(Context) {
    OwnNativeHandle = OwnZeMemHandle;
  }
  // Currently kernel can reference memory allocations from different contexts
  // and we need to know the context of a memory allocation when we release it
  // in piKernelRelease.
  // TODO: this should go away when memory isolation issue is fixed in the Level
  // Zero runtime.
  ur_context_handle_t Context;

  ur::RefCount RefCount;
};

} // namespace ur::level_zero::v1
