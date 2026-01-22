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
#include <unordered_map>
#include <vector>

#include "common/helpers/shared_helpers.hpp"
#include "ur_interface_loader.hpp"

namespace ur::level_zero {

// V1-local plain-data payload shared by every concrete v1 handle. Does NOT
// carry `ddi_table` — that lives on `ur_handle_base_t` below (or, for
// handle types that implement a `ur_<noun>_interface_t` from common/, on
// the interface itself). No dependency on UR_L0_V1_ADAPTER_ENABLED.
struct ur_object_t {
  ur_shared_mutex Mutex;
  bool OwnNativeHandle = false;
};

// Default base for v1 concrete handle types that don't implement a
// common/-level `_interface_t`. Auto-populates `ddi_table` with v1's DDI
// at construction, preserving the historical behavior of `ur_object`.
// Order matters: `ddi_table` must be the first member so it sits at
// offset 0 of every concrete handle (where the loader's intercept layer
// reads it).
struct ur_handle_base_t {
  const ur_dditable_t *ddi_table = ur::level_zero::ddi_getter::value();
  ur_shared_mutex Mutex;
  bool OwnNativeHandle = false;

  ur_handle_base_t() = default;
  ur_handle_base_t(const ur_handle_base_t &) = delete;
  ur_handle_base_t &operator=(const ur_handle_base_t &) = delete;
};

} // namespace ur::level_zero

namespace ur::level_zero::v1 {

// Forward declarations for v1-only concrete handle types. Full definitions
// live in the per-type v1 headers (context.hpp, event.hpp, queue.hpp, etc.).
struct ur_context_handle_t_;
struct ur_event_handle_t_;
struct ur_queue_handle_t_;
struct ur_kernel_handle_t_;
struct ur_mem_handle_t_;
struct ur_usm_pool_handle_t_;
struct ur_exp_command_buffer_handle_t_;

// Cast from an opaque UR handle to the v1 concrete type. The loader only ever
// reads offset 0 (ddi_table); v1 code uses this helper to reach v1-specific
// members on handles that the adapter itself allocated.
inline ur_context_handle_t_ *v1_cast(::ur_context_handle_t h) {
  return reinterpret_cast<ur_context_handle_t_ *>(h);
}
inline ur_event_handle_t_ *v1_cast(::ur_event_handle_t h) {
  return reinterpret_cast<ur_event_handle_t_ *>(h);
}
inline ur_queue_handle_t_ *v1_cast(::ur_queue_handle_t h) {
  return reinterpret_cast<ur_queue_handle_t_ *>(h);
}
inline ur_kernel_handle_t_ *v1_cast(::ur_kernel_handle_t h) {
  return reinterpret_cast<ur_kernel_handle_t_ *>(h);
}
inline ur_mem_handle_t_ *v1_cast(::ur_mem_handle_t h) {
  return reinterpret_cast<ur_mem_handle_t_ *>(h);
}
inline ur_usm_pool_handle_t_ *v1_cast(::ur_usm_pool_handle_t h) {
  return reinterpret_cast<ur_usm_pool_handle_t_ *>(h);
}
inline ur_exp_command_buffer_handle_t_ *
v1_cast(::ur_exp_command_buffer_handle_t h) {
  return reinterpret_cast<ur_exp_command_buffer_handle_t_ *>(h);
}

} // namespace ur::level_zero::v1

// Base for concrete handle types that are defined once and shared across v1
// and v2 adapters (device, program, sampler, physical_mem). Does NOT
// auto-populate `ddi_table` — the owning adapter's code must assign
// `ddi_table` at construction time from its own `ddi_getter::value()`. That
// way this base does not pull adapter-specific symbols, and the field sits
// at offset 0 as the loader's intercept layer requires.
struct ur_shared_handle_base_t {
  const ur_dditable_t *ddi_table = nullptr;
  ur_shared_mutex Mutex;
  bool OwnNativeHandle = false;

  ur_shared_handle_base_t() = default;
  ur_shared_handle_base_t(const ur_shared_handle_base_t &) = delete;
  ur_shared_handle_base_t &operator=(const ur_shared_handle_base_t &) = delete;
};

// Record for a memory allocation. This structure is used to keep information
// for each memory allocation.
struct MemAllocRecord : ur::level_zero::ur_handle_base_t {
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
