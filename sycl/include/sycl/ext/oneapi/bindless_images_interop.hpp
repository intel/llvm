//==----------- bindless_images_interop.hpp --- SYCL bindless images -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "interop_common.hpp" // For external_mem_handle_type.

#include <stddef.h> // For size_t.
#include <ur_api.h>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// Types of external semaphore handles
enum class external_semaphore_handle_type {
  opaque_fd = 0,
  win32_nt_handle = 1,
  win32_nt_dx12_fence = 2,
  timeline_fd = 3,
  timeline_win32_nt_handle = 4,
};

/// Opaque external memory handle type
struct external_mem {
  using raw_handle_type = ur_exp_external_mem_handle_t;
  raw_handle_type raw_handle;
};

/// Imported opaque external semaphore
struct external_semaphore {
  using raw_handle_type = ur_exp_external_semaphore_handle_t;
  raw_handle_type raw_handle;
  external_semaphore_handle_type handle_type;
};

// External resource file descriptor type
struct resource_fd {
  int file_descriptor;
};

// Windows external handle type
struct resource_win32_handle {
  void *handle;
};

// Windows external name type
struct resource_win32_name {
  const void *name;
};

/// Opaque external memory descriptor type
template <typename ResourceType> struct external_mem_descriptor {
  ResourceType external_resource;
  external_mem_handle_type handle_type;
  size_t size_in_bytes;
};

// Opaque external semaphore descriptor type
template <typename ResourceType> struct external_semaphore_descriptor {
  ResourceType external_resource;
  external_semaphore_handle_type handle_type;
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
