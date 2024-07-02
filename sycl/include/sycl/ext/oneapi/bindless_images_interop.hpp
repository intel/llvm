//==----------- bindless_images_interop.hpp --- SYCL bindless images -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/pi.h> // for pi_uint64

#include <stddef.h> // for size_t

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// Types of external memory handles
enum class external_mem_handle_type {
  opaque_fd = 0,
  win32_nt_handle = 1,
  win32_nt_dx12_resource = 2,
};

// Types of external semaphore handles
enum class external_semaphore_handle_type {
  opaque_fd = 0,
  win32_nt_handle = 1,
  win32_nt_dx12_fence = 2,
};

/// Opaque interop memory handle type
struct interop_mem_handle {
  using raw_handle_type = pi_uint64;
  raw_handle_type raw_handle;
};

/// Opaque interop semaphore handle type
struct interop_semaphore_handle {
  using raw_handle_type = pi_uint64;
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

/// EVERYTHING BELOW IS DEPRECATED

/// External memory file descriptor type
struct external_mem_fd {
  int file_descriptor;
};

/// Windows external memory type
struct external_mem_win32 {
  void *handle;
  const void *name;
};

/// External semaphore file descriptor type
struct external_semaphore_fd {
  int file_descriptor;
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
