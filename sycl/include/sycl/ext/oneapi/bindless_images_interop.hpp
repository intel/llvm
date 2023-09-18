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

/// Opaque interop memory handle type
struct interop_mem_handle {
  using raw_handle_type = pi_uint64;
  raw_handle_type raw_handle;
};

/// External memory file descriptor type
struct external_mem_fd {
  int file_descriptor;
};

/// Windows external memory type
struct external_mem_win32 {
  void *handle;
  const void *name;
};

/// Opaque external memory descriptor type
template <typename HandleType> struct external_mem_descriptor {
  HandleType external_handle;
  size_t size_in_bytes;
};

/// Opaque interop semaphore handle type
struct interop_semaphore_handle {
  using raw_handle_type = pi_uint64;
  raw_handle_type raw_handle;
};

/// External semaphore file descriptor type
struct external_semaphore_fd {
  int file_descriptor;
};

/// Opaque external semaphore descriptor type
template <typename HandleType> struct external_semaphore_descriptor {
  HandleType external_handle;
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
