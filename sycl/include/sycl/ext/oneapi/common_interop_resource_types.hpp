//==--- common_interop_resource_types.hpp --- SYCL interop resource types --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file contains interop resource types common to the Bindless Images and
// Memory Export extensions.

#pragma once

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// Types of external memory handles
enum class external_mem_handle_type {
  opaque_fd = 0,
  win32_nt_handle = 1,
  win32_nt_dx12_resource = 2,
};

// External resource file descriptor type
struct resource_fd {
  int file_descriptor;
};

// Windows external handle type
struct resource_win32_handle {
  void *handle;
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
