//==--------- interop_common.hpp --- Common Interop Definitions ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// Types of external memory handles.
enum class external_mem_handle_type {
  opaque_fd = 0,
  win32_nt_handle = 1,
  win32_nt_dx12_resource = 2,
  dma_buf = 3,
  win32_nt_dx11_resource = 4,
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
