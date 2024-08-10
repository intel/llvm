//==---------- bindless_images_mem_handle.hpp --- SYCL bindless images -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sycl/ur_api.h"

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
/// Opaque image memory handle type
struct image_mem_handle {
  using raw_handle_type = ur_exp_image_mem_native_handle_t;
  raw_handle_type raw_handle;
};
} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
