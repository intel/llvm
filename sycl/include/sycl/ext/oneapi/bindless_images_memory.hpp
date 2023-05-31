//==----------- bindless_images_memory.hpp --- SYCL bindless images --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/oneapi/bindless_images_descriptor.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

/// Opaque image memory handle type
struct image_mem_handle {
  using handle_type = void *;
  handle_type raw_handle;
};

/// A struct that represents image memory
struct image_mem {
  using raw_handle_type = image_mem_handle;

  image_mem() = default;
  image_mem(const image_mem &) = delete;
  image_mem(image_mem &&rhs) noexcept;

  image_mem(const context &syclContext, const image_descriptor &desc);
  ~image_mem();

  image_mem &operator=(const image_mem &) = delete;
  image_mem &operator=(image_mem &&) noexcept;

  raw_handle_type get_handle() const { return handle; }
  image_descriptor get_descriptor() const { return descriptor; }
  sycl::context get_context() const { return syclContext; }

  sycl::range<3> get_range() const;
  sycl::image_channel_type get_channel_type() const;
  sycl::image_channel_order get_channel_order() const;
  unsigned int get_num_channels() const;
  unsigned int get_flags() const;
  image_type get_type() const;

  raw_handle_type get_mip_level(const unsigned int level) const;

private:
  raw_handle_type handle{nullptr};
  image_descriptor descriptor;
  sycl::context syclContext;
};

/// Direction to copy data from bindless image handle
/// (Host -> Device) (Device -> Host) etc.
enum image_copy_flags : unsigned int {
  HtoD = 0,
  DtoH = 1,
  DtoD = 2,
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
