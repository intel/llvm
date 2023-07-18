//==----------- bindless_images_memory.hpp --- SYCL bindless images --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/context.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/oneapi/bindless_images_descriptor.hpp>
#include <sycl/image.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {

// Forward declaration
class queue;

namespace ext {
namespace oneapi {
namespace experimental {

/// Opaque image memory handle type
struct image_mem_handle {
  using handle_type = void *;
  handle_type raw_handle;
};

namespace detail {

class image_mem_impl {
  using raw_handle_type = image_mem_handle;

public:
  __SYCL_EXPORT image_mem_impl(const image_descriptor &desc,
                               const device &syclDevice,
                               const context &syclContext);
  __SYCL_EXPORT ~image_mem_impl();

  raw_handle_type get_handle() const { return handle; }
  const image_descriptor &get_descriptor() const { return descriptor; }
  sycl::device get_device() const { return syclDevice; }
  sycl::context get_context() const { return syclContext; }

private:
  raw_handle_type handle{nullptr};
  image_descriptor descriptor;
  sycl::device syclDevice;
  sycl::context syclContext;
};

} // namespace detail

/// A class that represents image memory
class image_mem {
  using raw_handle_type = image_mem_handle;

public:
  image_mem() = default;
  image_mem(const image_mem &) = default;
  image_mem(image_mem &&rhs) = default;

  __SYCL_EXPORT image_mem(const image_descriptor &desc,
                          const device &syclDevice, const context &syclContext);
  __SYCL_EXPORT image_mem(const image_descriptor &desc, const queue &syclQueue);
  ~image_mem() = default;

  image_mem &operator=(const image_mem &) = default;
  image_mem &operator=(image_mem &&) = default;

  bool operator==(const image_mem &rhs) const { return impl == rhs.impl; }
  bool operator!=(const image_mem &rhs) const { return !(*this == rhs); }

  raw_handle_type get_handle() const { return impl->get_handle(); }
  const image_descriptor &get_descriptor() const {
    return impl->get_descriptor();
  }
  sycl::device get_device() const { return impl->get_device(); }
  sycl::context get_context() const { return impl->get_context(); }

  __SYCL_EXPORT sycl::range<3> get_range() const;
  __SYCL_EXPORT sycl::image_channel_type get_channel_type() const;
  __SYCL_EXPORT sycl::image_channel_order get_channel_order() const;
  __SYCL_EXPORT unsigned int get_num_channels() const;
  __SYCL_EXPORT image_type get_type() const;

  __SYCL_EXPORT raw_handle_type __SYCL_EXPORT
  get_mip_level_mem_handle(const unsigned int level) const;

protected:
  std::shared_ptr<detail::image_mem_impl> impl;

  template <class Obj>
  friend decltype(Obj::impl)
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);
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

namespace std {
template <> struct hash<sycl::ext::oneapi::experimental::image_mem> {
  size_t operator()(
      const sycl::ext::oneapi::experimental::image_mem &image_mem) const {
    return hash<std::shared_ptr<
        sycl::ext::oneapi::experimental::detail::image_mem_impl>>()(
        sycl::detail::getSyclObjImpl(image_mem));
  }
};
} // namespace std
