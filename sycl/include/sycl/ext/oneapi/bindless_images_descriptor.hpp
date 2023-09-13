//==------ bindless_images_descriptor.hpp --- SYCL bindless images ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/array.hpp> // for array
#include <sycl/exception.hpp>    // for errc, exception
#include <sycl/image.hpp>        // for image_channel_order, image_channel_type
#include <sycl/range.hpp>        // for range

#include <algorithm>    // for max
#include <stddef.h>     // for size_t
#include <system_error> // for error_code

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

/// image type enum
enum class image_type : unsigned int {
  standard = 0,
  interop = 1,
  mipmap = 2,
  cubemap = 3, /* Not implemented */
  layered = 4, /* Not implemented */
};

/// A struct to describe the properties of an image.
struct image_descriptor {
  size_t width;
  size_t height;
  size_t depth;
  image_channel_order channel_order;
  image_channel_type channel_type;
  image_type type;
  unsigned int num_levels;

  image_descriptor() = default;

  image_descriptor(range<1> dims, image_channel_order channel_order,
                   image_channel_type channel_type,
                   image_type type = image_type::standard,
                   unsigned int num_levels = 1)
      : width(dims[0]), height(0), depth(0), channel_order(channel_order),
        channel_type(channel_type), type(type), num_levels(num_levels) {}

  image_descriptor(range<2> dims, image_channel_order channel_order,
                   image_channel_type channel_type,
                   image_type type = image_type::standard,
                   unsigned int num_levels = 1)
      : width(dims[0]), height(dims[1]), depth(0), channel_order(channel_order),
        channel_type(channel_type), type(type), num_levels(num_levels) {}

  image_descriptor(range<3> dims, image_channel_order channel_order,
                   image_channel_type channel_type,
                   image_type type = image_type::standard,
                   unsigned int num_levels = 1)
      : width(dims[0]), height(dims[1]), depth(dims[2]),
        channel_order(channel_order), channel_type(channel_type), type(type),
        num_levels(num_levels){};

  /// Get the descriptor for a mipmap level
  image_descriptor get_mip_level_desc(unsigned int level) const {
    // Check that this descriptor describes a mipmap - otherwise throw
    if (this->type != image_type::mipmap)
      throw sycl::exception(
          sycl::errc::invalid,
          "Invalid descriptor `image_type` passed to "
          "`get_mip_level_desc`. A mipmap level descriptor can only be "
          "requested by a descriptor with mipmap image type!");

    // Generate a new descriptor which represents the level accordingly
    // Do not allow height/depth values to be clamped to 1 when naturally 0
    size_t width = std::max<size_t>(this->width >> level, 1);
    size_t height = this->height == 0
                        ? this->height
                        : std::max<size_t>(this->height >> level, 1);
    size_t depth = this->depth == 0 ? this->depth
                                    : std::max<size_t>(this->depth >> level, 1);

    // This will generate the new descriptor with image_type standard
    // since individual mip levels are standard images
    sycl::ext::oneapi::experimental::image_descriptor levelDesc(
        {width, height, depth}, this->channel_order, this->channel_type);

    return levelDesc;
  }
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
