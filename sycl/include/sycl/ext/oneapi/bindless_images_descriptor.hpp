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
  mipmap = 1,
  array = 2,
  cubemap = 3, /* Not implemented */
  interop = 4,
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
  unsigned int array_size;

  image_descriptor() = default;

  image_descriptor(range<1> dims, image_channel_order channel_order,
                   image_channel_type channel_type,
                   image_type type = image_type::standard,
                   unsigned int num_levels = 1, unsigned int array_size = 1)
      : width(dims[0]), height(0), depth(0), channel_order(channel_order),
        channel_type(channel_type), type(type), num_levels(num_levels),
        array_size(array_size) {
    verify();
  }

  image_descriptor(range<2> dims, image_channel_order channel_order,
                   image_channel_type channel_type,
                   image_type type = image_type::standard,
                   unsigned int num_levels = 1, unsigned int array_size = 1)
      : width(dims[0]), height(dims[1]), depth(0), channel_order(channel_order),
        channel_type(channel_type), type(type), num_levels(num_levels),
        array_size(array_size) {
    verify();
  }

  image_descriptor(range<3> dims, image_channel_order channel_order,
                   image_channel_type channel_type,
                   image_type type = image_type::standard,
                   unsigned int num_levels = 1, unsigned int array_size = 1)
      : width(dims[0]), height(dims[1]), depth(dims[2]),
        channel_order(channel_order), channel_type(channel_type), type(type),
        num_levels(num_levels), array_size(array_size) {
    verify();
  };

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

    levelDesc.verify();
    return levelDesc;
  }

  void verify() const {
    switch (this->type) {
    case image_type::standard:
      if (this->array_size > 1) {
        // Not a standard image.
        throw sycl::exception(
            sycl::errc::invalid,
            "Standard images cannot have array_size greater than 1! Use "
            "image_type::array for image arrays.");
      }
      if (this->num_levels > 1) {
        // Image arrays cannot be mipmaps.
        throw sycl::exception(
            sycl::errc::invalid,
            "Standard images cannot have num_levels greater than 1! Use "
            "image_type::mipmap for mipmap images.");
      }
      return;

    case image_type::array:
      if (this->array_size <= 1) {
        // Not an image array.
        throw sycl::exception(sycl::errc::invalid,
                              "Image array must have array_size greater than "
                              "1! Use image_type::standard otherwise.");
      }
      if (this->depth != 0) {
        // Image arrays must only be 1D or 2D.
        throw sycl::exception(sycl::errc::invalid,
                              "Cannot have 3D image arrays! Either depth must "
                              "be 0 or array_size must be 1.");
      }
      if (this->num_levels != 1) {
        // Image arrays cannot be mipmaps.
        throw sycl::exception(sycl::errc::invalid,
                              "Cannot have mipmap image arrays! Either "
                              "num_levels or array_size must be 1.");
      }
      return;

    case image_type::mipmap:
      if (this->array_size > 1) {
        // Mipmap images cannot be arrays.
        throw sycl::exception(
            sycl::errc::invalid,
            "Mipmap images cannot have array_size greater than 1! Use "
            "image_type::array for image arrays.");
      }
      if (this->num_levels <= 1) {
        // Mipmaps must have more than one level.
        throw sycl::exception(sycl::errc::invalid,
                              "Mipmap images must have num_levels greater than "
                              "1! Use image_type::standard otherwise.");
      }
      return;

    case image_type::interop:
      // No checks to be made.
      return;

    default:
      // Invalid image type.
      throw sycl::exception(sycl::errc::invalid,
                            "Invalid image descriptor image type");
    }
  }
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
