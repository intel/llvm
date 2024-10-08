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

namespace detail {

inline image_channel_order
get_image_default_channel_order(unsigned int num_channels) {
  switch (num_channels) {
  case 1:
    return image_channel_order::r;
  case 2:
    return image_channel_order::rg;
  case 4:
    return image_channel_order::rgba;
  default:
    assert(false && "Invalid channel number");
    return static_cast<image_channel_order>(0);
  }
}

} // namespace detail

/// image type enum
enum class image_type : unsigned int {
  standard = 0,
  mipmap = 1,
  array = 2,
  cubemap = 3,
};

/// A struct to describe the properties of an image.
struct image_descriptor {
  size_t width{0};
  size_t height{0};
  size_t depth{0};
  unsigned int num_channels{4};
  image_channel_type channel_type{image_channel_type::fp32};
  image_type type{image_type::standard};
  unsigned int num_levels{1};
  unsigned int array_size{1};

  image_descriptor() = default;

  image_descriptor(range<1> dims, unsigned int num_channels,
                   image_channel_type channel_type,
                   image_type type = image_type::standard,
                   unsigned int num_levels = 1, unsigned int array_size = 1)
      : width(dims[0]), height(0), depth(0), num_channels(num_channels),
        channel_type(channel_type), type(type), num_levels(num_levels),
        array_size(array_size) {
    verify();
  }

  image_descriptor(range<2> dims, unsigned int num_channels,
                   image_channel_type channel_type,
                   image_type type = image_type::standard,
                   unsigned int num_levels = 1, unsigned int array_size = 1)
      : width(dims[0]), height(dims[1]), depth(0), num_channels(num_channels),
        channel_type(channel_type), type(type), num_levels(num_levels),
        array_size(array_size) {
    verify();
  }

  image_descriptor(range<3> dims, unsigned int num_channels,
                   image_channel_type channel_type,
                   image_type type = image_type::standard,
                   unsigned int num_levels = 1, unsigned int array_size = 1)
      : width(dims[0]), height(dims[1]), depth(dims[2]),
        num_channels(num_channels), channel_type(channel_type), type(type),
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
    image_descriptor levelDesc({width, height, depth}, this->num_channels,
                               this->channel_type);

    levelDesc.verify();
    return levelDesc;
  }

  void verify() const {

    if (this->num_channels != 1 && this->num_channels != 2 &&
        this->num_channels != 4) {
      // Images can only have 1, 2, or 4 channels.
      throw sycl::exception(sycl::errc::invalid,
                            "Images must have only 1, 2, or 4 channels! Use a "
                            "valid number of channels instead.");
    }

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

    case image_type::cubemap:
      if (this->array_size != 6) {
        // Cubemaps must have an array size of 6.
        throw sycl::exception(sycl::errc::invalid,
                              "Cubemap images must have array_size of 6 only! "
                              "Use image_type::array instead.");
      }
      if (this->depth != 0 || this->height == 0 ||
          this->width != this->height) {
        // Cubemaps must be 2D
        throw sycl::exception(
            sycl::errc::invalid,
            "Cubemap images must be square with valid and equivalent width and "
            "height! Use image_type::array instead.");
      }
      if (this->num_levels != 1) {
        // Cubemaps cannot be mipmaps.
        throw sycl::exception(sycl::errc::invalid,
                              "Cannot have mipmap cubemaps! Either num_levels "
                              "or array_size must be 1.");
      }
      return;
    }
  }
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
