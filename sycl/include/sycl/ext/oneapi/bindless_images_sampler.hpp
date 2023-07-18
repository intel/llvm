//==------ bindless_images_sampler.hpp --- SYCL bindless images ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/sampler.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

struct bindless_image_sampler {

  bindless_image_sampler(sycl::addressing_mode addressing,
                         sycl::coordinate_normalization_mode coordinate,
                         sycl::filtering_mode filtering)
      : addressing(addressing), coordinate(coordinate), filtering(filtering) {}

  bindless_image_sampler(sycl::addressing_mode addressing,
                         sycl::coordinate_normalization_mode coordinate,
                         sycl::filtering_mode filtering,
                         sycl::filtering_mode mipmapFiltering,
                         float minMipmapLevelClamp, float maxMipmapLevelClamp,
                         float maxAnisotropy)
      : addressing(addressing), coordinate(coordinate), filtering(filtering),
        mipmapFiltering(mipmapFiltering),
        minMipmapLevelClamp(minMipmapLevelClamp),
        maxMipmapLevelClamp(maxMipmapLevelClamp), maxAnisotropy(maxAnisotropy) {
  }

  sycl::addressing_mode addressing;
  sycl::coordinate_normalization_mode coordinate;
  sycl::filtering_mode filtering;
  sycl::filtering_mode mipmapFiltering;
  float minMipmapLevelClamp = 0.f;
  float maxMipmapLevelClamp = 0.f;
  float maxAnisotropy = 0.f;
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
