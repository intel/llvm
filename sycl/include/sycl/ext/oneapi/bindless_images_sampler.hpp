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
inline namespace _V1 {
namespace ext::oneapi::experimental {

struct bindless_image_sampler {

  bindless_image_sampler(sycl::addressing_mode addressing[3],
                         sycl::coordinate_normalization_mode coordinate,
                         sycl::filtering_mode filtering)
      : addressing{addressing[0], addressing[1], addressing[2]},
        coordinate(coordinate), filtering(filtering) {}

  bindless_image_sampler(sycl::addressing_mode addressing[3],
                         sycl::coordinate_normalization_mode coordinate,
                         sycl::filtering_mode filtering,
                         sycl::filtering_mode mipmapFiltering,
                         float minMipmapLevelClamp, float maxMipmapLevelClamp,
                         float maxAnisotropy)
      : addressing{addressing[0], addressing[1], addressing[2]},
        coordinate(coordinate), filtering(filtering),
        mipmap_filtering(mipmapFiltering),
        min_mipmap_level_clamp(minMipmapLevelClamp),
        max_mipmap_level_clamp(maxMipmapLevelClamp),
        max_anisotropy(maxAnisotropy) {}

  bindless_image_sampler() = default;

  bindless_image_sampler(sycl::addressing_mode addr,
                         sycl::coordinate_normalization_mode coordinate,
                         sycl::filtering_mode filtering)
      : addressing{addr, addr, addr}, coordinate(coordinate),
        filtering(filtering) {}

  bindless_image_sampler(sycl::addressing_mode addr,
                         sycl::coordinate_normalization_mode coordinate,
                         sycl::filtering_mode filtering,
                         sycl::filtering_mode mipmapFiltering,
                         float minMipmapLevelClamp, float maxMipmapLevelClamp,
                         float maxAnisotropy)
      : addressing{addr, addr, addr}, coordinate(coordinate),
        filtering(filtering), mipmap_filtering(mipmapFiltering),
        min_mipmap_level_clamp(minMipmapLevelClamp),
        max_mipmap_level_clamp(maxMipmapLevelClamp),
        max_anisotropy(maxAnisotropy) {}

  sycl::addressing_mode addressing[3] = {sycl::addressing_mode::none};
  sycl::coordinate_normalization_mode coordinate =
      sycl::coordinate_normalization_mode::unnormalized;
  sycl::filtering_mode filtering = sycl::filtering_mode::nearest;
  sycl::filtering_mode mipmap_filtering = sycl::filtering_mode::nearest;
  float min_mipmap_level_clamp = 0.f;
  float max_mipmap_level_clamp = 0.f;
  float max_anisotropy = 0.f;
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
