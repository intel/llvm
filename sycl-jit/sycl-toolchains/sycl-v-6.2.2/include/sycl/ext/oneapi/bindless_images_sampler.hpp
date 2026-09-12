//==------ bindless_images_sampler.hpp --- SYCL bindless images ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/sampler.hpp>
#include <ur_api.h>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

/// cubemap filtering mode enum
enum class cubemap_filtering_mode : unsigned int {
  disjointed = UR_EXP_SAMPLER_CUBEMAP_FILTER_MODE_DISJOINTED,
  seamless = UR_EXP_SAMPLER_CUBEMAP_FILTER_MODE_SEAMLESS,
};

struct bindless_image_sampler {

  bindless_image_sampler(sycl::addressing_mode addr[3],
                         sycl::coordinate_normalization_mode coordinate,
                         sycl::filtering_mode filtering)
      : addressing{addr[0], addr[1], addr[2]}, coordinate(coordinate),
        filtering(filtering) {}

  bindless_image_sampler(sycl::addressing_mode addr[3],
                         sycl::coordinate_normalization_mode coordinate,
                         sycl::filtering_mode filtering,
                         sycl::filtering_mode mipmapFiltering,
                         float minMipmapLevelClamp, float maxMipmapLevelClamp,
                         float maxAnisotropy)
      : addressing{addr[0], addr[1], addr[2]}, coordinate(coordinate),
        filtering(filtering), mipmap_filtering(mipmapFiltering),
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

  bindless_image_sampler(sycl::addressing_mode addr,
                         sycl::coordinate_normalization_mode coordinate,
                         sycl::filtering_mode filtering,
                         cubemap_filtering_mode cubemapFiltering)
      : addressing{addr, addr, addr}, coordinate(coordinate),
        filtering(filtering), cubemap_filtering(cubemapFiltering) {}

  bindless_image_sampler(sycl::addressing_mode addr[3],
                         sycl::coordinate_normalization_mode coordinate,
                         sycl::filtering_mode filtering,
                         cubemap_filtering_mode cubemapFiltering)
      : addressing{addr[0], addr[1], addr[2]}, coordinate(coordinate),
        filtering(filtering), cubemap_filtering(cubemapFiltering) {}

  sycl::addressing_mode addressing[3] = {sycl::addressing_mode::none};
  sycl::coordinate_normalization_mode coordinate =
      sycl::coordinate_normalization_mode::unnormalized;
  sycl::filtering_mode filtering = sycl::filtering_mode::nearest;
  sycl::filtering_mode mipmap_filtering = sycl::filtering_mode::nearest;
  cubemap_filtering_mode cubemap_filtering = cubemap_filtering_mode::disjointed;
  float min_mipmap_level_clamp = 0.f;
  float max_mipmap_level_clamp = 0.f;
  float max_anisotropy = 0.f;
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
