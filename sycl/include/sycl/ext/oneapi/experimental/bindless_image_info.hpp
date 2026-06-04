//==-- bindless_image_info.hpp - bindless image device info traits ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/info_desc_traits.hpp>
#include <unified-runtime/ur_api.h>

#include <cstddef>
#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental::info::device {

template <ur_device_info_t UrCode>
using device_traits =
    sycl::detail::ur_traits_base<sycl::detail::info_class::device, UrCode>;

struct image_row_pitch_align
    : device_traits<UR_DEVICE_INFO_IMAGE_PITCH_ALIGN_EXP> {
  using return_type = uint32_t;
};
struct max_image_linear_row_pitch
    : device_traits<UR_DEVICE_INFO_MAX_IMAGE_LINEAR_PITCH_EXP> {
  using return_type = size_t;
};
struct max_image_linear_width
    : device_traits<UR_DEVICE_INFO_MAX_IMAGE_LINEAR_WIDTH_EXP> {
  using return_type = size_t;
};
struct max_image_linear_height
    : device_traits<UR_DEVICE_INFO_MAX_IMAGE_LINEAR_HEIGHT_EXP> {
  using return_type = size_t;
};
struct mipmap_max_anisotropy
    : device_traits<UR_DEVICE_INFO_MIPMAP_MAX_ANISOTROPY_EXP> {
  using return_type = float;
};

} // namespace ext::oneapi::experimental::info::device
} // namespace _V1
} // namespace sycl
