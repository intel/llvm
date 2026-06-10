//==---------- composite_device.hpp - SYCL Composite Device ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/export.hpp>
#include <sycl/detail/info_desc_traits.hpp>
#include <unified-runtime/ur_api.h>

#include <vector>

namespace sycl {
inline namespace _V1 {

class device;

namespace ext::oneapi::experimental {

__SYCL_EXPORT std::vector<device> get_composite_devices();

namespace info::device {

struct component_devices
    : sycl::detail::ur_traits_base<sycl::detail::info_class::device,
                                   UR_DEVICE_INFO_COMPONENT_DEVICES> {
  using return_type = std::vector<sycl::device>;
};

struct composite_device
    : sycl::detail::ur_traits_base<sycl::detail::info_class::device,
                                   UR_DEVICE_INFO_COMPOSITE_DEVICE> {
  using return_type = sycl::device;
};

} // namespace info::device

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
