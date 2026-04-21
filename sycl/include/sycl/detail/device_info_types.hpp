//==---- device_info_types.hpp - SYCL device-info-only type aliases --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Isolates uuid_type / luid_type from the broad type_traits.hpp so that
// headers which include type_traits.hpp do not pay for <array>.  Only the
// device-info descriptor path (info/info_desc.hpp) and the runtime
// (source/detail/device_impl.hpp, via info_desc.hpp) need these types.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <array> // for array

namespace sycl {
inline namespace _V1 {
namespace detail {

// Types for Intel's device UUID and device LUID extension.
// For details about this extension, see
// sycl/doc/extensions/supported/sycl_ext_intel_device_info.md
using uuid_type = std::array<unsigned char, 16>;
using luid_type = std::array<unsigned char, 8>;

} // namespace detail
} // namespace _V1
} // namespace sycl
