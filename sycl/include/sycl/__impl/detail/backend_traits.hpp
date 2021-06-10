//==-------------- backend_traits.hpp - SYCL backend traits ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/backend_types.hpp>
#include <sycl/__impl/detail/common.hpp>

namespace __sycl_internal {
inline namespace __v1 {
namespace detail {
template <backend Backend> struct InteropFeatureSupportMap {
  static constexpr bool MakePlatform = false;
  static constexpr bool MakeDevice = false;
  static constexpr bool MakeContext = false;
  static constexpr bool MakeQueue = false;
  static constexpr bool MakeEvent = false;
  static constexpr bool MakeBuffer = false;
  static constexpr bool MakeKernel = false;
};
} // namespace detail
} // namespace sycl
} // namespace __sycl_internal
