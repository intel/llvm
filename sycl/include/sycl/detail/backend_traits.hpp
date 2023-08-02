//==-------------- backend_traits.hpp - SYCL backend traits ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/backend_types.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

template <backend Backend, typename SYCLObjectT> struct interop;

template <backend Backend, typename T> struct BackendInput;

template <backend Backend, typename T> struct BackendReturn;

template <backend Backend> struct InteropFeatureSupportMap {
  static constexpr bool MakePlatform = false;
  static constexpr bool MakeDevice = false;
  static constexpr bool MakeContext = false;
  static constexpr bool MakeQueue = false;
  static constexpr bool MakeEvent = false;
  static constexpr bool MakeBuffer = false;
  static constexpr bool MakeKernel = false;
  static constexpr bool MakeKernelBundle = false;
  static constexpr bool MakeImage = false;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
