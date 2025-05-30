//==---------- kernel_name_str_t.hpp ----- Kernel name type aliases --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/detail/string.hpp>
#include <sycl/detail/string_view.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
using KernelNameStrT = std::string_view;
using KernelNameStrRefT = std::string_view;
using ABINeutralKernelNameStrT = detail::string_view;

inline KernelNameStrT toKernelNameStrT(ABINeutralKernelNameStrT str) {
  return std::string_view(str);
}

#else
using KernelNameStrT = std::string;
using KernelNameStrRefT = const std::string &;
using ABINeutralKernelNameStrT = detail::string;

inline KernelNameStrT toKernelNameStrT(const ABINeutralKernelNameStrT &str) {
  return str.data();
}
#endif

} // namespace detail
} // namespace _V1
} // namespace sycl
