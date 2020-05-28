//==----------- bit_cast.hpp --- SYCL bit_cast -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/defines.hpp>
#include <CL/sycl/detail/helpers.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace intel {

template <typename To, typename From>
constexpr To bit_cast(const From &from) noexcept {
  return sycl::detail::bit_cast<To>(from);
}

} // namespace intel
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)