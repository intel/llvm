//==------------------- group_mask.cpp -------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/ext/oneapi/group_mask.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
group_mask operator&(const group_mask &lhs, const group_mask &rhs) {
  auto Res = lhs;
  Res &= rhs;
  return Res;
}
group_mask operator|(const group_mask &lhs, const group_mask &rhs) {
  auto Res = lhs;
  Res |= rhs;
  return Res;
}

group_mask operator^(const group_mask &lhs, const group_mask &rhs) {
  auto Res = lhs;
  Res ^= rhs;
  return Res;
}
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
