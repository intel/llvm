//==----------- dot_product.hpp ------- SYCL dot-product -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DP4A extension

#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {

union Us {
  char s[4];
  int32_t i;
};
union Uu {
  unsigned char s[4];
  uint32_t i;
};

int32_t dot_acc(int32_t pa, int32_t pb, int32_t c) {
  Us a = *(reinterpret_cast<Us *>(&pa));
  Us b = *(reinterpret_cast<Us *>(&pb));
  return a.s[0] * b.s[0] + a.s[1] * b.s[1] + a.s[2] * b.s[2] + a.s[3] * b.s[3] +
         c;
}

int32_t dot_acc(uint32_t pa, uint32_t pb, int32_t c) {
  Uu a = *(reinterpret_cast<Uu *>(&pa));
  Uu b = *(reinterpret_cast<Uu *>(&pb));
  return a.s[0] * b.s[0] + a.s[1] * b.s[1] + a.s[2] * b.s[2] + a.s[3] * b.s[3] +
         c;
}

int32_t dot_acc(int32_t pa, uint32_t pb, int32_t c) {
  Us a = *(reinterpret_cast<Us *>(&pa));
  Uu b = *(reinterpret_cast<Uu *>(&pb));
  return a.s[0] * b.s[0] + a.s[1] * b.s[1] + a.s[2] * b.s[2] + a.s[3] * b.s[3] +
         c;
}

int32_t dot_acc(uint32_t pa, int32_t pb, int32_t c) {
  Uu a = *(reinterpret_cast<Uu *>(&pa));
  Us b = *(reinterpret_cast<Us *>(&pb));
  return a.s[0] * b.s[0] + a.s[1] * b.s[1] + a.s[2] * b.s[2] + a.s[3] * b.s[3] +
         c;
}

int32_t dot_acc(vec<int8_t, 4> a, vec<int8_t, 4> b, int32_t c) {
  return a.s0() * b.s0() + a.s1() * b.s1() + a.s2() * b.s2() + a.s3() * b.s3() +
         c;
}

int32_t dot_acc(vec<uint8_t, 4> a, vec<uint8_t, 4> b, int32_t c) {
  return a.s0() * b.s0() + a.s1() * b.s1() + a.s2() * b.s2() + a.s3() * b.s3() +
         c;
}

int32_t dot_acc(vec<uint8_t, 4> a, vec<int8_t, 4> b, int32_t c) {
  return a.s0() * b.s0() + a.s1() * b.s1() + a.s2() * b.s2() + a.s3() * b.s3() +
         c;
}

int32_t dot_acc(vec<int8_t, 4> a, vec<uint8_t, 4> b, int32_t c) {
  return a.s0() * b.s0() + a.s1() * b.s1() + a.s2() * b.s2() + a.s3() * b.s3() +
         c;
}

} // namespace oneapi
} // namespace ext

namespace __SYCL2020_DEPRECATED("use 'ext::oneapi' instead") ONEAPI {
  using namespace ext::oneapi;
}
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
