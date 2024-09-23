//==--------------------------- imf_integer_utils.hpp ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// C++ APIs for simple integer util funcitons
//===----------------------------------------------------------------------===//

#pragma once

extern "C" {
unsigned __imf_brev(unsigned);
unsigned long long __imf_brevll(unsigned long long);
unsigned __imf_byte_perm(unsigned, unsigned, unsigned);
long long __imf_llmax(long long x, long long y);
long long __imf_llmin(long long x, long long y);
int __imf_max(int x, int y);
int __imf_min(int x, int y);
unsigned long long __imf_ullmax(unsigned long long x, unsigned long long y);
unsigned long long __imf_ullmin(unsigned long long x, unsigned long long y);
unsigned __imf_umax(unsigned x, unsigned y);
unsigned __imf_umin(unsigned x, unsigned y);
int __imf_clz(int);
int __imf_clzll(long long);
int __imf_ffs(int);
int __imf_ffsll(long long);
int __imf_mul24(int, int);
int __imf_mulhi(int, int);
long long __imf_mul64hi(long long, long long);
int __imf_popc(unsigned);
int __imf_popcll(unsigned long long);
int __imf_rhadd(int, int);
int __imf_hadd(int, int);
unsigned __imf_sad(int, int, unsigned);
unsigned __imf_uhadd(unsigned, unsigned);
unsigned __imf_umul24(unsigned, unsigned);
unsigned __imf_umulhi(unsigned, unsigned);
unsigned long long __imf_umul64hi(unsigned long long, unsigned long long);
unsigned __imf_urhadd(unsigned, unsigned);
unsigned __imf_usad(unsigned, unsigned, unsigned);
}

namespace sycl {
inline namespace _V1 {
namespace ext::intel::math {

template <typename Tp = unsigned> unsigned brev(Tp x) { return __imf_brev(x); }

template <typename Tp = unsigned long long> unsigned long long brevll(Tp x) {
  return __imf_brevll(x);
}

template <typename Tp = unsigned> unsigned byte_perm(Tp x, Tp y, Tp z) {
  return __imf_byte_perm(x, y, z);
}

template <typename Tp = int> int max(Tp x, Tp y) { return __imf_max(x, y); }

template <typename Tp = int> int min(Tp x, Tp y) { return __imf_min(x, y); }

template <typename Tp = unsigned> unsigned umax(Tp x, Tp y) {
  return __imf_umax(x, y);
}

template <typename Tp = unsigned> unsigned umin(Tp x, Tp y) {
  return __imf_umin(x, y);
}

template <typename Tp = long long> long long llmax(Tp x, Tp y) {
  return __imf_llmax(x, y);
}

template <typename Tp = long long> long long llmin(Tp x, Tp y) {
  return __imf_llmin(x, y);
}

template <typename Tp = unsigned long long>
unsigned long long ullmax(Tp x, Tp y) {
  return __imf_ullmax(x, y);
}

template <typename Tp = unsigned long long>
unsigned long long ullmin(Tp x, Tp y) {
  return __imf_ullmin(x, y);
}

template <typename Tp = int> int clz(Tp x) { return __imf_clz(x); }

template <typename Tp = long long> int clzll(Tp x) { return __imf_clzll(x); }

template <typename Tp = int> int ffs(Tp x) { return __imf_ffs(x); }

template <typename Tp = long long> int ffsll(Tp x) { return __imf_ffsll(x); }

template <typename Tp = int> int hadd(Tp x, Tp y) { return __imf_hadd(x, y); }

template <typename Tp = int> int rhadd(Tp x, Tp y) { return __imf_rhadd(x, y); }

template <typename Tp = unsigned> unsigned urhadd(Tp x, Tp y) {
  return __imf_urhadd(x, y);
}

template <typename Tp = int> int mul24(Tp x, Tp y) { return __imf_mul24(x, y); }

template <typename Tp = unsigned> unsigned umul24(Tp x, Tp y) {
  return __imf_umul24(x, y);
}

template <typename Tp = int> int mulhi(Tp x, Tp y) { return __imf_mulhi(x, y); }

template <typename Tp = unsigned> unsigned umulhi(Tp x, Tp y) {
  return __imf_umulhi(x, y);
}

template <typename Tp = long long> long long mul64hi(Tp x, Tp y) {
  return __imf_mul64hi(x, y);
}

template <typename Tp = unsigned long long>
unsigned long long umul64hi(Tp x, Tp y) {
  return __imf_umul64hi(x, y);
}

template <typename Tp = unsigned> int popc(Tp x) { return __imf_popc(x); }

template <typename Tp = unsigned long long> int popcll(Tp x) {
  return __imf_popcll(x);
}

template <typename Tp1 = int, typename Tp2 = unsigned>
unsigned sad(Tp1 x, Tp1 y, Tp2 z) {
  return __imf_sad(x, y, z);
}

template <typename Tp = unsigned> unsigned usad(Tp x, Tp y, Tp z) {
  return __imf_usad(x, y, z);
}

template <typename Tp = unsigned> unsigned uhadd(Tp x, Tp y) {
  return __imf_uhadd(x, y);
}

} // namespace ext::intel::math
} // namespace _V1
} // namespace sycl
