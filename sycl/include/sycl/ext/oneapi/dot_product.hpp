//==----------- dot_product.hpp ------- SYCL dot-product -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// DP4A extension

#pragma once

#include <sycl/bit_cast.hpp>
#include <sycl/vector.hpp>

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {

namespace detail {

// Scalar (portable) fallback implementations. The four combinations differ
// only in how each of the four packed bytes is extended to 32-bit before the
// multiply (signed vs. unsigned).
// Values are promoted to `int` for the multiply.
inline int32_t dp4a_ss_fallback(int32_t pa, int32_t pb, int32_t c) {
  const int8_t *a = reinterpret_cast<const int8_t *>(&pa);
  const int8_t *b = reinterpret_cast<const int8_t *>(&pb);
  return int{a[0]} * int{b[0]} + int{a[1]} * int{b[1]} +
         int{a[2]} * int{b[2]} + int{a[3]} * int{b[3]} + c;
}
inline int32_t dp4a_uu_fallback(uint32_t pa, uint32_t pb, int32_t c) {
  const uint8_t *a = reinterpret_cast<const uint8_t *>(&pa);
  const uint8_t *b = reinterpret_cast<const uint8_t *>(&pb);
  return int{a[0]} * int{b[0]} + int{a[1]} * int{b[1]} +
         int{a[2]} * int{b[2]} + int{a[3]} * int{b[3]} + c;
}
inline int32_t dp4a_su_fallback(int32_t pa, uint32_t pb, int32_t c) {
  const int8_t *a = reinterpret_cast<const int8_t *>(&pa);
  const uint8_t *b = reinterpret_cast<const uint8_t *>(&pb);
  return int{a[0]} * int{b[0]} + int{a[1]} * int{b[1]} +
         int{a[2]} * int{b[2]} + int{a[3]} * int{b[3]} + c;
}
inline int32_t dp4a_us_fallback(uint32_t pa, int32_t pb, int32_t c) {
  const uint8_t *a = reinterpret_cast<const uint8_t *>(&pa);
  const int8_t *b = reinterpret_cast<const int8_t *>(&pb);
  return int{a[0]} * int{b[0]} + int{a[1]} * int{b[1]} +
         int{a[2]} * int{b[2]} + int{a[3]} * int{b[3]} + c;
}

// Hardware-accelerated implementations.
//
// NVPTX (CUDA):  the `dp4a` instruction is available since sm_61 (Pascal).
//                PTX supports independent signedness of the two operands via
//                `dp4a.atype.btype` where atype/btype in {s32, u32}.
// AMDGCN (HIP):  `__builtin_amdgcn_sdot4` / `__builtin_amdgcn_udot4` map to the
//                `v_dot4_i32_i8` / `v_dot4_u32_u8` instructions available on
//                gfx906+, CDNA and RDNA2 (sdot4 needs `dot1-insts`, udot4 needs
//                `dot7-insts`). Mixed-signedness dot products use
//                `__builtin_amdgcn_sudot4` (`v_dot4_i32_iu8`, needs
//                `dot8-insts`), whose per-operand sign flags exist only on gfx11
//                (RDNA3) and gfx12/RDNA4 -- the CDNA3 gfx94x parts, including
//                MI300, lack `dot8-insts`. Since RDNA3+ dropped `v_dot4_i32_i8`,
//                the signed-signed case also falls through to `sudot4` (both
//                operands signed) there. Every path is guarded with
//                `__builtin_amdgcn_is_invocable` (a compile-time constant), so
//                architectures that lack a given instruction use the scalar
//                fallback.

inline int32_t dp4a_ss(int32_t a, int32_t b, int32_t c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                      \
    (__SYCL_CUDA_ARCH__ >= 610)
  int32_t d;
  asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_is_invocable) &&                            \
    __has_builtin(__builtin_amdgcn_sdot4)
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_sdot4))
    return __builtin_amdgcn_sdot4(a, b, c, false);
#endif
#if __has_builtin(__builtin_amdgcn_is_invocable) &&                            \
    __has_builtin(__builtin_amdgcn_sudot4)
  // RDNA3+/gfx12 dropped `v_dot4_i32_i8` (sdot4) but can still perform a signed
  // dot product via `v_dot4_i32_iu8` (sudot4) with both operands marked signed.
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_sudot4))
    return __builtin_amdgcn_sudot4(/*a_sign=*/true, a, /*b_sign=*/true, b, c,
                                   false);
#endif
  return dp4a_ss_fallback(a, b, c);
#else
  return dp4a_ss_fallback(a, b, c);
#endif
}

inline int32_t dp4a_uu(uint32_t a, uint32_t b, int32_t c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                      \
    (__SYCL_CUDA_ARCH__ >= 610)
  int32_t d;
  asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_is_invocable) &&                            \
    __has_builtin(__builtin_amdgcn_udot4)
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_udot4))
    return __builtin_amdgcn_udot4(a, b, c, false);
#endif
  return dp4a_uu_fallback(a, b, c);
#else
  return dp4a_uu_fallback(a, b, c);
#endif
}

inline int32_t dp4a_su(int32_t a, uint32_t b, int32_t c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                      \
    (__SYCL_CUDA_ARCH__ >= 610)
  int32_t d;
  asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_is_invocable) &&                            \
    __has_builtin(__builtin_amdgcn_sudot4)
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_sudot4))
    return __builtin_amdgcn_sudot4(/*a_sign=*/true, a, /*b_sign=*/false,
                                   static_cast<int32_t>(b), c, false);
#endif
  return dp4a_su_fallback(a, b, c);
#else
  return dp4a_su_fallback(a, b, c);
#endif
}

inline int32_t dp4a_us(uint32_t a, int32_t b, int32_t c) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) &&                      \
    (__SYCL_CUDA_ARCH__ >= 610)
  int32_t d;
  asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
#if __has_builtin(__builtin_amdgcn_is_invocable) &&                            \
    __has_builtin(__builtin_amdgcn_sudot4)
  if (__builtin_amdgcn_is_invocable(__builtin_amdgcn_sudot4))
    return __builtin_amdgcn_sudot4(/*a_sign=*/false, static_cast<int32_t>(a),
                                   /*b_sign=*/true, b, c, false);
#endif
  return dp4a_us_fallback(a, b, c);
#else
  return dp4a_us_fallback(a, b, c);
#endif
}

} // namespace detail

inline int32_t dot_acc(int32_t pa, int32_t pb, int32_t c) {
  return detail::dp4a_ss(pa, pb, c);
}

inline int32_t dot_acc(uint32_t pa, uint32_t pb, int32_t c) {
  return detail::dp4a_uu(pa, pb, c);
}

inline int32_t dot_acc(int32_t pa, uint32_t pb, int32_t c) {
  return detail::dp4a_su(pa, pb, c);
}

inline int32_t dot_acc(uint32_t pa, int32_t pb, int32_t c) {
  return detail::dp4a_us(pa, pb, c);
}

// The 4-byte vectors are bit-cast directly to the packed 32-bit word. `dot_acc`
// sums four independent lane products, so byte order (endianness) does not
// affect the result as long as both operands are packed identically.
inline int32_t dot_acc(vec<int8_t, 4> a, vec<int8_t, 4> b, int32_t c) {
  return detail::dp4a_ss(sycl::bit_cast<int32_t>(a), sycl::bit_cast<int32_t>(b),
                         c);
}

inline int32_t dot_acc(vec<uint8_t, 4> a, vec<uint8_t, 4> b, int32_t c) {
  return detail::dp4a_uu(sycl::bit_cast<uint32_t>(a),
                         sycl::bit_cast<uint32_t>(b), c);
}

inline int32_t dot_acc(vec<uint8_t, 4> a, vec<int8_t, 4> b, int32_t c) {
  return detail::dp4a_us(sycl::bit_cast<uint32_t>(a), sycl::bit_cast<int32_t>(b),
                         c);
}

inline int32_t dot_acc(vec<int8_t, 4> a, vec<uint8_t, 4> b, int32_t c) {
  return detail::dp4a_su(sycl::bit_cast<int32_t>(a), sycl::bit_cast<uint32_t>(b),
                         c);
}

} // namespace ext::oneapi

} // namespace _V1
} // namespace sycl
