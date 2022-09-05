//==---------------- common.hpp - DPC++ Explicit SIMD API   ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// definitions used in Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/detail/defines_elementary.hpp>
#include <sycl/ext/intel/esimd/native/common.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>

#include <sycl/detail/defines.hpp>

#include <cstdint> // for uint* types
#include <type_traits>

/// @cond ESIMD_DETAIL

#ifdef __SYCL_DEVICE_ONLY__
#define __ESIMD_UNSUPPORTED_ON_HOST
#else // __SYCL_DEVICE_ONLY__
#define __ESIMD_UNSUPPORTED_ON_HOST                                            \
  throw sycl::exception(sycl::errc::feature_not_supported,                     \
                        "This ESIMD feature is not supported on HOST")
#endif // __SYCL_DEVICE_ONLY__

/// @endcond ESIMD_DETAIL

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::esimd {

/// @addtogroup sycl_esimd_core
/// @{

using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;

/// Gen hardware supports applying saturation to results of certain operations.
/// This type tag represents "saturation on" behavior.
struct saturation_on_tag : std::true_type {};

/// This type tag represents "saturation off" behavior.
struct saturation_off_tag : std::false_type {};

/// Type tag object representing "saturation off" behavior.
static inline constexpr saturation_off_tag saturation_off{};

/// Type tag object representing "saturation on" behavior.
static inline constexpr saturation_on_tag saturation_on{};

/// Represents a pixel's channel.
enum class rgba_channel : uint8_t { R, G, B, A };

/// Surface index type. Surface is an internal representation of a memory block
/// addressable by GPU in "stateful" memory model, and each surface is
/// identified by its "binding table index" - surface index.
using SurfaceIndex = unsigned int;

namespace detail {

template <typename T>
struct is_saturation_tag {
  static constexpr bool value =
      std::is_same_v<T, __ESIMD_NS::saturation_on_tag> ||
      std::is_same_v<T, __ESIMD_NS::saturation_off_tag>;
};

template <class T>
inline constexpr bool is_saturation_tag_v = is_saturation_tag<T>::value;

/// Check if a given 32 bit positive integer is a power of 2 at compile time.
ESIMD_INLINE constexpr bool isPowerOf2(unsigned int n) {
  return (n & (n - 1)) == 0;
}

/// Check at compile time if given 32 bit positive integer is both:
/// - a power of 2
/// - less or equal to given limit
ESIMD_INLINE constexpr bool isPowerOf2(unsigned int n, unsigned int limit) {
  return (n & (n - 1)) == 0 && n <= limit;
}

template <rgba_channel Ch>
static inline constexpr uint8_t ch = 1 << static_cast<int>(Ch);
static inline constexpr uint8_t chR = ch<rgba_channel::R>;
static inline constexpr uint8_t chG = ch<rgba_channel::G>;
static inline constexpr uint8_t chB = ch<rgba_channel::B>;
static inline constexpr uint8_t chA = ch<rgba_channel::A>;

// Shared Local Memory Binding Table Index (aka surface index).
static inline constexpr SurfaceIndex SLM_BTI = 254;
static inline constexpr SurfaceIndex INVALID_BTI =
    static_cast<SurfaceIndex>(-1);
} // namespace detail

/// Represents a pixel's channel mask - all possible combinations of enabled
/// channels.
enum class rgba_channel_mask : uint8_t {
  R = detail::chR,
  G = detail::chG,
  GR = detail::chG | detail::chR,
  B = detail::chB,
  BR = detail::chB | detail::chR,
  BG = detail::chB | detail::chG,
  BGR = detail::chB | detail::chG | detail::chR,
  A = detail::chA,
  AR = detail::chA | detail::chR,
  AG = detail::chA | detail::chG,
  AGR = detail::chA | detail::chG | detail::chR,
  AB = detail::chA | detail::chB,
  ABR = detail::chA | detail::chB | detail::chR,
  ABG = detail::chA | detail::chB | detail::chG,
  ABGR = detail::chA | detail::chB | detail::chG | detail::chR,
};

constexpr int is_channel_enabled(rgba_channel_mask M, rgba_channel Ch) {
  int Pos = static_cast<int>(Ch);
  return (static_cast<int>(M) & (1 << Pos)) >> Pos;
}

constexpr int get_num_channels_enabled(rgba_channel_mask M) {
  return is_channel_enabled(M, rgba_channel::R) +
         is_channel_enabled(M, rgba_channel::G) +
         is_channel_enabled(M, rgba_channel::B) +
         is_channel_enabled(M, rgba_channel::A);
}

#define __ESIMD_USM_DWORD_ATOMIC_TO_LSC                                        \
  " is supported only on ACM, PVC. USM-based atomic will be auto-converted "   \
  "to LSC version."

/// Represents an atomic operation. Operations always return the old value(s) of
/// the target memory location(s) as it was before the operation was applied.
/// Each operation is annotated with a pseudocode illustrating its semantics,
/// \c addr is a memory address (one of the many, as the atomic operation is
/// vector) the operation is applied at, \c src0 is its first argumnet,
/// \c src1 - second.
enum class atomic_op : uint8_t {
  /// Addition: <code>*addr = *addr + src0</code>.
  add = 0x0,
  /// Subtraction: <code>*addr = *addr - src0</code>.
  sub = 0x1,
  /// Increment: <code>*addr = *addr + 1</code>.
  inc = 0x2,
  /// Decrement: <code>*addr = *addr - 1</code>.
  dec = 0x3,
  /// Minimum: <code>*addr = min(*addr, src0)</code>.
  umin = 0x4,
  min __SYCL_DEPRECATED("use umin") = umin,
  /// Maximum: <code>*addr = max(*addr, src0)</code>.
  umax = 0x5,
  max __SYCL_DEPRECATED("use smax") = umax,
  /// Exchange. <code>*addr == src0;</code>
  xchg = 0x6,
  /// Compare and exchange. <code>if (*addr == src0) *sddr = src1;</code>
  cmpxchg = 0x7,
  /// Bit \c and: <code>*addr = *addr & src0</code>.
  bit_and = 0x8,
  /// Bit \c or: <code>*addr = *addr | src0</code>.
  bit_or = 0x9,
  /// Bit \c xor: <code>*addr = *addr | src0</code>.
  bit_xor = 0xa,
  /// Minimum (signed integer): <code>*addr = min(*addr, src0)</code>.
  smin = 0xb,
  minsint __SYCL_DEPRECATED("use smin") = smin,
  /// Maximum (signed integer): <code>*addr = max(*addr, src0)</code>.
  smax = 0xc,
  maxsint __SYCL_DEPRECATED("use smax") = 0xc,
  /// Minimum (floating point): <code>*addr = min(*addr, src0)</code>.
  fmax __SYCL_DEPRECATED("fmax" __ESIMD_USM_DWORD_ATOMIC_TO_LSC) = 0x10,
  /// Maximum (floating point): <code>*addr = max(*addr, src0)</code>.
  fmin __SYCL_DEPRECATED("fmin" __ESIMD_USM_DWORD_ATOMIC_TO_LSC) = 0x11,
  /// Compare and exchange (floating point).
  /// <code>if (*addr == src0) *addr = src1;</code>
  fcmpxchg = 0x12,
  fcmpwr __SYCL_DEPRECATED("fcmpwr" __ESIMD_USM_DWORD_ATOMIC_TO_LSC) = fcmpxchg,
  fadd __SYCL_DEPRECATED("fadd" __ESIMD_USM_DWORD_ATOMIC_TO_LSC) = 0x13,
  fsub __SYCL_DEPRECATED("fsub" __ESIMD_USM_DWORD_ATOMIC_TO_LSC) = 0x14,
  load = 0x15,
  store = 0x16,
  /// Decrement: <code>*addr = *addr - 1</code>. The only operation which
  /// returns new value of the destination rather than old.
  predec = 0xff,
};

#undef __ESIMD_USM_DWORD_TO_LSC_MSG

/// @} sycl_esimd_core

namespace detail {
template <__ESIMD_NS::native::lsc::atomic_op Op> constexpr int get_num_args() {
  if constexpr (Op == __ESIMD_NS::native::lsc::atomic_op::inc ||
                Op == __ESIMD_NS::native::lsc::atomic_op::dec ||
                Op == __ESIMD_NS::native::lsc::atomic_op::load) {
    return 0;
  } else if constexpr (Op == __ESIMD_NS::native::lsc::atomic_op::store ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::add ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::sub ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::smin ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::smax ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::umin ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::umax ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::fadd ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::fsub ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::fmin ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::fmax ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::bit_and ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::bit_or ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::bit_xor) {
    return 1;
  } else if constexpr (Op == __ESIMD_NS::native::lsc::atomic_op::cmpxchg ||
                       Op == __ESIMD_NS::native::lsc::atomic_op::fcmpxchg) {
    return 2;
  } else {
    return -1; // error
  }
}

template <__ESIMD_NS::atomic_op Op> constexpr bool has_lsc_equivalent() {
  switch (Op) {
  case __ESIMD_NS::atomic_op::xchg:
  case __ESIMD_NS::atomic_op::predec:
    return false;
  default:
    return true;
  }
}

template <__ESIMD_NS::atomic_op Op>
constexpr __ESIMD_NS::native::lsc::atomic_op to_lsc_atomic_op() {
  switch (Op) {
  case __ESIMD_NS::atomic_op::add:
    return __ESIMD_NS::native::lsc::atomic_op::add;
  case __ESIMD_NS::atomic_op::sub:
    return __ESIMD_NS::native::lsc::atomic_op::sub;
  case __ESIMD_NS::atomic_op::inc:
    return __ESIMD_NS::native::lsc::atomic_op::inc;
  case __ESIMD_NS::atomic_op::dec:
    return __ESIMD_NS::native::lsc::atomic_op::dec;
  case __ESIMD_NS::atomic_op::min:
    return __ESIMD_NS::native::lsc::atomic_op::umin;
  case __ESIMD_NS::atomic_op::max:
    return __ESIMD_NS::native::lsc::atomic_op::umax;
  case __ESIMD_NS::atomic_op::cmpxchg:
    return __ESIMD_NS::native::lsc::atomic_op::cmpxchg;
  case __ESIMD_NS::atomic_op::bit_and:
    return __ESIMD_NS::native::lsc::atomic_op::bit_and;
  case __ESIMD_NS::atomic_op::bit_or:
    return __ESIMD_NS::native::lsc::atomic_op::bit_or;
  case __ESIMD_NS::atomic_op::bit_xor:
    return __ESIMD_NS::native::lsc::atomic_op::bit_xor;
  case __ESIMD_NS::atomic_op::minsint:
    return __ESIMD_NS::native::lsc::atomic_op::smin;
  case __ESIMD_NS::atomic_op::maxsint:
    return __ESIMD_NS::native::lsc::atomic_op::smax;
  case __ESIMD_NS::atomic_op::fmax:
    return __ESIMD_NS::native::lsc::atomic_op::fmax;
  case __ESIMD_NS::atomic_op::fmin:
    return __ESIMD_NS::native::lsc::atomic_op::fmin;
  case __ESIMD_NS::atomic_op::fcmpwr:
    return __ESIMD_NS::native::lsc::atomic_op::fcmpxchg;
  case __ESIMD_NS::atomic_op::fadd:
    return __ESIMD_NS::native::lsc::atomic_op::fadd;
  case __ESIMD_NS::atomic_op::fsub:
    return __ESIMD_NS::native::lsc::atomic_op::fsub;
  case __ESIMD_NS::atomic_op::load:
    return __ESIMD_NS::native::lsc::atomic_op::load;
  case __ESIMD_NS::atomic_op::store:
    return __ESIMD_NS::native::lsc::atomic_op::store;
  default:
    static_assert(has_lsc_equivalent<Op>() && "Unsupported LSC atomic op");
  }
}

template <__ESIMD_NS::native::lsc::atomic_op Op>
constexpr __ESIMD_NS::atomic_op to_atomic_op() {
  switch (Op) {
  case __ESIMD_NS::native::lsc::atomic_op::add:
    return __ESIMD_NS::atomic_op::add;
  case __ESIMD_NS::native::lsc::atomic_op::sub:
    return __ESIMD_NS::atomic_op::sub;
  case __ESIMD_NS::native::lsc::atomic_op::inc:
    return __ESIMD_NS::atomic_op::inc;
  case __ESIMD_NS::native::lsc::atomic_op::dec:
    return __ESIMD_NS::atomic_op::dec;
  case __ESIMD_NS::native::lsc::atomic_op::umin:
    return __ESIMD_NS::atomic_op::min;
  case __ESIMD_NS::native::lsc::atomic_op::umax:
    return __ESIMD_NS::atomic_op::max;
  case __ESIMD_NS::native::lsc::atomic_op::cmpxchg:
    return __ESIMD_NS::atomic_op::cmpxchg;
  case __ESIMD_NS::native::lsc::atomic_op::bit_and:
    return __ESIMD_NS::atomic_op::bit_and;
  case __ESIMD_NS::native::lsc::atomic_op::bit_or:
    return __ESIMD_NS::atomic_op::bit_or;
  case __ESIMD_NS::native::lsc::atomic_op::bit_xor:
    return __ESIMD_NS::atomic_op::bit_xor;
  case __ESIMD_NS::native::lsc::atomic_op::smin:
    return __ESIMD_NS::atomic_op::minsint;
  case __ESIMD_NS::native::lsc::atomic_op::smax:
    return __ESIMD_NS::atomic_op::maxsint;
  case __ESIMD_NS::native::lsc::atomic_op::fmax:
    return __ESIMD_NS::atomic_op::fmax;
  case __ESIMD_NS::native::lsc::atomic_op::fmin:
    return __ESIMD_NS::atomic_op::fmin;
  case __ESIMD_NS::native::lsc::atomic_op::fcmpxchg:
    return __ESIMD_NS::atomic_op::fcmpwr;
  case __ESIMD_NS::native::lsc::atomic_op::fadd:
    return __ESIMD_NS::atomic_op::fadd;
  case __ESIMD_NS::native::lsc::atomic_op::fsub:
    return __ESIMD_NS::atomic_op::fsub;
  case __ESIMD_NS::native::lsc::atomic_op::load:
    return __ESIMD_NS::atomic_op::load;
  case __ESIMD_NS::native::lsc::atomic_op::store:
    return __ESIMD_NS::atomic_op::store;
  }
}

template <__ESIMD_NS::atomic_op Op> constexpr int get_num_args() {
  if constexpr (has_lsc_equivalent<Op>()) {
    return get_num_args<to_lsc_atomic_op<Op>()>();
  } else {
    switch (Op) {
    case __ESIMD_NS::atomic_op::xchg:
    case __ESIMD_NS::atomic_op::predec:
      return 1;
    default:
      return -1; // error
    }
  }
}

} // namespace detail

} // namespace ext::intel::esimd
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
