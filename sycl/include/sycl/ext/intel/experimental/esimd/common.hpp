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

#include <CL/sycl/detail/defines.hpp>

#include <cstdint> // for uint* types

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_ESIMD_KERNEL __attribute__((sycl_explicit_simd))
#define SYCL_ESIMD_FUNCTION __attribute__((sycl_explicit_simd))
#else
#define SYCL_ESIMD_KERNEL
#define SYCL_ESIMD_FUNCTION
#endif

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;

#ifdef __SYCL_DEVICE_ONLY__
// Mark a function being nodebug.
#define ESIMD_NODEBUG __attribute__((nodebug))
// Mark a "ESIMD global": accessible from all functions in current translation
// unit, separate copy per subgroup (work-item), mapped to SPIR-V private
// storage class.
#define ESIMD_PRIVATE                                                          \
  __attribute__((opencl_private)) __attribute__((sycl_explicit_simd))
// Bind a ESIMD global variable to a specific register.
#define ESIMD_REGISTER(n) __attribute__((register_num(n)))
#else
// TODO ESIMD define what this means on Windows host
#define ESIMD_NODEBUG
// On host device ESIMD global is a thread local static var. This assumes that
// each work-item is mapped to a separate OS thread on host device.
#define ESIMD_PRIVATE thread_local
#define ESIMD_REGISTER(n)
#endif

// Mark a function being noinline
#define ESIMD_NOINLINE __attribute__((noinline))
// Force a function to be inlined. 'inline' is used to preserve ODR for
// functions defined in a header.
#define ESIMD_INLINE inline __attribute__((always_inline))

// Macros for internal use
#define __ESIMD_NS sycl::ext::intel::experimental::esimd
#define __ESIMD_QUOTE1(m) #m
#define __ESIMD_QUOTE(m) __ESIMD_QUOTE1(m)
#define __ESIMD_NS_QUOTED __ESIMD_QUOTE(__ESIMD_NS)
#define __ESIMD_DEPRECATED(new_api)                                            \
  __SYCL_DEPRECATED("use " __ESIMD_NS_QUOTED "::" __ESIMD_QUOTE(new_api))
// Defines a deprecated enum value. Use of this value will cause a deprecation
// message printed out by the compiler.
#define __ESIMD_DEPR_ENUM_V(old, new, t)                                       \
  old __ESIMD_DEPRECATED(new) = static_cast<t>(new)

/// Gen hardware supports applying saturation to results of some operation.
/// This enum allows to control this behavior.
enum class saturation : uint8_t { off, on };

/// Integer type short-cut to saturation::off.
static inline constexpr uint8_t saturation_off =
    static_cast<uint8_t>(saturation::off);
/// Integer type short-cut to saturation::on.
static inline constexpr uint8_t saturation_on =
    static_cast<uint8_t>(saturation::on);

enum {
  __ESIMD_DEPR_ENUM_V(GENX_NOSAT, saturation::off, uint8_t),
  __ESIMD_DEPR_ENUM_V(GENX_SAT, saturation::on, uint8_t)
};

/// Represents a pixel's channel.
enum class rgba_channel : uint8_t { R, G, B, A };

namespace detail {
template <rgba_channel Ch>
static inline constexpr uint8_t ch = 1 << static_cast<int>(Ch);
static inline constexpr uint8_t chR = ch<rgba_channel::R>;
static inline constexpr uint8_t chG = ch<rgba_channel::G>;
static inline constexpr uint8_t chB = ch<rgba_channel::B>;
static inline constexpr uint8_t chA = ch<rgba_channel::A>;
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
  // For backward compatibility ('ChannelMaskType::ESIMD_R_ENABLE' usage style):
  __ESIMD_DEPR_ENUM_V(ESIMD_R_ENABLE, rgba_channel_mask::R, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_G_ENABLE, rgba_channel_mask::G, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_GR_ENABLE, rgba_channel_mask::GR, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_B_ENABLE, rgba_channel_mask::B, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_BR_ENABLE, rgba_channel_mask::BR, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_BG_ENABLE, rgba_channel_mask::BG, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_BGR_ENABLE, rgba_channel_mask::BGR, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_A_ENABLE, rgba_channel_mask::A, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_AR_ENABLE, rgba_channel_mask::AR, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_AG_ENABLE, rgba_channel_mask::AG, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_AGR_ENABLE, rgba_channel_mask::AGR, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_AB_ENABLE, rgba_channel_mask::AB, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_ABR_ENABLE, rgba_channel_mask::ABR, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_ABG_ENABLE, rgba_channel_mask::ABG, uint8_t),
  __ESIMD_DEPR_ENUM_V(ESIMD_ABGR_ENABLE, rgba_channel_mask::ABGR, uint8_t)
};

#define __ESIMD_DEPR_CONST(old, new)                                           \
  static inline constexpr auto old __ESIMD_DEPRECATED(new) = new

// For backward compatibility ('ESIMD_R_ENABLE' usage style):
__ESIMD_DEPR_CONST(ESIMD_R_ENABLE, rgba_channel_mask::R);
__ESIMD_DEPR_CONST(ESIMD_G_ENABLE, rgba_channel_mask::G);
__ESIMD_DEPR_CONST(ESIMD_GR_ENABLE, rgba_channel_mask::GR);
__ESIMD_DEPR_CONST(ESIMD_B_ENABLE, rgba_channel_mask::B);
__ESIMD_DEPR_CONST(ESIMD_BR_ENABLE, rgba_channel_mask::BR);
__ESIMD_DEPR_CONST(ESIMD_BG_ENABLE, rgba_channel_mask::BG);
__ESIMD_DEPR_CONST(ESIMD_BGR_ENABLE, rgba_channel_mask::BGR);
__ESIMD_DEPR_CONST(ESIMD_A_ENABLE, rgba_channel_mask::A);
__ESIMD_DEPR_CONST(ESIMD_AR_ENABLE, rgba_channel_mask::AR);
__ESIMD_DEPR_CONST(ESIMD_AG_ENABLE, rgba_channel_mask::AG);
__ESIMD_DEPR_CONST(ESIMD_AGR_ENABLE, rgba_channel_mask::AGR);
__ESIMD_DEPR_CONST(ESIMD_AB_ENABLE, rgba_channel_mask::AB);
__ESIMD_DEPR_CONST(ESIMD_ABR_ENABLE, rgba_channel_mask::ABR);
__ESIMD_DEPR_CONST(ESIMD_ABG_ENABLE, rgba_channel_mask::ABG);
__ESIMD_DEPR_CONST(ESIMD_ABGR_ENABLE, rgba_channel_mask::ABGR);

#undef __ESIMD_DEPR_CONST

// For backward compatibility:
using ChannelMaskType = rgba_channel_mask;

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

/// Represents an atomic operation.
enum class atomic_op : uint8_t {
  add = 0x0,
  sub = 0x1,
  inc = 0x2,
  dec = 0x3,
  min = 0x4,
  max = 0x5,
  xchg = 0x6,
  cmpxchg = 0x7,
  bit_and = 0x8,
  bit_or = 0x9,
  bit_xor = 0xa,
  minsint = 0xb,
  maxsint = 0xc,
  fmax = 0x10,
  fmin = 0x11,
  fcmpwr = 0x12,
  predec = 0xff,
  // For backward compatibility:
  __ESIMD_DEPR_ENUM_V(ATOMIC_ADD, atomic_op::add, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_SUB, atomic_op::sub, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_INC, atomic_op::inc, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_DEC, atomic_op::dec, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_MIN, atomic_op::min, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_MAX, atomic_op::max, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_XCHG, atomic_op::xchg, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_CMPXCHG, atomic_op::cmpxchg, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_AND, atomic_op::bit_and, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_OR, atomic_op::bit_or, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_XOR, atomic_op::bit_xor, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_MINSINT, atomic_op::minsint, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_MAXSINT, atomic_op::maxsint, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_FMAX, atomic_op::fmax, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_FMIN, atomic_op::fmin, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_FCMPWR, atomic_op::fcmpwr, uint8_t),
  __ESIMD_DEPR_ENUM_V(ATOMIC_PREDEC, atomic_op::predec, uint8_t)
};

// For backward compatibility:
using EsimdAtomicOpType = atomic_op;

// TODO Cache hints APIs are being reworked.
// L1 or L3 cache hint kinds.
enum class CacheHint : uint8_t {
  None = 0,
  Uncached = 1,
  Cached = 2,
  WriteBack = 3,
  WriteThrough = 4,
  Streaming = 5,
  ReadInvalidate = 6
};

/// Represents a split barrier action.
enum class split_barrier_action : uint8_t {
  wait = 0,   // split barrier wait
  signal = 1, // split barrier signal
  // For backward compatibility:
  __ESIMD_DEPR_ENUM_V(WAIT, split_barrier_action::wait, uint8_t),
  __ESIMD_DEPR_ENUM_V(SIGNAL, split_barrier_action::signal, uint8_t)
};

// For backward compatibility:
using EsimdSbarrierType = split_barrier_action;

#undef __ESIMD_DEPR_ENUM_V

// Since EsimdSbarrierType values are deprecated, these macros will generate
// deprecation message.
#define ESIMD_SBARRIER_WAIT EsimdSbarrierType::WAIT
#define ESIMD_SBARRIER_SIGNAL EsimdSbarrierType::SIGNAL

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
