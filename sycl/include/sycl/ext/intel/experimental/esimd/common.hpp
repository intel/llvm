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

// Mark a function being nodebug.
#define ESIMD_NODEBUG __attribute__((nodebug))
// Mark a "ESIMD global": accessible from all functions in current translation
// unit, separate copy per subgroup (work-item), mapped to SPIR-V private
// storage class.
#define ESIMD_PRIVATE                                                          \
  __attribute__((opencl_private)) __attribute__((sycl_explicit_simd))
// Bind a ESIMD global variable to a specific register.
#define ESIMD_REGISTER(n) __attribute__((register_num(n)))

#define __ESIMD_API ESIMD_NODEBUG ESIMD_INLINE

#define __ESIMD_UNSUPPORTED_ON_HOST

#else // __SYCL_DEVICE_ONLY__
#define SYCL_ESIMD_KERNEL
#define SYCL_ESIMD_FUNCTION

// TODO ESIMD define what this means on Windows host
#define ESIMD_NODEBUG
// On host device ESIMD global is a thread local static var. This assumes that
// each work-item is mapped to a separate OS thread on host device.
#define ESIMD_PRIVATE thread_local
#define ESIMD_REGISTER(n)

#define __ESIMD_API ESIMD_INLINE

#define __ESIMD_UNSUPPORTED_ON_HOST throw cl::sycl::feature_not_supported()

#endif // __SYCL_DEVICE_ONLY__

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

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

/// @{
/// @ingroup sycl_esimd_core

using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;

/// Gen hardware supports applying saturation to results of some operation.
/// This enum allows to control this behavior.
enum class saturation : uint8_t { off, on };

/// Integer type short-cut to saturation::off.
static inline constexpr uint8_t saturation_off =
    static_cast<uint8_t>(saturation::off);
/// Integer type short-cut to saturation::on.
static inline constexpr uint8_t saturation_on =
    static_cast<uint8_t>(saturation::on);

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
};

/// Represents a split barrier action.
enum class split_barrier_action : uint8_t {
  wait = 0,   // split barrier wait
  signal = 1, // split barrier signal
};

/// Surface index type. Surface is an internal representation of a memory block
/// addressable by GPU in "stateful" memory model, and each surface is
/// identified by its "binding table index" - surface index.
using SurfaceIndex = unsigned int;

/// @} sycl_esimd_core

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
