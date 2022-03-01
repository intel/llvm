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
#include <type_traits>

/// @cond ESIMD_DETAIL

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

/// @endcond ESIMD_DETAIL

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

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

enum class argument_type {
  U1 = 0,   // unsigned 1 bit
  S1 = 1,   // signed 1 bit
  U2 = 2,   // unsigned 2 bits
  S2 = 3,   // signed 2 bits
  U4 = 4,   // unsigned 4 bits
  S4 = 5,   // signed 4 bits
  U8 = 6,   // unsigned 8 bits
  S8 = 7,   // signed 8 bits
  BF16 = 8, // bfloat 16
  FP16 = 9, // half float
  TF32 = 11 // tensorfloat 32
};

/// Represents a pixel's channel.
enum class rgba_channel : uint8_t { R, G, B, A };

/// Surface index type. Surface is an internal representation of a memory block
/// addressable by GPU in "stateful" memory model, and each surface is
/// identified by its "binding table index" - surface index.
using SurfaceIndex = unsigned int;

namespace detail {
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
  min = 0x4,
  /// Maximum: <code>*addr = max(*addr, src0)</code>.
  max = 0x5,
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
  minsint = 0xb,
  /// Maximum (signed integer): <code>*addr = max(*addr, src0)</code>.
  maxsint = 0xc,
  /// Minimum (floating point): <code>*addr = min(*addr, src0)</code>.
  fmax = 0x10,
  /// Maximum (floating point): <code>*addr = max(*addr, src0)</code>.
  fmin = 0x11,
  /// Compare and exchange (floating point).
  /// <code>if (*addr == src0) *addr = src1;</code>
  fcmpwr = 0x12,
  /// Decrement: <code>*addr = *addr - 1</code>. The only operation which
  /// returns new value of the destination rather than old.
  predec = 0xff,
};

/// Represents a split barrier action.
enum class split_barrier_action : uint8_t {
  wait = 0,   // split barrier wait
  signal = 1, // split barrier signal
};

/// @} sycl_esimd_core

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
