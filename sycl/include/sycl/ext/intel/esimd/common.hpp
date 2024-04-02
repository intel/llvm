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

#include <sycl/detail/defines.hpp>
#include <sycl/exception.hpp>
#include <sycl/ext/intel/esimd/detail/defines_elementary.hpp>
#include <sycl/ext/intel/esimd/memory_properties.hpp>
#include <sycl/ext/intel/esimd/native/common.hpp>

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
inline namespace _V1 {
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

/// Specify if end of thread should be set.
enum class raw_send_eot : uint8_t {
  not_eot = 0,
  eot = 1,
};

/// Specify if sendc should be used.
enum class raw_send_sendc : uint8_t {
  not_sendc = 0,
  sendc = 1,
};

namespace detail {

// Type used in internal functions to designate SLM access by
// providing dummy accessor of this type. Used to make it possible to delegate
// implemenations of SLM memory accesses to general surface-based memory
// accesses and thus reuse validity checks etc.
struct LocalAccessorMarker {};

template <typename T> struct is_saturation_tag {
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

/// Represents an atomic operation. Operations always return the old value(s) of
/// the target memory location(s) as it was before the operation was applied.
/// Each operation is annotated with a pseudocode illustrating its semantics,
/// \c addr is a memory address (one of the many, as the atomic operation is
/// vector) the operation is applied at, \c src0 is its first argumnet,
/// \c src1 - second.
/// Using the floating point atomic operations adds the requirement to running
/// the code with it on target devices with LSC features (ACM, PVC, etc).
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
  /// Maximum: <code>*addr = max(*addr, src0)</code>.
  umax = 0x5,
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
  /// Maximum (signed integer): <code>*addr = max(*addr, src0)</code>.
  smax = 0xc,
  /// ACM/PVC: Minimum (floating point): <code>*addr = min(*addr, src0)</code>.
  fmax = 0x10,
  /// ACM/PVC: Maximum (floating point): <code>*addr = max(*addr, src0)</code>.
  fmin = 0x11,
  /// ACM/PVC: Compare and exchange (floating point).
  /// <code>if (*addr == src0) *addr = src1;</code>
  fcmpxchg = 0x12,
  fcmpwr = fcmpxchg,
  /// ACM/PVC: Addition (floating point): <code>*addr = *addr + src0</code>.
  fadd = 0x13, //
  /// ACM/PVC: Subtraction (floating point): <code>*addr = *addr - src0</code>.
  fsub = 0x14,
  load = 0x15,
  store = 0x16,
  /// Decrement: <code>*addr = *addr - 1</code>. The only operation which
  /// returns new value of the destination rather than old.
  predec = 0xff,
};

#undef __ESIMD_USM_DWORD_TO_LSC_MSG

/// @} sycl_esimd_core

namespace detail {
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
  case __ESIMD_NS::atomic_op::umin:
    return __ESIMD_NS::native::lsc::atomic_op::umin;
  case __ESIMD_NS::atomic_op::umax:
    return __ESIMD_NS::native::lsc::atomic_op::umax;
  case __ESIMD_NS::atomic_op::cmpxchg:
    return __ESIMD_NS::native::lsc::atomic_op::cmpxchg;
  case __ESIMD_NS::atomic_op::bit_and:
    return __ESIMD_NS::native::lsc::atomic_op::bit_and;
  case __ESIMD_NS::atomic_op::bit_or:
    return __ESIMD_NS::native::lsc::atomic_op::bit_or;
  case __ESIMD_NS::atomic_op::bit_xor:
    return __ESIMD_NS::native::lsc::atomic_op::bit_xor;
  case __ESIMD_NS::atomic_op::smin:
    return __ESIMD_NS::native::lsc::atomic_op::smin;
  case __ESIMD_NS::atomic_op::smax:
    return __ESIMD_NS::native::lsc::atomic_op::smax;
  case __ESIMD_NS::atomic_op::fmax:
    return __ESIMD_NS::native::lsc::atomic_op::fmax;
  case __ESIMD_NS::atomic_op::fmin:
    return __ESIMD_NS::native::lsc::atomic_op::fmin;
  case __ESIMD_NS::atomic_op::fcmpxchg:
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
    return __ESIMD_NS::atomic_op::umin;
  case __ESIMD_NS::native::lsc::atomic_op::umax:
    return __ESIMD_NS::atomic_op::umax;
  case __ESIMD_NS::native::lsc::atomic_op::cmpxchg:
    return __ESIMD_NS::atomic_op::cmpxchg;
  case __ESIMD_NS::native::lsc::atomic_op::bit_and:
    return __ESIMD_NS::atomic_op::bit_and;
  case __ESIMD_NS::native::lsc::atomic_op::bit_or:
    return __ESIMD_NS::atomic_op::bit_or;
  case __ESIMD_NS::native::lsc::atomic_op::bit_xor:
    return __ESIMD_NS::atomic_op::bit_xor;
  case __ESIMD_NS::native::lsc::atomic_op::smin:
    return __ESIMD_NS::atomic_op::smin;
  case __ESIMD_NS::native::lsc::atomic_op::smax:
    return __ESIMD_NS::atomic_op::smax;
  case __ESIMD_NS::native::lsc::atomic_op::fmax:
    return __ESIMD_NS::atomic_op::fmax;
  case __ESIMD_NS::native::lsc::atomic_op::fmin:
    return __ESIMD_NS::atomic_op::fmin;
  case __ESIMD_NS::native::lsc::atomic_op::fcmpxchg:
    return __ESIMD_NS::atomic_op::fcmpxchg;
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
  switch (Op) {
  case __ESIMD_NS::atomic_op::inc:
  case __ESIMD_NS::atomic_op::dec:
  case __ESIMD_NS::atomic_op::load:
    return 0;
  case __ESIMD_NS::atomic_op::xchg:
  case __ESIMD_NS::atomic_op::predec:
  case __ESIMD_NS::atomic_op::store:
  case __ESIMD_NS::atomic_op::add:
  case __ESIMD_NS::atomic_op::sub:
  case __ESIMD_NS::atomic_op::smin:
  case __ESIMD_NS::atomic_op::smax:
  case __ESIMD_NS::atomic_op::umin:
  case __ESIMD_NS::atomic_op::umax:
  case __ESIMD_NS::atomic_op::fadd:
  case __ESIMD_NS::atomic_op::fsub:
  case __ESIMD_NS::atomic_op::fmin:
  case __ESIMD_NS::atomic_op::fmax:
  case __ESIMD_NS::atomic_op::bit_and:
  case __ESIMD_NS::atomic_op::bit_or:
  case __ESIMD_NS::atomic_op::bit_xor:
    return 1;
  case __ESIMD_NS::atomic_op::cmpxchg:
  case __ESIMD_NS::atomic_op::fcmpxchg:
    return 2;
  default:
    return -1; // error
  }
}

template <__ESIMD_NS::native::lsc::atomic_op Op> constexpr int get_num_args() {
  return get_num_args<to_atomic_op<Op>()>();
}

} // namespace detail

/// The scope that fence() operation should apply to.
/// Supported platforms: DG2, PVC
enum class fence_scope : uint8_t {
  /// Wait until all previous memory transactions from this thread are observed
  /// within the local thread-group.
  group = 0,

  /// Wait until all previous memory transactions from this thread are observed
  /// within the local sub-slice.
  local = 1,

  /// Wait until all previous memory transactions from this thread are observed
  /// in the local tile.
  tile = 2,

  /// Wait until all previous memory transactions from this thread are observed
  /// in the local GPU.
  gpu = 3,

  /// Wait until all previous memory transactions from this thread are observed
  /// across all GPUs in the system.
  gpus = 4,

  /// Global memory data-port only: wait until all previous memory transactions
  /// from this thread are observed at the "system" level.
  system = 5,

  /// Global memory data-port only: for GPUs that do not follow
  /// PCIe Write ordering for downstream writes targeting device memory,
  /// this op will commit to device memory all downstream and peer writes that
  /// have reached the device.
  system_acquire = 6
};

/// The cache flush operation to apply to caches after fence() is complete.
/// Supported platforms: DG2, PVC
enum class fence_flush_op : uint8_t {
  none = 0,       /// no operation;
  evict = 1,      /// R/W: evict dirty lines; R/W and RO: invalidate clean lines
  invalidate = 2, /// R/W and RO: invalidate all clean lines;

  // enum with the value 3 is reserved;

  clean = 4 /// R/W: dirty lines are written to memory, but retained in
            /// cache in clean state; RO: no effect.
};

/// The target memory kind for fence() operation.
/// Supported platforms: DG2, PVC
enum class memory_kind : uint8_t {
  global = 0, /// untyped global memory
  // enum with the value 1 is reserved;
  image = 2, /// image (also known as typed global memory)
  local = 3, /// shared local memory
};

namespace detail {

/// Data size or format to read or store
enum class lsc_data_size : uint8_t {
  default_size = 0,
  u8 = 1,
  u16 = 2,
  u32 = 3,
  u64 = 4,
  u8u32 = 5,   /// load 8b, zero extend to 32b; store the opposite
  u16u32 = 6,  /// load 16b, zero extend to 32b; store the opposite
  u16u32h = 7, /// load 16b into high 16 of each 32b; store the high 16
};

template <typename T, lsc_data_size DS> constexpr void check_lsc_data_size() {
  static_assert(DS != lsc_data_size::default_size || sizeof(T) == 1 ||
                    sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "Unsupported data type");
  static_assert(
      DS == lsc_data_size::default_size ||
          (sizeof(T) == 1 &&
           (DS == lsc_data_size::u8 || DS == lsc_data_size::u8u32)) ||
          (sizeof(T) == 2 &&
           (DS == lsc_data_size::u16 || DS == lsc_data_size::u16u32 ||
            DS == lsc_data_size::u16u32h)) ||
          (sizeof(T) == 4 &&
           (DS == lsc_data_size::u32 || DS == lsc_data_size::u8u32 ||
            DS == lsc_data_size::u16u32 || DS == lsc_data_size::u16u32h)) ||
          (sizeof(T) == 8 && DS == lsc_data_size::u64),
      "Data type does not match data size");
}

template <typename T, lsc_data_size DS>
constexpr lsc_data_size finalize_data_size() {
  check_lsc_data_size<T, DS>();
  if (DS != lsc_data_size::default_size)
    return DS;
  else if (sizeof(T) == 1)
    return lsc_data_size::u8;
  else if (sizeof(T) == 2)
    return lsc_data_size::u16;
  else if (sizeof(T) == 4)
    return lsc_data_size::u32;
  else if (sizeof(T) == 8)
    return lsc_data_size::u64;
  else
    return DS;
}

enum class lsc_vector_size : uint8_t {
  n1 = 1,
  n2 = 2,
  n3 = 3,
  n4 = 4,
  n8 = 5,
  n16 = 6,
  n32 = 7,
  n64 = 8,
};

template <int VS> constexpr void check_lsc_vector_size() {
  static_assert(VS == 1 || VS == 2 || VS == 3 || VS == 4 || VS == 8 ||
                    VS == 16 || VS == 32 || VS == 64,
                "Unsupported vector size");
}

template <lsc_vector_size VS> constexpr void check_lsc_vector_size() {
  static_assert(VS == lsc_vector_size::n1 || VS == lsc_vector_size::n2 ||
                    VS == lsc_vector_size::n3 || VS == lsc_vector_size::n4 ||
                    VS == lsc_vector_size::n8 || VS == lsc_vector_size::n16 ||
                    VS == lsc_vector_size::n64 || VS == lsc_vector_size::n32,
                "Unsupported vector size");
}

template <lsc_vector_size VS> constexpr uint8_t to_int() {
  check_lsc_vector_size<VS>();
  switch (VS) {
  case lsc_vector_size::n1:
    return 1;
  case lsc_vector_size::n2:
    return 2;
  case lsc_vector_size::n3:
    return 3;
  case lsc_vector_size::n4:
    return 4;
  case lsc_vector_size::n8:
    return 8;
  case lsc_vector_size::n16:
    return 16;
  case lsc_vector_size::n32:
    return 32;
  case lsc_vector_size::n64:
    return 64;
  default:
    return 1;
  }
}

template <int VS> constexpr lsc_vector_size to_lsc_vector_size() {
  check_lsc_vector_size<VS>();
  switch (VS) {
  case 1:
    return lsc_vector_size::n1;
  case 2:
    return lsc_vector_size::n2;
  case 3:
    return lsc_vector_size::n3;
  case 4:
    return lsc_vector_size::n4;
  case 8:
    return lsc_vector_size::n8;
  case 16:
    return lsc_vector_size::n16;
  case 32:
    return lsc_vector_size::n32;
  case 64:
    return lsc_vector_size::n64;
  default:
    return lsc_vector_size::n1;
  }
}

enum class lsc_data_order : uint8_t {
  nontranspose = 1,
  transpose = 2,
};

template <cache_hint Hint> class cache_hint_wrap {
  template <cache_hint...> struct is_one_of_t;
  template <cache_hint Last>
  struct is_one_of_t<Last>
      : std::conditional_t<Last == Hint, std::true_type, std::false_type> {};
  template <cache_hint Head, cache_hint... Tail>
  struct is_one_of_t<Head, Tail...>
      : std::conditional_t<Head == Hint, std::true_type, is_one_of_t<Tail...>> {
  };

public:
  constexpr operator cache_hint() const { return Hint; }
  template <cache_hint... Hints> constexpr bool is_one_of() const {
    return is_one_of_t<Hints...>::value;
  }
};

constexpr bool are_both(cache_hint First, cache_hint Second, cache_hint Val) {
  return First == Val && Second == Val;
}

enum class cache_action { prefetch, load, store, atomic };

template <typename PropertyListT> constexpr bool has_cache_hints() {
  constexpr cache_hint L1H =
      getPropertyValue<PropertyListT, cache_hint_L1_key>(cache_hint::none);
  constexpr cache_hint L2H =
      getPropertyValue<PropertyListT, cache_hint_L2_key>(cache_hint::none);
  return L1H != cache_hint::none || L2H != cache_hint::none;
}

// Currently, this is just a wrapper around 'check_cache_hint' function.
// It accepts the compile-time properties that may include cache-hints
// to be verified.
template <cache_action Action, typename PropertyListT>
void check_cache_hints() {
  constexpr auto L1H =
      cache_hint_wrap<getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none)>{};
  constexpr auto L2H =
      cache_hint_wrap<getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none)>{};
  if constexpr (Action == cache_action::prefetch) {
    static_assert(
        L1H.template is_one_of<cache_hint::cached, cache_hint::uncached,
                               cache_hint::streaming>() &&
            L2H.template is_one_of<cache_hint::cached,
                                   cache_hint::uncached>() &&
            !are_both(L1H, L2H, cache_hint::uncached),
        "unsupported cache hint");
  } else if constexpr (Action == cache_action::load) {
    static_assert(
        are_both(L1H, L2H, cache_hint::none) ||
            (L1H.template is_one_of<cache_hint::uncached, cache_hint::cached,
                                    cache_hint::streaming>() &&
             L2H.template is_one_of<cache_hint::uncached,
                                    cache_hint::cached>()) ||
            (L1H == cache_hint::read_invalidate && L2H == cache_hint::cached),
        "unsupported cache hint");
  } else if constexpr (Action == cache_action::store) {
    static_assert(are_both(L1H, L2H, cache_hint::none) ||
                      are_both(L1H, L2H, cache_hint::write_back) ||
                      (L1H.template is_one_of<cache_hint::uncached,
                                              cache_hint::write_through,
                                              cache_hint::streaming>() &&
                       L2H.template is_one_of<cache_hint::uncached,
                                              cache_hint::write_back>()),
                  "unsupported cache hint");
  } else if constexpr (Action == cache_action::atomic) {
    static_assert(are_both(L1H, L2H, cache_hint::none) ||
                      (L1H == cache_hint::uncached &&
                       L2H.template is_one_of<cache_hint::uncached,
                                              cache_hint::write_back>()),
                  "unsupported cache hint");
  }
}

constexpr lsc_data_size expand_data_size(lsc_data_size DS) {
  if (DS == lsc_data_size::u8)
    return lsc_data_size::u8u32;
  if (DS == lsc_data_size::u16)
    return lsc_data_size::u16u32;
  return DS;
}

template <typename T> struct lsc_expand_type {
  using type = std::conditional_t<
      sizeof(T) <= 4,
      std::conditional_t<std::is_signed_v<T>, int32_t, uint32_t>,
      std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>>;
};

} // namespace detail

} // namespace ext::intel::esimd
} // namespace _V1
} // namespace sycl
