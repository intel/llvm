//==---------------- common.hpp - DPC++ Explicit SIMD API   ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// definitions used in experimental Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/common.hpp>

/// @cond ESIMD_DETAIL

// Macros for internal use
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#define __ESIMD_EDNS sycl::ext::intel::experimental::esimd::detail

/// @endcond ESIMD_DETAIL

__SYCL_INLINE_NAMESPACE(cl) {
namespace __ESIMD_ENS {

/// @addtogroup sycl_esimd_core
/// @{

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

/// The scope that lsc_fence operation should apply to
/// Supported platforms: DG2, PVC
enum class lsc_scope : uint8_t {
  group = 0,  /// flush out to the threadgroup's scope
  local = 1,  /// flush out to the local scope
  tile = 2,   /// tile, flush out to several DSSs
  gpu = 3,    /// entire GPU, flush out to the GPUs LLC
  gpus = 4,   /// all GPUs in the system, flush out to memory shared by all GPUs
  system = 5, /// the entire system memory space
  sysacq = 6, /// the entire system memory space with system-acquire semantics
};

/// The lsc_fence operation to apply to caches
/// Supported platforms: DG2, PVC
enum class lsc_fence_op : uint8_t {
  none = 0,       /// no operation
  evict = 1,      /// dirty lines evicted and invalidated from L1
  invalidate = 2, /// invalidate all clean lines
  discard = 3,    /// direct and clean lines are discarded w/o eviction
  clean = 4,      /// dirty lines are written to memory, but retained in cache
                  /// in clean state
  flushl3 = 5,    /// flush only L3
};

/// The specific LSC shared function to fence with lsc_fence
/// Supported platforms: DG2, PVC
enum class lsc_memory_kind : uint8_t {
  untyped_global = 0,         /// untyped global memory
  untyped_global_low_pri = 1, /// low-priority untyped global memory
  typed_global = 2,           /// typed global memory
  shared_local = 3,           /// shared local memory
};

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

namespace detail {
/// LSC atomic operations op codes
enum class lsc_atomic_op : uint8_t {
  iinc = 0x08,    // atomic integer increment
  idec = 0x09,    // atomic integer decrement
  load = 0x0a,    // atomic load
  store = 0x0b,   // atomic store
  iadd = 0x0c,    // atomic integer add
  isub = 0x0d,    // atomic integer subtract
  smin = 0x0e,    // atomic signed int min
  smax = 0x0f,    // atomic signed int max
  umin = 0x10,    // atomic unsigned int min
  umax = 0x11,    // atomic unsigned int max
  icas = 0x12,    // atomic int compare and swap
  fadd = 0x13,    // floating-point add
  fsub = 0x14,    // floating-point subtract
  fmin = 0x15,    // floating-point min
  fmax = 0x16,    // floating-point max
  fcas = 0x17,    // floating-point CAS
  bit_and = 0x18, // logical (bitwise) AND
  bit_or = 0x19,  // logical (bitwise) OR
  bit_xor = 0x1a, // logical (bitwise) XOR
};

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

enum class lsc_data_order : uint8_t {
  nontranspose = 1,
  transpose = 2,
};

template <lsc_vector_size VS> constexpr void check_lsc_vector_size() {
  static_assert(VS == lsc_vector_size::n1 || VS == lsc_vector_size::n2 ||
                    VS == lsc_vector_size::n3 || VS == lsc_vector_size::n4 ||
                    VS == lsc_vector_size::n8 || VS == lsc_vector_size::n16 ||
                    VS == lsc_vector_size::n64 || VS == lsc_vector_size::n32,
                "Unsupported vector size");
}

template <uint8_t VS> constexpr void check_lsc_vector_size() {
  static_assert(VS == 1 || VS == 2 || VS == 3 || VS == 4 || VS == 8 ||
                    VS == 16 || VS == 32 || VS == 64,
                "Unsupported vector size");
}

template <typename T, lsc_data_size DS> constexpr void check_lsc_data_size() {
  static_assert(DS != lsc_data_size::default_size || sizeof(T) == 1 ||
                    sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                "Unsupported data type");
}

template <__ESIMD_NS::atomic_op Op> constexpr void check_lsc_atomic_op() {
  static_assert(Op == __ESIMD_NS::atomic_op::add ||
                    Op == __ESIMD_NS::atomic_op::sub ||
                    Op == __ESIMD_NS::atomic_op::inc ||
                    Op == __ESIMD_NS::atomic_op::dec ||
                    Op == __ESIMD_NS::atomic_op::min ||
                    Op == __ESIMD_NS::atomic_op::max ||
                    Op == __ESIMD_NS::atomic_op::cmpxchg ||
                    Op == __ESIMD_NS::atomic_op::bit_and ||
                    Op == __ESIMD_NS::atomic_op::bit_or ||
                    Op == __ESIMD_NS::atomic_op::bit_xor ||
                    Op == __ESIMD_NS::atomic_op::minsint ||
                    Op == __ESIMD_NS::atomic_op::maxsint ||
                    Op == __ESIMD_NS::atomic_op::fmax ||
                    Op == __ESIMD_NS::atomic_op::fmin ||
                    Op == __ESIMD_NS::atomic_op::fcmpwr ||
                    Op == __ESIMD_NS::atomic_op::fadd ||
                    Op == __ESIMD_NS::atomic_op::fsub ||
                    Op == __ESIMD_NS::atomic_op::load ||
                    Op == __ESIMD_NS::atomic_op::store,
                "Unsupported operation for LSC atomics");
}

/// Check the legality of lsc xatomic call in terms of size and type.
template <__ESIMD_NS::atomic_op Op, unsigned NumSrc>
constexpr void check_lsc_atomic() {
  check_lsc_atomic_op<Op>();
  if constexpr (Op == __ESIMD_NS::atomic_op::inc ||
                Op == __ESIMD_NS::atomic_op::dec ||
                Op == __ESIMD_NS::atomic_op::load) {
    static_assert(NumSrc == 0, "No source operands are expected");
  }
  if constexpr (Op == __ESIMD_NS::atomic_op::store ||
                Op == __ESIMD_NS::atomic_op::add ||
                Op == __ESIMD_NS::atomic_op::sub ||
                Op == __ESIMD_NS::atomic_op::minsint ||
                Op == __ESIMD_NS::atomic_op::maxsint ||
                Op == __ESIMD_NS::atomic_op::min ||
                Op == __ESIMD_NS::atomic_op::max ||
                Op == __ESIMD_NS::atomic_op::fadd ||
                Op == __ESIMD_NS::atomic_op::fsub ||
                Op == __ESIMD_NS::atomic_op::fmin ||
                Op == __ESIMD_NS::atomic_op::fmax ||
                Op == __ESIMD_NS::atomic_op::bit_and ||
                Op == __ESIMD_NS::atomic_op::bit_or ||
                Op == __ESIMD_NS::atomic_op::bit_xor) {
    static_assert(NumSrc == 1, "One source operand is expected");
  }
  if constexpr (Op == __ESIMD_NS::atomic_op::cmpxchg ||
                Op == __ESIMD_NS::atomic_op::fcmpwr) {
    static_assert(NumSrc == 2, "Two source operands are expected");
  }
}

template <__ESIMD_NS::atomic_op Op> constexpr lsc_atomic_op to_lsc_atomic_op() {
  check_lsc_atomic_op<Op>();
  switch (Op) {
  case __ESIMD_NS::atomic_op::add:
    return lsc_atomic_op::iadd;
  case __ESIMD_NS::atomic_op::sub:
    return lsc_atomic_op::isub;
  case __ESIMD_NS::atomic_op::inc:
    return lsc_atomic_op::iinc;
  case __ESIMD_NS::atomic_op::dec:
    return lsc_atomic_op::idec;
  case __ESIMD_NS::atomic_op::min:
    return lsc_atomic_op::umin;
  case __ESIMD_NS::atomic_op::max:
    return lsc_atomic_op::umax;
  case __ESIMD_NS::atomic_op::cmpxchg:
    return lsc_atomic_op::icas;
  case __ESIMD_NS::atomic_op::bit_and:
    return lsc_atomic_op::bit_and;
  case __ESIMD_NS::atomic_op::bit_or:
    return lsc_atomic_op::bit_or;
  case __ESIMD_NS::atomic_op::bit_xor:
    return lsc_atomic_op::bit_xor;
  case __ESIMD_NS::atomic_op::minsint:
    return lsc_atomic_op::smin;
  case __ESIMD_NS::atomic_op::maxsint:
    return lsc_atomic_op::smax;
  case __ESIMD_NS::atomic_op::fmax:
    return lsc_atomic_op::fmax;
  case __ESIMD_NS::atomic_op::fmin:
    return lsc_atomic_op::fmin;
  case __ESIMD_NS::atomic_op::fcmpwr:
    return lsc_atomic_op::fcas;
  case __ESIMD_NS::atomic_op::fadd:
    return lsc_atomic_op::fadd;
  case __ESIMD_NS::atomic_op::fsub:
    return lsc_atomic_op::fsub;
  case __ESIMD_NS::atomic_op::load:
    return lsc_atomic_op::load;
  case __ESIMD_NS::atomic_op::store:
    return lsc_atomic_op::store;
  default:
    return lsc_atomic_op::iinc;
  }
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

template <uint8_t VS> constexpr lsc_vector_size to_lsc_vector_size() {
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

constexpr lsc_data_size expand_data_size(lsc_data_size DS) {
  if (DS == lsc_data_size::u8)
    return lsc_data_size::u8u32;
  if (DS == lsc_data_size::u16)
    return lsc_data_size::u16u32;
  return DS;
}

template <typename T> struct lsc_expand_type {
  using type = typename std::conditional<sizeof(T) < 4, uint32_t, T>::type;
};

template <typename T> struct lsc_bitcast_type {
private:
  using _type1 = typename std::conditional<sizeof(T) == 2, uint16_t, T>::type;
  using _type2 = typename std::conditional<sizeof(T) == 1, uint8_t, T>::type;

public:
  using type =
      typename std::conditional<sizeof(_type2) == 1, _type2, _type1>::type;
};

} // namespace detail

/// L1 or L3 cache hint kinds.
enum class cache_hint : uint8_t {
  none = 0,
  uncached = 1,
  cached = 2,
  write_back = 3,
  write_through = 4,
  streaming = 5,
  read_invalidate = 6
};

namespace detail {

template <cache_hint Hint> class cache_hint_wrap {
  template <cache_hint...> class is_one_of_t;
  template <cache_hint Last>
  struct is_one_of_t<Last>
      : std::conditional<Last == Hint, std::true_type, std::false_type>::type {
  };
  template <cache_hint Head, cache_hint... Tail>
  struct is_one_of_t<Head, Tail...>
      : std::conditional<Head == Hint, std::true_type,
                         is_one_of_t<Tail...>>::type {};

public:
  constexpr operator cache_hint() const { return Hint; }
  template <cache_hint... Hints> constexpr bool is_one_of() const {
    return is_one_of_t<Hints...>::value;
  }
};

constexpr bool are_both(cache_hint First, cache_hint Second, cache_hint Val) {
  return First == Val && Second == Val;
}

enum class lsc_action { prefetch, load, store, atomic };

template <lsc_action Action, cache_hint L1, cache_hint L3>
constexpr void check_lsc_cache_hint() {
  constexpr auto L1H = cache_hint_wrap<L1>{};
  constexpr auto L3H = cache_hint_wrap<L3>{};
  if constexpr (Action == lsc_action::prefetch) {
    static_assert(
        L1H.template is_one_of<cache_hint::cached, cache_hint::uncached,
                               cache_hint::streaming>() &&
            L3H.template is_one_of<cache_hint::cached,
                                   cache_hint::uncached>() &&
            !are_both(L1H, L3H, cache_hint::uncached),
        "unsupported cache hint");
  } else if constexpr (Action == lsc_action::load) {
    static_assert(
        are_both(L1H, L3H, cache_hint::none) ||
            (L1H.template is_one_of<cache_hint::uncached, cache_hint::cached,
                                    cache_hint::streaming>() &&
             L3H.template is_one_of<cache_hint::uncached,
                                    cache_hint::cached>()) ||
            (L1H == cache_hint::read_invalidate && L3H == cache_hint::cached),
        "unsupported cache hint");
  } else if constexpr (Action == lsc_action::store) {
    static_assert(are_both(L1H, L3H, cache_hint::none) ||
                      are_both(L1H, L3H, cache_hint::write_back) ||
                      (L1H.template is_one_of<cache_hint::uncached,
                                              cache_hint::write_through,
                                              cache_hint::streaming>() &&
                       L3H.template is_one_of<cache_hint::uncached,
                                              cache_hint::write_back>()),
                  "unsupported cache hint");
  } else if constexpr (Action == lsc_action::atomic) {
    static_assert(are_both(L1H, L3H, cache_hint::none) ||
                      (L1H == cache_hint::uncached &&
                       L3H.template is_one_of<cache_hint::uncached,
                                              cache_hint::write_back>()),
                  "unsupported cache hint");
  }
}

} // namespace detail

/// Represents a split barrier action.
enum class split_barrier_action : uint8_t {
  wait = 0,   // split barrier wait
  signal = 1, // split barrier signal
};

/// @} sycl_esimd_core

} // namespace __ESIMD_ENS
} // __SYCL_INLINE_NAMESPACE(cl)
