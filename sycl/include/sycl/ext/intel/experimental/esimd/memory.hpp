//==-------------- memory.hpp - DPC++ Explicit SIMD API --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement experimental Explicit SIMD memory-access APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/common.hpp>
#include <sycl/ext/intel/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/util.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel {
namespace experimental::esimd {

/// @addtogroup sycl_esimd_memory
/// @{

/// Generic work-group split barrier.
/// @tparam flag  - split barrier action.
template <split_barrier_action flag> __ESIMD_API void split_barrier() {
  __esimd_sbarrier(flag);
}

__SYCL_DEPRECATED("use split_barrier<split_barrier_action>()")
__ESIMD_API void split_barrier(split_barrier_action flag) {
  __esimd_sbarrier(flag);
}

/// @} sycl_esimd_memory

// sycl_esimd_raw_send intrinsics are not available when stateless memory
// accesses are enforced.
#ifndef __ESIMD_FORCE_STATELESS_MEM

/// @addtogroup sycl_esimd_raw_send
/// @{

/// Raw sends.  "s" suffix designates "split" variant - i.e. two sources.
///
/// @param msgDst is the old value of the destination operand.
/// @param msgSrc0 is the first source operand of send message.
/// @param msgSrc1 is the second source operand of send message.
/// @param exDesc is the extended message descriptor.
/// @param msgDesc is the message descriptor.
/// @param execSize is the execution size, which must be a compile time
/// constant.
/// @param sfid is the shared function ID, which must be a compile time
/// constant.
/// @param numSrc0 is the number of GRFs for source-0, which must be a compile
/// time constant.
/// @param numSrc1 is the number of GRFs for source-1, which must be a compile
/// constant.
/// @param numDst is the number of GRFs for destination, which must be a compile
/// time constant.
/// @param isEOT is the flag that indicates whether this is an EOT message,
/// which must be a compile time constant (optional - default to 0).
/// @param isSendc is the flag that indicates whether sendc should be used,
/// which must be a compile time constant (optional - default to 0).
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
/// @return the vector value read from memory.
template <typename T1, int n1, typename T2, int n2, typename T3, int n3,
          int N = 16>
__ESIMD_API __ESIMD_NS::simd<T1, n1>
raw_sends(__ESIMD_NS::simd<T1, n1> msgDst, __ESIMD_NS::simd<T2, n2> msgSrc0,
          __ESIMD_NS::simd<T3, n3> msgSrc1, uint32_t exDesc, uint32_t msgDesc,
          uint8_t execSize, uint8_t sfid, uint8_t numSrc0, uint8_t numSrc1,
          uint8_t numDst, uint8_t isEOT = 0, uint8_t isSendc = 0,
          __ESIMD_NS::simd_mask<N> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send rspVar");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msgSrc0");
  constexpr unsigned _Width3 = n3 * sizeof(T3);
  static_assert(_Width3 % 32 == 0, "Invalid size for raw send msgSrc1");

  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  using ElemT2 = __ESIMD_DNS::__raw_t<T2>;
  using ElemT3 = __ESIMD_DNS::__raw_t<T3>;

  uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  return __esimd_raw_sends2<ElemT1, n1, ElemT2, n2, ElemT3, n3, N>(
      modifier, execSize, mask.data(), numSrc0, numSrc1, numDst, sfid, exDesc,
      msgDesc, msgSrc0.data(), msgSrc1.data(), msgDst.data());
}

template <typename T1, int n1, typename T2, int n2, typename T3, int n3,
          int N = 16>
__SYCL_DEPRECATED("raw_sends_load is deprecated. Use raw_sends")
__ESIMD_API __ESIMD_NS::simd<T1, n1> raw_sends_load(
    __ESIMD_NS::simd<T1, n1> msgDst, __ESIMD_NS::simd<T2, n2> msgSrc0,
    __ESIMD_NS::simd<T3, n3> msgSrc1, uint32_t exDesc, uint32_t msgDesc,
    uint8_t execSize, uint8_t sfid, uint8_t numSrc0, uint8_t numSrc1,
    uint8_t numDst, uint8_t isEOT = 0, uint8_t isSendc = 0,
    __ESIMD_NS::simd_mask<N> mask = 1) {
  return raw_sends(msgDst, msgSrc0, msgSrc1, exDesc, msgDesc, execSize, sfid,
                   numSrc0, numSrc1, numDst, isEOT, isSendc);
}

/// Raw send.
///
/// @param msgDst is the old value of the destination operand.
/// @param msgSrc0 is the first source operand of send message.
/// @param exDesc is the extended message descriptor.
/// @param msgDesc is the message descriptor.
/// @param execSize is the execution size, which must be a compile time
/// constant.
/// @param sfid is the shared function ID, which must be a compile time
/// constant.
/// @param numSrc0 is the number of GRFs for source-0, which must be a compile
/// time constant.
/// @param numDst is the number of GRFs for destination, which must be a compile
/// time constant.
/// @param isEOT is the flag that indicates whether this is an EOT message,
/// which must be a compile time constant (optional - default to 0).
/// @param isSendc is the flag that indicates whether sendc should be used,
/// which must be a compile time constant (optional - default to 0).
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
/// @return the vector value read from memory.
template <typename T1, int n1, typename T2, int n2, int N = 16>
__ESIMD_API __ESIMD_NS::simd<T1, n1>
raw_send(__ESIMD_NS::simd<T1, n1> msgDst, __ESIMD_NS::simd<T2, n2> msgSrc0,
         uint32_t exDesc, uint32_t msgDesc, uint8_t execSize, uint8_t sfid,
         uint8_t numSrc0, uint8_t numDst, uint8_t isEOT = 0,
         uint8_t isSendc = 0, __ESIMD_NS::simd_mask<N> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send rspVar");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msgSrc0");

  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  using ElemT2 = __ESIMD_DNS::__raw_t<T2>;

  uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  return __esimd_raw_send2<ElemT1, n1, ElemT2, n2, N>(
      modifier, execSize, mask.data(), numSrc0, numDst, sfid, exDesc, msgDesc,
      msgSrc0.data(), msgDst.data());
}

template <typename T1, int n1, typename T2, int n2, int N = 16>
__SYCL_DEPRECATED("raw_send_load is deprecated. Use raw_send")
__ESIMD_API __ESIMD_NS::simd<T1, n1> raw_send_load(
    __ESIMD_NS::simd<T1, n1> msgDst, __ESIMD_NS::simd<T2, n2> msgSrc0,
    uint32_t exDesc, uint32_t msgDesc, uint8_t execSize, uint8_t sfid,
    uint8_t numSrc0, uint8_t numDst, uint8_t isEOT = 0, uint8_t isSendc = 0,
    __ESIMD_NS::simd_mask<N> mask = 1) {
  return raw_send(msgDst, msgSrc0, exDesc, msgDesc, execSize, sfid, numSrc0,
                  numDst, isEOT, isSendc, mask);
}

/// Raw sends. "s" suffix designates "split" variant - i.e. two sources.
///
/// @param msgSrc0 is the first source operand of send message.
/// @param msgSrc1 is the second source operand of send message.
/// @param exDesc is the extended message descriptor.
/// @param msgDesc is the message descriptor.
/// @param execSize is the execution size, which must be a compile time
/// constant.
/// @param sfid is the shared function ID, which must be a compile time
/// constant.
/// @param numSrc0 is the number of GRFs for source-0, which must be a compile
/// time constant.
/// @param numSrc1 is the number of GRFs for source-1, which must be a compile
/// time constant.
/// @param isEOT is the flag that indicates whether this is an EOT message,
/// which must be a compile time constant (optional - default to 0).
/// @param isSendc is the flag that indicates whether sendc should be used,
/// which must be a compile time constant (optional - default to 0).
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
template <typename T1, int n1, typename T2, int n2, int N = 16>
__ESIMD_API void
raw_sends(__ESIMD_NS::simd<T1, n1> msgSrc0, __ESIMD_NS::simd<T2, n2> msgSrc1,
          uint32_t exDesc, uint32_t msgDesc, uint8_t execSize, uint8_t sfid,
          uint8_t numSrc0, uint8_t numSrc1, uint8_t isEOT = 0,
          uint8_t isSendc = 0, __ESIMD_NS::simd_mask<N> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send msgSrc0");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msgSrc1");

  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  using ElemT2 = __ESIMD_DNS::__raw_t<T2>;

  uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  __esimd_raw_sends2_noresult<ElemT1, n1, ElemT2, n2, N>(
      modifier, execSize, mask.data(), numSrc0, numSrc1, sfid, exDesc, msgDesc,
      msgSrc0.data(), msgSrc1.data());
}

template <typename T1, int n1, typename T2, int n2, int N = 16>
__SYCL_DEPRECATED("raw_sends_store is deprecated. Use raw_sends")
__ESIMD_API
    void raw_sends_store(__ESIMD_NS::simd<T1, n1> msgSrc0,
                         __ESIMD_NS::simd<T2, n2> msgSrc1, uint32_t exDesc,
                         uint32_t msgDesc, uint8_t execSize, uint8_t sfid,
                         uint8_t numSrc0, uint8_t numSrc1, uint8_t isEOT = 0,
                         uint8_t isSendc = 0,
                         __ESIMD_NS::simd_mask<N> mask = 1) {
  raw_sends(msgSrc0, msgSrc1, exDesc, msgDesc, execSize, sfid, numSrc0, numSrc1,
            isEOT, isSendc, mask);
}

/// Raw send. Generates a \c send or \c sendc instruction for the message
/// gateway.
///
/// @param msgSrc0 is the first source operand of send message.
/// @param exDesc is the extended message descriptor.
/// @param msgDesc is the message descriptor.
/// @param execSize is the execution size, which must be a compile time
/// constant.
/// @param sfid is the shared function ID, which must be a compile time
/// constant.
/// @param numSrc0 is the number of GRFs for source-0, which must be a compile
/// time constant.
/// @param isEOT is the flag that indicates whether this is an EOT message,
/// which must be a compile time constant (optional - default to 0).
/// @param isSendc is the flag that indicates whether sendc should be used,
/// which must be a compile time constant (optional - default to 0).
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
template <typename T1, int n1, int N = 16>
__ESIMD_API void
raw_send(__ESIMD_NS::simd<T1, n1> msgSrc0, uint32_t exDesc, uint32_t msgDesc,
         uint8_t execSize, uint8_t sfid, uint8_t numSrc0, uint8_t isEOT = 0,
         uint8_t isSendc = 0, __ESIMD_NS::simd_mask<N> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send msgSrc0");
  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  __esimd_raw_send2_noresult<ElemT1, n1, N>(modifier, execSize, mask.data(),
                                            numSrc0, sfid, exDesc, msgDesc,
                                            msgSrc0.data());
}

template <typename T1, int n1, int N = 16>
__SYCL_DEPRECATED("raw_send_store is deprecated. Use raw_send")
__ESIMD_API
    void raw_send_store(__ESIMD_NS::simd<T1, n1> msgSrc0, uint32_t exDesc,
                        uint32_t msgDesc, uint8_t execSize, uint8_t sfid,
                        uint8_t numSrc0, uint8_t isEOT = 0, uint8_t isSendc = 0,
                        __ESIMD_NS::simd_mask<N> mask = 1) {
  raw_send(msgSrc0, exDesc, msgDesc, execSize, sfid, numSrc0, isEOT, isSendc,
           mask);
}

/// @} sycl_esimd_raw_send

#endif // !__ESIMD_FORCE_STATELESS_MEM

/// @defgroup sycl_esimd_memory_nbarrier Named barrier APIs.
/// @ingroup sycl_esimd_memory

/// @addtogroup sycl_esimd_memory_nbarrier
/// @{

/// Wait on a named barrier
/// Available only on PVC
///
/// @param id  - named barrier id
__ESIMD_API void named_barrier_wait(uint8_t id) {
  __esimd_nbarrier(0 /*wait*/, id, 0 /*thread count*/);
}

/// Initialize number of named barriers for a kernel
/// Available only on PVC
///
/// @tparam NbarCount  - number of named barriers
template <uint8_t NbarCount> __ESIMD_API void named_barrier_init() {
  __esimd_nbarrier_init(NbarCount);
}

/// Perform signal operation for the given named barrier
/// Available only on PVC
///
/// @param barrier_id  - named barrier id
///
/// @param producer_consumer_mode  - 2-bit flag to indicate if it's producer
/// mode (0x1) or consumer mode (0x2). User must ensure the input value is set
/// correctly and higher order bits are cleared.
///
/// @param num_producers  - number of producers
///
/// @param num_consumers  - number of consumers
__ESIMD_API void named_barrier_signal(uint8_t barrier_id,
                                      uint8_t producer_consumer_mode,
                                      uint32_t num_producers,
                                      uint32_t num_consumers) {
  constexpr uint32_t gateway = 3;
  constexpr uint32_t barrier = 4;
  constexpr uint32_t descriptor = 1 << 25 | // Message length: 1 register
                                  0 << 12 | // Fence Data Ports: No fence
                                  barrier;  // Barrier subfunction

  __ESIMD_DNS::vector_type_t<uint32_t, 8> payload = 0;
  payload[2] = (num_consumers & 0xff) << 24 | (num_producers & 0xff) << 16 |
               producer_consumer_mode << 14 | (barrier_id & 0b11111) << 0;

  __esimd_raw_send_nbarrier_signal<uint32_t, 8>(
      0 /*sendc*/, gateway, descriptor, payload, 1 /*pred*/);
}

/// Create explicit scoreboard dependency to avoid device code motion
/// across this call and preserve the \p value computation even
/// if it is unused.
template <typename T, int N>
__ESIMD_API std::enable_if_t<(sizeof(T) * N >= 2)>
wait(__ESIMD_NS::simd<T, N> value) {
#ifdef __SYCL_DEVICE_ONLY__
  uint16_t Word = value.template bit_cast_view<uint16_t>()[0];
  __esimd_wait(Word);
#endif // __SYCL_DEVICE_ONLY__
}

/// Create explicit scoreboard dependency to avoid device code motion
/// across this call and preserve the \p value computation even
/// if it is unused.
template <typename T, typename RegionT>
__ESIMD_API std::enable_if_t<
    (RegionT::length * sizeof(typename RegionT::element_type) >= 2)>
wait(__ESIMD_NS::simd_view<T, RegionT> value) {
#ifdef __SYCL_DEVICE_ONLY__
  uint16_t Word = value.template bit_cast_view<uint16_t>()[0];
  __esimd_wait(Word);
#endif // __SYCL_DEVICE_ONLY__
}

/// @} sycl_esimd_memory_nbarrier

/// @defgroup sycl_esimd_memory_lsc LSC memory access APIs.
/// @ingroup sycl_esimd_memory

/// @addtogroup sycl_esimd_memory_lsc
/// @{

namespace detail {
// Compute the data size for 2d block load or store.
template <typename T, int NBlocks, int Height, int Width, bool Transposed,
          bool Transformed>
constexpr int get_lsc_block_2d_data_size() {
  if (Transformed)
    return detail::roundUpNextMultiple<Height, 4 / sizeof(T)>() *
           __ESIMD_DNS::getNextPowerOf2<Width>() * NBlocks;
  return Width * Height * NBlocks;
}

// Format u8 and u16 to u8u32 and u16u32 by doing garbage-extension.
template <typename RT, typename T, int N>
ESIMD_INLINE __ESIMD_NS::simd<RT, N>
lsc_format_input(__ESIMD_NS::simd<T, N> Vals) {
  if constexpr (sizeof(T) == 1) {
    // Extend bytes to RT.
    return Vals.template bit_cast_view<uint8_t>();
  } else if constexpr (sizeof(T) == 2) {
    // Extend words to RT.
    return Vals.template bit_cast_view<uint16_t>();
  } else {
    return Vals.template bit_cast_view<RT>();
  }
}

// Format u8u32 and u16u32 back to u8 and u16.
template <typename T, typename T1, int N>
ESIMD_INLINE __ESIMD_NS::simd<T, N>
lsc_format_ret(__ESIMD_NS::simd<T1, N> Vals) {
  auto Formatted = Vals.template bit_cast_view<T>();
  if constexpr (sizeof(T) == sizeof(T1)) {
    return Formatted;
  } else {
    constexpr int Stride = Formatted.length / N;
    return Formatted.template select<N, Stride>(0);
  }
}

/// Check the legality of lsc atomic call in terms of size and type.
template <__ESIMD_NS::native::lsc::atomic_op Op, typename T, int N,
          unsigned NumSrc>
constexpr void check_lsc_atomic() {
  if constexpr (!__ESIMD_DNS::isPowerOf2(N, 32)) {
    static_assert((__ESIMD_DNS::isPowerOf2(N, 32)),
                  "Execution size 1, 2, 4, 8, 16, 32 are supported");
  }
  if constexpr (NumSrc != __ESIMD_DNS::get_num_args<Op>()) {
    static_assert(NumSrc == __ESIMD_DNS::get_num_args<Op>(),
                  "wrong number of operands");
  }
  if constexpr (Op == __ESIMD_NS::native::lsc::atomic_op::fcmpxchg) {
    static_assert(__ESIMD_DNS::is_type<T, float, sycl::half, double>(),
                  "float, double or sycl::half type is expected");
  } else {
    __ESIMD_DNS::check_atomic<__ESIMD_DNS::to_atomic_op<Op>(), T, N, NumSrc>();
  }
}

template <cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
constexpr uint32_t get_lsc_load_cache_mask() {
  if constexpr (L1H == cache_hint::read_invalidate &&
                L3H == cache_hint::cached) {
    return 7;
  }
  if constexpr (L1H == cache_hint::streaming && L3H == cache_hint::cached) {
    return 6;
  }
  if constexpr (L1H == cache_hint::streaming && L3H == cache_hint::uncached) {
    return 5;
  }
  if constexpr (L1H == cache_hint::cached && L3H == cache_hint::cached) {
    return 4;
  }
  if constexpr (L1H == cache_hint::cached && L3H == cache_hint::uncached) {
    return 3;
  }
  if constexpr (L1H == cache_hint::uncached && L3H == cache_hint::cached) {
    return 2;
  }
  if constexpr (L1H == cache_hint::uncached && L3H == cache_hint::uncached) {
    return 1;
  }
  return 0;
}

template <cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
constexpr uint32_t get_lsc_store_cache_mask() {
  if constexpr (L1H == cache_hint::write_back && L3H == cache_hint::cached) {
    return 7;
  }
  if constexpr (L1H == cache_hint::streaming && L3H == cache_hint::cached) {
    return 6;
  }
  if constexpr (L1H == cache_hint::streaming && L3H == cache_hint::uncached) {
    return 5;
  }
  if constexpr (L1H == cache_hint::write_through && L3H == cache_hint::cached) {
    return 4;
  }
  if constexpr (L1H == cache_hint::write_through &&
                L3H == cache_hint::uncached) {
    return 3;
  }
  if constexpr (L1H == cache_hint::uncached && L3H == cache_hint::cached) {
    return 2;
  }
  if constexpr (L1H == cache_hint::uncached && L3H == cache_hint::uncached) {
    return 1;
  }
  return 0;
}

} // namespace detail

/// SLM gather.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.slm
///
/// Collects elements located at slm and returns them
/// as a single \ref simd object.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam N is the number of channels (platform dependent).
/// @param offsets is the zero-based offsets for SLM buffer in bytes.
/// @param pred is predicates.
/// @return is a vector of type T and size N * NElts
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size, int N>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
lsc_slm_gather(__ESIMD_NS::simd<uint32_t, N> offsets,
               __ESIMD_NS::simd_mask<N> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr auto _Transposed = detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<_MsgT, N *NElts> Tmp =
      __esimd_lsc_load_slm<_MsgT, cache_hint::none, cache_hint::none,
                           _AddressScale, _ImmOffset, _DS, _VS, _Transposed, N>(
          pred.data(), offsets.data());
  return detail::lsc_format_ret<T>(Tmp);
}

/// SLM gather.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.slm
///
/// Collects elements located at slm and returns them
/// as a single \ref simd object.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam N is the number of channels (platform dependent).
/// @param offsets is the zero-based offsets for SLM buffer in bytes.
/// @param pred is predicates.
/// @param old_values values copied to the result when the corresponding
/// element of \p pred is zero..
/// @return is a vector of type T and size N * NElts
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size, int N>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
lsc_slm_gather(__ESIMD_NS::simd<uint32_t, N> offsets,
               __ESIMD_NS::simd_mask<N> pred,
               __ESIMD_NS::simd<T, N * NElts> old_values) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<_MsgT, N *NElts> OldValuesExpanded =
      detail::lsc_format_input<_MsgT>(old_values);
  __ESIMD_NS::simd<_MsgT, N *NElts> Result =
      __esimd_lsc_load_merge_slm<_MsgT, cache_hint::none, cache_hint::none,
                                 _AddressScale, _ImmOffset, _DS, _VS,
                                 _Transposed, N>(pred.data(), offsets.data(),
                                                 OldValuesExpanded.data());
  return detail::lsc_format_ret<T>(Result);
}

/// Transposed SLM gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.slm
///
/// Collects elements located at slm and returns them
/// as a single \ref simd object.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @param offset is the zero-based offset for SLM buffer in bytes.
/// @param pred is the predicate; if it contains 0, then the actual load
/// is not performed and the returned value is undefined.
/// @return is a vector of type T and size NElts
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, NElts>
lsc_slm_block_load(uint32_t offset, __ESIMD_NS::simd_mask<1> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size FDS = detail::finalize_data_size<T, DS>();
  static_assert(FDS == lsc_data_size::u32 || FDS == lsc_data_size::u64,
                "Transposed load is supported only for data size u32 or u64");
  constexpr detail::lsc_vector_size VS = detail::to_lsc_vector_size<NElts>();
  constexpr auto Transposed = detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd<uint32_t, N> offsets = offset;
  return __esimd_lsc_load_slm<T, cache_hint::none, cache_hint::none,
                              AddressScale, ImmOffset, FDS, VS, Transposed, N>(
      pred.data(), offsets.data());
}

/// Transposed SLM gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.slm
///
/// Collects elements located at slm and returns them
/// as a single \ref simd object.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @param offset is the zero-based offset for SLM buffer in bytes.
/// @param pred is the predicate; if it contains 0, then the actual load
/// is not performed and \p old_values is returned.
/// @param old_values contains the vector that is returned if
/// the parameter \p pred contains 0.
/// @return is a vector of type T and size NElts.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, NElts>
lsc_slm_block_load(uint32_t offset, __ESIMD_NS::simd_mask<1> pred,
                   __ESIMD_NS::simd<T, NElts> old_values) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size FDS = detail::finalize_data_size<T, DS>();
  static_assert(FDS == lsc_data_size::u32 || FDS == lsc_data_size::u64,
                "Transposed load is supported only for data size u32 or u64");
  constexpr detail::lsc_vector_size VS = detail::to_lsc_vector_size<NElts>();
  constexpr auto Transposed = detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd<uint32_t, N> offsets = offset;
  return __esimd_lsc_load_merge_slm<T, cache_hint::none, cache_hint::none,
                                    AddressScale, ImmOffset, FDS, VS,
                                    Transposed, N>(pred.data(), offsets.data(),
                                                   old_values.data());
}

/// USM pointer gather.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Collects elements located at specified address and returns them
/// as a single \ref simd object.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
/// @return is a vector of type T and size N * NElts
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename Toffset>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
lsc_gather(const T *p, __ESIMD_NS::simd<Toffset, N> offsets,
           __ESIMD_NS::simd_mask<N> pred = 1) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr auto _Transposed = detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __ESIMD_NS::simd<_MsgT, N *NElts> Tmp =
      __esimd_lsc_load_stateless<_MsgT, L1H, L3H, _AddressScale, _ImmOffset,
                                 _DS, _VS, _Transposed, N>(pred.data(),
                                                           addrs.data());
  return detail::lsc_format_ret<T>(Tmp);
}

/// USM pointer gather.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Collects elements located at specified address and returns them
/// as a single \ref simd object.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
/// @param old_values contains the vector which elements are copied
/// to the returned result when the corresponding element of \p pred is 0.
/// @return is a vector of type T and size N * NElts
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename Toffset>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
lsc_gather(const T *p, __ESIMD_NS::simd<Toffset, N> offsets,
           __ESIMD_NS::simd_mask<N> pred,
           __ESIMD_NS::simd<T, N * NElts> old_values) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr auto _Transposed = detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<uintptr_t, N> Addrs = reinterpret_cast<uintptr_t>(p);
  Addrs += convert<uintptr_t>(offsets);
  __ESIMD_NS::simd<_MsgT, N *NElts> OldValuesExpanded =
      detail::lsc_format_input<_MsgT>(old_values);
  __ESIMD_NS::simd<_MsgT, N *NElts> Result =
      __esimd_lsc_load_merge_stateless<_MsgT, L1H, L3H, _AddressScale,
                                       _ImmOffset, _DS, _VS, _Transposed, N>(
          pred.data(), Addrs.data(), OldValuesExpanded.data());
  return detail::lsc_format_ret<T>(Result);
}

template <
    typename T, int NElts = 1, lsc_data_size DS = lsc_data_size::default_size,
    cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none, int N,
    typename Toffset, typename RegionTy = __ESIMD_NS::region1d_t<Toffset, N, 1>>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
lsc_gather(const T *p, __ESIMD_NS::simd_view<Toffset, RegionTy> offsets,
           __ESIMD_NS::simd_mask<N> pred = 1) {
  using Ty = typename __ESIMD_NS::simd_view<Toffset, RegionTy>::element_type;
  return lsc_gather<T, NElts, DS, L1H, L3H, N>(
      p, __ESIMD_NS::simd<Ty, N>(offsets), pred);
}

template <
    typename T, int NElts = 1, lsc_data_size DS = lsc_data_size::default_size,
    cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none, int N,
    typename Toffset, typename RegionTy = __ESIMD_NS::region1d_t<Toffset, N, 1>>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
lsc_gather(const T *p, __ESIMD_NS::simd_view<Toffset, RegionTy> offsets,
           __ESIMD_NS::simd_mask<N> pred,
           __ESIMD_NS::simd<T, N * NElts> old_values) {
  using Ty = typename __ESIMD_NS::simd_view<Toffset, RegionTy>::element_type;
  return lsc_gather<T, NElts, DS, L1H, L3H, N>(
      p, __ESIMD_NS::simd<Ty, N>(offsets), pred, old_values);
}

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset>,
                             __ESIMD_NS::simd<T, N * NElts>>
lsc_gather(const T *p, Toffset offset, __ESIMD_NS::simd_mask<N> pred = 1) {
  return lsc_gather<T, NElts, DS, L1H, L3H, N>(
      p, __ESIMD_NS::simd<Toffset, N>(offset), pred);
}

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset>,
                             __ESIMD_NS::simd<T, N * NElts>>
lsc_gather(const T *p, Toffset offset, __ESIMD_NS::simd_mask<N> pred,
           __ESIMD_NS::simd<T, N * NElts> old_values) {
  return lsc_gather<T, NElts, DS, L1H, L3H, N>(
      p, __ESIMD_NS::simd<Toffset, N>(offset), pred, old_values);
}

/// Accessor-based gather.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
/// @return is a vector of type T and size N * NElts
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer_v<AccessorTy>,
                             __ESIMD_NS::simd<T, N * NElts>>
lsc_gather(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
           __ESIMD_NS::simd_mask<N> pred = 1) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_gather<T, NElts, DS, L1H, L3H>(acc.get_pointer().get(), offsets,
                                            pred);
#else
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  auto si = __ESIMD_NS::get_surface_index(acc);
  __ESIMD_NS::simd<_MsgT, N *NElts> Tmp =
      __esimd_lsc_load_bti<_MsgT, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(), si);
  return detail::lsc_format_ret<T>(Tmp);
#endif
}

/// Accessor-based gather.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
/// @param old_values contains the vector which elements are copied
/// to the returned result when the corresponding element of \p pred is 0.
/// @return is a vector of type T and size N * NElts
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer_v<AccessorTy>,
                             __ESIMD_NS::simd<T, N * NElts>>
lsc_gather(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
           __ESIMD_NS::simd_mask<N> pred,
           __ESIMD_NS::simd<T, N * NElts> old_values) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_gather<T, NElts, DS, L1H, L3H>(acc.get_pointer().get(), offsets,
                                            pred, old_values);
#else
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr auto _Transposed = detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  auto SI = __ESIMD_NS::get_surface_index(acc);
  __ESIMD_NS::simd<_MsgT, N *NElts> OldValuesExpanded =
      detail::lsc_format_input<_MsgT>(old_values);
  __ESIMD_NS::simd<_MsgT, N *NElts> Result =
      __esimd_lsc_load_merge_bti<_MsgT, L1H, L3H, _AddressScale, _ImmOffset,
                                 _DS, _VS, _Transposed, N>(
          pred.data(), offsets.data(), SI, OldValuesExpanded.data());
  return detail::lsc_format_ret<T>(Result);
#endif
}

/// USM pointer transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Accesses contiguous block of memory of `NElts * S` bytes  starting from
/// given address, where S is a byte size of an "element" defined by the \c DS
/// template parameter. The maximum size of accessed block is 512 bytes for PVC
/// and 256 bytes for ACM (DG2).
/// When \c DS equals \c lsc_data_size::u64 or \c sizeof(T) equal to 8 the
/// address must be 8-byte aligned, otherwise - 4-bytes aligned. Allowed values
/// for the data size are \c lsc_data_size::u32, \c lsc_data_size::u64,
/// \c lsc_data_size::u8, \c lsc_data_size::u16.
/// When data size is either  \c lsc_data_size::u8 or \c lsc_data_size::u16
/// the data is treated as 32 bit data. Allowed NElts values for 64 bit
/// data are 1, 2, 3, 4, 8, 16, 32, 64.
/// When NElts is greater than 64 and the data size is not \c
/// lsc_data_size::u64, the data is treated as 64 bit blocks and hence must
/// follow the rules set for 64 bit data i.e. 8 byte alignment, the data must
/// fit whole number of 64 bit words (i.e. \c sizeof(T) \c * \c NELTS
/// is multiple of 8) and less than 64 words to transfer.
/// For example to access 512 bytes DS can be \c lsc_data_size::u64 and \c NElts
/// 64 or DS can be lsc_data_size::u8 and \c NElts 512. However in both cases
/// the data must be 8 bytes aligned.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p is the base pointer.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed. The default is '1' -
/// perform the operation.
/// @param flags is the alignment specifier type tag.
/// @return is a vector of type T and size NElts. The elements of the
/// returned vector for which the corresponding element in \p pred is 0
/// are undefined.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<__ESIMD_NS::is_simd_flag_type_v<FlagsT>,
                             __ESIMD_NS::simd<T, NElts>>
lsc_block_load(const T *p, __ESIMD_NS::simd_mask<1> pred = 1,
               FlagsT flags = FlagsT{}) {
  // Verify input template arguments.
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr lsc_data_size FDS = detail::finalize_data_size<T, DS>();

  static_assert(FDS == lsc_data_size::u16 || FDS == lsc_data_size::u8 ||
                    FDS == lsc_data_size::u32 || FDS == lsc_data_size::u64,
                "Conversion data types are not supported");
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor32Bit =
      (FDS == lsc_data_size::u16) ? 2 : (FDS == lsc_data_size::u8 ? 4 : 1);
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");

  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (sizeof(T) == 8 ||
       (DS == lsc_data_size::default_size && NElts / SmallIntFactor32Bit > 64 &&
        (NElts * sizeof(T)) % 8 == 0));
  constexpr int SmallIntFactor64Bit =
      (FDS == lsc_data_size::u16)
          ? 4
          : (FDS == lsc_data_size::u8 ? 8
                                      : (FDS == lsc_data_size::u32 ? 2 : 1));
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  detail::check_lsc_vector_size<FactoredNElts>();

  // Prepare template arguments for the call of intrinsic.
  constexpr lsc_data_size ActualDS = Use64BitData
                                         ? __ESIMD_ENS::lsc_data_size::u64
                                         : __ESIMD_ENS::lsc_data_size::u32;

  constexpr detail::lsc_vector_size _VS =
      detail::to_lsc_vector_size<FactoredNElts>();
  using LoadElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;

  constexpr auto _Transposed = detail::lsc_data_order::transpose;
  constexpr int N = 1;

  __ESIMD_NS::simd<uintptr_t, N> Addrs = reinterpret_cast<uintptr_t>(p);

  __ESIMD_NS::simd<LoadElemT, FactoredNElts> Result =
      __esimd_lsc_load_stateless<LoadElemT, L1H, L3H, _AddressScale, _ImmOffset,
                                 ActualDS, _VS, _Transposed, N>(pred.data(),
                                                                Addrs.data());
  return Result.template bit_cast_view<T>();
}

/// A variation of lsc_block_load without predicate parameter to simplify use
/// of alignment parameter
///
/// Accesses contiguous block of memory of `NElts * S` bytes  starting from
/// given address, where S is a byte size of an "element" defined by the \c DS
/// template parameter. The maximum size of accessed block is 512 bytes for PVC
/// and 256 bytes for ACM (DG2).
/// When \c DS equals \c lsc_data_size::u64 or \c sizeof(T) equal to 8 the
/// address must be 8-byte aligned, otherwise - 4-bytes aligned. Allowed values
/// for the data size are \c lsc_data_size::u32, \c lsc_data_size::u64,
/// \c lsc_data_size::u8, \c lsc_data_size::u16.
/// When data size is either  \c lsc_data_size::u8 or \c lsc_data_size::u16
/// the data is treated as 32 bit data. Allowed NElts values for 64 bit
/// data are 1, 2, 3, 4, 8, 16, 32, 64.
/// When NElts is greater than 64 and the data size is not \c
/// lsc_data_size::u64, the data is treated as 64 bit blocks and hence must
/// follow the rules set for 64 bit data i.e. 8 byte alignment, the data must
/// fit whole number of 64 bit words (i.e. \c sizeof(T) \c * \c NELTS
/// is multiple of 8) and less than 64 words to transfer.
/// For example to access 512 bytes DS can be \c lsc_data_size::u64 and \c NElts
/// 64 or DS can be lsc_data_size::u8 and \c NElts 512. However in both cases
/// the data must be 8 bytes aligned.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p is the base pointer.
/// @param flags is the alignment specifier type tag.
/// @return is a vector of type T and size NElts. The elements of the
/// returned vector for which the corresponding element in \p pred is 0
/// are undefined.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<__ESIMD_NS::is_simd_flag_type_v<FlagsT>,
                             __ESIMD_NS::simd<T, NElts>>
lsc_block_load(const T *p, FlagsT flags) {
  return lsc_block_load<T, NElts, DS, L1H, L3H>(p, __ESIMD_NS::simd_mask<1>(1),
                                                flags);
}

/// USM pointer transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Accesses contiguous block of memory of `NElts * S` bytes  starting from
/// given address, where S is a byte size of an "element" defined by the \c DS
/// template parameter. The maximum size of accessed block is 512 bytes for PVC
/// and 256 bytes for ACM (DG2).
/// When \c DS equals \c lsc_data_size::u64 or \c sizeof(T) equal to 8 the
/// address must be 8-byte aligned, otherwise - 4-bytes aligned. Allowed values
/// for the data size are \c lsc_data_size::u32, \c lsc_data_size::u64,
/// \c lsc_data_size::u8, \c lsc_data_size::u16.
/// When data size is either  \c lsc_data_size::u8 or \c lsc_data_size::u16
/// the data is treated as 32 bit data. Allowed NElts values for 64 bit
/// data are 1, 2, 3, 4, 8, 16, 32, 64.
/// When NElts is greater than 64 and the data size is not \c
/// lsc_data_size::u64, the data is treated as 64 bit blocks and hence must
/// follow the rules set for 64 bit data i.e. 8 byte alignment, the data must
/// fit whole number of 64 bit words (i.e. \c sizeof(T) \c * \c NELTS
/// is multiple of 8) and less than 64 words to transfer.
/// For example to access 512 bytes DS can be \c lsc_data_size::u64 and \c NElts
/// 64 or DS can be lsc_data_size::u8 and \c NElts 512. However in both cases
/// the data must be 8 bytes aligned.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p is the base pointer.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed.
/// @param old_values contains the vector which elements are copied
/// to the returned result when the corresponding element of \p pred is 0.
/// @param flags is the alignment specifier type tag.
/// @return is a vector of type T and size NElts.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<__ESIMD_NS::is_simd_flag_type_v<FlagsT>,
                             __ESIMD_NS::simd<T, NElts>>
lsc_block_load(const T *p, __ESIMD_NS::simd_mask<1> pred,
               __ESIMD_NS::simd<T, NElts> old_values, FlagsT flags = FlagsT{}) {
  // Verify input template arguments.
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr lsc_data_size FDS = detail::finalize_data_size<T, DS>();
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");
  static_assert(FDS == lsc_data_size::u16 || FDS == lsc_data_size::u8 ||
                    FDS == lsc_data_size::u32 || FDS == lsc_data_size::u64,
                "Conversion data types are not supported");
  constexpr int SmallIntFactor32Bit =
      (FDS == lsc_data_size::u16) ? 2 : (FDS == lsc_data_size::u8 ? 4 : 1);
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");

  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (sizeof(T) == 8 ||
       (DS == lsc_data_size::default_size && NElts / SmallIntFactor32Bit > 64 &&
        (NElts * sizeof(T)) % 8 == 0));
  constexpr int SmallIntFactor64Bit =
      (FDS == lsc_data_size::u16)
          ? 4
          : (FDS == lsc_data_size::u8 ? 8
                                      : (FDS == lsc_data_size::u32 ? 2 : 1));
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  detail::check_lsc_vector_size<FactoredNElts>();

  // Prepare template arguments for the call of intrinsic.
  constexpr lsc_data_size ActualDS = Use64BitData
                                         ? __ESIMD_ENS::lsc_data_size::u64
                                         : __ESIMD_ENS::lsc_data_size::u32;

  constexpr detail::lsc_vector_size _VS =
      detail::to_lsc_vector_size<FactoredNElts>();
  using LoadElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;

  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;

  constexpr auto _Transposed = detail::lsc_data_order::transpose;
  constexpr int N = 1;

  __ESIMD_NS::simd<uintptr_t, N> Addrs = reinterpret_cast<uintptr_t>(p);
  __ESIMD_NS::simd<LoadElemT, FactoredNElts> OldVals =
      old_values.template bit_cast_view<LoadElemT>();
  __ESIMD_NS::simd<LoadElemT, FactoredNElts> Result =
      __esimd_lsc_load_merge_stateless<LoadElemT, L1H, L3H, _AddressScale,
                                       _ImmOffset, ActualDS, _VS, _Transposed,
                                       N>(pred.data(), Addrs.data(),
                                          OldVals.data());
  return Result.template bit_cast_view<T>();
}

/// Accessor-based transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
/// See comments in the  \ref lsc_block_load API for description and parameter
/// constraints.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed. The default is '1' - perform
/// the operation.
/// @param flags is the alignment specifier type tag.
/// @return is a vector of type T and size NElts. The elements of the returned
/// vector for which the corresponding element in \p pred is 0 are undefined.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value &&
                                 __ESIMD_NS::is_simd_flag_type_v<FlagsT>,
                             __ESIMD_NS::simd<T, NElts>>
lsc_block_load(AccessorTy acc, uint32_t offset,
               __ESIMD_NS::simd_mask<1> pred = 1, FlagsT flags = FlagsT{}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_block_load<T, NElts, DS, L1H, L3H>(
      __ESIMD_DNS::accessorToPointer<T>(acc, offset), pred, flags);
#else  // !__ESIMD_FORCE_STATELESS_MEM
  // Verify input template arguments.
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();

  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr lsc_data_size FDS = detail::finalize_data_size<T, DS>();
  static_assert(FDS == lsc_data_size::u16 || FDS == lsc_data_size::u8 ||
                    FDS == lsc_data_size::u32 || FDS == lsc_data_size::u64,
                "Conversion data types are not supported");
  constexpr int SmallIntFactor32Bit =
      (FDS == lsc_data_size::u16) ? 2 : (FDS == lsc_data_size::u8 ? 4 : 1);
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (sizeof(T) == 8 ||
       (DS == lsc_data_size::default_size && NElts / SmallIntFactor32Bit > 64 &&
        (NElts * sizeof(T)) % 8 == 0));
  constexpr int SmallIntFactor64Bit =
      (FDS == lsc_data_size::u16)
          ? 4
          : (FDS == lsc_data_size::u8 ? 8
                                      : (FDS == lsc_data_size::u32 ? 2 : 1));
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  constexpr lsc_data_size ActualDS = Use64BitData
                                         ? __ESIMD_ENS::lsc_data_size::u64
                                         : __ESIMD_ENS::lsc_data_size::u32;

  detail::check_lsc_vector_size<FactoredNElts>();

  // Prepare template arguments for the call of intrinsic.
  using LoadElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;

  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr auto _VS = detail::to_lsc_vector_size<FactoredNElts>();
  constexpr auto _Transposed = detail::lsc_data_order::transpose;
  constexpr int N = 1;

  __ESIMD_NS::simd<uint32_t, N> Offsets = offset;
  auto SI = __ESIMD_NS::get_surface_index(acc);
  __ESIMD_NS::simd<LoadElemT, FactoredNElts> Result =
      __esimd_lsc_load_bti<LoadElemT, L1H, L3H, _AddressScale, _ImmOffset,
                           ActualDS, _VS, _Transposed, N>(pred.data(),
                                                          Offsets.data(), SI);
  return Result.template bit_cast_view<T>();
#endif // !__ESIMD_FORCE_STATELESS_MEM
}

/// A variation of lsc_block_load without predicate parameter to simplify use
/// of alignment parameter
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
/// See comments in the  \ref lsc_block_load API for description and parameter
/// constraints.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param flags is the alignment specifier type tag.
/// @return is a vector of type T and size NElts. The elements of the returned
/// vector for which the corresponding element in \p pred is 0 are undefined.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value &&
                                 __ESIMD_NS::is_simd_flag_type_v<FlagsT>,
                             __ESIMD_NS::simd<T, NElts>>
lsc_block_load(AccessorTy acc, uint32_t offset, FlagsT flags) {
  return lsc_block_load<T, NElts, DS, L1H, L3H>(
      acc, offset, __ESIMD_NS::simd_mask<1>(1), flags);
}

/// Accessor-based transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
/// See comments in the  \ref lsc_block_load API for description and parameter
/// constraints.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param pred is operation predicate. Operation is skipped for index 'i'
/// if pred[i] == 0 and the result element is taken from \p old_values[i].
/// Otherwise, the operation is performed.
/// @param old_values contains the values copied to the result when
/// the corresponding element from \p pred is zero.
/// @param flags is the alignment specifier type tag.
/// @return is a vector of type T and size NElts
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value &&
                                 __ESIMD_NS::is_simd_flag_type_v<FlagsT>,
                             __ESIMD_NS::simd<T, NElts>>
lsc_block_load(AccessorTy acc, uint32_t offset, __ESIMD_NS::simd_mask<1> pred,
               __ESIMD_NS::simd<T, NElts> old_values, FlagsT flags = FlagsT{}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_block_load<T, NElts, DS, L1H, L3H>(
      __ESIMD_DNS::accessorToPointer<T>(acc, offset), pred, old_values, flags);
#else  // !__ESIMD_FORCE_STATELESS_MEM
  // Verify input template arguments.
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr lsc_data_size FDS = detail::finalize_data_size<T, DS>();
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");
  static_assert(FDS == lsc_data_size::u16 || FDS == lsc_data_size::u8 ||
                    FDS == lsc_data_size::u32 || FDS == lsc_data_size::u64,
                "Conversion data types are not supported");
  constexpr int SmallIntFactor32Bit =
      (FDS == lsc_data_size::u16) ? 2 : (FDS == lsc_data_size::u8 ? 4 : 1);
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (sizeof(T) == 8 ||
       (DS == lsc_data_size::default_size && NElts / SmallIntFactor32Bit > 64 &&
        (NElts * sizeof(T)) % 8 == 0));
  constexpr int SmallIntFactor64Bit =
      (FDS == lsc_data_size::u16)
          ? 4
          : (FDS == lsc_data_size::u8 ? 8
                                      : (FDS == lsc_data_size::u32 ? 2 : 1));
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  constexpr lsc_data_size ActualDS = Use64BitData
                                         ? __ESIMD_ENS::lsc_data_size::u64
                                         : __ESIMD_ENS::lsc_data_size::u32;

  detail::check_lsc_vector_size<FactoredNElts>();

  // Prepare template arguments for the call of intrinsic.
  using LoadElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr auto _VS = detail::to_lsc_vector_size<FactoredNElts>();
  constexpr auto _Transposed = detail::lsc_data_order::transpose;
  constexpr int N = 1;

  __ESIMD_NS::simd<uint32_t, N> Offsets = offset;
  auto SI = __ESIMD_NS::get_surface_index(acc);
  __ESIMD_NS::simd<LoadElemT, FactoredNElts> OldVals =
      old_values.template bit_cast_view<LoadElemT>();
  __ESIMD_NS::simd<LoadElemT, FactoredNElts> Result =
      __esimd_lsc_load_merge_bti<LoadElemT, L1H, L3H, _AddressScale, _ImmOffset,
                                 ActualDS, _VS, _Transposed, N>(
          pred.data(), Offsets.data(), SI, OldVals.data());
  return Result.template bit_cast_view<T>();
#endif // !__ESIMD_FORCE_STATELESS_MEM
}

/// USM pointer prefetch gather.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Prefetches elements located at specified address.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename Toffset>
__ESIMD_API void lsc_prefetch(const T *p, __ESIMD_NS::simd<Toffset, N> offsets,
                              __ESIMD_NS::simd_mask<N> pred = 1) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::prefetch, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __esimd_lsc_prefetch_stateless<_MsgT, L1H, L3H, _AddressScale, _ImmOffset,
                                 _DS, _VS, _Transposed, N>(pred.data(),
                                                           addrs.data());
}

template <
    typename T, int NElts = 1, lsc_data_size DS = lsc_data_size::default_size,
    cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none, int N,
    typename Toffset, typename RegionTy = __ESIMD_NS::region1d_t<Toffset, N, 1>>
__ESIMD_API void lsc_prefetch(const T *p,
                              __ESIMD_NS::simd_view<Toffset, RegionTy> offsets,
                              __ESIMD_NS::simd_mask<N> pred = 1) {
  using Ty = typename __ESIMD_NS::simd_view<Toffset, RegionTy>::element_type;
  lsc_prefetch<T, NElts, DS, L1H, L3H, N>(p, __ESIMD_NS::simd<Ty, N>(offsets),
                                          pred);
}

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset>>
lsc_prefetch(const T *p, Toffset offset, __ESIMD_NS::simd_mask<N> pred = 1) {
  lsc_prefetch<T, NElts, DS, L1H, L3H, N>(
      p, __ESIMD_NS::simd<Toffset, N>(offset), pred);
}

/// USM pointer prefetch transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Prefetches elements located at specified address.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p is the base pointer.
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__ESIMD_API void lsc_prefetch(const T *p) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::prefetch, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();

  static_assert(
      _DS == lsc_data_size::u32 || _DS == lsc_data_size::u64,
      "Transposed prefetch is supported only for data size u32 or u64");
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd_mask<N> pred = 1;

  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  __esimd_lsc_prefetch_stateless<T, L1H, L3H, _AddressScale, _ImmOffset, _DS,
                                 _VS, _Transposed, N>(pred.data(),
                                                      addrs.data());
}

/// Accessor-based prefetch gather.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Prefetches elements located at surface.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value>
lsc_prefetch(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
             __ESIMD_NS::simd_mask<N> pred = 1) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_prefetch<T, NElts, DS, L1H, L3H>(
      __ESIMD_DNS::accessorToPointer<T>(acc), offsets, pred);
#else
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::prefetch, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  auto si = __ESIMD_NS::get_surface_index(acc);
  __esimd_lsc_prefetch_bti<_MsgT, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(), si);
#endif
}

/// Accessor-based transposed prefetch gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Prefetches elements located at surface.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value>
lsc_prefetch(AccessorTy acc, uint32_t offset) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  lsc_prefetch<T, NElts, DS, L1H, L3H>(
      __ESIMD_DNS::accessorToPointer<T>(acc, offset));
#else
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::prefetch, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  static_assert(
      _DS == lsc_data_size::u32 || _DS == lsc_data_size::u64,
      "Transposed prefetch is supported only for data size u32 or u64");
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd_mask<N> pred = 1;
  __ESIMD_NS::simd<uint32_t, N> offsets = offset;
  auto si = __ESIMD_NS::get_surface_index(acc);
  __esimd_lsc_prefetch_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(), si);
#endif
}

/// SLM scatter.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_store.slm
///
/// Scatters elements located to slm.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam N is the number of channels (platform dependent).
/// @param offsets is the zero-based offsets for SLM buffer in bytes.
/// @param vals is values to store.
/// @param pred is predicates.
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size, int N>
__ESIMD_API void lsc_slm_scatter(__ESIMD_NS::simd<uint32_t, N> offsets,
                                 __ESIMD_NS::simd<T, N * NElts> vals,
                                 __ESIMD_NS::simd_mask<N> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  using _CstT = typename detail::lsc_bitcast_type<T>::type;
  __ESIMD_NS::simd<_MsgT, N *NElts> Tmp = vals.template bit_cast_view<_CstT>();
  __esimd_lsc_store_slm<_MsgT, cache_hint::none, cache_hint::none,
                        _AddressScale, _ImmOffset, _DS, _VS, _Transposed, N>(
      pred.data(), offsets.data(), Tmp.data());
}

/// Transposed SLM scatter with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_store.slm
///
/// Scatters elements located to slm.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @param offset is the zero-based offset for SLM buffer in bytes.
/// @param vals is values to store.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API void lsc_slm_block_store(uint32_t offset,
                                     __ESIMD_NS::simd<T, NElts> vals) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  static_assert(_DS == lsc_data_size::u32 || _DS == lsc_data_size::u64,
                "Transposed store is supported only for data size u32 or u64");
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd_mask<N> pred = 1;
  __ESIMD_NS::simd<uint32_t, N> offsets = offset;
  __esimd_lsc_store_slm<T, cache_hint::none, cache_hint::none, _AddressScale,
                        _ImmOffset, _DS, _VS, _Transposed, N>(
      pred.data(), offsets.data(), vals.data());
}

/// USM pointer scatter.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_store.ugm
///
/// Scatters elements to specific address.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets in bytes.
/// @param vals is values to store.
/// @param pred is predicates.
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename Toffset>
__ESIMD_API void lsc_scatter(T *p, __ESIMD_NS::simd<Toffset, N> offsets,
                             __ESIMD_NS::simd<T, N * NElts> vals,
                             __ESIMD_NS::simd_mask<N> pred = 1) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  using _CstT = typename detail::lsc_bitcast_type<T>::type;
  __ESIMD_NS::simd<_MsgT, N *NElts> Tmp = vals.template bit_cast_view<_CstT>();
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __esimd_lsc_store_stateless<_MsgT, L1H, L3H, _AddressScale, _ImmOffset, _DS,
                              _VS, _Transposed, N>(pred.data(), addrs.data(),
                                                   Tmp.data());
}

template <
    typename T, int NElts = 1, lsc_data_size DS = lsc_data_size::default_size,
    cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none, int N,
    typename Toffset, typename RegionTy = __ESIMD_NS::region1d_t<Toffset, N, 1>>
__ESIMD_API void lsc_scatter(T *p,
                             __ESIMD_NS::simd_view<Toffset, RegionTy> offsets,
                             __ESIMD_NS::simd<T, N * NElts> vals,
                             __ESIMD_NS::simd_mask<N> pred = 1) {
  using Ty = typename __ESIMD_NS::simd_view<Toffset, RegionTy>::element_type;
  lsc_scatter<T, NElts, DS, L1H, L3H, N>(p, __ESIMD_NS::simd<Ty, N>(offsets),
                                         vals, pred);
}

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> && N == 1>
lsc_scatter(T *p, Toffset offset, __ESIMD_NS::simd<T, N * NElts> vals,
            __ESIMD_NS::simd_mask<N> pred = 1) {
  lsc_scatter<T, NElts, DS, L1H, L3H, N>(
      p, __ESIMD_NS::simd<Toffset, N>(offset), vals, pred);
}

/// Accessor-based scatter.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_store.ugm
///
/// Scatters elements to surface.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets in bytes.
/// @param vals is values to store.
/// @param pred is predicates.
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value>
lsc_scatter(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
            __ESIMD_NS::simd<T, N * NElts> vals,
            __ESIMD_NS::simd_mask<N> pred = 1) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  lsc_scatter<T, NElts, DS, L1H, L3H>(__ESIMD_DNS::accessorToPointer<T>(acc),
                                      offsets, vals, pred);
#else
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  using _CstT = typename detail::lsc_bitcast_type<T>::type;
  __ESIMD_NS::simd<_MsgT, N *NElts> Tmp = vals.template bit_cast_view<_CstT>();
  auto si = __ESIMD_NS::get_surface_index(acc);
  __esimd_lsc_store_bti<_MsgT, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                        _Transposed, N>(pred.data(), offsets.data(), Tmp.data(),
                                        si);
#endif
}

/// USM pointer transposed scatter with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_store.ugm
///
/// Scatters elements to specific address.
/// See comments in the  \ref lsc_block_load API for description and parameter
/// constraints.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p is the base pointer.
/// @param vals is values to store.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed. The default is '1' - perform
/// the operation.
/// @param flags is the alignment specifier type tag.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<__ESIMD_NS::is_simd_flag_type_v<FlagsT>>
lsc_block_store(T *p, __ESIMD_NS::simd<T, NElts> vals,
                __ESIMD_NS::simd_mask<1> pred = 1, FlagsT flags = FlagsT{}) {
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  // Prepare template arguments for the call of intrinsic.
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  static_assert(_DS == lsc_data_size::u16 || _DS == lsc_data_size::u8 ||
                    _DS == lsc_data_size::u32 || _DS == lsc_data_size::u64,
                "Conversion data types are not supported");
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd<uintptr_t, N> Addrs = reinterpret_cast<uintptr_t>(p);

  constexpr int SmallIntFactor32Bit =
      (_DS == lsc_data_size::u16) ? 2 : (_DS == lsc_data_size::u8 ? 4 : 1);
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed store");

  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (sizeof(T) == 8 ||
       (DS == lsc_data_size::default_size && NElts / SmallIntFactor32Bit > 64 &&
        (NElts * sizeof(T)) % 8 == 0));
  constexpr int SmallIntFactor64Bit =
      (_DS == lsc_data_size::u16)
          ? 4
          : (_DS == lsc_data_size::u8 ? 8
                                      : (_DS == lsc_data_size::u32 ? 2 : 1));
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  constexpr lsc_data_size ActualDS = Use64BitData
                                         ? __ESIMD_ENS::lsc_data_size::u64
                                         : __ESIMD_ENS::lsc_data_size::u32;

  detail::check_lsc_vector_size<FactoredNElts>();
  constexpr detail::lsc_vector_size _VS =
      detail::to_lsc_vector_size<FactoredNElts>();

  using StoreType = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;

  __esimd_lsc_store_stateless<StoreType, L1H, L3H, _AddressScale, _ImmOffset,
                              ActualDS, _VS, _Transposed, N>(
      pred.data(), Addrs.data(),
      sycl::bit_cast<__ESIMD_DNS::vector_type_t<StoreType, FactoredNElts>>(
          vals.data()));
}

/// A variation of lsc_block_store without predicate parameter to simplify
/// use of alignment parameter
///
/// Scatters elements to specific address.
/// See comments in the  \ref lsc_block_load API for description and parameter
/// constraints.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p is the base pointer.
/// @param vals is values to store.
/// @param flags is the alignment specifier type tag.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<__ESIMD_NS::is_simd_flag_type_v<FlagsT>>
lsc_block_store(T *p, __ESIMD_NS::simd<T, NElts> vals, FlagsT flags) {
  lsc_block_store<T, NElts, DS, L1H, L3H>(p, vals, __ESIMD_NS::simd_mask<1>(1),
                                          flags);
}

/// Accessor-based transposed scatter with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_store.ugm
///
/// Scatters elements to surface.
/// See comments in the  \ref lsc_block_load API for description and parameter
/// constraints.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param vals is values to store.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed. The default is '1' - perform
/// the operation.
/// @param flags is the alignment specifier type tag.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value &&
                             __ESIMD_NS::is_simd_flag_type_v<FlagsT>>
lsc_block_store(AccessorTy acc, uint32_t offset,
                __ESIMD_NS::simd<T, NElts> vals,
                __ESIMD_NS::simd_mask<1> pred = 1, FlagsT flags = FlagsT{}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  lsc_block_store<T, NElts, DS, L1H, L3H>(
      __ESIMD_DNS::accessorToPointer<T>(acc, offset), vals, pred, flags);
#else
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");
  // Prepare template arguments for the call of intrinsic.
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  static_assert(_DS == lsc_data_size::u16 || _DS == lsc_data_size::u8 ||
                    _DS == lsc_data_size::u32 || _DS == lsc_data_size::u64,
                "Conversion data types are not supported");
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;

  __ESIMD_NS::simd<uint32_t, N> Offsets = offset;
  auto si = __ESIMD_NS::get_surface_index(acc);

  constexpr int SmallIntFactor32Bit =
      (_DS == lsc_data_size::u16) ? 2 : (_DS == lsc_data_size::u8 ? 4 : 1);
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed store");

  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (sizeof(T) == 8 ||
       (DS == lsc_data_size::default_size && NElts / SmallIntFactor32Bit > 64 &&
        (NElts * sizeof(T)) % 8 == 0));
  constexpr int SmallIntFactor64Bit =
      (_DS == lsc_data_size::u16)
          ? 4
          : (_DS == lsc_data_size::u8 ? 8
                                      : (_DS == lsc_data_size::u32 ? 2 : 1));
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  constexpr lsc_data_size ActualDS = Use64BitData
                                         ? __ESIMD_ENS::lsc_data_size::u64
                                         : __ESIMD_ENS::lsc_data_size::u32;

  detail::check_lsc_vector_size<FactoredNElts>();
  constexpr detail::lsc_vector_size _VS =
      detail::to_lsc_vector_size<FactoredNElts>();

  using StoreType = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;

  __esimd_lsc_store_bti<StoreType, L1H, L3H, _AddressScale, _ImmOffset,
                        ActualDS, _VS, _Transposed, N>(
      pred.data(), Offsets.data(),
      sycl::bit_cast<__ESIMD_DNS::vector_type_t<StoreType, FactoredNElts>>(
          vals.data()),
      si);
#endif
}

/// A variation of lsc_block_store without predicate parameter to simplify
/// use of alignment parameter
///
/// Scatters elements to surface.
/// See comments in the  \ref lsc_block_load API for description and parameter
/// constraints.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param vals is values to store.
/// @param flags is the alignment specifier type tag.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value &&
                             __ESIMD_NS::is_simd_flag_type_v<FlagsT>>
lsc_block_store(AccessorTy acc, uint32_t offset,
                __ESIMD_NS::simd<T, NElts> vals, FlagsT flags) {
  lsc_block_store<T, NElts, DS, L1H, L3H>(acc, offset, vals,
                                          __ESIMD_NS::simd_mask<1>(1), flags);
}

namespace detail {
// Compile-time checks for lsc_load_2d/prefetch_2d/store_2d restrictions.
template <typename T, int BlockWidth, int BlockHeight, int NBlocks,
          bool Transposed, bool Transformed, bool IsStore = false>
constexpr void check_lsc_block_2d_restrictions() {
  constexpr int GRFByteSize = BlockWidth * BlockHeight * NBlocks * sizeof(T);
  static_assert(BlockWidth > 0, "Block width must be positive");
  static_assert(BlockHeight > 0, "Block height must be positive");
  // Restrictions based on documentation.
  static_assert(!IsStore || GRFByteSize <= 512,
                "2D store supports 512 bytes max");
  static_assert(IsStore || GRFByteSize <= 2048,
                "2D load supports 2048 bytes max");
  static_assert(!Transposed || !Transformed,
                "Transposed and transformed is not supported");
  static_assert((sizeof(T) * BlockWidth) % 4 == 0,
                "Block width must be aligned by DW");
  if constexpr (Transposed) {
    static_assert(NBlocks == 1, "Transposed expected to be 1 block only");
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "Transposed load is supported only for data size u32 or u64");
    static_assert(sizeof(T) == 8 ? BlockHeight == 8
                                 : BlockHeight >= 1 && BlockHeight <= 32,
                  "Unsupported block height");
    static_assert(sizeof(T) == 8 ? __ESIMD_DNS::isPowerOf2(BlockWidth, 4)
                                 : BlockWidth >= 1 && BlockWidth <= 8,
                  "Unsupported block width");
  } else if constexpr (Transformed) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2,
                  "VNNI transform is supported only for data size u8 or u16");
    static_assert(__ESIMD_DNS::isPowerOf2(NBlocks, 4),
                  "Unsupported number of blocks");
    static_assert(BlockHeight * sizeof(T) >= 4 && BlockHeight <= 32,
                  "Unsupported block height");
    static_assert(BlockWidth * sizeof(T) >= 4 && BlockWidth <= 16 &&
                      BlockWidth * NBlocks * sizeof(T) <= 64,
                  "Unsupported block width");
  } else {
    static_assert(
        __ESIMD_DNS::isPowerOf2(NBlocks, sizeof(T) == 1 ? 4 : 8 / sizeof(T)),
        "Unsupported number of blocks");
    if constexpr (IsStore)
      static_assert(BlockHeight <= 8, "Unsupported block height for store");
    else
      static_assert(BlockHeight <= 32, "Unsupported block height for load");
    static_assert(BlockWidth * sizeof(T) >= 4 &&
                      BlockWidth * NBlocks * sizeof(T) <= 64,
                  "Unsupported block width");
  }
}

} // namespace detail

/// 2D USM pointer block load.
/// Supported platforms: PVC
/// VISA instruction: lsc_load_block2d.ugm
///
/// Collects elements located at specified address and returns them
/// as a single \ref simd object.
///
/// @tparam T is element type.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam NBlocks is the number of blocks.
/// @tparam Transposed is the transposed version or not.
/// @tparam Transformed is apply VNNI transform or not.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the data size
/// @param Ptr is the surface base address for this operation.
/// @param SurfaceWidth is the surface width minus 1 in bytes
/// @param SurfaceHeight is the surface height minus 1 in rows
/// @param SurfacePitch is the surface pitch minus 1 in bytes
/// @param X is zero based X-coordinate of the left upper rectangle corner in
/// number of elements.
/// @param Y is zero based Y-coordinate of the left upper rectangle corner in
/// rows.
/// @return is a vector of type T and size N, where N is
///  BlockWidth * BlockHeight * NBlocks, if transformed;
///  otherwise,
///  N = roundUpNextMultiple(BlockHeight, 4 / sizeof(T)) *
///   getNextPowerOf2(BlockWidth) * NBlocks
///
template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          bool Transposed = false, bool Transformed = false,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N = detail::get_lsc_block_2d_data_size<
              T, NBlocks, BlockHeight, BlockWidth, Transposed, Transformed>()>
__ESIMD_API __ESIMD_NS::simd<T, N>
lsc_load_2d(const T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
            unsigned SurfacePitch, int X, int Y) {
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  detail::check_lsc_block_2d_restrictions<T, BlockWidth, BlockHeight, NBlocks,
                                          Transposed, Transformed>();
  constexpr int ElemsPerDword = 4 / sizeof(T);
  constexpr int GRFRowSize = Transposed    ? BlockHeight
                             : Transformed ? BlockWidth * ElemsPerDword
                                           : BlockWidth;
  constexpr int GRFRowPitch = __ESIMD_DNS::getNextPowerOf2<GRFRowSize>();
  constexpr int GRFColSize =
      Transposed
          ? BlockWidth
          : (Transformed ? (BlockHeight + ElemsPerDword - 1) / ElemsPerDword
                         : BlockHeight);
  constexpr int GRFBlockSize = GRFRowPitch * GRFColSize;
  constexpr int GRFBlockPitch =
      detail::roundUpNextMultiple<64 / sizeof(T), GRFBlockSize>();
  constexpr int ActualN = NBlocks * GRFBlockPitch;

  constexpr int DstBlockElements = GRFColSize * GRFRowSize;
  constexpr int DstElements = DstBlockElements * NBlocks;

  static_assert(N == ActualN || N == DstElements, "Incorrect element count");

  constexpr lsc_data_size DS =
      detail::finalize_data_size<T, lsc_data_size::default_size>();
  __ESIMD_NS::simd_mask<ActualN> pred = 1;
  uintptr_t surf_addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr detail::lsc_data_order _Transposed =
      Transposed ? detail::lsc_data_order::transpose
                 : detail::lsc_data_order::nontranspose;
  __ESIMD_NS::simd<T, ActualN> Raw =
      __esimd_lsc_load2d_stateless<T, L1H, L3H, DS, _Transposed, NBlocks,
                                   BlockWidth, BlockHeight, Transformed,
                                   ActualN>(pred.data(), surf_addr,
                                            SurfaceWidth, SurfaceHeight,
                                            SurfacePitch, X, Y);

  if constexpr (ActualN == N) {
    return Raw;
  } else {
    // HW restrictions force data which is read to contain padding filled with
    // garbage for 2d lsc loads. This code eliminates such padding.

    __ESIMD_NS::simd<T, DstElements> Dst;

    for (auto i = 0; i < NBlocks; i++) {
      auto DstBlock =
          Dst.template select<DstBlockElements, 1>(i * DstBlockElements);

      auto RawBlock = Raw.template select<GRFBlockSize, 1>(i * GRFBlockPitch);
      DstBlock = RawBlock.template bit_cast_view<T, GRFColSize, GRFRowPitch>()
                     .template select<GRFColSize, 1, GRFRowSize, 1>(0, 0)
                     .template bit_cast_view<T>();
    }

    return Dst;
  }
}

template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          bool Transposed = false, bool Transformed = false,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N = detail::get_lsc_block_2d_data_size<
              T, NBlocks, BlockHeight, BlockWidth, Transposed, Transformed>()>
__SYCL_DEPRECATED("use lsc_load_2d()")
__ESIMD_API __ESIMD_NS::simd<T, N> lsc_load2d(const T *Ptr,
                                              unsigned SurfaceWidth,
                                              unsigned SurfaceHeight,
                                              unsigned SurfacePitch, int X,
                                              int Y) {
  return lsc_load_2d<T, BlockWidth, BlockHeight, NBlocks, Transposed,
                     Transformed, L1H, L3H>(Ptr, SurfaceWidth, SurfaceHeight,
                                            SurfacePitch, X, Y);
}

/// 2D USM pointer block prefetch.
/// Supported platforms: PVC
/// VISA instruction: lsc_load_block2d.ugm
///
/// Prefetches elements located at specified address.
///
/// @tparam T is element type.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam NBlocks is the number of blocks.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the data size
/// @param Ptr is the surface base address for this operation.
/// @param SurfaceWidth is the surface width minus 1 in bytes
/// @param SurfaceHeight is the surface height minus 1 in rows
/// @param SurfacePitch is the surface pitch minus 1 in bytes
/// @param X is zero based X-coordinate of the left upper rectangle corner in
/// number of elements.
/// @param Y is zero based Y-coordinate of the left upper rectangle corner in
/// rows.
///
template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N = detail::get_lsc_block_2d_data_size<
              T, NBlocks, BlockHeight, BlockWidth, false, false>()>
__ESIMD_API void lsc_prefetch_2d(const T *Ptr, unsigned SurfaceWidth,
                                 unsigned SurfaceHeight, unsigned SurfacePitch,
                                 int X, int Y) {
  detail::check_lsc_cache_hint<detail::lsc_action::prefetch, L1H, L3H>();
  detail::check_lsc_block_2d_restrictions<T, BlockWidth, BlockHeight, NBlocks,
                                          false, false>();
  constexpr lsc_data_size DS =
      detail::finalize_data_size<T, lsc_data_size::default_size>();
  __ESIMD_NS::simd_mask<N> pred = 1;
  uintptr_t surf_addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  __esimd_lsc_prefetch2d_stateless<T, L1H, L3H, DS, _Transposed, NBlocks,
                                   BlockWidth, BlockHeight, false, N>(
      pred.data(), surf_addr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y);
}

template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N = detail::get_lsc_block_2d_data_size<
              T, NBlocks, BlockHeight, BlockWidth, false, false>()>
__SYCL_DEPRECATED("use lsc_prefetch_2d()")
__ESIMD_API void lsc_prefetch2d(const T *Ptr, unsigned SurfaceWidth,
                                unsigned SurfaceHeight, unsigned SurfacePitch,
                                int X, int Y) {
  lsc_prefetch_2d<T, BlockWidth, BlockHeight, NBlocks, L1H, L3H>(
      Ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y);
}
/// 2D USM pointer block store.
/// Supported platforms: PVC
/// VISA instruction: lsc_store_block2d.ugm
///
/// Stores elements at specified address.
///
/// @tparam T is element type.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the data size
/// @param Ptr is the surface base address for this operation.
/// @param SurfaceWidth is the surface width minus 1 in bytes
/// @param SurfaceHeight is the surface height minus 1 in rows
/// @param SurfacePitch is the surface pitch minus 1 in bytes
/// @param X is zero based X-coordinate of the left upper rectangle corner in
/// number of elements.
/// @param Y is zero based Y-coordinate of the left upper rectangle corner in
/// rows.
/// @param Vals is a vector to store of type T and size N, where
///  N = roundUpNextMultiple(BlockHeight, 4 / sizeof(T)) *
///   getNextPowerOf2(BlockWidth) * NBlocks
///
template <typename T, int BlockWidth, int BlockHeight = 1,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N = detail::get_lsc_block_2d_data_size<
              T, 1u, BlockHeight, BlockWidth, false, false>()>
__ESIMD_API void lsc_store_2d(T *Ptr, unsigned SurfaceWidth,
                              unsigned SurfaceHeight, unsigned SurfacePitch,
                              int X, int Y, __ESIMD_NS::simd<T, N> Vals) {
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();
  detail::check_lsc_block_2d_restrictions<T, BlockWidth, BlockHeight, 1, false,
                                          false, true /*IsStore*/>();
  constexpr lsc_data_size DS =
      detail::finalize_data_size<T, lsc_data_size::default_size>();
  uintptr_t surf_addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;

  constexpr int Pitch = __ESIMD_DNS::getNextPowerOf2<BlockWidth>();
  __ESIMD_NS::simd<T, BlockHeight * Pitch> Raw;

  if constexpr (BlockHeight * Pitch == N) {
    Raw = Vals;
  } else {
    auto Data2D = Vals.template bit_cast_view<T, BlockHeight, BlockWidth>();
    auto Raw2D = Raw.template bit_cast_view<T, BlockHeight, Pitch>();
    Raw2D.template select<BlockHeight, 1, BlockWidth, 1>(0, 0) = Data2D;
  }

  __ESIMD_NS::simd_mask<BlockHeight * Pitch> pred = 1;
  __esimd_lsc_store2d_stateless<T, L1H, L3H, DS, _Transposed, 1u, BlockWidth,
                                BlockHeight, false, BlockHeight * Pitch>(
      pred.data(), surf_addr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y,
      Raw.data());
}

template <typename T, int BlockWidth, int BlockHeight = 1,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N = detail::get_lsc_block_2d_data_size<
              T, 1u, BlockHeight, BlockWidth, false, false>()>
__SYCL_DEPRECATED("use lsc_store_2d()")
__ESIMD_API void lsc_store2d(T *Ptr, unsigned SurfaceWidth,
                             unsigned SurfaceHeight, unsigned SurfacePitch,
                             int X, int Y, __ESIMD_NS::simd<T, N> Vals) {
  lsc_store_2d<T, BlockWidth, BlockHeight, L1H, L3H>(
      Ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y, Vals);
}

/// <summary>
///  Container class to hold parameters for \c load2d/store2d \c functions
/// </summary>
/// @tparam T Type of data to load/store
/// @tparam BlockWidth the block width in number of elements
/// @tparam BlockHeight block height in number of elements
/// @tparam NBlocks Number of blocks
template <typename T, int BlockWidth, int BlockHeight, int NBlocks>
class config_2d_mem_access {
public:
  /// <summary>
  /// Default constructor
  /// </summary>
  config_2d_mem_access() : payload_data(0) {
    payload_data.template select<1, 1>(7) =
        ((NBlocks - 1) << 16) | ((BlockHeight - 1) << 8) | (BlockWidth - 1);
  }

  /// <summary>
  /// Copy constructor
  /// </summary>
  config_2d_mem_access(const config_2d_mem_access &other)
      : payload_data(other.payload) {}

  /// <summary>
  /// Constructor
  /// </summary>
  /// <param name="Ptr">surface base address</param>
  /// <param name="SurfaceWidth">surface width minus 1 in bytes</param>
  /// <param name="SurfaceHeight">surface height minus 1 in rows</param>
  /// <param name="SurfacePitch">surface pitch minus 1 in bytes</param>
  /// <param name="X">zero based X-coordinate of the left upper rectangle corner
  /// in number of elements</param>
  /// <param name="Y">zero based Y-coordinate of the left upper rectangle corner
  /// in rows</param>
  config_2d_mem_access(const T *Ptr, uint32_t SurfaceWidth,
                       uint32_t SurfaceHeight, uint32_t SurfacePitch, int32_t X,
                       int32_t Y)
      : config_2d_mem_access() {
    payload_data.template bit_cast_view<uint64_t>().template select<1, 1>(0) =
        (uint64_t)Ptr;
    payload_data.template select<1, 1>(2) = SurfaceWidth;
    payload_data.template select<1, 1>(3) = SurfaceHeight;
    payload_data.template select<1, 1>(4) = SurfacePitch;
    payload_data.template select<1, 1>(5) = X;
    payload_data.template select<1, 1>(6) = Y;
  }

  /// <summary>
  /// Get a surface base address
  /// </summary>
  /// <returns>surface base address</returns>
  T *get_data_pointer() const {
    return (T *)(const_cast<config_2d_mem_access>(this)
                     ->payload_data.template bit_cast_view<uint64_t>()
                     .template select<1, 1>(0));
  }

  /// <summary>
  /// Get surface width
  /// </summary>
  /// <returns>Surface Width</returns>
  uint32_t get_surface_width() const {
    return const_cast<config_2d_mem_access>(this)
        ->payload_data.template select<1, 1>(2);
  }

  /// <summary>
  /// Get surface height
  /// </summary>
  /// <returns>Surface Height</returns>
  uint32_t get_surface_height() const {
    return const_cast<config_2d_mem_access>(this)
        ->payload_data.template select<1, 1>(3);
  }

  /// <summary>
  /// Get surface pitch
  /// </summary>
  /// <returns>Surface Pitch</returns>
  uint32_t get_surface_pitch() const {
    return const_cast<config_2d_mem_access>(this)
        ->payload_data.template select<1, 1>(4);
  }

  /// <summary>
  /// Get top left corner X coordinate of the block
  /// </summary>
  /// <returns>Top left corner X coordinate of the block</returns>
  int32_t get_x() const {
    return const_cast<config_2d_mem_access>(this)
        ->payload_data.template select<1, 1>(5);
  }

  /// <summary>
  /// Get top left corner Y coordinate of the block
  /// </summary>
  /// <returns>Top left corner Y coordinate of the block</returns>
  int32_t get_y() const {
    return const_cast<config_2d_mem_access>(this)
        ->payload_data.template select<1, 1>(6);
  }

  /// <summary>
  /// Get width of the block
  /// </summary>
  /// <returns>Width of the block</returns>
  constexpr int32_t get_width() const { return BlockWidth; }

  /// <summary>
  /// Get height of the block
  /// </summary>
  /// <returns>Height of the block</returns>
  constexpr int32_t get_height() const { return BlockHeight; }

  /// <summary>
  /// Get number of blocks
  /// </summary>
  /// <returns>Height of the block</returns>
  constexpr int32_t get_number_of_blocks() const { return NBlocks; }

  /// <summary>
  /// Sets surface base address
  /// </summary>
  /// <param name="Ptr">surface base address</param>
  /// <returns>Reference to the modified object</returns>
  config_2d_mem_access &set_data_pointer(T *Ptr) {
    payload_data.template bit_cast_view<uint64_t>().template select<1, 1>(0) =
        (uint64_t)Ptr;
    return *this;
  }

  /// <summary>
  /// Sets surface width
  /// </summary>
  /// <param name="SurfaceWidth">Surface Width</param>
  /// <returns>Reference to the modified object</returns>
  config_2d_mem_access &set_surface_width(uint32_t SurfaceWidth) {
    payload_data.template select<1, 1>(2) = SurfaceWidth;
    return *this;
  }

  /// <summary>
  /// Sets surface height
  /// </summary>
  /// <param name="SurfaceHeight">Surface Height</param>
  /// <returns>Reference to the modified object</returns>
  config_2d_mem_access &set_surface_height(uint32_t SurfaceHeight) {
    payload_data.template select<1, 1>(3) = SurfaceHeight;
    return *this;
  }

  /// <summary>
  /// Sets surface pitch
  /// </summary>
  /// <param name="SurfacePitch">Surface Pitch</param>
  /// <returns>Reference to the modified object</returns>
  config_2d_mem_access &set_surface_pitch(uint32_t SurfacePitch) {
    payload_data.template select<1, 1>(4) = SurfacePitch;
    return *this;
  }

  /// <summary>
  /// Sets top left corner X coordinate of the block
  /// </summary>
  /// <param name="X">Top left corner X coordinate of the block</param>
  /// <returns>Reference to the modified object</returns>
  config_2d_mem_access &set_x(int32_t X) {
    payload_data.template select<1, 1>(5) = X;
    return *this;
  }

  /// <summary>
  /// Sets top left corner Y coordinate of the block
  /// </summary>
  /// <param name="Y">Top left corner Y coordinate of the block</param>
  /// <returns>Reference to the modified object</returns>
  config_2d_mem_access &set_y(int32_t Y) {
    payload_data.template select<1, 1>(6) = Y;
    return *this;
  }

private:
  __ESIMD_NS::simd<uint32_t, 16> get_raw_data() { return payload_data; }
  __ESIMD_NS::simd<uint32_t, 16> payload_data;

  template <typename T1, int BlockWidth1, int BlockHeight1, int NBlocks1,
            bool Transposed1, bool Transformed1, cache_hint L1H, cache_hint L3H,
            int N>
  friend ESIMD_INLINE SYCL_ESIMD_FUNCTION __ESIMD_NS::simd<T1, N> lsc_load_2d(
      config_2d_mem_access<T1, BlockWidth1, BlockHeight1, NBlocks1> &payload);

  template <typename T1, int BlockWidth1, int BlockHeight1, int NBlocks1,
            cache_hint L1H, cache_hint L3H, int N>
  friend ESIMD_INLINE SYCL_ESIMD_FUNCTION void lsc_store_2d(
      config_2d_mem_access<T1, BlockWidth1, BlockHeight1, NBlocks1> &payload,
      __ESIMD_NS::simd<T1, N> Data);

  template <typename T1, int BlockWidth1, int BlockHeight1, int NBlocks1,
            bool Transposed1, bool Transformed1, cache_hint L1H, cache_hint L3H,
            int N>
  friend ESIMD_INLINE SYCL_ESIMD_FUNCTION void lsc_prefetch_2d(
      config_2d_mem_access<T1, BlockWidth1, BlockHeight1, NBlocks1> &payload);
};

/// A variation of \c 2D stateless block load \c with parameters passed as
/// \c config_2d_mem_access \c object
/// Note: Compatibility with future hardware versions is not guaranteed.
/// Note: No software mitigation for hardware bugs is possible for this
/// function.
/// @tparam T is the element data type
/// @tparam BlockWidth the block width in number of elements
/// @tparam BlockHeight block height in number of elements
/// @tparam NBlocks Number of blocks
/// @tparam Transposed is the transposed version or not.
/// @tparam Transformed is apply VNNI transform or not.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the data size
/// @param payload is \c config_2d_mem_access \c object holding all the data
/// @return is a vector of type T and size N, where N is
///  getNextPowerOf2(Height) * Width * NBlocks, if transposed
///  getNextPowerOf2(Width) * Height * NBlocks, otherwise
///
template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          bool Transposed = false, bool Transformed = false,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N = detail::get_lsc_block_2d_data_size<
              T, NBlocks, BlockHeight, BlockWidth, Transposed, Transformed>()>
ESIMD_INLINE SYCL_ESIMD_FUNCTION __ESIMD_NS::simd<T, N> lsc_load_2d(
    config_2d_mem_access<T, BlockWidth, BlockHeight, NBlocks> &payload) {
  detail::check_lsc_block_2d_restrictions<T, BlockWidth, BlockHeight, NBlocks,
                                          Transposed, Transformed, false>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr int ElemsPerDword = 4 / sizeof(T);
  constexpr int GRFRowSize = Transposed    ? BlockHeight
                             : Transformed ? BlockWidth * ElemsPerDword
                                           : BlockWidth;
  constexpr int GRFRowPitch = __ESIMD_DNS::getNextPowerOf2<GRFRowSize>();
  constexpr int GRFColSize =
      Transposed
          ? BlockWidth
          : (Transformed ? (BlockHeight + ElemsPerDword - 1) / ElemsPerDword
                         : BlockHeight);
  constexpr int GRFBlockSize = GRFRowPitch * GRFColSize;
  constexpr int GRFBlockPitch =
      detail::roundUpNextMultiple<64 / sizeof(T), GRFBlockSize>();
  constexpr int ActualN = NBlocks * GRFBlockPitch;

  constexpr int DstBlockElements = GRFColSize * GRFRowSize;
  constexpr int DstElements = DstBlockElements * NBlocks;

  static_assert(N == ActualN || N == DstElements, "Incorrect element count");

  constexpr uint32_t cache_mask = detail::get_lsc_load_cache_mask<L1H, L3H>()
                                  << 17;
  constexpr uint32_t base_desc = 0x2800403;
  constexpr uint32_t transformMask = Transformed ? 1 << 7 : 0;
  constexpr uint32_t transposeMask = Transposed ? 1 << 15 : 0;
  __ESIMD_NS::simd<T, N> oldDst;
  constexpr uint32_t exDesc = 0x0;
  constexpr uint32_t desc =
      base_desc | cache_mask | transformMask | transposeMask;
  constexpr uint8_t execSize = 0x0;
  constexpr uint8_t sfid = 0xF;
  constexpr uint8_t numSrc0 = 0x1;
  constexpr uint8_t numDst = (N * sizeof(T)) / 64;
  __ESIMD_NS::simd<T, ActualN> Raw =
      raw_send(oldDst, payload.get_raw_data(), exDesc, desc, execSize, sfid,
               numSrc0, numDst);

  if constexpr (ActualN == N) {
    return Raw;
  } else {
    // HW restrictions force data which is read to contain padding filled with
    // garbage for 2d lsc loads. This code eliminates such padding.

    __ESIMD_NS::simd<T, DstElements> Dst;

    for (auto i = 0; i < NBlocks; i++) {
      auto DstBlock =
          Dst.template select<DstBlockElements, 1>(i * DstBlockElements);

      auto RawBlock = Raw.template select<GRFBlockSize, 1>(i * GRFBlockPitch);
      DstBlock = RawBlock.template bit_cast_view<T, GRFColSize, GRFRowPitch>()
                     .template select<GRFColSize, 1, GRFRowSize, 1>(0, 0)
                     .template bit_cast_view<T>();
    }

    return Dst;
  }
}

/// A variation of \c 2D stateless block prefetch \c with parameters passed as
/// \c config_2d_mem_access \c object
/// Note: Compatibility with future hardware versions is not guaranteed.
/// Note: No software mitigation for hardware bugs is possible for this
/// function.
/// @tparam T is the element data type
/// @tparam BlockWidth the block width in number of elements
/// @tparam BlockHeight block height in number of elements
/// @tparam NBlocks Number of blocks
/// @tparam Transposed is the transposed version or not.
/// @tparam Transformed is apply VNNI transform or not.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the data size
/// @param payload is \c config_2d_mem_access \c object holding all the data
///
template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          bool Transposed = false, bool Transformed = false,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N = detail::get_lsc_block_2d_data_size<
              T, NBlocks, BlockHeight, BlockWidth, Transposed, Transformed>()>
ESIMD_INLINE SYCL_ESIMD_FUNCTION void lsc_prefetch_2d(
    config_2d_mem_access<T, BlockWidth, BlockHeight, NBlocks> &payload) {
  detail::check_lsc_cache_hint<detail::lsc_action::prefetch, L1H, L3H>();
  detail::check_lsc_block_2d_restrictions<T, BlockWidth, BlockHeight, NBlocks,
                                          Transposed, Transformed, false>();
  static_assert(!Transposed || !Transformed,
                "Transposed and transformed is not supported");
  constexpr uint32_t cache_mask = detail::get_lsc_load_cache_mask<L1H, L3H>()
                                  << 17;
  constexpr uint32_t base_desc = 0x2000403;
  constexpr uint32_t transformMask = Transformed ? 1 << 7 : 0;
  constexpr uint32_t transposeMask = Transposed ? 1 << 15 : 0;
  constexpr uint32_t exDesc = 0x0;
  constexpr uint32_t desc =
      base_desc | cache_mask | transformMask | transposeMask;
  constexpr uint8_t execSize = 0x0;
  constexpr uint8_t sfid = 0xF;
  constexpr uint8_t numDst = (N * sizeof(T)) / 64;
  raw_send(payload.get_raw_data(), exDesc, desc, execSize, sfid, numDst);
}

/// A variation of \c 2D stateless block store \c with parameters passed as
/// \c config_2d_mem_access \c object
/// Note: Compatibility with future hardware versions is not guaranteed.
/// Note: No software mitigation for hardware bugs is possible for this
/// function.
/// @tparam T is the element data type
/// @tparam BlockWidth the block width in number of elements
/// @tparam BlockHeight block height in number of elements
/// @tparam NBlocks Number of blocks
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the data size
/// @param payload is \c config_2d_mem_access \c object holding all the data
/// @param Data is the data to be stored.
///
template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N = detail::get_lsc_block_2d_data_size<
              T, NBlocks, BlockHeight, BlockWidth, false, false>()>
ESIMD_INLINE SYCL_ESIMD_FUNCTION void
lsc_store_2d(config_2d_mem_access<T, BlockWidth, BlockHeight, NBlocks> &payload,
             __ESIMD_NS::simd<T, N> Data) {
  detail::check_lsc_block_2d_restrictions<T, BlockWidth, BlockHeight, NBlocks,
                                          false, false, true>();
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();

  constexpr uint32_t cache_mask = detail::get_lsc_store_cache_mask<L1H, L3H>()
                                  << 17;
  constexpr uint32_t base_desc = 0x2000407;

  constexpr uint32_t exDesc = 0x0;
  constexpr uint32_t desc = base_desc | cache_mask;
  constexpr uint8_t execSize = 0x0;
  constexpr uint8_t sfid = 0xF;
  constexpr uint8_t numSrc0 = 0x1;
  constexpr uint8_t numSrc1 = (N * sizeof(T)) / 64;

  raw_sends(payload.get_raw_data(), Data, exDesc, desc, execSize, sfid, numSrc0,
            numSrc1);
}

/// SLM atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets is the zero-based offsets.
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, N>
lsc_slm_atomic_update(__ESIMD_NS::simd<uint32_t, N> offsets,
                      __ESIMD_NS::simd_mask<N> pred) {
  __ESIMD_EDNS::check_lsc_vector_size<1>();
  __ESIMD_EDNS::check_lsc_data_size<T, DS>();
  constexpr __ESIMD_NS::native::lsc::atomic_op _Op =
      __ESIMD_DNS::to_lsc_atomic_op<Op>();
  __ESIMD_EDNS::check_lsc_atomic<_Op, T, N, 0>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_slm_0<_MsgT, _Op, cache_hint::none, cache_hint::none,
                                _AddressScale, _ImmOffset, _DS, _VS,
                                _Transposed, N>(pred.data(), offsets.data());
  return detail::lsc_format_ret<T>(Tmp);
}

/// SLM atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, N>
lsc_slm_atomic_update(__ESIMD_NS::simd<uint32_t, N> offsets,
                      __ESIMD_NS::simd<T, N> src0,
                      __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  constexpr __ESIMD_NS::native::lsc::atomic_op _Op =
      __ESIMD_DNS::to_lsc_atomic_op<Op>();
  __ESIMD_EDNS::check_lsc_atomic<_Op, T, N, 1>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_slm_1<_MsgT, _Op, cache_hint::none, cache_hint::none,
                                _AddressScale, _ImmOffset, _DS, _VS,
                                _Transposed, N>(pred.data(), offsets.data(),
                                                src0.data());
  return detail::lsc_format_ret<T>(Tmp);
}

/// SLM atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, N>
lsc_slm_atomic_update(__ESIMD_NS::simd<uint32_t, N> offsets,
                      __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                      __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  constexpr __ESIMD_NS::native::lsc::atomic_op _Op =
      __ESIMD_DNS::to_lsc_atomic_op<Op>();
  __ESIMD_EDNS::check_lsc_atomic<_Op, T, N, 2>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_slm_2<_MsgT, _Op, cache_hint::none, cache_hint::none,
                                _AddressScale, _ImmOffset, _DS, _VS,
                                _Transposed, N>(pred.data(), offsets.data(),
                                                src0.data(), src1.data());
  return detail::lsc_format_ret<T>(Tmp);
}

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param pred is predicates.
///
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<__ESIMD_DNS::to_lsc_atomic_op<Op>()>() == 0,
    __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd<Toffset, N> offsets,
                  __ESIMD_NS::simd_mask<N> pred) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  constexpr __ESIMD_NS::native::lsc::atomic_op _Op =
      __ESIMD_DNS::to_lsc_atomic_op<Op>();
  __ESIMD_EDNS::check_lsc_atomic<_Op, T, N, 0>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_stateless_0<_MsgT, _Op, L1H, L3H, _AddressScale,
                                      _ImmOffset, _DS, _VS, _Transposed, N>(
          pred.data(), addrs.data());
  return detail::lsc_format_ret<T>(Tmp);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset,
          typename RegionTy = __ESIMD_NS::region1d_t<Toffset, N, 1>>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<__ESIMD_DNS::to_lsc_atomic_op<Op>()>() == 0,
    __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd_view<Toffset, RegionTy> offsets,
                  __ESIMD_NS::simd_mask<N> pred = 1) {
  using Ty = typename __ESIMD_NS::simd_view<Toffset, RegionTy>::element_type;
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(
      p, __ESIMD_NS::simd<Ty, N>(offsets), pred);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> &&
        __ESIMD_DNS::get_num_args<__ESIMD_DNS::to_lsc_atomic_op<Op>()>() == 0,
    __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, Toffset offset, __ESIMD_NS::simd_mask<N> pred = 1) {
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(
      p, __ESIMD_NS::simd<Toffset, N>(offset), pred);
}

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<__ESIMD_DNS::to_lsc_atomic_op<Op>()>() == 1,
    __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd<Toffset, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd_mask<N> pred) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  constexpr __ESIMD_NS::native::lsc::atomic_op _Op =
      __ESIMD_DNS::to_lsc_atomic_op<Op>();
  __ESIMD_EDNS::check_lsc_atomic<_Op, T, N, 1>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_stateless_1<_MsgT, _Op, L1H, L3H, _AddressScale,
                                      _ImmOffset, _DS, _VS, _Transposed, N>(
          pred.data(), addrs.data(),
          src0.template bit_cast_view<_MsgT>().data());
  return detail::lsc_format_ret<T>(Tmp);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset,
          typename RegionTy = __ESIMD_NS::region1d_t<Toffset, N, 1>>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<__ESIMD_DNS::to_lsc_atomic_op<Op>()>() == 1,
    __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd_view<Toffset, RegionTy> offsets,
                  __ESIMD_NS::simd<T, N> src0,
                  __ESIMD_NS::simd_mask<N> pred = 1) {
  using Ty = typename __ESIMD_NS::simd_view<Toffset, RegionTy>::element_type;
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(
      p, __ESIMD_NS::simd<Ty, N>(offsets), src0, pred);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> &&
        __ESIMD_DNS::get_num_args<__ESIMD_DNS::to_lsc_atomic_op<Op>()>() == 1 &&
        ((Op != __ESIMD_NS::atomic_op::store &&
          Op != __ESIMD_NS::atomic_op::xchg) ||
         N == 1),
    __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, Toffset offset, __ESIMD_NS::simd<T, N> src0,
                  __ESIMD_NS::simd_mask<N> pred = 1) {
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(
      p, __ESIMD_NS::simd<Toffset, N>(offset), src0, pred);
}
/// USM pointer atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand (expected value).
/// @param src1 is the second atomic operand (new value).
/// @param pred predicates.
///
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<__ESIMD_DNS::to_lsc_atomic_op<Op>()>() == 2,
    __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd<Toffset, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                  __ESIMD_NS::simd_mask<N> pred) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  constexpr __ESIMD_NS::native::lsc::atomic_op _Op =
      __ESIMD_DNS::to_lsc_atomic_op<Op>();
  __ESIMD_EDNS::check_lsc_atomic<_Op, T, N, 2>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_stateless_2<_MsgT, _Op, L1H, L3H, _AddressScale,
                                      _ImmOffset, _DS, _VS, _Transposed, N>(
          pred.data(), addrs.data(),
          src0.template bit_cast_view<_MsgT>().data(),
          src1.template bit_cast_view<_MsgT>().data());
  return detail::lsc_format_ret<T>(Tmp);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset,
          typename RegionTy = __ESIMD_NS::region1d_t<Toffset, N, 1>>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<__ESIMD_DNS::to_lsc_atomic_op<Op>()>() == 2,
    __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd_view<Toffset, RegionTy> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                  __ESIMD_NS::simd_mask<N> pred = 1) {
  using Ty = typename __ESIMD_NS::simd_view<Toffset, RegionTy>::element_type;
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(
      p, __ESIMD_NS::simd<Ty, N>(offsets), src0, src1, pred);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> &&
        __ESIMD_DNS::get_num_args<__ESIMD_DNS::to_lsc_atomic_op<Op>()>() == 2,
    __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, Toffset offset, __ESIMD_NS::simd<T, N> src0,
                  __ESIMD_NS::simd<T, N> src1,
                  __ESIMD_NS::simd_mask<N> pred = 1) {
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(
      p, __ESIMD_NS::simd<Toffset, N>(offset), src0, src1, pred);
}

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets.
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd_mask<N> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(
      __ESIMD_DNS::accessorToPointer<T>(acc), offsets, pred);
#else
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  constexpr __ESIMD_NS::native::lsc::atomic_op _Op =
      __ESIMD_DNS::to_lsc_atomic_op<Op>();
  __ESIMD_EDNS::check_lsc_atomic<_Op, T, N, 0>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  auto si = __ESIMD_NS::get_surface_index(acc);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_bti_0<_MsgT, _Op, L1H, L3H, _AddressScale, _ImmOffset,
                                _DS, _VS, _Transposed, N>(pred.data(),
                                                          offsets.data(), si);
  return detail::lsc_format_ret<T>(Tmp);
#endif
}

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd_mask<N> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(
      __ESIMD_DNS::accessorToPointer<T>(acc), offsets, src0, pred);
#else
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  constexpr __ESIMD_NS::native::lsc::atomic_op _Op =
      __ESIMD_DNS::to_lsc_atomic_op<Op>();
  __ESIMD_EDNS::check_lsc_atomic<_Op, T, N, 1>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  auto si = __ESIMD_NS::get_surface_index(acc);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_bti_1<_MsgT, _Op, L1H, L3H, _AddressScale, _ImmOffset,
                                _DS, _VS, _Transposed, N>(
          pred.data(), offsets.data(), src0.data(), si);
  return detail::lsc_format_ret<T>(Tmp);
#endif
}

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                  __ESIMD_NS::simd_mask<N> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(
      __ESIMD_DNS::accessorToPointer<T>(acc), offsets, src0, src1, pred);
#else
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  constexpr __ESIMD_NS::native::lsc::atomic_op _Op =
      __ESIMD_DNS::to_lsc_atomic_op<Op>();
  __ESIMD_EDNS::check_lsc_atomic<_Op, T, N, 2>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  auto si = __ESIMD_NS::get_surface_index(acc);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_bti_2<_MsgT, _Op, L1H, L3H, _AddressScale, _ImmOffset,
                                _DS, _VS, _Transposed, N>(
          pred.data(), offsets.data(), src0.data(), src1.data(), si);
  return detail::lsc_format_ret<T>(Tmp);
#endif
}

/// Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the Sfid shaded function.
/// @tparam FenceOp is the fence operation.
/// @tparam Scope is the operation scope.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
template <lsc_memory_kind Kind = lsc_memory_kind::untyped_global,
          lsc_fence_op FenceOp = lsc_fence_op::none,
          lsc_scope Scope = lsc_scope::group, int N = 16>
__ESIMD_API void lsc_fence(__ESIMD_NS::simd_mask<N> pred = 1) {
  static_assert(
      Kind != lsc_memory_kind::shared_local ||
          (FenceOp == lsc_fence_op::none && Scope == lsc_scope::group),
      "SLM fence must have 'none' lsc_fence_op and 'group' scope");
  __esimd_lsc_fence<Kind, FenceOp, Scope, N>(pred.data());
}

/// @} sycl_esimd_memory_lsc

/// @defgroup sycl_esimd_hw_thread_queries HW thread .
/// @ingroup sycl_esimd_memory

/// @addtogroup sycl_esimd_hw_thread_queries
/// @{

/// Get HW Thread ID
__ESIMD_API int32_t get_hw_thread_id() {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInGlobalHWThreadIDINTEL();
#else
  return std::rand();
#endif // __SYCL_DEVICE_ONLY__
}
/// Get subdevice ID
__ESIMD_API int32_t get_subdevice_id() {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_BuiltInSubDeviceIDINTEL();
#else
  return 0;
#endif
}

/// @} sycl_esimd_hw_thread_queries

} // namespace experimental::esimd

namespace esimd {

/// LSC version of no argument variant of the \c atomic_update - accepts
/// <tt>native::lsc::atomic_op</tt> instead of <tt>atomic_op</tt> as atomic
/// operation template argument.
template <native::lsc::atomic_op Op, typename T, int N, typename Toffset>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 0, simd<T, N>>
atomic_update(T *p, simd<Toffset, N> offset, simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offset, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename Toffset,
          typename RegionTy = __ESIMD_NS::region1d_t<Toffset, N, 1>>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 0, simd<T, N>>
atomic_update(T *p, simd_view<Toffset, RegionTy> offsets,
              simd_mask<N> mask = 1) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offsets, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 0,
                             simd<T, N>>
atomic_update(T *p, Toffset offset, simd_mask<N> mask = 1) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offset, mask);
}

/// LSC version of the single-argument atomic update.
template <native::lsc::atomic_op Op, typename T, int N, typename Toffset>
__ESIMD_API __ESIMD_API
    std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1, simd<T, N>>
    atomic_update(T *p, simd<Toffset, N> offset, simd<T, N> src0,
                  simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offset, src0, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename Toffset,
          typename RegionTy = __ESIMD_NS::region1d_t<Toffset, N, 1>>
__ESIMD_API __ESIMD_API
    std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1, simd<T, N>>
    atomic_update(T *p, simd_view<Toffset, RegionTy> offsets, simd<T, N> src0,
                  simd_mask<N> mask = 1) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offsets, src0, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 1,
                             simd<T, N>>
atomic_update(T *p, Toffset offset, simd<T, N> src0, simd_mask<N> mask = 1) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offset, src0, mask);
}

/// LSC version of the two-argument atomic update.
template <native::lsc::atomic_op Op, typename T, int N, typename Toffset>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2, simd<T, N>>
atomic_update(T *p, simd<Toffset, N> offset, simd<T, N> src0, simd<T, N> src1,
              simd_mask<N> mask) {
  // 2-argument lsc_atomic_update arguments order matches the standard one -
  // expected value first, then new value. But atomic_update uses reverse
  // order, hence the src1/src0 swap.
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offset, src1, src0, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename Toffset,
          typename RegionTy = __ESIMD_NS::region1d_t<Toffset, N, 1>>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2, simd<T, N>>
atomic_update(T *p, simd_view<Toffset, RegionTy> offsets, simd<T, N> src0,
              simd<T, N> src1, simd_mask<N> mask = 1) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offsets, src1, src0, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 2,
                             __ESIMD_NS::simd<T, N>>
atomic_update(T *p, Toffset offset, simd<T, N> src0, simd<T, N> src1,
              simd_mask<N> mask = 1) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offset, src1, src0, mask);
}

/// RAII-style class used to implement "semi-dynamic" SLM allocation.
/// SLM is allocated in the constructor and released in the destructor, that's
/// why it is "dynamic", as opposed to fully static allocation style of
/// 'slm_init'. Actual offset of SLM chunk allocated by the call is calculated
/// at compile time, that's why it is "semi-". To calculate SLM usage by a
/// kernel, compiler finds a path in a callgraph with the largest amount of SLM
/// "locked" by slm_allocator objects live along the paths. slm_init call also
/// participates in calculating SLM budget. It can be modelled as
/// \c slm_allocator object declared at the very beginning of a kernel and live
/// till its the very end.
/// Only compile-time constant SLM amount is supported for now, it is provided
/// as a class' template argument.
///
/// Since a call graph is used, function pointers and recursion is not
/// supported.
///
/// @tparam SLMAmount The amount allocated in bytes
template <int SLMAmount> class slm_allocator {
  int offset;

public:
  /// Allocates the amount of SLM which is class' template parameter.
  slm_allocator() { offset = __esimd_slm_alloc(SLMAmount); }

  /// @return The allocated chunk's offset in bytes.
  int get_offset() const { return offset; }

  /// Releases the SLM chunk allocated in the constructor.
  ~slm_allocator() { __esimd_slm_free(offset); }
};

} // namespace esimd
} // namespace ext::intel
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
