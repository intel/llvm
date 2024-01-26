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
inline namespace _V1 {
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

/// Raw sends. "s" suffix designates "split" variant - i.e. two sources.
///
/// @tparam execSize is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam numSrc0 is the number of GRFs for source-0.
/// @tparam numSrc1 is the number of GRFs for source-1.
/// @tparam numDst is the number of GRFs for destination.
/// @tparam isEOT is the flag that indicates whether this is an EOT message
/// (optional - default to 0).
/// @tparam isSendc is the flag that indicates whether sendc should be used
/// (optional - default to 0).
/// @param msgDst is the old value of the destination operand.
/// @param msgSrc0 is the first source operand of send message.
/// @param msgSrc1 is the second source operand of send message.
/// @param exDesc is the extended message descriptor.
/// @param msgDesc is the message descriptor.
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
/// @return the vector value read from memory.
template <uint8_t execSize, uint8_t sfid, uint8_t numSrc0, uint8_t numSrc1,
          uint8_t numDst, uint8_t isEOT = 0, uint8_t isSendc = 0, typename T1,
          int n1, typename T2, int n2, typename T3, int n3>
__SYCL_DEPRECATED("use sycl::ext::intel::esimd::raw_sends")
__ESIMD_API __ESIMD_NS::simd<T1, n1> raw_sends(
    __ESIMD_NS::simd<T1, n1> msgDst, __ESIMD_NS::simd<T2, n2> msgSrc0,
    __ESIMD_NS::simd<T3, n3> msgSrc1, uint32_t exDesc, uint32_t msgDesc,
    __ESIMD_NS::simd_mask<execSize> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send rspVar");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msgSrc0");
  constexpr unsigned _Width3 = n3 * sizeof(T3);
  static_assert(_Width3 % 32 == 0, "Invalid size for raw send msgSrc1");

  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  using ElemT2 = __ESIMD_DNS::__raw_t<T2>;
  using ElemT3 = __ESIMD_DNS::__raw_t<T3>;

  constexpr uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);

  return __esimd_raw_sends2<ElemT1, n1, ElemT2, n2, ElemT3, n3, execSize>(
      modifier, execSize, mask.data(), numSrc0, numSrc1, numDst, sfid, exDesc,
      msgDesc, msgSrc0.data(), msgSrc1.data(), msgDst.data());
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

/// Raw send.
///
/// @tparam execSize is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam numSrc0 is the number of GRFs for source-0.
/// @tparam numDst is the number of GRFs for destination.
/// @tparam isEOT is the flag that indicates whether this is an EOT message
/// (optional - default to 0).
/// @tparam isSendc is the flag that indicates whether sendc should be used
/// (optional - default to 0).
/// @param msgDst is the old value of the destination operand.
/// @param msgSrc0 is the first source operand of send message.
/// @param exDesc is the extended message descriptor.
/// @param msgDesc is the message descriptor.
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
/// @return the vector value read from memory
template <uint8_t execSize, uint8_t sfid, uint8_t numSrc0, uint8_t numDst,
          uint8_t isEOT = 0, uint8_t isSendc = 0, typename T1, int n1,
          typename T2, int n2>
__SYCL_DEPRECATED("use sycl::ext::intel::esimd::raw_send")
__ESIMD_API __ESIMD_NS::simd<T1, n1> raw_send(
    __ESIMD_NS::simd<T1, n1> msgDst, __ESIMD_NS::simd<T2, n2> msgSrc0,
    uint32_t exDesc, uint32_t msgDesc,
    __ESIMD_NS::simd_mask<execSize> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send rspVar");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msgSrc0");

  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  using ElemT2 = __ESIMD_DNS::__raw_t<T2>;

  constexpr uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  return __esimd_raw_send2<ElemT1, n1, ElemT2, n2, execSize>(
      modifier, execSize, mask.data(), numSrc0, numDst, sfid, exDesc, msgDesc,
      msgSrc0.data(), msgDst.data());
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

/// Raw sends. "s" suffix designates "split" variant - i.e. two sources.
///
/// @tparam execSize is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam numSrc0 is the number of GRFs for source-0.
/// @tparam numSrc1 is the number of GRFs for source-1.
/// @tparam isEOT is the flag that indicates whether this is an EOT message
/// (optional - default to 0).
/// @tparam isSendc is the flag that indicates whether sendc should be used
/// (optional - default to 0).
/// @param msgSrc0 is the first source operand of send message.
/// @param msgSrc1 is the second source operand of send message.
/// @param exDesc is the extended message descriptor.
/// @param msgDesc is the message descriptor.
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
template <uint8_t execSize, uint8_t sfid, uint8_t numSrc0, uint8_t numSrc1,
          uint8_t isEOT = 0, uint8_t isSendc = 0, typename T1, int n1,
          typename T2, int n2>
__SYCL_DEPRECATED("use sycl::ext::intel::esimd::raw_sends")
__ESIMD_API
    void raw_sends(__ESIMD_NS::simd<T1, n1> msgSrc0,
                   __ESIMD_NS::simd<T2, n2> msgSrc1, uint32_t exDesc,
                   uint32_t msgDesc, __ESIMD_NS::simd_mask<execSize> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send msgSrc0");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msgSrc1");

  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  using ElemT2 = __ESIMD_DNS::__raw_t<T2>;

  constexpr uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  __esimd_raw_sends2_noresult<ElemT1, n1, ElemT2, n2, execSize>(
      modifier, execSize, mask.data(), numSrc0, numSrc1, sfid, exDesc, msgDesc,
      msgSrc0.data(), msgSrc1.data());
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

/// Raw send. Generates a \c send or \c sendc instruction for the message
/// gateway.
///
/// @tparam execSize is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam numSrc0 is the number of GRFs for source-0.
/// @tparam isEOT is the flag that indicates whether this is an EOT message
/// (optional - default to 0).
/// @tparam isSendc is the flag that indicates whether sendc should be used
/// (optional - default to 0).
/// @param msgSrc0 is the first source operand of send message.
/// @param exDesc is the extended message descriptor.
/// @param msgDesc is the message descriptor.
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
template <uint8_t execSize, uint8_t sfid, uint8_t numSrc0, uint8_t isEOT = 0,
          uint8_t isSendc = 0, typename T1, int n1>
__SYCL_DEPRECATED("use sycl::ext::intel::esimd::raw_send")
__ESIMD_API
    void raw_send(__ESIMD_NS::simd<T1, n1> msgSrc0, uint32_t exDesc,
                  uint32_t msgDesc, __ESIMD_NS::simd_mask<execSize> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send msgSrc0");
  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  constexpr uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  __esimd_raw_send2_noresult<ElemT1, n1, execSize>(
      modifier, execSize, mask.data(), numSrc0, sfid, exDesc, msgDesc,
      msgSrc0.data());
}

/// @} sycl_esimd_raw_send

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
  if constexpr (Transformed)
    return detail::roundUpNextMultiple<Height, 4 / sizeof(T)>() *
           __ESIMD_DNS::getNextPowerOf2<Width>() * NBlocks;
  return Width * Height * NBlocks;
}

// Format u8 and u16 to u8u32 and u16u32 by doing garbage-extension.
template <typename RT, typename T, int N>
ESIMD_INLINE __ESIMD_NS::simd<RT, N>
lsc_format_input(__ESIMD_NS::simd<T, N> Vals) {
  return __ESIMD_DNS::lsc_format_input<RT, T, N>(Vals);
}

// Format u8u32 and u16u32 back to u8 and u16.
template <typename T, typename T1, int N>
ESIMD_INLINE __ESIMD_NS::simd<T, N>
lsc_format_ret(__ESIMD_NS::simd<T1, N> Vals) {
  return __ESIMD_DNS::lsc_format_ret<T, T1, N>(Vals);
}

template <typename T> constexpr uint32_t get_lsc_data_size() {
  switch (sizeof(T)) {
  case 1:
    return 0;
  case 2:
    return 1;
  case 4:
    return 2;
  case 8:
    return 3;
  default:
    static_assert(true, "Unsupported data type.");
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
  using MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<MsgT, N * NElts> Tmp =
      __esimd_lsc_load_slm<MsgT, cache_hint::none, cache_hint::none,
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
/// @param pass_thru values copied to the result when the corresponding
/// element of \p pred is zero..
/// @return is a vector of type T and size N * NElts
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size, int N>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
lsc_slm_gather(__ESIMD_NS::simd<uint32_t, N> offsets,
               __ESIMD_NS::simd_mask<N> pred,
               __ESIMD_NS::simd<T, N * NElts> pass_thru) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<MsgT, N * NElts> PassThruExpanded =
      detail::lsc_format_input<MsgT>(pass_thru);
  __ESIMD_NS::simd<MsgT, N * NElts> Result =
      __esimd_lsc_load_merge_slm<MsgT, cache_hint::none, cache_hint::none,
                                 _AddressScale, _ImmOffset, _DS, _VS,
                                 _Transposed, N>(pred.data(), offsets.data(),
                                                 PassThruExpanded.data());
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
/// @tparam DS is the data size (unused/obsolete).
/// @param offset is the zero-based offset for SLM buffer in bytes.
/// @param pred is the predicate; if it contains 0, then the actual load
/// is not performed and the returned value is undefined.
/// @return is a vector of type T and size NElts
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, NElts>
lsc_slm_block_load(uint32_t offset, __ESIMD_NS::simd_mask<1> pred = 1) {
  constexpr size_t DefaultAlignment = sizeof(T) <= 4 ? 4 : sizeof(T);
  __ESIMD_NS::properties Props{__ESIMD_NS::alignment<DefaultAlignment>};
  return __ESIMD_NS::slm_block_load<T, NElts>(offset, pred, Props);
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
/// @tparam DS is the data size (unused/obsolete).
/// @param offset is the zero-based offset for SLM buffer in bytes.
/// @param pred is the predicate; if it contains 0, then the actual load
/// is not performed and \p pass_thru is returned.
/// @param pass_thru contains the vector that is returned if
/// the parameter \p pred contains 0.
/// @return is a vector of type T and size NElts.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, NElts>
lsc_slm_block_load(uint32_t offset, __ESIMD_NS::simd_mask<1> pred,
                   __ESIMD_NS::simd<T, NElts> pass_thru) {
  constexpr size_t DefaultAlignment = sizeof(T) <= 4 ? 4 : sizeof(T);
  __ESIMD_NS::properties Props{__ESIMD_NS::alignment<DefaultAlignment>};
  return __ESIMD_NS::slm_block_load<T, NElts>(offset, pred, pass_thru, Props);
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
  return __ESIMD_DNS::gather_impl<T, NElts, DS, L1H, L3H>(p, offsets, pred);
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
/// @param pass_thru contains the vector which elements are copied
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
           __ESIMD_NS::simd<T, N * NElts> pass_thru) {
  return __ESIMD_DNS::gather_impl<T, NElts, DS, L1H, L3H>(p, offsets, pred,
                                                          pass_thru);
}

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename OffsetObjT, typename RegionTy>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
lsc_gather(const T *p, __ESIMD_NS::simd_view<OffsetObjT, RegionTy> offsets,
           __ESIMD_NS::simd_mask<N> pred = 1) {
  return lsc_gather<T, NElts, DS, L1H, L3H, N>(p, offsets.read(), pred);
}

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename OffsetObjT, typename RegionTy>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
lsc_gather(const T *p, __ESIMD_NS::simd_view<OffsetObjT, RegionTy> offsets,
           __ESIMD_NS::simd_mask<N> pred,
           __ESIMD_NS::simd<T, N * NElts> pass_thru) {
  return lsc_gather<T, NElts, DS, L1H, L3H, N>(p, offsets.read(), pred,
                                               pass_thru);
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
           __ESIMD_NS::simd<T, N * NElts> pass_thru) {
  return lsc_gather<T, NElts, DS, L1H, L3H, N>(
      p, __ESIMD_NS::simd<Toffset, N>(offset), pred, pass_thru);
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
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::is_device_accessor_with_v<
                         AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read>,
                     __ESIMD_NS::simd<T, N * NElts>>
    lsc_gather(AccessorTy acc,
               __ESIMD_NS::simd<__ESIMD_DNS::DeviceAccessorOffsetT, N> offsets,
               __ESIMD_NS::simd_mask<N> pred = 1) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_gather<T, NElts, DS, L1H, L3H>(
      reinterpret_cast<T *>(acc.get_pointer().get()), offsets, pred);
#else
  __ESIMD_NS::simd<T, N * NElts> PassThru; // Intentionally unitialized.
  return __ESIMD_DNS::gather_impl<T, N * NElts, NElts, L1H, L3H, DS>(
      acc, offsets, pred, PassThru);
#endif // __ESIMD_FORCE_STATELESS_MEM
}

#ifdef __ESIMD_FORCE_STATELESS_MEM
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_device_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read> &&
        std::is_integral_v<Toffset> && !std::is_same_v<Toffset, uint64_t>,
    __ESIMD_NS::simd<T, N * NElts>>
lsc_gather(AccessorTy acc, __ESIMD_NS::simd<Toffset, N> offsets,
           __ESIMD_NS::simd_mask<N> pred = 1) {
  return lsc_gather<T, NElts, DS, L1H, L3H, N, AccessorTy>(
      acc, convert<uint64_t>(offsets), pred);
}
#endif

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::is_local_accessor_with_v<
                         AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read>,
                     __ESIMD_NS::simd<T, N * NElts>>
    lsc_gather(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
               __ESIMD_NS::simd_mask<N> pred = 1) {
  return lsc_slm_gather<T, NElts, DS>(
      offsets + __ESIMD_DNS::localAccessorToOffset(acc), pred);
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
/// @param pass_thru contains the vector which elements are copied
/// to the returned result when the corresponding element of \p pred is 0.
/// @return is a vector of type T and size N * NElts
///
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::is_device_accessor_with_v<
                         AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read>,
                     __ESIMD_NS::simd<T, N * NElts>>
    lsc_gather(AccessorTy acc,
               __ESIMD_NS::simd<__ESIMD_DNS::DeviceAccessorOffsetT, N> offsets,
               __ESIMD_NS::simd_mask<N> pred,
               __ESIMD_NS::simd<T, N * NElts> pass_thru) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_gather<T, NElts, DS, L1H, L3H>(
      reinterpret_cast<T *>(acc.get_pointer().get()), offsets, pred, pass_thru);

#else
  return __ESIMD_DNS::gather_impl<T, N * NElts, NElts, L1H, L3H, DS>(
      acc, offsets, pred, pass_thru);
#endif // __ESIMD_FORCE_STATELESS_MEM
}

#ifdef __ESIMD_FORCE_STATELESS_MEM
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_device_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read> &&
        std::is_integral_v<Toffset> && !std::is_same_v<Toffset, uint64_t>,
    __ESIMD_NS::simd<T, N * NElts>>
lsc_gather(AccessorTy acc, __ESIMD_NS::simd<Toffset, N> offsets,
           __ESIMD_NS::simd_mask<N> pred,
           __ESIMD_NS::simd<T, N * NElts> pass_thru) {
  return lsc_gather<T, NElts, DS, L1H, L3H, N, AccessorTy>(
      acc, convert<uint64_t>(offsets), pred, pass_thru);
}
#endif

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<
    sycl::detail::acc_properties::is_local_accessor_v<AccessorTy>,
    __ESIMD_NS::simd<T, N * NElts>>
lsc_gather(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
           __ESIMD_NS::simd_mask<N> pred,
           __ESIMD_NS::simd<T, N * NElts> pass_thru) {
  return lsc_slm_gather<T, NElts, DS>(
      offsets + __ESIMD_DNS::localAccessorToOffset(acc), pred, pass_thru);
}

/// USM pointer transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Accesses contiguous block of memory of `NElts * S` bytes  starting from
/// given address, where S is a byte size of an "element" defined by the \c DS
/// template parameter. The maximum size of accessed block is 512 bytes for PVC
/// and 256 bytes for ACM (DG2).
/// When sizeof(T) equal to 8 the address must be 8-byte aligned.
/// Also, 8-bytes alignment is required when the function has to load
/// more than 256-bytes. In all other cases 4-byte alignment is required.
/// When T is 1- or 2-byte type the data is treated as 4-byte data.
/// Allowed \c NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
/// 8 bytes alignment is required for 64 bit data, 32 bit data and \c NElts
/// equal to 128, 16 bit data and \c NElts equal to 256, 8 bit data and \c
/// NElts equal to 512. Otherwise 4 bytes alignment is required.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size (unused/obsolete).
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
  return __ESIMD_DNS::block_load_impl<T, NElts, L1H, L3H>(p, pred, flags);
}

/// A variation of lsc_block_load without predicate parameter to simplify use
/// of alignment parameter
///
/// Accesses contiguous block of memory of `NElts * S` bytes  starting from
/// given address, where S is a byte size of an "element" defined by the \c DS
/// template parameter. The maximum size of accessed block is 512 bytes for PVC
/// and 256 bytes for ACM (DG2).
/// When sizeof(T) equal to 8 the address must be 8-byte aligned.
/// Also, 8-bytes alignment is required when the function has to load
/// more than 256-bytes. In all other cases 4-byte alignment is required.
/// When T is 1- or 2-byte type the data is treated as 4-byte data.
/// Allowed \c NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size (unused/obsolete).
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
  return __ESIMD_DNS::block_load_impl<T, NElts, L1H, L3H>(
      p, __ESIMD_NS::simd_mask<1>(1), flags);
}

/// USM pointer transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Accesses contiguous block of memory of `NElts * S` bytes  starting from
/// given address, where S is a byte size of an "element" defined by the \c DS
/// template parameter. The maximum size of accessed block is 512 bytes for PVC
/// and 256 bytes for ACM (DG2).
/// When sizeof(T) equal to 8 the address must be 8-byte aligned.
/// Also, 8-bytes alignment is required when the function has to load
/// more than 256-bytes. In all other cases 4-byte alignment is required.
/// When T is 1- or 2-byte type the data is treated as 4-byte data.
/// Allowed \c NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size (unused/obsolete).
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @param p is the base pointer.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed.
/// @param pass_thru contains the vector which elements are copied
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
               __ESIMD_NS::simd<T, NElts> pass_thru, FlagsT flags = FlagsT{}) {
  return __ESIMD_DNS::block_load_impl<T, NElts, L1H, L3H>(p, pred, pass_thru,
                                                          flags);
}

/// Accessor-based transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
/// When sizeof(T) equal to 8 the address must be 8-byte aligned.
/// Also, 8-bytes alignment is required when the function has to load
/// more than 256-bytes. In all other cases 4-byte alignment is required.
/// When T is 1- or 2-byte type the data is treated as 4-byte data.
/// Allowed \c NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size (unused/obsolete).
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
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_device_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read> &&
        __ESIMD_NS::is_simd_flag_type_v<FlagsT>,
    __ESIMD_NS::simd<T, NElts>>
lsc_block_load(AccessorTy acc, __ESIMD_DNS::DeviceAccessorOffsetT offset,
               __ESIMD_NS::simd_mask<1> pred = 1, FlagsT flags = FlagsT{}) {
  return __ESIMD_DNS::block_load_impl<T, NElts, L1H, L3H>(acc, offset, pred,
                                                          flags);
}

template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_local_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read> &&
        __ESIMD_NS::is_simd_flag_type_v<FlagsT>,
    __ESIMD_NS::simd<T, NElts>>
lsc_block_load(AccessorTy acc, uint32_t offset,
               __ESIMD_NS::simd_mask<1> pred = 1, FlagsT flags = FlagsT{}) {
  return lsc_slm_block_load<T, NElts, DS>(
      offset + __ESIMD_DNS::localAccessorToOffset(acc), pred);
}

/// A variation of lsc_block_load without predicate parameter to simplify use
/// of alignment parameter
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
/// When sizeof(T) equal to 8 the address must be 8-byte aligned.
/// Also, 8-bytes alignment is required when the function has to load
/// more than 256-bytes. In all other cases 4-byte alignment is required.
/// When T is 1- or 2-byte type the data is treated as 4-byte data.
/// Allowed \c NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size (unused/obsolete).
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
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_device_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read> &&
        __ESIMD_NS::is_simd_flag_type_v<FlagsT>,
    __ESIMD_NS::simd<T, NElts>>
lsc_block_load(AccessorTy acc, __ESIMD_DNS::DeviceAccessorOffsetT offset,
               FlagsT flags) {
  return lsc_block_load<T, NElts, DS, L1H, L3H>(
      acc, offset, __ESIMD_NS::simd_mask<1>(1), flags);
}

template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_local_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read> &&
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
/// When sizeof(T) equal to 8 the address must be 8-byte aligned.
/// Also, 8-bytes alignment is required when the function has to load
/// more than 256-bytes. In all other cases 4-byte alignment is required.
/// When T is 1- or 2-byte type the data is treated as 4-byte data.
/// Allowed \c NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size (unused/obsolete).
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param pred is operation predicate. Operation is skipped for index 'i'
/// if pred[i] == 0 and the result element is taken from \p pass_thru[i].
/// Otherwise, the operation is performed.
/// @param pass_thru contains the values copied to the result when
/// the corresponding element from \p pred is zero.
/// @param flags is the alignment specifier type tag.
/// @return is a vector of type T and size NElts
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_device_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read> &&
        __ESIMD_NS::is_simd_flag_type_v<FlagsT>,
    __ESIMD_NS::simd<T, NElts>>
lsc_block_load(AccessorTy acc, __ESIMD_DNS::DeviceAccessorOffsetT offset,
               __ESIMD_NS::simd_mask<1> pred,
               __ESIMD_NS::simd<T, NElts> pass_thru, FlagsT flags = FlagsT{}) {
  return __ESIMD_DNS::block_load_impl<T, NElts, L1H, L3H>(acc, offset, pred,
                                                          pass_thru, flags);
}

template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_local_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read> &&
        __ESIMD_NS::is_simd_flag_type_v<FlagsT>,
    __ESIMD_NS::simd<T, NElts>>
lsc_block_load(AccessorTy acc, uint32_t offset, __ESIMD_NS::simd_mask<1> pred,
               __ESIMD_NS::simd<T, NElts> pass_thru, FlagsT flags = FlagsT{}) {
  return lsc_slm_block_load<T, NElts, DS>(
      offset + __ESIMD_DNS::localAccessorToOffset(acc), pred, pass_thru);
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
  using MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __esimd_lsc_prefetch_stateless<MsgT, L1H, L3H, _AddressScale, _ImmOffset, _DS,
                                 _VS, _Transposed, N>(pred.data(),
                                                      addrs.data());
}

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename OffsetObjT, typename RegionTy>
__ESIMD_API void
lsc_prefetch(const T *p, __ESIMD_NS::simd_view<OffsetObjT, RegionTy> offsets,
             __ESIMD_NS::simd_mask<N> pred = 1) {
  lsc_prefetch<T, NElts, DS, L1H, L3H, N>(p, offsets.read(), pred);
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
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_device_accessor_with_v<
    AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read>>
lsc_prefetch(AccessorTy acc,
#ifdef __ESIMD_FORCE_STATELESS_MEM
             __ESIMD_NS::simd<uint64_t, N> offsets,
#else
             __ESIMD_NS::simd<uint32_t, N> offsets,
#endif
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
  using MsgT = typename detail::lsc_expand_type<T>::type;
  auto si = __ESIMD_NS::get_surface_index(acc);
  __esimd_lsc_prefetch_bti<MsgT, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(), si);
#endif
}

#ifdef __ESIMD_FORCE_STATELESS_MEM
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_device_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read> &&
    std::is_integral_v<Toffset> && !std::is_same_v<Toffset, uint64_t>>
lsc_prefetch(AccessorTy acc, __ESIMD_NS::simd<Toffset, N> offsets,
             __ESIMD_NS::simd_mask<N> pred = 1) {
  lsc_prefetch<T, NElts, DS, L1H, L3H, N, AccessorTy>(
      acc, convert<uint64_t>(offsets), pred);
}
#endif

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
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_device_accessor_with_v<
    AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read>>
lsc_prefetch(AccessorTy acc, __ESIMD_DNS::DeviceAccessorOffsetT offset) {
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
  using MsgT = typename detail::lsc_expand_type<T>::type;
  using CstT = typename detail::lsc_bitcast_type<T>::type;
  __ESIMD_NS::simd<MsgT, N * NElts> Tmp = vals.template bit_cast_view<CstT>();
  __esimd_lsc_store_slm<MsgT, cache_hint::none, cache_hint::none, _AddressScale,
                        _ImmOffset, _DS, _VS, _Transposed, N>(
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
/// @tparam DS is the data size (unused/obsolete).
/// @param offset is the zero-based offset for SLM buffer in bytes.
/// @param vals is values to store.
///
template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API void lsc_slm_block_store(uint32_t offset,
                                     __ESIMD_NS::simd<T, NElts> vals) {
  // Make sure we generate an LSC block store
  constexpr size_t DefaultAlignment = sizeof(T) <= 4 ? 4 : sizeof(T);
  __ESIMD_NS::properties Props{__ESIMD_NS::alignment<DefaultAlignment>};
  __ESIMD_NS::simd_mask<1> pred = 1;
  __ESIMD_NS::slm_block_store<T, NElts>(offset, vals, pred, Props);
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
  __ESIMD_DNS::scatter_impl<T, NElts, DS, L1H, L3H, N, Toffset>(p, offsets,
                                                                vals, pred);
}

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename OffsetObjT, typename RegionTy>
__ESIMD_API void
lsc_scatter(T *p, __ESIMD_NS::simd_view<OffsetObjT, RegionTy> offsets,
            __ESIMD_NS::simd<T, N * NElts> vals,
            __ESIMD_NS::simd_mask<N> pred = 1) {
  lsc_scatter<T, NElts, DS, L1H, L3H, N>(p, offsets.read(), vals, pred);
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
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_device_accessor_with_v<
    AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_write>>
lsc_scatter(AccessorTy acc,
            __ESIMD_NS::simd<__ESIMD_DNS::DeviceAccessorOffsetT, N> offsets,
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
  using MsgT = typename detail::lsc_expand_type<T>::type;
  using _CstT = typename detail::lsc_bitcast_type<T>::type;
  __ESIMD_NS::simd<MsgT, N * NElts> Tmp = vals.template bit_cast_view<_CstT>();
  auto si = __ESIMD_NS::get_surface_index(acc);
  __esimd_lsc_store_bti<MsgT, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                        _Transposed, N>(pred.data(), offsets.data(), Tmp.data(),
                                        si);
#endif
}

#ifdef __ESIMD_FORCE_STATELESS_MEM
template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_device_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_write> &&
    std::is_integral_v<Toffset> && !std::is_same_v<Toffset, uint64_t>>
lsc_scatter(AccessorTy acc, __ESIMD_NS::simd<Toffset, N> offsets,
            __ESIMD_NS::simd<T, N * NElts> vals,
            __ESIMD_NS::simd_mask<N> pred = 1) {
  lsc_scatter<T, NElts, DS, L1H, L3H, N, AccessorTy>(
      acc, convert<uint64_t>(offsets), vals, pred);
}
#endif

template <typename T, int NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_local_accessor_with_v<
    AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_write>>
lsc_scatter(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
            __ESIMD_NS::simd<T, N * NElts> vals,
            __ESIMD_NS::simd_mask<N> pred = 1) {
  lsc_slm_scatter<T, NElts, DS>(
      offsets + __ESIMD_DNS::localAccessorToOffset(acc), vals, pred);
}

/// USM pointer transposed scatter with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_store.ugm
///
/// Scatters elements to specific address.
/// When \c DS equals \c lsc_data_size::u64 or \c sizeof(T) equal to 8 the
/// address must be 8-byte aligned, otherwise - 4-bytes aligned. Allowed values
/// for the data size are \c lsc_data_size::u32, \c lsc_data_size::u64,
/// \c lsc_data_size::u8, \c lsc_data_size::u16.
/// When data size is either  \c lsc_data_size::u8 or \c lsc_data_size::u16
/// the data is treated as 32 bit data.
/// Allowed \c NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
/// 8 bytes alignment is required for 64 bit data, 32 bit data and \c NElts
/// equal to 128, 16 bit data and \c NElts equal to 256, 8 bit data and \c
/// NElts equal to 512. Otherwise 4 bytes alignment is required.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size (unused/obsolete).
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
  return __ESIMD_DNS::block_store_impl<T, NElts, L1H, L3H>(p, vals, pred,
                                                           flags);
}

/// A variation of lsc_block_store without predicate parameter to simplify
/// use of alignment parameter
///
/// Scatters elements to specific address.
/// When \c DS equals \c lsc_data_size::u64 or \c sizeof(T) equal to 8 the
/// address must be 8-byte aligned, otherwise - 4-bytes aligned. Allowed values
/// for the data size are \c lsc_data_size::u32, \c lsc_data_size::u64,
/// \c lsc_data_size::u8, \c lsc_data_size::u16.
/// When data size is either  \c lsc_data_size::u8 or \c lsc_data_size::u16
/// the data is treated as 32 bit data.
/// Allowed \c NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
/// 8 bytes alignment is required for 64 bit data, 32 bit data and \c NElts
/// equal to 128, 16 bit data and \c NElts equal to 256, 8 bit data and \c
/// NElts equal to 512. Otherwise 4 bytes alignment is required.
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
/// When \c DS equals \c lsc_data_size::u64 or \c sizeof(T) equal to 8 the
/// address must be 8-byte aligned, otherwise - 4-bytes aligned. Allowed values
/// for the data size are \c lsc_data_size::u32, \c lsc_data_size::u64,
/// \c lsc_data_size::u8, \c lsc_data_size::u16.
/// When data size is either  \c lsc_data_size::u8 or \c lsc_data_size::u16
/// the data is treated as 32 bit data.
/// Allowed \c NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
/// 8 bytes alignment is required for 64 bit data, 32 bit data and \c NElts
/// equal to 128, 16 bit data and \c NElts equal to 256, 8 bit data and \c
/// NElts equal to 512. Otherwise 4 bytes alignment is required.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size (unused/obsolete).
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
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_device_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_write> &&
    __ESIMD_NS::is_simd_flag_type_v<FlagsT>>
lsc_block_store(AccessorTy acc, __ESIMD_DNS::DeviceAccessorOffsetT offset,
                __ESIMD_NS::simd<T, NElts> vals,
                __ESIMD_NS::simd_mask<1> pred = 1, FlagsT flags = FlagsT{}) {
  __ESIMD_DNS::block_store_impl<T, NElts, L1H, L3H>(acc, offset, vals, pred,
                                                    flags);
}

template <typename T, int NElts, lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_local_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_write> &&
    __ESIMD_NS::is_simd_flag_type_v<FlagsT>>
lsc_block_store(AccessorTy acc, uint32_t offset,
                __ESIMD_NS::simd<T, NElts> vals, FlagsT flags = FlagsT{}) {
  lsc_slm_block_store<T, NElts, DS>(
      offset + __ESIMD_DNS::localAccessorToOffset(acc), vals);
}

/// A variation of lsc_block_store without predicate parameter to simplify
/// use of alignment parameter
///
/// Scatters elements to surface.
/// When \c DS equals \c lsc_data_size::u64 or \c sizeof(T) equal to 8 the
/// address must be 8-byte aligned, otherwise - 4-bytes aligned. Allowed values
/// for the data size are \c lsc_data_size::u32, \c lsc_data_size::u64,
/// \c lsc_data_size::u8, \c lsc_data_size::u16.
/// When data size is either  \c lsc_data_size::u8 or \c lsc_data_size::u16
/// the data is treated as 32 bit data.
/// Allowed \c NElts values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 8 bit data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
/// 8 bytes alignment is required for 64 bit data, 32 bit data and \c NElts
/// equal to 128, 16 bit data and \c NElts equal to 256, 8 bit data and \c
/// NElts equal to 512. Otherwise 4 bytes alignment is required.
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
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_write> &&
    __ESIMD_NS::is_simd_flag_type_v<FlagsT>>
lsc_block_store(AccessorTy acc, __ESIMD_DNS::DeviceAccessorOffsetT offset,
                __ESIMD_NS::simd<T, NElts> vals, FlagsT flags) {
  lsc_block_store<T, NElts, DS, L1H, L3H>(acc, offset, vals,
                                          __ESIMD_NS::simd_mask<1>(1), flags);
}

namespace detail {
#ifndef __ESIMD_DWORD_BLOCK_2D_WIDTH_SCALE
#define __ESIMD_DWORD_BLOCK_2D_WIDTH_SCALE (1)
#endif

#ifndef __ESIMD_BLOCK_2D_WIDTH_CHECK
#define __ESIMD_BLOCK_2D_WIDTH_CHECK(OP, BLOCK_WIDTH, NBLOCKS, SIZE)           \
  static_assert((BLOCK_WIDTH) * (NBLOCKS) * (SIZE) <= 64,                      \
                "Unsupported block width");
#endif

enum class block_2d_op { prefetch, load, store };

// Compile-time checks for lsc_load_2d/prefetch_2d/store_2d restrictions.
template <typename T, int BlockWidth, int BlockHeight, int NBlocks,
          bool Transposed, bool Transformed, block_2d_op Op>
constexpr void check_lsc_block_2d_restrictions() {
  constexpr int GRFByteSize = BlockWidth * BlockHeight * NBlocks * sizeof(T);
  static_assert(BlockWidth > 0, "Block width must be positive");
  static_assert(BlockHeight > 0, "Block height must be positive");
  // Restrictions based on documentation.
  if constexpr (Op == block_2d_op::store)
    static_assert(GRFByteSize <= 512, "2D store supports 512 bytes max");
  else
    static_assert(GRFByteSize <= 2048,
                  "2D load/prefetch supports 2048 bytes max");
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
    static_assert(sizeof(T) == 8
                      ? __ESIMD_DNS::isPowerOf2(BlockWidth, 4)
                      : BlockWidth >= 1 &&
                            BlockWidth <=
                                8 * __ESIMD_DWORD_BLOCK_2D_WIDTH_SCALE,
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
    if constexpr (Op == block_2d_op::store) {
      static_assert(NBlocks == 1, "Unsupported number of blocks for 2D store");
      static_assert(BlockHeight <= 8, "Unsupported block height for store");
    } else {
      static_assert(
          __ESIMD_DNS::isPowerOf2(NBlocks, sizeof(T) == 1 ? 4 : 8 / sizeof(T)),
          "Unsupported number of blocks for 2D load/prefetch");
      static_assert(BlockHeight <= 32, "Unsupported block height for load");
    }
    static_assert(BlockWidth * sizeof(T) >= 4, "Unsupported block width");
    __ESIMD_BLOCK_2D_WIDTH_CHECK(Op, BlockWidth, NBlocks, sizeof(T));
  }
}
#undef __ESIMD_DWORD_BLOCK_2D_WIDTH_SCALE
#undef __ESIMD_BLOCK_2D_WIDTH_CHECK

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
  using RawT = __ESIMD_DNS::__raw_t<T>;
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  detail::check_lsc_block_2d_restrictions<RawT, BlockWidth, BlockHeight,
                                          NBlocks, Transposed, Transformed,
                                          detail::block_2d_op::load>();
  // For Load BlockWidth is padded up to the next power-of-two value.
  // For Load with Transpose the pre-operation BlockHeight is padded up
  // to the next power-of-two value.
  // For Load with Transform pre-operation BlockHeight is padded up to
  // multiple of K, where K = 4B / sizeof(T).
  constexpr int ElemsPerDword = 4 / sizeof(RawT);
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
      detail::roundUpNextMultiple<64 / sizeof(RawT), GRFBlockSize>();
  constexpr int ActualN = NBlocks * GRFBlockPitch;

  constexpr int DstBlockElements = GRFColSize * GRFRowSize;
  constexpr int DstElements = DstBlockElements * NBlocks;

  static_assert(N == ActualN || N == DstElements, "Incorrect element count");

  constexpr lsc_data_size DS =
      detail::finalize_data_size<RawT, lsc_data_size::default_size>();
  __ESIMD_NS::simd_mask<ActualN> pred = 1;
  uintptr_t surf_addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr detail::lsc_data_order _Transposed =
      Transposed ? detail::lsc_data_order::transpose
                 : detail::lsc_data_order::nontranspose;
  __ESIMD_NS::simd<RawT, ActualN> Raw =
      __esimd_lsc_load2d_stateless<RawT, L1H, L3H, DS, _Transposed, NBlocks,
                                   BlockWidth, BlockHeight, Transformed,
                                   ActualN>(pred.data(), surf_addr,
                                            SurfaceWidth, SurfaceHeight,
                                            SurfacePitch, X, Y);

  if constexpr (ActualN == N) {
    return Raw;
  } else {
    // HW restrictions force data which is read to contain padding filled with
    // zeros for 2d lsc loads. This code eliminates such padding.

    // For example, 2D block load of 5 elements of 1 byte data type will
    // take 8 bytes per row for each block.
    //
    // +----+----+----+----+----+----+-----+-----+
    // | 00 | 01 | 02 | 03 | 04 | 05 | 06* | 07* |
    // +----+----+----+----+----+----+-----+-----+
    // | 10 | 11 | 12 | 13 | 14 | 15 | 16* | 17* |
    // +----+----+----+----+----+----+-----+-----+
    // | 20 | 21 | 22 | 23 | 24 | 25 | 26* | 27* |
    // +----+----+----+----+----+----+-----+-----+
    // | 30 | 31 | 32 | 33 | 34 | 35 | 36* | 37* |
    // +----+----+----+----+----+----+-----+-----+
    // * signifies the padded element.

    __ESIMD_NS::simd<RawT, DstElements> Dst;

    for (auto i = 0; i < NBlocks; i++) {
      auto DstBlock =
          Dst.template select<DstBlockElements, 1>(i * DstBlockElements);

      auto RawBlock = Raw.template select<GRFBlockSize, 1>(i * GRFBlockPitch);
      DstBlock =
          RawBlock.template bit_cast_view<RawT, GRFColSize, GRFRowPitch>()
              .template select<GRFColSize, 1, GRFRowSize, 1>(0, 0)
              .template bit_cast_view<RawT>();
    }

    return Dst;
  }
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
                                          false, false,
                                          detail::block_2d_op::prefetch>();
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
  using RawT = __ESIMD_DNS::__raw_t<T>;
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();
  detail::check_lsc_block_2d_restrictions<RawT, BlockWidth, BlockHeight, 1,
                                          false, false,
                                          detail::block_2d_op::store>();
  constexpr lsc_data_size DS =
      detail::finalize_data_size<RawT, lsc_data_size::default_size>();
  uintptr_t surf_addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;

  constexpr int Pitch = __ESIMD_DNS::getNextPowerOf2<BlockWidth>();
  __ESIMD_NS::simd<RawT, BlockHeight * Pitch> Raw;

  if constexpr (BlockHeight * Pitch == N) {
    Raw = Vals;
  } else {
    // For store with padding, allocate the block with padding, and place
    // original data there.
    auto Data2D = Vals.template bit_cast_view<RawT, BlockHeight, BlockWidth>();
    auto Raw2D = Raw.template bit_cast_view<RawT, BlockHeight, Pitch>();
    Raw2D.template select<BlockHeight, 1, BlockWidth, 1>(0, 0) = Data2D;
  }

  __ESIMD_NS::simd_mask<BlockHeight * Pitch> pred = 1;
  __esimd_lsc_store2d_stateless<RawT, L1H, L3H, DS, _Transposed, 1u, BlockWidth,
                                BlockHeight, false, BlockHeight * Pitch>(
      pred.data(), surf_addr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y,
      Raw.data());
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
    return (T *)((
        uint64_t)(const_cast<config_2d_mem_access *>(this)
                      ->payload_data.template bit_cast_view<uint64_t>()[0]));
  }

  /// <summary>
  /// Get surface width
  /// </summary>
  /// <returns>Surface Width</returns>
  uint32_t get_surface_width() const {
    return const_cast<config_2d_mem_access *>(this)
        ->payload_data.template select<1, 1>(2);
  }

  /// <summary>
  /// Get surface height
  /// </summary>
  /// <returns>Surface Height</returns>
  uint32_t get_surface_height() const {
    return const_cast<config_2d_mem_access *>(this)
        ->payload_data.template select<1, 1>(3);
  }

  /// <summary>
  /// Get surface pitch
  /// </summary>
  /// <returns>Surface Pitch</returns>
  uint32_t get_surface_pitch() const {
    return const_cast<config_2d_mem_access *>(this)
        ->payload_data.template select<1, 1>(4);
  }

  /// <summary>
  /// Get top left corner X coordinate of the block
  /// </summary>
  /// <returns>Top left corner X coordinate of the block</returns>
  int32_t get_x() const {
    return const_cast<config_2d_mem_access *>(this)
        ->payload_data.template select<1, 1>(5);
  }

  /// <summary>
  /// Get top left corner Y coordinate of the block
  /// </summary>
  /// <returns>Top left corner Y coordinate of the block</returns>
  int32_t get_y() const {
    return const_cast<config_2d_mem_access *>(this)
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
                                          Transposed, Transformed,
                                          detail::block_2d_op::load>();
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

  constexpr uint32_t GrfBytes = 64;
  constexpr uint32_t DstBlockSize =
      detail::roundUpNextMultiple<DstElements * sizeof(T), GrfBytes>();
  constexpr uint32_t DstLength =
      (DstBlockSize / GrfBytes) > 31 ? 31 : (DstBlockSize / GrfBytes);
  constexpr uint32_t DstLengthMask = DstLength << 20;

  static_assert(N == ActualN || N == DstElements, "Incorrect element count");

  constexpr uint32_t cache_mask = detail::get_lsc_load_cache_mask<L1H, L3H>()
                                  << 17;
  constexpr uint32_t base_desc = 0x2000003;
  constexpr uint32_t transformMask = Transformed ? 1 << 7 : 0;
  constexpr uint32_t transposeMask = Transposed ? 1 << 15 : 0;
  constexpr uint32_t dataSizeMask = detail::get_lsc_data_size<T>() << 9;
  __ESIMD_NS::simd<T, N> oldDst;
  constexpr uint32_t exDesc = 0x0;
  constexpr uint32_t desc = base_desc | cache_mask | transformMask |
                            transposeMask | dataSizeMask | DstLengthMask;
  constexpr uint8_t execSize = 1;
  constexpr uint8_t sfid = 0xF;
  constexpr uint8_t numSrc0 = 0x1;
  constexpr uint8_t numDst = (N * sizeof(T)) / 64;
  __ESIMD_NS::simd<T, ActualN> Raw =
      __ESIMD_NS::raw_send<execSize, sfid, numSrc0, numDst>(
          oldDst, payload.get_raw_data(), exDesc, desc);

  if constexpr (ActualN == N) {
    return Raw;
  } else {
    // HW restrictions force data which is read to contain padding filled with
    // zeros for 2d lsc loads. This code eliminates such padding.

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
                                          Transposed, Transformed,
                                          detail::block_2d_op::prefetch>();
  static_assert(!Transposed || !Transformed,
                "Transposed and transformed is not supported");
  constexpr uint32_t cache_mask = detail::get_lsc_load_cache_mask<L1H, L3H>()
                                  << 17;
  constexpr uint32_t dataSizeMask = detail::get_lsc_data_size<T>() << 9;
  constexpr uint32_t base_desc = 0x2000003;
  constexpr uint32_t transformMask = Transformed ? 1 << 7 : 0;
  constexpr uint32_t transposeMask = Transposed ? 1 << 15 : 0;
  constexpr uint32_t exDesc = 0x0;
  constexpr uint32_t desc =
      base_desc | cache_mask | transformMask | transposeMask | dataSizeMask;
  constexpr uint8_t execSize = 1;
  constexpr uint8_t sfid = 0xF;
  constexpr uint8_t numDst = (N * sizeof(T)) / 64;
  __ESIMD_NS::raw_send<execSize, sfid, numDst>(payload.get_raw_data(), exDesc,
                                               desc);
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
                                          false, false,
                                          detail::block_2d_op::store>();
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();

  constexpr uint32_t cache_mask = detail::get_lsc_store_cache_mask<L1H, L3H>()
                                  << 17;
  constexpr uint32_t dataSizeMask = detail::get_lsc_data_size<T>() << 9;
  constexpr uint32_t base_desc = 0x2000007;

  constexpr uint32_t exDesc = 0x0;
  constexpr uint32_t desc = base_desc | cache_mask | dataSizeMask;
  constexpr uint8_t execSize = 1;
  constexpr uint8_t sfid = 0xF;
  constexpr uint8_t numSrc0 = 0x1;
  constexpr uint8_t numSrc1 = (N * sizeof(T)) / 64;

  __ESIMD_NS::raw_sends<execSize, sfid, numSrc0, numSrc1>(
      payload.get_raw_data(), Data, exDesc, desc);
}

namespace detail {

// lsc_atomic_update() operations may share atomic_op values for data types
// of the same (fp vs integral) class for convenience (e.g. re-use 'fmax' for
// all FP types). In fact those data types may require using different internal
// opcodes. This function returns the corresponding internal opcode for
// the input type 'T' and operation 'Op'.
template <typename T, __ESIMD_NS::atomic_op Op>
constexpr int lsc_to_internal_atomic_op() {
  constexpr __ESIMD_NS::native::lsc::atomic_op LSCOp =
      __ESIMD_DNS::to_lsc_atomic_op<Op>();
  return static_cast<int>(LSCOp);
}
} // namespace detail

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
  return __ESIMD_DNS::slm_atomic_update_impl<Op, T, N, DS>(offsets, pred);
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
  return __ESIMD_DNS::slm_atomic_update_impl<Op, T, N, DS>(offsets, src0, pred);
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
/// @param src0 is the first atomic operand (expected value).
/// @param src1 is the second atomic operand (new value).
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
  return __ESIMD_DNS::slm_atomic_update_impl<Op, T, N, DS>(offsets, src0, src1,
                                                           pred);
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
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 0,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd<Toffset, N> offsets,
                  __ESIMD_NS::simd_mask<N> pred) {
  return __ESIMD_DNS::atomic_update_impl<Op, T, N, DS, L1H, L3H, Toffset>(
      p, offsets, pred);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 0,
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
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd<Toffset, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd_mask<N> pred) {
  return __ESIMD_DNS::atomic_update_impl<Op, T, N, DS, L1H, L3H, Toffset>(
      p, offsets, src0, pred);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename OffsetObjT, typename RegionTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd_view<OffsetObjT, RegionTy> offsets,
                  __ESIMD_NS::simd<T, N> src0,
                  __ESIMD_NS::simd_mask<N> pred = 1) {
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(p, offsets.read(), src0,
                                                   pred);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 1 &&
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
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd<Toffset, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                  __ESIMD_NS::simd_mask<N> pred) {
  return __ESIMD_DNS::atomic_update_impl<Op, T, N, DS, L1H, L3H, Toffset>(
      p, offsets, src0, src1, pred);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename OffsetObjT, typename RegionTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(T *p, __ESIMD_NS::simd_view<OffsetObjT, RegionTy> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                  __ESIMD_NS::simd_mask<N> pred = 1) {
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(p, offsets.read(), src0,
                                                   src1, pred);
}

template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 2,
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
          typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_device_accessor_with_v<
        AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_read> &&
        (Op == __ESIMD_NS::atomic_op::load ||
         __ESIMD_DNS::is_device_accessor_with_v<
             AccessorTy, __ESIMD_DNS::accessor_mode_cap::can_write>),
    __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<Toffset, N> offsets,
                  __ESIMD_NS::simd_mask<N> pred) {
  return __ESIMD_DNS::atomic_update_impl<Op, T, N, DS, L1H, L3H>(acc, offsets,
                                                                 pred);
}

/// Variant of \c lsc_atomic_update that uses \c local_accessor as a parameter.
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
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_rw_local_accessor_v<AccessorTy>,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd_mask<N> pred) {
  return lsc_slm_atomic_update<Op, T, N, DS>(
      offsets + __ESIMD_DNS::localAccessorToOffset(acc), pred);
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
          typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_rw_device_accessor_v<AccessorTy>,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<Toffset, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd_mask<N> pred) {
  return __ESIMD_DNS::atomic_update_impl<Op, T, N, DS, L1H, L3H>(acc, offsets,
                                                                 src0, pred);
}

/// Variant of \c lsc_atomic_update that uses \c local_accessor as a parameter.
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
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_rw_local_accessor_v<AccessorTy>,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd_mask<N> pred) {
  return lsc_slm_atomic_update<Op, T, N, DS>(
      offsets + __ESIMD_DNS::localAccessorToOffset(acc), src0, pred);
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
/// @param src0 is the first atomic operand (expected value).
/// @param src1 is the second atomic operand (new value).
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_rw_device_accessor_v<AccessorTy>,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<Toffset, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                  __ESIMD_NS::simd_mask<N> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return lsc_atomic_update<Op, T, N, DS, L1H, L3H>(
      __ESIMD_DNS::accessorToPointer<T>(acc), offsets, src0, src1, pred);
#else
  static_assert(std::is_integral_v<Toffset> && sizeof(Toffset) == 4,
                "Unsupported offset type");
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  __ESIMD_DNS::check_atomic<Op, T, N, 2>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using MsgT = typename detail::lsc_expand_type<T>::type;
  constexpr int IOp = detail::lsc_to_internal_atomic_op<T, Op>();
  __ESIMD_NS::simd<MsgT, N> Msg_data0 = detail::lsc_format_input<MsgT>(src0);
  __ESIMD_NS::simd<MsgT, N> Msg_data1 = detail::lsc_format_input<MsgT>(src1);
  auto si = __ESIMD_NS::get_surface_index(acc);
  __ESIMD_NS::simd<MsgT, N> Tmp =
      __esimd_lsc_xatomic_bti_2<MsgT, IOp, L1H, L3H, _AddressScale, _ImmOffset,
                                _DS, _VS, _Transposed, N>(
          pred.data(), offsets.data(), Msg_data0.data(), Msg_data1.data(), si);
  return detail::lsc_format_ret<T>(Tmp);
#endif
}

/// Variant of \c lsc_atomic_update that uses \c local_accessor as a parameter.
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
/// @param src0 is the first atomic operand (expected value).
/// @param src1 is the second atomic operand (new value).
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_rw_local_accessor_v<AccessorTy>,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                  __ESIMD_NS::simd_mask<N> pred) {
  return lsc_slm_atomic_update<Op, T, N, DS>(
      offsets + __ESIMD_DNS::localAccessorToOffset(acc), src0, src1, pred);
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
__SYCL_DEPRECATED("use sycl::ext::intel::esimd::fence<Kind, FenceOp, Scope>()")
__ESIMD_API void lsc_fence(__ESIMD_NS::simd_mask<N> pred = 1) {
  static_assert(
      Kind != lsc_memory_kind::shared_local ||
          (FenceOp == lsc_fence_op::none && Scope == lsc_scope::group),
      "SLM fence must have 'none' lsc_fence_op and 'group' scope");
  static_assert(Kind != lsc_memory_kind::untyped_global_low_pri,
                "lsc_memory_kind::untyped_global_low_pri is not supported in HW"
                " and/or GPU drivers");
  __esimd_lsc_fence<static_cast<uint8_t>(Kind), static_cast<uint8_t>(FenceOp),
                    static_cast<uint8_t>(Scope), N>(pred.data());
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
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 0,
                             simd<T, N>>
atomic_update(T *p, simd<Toffset, N> offset, simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offset, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename OffsetObjT,
          typename RegionTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 0, simd<T, N>>
atomic_update(T *p, simd_view<OffsetObjT, RegionTy> offsets,
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
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 1,
                             simd<T, N>>
atomic_update(T *p, simd<Toffset, N> offset, simd<T, N> src0,
              simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offset, src0, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename OffsetObjT,
          typename RegionTy>
__ESIMD_API __ESIMD_API
    std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1, simd<T, N>>
    atomic_update(T *p, simd_view<OffsetObjT, RegionTy> offsets,
                  simd<T, N> src0, simd_mask<N> mask = 1) {
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
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 2,
                             simd<T, N>>
atomic_update(T *p, simd<Toffset, N> offset, simd<T, N> src0, simd<T, N> src1,
              simd_mask<N> mask) {
  // 2-argument lsc_atomic_update arguments order matches the standard one -
  // expected value first, then new value. But atomic_update uses reverse
  // order, hence the src1/src0 swap.
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      p, offset, src1, src0, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename OffsetObjT,
          typename RegionTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2, simd<T, N>>
atomic_update(T *p, simd_view<OffsetObjT, RegionTy> offsets, simd<T, N> src0,
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

template <native::lsc::atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 0 &&
                                 __ESIMD_DNS::is_rw_accessor_v<AccessorTy>,
                             simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> offset, simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      acc, offset, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename OffsetObjT,
          typename RegionTy, typename AccessorTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 0 &&
                                 __ESIMD_DNS::is_rw_accessor_v<AccessorTy>,
                             simd<T, N>>
atomic_update(AccessorTy acc, simd_view<OffsetObjT, RegionTy> offsets,
              simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      acc, offsets, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 0 &&
                                 __ESIMD_DNS::is_rw_accessor_v<AccessorTy>,
                             simd<T, N>>
atomic_update(AccessorTy acc, Toffset offset, simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      acc, offset, mask);
}

/// LSC version of the single-argument atomic update.
template <native::lsc::atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 1 &&
                                 __ESIMD_DNS::is_rw_accessor_v<AccessorTy>,
                             simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> offset, simd<T, N> src0,
              simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      acc, offset, src0, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename OffsetObjT,
          typename RegionTy, typename AccessorTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1 &&
                                 __ESIMD_DNS::is_rw_accessor_v<AccessorTy>,
                             simd<T, N>>
atomic_update(AccessorTy acc, simd_view<OffsetObjT, RegionTy> offsets,
              simd<T, N> src0, simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      acc, offsets, src0, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 1 &&
                                 __ESIMD_DNS::is_rw_accessor_v<AccessorTy>,
                             simd<T, N>>
atomic_update(AccessorTy acc, Toffset offset, simd<T, N> src0,
              simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      acc, offset, src0, mask);
}

/// LSC version of the two-argument atomic update.
template <native::lsc::atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 __ESIMD_DNS::is_rw_accessor_v<AccessorTy>,
                             simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> offset, simd<T, N> src0,
              simd<T, N> src1, simd_mask<N> mask) {
  // 2-argument lsc_atomic_update arguments order matches the standard one -
  // expected value first, then new value. But atomic_update uses reverse
  // order, hence the src1/src0 swap.
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      acc, offset, src1, src0, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename OffsetObjT,
          typename RegionTy, typename AccessorTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 __ESIMD_DNS::is_rw_accessor_v<AccessorTy>,
                             simd<T, N>>
atomic_update(AccessorTy acc, simd_view<OffsetObjT, RegionTy> offsets,
              simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      acc, offsets, src1, src0, mask);
}

template <native::lsc::atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 __ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 __ESIMD_DNS::is_rw_accessor_v<AccessorTy>,
                             __ESIMD_NS::simd<T, N>>
atomic_update(AccessorTy acc, Toffset offset, simd<T, N> src0, simd<T, N> src1,
              simd_mask<N> mask) {
  return __ESIMD_ENS::lsc_atomic_update<detail::to_atomic_op<Op>(), T, N>(
      acc, offset, src1, src0, mask);
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
  ESIMD_INLINE int get_offset() const { return offset; }

  /// Releases the SLM chunk allocated in the constructor.
  ~slm_allocator() { __esimd_slm_free(offset); }
};

} // namespace esimd
} // namespace ext::intel
} // namespace _V1
} // namespace sycl
