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

#include <sycl/ext/intel/esimd/memory.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/util.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

#define __ESIMD_GET_SURF_HANDLE(acc) __ESIMD_NS::get_surface_index(acc)

/// @addtogroup sycl_esimd_memory
/// @{

/// Generic work-group split barrier
__ESIMD_API void sbarrier(split_barrier_action flag) { __esimd_sbarrier(flag); }

/// @} sycl_esimd_memory

/// @addtogroup sycl_esimd_raw_send
/// @{

/// Raw sends load.  "s" suffix designates "split" variant - i.e. two sources.
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
__ESIMD_API __ESIMD_NS::simd<T1, n1> raw_sends_load(
    __ESIMD_NS::simd<T1, n1> msgDst, __ESIMD_NS::simd<T2, n2> msgSrc0,
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

  uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  return __esimd_raw_sends2<T1, n1, T2, n2, T3, n3, N>(
      modifier, execSize, mask.data(), numSrc0, numSrc1, numDst, sfid, exDesc,
      msgDesc, msgSrc0.data(), msgSrc1.data(), msgDst.data());
}

/// Raw send load.
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
raw_send_load(__ESIMD_NS::simd<T1, n1> msgDst, __ESIMD_NS::simd<T2, n2> msgSrc0,
              uint32_t exDesc, uint32_t msgDesc, uint8_t execSize, uint8_t sfid,
              uint8_t numSrc0, uint8_t numDst, uint8_t isEOT = 0,
              uint8_t isSendc = 0, __ESIMD_NS::simd_mask<N> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send rspVar");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msgSrc0");

  uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  return __esimd_raw_send2<T1, n1, T2, n2, N>(
      modifier, execSize, mask.data(), numSrc0, numDst, sfid, exDesc, msgDesc,
      msgSrc0.data(), msgDst.data());
}

/// Raw sends store. "s" suffix designates "split" variant - i.e. two sources.
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
raw_sends_store(__ESIMD_NS::simd<T1, n1> msgSrc0,
                __ESIMD_NS::simd<T2, n2> msgSrc1, uint32_t exDesc,
                uint32_t msgDesc, uint8_t execSize, uint8_t sfid,
                uint8_t numSrc0, uint8_t numSrc1, uint8_t isEOT = 0,
                uint8_t isSendc = 0, __ESIMD_NS::simd_mask<N> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send msgSrc0");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msgSrc1");

  uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  __esimd_raw_sends2_noresult<T1, n1, T2, n2, N>(
      modifier, execSize, mask.data(), numSrc0, numSrc1, sfid, exDesc, msgDesc,
      msgSrc0.data(), msgSrc1.data());
}

/// Raw send store. Generates a \c send or \c sendc instruction for the message
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
__ESIMD_API void raw_send_store(__ESIMD_NS::simd<T1, n1> msgSrc0,
                                uint32_t exDesc, uint32_t msgDesc,
                                uint8_t execSize, uint8_t sfid, uint8_t numSrc0,
                                uint8_t isEOT = 0, uint8_t isSendc = 0,
                                __ESIMD_NS::simd_mask<N> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send msgSrc0");

  uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  __esimd_raw_send2_noresult<T1, n1, N>(modifier, execSize, mask.data(),
                                        numSrc0, sfid, exDesc, msgDesc,
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
__ESIMD_API void nbarrier_wait(uint8_t id) {
  __esimd_nbarrier(0 /*wait*/, id, 0 /*thread count*/);
}

/// Initialize number of named barriers for a kernel
/// Available only on PVC
///
/// @tparam NbarCount  - number of named barriers
template <uint8_t NbarCount> __ESIMD_API void nbarrier_init() {
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
__ESIMD_API void nbarrier_signal(uint8_t barrier_id,
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

// Format u8u32 and u16u32 back to u8 and u16.
template <typename T, typename T1, int N>
ESIMD_INLINE __ESIMD_NS::simd<T, N>
lsc_format_ret(__ESIMD_NS::simd<T1, N> Vals) {
  auto Formatted = Vals.template bit_cast_view<T>();
  constexpr int Stride = Formatted.length / N;
  return Formatted.template select<N, Stride>(0);
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
template <typename T, uint8_t NElts = 1,
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
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<_MsgT, N *NElts> Tmp =
      __esimd_lsc_load_slm<_MsgT, cache_hint::none, cache_hint::none,
                           _AddressScale, _ImmOffset, _DS, _VS, _Transposed, N>(
          pred.data(), offsets.data());
  return detail::lsc_format_ret<T>(Tmp);
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
/// @return is a vector of type T and size NElts
///
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, NElts> lsc_slm_block_load(uint32_t offset) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  static_assert(_DS == lsc_data_size::u32 || _DS == lsc_data_size::u64,
                "Transposed load is supported only for data size u32 or u64");
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd_mask<N> pred = 1;
  __ESIMD_NS::simd<uint32_t, N> offsets = offset;
  return __esimd_lsc_load_slm<T, cache_hint::none, cache_hint::none,
                              _AddressScale, _ImmOffset, _DS, _VS, _Transposed,
                              N>(pred.data(), offsets.data());
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
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value,
                             __ESIMD_NS::simd<T, N * NElts>>
lsc_gather(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
           __ESIMD_NS::simd_mask<N> pred = 1) {
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
  auto si = __ESIMD_GET_SURF_HANDLE(acc);
  __ESIMD_NS::simd<_MsgT, N *NElts> Tmp =
      __esimd_lsc_load_bti<_MsgT, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(), si);
  return detail::lsc_format_ret<T>(Tmp);
}

/// Accessor-based transposed gather with 1 channel.
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
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @return is a vector of type T and size NElts
///
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value,
                             __ESIMD_NS::simd<T, NElts>>
lsc_block_load(AccessorTy acc, uint32_t offset) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  static_assert(_DS == lsc_data_size::u32 || _DS == lsc_data_size::u64,
                "Transposed load is supported only for data size u32 or u64");
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd_mask<N> pred = 1;
  __ESIMD_NS::simd<uint32_t, N> offsets = offset;
  auto si = __ESIMD_GET_SURF_HANDLE(acc);
  return __esimd_lsc_load_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                              _Transposed, N>(pred.data(), offsets.data(), si);
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
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
lsc_gather(const T *p, __ESIMD_NS::simd<uint32_t, N> offsets,
           __ESIMD_NS::simd_mask<N> pred = 1) {
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
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __ESIMD_NS::simd<_MsgT, N *NElts> Tmp =
      __esimd_lsc_load_stateless<_MsgT, L1H, L3H, _AddressScale, _ImmOffset,
                                 _DS, _VS, _Transposed, N>(pred.data(),
                                                           addrs.data());
  return detail::lsc_format_ret<T>(Tmp);
}

/// USM pointer transposed gather with 1 channel.
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
/// @param p is the base pointer.
/// @return is a vector of type T and size NElts
///
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__ESIMD_API __ESIMD_NS::simd<T, NElts> lsc_block_load(const T *p) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  static_assert(_DS == lsc_data_size::u32 || _DS == lsc_data_size::u64,
                "Transposed load is supported only for data size u32 or u64");
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd_mask<N> pred = 1;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  return __esimd_lsc_load_stateless<T, L1H, L3H, _AddressScale, _ImmOffset, _DS,
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
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value>
lsc_prefetch(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
             __ESIMD_NS::simd_mask<N> pred = 1) {
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
  auto si = __ESIMD_GET_SURF_HANDLE(acc);
  __esimd_lsc_prefetch_bti<_MsgT, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(), si);
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
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value>
lsc_prefetch(AccessorTy acc, uint32_t offset) {
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
  auto si = __ESIMD_GET_SURF_HANDLE(acc);
  __esimd_lsc_prefetch_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(), si);
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
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N>
__ESIMD_API void lsc_prefetch(const T *p, __ESIMD_NS::simd<uint32_t, N> offsets,
                              __ESIMD_NS::simd_mask<N> pred = 1) {
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
template <typename T, uint8_t NElts = 1,
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
template <typename T, uint8_t NElts = 1,
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
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size>
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
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value>
lsc_scatter(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
            __ESIMD_NS::simd<T, N * NElts> vals,
            __ESIMD_NS::simd_mask<N> pred = 1) {
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
  auto si = __ESIMD_GET_SURF_HANDLE(acc);
  __esimd_lsc_store_bti<_MsgT, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                        _Transposed, N>(pred.data(), offsets.data(), Tmp.data(),
                                        si);
}

/// Accessor-based transposed scatter with 1 channel.
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
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param vals is values to store.
///
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value>
lsc_block_store(AccessorTy acc, uint32_t offset,
                __ESIMD_NS::simd<T, NElts> vals) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();
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
  auto si = __ESIMD_GET_SURF_HANDLE(acc);
  __esimd_lsc_store_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                        _Transposed, N>(pred.data(), offsets.data(),
                                        vals.data(), si);
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
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N>
__ESIMD_API void lsc_scatter(T *p, __ESIMD_NS::simd<uint32_t, N> offsets,
                             __ESIMD_NS::simd<T, N * NElts> vals,
                             __ESIMD_NS::simd_mask<N> pred = 1) {
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

/// USM pointer transposed scatter with 1 channel.
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
/// @param p is the base pointer.
/// @param vals is values to store.
///
template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__ESIMD_API void lsc_block_store(T *p, __ESIMD_NS::simd<T, NElts> vals) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();
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
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  __esimd_lsc_store_stateless<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                              _Transposed, N>(pred.data(), addrs.data(),
                                              vals.data());
}

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
lsc_load2d(const T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
           unsigned SurfacePitch, int X, int Y) {
  static_assert(!Transposed || !Transformed,
                "Transposed and transformed is not supported");
  static_assert(!Transposed || (Transposed && NBlocks == 1),
                "Transposed expected to be 1 block only");
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr int ElemsPerDword = 4 / sizeof(T);
  constexpr int GRFRowSize = Transposed ? BlockHeight : BlockWidth;
  constexpr int GRFRowPitch = __ESIMD_DNS::getNextPowerOf2<GRFRowSize>();
  constexpr int GRFBlockSize =
      GRFRowPitch * (Transposed ? BlockWidth : BlockHeight);
  constexpr int GRFBlockPitch =
      detail::roundUpNextMultiple<64 / sizeof(T), GRFBlockSize>();
  constexpr int ActualN = NBlocks * GRFBlockPitch;
  static_assert(
      ActualN == N,
      "These parameters require unpadding. It is not implemented yet");
  constexpr lsc_data_size DS =
      detail::finalize_data_size<T, lsc_data_size::default_size>();
  static_assert(!Transformed ||
                    (DS == lsc_data_size::u8 || DS == lsc_data_size::u16),
                "VNNI transform is supported only for data size U8 or U16");
  static_assert(!Transposed ||
                    (DS == lsc_data_size::u32 || DS == lsc_data_size::u64),
                "Transposed load is supported only for data size u32 or u64");
  __ESIMD_NS::simd_mask<N> pred = 1;
  uintptr_t surf_addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr detail::lsc_data_order _Transposed =
      Transposed ? detail::lsc_data_order::transpose
                 : detail::lsc_data_order::nontranspose;
  return __esimd_lsc_load2d_stateless<T, L1H, L3H, DS, _Transposed, NBlocks,
                                      BlockWidth, BlockHeight, Transformed, N>(
      pred.data(), surf_addr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y);
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
__ESIMD_API void lsc_prefetch2d(const T *Ptr, unsigned SurfaceWidth,
                                unsigned SurfaceHeight, unsigned SurfacePitch,
                                int X, int Y) {
  detail::check_lsc_cache_hint<detail::lsc_action::prefetch, L1H, L3H>();
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
__ESIMD_API void lsc_store2d(T *Ptr, unsigned SurfaceWidth,
                             unsigned SurfaceHeight, unsigned SurfacePitch,
                             int X, int Y, __ESIMD_NS::simd<T, N> Vals) {
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();
  constexpr lsc_data_size DS =
      detail::finalize_data_size<T, lsc_data_size::default_size>();
  __ESIMD_NS::simd_mask<N> pred = 1;
  uintptr_t surf_addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  __esimd_lsc_store2d_stateless<T, L1H, L3H, DS, _Transposed, 1u, BlockWidth,
                                BlockHeight, false, N>(
      pred.data(), surf_addr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y,
      Vals.data());
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
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, N>
lsc_slm_atomic_update(__ESIMD_NS::simd<uint32_t, N> offsets,
                      __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 0>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
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
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, N>
lsc_slm_atomic_update(__ESIMD_NS::simd<uint32_t, N> offsets,
                      __ESIMD_NS::simd<T, N> src0,
                      __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 1>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
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
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size>
__ESIMD_API __ESIMD_NS::simd<T, N>
lsc_slm_atomic_update(__ESIMD_NS::simd<uint32_t, N> offsets,
                      __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                      __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 2>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_slm_2<_MsgT, _Op, cache_hint::none, cache_hint::none,
                                _AddressScale, _ImmOffset, _DS, _VS,
                                _Transposed, N>(pred.data(), offsets.data(),
                                                src0.data(), src1.data());
  return detail::lsc_format_ret<T>(Tmp);
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
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 0>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  auto si = __ESIMD_GET_SURF_HANDLE(acc);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_bti_0<_MsgT, _Op, L1H, L3H, _AddressScale, _ImmOffset,
                                _DS, _VS, _Transposed, N>(pred.data(),
                                                          offsets.data(), si);
  return detail::lsc_format_ret<T>(Tmp);
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
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 1>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  auto si = __ESIMD_GET_SURF_HANDLE(acc);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_bti_1<_MsgT, _Op, L1H, L3H, _AddressScale, _ImmOffset,
                                _DS, _VS, _Transposed, N>(
          pred.data(), offsets.data(), src0.data(), si);
  return detail::lsc_format_ret<T>(Tmp);
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
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value,
                             __ESIMD_NS::simd<T, N>>
lsc_atomic_update(AccessorTy acc, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                  __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 2>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  auto si = __ESIMD_GET_SURF_HANDLE(acc);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_bti_2<_MsgT, _Op, L1H, L3H, _AddressScale, _ImmOffset,
                                _DS, _VS, _Transposed, N>(
          pred.data(), offsets.data(), src0.data(), src1.data(), si);
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
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__ESIMD_API __ESIMD_NS::simd<T, N>
lsc_atomic_update(T *p, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 0>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_stateless_0<_MsgT, _Op, L1H, L3H, _AddressScale,
                                      _ImmOffset, _DS, _VS, _Transposed, N>(
          pred.data(), addrs.data());
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
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__ESIMD_API __ESIMD_NS::simd<T, N>
lsc_atomic_update(T *p, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 1>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_stateless_1<_MsgT, _Op, L1H, L3H, _AddressScale,
                                      _ImmOffset, _DS, _VS, _Transposed, N>(
          pred.data(), addrs.data(), src0.data());
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
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
/// @param pred is predicates.
///
template <__ESIMD_NS::atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__ESIMD_API __ESIMD_NS::simd<T, N>
lsc_atomic_update(T *p, __ESIMD_NS::simd<uint32_t, N> offsets,
                  __ESIMD_NS::simd<T, N> src0, __ESIMD_NS::simd<T, N> src1,
                  __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 2>();
  detail::check_lsc_cache_hint<detail::lsc_action::atomic, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  using _MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __ESIMD_NS::simd<_MsgT, N> Tmp =
      __esimd_lsc_xatomic_stateless_2<_MsgT, _Op, L1H, L3H, _AddressScale,
                                      _ImmOffset, _DS, _VS, _Transposed, N>(
          pred.data(), addrs.data(), src0.data(), src1.data());
  return detail::lsc_format_ret<T>(Tmp);
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

#undef __ESIMD_GET_SURF_HANDLE

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
