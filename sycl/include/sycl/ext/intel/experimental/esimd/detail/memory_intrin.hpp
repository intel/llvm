//==------------ memory_intrin.hpp - DPC++ Explicit SIMD API ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declares experimental memory Explicit SIMD intrinsics.
//===----------------------------------------------------------------------===//

/// @cond ESIMD_DETAIL

#pragma once

#include <sycl/ext/intel/esimd/detail/memory_intrin.hpp>

// generic work-group split barrier
__ESIMD_INTRIN void __esimd_sbarrier(__ESIMD_ENS::split_barrier_action flag)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  sycl::detail::getESIMDDeviceInterface()->cm_sbarrier_ptr((uint32_t)flag);
}
#endif // __SYCL_DEVICE_ONLY__

// \brief Raw sends load.
//
// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
//
// @param execSize the execution size, which must be a compile time constant.
//
// @param pred the predicate to specify enabled channels.
//
// @param numSrc0 the number of GRFs for source-0, which must be a compile time
// constant.
//
// @param numSrc1 the number of GRFs for source-1, which must be a compile time
// constant.
//
// @param numDst the number of GRFs for destination, which must be a compile
// time constant.
//
// @param sfid the shared function ID, which must be a compile time constant.
//
// @param exDesc the extended message descriptor.
//
// @param msgDesc the message descriptor.
//
// @param msgSrc0 the first source operand of send message.
//
// @param msgSrc1 the second source operand of send message.
//
// @param msgDst the destination operand of send message.
//
// Returns a simd vector of type Ty1 and size N1.
//
template <typename Ty1, int N1, typename Ty2, int N2, typename Ty3, int N3,
          int N = 16>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty1, N1>
__esimd_raw_sends2(uint8_t modifier, uint8_t execSize,
                   __ESIMD_DNS::simd_mask_storage_t<N> pred, uint8_t numSrc0,
                   uint8_t numSrc1, uint8_t numDst, uint8_t sfid,
                   uint32_t exDesc, uint32_t msgDesc,
                   __ESIMD_DNS::vector_type_t<Ty2, N2> msgSrc0,
                   __ESIMD_DNS::vector_type_t<Ty3, N3> msgSrc1,
                   __ESIMD_DNS::vector_type_t<Ty1, N1> msgDst)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

// \brief Raw send load.
//
// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
//
// @param execSize the execution size, which must be a compile time constant.
//
// @param pred the predicate to specify enabled channels.
//
// @param numSrc0 the number of GRFs for source-0, which must be a compile time
// constant.
//
// @param numDst the number of GRFs for destination, which must be a compile
// time constant.
//
// @param sfid the shared function ID, which must be a compile time constant.
//
// @param exDesc the extended message descriptor.
//
// @param msgDesc the message descriptor.
//
// @param msgSrc0 the first source operand of send message.
//
// @param msgDst the destination operand of send message.
//
// Returns a simd vector of type Ty1 and size N1.
//
template <typename Ty1, int N1, typename Ty2, int N2, int N = 16>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty1, N1>
__esimd_raw_send2(uint8_t modifier, uint8_t execSize,
                  __ESIMD_DNS::simd_mask_storage_t<N> pred, uint8_t numSrc0,
                  uint8_t numDst, uint8_t sfid, uint32_t exDesc,
                  uint32_t msgDesc, __ESIMD_DNS::vector_type_t<Ty2, N2> msgSrc0,
                  __ESIMD_DNS::vector_type_t<Ty1, N1> msgDst)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

// \brief Raw sends store.
//
// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
//
// @param execSize the execution size, which must be a compile time constant.
//
// @param pred the predicate to specify enabled channels.
//
// @param numSrc0 the number of GRFs for source-0, which must be a compile time
// constant.
//
// @param numSrc1 the number of GRFs for source-1, which must be a compile time
// constant.
//
// @param sfid the shared function ID, which must be a compile time constant.
//
// @param exDesc the extended message descriptor.
//
// @param msgDesc the message descriptor.
//
// @param msgSrc0 the first source operand of send message.
//
// @param msgSrc1 the second source operand of send message.
//
template <typename Ty1, int N1, typename Ty2, int N2, int N = 16>
__ESIMD_INTRIN void
__esimd_raw_sends2_noresult(uint8_t modifier, uint8_t execSize,
                            __ESIMD_DNS::simd_mask_storage_t<N> pred,
                            uint8_t numSrc0, uint8_t numSrc1, uint8_t sfid,
                            uint32_t exDesc, uint32_t msgDesc,
                            __ESIMD_DNS::vector_type_t<Ty1, N1> msgSrc0,
                            __ESIMD_DNS::vector_type_t<Ty2, N2> msgSrc1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

// \brief Raw send store.
//
// @param modifier	the send message flags (Bit-0: isSendc, Bit-1: isEOT).
//
// @param execSize the execution size, which must be a compile time constant.
//
// @param pred the predicate to specify enabled channels.
//
// @param numSrc0 the number of GRFs for source-0, which must be a compile time
// constant.
//
// @param sfid the shared function ID, which must be a compile time constant.
//
// @param exDesc the extended message descriptor.
//
// @param msgDesc the message descriptor.
//
// @param msgSrc0 the first source operand of send message.
//
template <typename Ty1, int N1, int N = 16>
__ESIMD_INTRIN void
__esimd_raw_send2_noresult(uint8_t modifier, uint8_t execSize,
                           __ESIMD_DNS::simd_mask_storage_t<N> pred,
                           uint8_t numSrc0, uint8_t sfid, uint32_t exDesc,
                           uint32_t msgDesc,
                           __ESIMD_DNS::vector_type_t<Ty1, N1> msgSrc0)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// Represents named barrier synchronization for a subgroup of threads.
/// Available only on PVC
///
/// @param mode  - is wait(0) or signal(1)
///
/// @param id  - barrier id
///
/// @param thread_count  - number of threads, ignored in 'wait' mode
__ESIMD_INTRIN void __esimd_nbarrier(uint8_t mode, uint8_t id,
                                     uint8_t thread_count)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// Initialize number of named barriers for a kernel
/// Available only on PVC
///
/// @param count  - number of named barriers
__ESIMD_INTRIN void __esimd_nbarrier_init(uint8_t count)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// Raw send signal to perform signal operation on named barriers
/// Available only on PVC
/// @tparam Ty  - message element type
///
/// @tparam N  - message length
///
/// @param is_sendc  - is sendc
///
/// @param extended_descriptor  - extended message descriptor
///
/// @param descriptor  - message descriptor
///
/// @param msg_var  - source operand of send message
///
/// @param pred  - predicate for enabled channels
template <typename Ty, int N>
__ESIMD_INTRIN void __esimd_raw_send_nbarrier_signal(
    uint32_t is_sendc, uint32_t extended_descriptor, uint32_t descriptor,
    __ESIMD_DNS::vector_type_t<Ty, N> msg_var, uint16_t pred = 1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// SLM gather.
/// Supported platforms: DG2, PVC
///
/// Collects elements located at slm and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param offsets is the zero-based offsets for SLM buffer in bytes.
/// @return is a vector of type T and size N * to_int<VS>()
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_load_slm(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                     __ESIMD_DNS::vector_type_t<uint32_t, N> offsets)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Surface-based gather.
/// Supported platforms: DG2, PVC
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets in bytes.
/// @param surf_ind is the surface index.
/// @return is a vector of type T and N * to_int<VS>()
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N,
          typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_load_bti(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                     __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
                     SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer gather.
/// Supported platforms: DG2, PVC
///
/// Collects elements located at specified address and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the load addresses.
/// @return is a vector of type T and N * to_int<VS>()
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_load_stateless(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                           __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Surface-based prefetch gather.
/// Supported platforms: DG2, PVC
///
/// Prefetches elements located at surface.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets in bytes.
/// @param surf_ind is the surface index.
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N,
          typename SurfIndAliasTy>
__ESIMD_INTRIN void
__esimd_lsc_prefetch_bti(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                         __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
                         SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer prefetch gather.
/// Supported platforms: DG2, PVC
///
/// Prefetches elements located at specified address.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN void
__esimd_lsc_prefetch_stateless(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                               __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// SLM scatter.
/// Supported platforms: DG2, PVC
///
/// Scatters elements located to slm.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param offsets is the zero-based offsets for SLM buffer in bytes.
/// @param vals is values to store.
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN void __esimd_lsc_store_slm(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// Surface-based scatter.
/// Supported platforms: DG2, PVC
///
/// Scatters elements to surface.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets in bytes.
/// @param vals is values to store.
/// @param surf_ind is the surface index.
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N,
          typename SurfIndAliasTy>
__ESIMD_INTRIN void __esimd_lsc_store_bti(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> vals,
    SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer scatter.
/// Supported platforms: DG2, PVC
///
/// Scatters elements to specific address.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
/// @param vals is values to store.
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN void __esimd_lsc_store_stateless(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// 2D USM pointer block load.
/// Supported platforms: PVC
///
/// Collects elements located at specified address and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam DS is the data size.
/// @tparam Transposed is the transposed version or not.
/// @tparam NBlocks is the number of blocks.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam Transformed is apply VNNI transform or not.
/// @tparam N is the data size
/// @param Pred is predicates.
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
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_data_order _Transposed, uint8_t NBlocks,
          int BlockWidth, int BlockHeight, bool Transformed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_lsc_load2d_stateless(__ESIMD_DNS::simd_mask_storage_t<N> Pred,
                             uintptr_t Ptr, int SurfaceWidth, int SurfaceHeight,
                             int SurfacePitch, int X, int Y)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// 2D USM pointer block prefetch.
/// Supported platforms: PVC
///
/// Prefetches elements located at specified address.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam DS is the data size.
/// @tparam NBlocks is the number of blocks.
/// @tparam Transposed is the transposed version or not.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam Transformed is apply VNNI transform or not.
/// @tparam N is the data size
/// @param Pred is predicates.
/// @param Ptr is the surface base address for this operation.
/// @param SurfaceWidth is the surface width minus 1 in bytes
/// @param SurfaceHeight is the surface height minus 1 in rows
/// @param SurfacePitch is the surface pitch minus 1 in bytes
/// @param X is zero based X-coordinate of the left upper rectangle corner in
/// number of elements.
/// @param Y is zero based Y-coordinate of the left upper rectangle corner in
/// rows.
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_data_order _Transposed, uint8_t NBlocks,
          int BlockWidth, int BlockHeight, bool Transformed, int N>
__ESIMD_INTRIN void __esimd_lsc_prefetch2d_stateless(
    __ESIMD_DNS::simd_mask_storage_t<N> Pred, uintptr_t Ptr, int SurfaceWidth,
    int SurfaceHeight, int SurfacePitch, int X, int Y)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// 2D USM pointer block store.
/// Supported platforms: PVC
///
/// Stores elements at specified address.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam DS is the data size.
/// @tparam Transposed is the transposed version or not.
/// @tparam NBlocks is the number of blocks.
/// @tparam BlockWidth is the block width in number of elements.
/// @tparam BlockHeight is the block height in number of elements.
/// @tparam Transformed is apply VNNI transform or not.
/// @tparam N is the data size
/// @param Pred is predicates.
/// @param Ptr is the surface base address for this operation.
/// @param SurfaceWidth is the surface width minus 1 in bytes
/// @param SurfaceHeight is the surface height minus 1 in rows
/// @param SurfacePitch is the surface pitch minus 1 in bytes
/// @param X is zero based X-coordinate of the left upper rectangle corner in
/// number of elements.
/// @param Y is zero based Y-coordinate of the left upper rectangle corner in
/// rows.
/// @param Vals is a vector to store of type T and size N, where N is
///  BlockWidth * BlockHeight * NBlocks, if transformed;
///  otherwise,
///  N = roundUpNextMultiple(BlockHeight, 4 / sizeof(T)) *
///   getNextPowerOf2(BlockWidth) * NBlocks
template <typename Ty, __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_data_order _Transposed, uint8_t NBlocks,
          int BlockWidth, int BlockHeight, bool Transformed, int N>
__ESIMD_INTRIN void
__esimd_lsc_store2d_stateless(__ESIMD_DNS::simd_mask_storage_t<N> Pred,
                              uintptr_t Ptr, int SurfaceWidth,
                              int SurfaceHeight, int SurfacePitch, int X, int Y,
                              __ESIMD_DNS::vector_type_t<Ty, N> vals)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// SLM atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
template <typename Ty, __ESIMD_EDNS::lsc_atomic_op Op,
          __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_xatomic_slm_0(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                          __ESIMD_DNS::vector_type_t<uint32_t, N> offsets)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// SLM atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
template <typename Ty, __ESIMD_EDNS::lsc_atomic_op Op,
          __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_xatomic_slm_1(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> src0)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// SLM atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
template <typename Ty, __ESIMD_EDNS::lsc_atomic_op Op,
          __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_xatomic_slm_2(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> src0,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> src1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param surf_ind is the surface index.
template <
    typename Ty, __ESIMD_EDNS::lsc_atomic_op Op, __ESIMD_ENS::cache_hint L1H,
    __ESIMD_ENS::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
    __ESIMD_ENS::lsc_data_size DS, __ESIMD_EDNS::lsc_vector_size VS,
    __ESIMD_EDNS::lsc_data_order _Transposed, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_xatomic_bti_0(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                          __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
                          SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param surf_ind is the surface index.
template <
    typename Ty, __ESIMD_EDNS::lsc_atomic_op Op, __ESIMD_ENS::cache_hint L1H,
    __ESIMD_ENS::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
    __ESIMD_ENS::lsc_data_size DS, __ESIMD_EDNS::lsc_vector_size VS,
    __ESIMD_EDNS::lsc_data_order _Transposed, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_xatomic_bti_1(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> src0,
    SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
/// @param surf_ind is the surface index.
template <
    typename Ty, __ESIMD_EDNS::lsc_atomic_op Op, __ESIMD_ENS::cache_hint L1H,
    __ESIMD_ENS::cache_hint L3H, uint16_t AddressScale, int ImmOffset,
    __ESIMD_ENS::lsc_data_size DS, __ESIMD_EDNS::lsc_vector_size VS,
    __ESIMD_EDNS::lsc_data_order _Transposed, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_xatomic_bti_2(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> src0,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> src1,
    SurfIndAliasTy surf_ind)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
template <typename Ty, __ESIMD_EDNS::lsc_atomic_op Op,
          __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_xatomic_stateless_0(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                                __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
/// @param src0 is the first atomic operand.
template <typename Ty, __ESIMD_EDNS::lsc_atomic_op Op,
          __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_xatomic_stateless_1(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> src0)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam Op is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L3H is L3 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
template <typename Ty, __ESIMD_EDNS::lsc_atomic_op Op,
          __ESIMD_ENS::cache_hint L1H, __ESIMD_ENS::cache_hint L3H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_ENS::lsc_data_size DS,
          __ESIMD_EDNS::lsc_vector_size VS,
          __ESIMD_EDNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()>
__esimd_lsc_xatomic_stateless_2(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> src0,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_EDNS::to_int<VS>()> src1)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
  return 0;
}
#endif // __SYCL_DEVICE_ONLY__

/// Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the Sfid shaded function.
/// @tparam FenceOp is the fence operation.
/// @tparam Scope is the operation scope.
/// @tparam N is the number of channels (platform dependent).
/// @param pred is predicates.
template <__ESIMD_ENS::lsc_memory_kind Kind, __ESIMD_ENS::lsc_fence_op FenceOp,
          __ESIMD_ENS::lsc_scope Scope, int N>
__ESIMD_INTRIN void __esimd_lsc_fence(__ESIMD_DNS::simd_mask_storage_t<N> pred)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else  // __SYCL_DEVICE_ONLY__
{
  throw cl::sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

/// @endcond ESIMD_DETAIL
