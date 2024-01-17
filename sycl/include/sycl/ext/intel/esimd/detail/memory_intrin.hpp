//==------------ memory_intrin.hpp - DPC++ Explicit SIMD API ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Declares Explicit SIMD intrinsics used to implement working with
// the SIMD classes objects.
//===----------------------------------------------------------------------===//

/// @cond ESIMD_DETAIL

#pragma once

#include <sycl/accessor.hpp>
#include <sycl/ext/intel/esimd/common.hpp>
#include <sycl/ext/intel/esimd/detail/types.hpp>
#include <sycl/ext/intel/esimd/detail/util.hpp>
#include <sycl/types.hpp>

#include <cstdint>

namespace sycl {
inline namespace _V1 {

namespace ext::intel::esimd {
template <typename AccessorTy>
__ESIMD_API SurfaceIndex get_surface_index(AccessorTy acc);
} // namespace ext::intel::esimd

namespace ext::intel::esimd::detail {

// Provides access to sycl accessor class' private members.
class AccessorPrivateProxy {
public:
  template <typename AccessorTy>
  static auto getQualifiedPtrOrImageObj(const AccessorTy &Acc) {
#ifdef __SYCL_DEVICE_ONLY__
    if constexpr (sycl::detail::acc_properties::is_image_accessor_v<AccessorTy>)
      return Acc.getNativeImageObj();
    else
      return Acc.getQualifiedPtr();
#else  // __SYCL_DEVICE_ONLY__
    return Acc;
#endif // __SYCL_DEVICE_ONLY__
  }

#ifndef __SYCL_DEVICE_ONLY__
  static void *getPtr(const sycl::detail::AccessorBaseHost &Acc) {
    return Acc.getPtr();
  }
#endif // __SYCL_DEVICE_ONLY__
};

template <int ElemsPerAddr,
          typename = std::enable_if_t<(ElemsPerAddr == 1 || ElemsPerAddr == 2 ||
                                       ElemsPerAddr == 4)>>
constexpr unsigned int ElemsPerAddrEncoding() {
  // encoding requires log2 of ElemsPerAddr
  if constexpr (ElemsPerAddr == 1)
    return 0;
  else if constexpr (ElemsPerAddr == 2)
    return 1;
  else if constexpr (ElemsPerAddr == 4)
    return 2;

  // other cases not needed since std::enable_if disallows other values
}

constexpr unsigned int ElemsPerAddrDecoding(unsigned int ElemsPerAddrEncoded) {
  // encoding requires 2^ElemsPerAddrEncoded
  return (1 << ElemsPerAddrEncoded);
}

} // namespace ext::intel::esimd::detail
} // namespace _V1
} // namespace sycl

// flat_read does flat-address gather
template <typename Ty, int N, int NumBlk = 0, int ElemsPerAddr = 0>
__ESIMD_INTRIN
    __ESIMD_DNS::vector_type_t<Ty,
                               N * __ESIMD_DNS::ElemsPerAddrDecoding(NumBlk)>
    __esimd_svm_gather(__ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
                       __ESIMD_DNS::simd_mask_storage_t<N> pred = 1)
        __ESIMD_INTRIN_END;

// flat_write does flat-address scatter
template <typename Ty, int N, int NumBlk = 0, int ElemsPerAddr = 0>
__ESIMD_INTRIN void __esimd_svm_scatter(
    __ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty,
                               N * __ESIMD_DNS::ElemsPerAddrDecoding(NumBlk)>
        vals,
    __ESIMD_DNS::simd_mask_storage_t<N> pred = 1) __ESIMD_INTRIN_END;

// Reads a block of data from given surface at given offset.
template <typename Ty, int N, typename SurfIndAliasTy, int32_t IsModified = 0>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_oword_ld_unaligned(SurfIndAliasTy surf_ind,
                           uint32_t offset) __ESIMD_INTRIN_END;

// Writes given block of data to a surface with given index at given offset.
template <typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN void
__esimd_oword_st(SurfIndAliasTy surf_ind, uint32_t owords_offset,
                 __ESIMD_DNS::vector_type_t<Ty, N> vals) __ESIMD_INTRIN_END;

// Read a block of data from the given address.
template <typename Ty, int N, size_t Align>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N> __esimd_svm_block_ld(
    const __ESIMD_DNS::vector_type_t<Ty, N> *addr) __ESIMD_INTRIN_END;

// flat_block_write writes a block of data using one flat address
template <typename Ty, int N, size_t Align>
__ESIMD_INTRIN void
__esimd_slm_block_st(uint32_t offset,
                     __ESIMD_DNS::vector_type_t<Ty, N> vals) __ESIMD_INTRIN_END;

/// SLM block_store/scatter.
/// Supported platforms: DG2, PVC
///
/// Scatters elements located to slm.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to store per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
/// @param offsets is the zero-based offsets for SLM buffer in bytes.
/// @param vals is values to store.
template <typename Ty, __ESIMD_NS::cache_hint L1H, __ESIMD_NS::cache_hint L2H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_DNS::lsc_data_size DS,
          __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN void __esimd_lsc_store_slm(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> vals)
    __ESIMD_INTRIN_END;

// Read a block of data from SLM at the given offset.
template <typename Ty, int N, size_t Align>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_slm_block_ld(uint32_t offset) __ESIMD_INTRIN_END;

// flat_block_write writes a block of data using one flat address
template <typename Ty, int N, size_t Align>
__ESIMD_INTRIN void
__esimd_svm_block_st(__ESIMD_DNS::vector_type_t<Ty, N> *addr,
                     __ESIMD_DNS::vector_type_t<Ty, N> vals) __ESIMD_INTRIN_END;

/// SLM gather/block_load.
/// Supported platforms: DG2, PVC
///
/// Collects elements located at slm and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
/// @param offsets is the zero-based offsets for SLM buffer in bytes.
/// @param pass_thru contains the vector which elements are copied
/// to the returned result when the corresponding element of \p pred is 0.
/// @return is a vector of type T and size N * to_int<VS>()
template <typename Ty, __ESIMD_NS::cache_hint L1H, __ESIMD_NS::cache_hint L2H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_DNS::lsc_data_size DS,
          __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_load_merge_slm(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> pass_thru)
    __ESIMD_INTRIN_END;

/// Similar to __esimd_lsc_load_merge_slm(), but the argument pass_thru is not
/// explicitly specified, which results into random values in those elements of
/// the returned result for which the corresponding element in \p pred is 0.
template <typename Ty, __ESIMD_NS::cache_hint L1H, __ESIMD_NS::cache_hint L2H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_DNS::lsc_data_size DS,
          __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_load_slm(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                     __ESIMD_DNS::vector_type_t<uint32_t, N> offsets)
    __ESIMD_INTRIN_END;

// Gather data from the given global or private addresses.
template <typename T, int N, size_t Align>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N> __esimd_gather_ld(
    __ESIMD_DNS::vector_type_t<uint64_t, N> vptr,
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<T, N> pass_thru) __ESIMD_INTRIN_END;

// Gather data from the given SLM addresses.
template <typename T, int N, size_t Align>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N> __esimd_slm_gather_ld(
    __ESIMD_DNS::vector_type_t<uint32_t, N> vptr,
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<T, N> pass_thru) __ESIMD_INTRIN_END;

/// Surface-based gather.
/// Supported platforms: DG2, PVC
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
///
/// @tparam T is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the
/// transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to
/// access)
/// @tparam SurfIndAliasT is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets in bytes.
/// @param surf_ind is the surface index.
/// @param PassThru contains the vector which elements are copied
/// to the returned result when the corresponding element of \p pred is 0.
/// @return is a vector of type T and N * to_int<VS>()
template <typename T, __ESIMD_NS::cache_hint L1H, __ESIMD_NS::cache_hint L2H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_DNS::lsc_data_size DS,
          __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N, typename SurfIndAliasT>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_load_merge_bti(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets, SurfIndAliasT surf_ind,
    __ESIMD_DNS::vector_type_t<T, N * __ESIMD_DNS::to_int<VS>()> PassThru)
    __ESIMD_INTRIN_END;

/// Similar to __esimd_lsc_load_merge_bti(), but the argument PassThru is not
/// explicitly specified, which results into random values in those elements of
/// the returned result for which the corresponding element in \p pred is 0.
template <typename T, __ESIMD_NS::cache_hint L1H, __ESIMD_NS::cache_hint L2H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_DNS::lsc_data_size DS,
          __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N, typename SurfIndAliasT>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<T, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_load_bti(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                     __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
                     SurfIndAliasT surf_ind) __ESIMD_INTRIN_END;

// flat_read4 does flat-address gather4
template <typename Ty, int N, __ESIMD_NS::rgba_channel_mask Mask>
__ESIMD_DNS::vector_type_t<Ty,
                           N * get_num_channels_enabled(Mask)> __ESIMD_INTRIN
__esimd_svm_gather4_scaled(__ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
                           __ESIMD_DNS::simd_mask_storage_t<N> pred = 1)
    __ESIMD_INTRIN_END;

// flat_write does flat-address scatter
template <typename Ty, int N, __ESIMD_NS::rgba_channel_mask Mask>
__ESIMD_INTRIN void __esimd_svm_scatter4_scaled(
    __ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals,
    __ESIMD_DNS::simd_mask_storage_t<N> pred = 1) __ESIMD_INTRIN_END;

// Low-level surface-based scatter. Writes elements of a \ref simd object into a
// surface at given offsets. Element can be a 1, 2 or 4-byte value, but it is
// always represented as a 4-byte value within the input simd object,
// unused (not written) upper bytes are ignored.
// Template (compile-time constant) parameters:
// @tparam Ty - element type; can only be a 4-byte integer or \c float,
// @tparam N  - the number of elements to write
// @tparam SurfIndAliasTy - "surface index alias" type - internal type in the
//   accessor used to denote the surface
// @tparam TySizeLog2 - Log2 of the number of bytes written per element:
//   0 - 1 byte, 1 - 2 bytes, 2 - 4 bytes
// @tparam Scale - offset scale; only 0 is supported for now
//
// Formal parameters:
// @param pred - per-element predicates; elements with zero corresponding
//   predicates are not written
// @param surf_ind - the surface index, taken from the SYCL memory object
// @param global_offset - offset added to each individual element's offset to
//   compute actual memory access offset for that element
// @param elem_offsets - per-element offsets
// @param vals - values to write
//
template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          int16_t Scale = 0>
__ESIMD_INTRIN void __esimd_scatter_scaled(
    __ESIMD_DNS::simd_mask_storage_t<N> pred, SurfIndAliasTy surf_ind,
    uint32_t global_offset,
    __ESIMD_DNS::vector_type_t<uint32_t, N> elem_offsets,
    __ESIMD_DNS::vector_type_t<Ty, N> vals) __ESIMD_INTRIN_END;

// flat_atomic: flat-address atomic
template <__ESIMD_NS::atomic_op Op, typename Ty, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N> __esimd_svm_atomic0(
    __ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
    __ESIMD_DNS::simd_mask_storage_t<N> pred) __ESIMD_INTRIN_END;

template <__ESIMD_NS::atomic_op Op, typename Ty, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N> __esimd_svm_atomic1(
    __ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N> src0,
    __ESIMD_DNS::simd_mask_storage_t<N> pred) __ESIMD_INTRIN_END;

template <__ESIMD_NS::atomic_op Op, typename Ty, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N> __esimd_svm_atomic2(
    __ESIMD_DNS::vector_type_t<uint64_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N> src0,
    __ESIMD_DNS::vector_type_t<Ty, N> src1,
    __ESIMD_DNS::simd_mask_storage_t<N> pred) __ESIMD_INTRIN_END;

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam InternalOp is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
template <typename Ty, int InternalOp, __ESIMD_NS::cache_hint L1H,
          __ESIMD_NS::cache_hint L2H, uint16_t AddressScale, int ImmOffset,
          __ESIMD_DNS::lsc_data_size DS, __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_xatomic_stateless_0(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                                __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs)
    __ESIMD_INTRIN_END;

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam InternalOp is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)

/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
/// @param src0 is the first atomic operand.
template <typename Ty, int InternalOp, __ESIMD_NS::cache_hint L1H,
          __ESIMD_NS::cache_hint L2H, uint16_t AddressScale, int ImmOffset,
          __ESIMD_DNS::lsc_data_size DS, __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_xatomic_stateless_1(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> src0)
    __ESIMD_INTRIN_END;

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam InternalOp is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
template <typename Ty, int InternalOp, __ESIMD_NS::cache_hint L1H,
          __ESIMD_NS::cache_hint L2H, uint16_t AddressScale, int ImmOffset,
          __ESIMD_DNS::lsc_data_size DS, __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_xatomic_stateless_2(
    __ESIMD_DNS::simd_mask_storage_t<N> Pred,
    __ESIMD_DNS::vector_type_t<uintptr_t, N> Addrs,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> src0,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> src1)
    __ESIMD_INTRIN_END;

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam InternalOp is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param surf_ind is the surface index.
template <typename Ty, int InternalOp, __ESIMD_NS::cache_hint L1H,
          __ESIMD_NS::cache_hint L2H, uint16_t AddressScale, int ImmOffset,
          __ESIMD_DNS::lsc_data_size DS, __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N,
          typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_xatomic_bti_0(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                          __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
                          SurfIndAliasTy surf_ind) __ESIMD_INTRIN_END;

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam InternalOp is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param surf_ind is the surface index.
template <typename Ty, int InternalOp, __ESIMD_NS::cache_hint L1H,
          __ESIMD_NS::cache_hint L2H, uint16_t AddressScale, int ImmOffset,
          __ESIMD_DNS::lsc_data_size DS, __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order _Transposed, int N,
          typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_xatomic_bti_1(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> src0,
    SurfIndAliasTy surf_ind) __ESIMD_INTRIN_END;

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam InternalOp is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
/// @param surf_ind is the surface index.
template <typename Ty, int InternalOp, __ESIMD_NS::cache_hint L1H,
          __ESIMD_NS::cache_hint L2H, uint16_t AddressScale, int ImmOffset,
          __ESIMD_DNS::lsc_data_size DS, __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N,
          typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_xatomic_bti_2(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> src0,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> src1,
    SurfIndAliasTy surf_ind) __ESIMD_INTRIN_END;

/// SLM atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam InternalOp is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
template <typename Ty, int InternalOpOp, __ESIMD_NS::cache_hint L1H,
          __ESIMD_NS::cache_hint L2H, uint16_t AddressScale, int ImmOffset,
          __ESIMD_DNS::lsc_data_size DS, __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_xatomic_slm_0(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                          __ESIMD_DNS::vector_type_t<uint32_t, N> offsets)
    __ESIMD_INTRIN_END;

/// SLM atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam InternalOp is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
template <typename Ty, int InternalOp, __ESIMD_NS::cache_hint L1H,
          __ESIMD_NS::cache_hint L2H, uint16_t AddressScale, int ImmOffset,
          __ESIMD_DNS::lsc_data_size DS, __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_xatomic_slm_1(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> src0)
    __ESIMD_INTRIN_END;

/// SLM atomic.
/// Supported platforms: DG2, PVC
///
/// @tparam Ty is element type.
/// @tparam InternalOp is operation type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
template <typename Ty, int InternalOp, __ESIMD_NS::cache_hint L1H,
          __ESIMD_NS::cache_hint L2H, uint16_t AddressScale, int ImmOffset,
          __ESIMD_DNS::lsc_data_size DS, __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_xatomic_slm_2(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> src0,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> src1)
    __ESIMD_INTRIN_END;

__ESIMD_INTRIN void __esimd_slm_init(uint32_t size) __ESIMD_INTRIN_END;

// esimd_barrier, generic group barrier
__ESIMD_INTRIN void __esimd_barrier() __ESIMD_INTRIN_END;

// slm_fence sets the SLM read/write order
__ESIMD_INTRIN void __esimd_fence(uint8_t cntl) __ESIMD_INTRIN_END;

/// Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the Sfid shaded function.
/// @tparam FenceOp is the fence operation.
/// @tparam Scope is the operation scope.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
template <uint8_t Kind, uint8_t FenceOp, uint8_t Scope, int N>
__ESIMD_INTRIN void
__esimd_lsc_fence(__ESIMD_DNS::simd_mask_storage_t<N> pred) __ESIMD_INTRIN_END;

// Predicated (masked) scaled gather from a surface.
//
// Template (compile-time constant) parameters:
// @tparam Ty - element type
// @tparam N  - the number of elements to read
// @tparam SurfIndAliasTy - "surface index alias" type - internal type in the
//   accessor used to denote the surface
// @tparam TySizeLog2 - Log2 of the number of bytes written per element:
//   0 - 1 byte, 1 - 2 bytes, 2 - 4 bytes
// @tparam Scale - offset scale; only 0 is supported for now
//
// Formal parameters:
// @param surf_ind - the surface index, taken from the SYCL memory object
// @param global_offset - offset added to each individual element's offset to
//   compute actual memory access offset for that element
// @param offsets - per-element offsets
// @param pred - per-element predicates; elements with zero corresponding
//   predicates are not written
// @return - elements read ("gathered") from memory

template <typename Ty, int N, typename SurfIndAliasTy, int TySizeLog2,
          int16_t Scale = 0>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N> __esimd_gather_masked_scaled2(
    SurfIndAliasTy surf_ind, uint32_t global_offset,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::simd_mask_storage_t<N> pred) __ESIMD_INTRIN_END;

// Reads a block of data from given surface at given `offset` counted
// in 16-byte chunks.
template <typename Ty, int N, typename SurfIndAliasTy, int32_t IsModified = 0>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N>
__esimd_oword_ld(SurfIndAliasTy surf_ind,
                 uint32_t owords_offset) __ESIMD_INTRIN_END;

// gather4 scaled masked from a surface/SLM
template <typename Ty, int N, __ESIMD_NS::rgba_channel_mask Mask,
          typename SurfIndAliasTy, int16_t Scale = 0>
__ESIMD_INTRIN
    __ESIMD_DNS::vector_type_t<Ty, N * get_num_channels_enabled(Mask)>
    __esimd_gather4_masked_scaled2(
        SurfIndAliasTy surf_ind, int global_offset,
        __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
        __ESIMD_DNS::simd_mask_storage_t<N> pred) __ESIMD_INTRIN_END;

// scatter4 scaled to a surface/SLM
template <typename Ty, int N, typename SurfIndAliasTy,
          __ESIMD_NS::rgba_channel_mask Mask, int16_t Scale = 0>
__ESIMD_INTRIN void __esimd_scatter4_scaled(
    __ESIMD_DNS::simd_mask_storage_t<N> pred, SurfIndAliasTy surf_ind,
    int global_offset, __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * get_num_channels_enabled(Mask)> vals)
    __ESIMD_INTRIN_END;

// Surface-based atomic operations
template <__ESIMD_NS::atomic_op Op, typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N> __esimd_dword_atomic0(
    __ESIMD_DNS::simd_mask_storage_t<N> pred, SurfIndAliasTy surf_ind,
    __ESIMD_DNS::vector_type_t<uint32_t, N> addrs) __ESIMD_INTRIN_END;

template <__ESIMD_NS::atomic_op Op, typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N> __esimd_dword_atomic1(
    __ESIMD_DNS::simd_mask_storage_t<N> pred, SurfIndAliasTy surf_ind,
    __ESIMD_DNS::vector_type_t<uint32_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N> src0) __ESIMD_INTRIN_END;

template <__ESIMD_NS::atomic_op Op, typename Ty, int N, typename SurfIndAliasTy>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N> __esimd_dword_atomic2(
    __ESIMD_DNS::simd_mask_storage_t<N> pred, SurfIndAliasTy surf_ind,
    __ESIMD_DNS::vector_type_t<uint32_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N> src0,
    __ESIMD_DNS::vector_type_t<Ty, N> src1) __ESIMD_INTRIN_END;

// Media block load.
//
// @tparam Ty the element data type.
// @tparam M the hight of the 2D block.
// @tparam N the width of the 2D block.
// @tparam Modifier top/bottom field surface access control.
// @tparam TACC type of the surface handle.
// @tparam Plane planar surface index.
// @tparam BlockWidth the width of the return block.
// @param handle the surface handle.
// @param x X-coordinate of the left upper rectangle corner in BYTES.
// @param y Y-coordinate of the left upper rectangle corner in ROWS.
//
// @return the linearized 2D block data read from surface.
//
template <typename Ty, int M, int N, int Modifier, typename TACC, int Plane,
          int BlockWidth>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, M * N>
__esimd_media_ld(TACC handle, unsigned x, unsigned y) __ESIMD_INTRIN_END;

// Media block store
//
// @tparam Ty the element data type.
// @tparam M the hight of the 2D block.
// @tparam N the width of the 2D block.
// @tparam Modifier top/bottom field surface access control.
// @tparam TACC type of the surface handle.
// @tparam Plane planar surface index.
// @tparam BlockWidth the width of the return block.
// @param handle the surface handle.
// @param x X-coordinate of the left upper rectangle corner in BYTES.
// @param y Y-coordinate of the left upper rectangle corner in ROWS.
// @param vals the linearized 2D block data to be written to surface.
//
template <typename Ty, int M, int N, int Modifier, typename TACC, int Plane,
          int BlockWidth>
__ESIMD_INTRIN void
__esimd_media_st(TACC handle, unsigned x, unsigned y,
                 __ESIMD_DNS::vector_type_t<Ty, M * N> vals) __ESIMD_INTRIN_END;

// \brief Converts given value to a surface index.
// The input must always be a result of
//   detail::AccessorPrivateProxy::getQualifiedPtrOrImageObj(acc)
// where acc is a buffer or image accessor. If the result is, say, 'obj', then
// 'obj' is really a value of the surface index kept in a differently typed
// accessor field. Front-end compilation time type of 'obj' is either
//   ConcreteASPtrType (detail::DecoratedType<DataT, AS>::type *), for a buffer
// or
//   image{1,2,3}d_t OpenCL type for an image
// But when doing code generation, FE replaces e.g. '__read_only image2d_t' FE
// type with '%opencl.image2d_ro_t addrspace(1) *' LLVM type or a Target
// Extension Type if using opaque pointers. These types can neither be
// reinterpret_cast'ed from pointer to intptr_t (because they are not a pointer
// at FE translation time), nor can they be bit_cast'ed to intptr_t (because
// they are not trivially copyable). This function takes advantage of the fact
// that in SPIR-V 'obj' is always a pointer, where we can do ptr to uint32_t
// conversion. This function can be called only from the device code, as
// accessor => memory handle translation for host is different.
// @param acc the SYCL accessor.
// Returns the binding table index value.
template <typename MemObjTy>
ESIMD_INLINE __ESIMD_NS::SurfaceIndex __esimd_get_surface_index(MemObjTy obj) {
#ifdef __SYCL_DEVICE_ONLY__
  return __spirv_ConvertPtrToU<MemObjTy, uint32_t>(obj);
#else  // __SYCL_DEVICE_ONLY__
  __ESIMD_UNSUPPORTED_ON_HOST;
#endif // __SYCL_DEVICE_ONLY__
}

/// USM pointer gather.
/// Supported platforms: DG2, PVC
///
/// Collects elements located at specified address and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
/// @param addrs is the load addresses.
/// @param pass_thru is the vector of values copied to the result when the
/// corresponding element in \p pred is unset.
/// @return is a vector of type T and N * to_int<VS>()
template <typename Ty, __ESIMD_NS::cache_hint L1H, __ESIMD_NS::cache_hint L2H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_DNS::lsc_data_size DS,
          __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_load_merge_stateless(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> pass_thru = 0)
    __ESIMD_INTRIN_END;

/// USM pointer gather.
/// Supported platforms: DG2, PVC
///
/// Collects elements located at specified address and returns them
/// as a single \ref simd object.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
/// @param addrs is the load addresses.
/// @return is a vector of type T and N * to_int<VS>()
template <typename Ty, __ESIMD_NS::cache_hint L1H, __ESIMD_NS::cache_hint L2H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_DNS::lsc_data_size DS,
          __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order Transposed, int N>
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()>
__esimd_lsc_load_stateless(__ESIMD_DNS::simd_mask_storage_t<N> pred,
                           __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs)
    __ESIMD_INTRIN_END;

/// USM pointer scatter.
/// Supported platforms: DG2, PVC
///
/// Scatters elements to specific address.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @param pred is predicates.
/// @param addrs is the prefetch addresses.
/// @param vals is values to store.
template <typename Ty, __ESIMD_NS::cache_hint L1H, __ESIMD_NS::cache_hint L2H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_DNS::lsc_data_size DS,
          __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order _Transposed, int N>
__ESIMD_INTRIN void __esimd_lsc_store_stateless(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uintptr_t, N> addrs,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> vals)
    __ESIMD_INTRIN_END;

/// Surface-based scatter.
/// Supported platforms: DG2, PVC
///
/// Scatters elements to surface.
///
/// @tparam Ty is element type.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AddressScale is the address scale.
/// @tparam ImmOffset is the immediate offset added to each address.
/// @tparam DS is the data size.
/// @tparam VS is the number of elements to load per address.
/// @tparam Transposed indicates if the data is transposed during the transfer.
/// @tparam N is the SIMD size of operation (the number of addresses to access)
/// @tparam SurfIndAliasTy is the \ref sycl::accessor type.
/// @param pred is predicates.
/// @param offsets is the zero-based offsets in bytes.
/// @param vals is values to store.
/// @param surf_ind is the surface index.
template <typename Ty, __ESIMD_NS::cache_hint L1H, __ESIMD_NS::cache_hint L2H,
          uint16_t AddressScale, int ImmOffset, __ESIMD_DNS::lsc_data_size DS,
          __ESIMD_DNS::lsc_vector_size VS,
          __ESIMD_DNS::lsc_data_order _Transposed, int N,
          typename SurfIndAliasTy>
__ESIMD_INTRIN void __esimd_lsc_store_bti(
    __ESIMD_DNS::simd_mask_storage_t<N> pred,
    __ESIMD_DNS::vector_type_t<uint32_t, N> offsets,
    __ESIMD_DNS::vector_type_t<Ty, N * __ESIMD_DNS::to_int<VS>()> vals,
    SurfIndAliasTy surf_ind) __ESIMD_INTRIN_END;

// \brief Raw sends.
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
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty1, N1> __esimd_raw_sends2(
    uint8_t modifier, uint8_t execSize,
    __ESIMD_DNS::simd_mask_storage_t<N> pred, uint8_t numSrc0, uint8_t numSrc1,
    uint8_t numDst, uint8_t sfid, uint32_t exDesc, uint32_t msgDesc,
    __ESIMD_DNS::vector_type_t<Ty2, N2> msgSrc0,
    __ESIMD_DNS::vector_type_t<Ty3, N3> msgSrc1,
    __ESIMD_DNS::vector_type_t<Ty1, N1> msgDst) __ESIMD_INTRIN_END;

// \brief Raw send.
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
__ESIMD_INTRIN __ESIMD_DNS::vector_type_t<Ty1, N1> __esimd_raw_send2(
    uint8_t modifier, uint8_t execSize,
    __ESIMD_DNS::simd_mask_storage_t<N> pred, uint8_t numSrc0, uint8_t numDst,
    uint8_t sfid, uint32_t exDesc, uint32_t msgDesc,
    __ESIMD_DNS::vector_type_t<Ty2, N2> msgSrc0,
    __ESIMD_DNS::vector_type_t<Ty1, N1> msgDst) __ESIMD_INTRIN_END;

// \brief Raw sends.
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
__ESIMD_INTRIN void __esimd_raw_sends2_noresult(
    uint8_t modifier, uint8_t execSize,
    __ESIMD_DNS::simd_mask_storage_t<N> pred, uint8_t numSrc0, uint8_t numSrc1,
    uint8_t sfid, uint32_t exDesc, uint32_t msgDesc,
    __ESIMD_DNS::vector_type_t<Ty1, N1> msgSrc0,
    __ESIMD_DNS::vector_type_t<Ty2, N2> msgSrc1) __ESIMD_INTRIN_END;

// \brief Raw send.
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
__ESIMD_INTRIN void __esimd_raw_send2_noresult(
    uint8_t modifier, uint8_t execSize,
    __ESIMD_DNS::simd_mask_storage_t<N> pred, uint8_t numSrc0, uint8_t sfid,
    uint32_t exDesc, uint32_t msgDesc,
    __ESIMD_DNS::vector_type_t<Ty1, N1> msgSrc0) __ESIMD_INTRIN_END;

/// @endcond ESIMD_DETAIL
