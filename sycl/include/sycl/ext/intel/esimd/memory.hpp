//==-------------- memory.hpp - DPC++ Explicit SIMD API --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Implement Explicit SIMD memory-access APIs.
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/ext/intel/esimd/common.hpp>
#include <sycl/ext/intel/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/esimd/detail/types.hpp>
#include <sycl/ext/intel/esimd/detail/util.hpp>
#include <sycl/ext/intel/esimd/memory_properties.hpp>
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/simd_view.hpp>
#include <sycl/half_type.hpp>

#include <algorithm>
#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd {

/// @addtogroup sycl_esimd_memory
/// @{

/// @defgroup sycl_esimd_memory_atomics Atomic memory access.
/// Memory access functions which perform per-lane atomic update using given
/// operation. "Per-lane" means that the atomicity guarantees of a vector atomic
/// operation are the same as of N independent scalar atomic operations per
/// lane (N is number of lanes).

/// @defgroup sycl_esimd_memory_slm Shared local memory access functions.

/// @defgroup sycl_esimd_memory_block Block load/prefetch/store functions.

/// @} sycl_esimd_memory

/// @addtogroup sycl_esimd_memory
/// @{

/// Get surface index corresponding to a SYCL accessor.
///
/// @param acc a SYCL buffer or image accessor.
/// @return the index of the corresponding surface (aka "binding table index").
///
template <typename AccessorTy>
__ESIMD_API SurfaceIndex get_surface_index(AccessorTy acc) {
  if constexpr (std::is_same_v<detail::LocalAccessorMarker, AccessorTy> ||
                sycl::detail::acc_properties::is_local_accessor_v<AccessorTy>) {
    return detail::SLM_BTI;
  } else {
#ifdef __ESIMD_FORCE_STATELESS_MEM
    static_assert(sycl::detail::acc_properties::is_image_accessor_v<AccessorTy>,
                  "The function get_surface_index() is available only for "
                  "image- and local-accessors in stateless-only memory mode. "
                  "Consider using "
                  "-fno-sycl-esimd-force-stateless-mem compilation switch.");
#endif // __ESIMD_FORCE_STATELESS_MEM
    return __esimd_get_surface_index(
        detail::AccessorPrivateProxy::getQualifiedPtrOrImageObj(acc));
  }
}

namespace detail {

// Format u8 and u16 to u8u32 and u16u32 by doing garbage-extension.
template <typename RT, typename T, int N>
ESIMD_INLINE simd<RT, N> lsc_format_input(simd<T, N> Vals) {
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
ESIMD_INLINE simd<T, N> lsc_format_ret(simd<T1, N> Vals) {
  auto Formatted = Vals.template bit_cast_view<T>();
  if constexpr (sizeof(T) == sizeof(T1)) {
    return Formatted;
  } else {
    constexpr int Stride = Formatted.length / N;
    return Formatted.template select<N, Stride>(0);
  }
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
/// @tparam L2H is L2 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
/// @return is a vector of type T and size N * NElts
template <typename T, int NElts, lsc_data_size DS, cache_hint L1H,
          cache_hint L2H, int N, typename OffsetT>
__ESIMD_API simd<T, N * NElts> gather_impl(const T *p, simd<OffsetT, N> offsets,
                                           simd_mask<N> pred) {
  static_assert(std::is_integral_v<OffsetT>, "Unsupported offset type");
  check_lsc_vector_size<NElts>();
  check_lsc_data_size<T, DS>();
  check_cache_hint<cache_action::load, L1H, L2H>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<NElts>();
  constexpr auto Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  simd<MsgT, N * NElts> Tmp =
      __esimd_lsc_load_stateless<MsgT, L1H, L2H, AddressScale, ImmOffset, EDS,
                                 VS, Transposed, N>(pred.data(), addrs.data());
  return lsc_format_ret<T>(Tmp);
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
/// @tparam L2H is L2 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
/// @param pass_thru contains the vector which elements are copied
/// to the returned result when the corresponding element of \p pred is 0.
/// @return is a vector of type T and size N * NElts
///
template <typename T, int NElts, lsc_data_size DS, cache_hint L1H,
          cache_hint L2H, int N, typename OffsetT>
__ESIMD_API simd<T, N * NElts> gather_impl(const T *p, simd<OffsetT, N> offsets,
                                           simd_mask<N> pred,
                                           simd<T, N * NElts> pass_thru) {
  static_assert(std::is_integral_v<OffsetT>, "Unsupported offset type");
  check_lsc_vector_size<NElts>();
  check_lsc_data_size<T, DS>();
  check_cache_hint<cache_action::load, L1H, L2H>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<NElts>();
  constexpr auto Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  simd<uintptr_t, N> Addrs = reinterpret_cast<uintptr_t>(p);
  Addrs += convert<uintptr_t>(offsets);
  simd<MsgT, N * NElts> PassThruExpanded = lsc_format_input<MsgT>(pass_thru);
  simd<MsgT, N * NElts> Result =
      __esimd_lsc_load_merge_stateless<MsgT, L1H, L2H, AddressScale, ImmOffset,
                                       EDS, VS, Transposed, N>(
          pred.data(), Addrs.data(), PassThruExpanded.data());
  return lsc_format_ret<T>(Result);
}

// Returns true iff it is Ok to use llvm.masked.gather and llvm.masked.scatter.
// By default (without use specifying __ESIMD_GATHER_SCATTER_LLVM_IR) it is
// not used because of an issue in GPU driver, which does not recognize
// those operations in SPIR-V when they are used in mixed (scalar and vector)
// kernels using invoke_simd() API.
constexpr bool isMaskedGatherScatterLLVMAvailable() {
#ifdef __ESIMD_GATHER_SCATTER_LLVM_IR
  return true;
#else
  return false;
#endif
}

} // namespace detail

/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (usm-ga-1)
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (usm-ga-2)
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (usm-ga-3)
///
/// The next 3 functions are similar to the above and were added for
/// convenience. They assume the VS parameter is set to 1 and do not require
/// specifying the template parameters <T, N, VS> at function calls.
/// template <typename T, int N, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (usm-ga-4)
/// simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask, PropertyListT props = {});// (usm-ga-5)
/// simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
///                   PropertyListT props = {});                   // (usm-ga-6)
///
/// The next 3 functions are variations of the first 3 above (usm-ga-1,2,3)
/// and were added only to support simd_view instead of simd for byte_offsets
/// and/or pass_thru operands.
/// template <typename T, int N, int VS = 1, typename OffsetObjT,
///           typename OffsetRegionT, typename PropertyListT = empty_props_t>
/// simd <T, N> gather(const T *p,
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             simd_mask<N / VS> mask, simd<T, N> pass_thru,
///             PropertyListT props = {});                         // (usm-ga-7)
/// simd <T, N> gather(const T *p,
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-ga-8)
/// simd <T, N> gather(const T *p,
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             PropertyListT props = {});                         // (usm-ga-9)

/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (usm-ga-1)
#ifndef __ESIMD_GATHER_SCATTER_LLVM_IR
/// Supported platforms: DG2, PVC only - Temporary restriction for the variant
/// with pass_thru operand.
#endif // __ESIMD_GATHER_SCATTER_LLVM_IR
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the load from
/// (p + byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param pass_thru The vector pass through values.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <typename T, int N, int VS, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
       simd<T, N> pass_thru, PropertyListT props = {}) {
  static_assert(std::is_integral_v<OffsetT>, "Unsupported offset type");
  static_assert(N / VS >= 1 && N % VS == 0, "N must be divisible by VS");

  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(Alignment >= sizeof(T),
                "gather() requires at least element-size alignment");
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);

  // Use LSC lowering if L1/L2 or VS > 1. Also, if masked gather is
  // not available, then LSC is the only lowering option.
  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                VS > 1 || !detail::isMaskedGatherScatterLLVMAvailable()) {
    static_assert(VS == 1 || sizeof(T) >= 4,
                  "VS > 1 is supprted only for 4- and 8-byte elements");
    return detail::gather_impl<T, VS, detail::lsc_data_size::default_size,
                               L1Hint, L2Hint>(p, byte_offsets, mask,
                                               pass_thru);
  } else {
    simd<uint64_t, N> Addrs(reinterpret_cast<uint64_t>(p));
    Addrs = Addrs + convert<uint64_t>(byte_offsets);

    using MsgT = detail::__raw_t<T>;
    return __esimd_gather_ld<MsgT, N, Alignment>(
        Addrs.data(), mask.data(),
        sycl::bit_cast<__ESIMD_DNS::vector_type_t<MsgT, N>>(pass_thru.data()));
  }
}

/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (usm-ga-2)
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the load from
/// (p + byte_offsets[i]) is skipped and the corresponding i-th element of the
/// returned vector is undefined.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
template <typename T, int N, int VS, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
       PropertyListT props = {}) {
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(Alignment >= sizeof(T),
                "gather() requires at least element-size alignment");
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);

  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                VS > 1 || detail::isMaskedGatherScatterLLVMAvailable()) {
    simd<T, N> PassThru; // it is intentionally undefined
    return gather<T, N, VS>(p, byte_offsets, mask, PassThru, props);
  } else {
    static_assert(detail::isPowerOf2(N, 32), "Unsupported value of N");
    simd<uintptr_t, N> Addrs = reinterpret_cast<uintptr_t>(p);
    Addrs += convert<uintptr_t>(byte_offsets);
    using MsgT = detail::__raw_t<T>;
    if constexpr (sizeof(T) == 1) {
      auto Ret = __esimd_svm_gather<MsgT, N, detail::ElemsPerAddrEncoding<4>(),
                                    detail::ElemsPerAddrEncoding<1>()>(
          Addrs.data(), mask.data());
      return __esimd_rdregion<MsgT, N * 4, N, /*VS*/ 0, N, 4>(Ret, 0);
    } else if constexpr (sizeof(T) == 2) {
      auto Ret = __esimd_svm_gather<MsgT, N, detail::ElemsPerAddrEncoding<2>(),
                                    detail::ElemsPerAddrEncoding<2>()>(
          Addrs.data(), mask.data());
      return __esimd_rdregion<MsgT, N * 2, N, /*VS*/ 0, N, 2>(Ret, 0);
    } else {
      return __esimd_svm_gather<MsgT, N, detail::ElemsPerAddrEncoding<1>(),
                                detail::ElemsPerAddrEncoding<1>()>(Addrs.data(),
                                                                   mask.data());
    }
  }
}

/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (usm-ga-3)
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <typename T, int N, int VS, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd<OffsetT, N / VS> byte_offsets,
       PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  return gather<T, N, VS>(p, byte_offsets, Mask, props);
}

/// template <typename T, int N, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (usm-ga-4)
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the load from
/// (p + byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param pass_thru The vector pass through values.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <typename T, int N, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd<OffsetT, N> byte_offsets, simd_mask<N> mask,
       simd<T, N> pass_thru, PropertyListT props = {}) {
  constexpr int VS = 1;
  return gather<T, N, VS>(p, byte_offsets, mask, pass_thru, props);
}

/// template <typename T, int N, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask, PropertyListT props = {});// (usm-ga-5)
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the load from
/// (p + byte_offsets[i]) is skipped and the corresponding i-th element of the
/// returned vector is undefined.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
template <typename T, int N, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd<OffsetT, N> byte_offsets, simd_mask<N> mask,
       PropertyListT props = {}) {
  constexpr int VS = 1;
  return gather<T, N, VS>(p, byte_offsets, mask, props);
}

/// template <typename T, int N, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(const T *p, simd<OffsetT, N> byte_offsets,
///                   PropertyListT props = {});                   // (usm-ga-6)
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <typename T, int N, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd<OffsetT, N> byte_offsets, PropertyListT props = {}) {
  constexpr int VS = 1;
  return gather<T, N, VS>(p, byte_offsets, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetObjT,
///           typename OffsetRegionT, typename PropertyListT = empty_props_t>
/// simd <T, N> gather(const T *p,
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             simd_mask<N / VS> mask, simd<T, N> pass_thru,
///             PropertyListT props = {});                         // (usm-ga-7)
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the load from
/// (p + byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param pass_thru The vector pass through values.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <typename T, int N, int VS = 1, typename OffsetObjT,
          typename OffsetRegionT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
       simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {}) {
  return gather<T, N, VS>(p, byte_offsets.read(), mask, pass_thru, props);
}

/// simd <T, N> gather(const T *p,
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-ga-8)
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the load from
/// (p + byte_offsets[i]) is skipped and the corresponding i-th element of the
/// returned vector is undefined.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
template <typename T, int N, int VS = 1, typename OffsetObjT,
          typename OffsetRegionT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
       simd_mask<N / VS> mask, PropertyListT props = {}) {
  return gather<T, N, VS>(p, byte_offsets.read(), mask, props);
}

/// simd <T, N> gather(const T *p,
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             PropertyListT props = {});                         // (usm-ga-9)
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <typename T, int N, int VS = 1, typename OffsetObjT,
          typename OffsetRegionT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
       PropertyListT props = {}) {
  return gather<T, N, VS>(p, byte_offsets.read(), props);
}

/// A variation of \c gather API with \c offsets represented as scalar.
///
/// @tparam Tx Element type, must be of size 4 or less.
/// @tparam N Number of elements to read; can be \c 1, \c 2, \c 4, \c 8, \c 16
///   or \c 32.
/// @param p The base address.
/// @param offset the scalar 32-bit or 64-bit offset in bytes.
/// ((byte*)p + offset) must be element size aligned.
/// @param mask The access mask, defaults to all 1s.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
///
template <typename Tx, int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset>, simd<Tx, N>>
gather(const Tx *p, Toffset offset, simd_mask<N> mask = 1) {
  return gather<Tx, N>(p, simd<Toffset, N>(offset), mask);
}

/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector. Access to
/// any element's memory location can be disabled via the input mask.
/// @tparam Tx Element type, must be of size 4 or less.
/// @tparam N Number of elements to write; can be \c 1, \c 2, \c 4, \c 8, \c 16
///   or \c 32.
/// @param p The base address.
/// @param offsets A vector of 32-bit or 64-bit offsets in bytes. For each lane
/// \c i,   ((byte*)p + offsets[i]) must be element size aligned.
/// @param vals The vector to scatter.
/// @param mask The access mask, defaults to all 1s.
///
template <typename Tx, int N, typename Toffset>
__ESIMD_API void scatter(Tx *p, simd<Toffset, N> offsets, simd<Tx, N> vals,
                         simd_mask<N> mask = 1) {
  using T = detail::__raw_t<Tx>;
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert(detail::isPowerOf2(N, 32), "Unsupported value of N");
  simd<uint64_t, N> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  if constexpr (sizeof(T) == 1) {
    simd<T, N * 4> D;
    D = __esimd_wrregion<T, N * 4, N, /*VS*/ 0, N, 4>(D.data(), vals.data(), 0);
    __esimd_svm_scatter<T, N, detail::ElemsPerAddrEncoding<4>(),
                        detail::ElemsPerAddrEncoding<1>()>(
        addrs.data(), D.data(), mask.data());
  } else if constexpr (sizeof(T) == 2) {
    simd<T, N * 2> D;
    D = __esimd_wrregion<T, N * 2, N, /*VS*/ 0, N, 2>(D.data(), vals.data(), 0);
    __esimd_svm_scatter<T, N, detail::ElemsPerAddrEncoding<2>(),
                        detail::ElemsPerAddrEncoding<2>()>(
        addrs.data(), D.data(), mask.data());
  } else
    __esimd_svm_scatter<T, N, detail::ElemsPerAddrEncoding<1>(),
                        detail::ElemsPerAddrEncoding<1>()>(
        addrs.data(), vals.data(), mask.data());
}

/// A variation of \c scatter API with \c offsets represented as \c simd_view
/// object.
///
/// @tparam Tx Element type, must be of size 4 or less.
/// @tparam N Number of elements to write; can be \c 1, \c 2, \c 4, \c 8, \c 16
///   or \c 32.
/// @param p The base address.
/// @param offsets A simd_view of 32-bit or 64-bit offsets in bytes. For each
/// lane \c i,   ((byte*)p + offsets[i]) must be element size aligned.
/// @param vals The vector to scatter.
/// @param mask The access mask, defaults to all 1s.
///
template <typename Tx, int N, typename OffsetObjT, typename RegionTy>
__ESIMD_API void scatter(Tx *p, simd_view<OffsetObjT, RegionTy> offsets,
                         simd<Tx, N> vals, simd_mask<N> mask = 1) {
  scatter<Tx, N>(p, offsets.read(), vals, mask);
}

/// A variation of \c scatter API with \c offsets represented as scalar.
///
/// @tparam Tx Element type, must be of size 4 or less.
/// @tparam N Number of elements to write; can be \c 1, \c 2, \c 4, \c 8, \c 16
///   or \c 32.
/// @param p The base address.
/// @param offset the scalar 32-bit or 64-bit offset in bytes.
/// ((byte*)p + offset) must be element size aligned.
/// @param vals The vector to scatter.
/// @param mask The access mask, defaults to all 1s.
///
template <typename Tx, int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> && N == 1>
scatter(Tx *p, Toffset offset, simd<Tx, N> vals, simd_mask<N> mask = 1) {
  scatter<Tx, N>(p, simd<Toffset, N>(offset), vals, mask);
}

namespace detail {
// Accessors may get either 32-bit offset or 64-bit depending on
// the -fsycl-esimd-force-stateles-mem mode setting.
#ifdef __ESIMD_FORCE_STATELESS_MEM
using DeviceAccessorOffsetT = uint64_t;
#else
using DeviceAccessorOffsetT = uint32_t;
#endif

template <typename T, int NElts, cache_hint L1H, cache_hint L2H,
          typename FlagsT>
__ESIMD_API std::enable_if_t<is_simd_flag_type_v<FlagsT>, simd<T, NElts>>
block_load_impl(const T *p, simd_mask<1> pred, FlagsT flags) {
  // Verify input template arguments.
  check_cache_hint<cache_action::load, L1H, L2H>();
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor64Bit = sizeof(uint64_t) / sizeof(T);
  constexpr int SmallIntFactor32Bit =
      sizeof(uint32_t) / sizeof(T) > 1 ? sizeof(uint32_t) / sizeof(T) : 1;
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");

  // If alignment >= 8 and (NElts * sizeof(T)) % 8 == 0) we can load QWORDs.
  // Don't do it for 4-byte vectors (unless it is greater than 256-bytes),
  // because it would require a bit-cast, which is supposed to be NO-OP, but
  // might confuse GPU BE sometimes. 1- and 2-byte vectors are casted anyways.
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (NElts * sizeof(T)) % sizeof(uint64_t) == 0 &&
      (sizeof(T) != sizeof(uint32_t) || NElts * sizeof(T) > 256);
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  check_lsc_vector_size<FactoredNElts>();

  // Prepare template arguments for the call of intrinsic.
  using LoadElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size ActualDS =
      Use64BitData ? lsc_data_size::u64 : lsc_data_size::u32;
  constexpr lsc_vector_size VS = to_lsc_vector_size<FactoredNElts>();
  constexpr auto Transposed = lsc_data_order::transpose;
  constexpr int N = 1;

  // Prepare non-template arguments and call the intrinsic.
  simd<uintptr_t, N> Addrs = reinterpret_cast<uintptr_t>(p);
  simd<LoadElemT, FactoredNElts> Result =
      __esimd_lsc_load_stateless<LoadElemT, L1H, L2H, AddressScale, ImmOffset,
                                 ActualDS, VS, Transposed, N>(pred.data(),
                                                              Addrs.data());
  return Result.template bit_cast_view<T>();
}

/// USM pointer transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
/// Instruction can load max: DG2(64xD32 or 32xD64), PVC(64xD32 or 64xD64).
///
/// Accesses contiguous block of memory of `NElts * sizeof(T)` bytes  starting
/// from the given address \p p. The maximum size of accessed block is 512 bytes
/// for PVC and 256 bytes for ACM (DG2).
/// When \c sizeof(T) equal to 8 the address must be 8-byte aligned,
/// otherwise - 4-byte aligned.
/// When T is 1- or 2-byte type, the memory block is loaded with DWORDs
/// or QWORDs depending on the alignment.
/// Allowed \c NElts values for 8-byte data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 4-byte data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 2-byte data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 1-byte data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
/// 8-byte alignment is required for 8-byte data, or if sizeof(T) * NElts > 256.
/// Otherwise, 4-byte alignment is required.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @param p is the base pointer.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed.
/// @param pass_thru contains the vector which elements are copied
/// to the returned result when the corresponding element of \p pred is 0.
/// @param flags is the alignment specifier type tag.
/// @return is a vector of type T and size NElts.
///
template <typename T, int NElts, cache_hint L1H, cache_hint L2H,
          typename FlagsT>
__ESIMD_API std::enable_if_t<is_simd_flag_type_v<FlagsT>, simd<T, NElts>>
block_load_impl(const T *p, simd_mask<1> pred, simd<T, NElts> pass_thru,
                FlagsT flags) {
  // Verify input template arguments.
  check_cache_hint<cache_action::load, L1H, L2H>();
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor64Bit = sizeof(uint64_t) / sizeof(T);
  constexpr int SmallIntFactor32Bit =
      sizeof(uint32_t) / sizeof(T) > 1 ? sizeof(uint32_t) / sizeof(T) : 1;
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");

  // If alignment >= 8 and (NElts * sizeof(T)) % 8 == 0) we can load QWORDs.
  // Don't do it for 4-byte vectors (unless it is greater than 256-bytes),
  // because it would require a bit-cast, which is supposed to be NO-OP, but
  // might confuse GPU BE sometimes. 1- and 2-byte vectors are casted anyways.
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (NElts * sizeof(T)) % sizeof(uint64_t) == 0 &&
      (sizeof(T) != sizeof(uint32_t) || NElts * sizeof(T) > 256);
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  check_lsc_vector_size<FactoredNElts>();

  // Prepare template arguments for the call of intrinsic.
  using LoadElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size ActualDS =
      Use64BitData ? lsc_data_size::u64 : lsc_data_size::u32;
  constexpr lsc_vector_size VS = to_lsc_vector_size<FactoredNElts>();
  constexpr auto Transposed = lsc_data_order::transpose;
  constexpr int N = 1;

  // Prepare non-template arguments and call the intrinsic.
  simd<uintptr_t, N> Addrs = reinterpret_cast<uintptr_t>(p);
  simd<LoadElemT, FactoredNElts> PassThru =
      pass_thru.template bit_cast_view<LoadElemT>();
  simd<LoadElemT, FactoredNElts> Result =
      __esimd_lsc_load_merge_stateless<LoadElemT, L1H, L2H, AddressScale,
                                       ImmOffset, ActualDS, VS, Transposed, N>(
          pred.data(), Addrs.data(), PassThru.data());
  return Result.template bit_cast_view<T>();
}

/// Accessor-based transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
/// Instruction can load max: DG2(64xD32 or 32xD64), PVC(64xD32 or 64xD64).
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
/// When \c sizeof(T) equal to 8 the address must be 8-byte aligned,
/// otherwise - 4-byte aligned.
/// When T is 1- or 2-byte type, the memory block is loaded with DWORDs
/// or QWORDs depending on the alignment.
/// Allowed \c NElts values for 8-byte data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 4-byte data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 2-byte data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 1-byte data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
/// 8-byte alignment is required for 8-byte data, or if sizeof(T) * NElts > 256.
/// Otherwise, 4-byte alignment is required.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AccessorT is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed. The default is '1' - perform
/// the operation.
/// @param flags is the alignment specifier type tag.
/// @return is a vector of type T and size NElts. The elements of the returned
/// vector for which the corresponding element in \p pred is 0 are undefined.
///
template <typename T, int NElts, cache_hint L1H, cache_hint L2H,
          typename AccessorT, typename FlagsT>
__ESIMD_API
    std::enable_if_t<detail::is_device_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_read> &&
                         is_simd_flag_type_v<FlagsT>,
                     simd<T, NElts>>
    block_load_impl(AccessorT acc, DeviceAccessorOffsetT offset,
                    simd_mask<1> pred, FlagsT flags) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return block_load_impl<T, NElts, L1H, L2H>(accessorToPointer<T>(acc, offset),
                                             pred, flags);
#else  // !__ESIMD_FORCE_STATELESS_MEM
  // Verify input template arguments.
  check_cache_hint<cache_action::load, L1H, L2H>();
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor64Bit = sizeof(uint64_t) / sizeof(T);
  constexpr int SmallIntFactor32Bit =
      sizeof(uint32_t) / sizeof(T) > 1 ? sizeof(uint32_t) / sizeof(T) : 1;
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");

  // If alignment >= 8 and (NElts * sizeof(T)) % 8 == 0) we can load QWORDs.
  // Don't do it for 4-byte vectors (unless it is greater than 256-bytes),
  // because it would require a bit-cast, which is supposed to be NO-OP, but
  // might confuse GPU BE sometimes. 1- and 2-byte vectors are casted anyways.
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (NElts * sizeof(T)) % sizeof(uint64_t) == 0 &&
      (sizeof(T) != sizeof(uint32_t) || NElts * sizeof(T) > 256);
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  check_lsc_vector_size<FactoredNElts>();

  // Prepare template arguments for the call of intrinsic.
  using LoadElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;

  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size ActualDS =
      Use64BitData ? lsc_data_size::u64 : lsc_data_size::u32;
  constexpr auto VS = to_lsc_vector_size<FactoredNElts>();
  constexpr auto Transposed = lsc_data_order::transpose;
  constexpr int N = 1;

  // Prepare non-template arguments and call the intrinsic.
  simd<uint32_t, N> Offsets = offset;
  auto SI = get_surface_index(acc);
  simd<LoadElemT, FactoredNElts> Result =
      __esimd_lsc_load_bti<LoadElemT, L1H, L2H, AddressScale, ImmOffset,
                           ActualDS, VS, Transposed, N>(pred.data(),
                                                        Offsets.data(), SI);
  return Result.template bit_cast_view<T>();
#endif // !__ESIMD_FORCE_STATELESS_MEM
}

/// Accessor-based transposed gather with 1 channel.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
/// Instruction can load max: DG2(64xD32 or 32xD64), PVC(64xD32 or 64xD64).
///
/// Collects elements located at surface and returns them
/// as a single \ref simd object.
/// When \c sizeof(T) equal to 8 the address must be 8-byte aligned,
/// otherwise - 4-byte aligned.
/// When T is 1- or 2-byte type, the memory block is loaded with DWORDs
/// or QWORDs depending on the alignment.
/// Allowed \c NElts values for 8-byte data are 1, 2, 3, 4, 8, 16, 32, 64.
/// Allowed \c NElts values for 4-byte data are 1, 2, 3, 4, 8, 16, 32, 64, 128.
/// Allowed \c NElts values for 2-byte data are 2, 4, 8, 16, 32, 64, 128, 256.
/// Allowed \c NElts values for 1-byte data are 4, 8, 12, 16, 32, 64, 128, 256,
/// 512.
/// 8-byte alignment is required for 8-byte data, or if sizeof(T) * NElts > 256.
/// Otherwise, 4-byte alignment is required.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param pred is operation predicate. Operation is skipped for index 'i'
/// if pred[0] == 0 the result element is taken from \p pass_thru[i].
/// Otherwise, the operation is performed and the result if it copied to
/// the result.
/// @param pass_thru contains the values copied to the result if \p pred is 0.
/// @param flags is the alignment specifier type tag.
/// @return is a vector of type T and size NElts
///
template <typename T, int NElts, cache_hint L1H, cache_hint L2H,
          typename AccessorT, typename FlagsT>
__ESIMD_API
    std::enable_if_t<detail::is_device_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_read> &&
                         is_simd_flag_type_v<FlagsT>,
                     simd<T, NElts>>
    block_load_impl(AccessorT acc, DeviceAccessorOffsetT offset,
                    simd_mask<1> pred, simd<T, NElts> pass_thru, FlagsT flags) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return block_load_impl<T, NElts, L1H, L2H>(accessorToPointer<T>(acc, offset),
                                             pred, pass_thru, flags);
#else  // !__ESIMD_FORCE_STATELESS_MEM
  // Verify input template arguments.
  check_cache_hint<cache_action::load, L1H, L2H>();
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor64Bit = sizeof(uint64_t) / sizeof(T);
  constexpr int SmallIntFactor32Bit =
      sizeof(uint32_t) / sizeof(T) > 1 ? sizeof(uint32_t) / sizeof(T) : 1;
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");

  // If alignment >= 8 and (NElts * sizeof(T)) % 8 == 0) we can load QWORDs.
  // Don't do it for 4-byte vectors (unless it is greater than 256-bytes),
  // because it would require a bit-cast, which is supposed to be NO-OP, but
  // might confuse GPU BE sometimes. 1- and 2-byte vectors are casted anyways.
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (NElts * sizeof(T)) % sizeof(uint64_t) == 0 &&
      (sizeof(T) != sizeof(uint32_t) || NElts * sizeof(T) > 256);
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  check_lsc_vector_size<FactoredNElts>();

  // Prepare template arguments for the call of intrinsic.
  using LoadElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;

  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size ActualDS =
      Use64BitData ? lsc_data_size::u64 : lsc_data_size::u32;
  constexpr auto VS = to_lsc_vector_size<FactoredNElts>();
  constexpr auto Transposed = lsc_data_order::transpose;
  constexpr int N = 1;

  // Prepare non-template arguments and call the intrinsic.
  simd<uint32_t, N> Offsets = offset;
  auto SI = get_surface_index(acc);
  simd<LoadElemT, FactoredNElts> PassThru =
      pass_thru.template bit_cast_view<LoadElemT>();
  simd<LoadElemT, FactoredNElts> Result =
      __esimd_lsc_load_merge_bti<LoadElemT, L1H, L2H, AddressScale, ImmOffset,
                                 ActualDS, VS, Transposed, N>(
          pred.data(), Offsets.data(), SI, PassThru.data());
  return Result.template bit_cast_view<T>();
#endif // !__ESIMD_FORCE_STATELESS_MEM
}

template <typename T, int NElts, cache_hint L1H, cache_hint L2H,
          typename FlagsT>
__ESIMD_API std::enable_if_t<is_simd_flag_type_v<FlagsT>>
block_store_impl(T *p, simd<T, NElts> vals, simd_mask<1> pred, FlagsT flags) {
  detail::check_cache_hint<cache_action::store, L1H, L2H>();
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor64Bit = sizeof(uint64_t) / sizeof(T);
  constexpr int SmallIntFactor32Bit =
      sizeof(uint32_t) / sizeof(T) > 1 ? sizeof(uint32_t) / sizeof(T) : 1;
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed store");

  // If alignment >= 8 and (NElts * sizeof(T)) % 8 == 0) we can store QWORDs.
  // Don't do it for 4-byte vectors (unless it is greater than 256-bytes),
  // because it would require a bit-cast, which is supposed to be NO-OP, but
  // might confuse GPU BE sometimes. 1- and 2-byte vectors are casted anyways.
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (NElts * sizeof(T)) % sizeof(uint64_t) == 0 &&
      (sizeof(T) != sizeof(uint32_t) || NElts * sizeof(T) > 256);

  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;

  check_lsc_vector_size<FactoredNElts>();

  using StoreType = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size ActualDS =
      Use64BitData ? lsc_data_size::u64 : lsc_data_size::u32;
  constexpr lsc_vector_size VS = to_lsc_vector_size<FactoredNElts>();
  constexpr auto Transposed = lsc_data_order::transpose;
  constexpr int N = 1;
  simd<uintptr_t, N> Addrs = reinterpret_cast<uintptr_t>(p);

  __esimd_lsc_store_stateless<StoreType, L1H, L2H, AddressScale, ImmOffset,
                              ActualDS, VS, Transposed, N>(
      pred.data(), Addrs.data(),
      sycl::bit_cast<__ESIMD_DNS::vector_type_t<StoreType, FactoredNElts>>(
          vals.data()));
}

template <typename T, int NElts, cache_hint L1H, cache_hint L2H,
          typename AccessorT, typename FlagsT>
__ESIMD_API
    std::enable_if_t<detail::is_device_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_write> &&
                     is_simd_flag_type_v<FlagsT>>
    block_store_impl(AccessorT acc, DeviceAccessorOffsetT offset,
                     simd<T, NElts> vals, simd_mask<1> pred, FlagsT flags) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  block_store_impl<T, NElts, L1H, L2H>(accessorToPointer<T>(acc, offset), vals,
                                       pred, flags);
#else
  // Verify input template arguments.
  check_cache_hint<cache_action::store, L1H, L2H>();
  constexpr auto Alignment =
      FlagsT::template alignment<__ESIMD_DNS::__raw_t<T>>;
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor64Bit = sizeof(uint64_t) / sizeof(T);
  constexpr int SmallIntFactor32Bit =
      sizeof(uint32_t) / sizeof(T) > static_cast<size_t>(1)
          ? sizeof(uint32_t) / sizeof(T)
          : static_cast<size_t>(1);
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed store");

  // If alignment >= 8 and (NElts * sizeof(T)) % 8 == 0) we can store QWORDs.
  // Don't do it for 4-byte vectors (unless it is greater than 256-bytes),
  // because it would require a bit-cast, which is supposed to be NO-OP, but
  // might confuse GPU BE sometimes. 1- and 2-byte vectors are casted anyways.
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (NElts * sizeof(T)) % sizeof(uint64_t) == 0 &&
      (sizeof(T) != sizeof(uint32_t) || NElts * sizeof(T) > 256);
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredNElts = NElts / SmallIntFactor;
  check_lsc_vector_size<FactoredNElts>();

  // Prepare template arguments for the call of intrinsic.
  using StoreElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;

  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size ActualDS =
      Use64BitData ? lsc_data_size::u64 : lsc_data_size::u32;
  constexpr auto VS = to_lsc_vector_size<FactoredNElts>();
  constexpr auto Transposed = lsc_data_order::transpose;
  constexpr int N = 1;

  // Prepare non-template arguments and call the intrinsic.
  simd<uint32_t, N> Offsets = offset;
  auto SI = get_surface_index(acc);

  __esimd_lsc_store_bti<StoreElemT, L1H, L2H, AddressScale, ImmOffset, ActualDS,
                        VS, Transposed, N>(
      pred.data(), Offsets.data(),
      sycl::bit_cast<__ESIMD_DNS::vector_type_t<StoreElemT, FactoredNElts>>(
          vals.data()),
      SI);
#endif
}

} // namespace detail

/// Stores elements of the vector \p vals to a contiguous block of memory
/// at the given address \p addr.
/// The generated code depends on the combination {T, N, Flags}.
/// Providing flags specifying the alignment of 16-bytes or more produces more
/// efficient code. If the alignment is smaller than 16-bytes, then less
/// efficient scatter is generated. If the stored vector is too long
/// for 1 flat-store GPU instruction, then a series of flat-store and/or
/// scatters may be generated.
/// @tparam Tx Element type.
/// @tparam N Number of elements to store.
/// @tparam Flags The alignment specifier type tag.
/// @param addr The memory address to store at.
/// @param vals The vector to store.
/// @param Flags Specifies the alignment.
template <typename Tx, int N,
          typename Flags = overaligned_tag<detail::OperandSize::OWORD>>
__ESIMD_API std::enable_if_t<is_simd_flag_type_v<Flags>>
block_store(Tx *addr, simd<Tx, N> vals, Flags) {
  using T = typename detail::__raw_t<Tx>;
  using VecT = typename simd<T, N>::raw_vector_type;
  constexpr size_t Align = Flags::template alignment<simd<T, N>>;
  __esimd_svm_block_st<T, N, Align>(reinterpret_cast<VecT *>(addr),
                                    vals.data());
}

/// @addtogroup sycl_esimd_memory_block
/// @{

/// Each of the following block load functions loads a contiguous memory block
/// from the address referenced by the USM pointer 'ptr', or from 'ptr +
/// offset', where 'offset' is the offset in bytes (not in elements!). The
/// parameter 'pred' is the one element predicate. If it is set to 1, then all
/// 'N' elements are loaded. Otherwise, the block load operation is a NO-OP.
/// The parameter 'pass_thru' specifies the values being copied to the returned
/// result if 'pred' is set to 0.
/// The parameter 'props' specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::cache_hint_L3, esimd::alignment.

/// simd<T, N> block_load(const T* ptr, props={});                 // (usm-bl-1)
/// simd<T, N> block_load(const T* ptr, size_t byte_offset,
///                       props={});                               // (usm-bl-2)

/// simd<T, N> block_load(const T* ptr, simd_mask<1> pred,
///                       props={});                               // (usm-bl-3)
/// simd<T, N> block_load(const T* ptr, size_t byte_offset,
///                       simd_mask<1> pred, props={});            // (usm-bl-4)

/// simd<T, N> block_load(const T* ptr, simd_mask<1> pred,
///                       simd<T, N> pass_thru, props={});         // (usm-bl-5)
/// simd<T, N> block_load(const T* ptr, size_t byte_offset,
///                       simd_mask<1> pred, simd<T, N> pass_thru,
///                       props={});                               // (usm-bl-6)

/// simd<T, N> block_load(const T* ptr, props={});                 // (usm-bl-1)
/// This function loads a contiguous memory block from USM pointer \p ptr.
///
/// There may be temporary restrictions depending on L1, L2 cache hints,
/// See details in the 'Restrictions' section below. The restrictions will be
/// relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is 4-bytes for 4-byte or smaller elements
/// and 8-bytes for 8-byte elements. The address may be element-size aligned
/// even for byte- and word-elements, but in such case the smaller alignment
/// property must explicitly passed to this function. Extra restrictions
/// may be in place - see Restrictions/R1 below.
///
/// Restrictions - cache hint imposed - temporary:
/// If L1 or L2 cache hint is passed, then:
/// R1: The pointer must be at least 4-byte aligned for elements of 4-bytes or
///     smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
block_load(const T *ptr, PropertyListT props = {}) {
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);
  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none) {
    detail::check_cache_hint<detail::cache_action::load, L1Hint, L2Hint>();

    simd_mask<1> Mask = 1;
    return detail::block_load_impl<T, N, L1Hint, L2Hint>(
        ptr, Mask, overaligned_tag<Alignment>{});
  } else {
    return block_load<T, N>(ptr, overaligned_tag<Alignment>{});
  }
}

/// simd<T, N> block_load(const T* ptr, size_t byte_offset,
///                       props={});  // (usm-bl-2)
/// This function loads a contiguous memory block from address referenced
/// by USM pointer \p ptr and the given \p byte_offset.
///
/// There may be temporary restrictions depending on L1, L2 cache hints,
/// See details in the 'Restrictions' section below. The restrictions will be
/// relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is 4-bytes for 4-byte or smaller elements
/// and 8-bytes for 8-byte elements. The address may be element-size aligned
/// even for byte- and word-elements, but in such case the smaller alignment
/// property must explicitly passed to this function. Extra restrictions
/// may be in place - see Restrictions/R1 below.
///
/// Restrictions - cache hint imposed - temporary:
/// If L1 or L2 cache hint is passed, then:
/// R1: The pointer must be at least 4-byte aligned for elements of 4-bytes or
///     smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
block_load(const T *ptr, size_t byte_offset, PropertyListT props = {}) {
  const T *AdjustedPtr = reinterpret_cast<const T *>(
      reinterpret_cast<const int8_t *>(ptr) + byte_offset);
  return block_load<T, N>(AdjustedPtr, props);
}

/// simd<T, N> block_load(const T* ptr, simd_mask<1> pred,
///                       props={});                               // (usm-bl-3)
/// This function loads a contiguous memory block from USM pointer \p ptr.
/// If the predicate \p pred is set to 0, then the load is omitted and the
/// returned value is undefined.
///
/// This function has temporary restrictions. See details in the 'Restrictions'
/// section below. The restrictions will be relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is the minimally required element-size
/// alignment. Note that additional/temporary restrictions are applied
/// (see Restrictions below).
///
/// Restrictions - cache hint and mask imposed - temporary:
/// R1: The pointer must be at least 4-byte aligned for elements of 4-bytes or
///     smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
block_load(const T *ptr, simd_mask<1> pred, PropertyListT props = {}) {
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  detail::check_cache_hint<detail::cache_action::load, L1Hint, L2Hint>();
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);

  return detail::block_load_impl<T, N, L1Hint, L2Hint>(
      ptr, pred, overaligned_tag<Alignment>{});
}

/// simd<T, N> block_load(const T* ptr, size_t byte_offset,
///                       simd_mask<1> pred, props={});            // (usm-bl-4)
/// This function loads a contiguous memory block from address referenced
/// by USM pointer \p ptr and the given \p byte_offset.
/// If the predicate \p pred is set to 0, then the load is omitted and the
/// returned value is undefined.
///
/// This function has temporary restrictions. See details in the 'Restrictions'
/// section below. The restrictions will be relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is the minimally required element-size
/// alignment. Note that additional/temporary restrictions are applied
/// (see Restrictions below).
///
/// Restrictions - cache hint and mask imposed - temporary:
/// R1: The pointer must be at least 4-byte aligned for elements of 4-bytes or
///     smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
block_load(const T *ptr, size_t byte_offset, simd_mask<1> pred,
           PropertyListT props = {}) {
  const T *AdjustedPtr = reinterpret_cast<const T *>(
      reinterpret_cast<const int8_t *>(ptr) + byte_offset);
  return block_load<T, N>(AdjustedPtr, pred, props);
}

/// simd<T, N> block_load(const T* ptr, simd_mask<1> pred,
///                       simd<T, N> pass_thru, props={});         // (usm-bl-5)
/// This function loads a contiguous memory block from USM pointer \p ptr.
/// If the predicate \p pred is set to 0, then the load is omitted and the
/// vector \p pass_thru is returned.
///
/// This function has temporary restrictions. See details in the 'Restrictions'
/// section below. The restrictions will be relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is the minimally required element-size
/// alignment. Note that additional/temporary restrictions are applied
/// (see Restrictions below).
///
/// Restrictions - cache hint and mask imposed - temporary:
/// R1: The pointer must be at least 4-byte aligned for elements of 4-bytes or
///     smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
block_load(const T *ptr, simd_mask<1> pred, simd<T, N> pass_thru,
           PropertyListT props = {}) {
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  detail::check_cache_hint<detail::cache_action::load, L1Hint, L2Hint>();
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);

  return detail::block_load_impl<T, N, L1Hint, L2Hint>(
      ptr, pred, pass_thru, overaligned_tag<Alignment>{});
}

/// simd<T, N> block_load(const T* ptr, size_t byte_offset,
///                       simd_mask<1> pred, simd<T, N> pass_thru,
///                       props={});                               // (usm-bl-6)
/// This function loads a contiguous memory block from address referenced
/// by USM pointer \p ptr and the given \p byte_offset.
/// If the predicate \p pred is set to 0, then the load is omitted and the
/// vector \p pass_thru is returned.
///
/// This function has temporary restrictions. See details in the 'Restrictions'
/// section below. The restrictions will be relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is the minimally required element-size
/// alignment. Note that additional/temporary restrictions are applied
/// (see Restrictions below).
///
/// Restrictions - cache hint and mask imposed - temporary:
/// R1: The pointer must be at least 4-byte aligned for elements of 4-bytes or
///     smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
block_load(const T *ptr, size_t byte_offset, simd_mask<1> pred,
           simd<T, N> pass_thru, PropertyListT props = {}) {
  const T *AdjustedPtr = reinterpret_cast<const T *>(
      reinterpret_cast<const int8_t *>(ptr) + byte_offset);
  return block_load<T, N>(AdjustedPtr, pred, pass_thru, props);
}

/// Loads a contiguous block of memory from the given memory address \p addr
/// and returns the loaded data as a vector.
/// The generated code depends on the combination {T, N, Flags}.
/// Providing flags specifying the alignment of 16-bytes or more produces more
/// efficient code. If the alignment is smaller than 16-bytes, then less
/// efficient gather is generated. If the loaded vector is too long
/// for 1 flat-load GPU instruction, then a series of flat-loads and/or gathers
/// may be generated.
/// @tparam Tx Element type.
/// @tparam N Number of elements to load.
/// @tparam Flags The alignment specifier type tag.
/// @param addr The address to load from.
/// @param Flags Specifies the alignment.
/// @return A vector of loaded elements.
///
template <typename Tx, int N,
          typename Flags = overaligned_tag<detail::OperandSize::OWORD>>
__ESIMD_API std::enable_if_t<is_simd_flag_type_v<Flags>, simd<Tx, N>>
block_load(const Tx *addr, Flags) {
  using T = typename detail::__raw_t<Tx>;
  using VecT = typename simd<T, N>::raw_vector_type;
  constexpr size_t Align = Flags::template alignment<simd<T, N>>;
  return __esimd_svm_block_ld<T, N, Align>(
      reinterpret_cast<const VecT *>(addr));
}

/// Loads a contiguous block of memory from the given accessor \p acc and
/// \p byte_offset and returns the loaded data as a vector.
/// Actual code generated depends on the alignment parameter.
/// @tparam Tx Element type.
/// @tparam N Number of elements to load, <code>N * sizeof(Tx)</code> must be
///    1, 2, 4 or 8 owords long.
/// @tparam AccessorTy Accessor type (auto-deduced).
/// @tparam Flags The alignment specifier type tag. Auto-deduced from the
///    \c Flags parameter. If it is less than \c 16, then slower unaligned
///    access is generated, otherwise the access is aligned.
/// @param acc The accessor.
/// @param byte_offset The offset to load from in bytes.
/// @param Flags Specifies the alignment.
/// @return A vector of loaded elements.
///
template <typename Tx, int N, typename AccessorTy,
          typename Flags = vector_aligned_tag,
          typename = std::enable_if_t<
              is_simd_flag_type_v<Flags> &&
              detail::is_device_accessor_with_v<
                  AccessorTy, detail::accessor_mode_cap::can_read>>,
          class T = detail::__raw_t<Tx>>
__ESIMD_API simd<Tx, N> block_load(AccessorTy acc,
                                   detail::DeviceAccessorOffsetT byte_offset,
                                   Flags flags) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return block_load<Tx, N>(__ESIMD_DNS::accessorToPointer<Tx>(acc, byte_offset),
                           flags);
#else
  std::ignore = flags;
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");

  auto surf_ind = __esimd_get_surface_index(
      detail::AccessorPrivateProxy::getQualifiedPtrOrImageObj(acc));

  if constexpr (Flags::template alignment<simd<T, N>> >=
                detail::OperandSize::OWORD) {
    return __esimd_oword_ld<T, N>(surf_ind, byte_offset >> 4);
  } else {
    return __esimd_oword_ld_unaligned<T, N>(surf_ind, byte_offset);
  }
#endif
}

/// Each of the following block load functions loads a contiguous memory block
/// from the address referenced by accessor 'acc', or from 'acc + byte_offset',
/// The parameter 'pred' is the one element predicate. If it is set to 1, then
/// all 'N' elements are loaded. Otherwise, the block load operation is a NO-OP.
/// The parameter 'pass_thru' specifies the values being copied to the returned
/// result if 'pred' is set to 0.
/// The parameter 'props' specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::cache_hint_L3, esimd::alignment.

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT byte_offset, props = {});    // (acc-bl-1)
/// simd<T, N> block_load(AccessorT acc, props = {});              // (acc-bl-2)

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT byte_offset, simd_mask<1> pred,
///            simd<T, N> pass_thru, props = {});                  // (acc-bl-3)
/// simd<T, N>
/// block_load(AccessorT acc, OffsetT byte_offset,
///            simd_mask<1> pred, props = {});                     // (acc-bl-4)

/// simd<T, N>
/// block_load(AccessorT acc, simd_mask<1> pred,
///            simd<T, N> pass_thru, props = {});                  // (acc-bl-5)
/// simd<T, N>
/// block_load(AccessorT acc, simd_mask<1> pred, props = {});      // (acc-bl-6)

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT byte_offset, props = {});    // (acc-bl-1)
/// This function loads a contiguous memory block referenced
/// by accessor \p acc and \p byte_offset.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the \p byte_offset must be at least 4-byte aligned for elements of 4-bytes
/// or smaller and 8-byte aligned for 8-byte elements.
/// The alignment requirement may be less strict if stateless memory mode is ON,
/// see block_load(usm_ptr, props) (aka usm-bl-01) for details/requirements.
///
/// Restrictions: there may be some extra restrictions depending on
///    a) stateless memory mode enforcement is ON,
///    b) cache hints are used,
///    c) number of bytes loaded is either 16,32,64, or 128.
/// If (b) || !(c), then the target device must be DG2 or PVC (not Gen12).
/// If (a) && !(b), then there is no restriction on the number of elements
/// to be loaded and \p byte_offset must be only element-aligned.
///
/// Gen12 requirements: !(b) && (c).
///   It can load 16-, 32-, 64-, or 128-bytes only.
/// DG2/PVC requirements:
///   It can load such number of elements depending on the type 'T':
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, detail::DeviceAccessorOffsetT byte_offset,
           PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return block_load<T, N>(detail::accessorToPointer<T>(acc, byte_offset),
                          props);
#else  // !__ESIMD_FORCE_STATELESS_MEM
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  // If the alignment property is not passed, then assume the pointer
  // is element-aligned.
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);

  // Legacy surface index loads must be 1, 2, 4 or 8 owords long.
  constexpr size_t Size = sizeof(T) * N;
  constexpr size_t OWord = detail::OperandSize::OWORD;
  constexpr bool IsLegacySize = Size == OWord || Size == 2 * OWord ||
                                Size == 4 * OWord || Size == 8 * OWord;

  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                !IsLegacySize) {
    return detail::block_load_impl<T, N, L1Hint, L2Hint>(
        acc, byte_offset, simd_mask<1>(1), overaligned_tag<Alignment>{});
  } else {
    return block_load<T, N>(acc, byte_offset, overaligned_tag<Alignment>{});
  }
#endif // !__ESIMD_FORCE_STATELESS_MEM
}

/// simd<T, N> block_load(AccessorT acc, props = {});              // (acc-bl-2)
/// This function loads a contiguous memory block referenced
/// by accessor \p acc and implied offset=0.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2. Other properties are ignored. If \p props specifies
/// the alignment property, then it is ignored because this variant implies
/// zero offset, which means the most favourable 16-byte alignment is used.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Restrictions: there may be some extra restrictions depending on
///    a) stateless memory mode enforcement is ON,
///    b) cache hints are used,
///    c) number of bytes loaded is either 16,32,64, or 128.
/// If (b) || !(c), then the target device must be DG2 or PVC (not Gen12).
/// If (a) && !(b), then there is no restriction on the number of elements
/// to be loaded and \p byte_offset must be only element-aligned.
///
/// Gen12 requirements: !(b) && (c).
///   It can load 16-, 32-, 64-, or 128-bytes only.
/// DG2/PVC requirements:
///   It can load such number of elements depending on the type 'T':
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2), or 128;
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2), or 256;
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2), or 512.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, PropertyListT /* props */ = {}) {
  // Create new properties without the alignment property passed in 'props',
  // and add alignment<16> as it is usable and most favourable in this case.
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");
  properties Props{cache_hint_L1<L1Hint>, cache_hint_L2<L2Hint>, alignment<16>};
  return block_load<T, N>(acc, 0, Props);
}

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT byte_offset, simd_mask<1> pred,
///            simd<T, N> pass_thru, props = {});                  // (acc-bl-3)
/// This function loads a contiguous memory block referenced
/// by accessor \p acc and the given \p byte_offset.
/// If the predicate \p pred is set to 0, then the load is omitted and the
/// \p pass_thru value is returned.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the \p byte_offset must be at least 4-byte aligned for elements of 4-bytes
/// or smaller and 8-byte aligned for 8-byte elements.
///
/// Restrictions - cache hint and predicate imposed - temporary:
/// R1: \p byte_offset must be at least 4-byte aligned for elements of 4-bytes
///     or  smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, detail::DeviceAccessorOffsetT byte_offset,
           simd_mask<1> pred, simd<T, N> pass_thru,
           PropertyListT /* props */ = {}) {
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  // If the alignment property is not passed, then assume the byte_offset
  // is element-aligned and is at leat 4-bytes.
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);
  return detail::block_load_impl<T, N, L1Hint, L2Hint>(
      acc, byte_offset, pred, pass_thru, overaligned_tag<Alignment>{});
}

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT byte_offset, simd_mask<1> pred,
///            props = {});                                        // (acc-bl-4)
/// This function loads a contiguous memory block referenced
/// by accessor \p acc and the given \p byte_offset.
/// If the predicate \p pred is set to 0, then the load is omitted and the
/// returned value is undefined.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the \p offset must be at least 4-byte aligned for elements of 4-bytes or
///     smaller and 8-byte aligned for 8-byte elements.
///
/// Restrictions - cache hint and predicate imposed - temporary:
/// R1: \p byte_offset must be at least 4-byte aligned for elements of 4-bytes
///     or  smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, detail::DeviceAccessorOffsetT byte_offset,
           simd_mask<1> pred, PropertyListT props = {}) {
  simd<T, N> PassThru; // Intentionally uninitialized.
  return block_load<T, N>(acc, byte_offset, pred, PassThru, props);
}

/// simd<T, N>
/// block_load(AccessorT acc, simd_mask<1> pred,
///            simd<T, N> pass_thru, props = {});                  // (acc-bl-5)
/// This function loads a contiguous memory block referenced
/// by accessor \p acc and implied offset=0.
/// If the predicate \p pred is set to 0, then the load is omitted and the
/// \p pass_thru value is returned.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2. Other properties are ignored. If \p props specifies
/// the alignment property, then it is ignored because this variant implies
/// zero offset, which means the most favourable 16-byte alignment is used.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Restrictions - cache hint and predicate imposed - temporary:
/// R1: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R2: The target device must be DG2, PVC or newer GPU.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, simd_mask<1> pred, simd<T, N> pass_thru,
           PropertyListT /* props */ = {}) {
  // Create new properties without the alignment property passed in 'props',
  // and add alignment<16> as it is usable and most favourable in this case.
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");
  properties Props{cache_hint_L1<L1Hint>, cache_hint_L2<L2Hint>, alignment<16>};
  return block_load<T, N>(acc, 0, pred, pass_thru, Props);
}

/// simd<T, N>
/// block_load(AccessorT acc, simd_mask<1> pred, props = {});      // (acc-bl-6)
/// This function loads a contiguous memory block referenced
/// by accessor \p acc and implied offset=0.
/// If the predicate \p pred is set to 0, then the load is omitted and some
/// undefined value is returned.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2. Other properties are ignored. If \p props specifies
/// the alignment property, then it is ignored because this variant implies
/// zero offset, which means the most favourable 16-byte alignment is used.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Restrictions - cache hint and predicate imposed - temporary:
/// R1: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R2: The target device must be DG2, PVC or newer GPU.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, simd_mask<1> pred, PropertyListT /* props */ = {}) {
  // Create new properties without the alignment property passed in 'props',
  // and add alignment<16> as it is usable and most favourable in this case.
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");
  properties Props{cache_hint_L1<L1Hint>, cache_hint_L2<L2Hint>, alignment<16>};

  simd<T, N> PassThru; // Intentionally uninitialized.
  return block_load<T, N>(acc, 0, pred, PassThru, Props);
}

/// Each of the following block store functions stores a contiguous memory block
/// to the address referenced by the USM pointer 'ptr', or from 'ptr +
/// offset', where 'offset' is the offset in bytes (not in elements!) with data
/// specified by 'vals'.
/// The parameter 'pred' is the one element predicate. If it is set to 1, then
/// all 'N' elements are stored. Otherwise, the block store operation is a
/// NO-OP. The parameter 'props' specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::cache_hint_L3, esimd::alignment.
///
/// void block_store(T* ptr, simd<T, N> vals, props={}); // (usm-bs-1)
/// void block_store(T* ptr, size_t byte_offset,         // (usm-bs-2)
///                          simd<T, N> vals, props={});

/// void block_store(T* ptr, simd<T, N> vals,            // (usm-bs-3)
///             simd_mask<1> pred, props={});

/// void block_store(T* ptr, size_t byte_offset,         // (usm-bs-4)
/// simd<T, N> vals, simd_mask<1> pred, props={});
///
/// void block_store(T* ptr, simd<T, N> vals, props={}); // (usm-bs-1)
/// This function stores a contiguous memory block to USM pointer \p ptr
/// with data specified by \p vals.
///
/// There may be temporary restrictions depending on L1, L2 cache hints,
/// See details in the 'Restrictions' section below. The restrictions will be
/// relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is 16 bytes if \p props does not specify any
/// L1 or L2 cache hints, and the minimally required element-size
/// alignment otherwise. Note that additional/temporary restrictions may apply
/// (see Restrictions below).
///
/// Restrictions - cache hint imposed - temporary:
/// If L1 or L2 cache hint is passed, then:
/// R1: The pointer must be at least 4-byte aligned for elements of 4-bytes or
///     smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(T *ptr, simd<T, N> vals, PropertyListT props = {}) {
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");
  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none) {
    detail::check_cache_hint<detail::cache_action::store, L1Hint, L2Hint>();
    constexpr int DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
    constexpr size_t Alignment =
        detail::getPropertyValue<PropertyListT, alignment_key>(
            DefaultAlignment);

    simd_mask<1> Mask = 1;
    detail::block_store_impl<T, N, L1Hint, L2Hint>(
        ptr, vals, Mask, overaligned_tag<Alignment>{});
  } else {
    // If the alignment property is not passed, then assume the pointer
    // is OWORD-aligned.
    constexpr size_t Alignment =
        detail::getPropertyValue<PropertyListT, alignment_key>(
            detail::OperandSize::OWORD);
    block_store<T, N>(ptr, vals, overaligned_tag<Alignment>{});
  }
}

/// void block_store(T* ptr, size_t byte_offset,         // (usm-bs-2)
///                          simd<T, N> vals, props={});
/// This function stores a contiguous memory block to USM pointer \p ptr and
/// byte-offset \p byte_offset with data specified by \p vals.
///
/// There may be temporary restrictions depending on L1, L2 cache hints,
/// See details in the 'Restrictions' section below. The restrictions will be
/// relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is 16 bytes if \p props does not specify any
/// L1 or L2 cache hints, and the minimally required element-size
/// alignment otherwise. Note that additional/temporary restrictions may apply
/// (see Restrictions below).
///
/// Restrictions - cache hint imposed - temporary:
/// If L1 or L2 cache hint is passed, then:
/// R1: The pointer plus byte offset must be at least 4-byte aligned for
/// elements of 4-bytes or smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(T *ptr, size_t byte_offset, simd<T, N> vals,
            PropertyListT props = {}) {
  T *AdjustedPtr =
      reinterpret_cast<T *>(reinterpret_cast<int8_t *>(ptr) + byte_offset);
  block_store<T, N>(AdjustedPtr, vals, props);
}

/// void block_store(T* ptr, simd<T, N> vals,            // (usm-bs-3)
///             simd_mask<1> pred, props={});
/// This function stores a contiguous memory block to USM pointer \p ptr
/// with data specified by \p vals. If the predicate \p pred is set to 0,
/// then the store is omitted.
///
/// There are temporary restrictions.  See details in the 'Restrictions'
/// section below. The restrictions will be relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is the minimally required element-size
/// alignment. Note that additional/temporary restrictions apply (see
/// Restrictions below).
///
/// Restrictions - predicate imposed - temporary:
/// R1: The pointer must be at least 4-byte aligned for elements of 4-bytes or
///     smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(T *ptr, simd<T, N> vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);

  detail::block_store_impl<T, N, L1Hint, L2Hint>(ptr, vals, pred,
                                                 overaligned_tag<Alignment>{});
}

/// void block_store(T* ptr, size_t byte_offset,         // (usm-bs-4)
/// simd<T, N> vals, simd_mask<1> pred, props={});
/// This function stores a contiguous memory block to USM pointer \p ptr
/// and byte-offset \p byte_offset with data specified by \p vals.
/// If the predicate \p pred is set to 0, then the store is omitted.
///
/// There may be temporary restrictions depending on L1, L2 cache hints,
/// See details in the 'Restrictions' section below. The restrictions will be
/// relaxed in the future.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default assumed alignment is 16 bytes if \p props does not specify any
/// L1 or L2 cache hints and \p pred is set to 1, and
//  the minimally required element-size alignment otherwise.
/// Note that additional/temporary restrictions may apply
/// (see Restrictions below).
///
/// Restrictions - cache hint or predicate imposed - temporary:
/// If a predicate, L1 or L2 cache hint is passed, then:
/// R1: The pointer plus byte offset must be at least 4-byte aligned for
/// elements of 4-bytes or smaller and 8-byte aligned for 8-byte elements.
/// R2: The number of elements for 8-byte data: 1, 2, 3, 4, 8, 16, 32, 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64,
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128,
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256,
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(T *ptr, size_t byte_offset, simd<T, N> vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  T *AdjustedPtr =
      reinterpret_cast<T *>(reinterpret_cast<int8_t *>(ptr) + byte_offset);
  block_store<T, N>(AdjustedPtr, vals, pred, props);
}

/// Each of the following block_store functions stores the vector 'vals' to a
/// contiguous memory block at the address referenced by accessor 'acc', or from
/// 'acc + byte_offset', The parameter 'pred' is the one element predicate. If
/// it is set to 1, then all 'N' elements are stored. Otherwise, the block store
/// operation is a NO-OP. The parameter 'props' specifies the optional
/// compile-time properties of the type esimd::properties and may include
/// esimd::cache_hint_L1, esimd::cache_hint_L2, esimd::cache_hint_L3,
/// esimd::alignment.

/// void block_store(AccessorT acc, OffsetT byte_offset,          // (acc-bs-1)
///                   simd<T, N> vals, props = {});

/// void block_store(AccessorT acc, simd<T, N> vals, props = {}); // (acc-bs-2)
/// void block_store(AccessorT acc, OffsetT byte_offset,          // (acc-bs-3)
///     simd<T, N> vals, simd_mask<1> pred, props = {});

/// void block_store(AccessorT acc, simd<T, N> vals,              // (acc-bs-4)
///                  simd_mask<1> pred, props = {});

/// void block_store(AccessorT acc, OffsetT byte_offset,          // (acc-bs-1)
///                   simd<T, N> vals, props = {});
/// This function stores a contiguous memory block to
/// accessor \p acc and \p byte_offset with data specified by \p vals.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the \p byte_offset must be at least 16-byte aligned if (!(b) && (c))
/// from the below restrictions, and must be at least 4-byte aligned for
/// elements of 4-bytes or smaller and 8-byte aligned for 8-byte elements
/// otherwise. If the 'alignment' property is specified as less than 16 bytes,
/// then the target device must be DG2 or PVC (not Gen12). The alignment
/// requirement may be less strict if stateless memory mode is ON, see
/// block_store(usm_ptr, props) (aka usm-bs-01) for details/requirements.
///
/// Restrictions: there may be some extra restrictions depending on
///    a) stateless memory mode enforcement is ON,
///    b) cache hints are used,
///    c) number of bytes stored is either 16,32,64, or 128.
///    d) the 'alignment' property is specified as less than 16 bytes.
///
/// If (b) || !(c) || (d), then the target device must be DG2 or PVC (not
/// Gen12).
/// If (a) && !(b), then there is no restriction on the number of
/// elements to be stored and \p byte_offset must be only element-aligned.
///
/// Gen12 requirements: !(b) && (c) && !(d).
///   It can store 16-, 32-, 64-, or 128-bytes only.
/// DG2/PVC requirements:
///   It can store such number of elements depending on the type 'T':
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, detail::DeviceAccessorOffsetT byte_offset,
            simd<T, N> vals, PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  block_store<T, N>(detail::accessorToPointer<T>(acc, byte_offset), vals,
                    props);
#else
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");
  constexpr int DefaultLSCAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(
          DefaultLSCAlignment);
  constexpr bool AlignmentRequiresLSC =
      PropertyListT::template has_property<alignment_key>() && Alignment < 16;
  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                AlignmentRequiresLSC) {
    detail::check_cache_hint<detail::cache_action::store, L1Hint, L2Hint>();
    simd_mask<1> Mask = 1;
    detail::block_store_impl<T, N, L1Hint, L2Hint>(
        acc, byte_offset, vals, Mask, overaligned_tag<Alignment>{});
  } else {
    using Tx = detail::__raw_t<T>;
    constexpr unsigned Sz = sizeof(Tx) * N;
    static_assert(Sz >= detail::OperandSize::OWORD,
                  "block size must be at least 1 oword");
    static_assert(Sz % detail::OperandSize::OWORD == 0,
                  "block size must be whole number of owords");
    static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                  "block must be 1, 2, 4 or 8 owords long");
    static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                  "block size must be at most 8 owords");

    auto surf_ind = __esimd_get_surface_index(
        detail::AccessorPrivateProxy::getQualifiedPtrOrImageObj(acc));
    __esimd_oword_st<Tx, N>(surf_ind, byte_offset >> 4, vals.data());
  }
#endif
}

/// void block_store(AccessorT acc, simd<T, N> vals, props = {}); // (acc-bs-2)
/// This function stores a contiguous memory block to
/// accessor \p acc with data specified by \p vals and implied offset=0.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2. Other properties are ignored. If \p props specifies
/// the alignment property, then it is ignored because this variant implies
/// zero offset, which means the most favourable 16-byte alignment is used.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Restrictions: there may be some extra restrictions depending on
///    a) stateless memory mode enforcement is ON,
///    b) cache hints are used,
///    c) number of bytes stored is either 16,32,64, or 128.
/// If (b) || !(c), then the target device must be DG2 or PVC (not Gen12).
/// If (a) && !(b), then there is no restriction on the number of elements
/// to be stored.
///
/// Gen12 requirements: !(b) && (c).
///   It can store 16-, 32-, 64-, or 128-bytes only.
/// DG2/PVC requirements:
///   It can store such number of elements depending on the type 'T':
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2), or 128;
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2), or 256;
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2), or 512.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, simd<T, N> vals, PropertyListT props = {}) {
  // Create new properties without the alignment property passed in 'props',
  // and add alignment<16> as it is usable and most favourable in this case.
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");
  properties Props{cache_hint_L1<L1Hint>, cache_hint_L2<L2Hint>, alignment<16>};

  block_store<T, N>(acc, 0, vals, Props);
}

/// void block_store(AccessorT acc, OffsetT byte_offset,          // (acc-bs-3)
///     simd<T, N> vals, simd_mask<1> pred, props = {});
/// This function stores a contiguous memory block to
/// accessor \p acc and \p byte_offset with data specified by \p vals.
/// If the predicate \p pred is set to 0, then the store is omitted.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::alignment. Other properties are ignored.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the \p byte_offset must be at least 4-byte aligned for elements of 4-bytes
/// or smaller and 8-byte aligned for 8-byte elements.
/// The alignment requirement may be less strict if stateless memory mode is ON,
/// see block_store(usm_ptr, props) (aka usm-bs-01) for details/requirements.
///
/// Restrictions:
/// R1: The target device must be DG2 or PVC (not Gen12).
///
/// R2:
///   It can store such number of elements depending on the type 'T':
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, detail::DeviceAccessorOffsetT byte_offset,
            simd<T, N> vals, simd_mask<1> pred, PropertyListT props = {}) {
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);

  detail::block_store_impl<T, N, L1Hint, L2Hint>(acc, byte_offset, vals, pred,
                                                 overaligned_tag<Alignment>{});
}

/// void block_store(AccessorT acc, simd<T, N> vals,              // (acc-bs-4)
///                  simd_mask<1> pred, props = {});
/// This function stores a contiguous memory block to
/// accessor \p acc with data specified by \p vals and implied offset=0.
/// If the predicate \p pred is set to 0, then the store is omitted.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2. Other properties are ignored. If \p props specifies
/// the alignment property, then it is ignored because this variant implies
/// zero offset, which means the most favourable 16-byte alignment is used.
///
/// Cache hints: If \p props does not specify any L1 or L2 cache hints, then
/// the cache_hint::none value is assumed by default.
///
/// Restrictions:
/// R1: The target device must be DG2 or PVC (not Gen12).
///
/// R2:
///   It can store such number of elements depending on the type 'T':
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2), or 128;
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2), or 256;
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2), or 512.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, simd<T, N> vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  // Create new properties without the alignment property passed in 'props',
  // and add alignment<16> as it is usable and most favourable in this case.
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");
  properties Props{cache_hint_L1<L1Hint>, cache_hint_L2<L2Hint>, alignment<16>};
  block_store<T, N>(acc, 0, vals, pred, Props);
}

/// @} sycl_esimd_memory_block

/// @} sycl_esimd_memory

/// @cond ESIMD_DETAIL

// Implementations of accessor-based gather and scatter functions
namespace detail {
template <typename T, int N, typename AccessorTy>
ESIMD_INLINE ESIMD_NODEBUG std::enable_if_t<
    (sizeof(T) <= 4) && (N == 1 || N == 8 || N == 16 || N == 32) &&
    (std::is_same_v<detail::LocalAccessorMarker, AccessorTy> ||
     is_accessor_with_v<AccessorTy, detail::accessor_mode_cap::can_write>)>
scatter_impl(AccessorTy acc, simd<T, N> vals, simd<uint32_t, N> offsets,
             uint32_t glob_offset, simd_mask<N> mask) {
  constexpr int TypeSizeLog2 = detail::ElemsPerAddrEncoding<sizeof(T)>();
  // TODO (performance) use hardware-supported scale once BE supports it
  constexpr int16_t scale = 0;
  const auto si = __ESIMD_NS::get_surface_index(acc);

  if constexpr (sizeof(T) < 4) {
    using Tint = std::conditional_t<std::is_integral_v<T>, T,
                                    detail::uint_type_t<sizeof(T)>>;
    using Treal = __raw_t<T>;
    simd<Tint, N> vals_int = bitcast<Tint, Treal, N>(std::move(vals).data());
    using PromoT = typename std::conditional_t<std::is_signed<Tint>::value,
                                               int32_t, uint32_t>;
    const simd<PromoT, N> promo_vals = convert<PromoT>(std::move(vals_int));
    __esimd_scatter_scaled<PromoT, N, decltype(si), TypeSizeLog2, scale>(
        mask.data(), si, glob_offset, offsets.data(), promo_vals.data());
  } else {
    using Treal = __raw_t<T>;
    if constexpr (!std::is_same_v<Treal, T>) {
      simd<Treal, N> Values = vals.template bit_cast_view<Treal>();
      __esimd_scatter_scaled<Treal, N, decltype(si), TypeSizeLog2, scale>(
          mask.data(), si, glob_offset, offsets.data(), Values.data());
    } else {
      __esimd_scatter_scaled<T, N, decltype(si), TypeSizeLog2, scale>(
          mask.data(), si, glob_offset, offsets.data(), vals.data());
    }
  }
}

#ifndef __ESIMD_FORCE_STATELESS_MEM
template <typename T, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<
    (std::is_same_v<detail::LocalAccessorMarker, AccessorTy> ||
     is_accessor_with_v<AccessorTy, detail::accessor_mode_cap::can_read>),
    simd<T, N>>
gather_impl(AccessorTy acc, simd<uint32_t, N> offsets, uint32_t glob_offset,
            simd_mask<N> mask) {
  static_assert(sizeof(T) <= 4 && (N == 1 || N == 8 || N == 16 || N == 32),
                "Unexpected type or vector length");

  constexpr int TypeSizeLog2 = detail::ElemsPerAddrEncoding<sizeof(T)>();
  // TODO (performance) use hardware-supported scale once BE supports it
  constexpr uint32_t scale = 0;
  const auto si = get_surface_index(acc);

  if constexpr (sizeof(T) < 4) {
    using Tint = std::conditional_t<std::is_integral_v<T>, T,
                                    detail::uint_type_t<sizeof(T)>>;
    using Treal = __raw_t<T>;
    static_assert(std::is_integral<Tint>::value,
                  "only integral 1- & 2-byte types are supported");
    using PromoT = typename std::conditional_t<std::is_signed<Tint>::value,
                                               int32_t, uint32_t>;
    simd<PromoT, N> promo_vals =
        __esimd_gather_masked_scaled2<PromoT, N, decltype(si), TypeSizeLog2,
                                      scale>(si, glob_offset, offsets.data(),
                                             mask.data());
    auto Res = convert<Tint>(promo_vals);

    if constexpr (!std::is_same_v<Tint, T>) {
      return detail::bitcast<Treal, Tint, N>(Res.data());
    } else {
      return Res;
    }
  } else {
    using Treal = __raw_t<T>;
    simd<Treal, N> Res = __esimd_gather_masked_scaled2<Treal, N, decltype(si),
                                                       TypeSizeLog2, scale>(
        si, glob_offset, offsets.data(), mask.data());
    if constexpr (!std::is_same_v<Treal, T>) {
      return Res.template bit_cast_view<T>();
    } else {
      return Res;
    }
  }
}

/// SLM gather implementation.
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

template <typename T, int NElts, lsc_data_size DS, int N>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
slm_gather_impl(__ESIMD_NS::simd<uint32_t, N> offsets,
                __ESIMD_NS::simd_mask<N> pred) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr auto Transposed = detail::lsc_data_order::nontranspose;
  using MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<MsgT, N * NElts> Tmp =
      __esimd_lsc_load_slm<MsgT, cache_hint::none, cache_hint::none,
                           AddressScale, ImmOffset, _DS, _VS, Transposed, N>(
          pred.data(), offsets.data());
  return detail::lsc_format_ret<T>(Tmp);
}

/// SLM gather implementation.
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
template <typename T, int NElts, lsc_data_size DS, int N>
__ESIMD_API __ESIMD_NS::simd<T, N * NElts>
slm_gather_impl(__ESIMD_NS::simd<uint32_t, N> offsets,
                __ESIMD_NS::simd_mask<N> pred,
                __ESIMD_NS::simd<T, N * NElts> pass_thru) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size _DS =
      detail::expand_data_size(detail::finalize_data_size<T, DS>());
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order Transposed =
      detail::lsc_data_order::nontranspose;
  using MsgT = typename detail::lsc_expand_type<T>::type;
  __ESIMD_NS::simd<MsgT, N * NElts> PassThruExpanded =
      detail::lsc_format_input<MsgT>(pass_thru);
  __ESIMD_NS::simd<MsgT, N * NElts> Result =
      __esimd_lsc_load_merge_slm<MsgT, cache_hint::none, cache_hint::none,
                                 AddressScale, ImmOffset, _DS, _VS, Transposed,
                                 N>(pred.data(), offsets.data(),
                                    PassThruExpanded.data());
  return detail::lsc_format_ret<T>(Result);
}

template <typename T, int N, int VS, cache_hint L1H, cache_hint L2H,
          lsc_data_size DS, typename OffsetT, typename AccessorT>
__ESIMD_API std::enable_if_t<
    is_device_accessor_with_v<AccessorT, accessor_mode_cap::can_read>,
    simd<T, N>>
gather_impl(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
            simd_mask<N / VS> pred, simd<T, N> pass_thru) {
  static_assert(N / VS >= 1 && N % VS == 0, "N must be divisible by VS");
  static_assert(std::is_integral_v<OffsetT>,
                "Gather must have integral byte_offset type");
  static_assert(sizeof(OffsetT) <= 4,
                "Implicit truncation of 64-bit byte_offset to 32-bit is "
                "disabled. Use -fsycl-esimd-force-stateless-mem or explicitly "
                "convert offsets to a 32-bit vector");
  static_assert(VS == 1 || sizeof(T) >= 4,
                "VS > 1 is supprted only for 4- and 8-byte elements");
  check_lsc_vector_size<VS>();
  check_lsc_data_size<T, DS>();
  check_cache_hint<cache_action::load, L1H, L2H>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size LSCVS = to_lsc_vector_size<VS>();
  constexpr auto Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  auto SI = get_surface_index(acc);
  simd<uint32_t, N / VS> ByteOffsets32 = convert<uint32_t>(byte_offsets);
  simd<MsgT, N> PassThruExpanded = lsc_format_input<MsgT>(pass_thru);
  simd<MsgT, N> Result =
      __esimd_lsc_load_merge_bti<MsgT, L1H, L2H, AddressScale, ImmOffset, EDS,
                                 LSCVS, Transposed, N / VS>(
          pred.data(), ByteOffsets32.data(), SI, PassThruExpanded.data());
  return lsc_format_ret<T>(Result);
}
#endif // __ESIMD_FORCE_STATELESS_MEM

} // namespace detail

/// @endcond ESIMD_DETAIL

/// @addtogroup sycl_esimd_memory
/// @{

/// @anchor accessor_gather Accessor-based gather.
///
/// Collects elements from memory referenced by the accessor \p acc, byte
/// offsets \p byte_offsets and common offset \glob_offset, then returns
/// the loaded elements as a single \ref simd object.
///
/// Supported platforms: DG2/PVC if sizeof(T) > 4 or the number of elements to
/// load is not equal to 1, 8, 16, 32. Otherwise, it is supported on ALL
/// platforms.
///
/// @tparam T Element type.
/// @tparam N The number of vector elements.
/// @tparam AccessorT The accessor type.
/// @param acc The accessor to gather from.
/// @param byte_offsets Per-element offsets in bytes.
/// @param glob_offset Offset in bytes added to each individual element's offset
///   to compute actual memory access offset for that element.
/// @param mask Memory access mask. Elements with zero corresponding mask's
///   predicate are not accessed, their values in the resulting vector are
///   undefined.
///
// Dev note: the argument \p glob_offset of this function does not have
// a default value to not conflict with more generic variant (acc-ga-3)
// defined below. This restriction though requires adding an additional
// variant: simd<T, N> gather(acc, glob_offset) to support calls that require
// implicit conversion of a scalar offset to a vector of offsets, e.g.
// 'res = gather<T, N>(acc, 0);'
template <typename T, int N, typename AccessorT>
__ESIMD_API
    std::enable_if_t<detail::is_device_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_read>,
                     simd<T, N>>
    gather(AccessorT acc, simd<detail::DeviceAccessorOffsetT, N> byte_offsets,
           detail::DeviceAccessorOffsetT glob_offset, simd_mask<N> mask = 1) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return gather<T, N>(__ESIMD_DNS::accessorToPointer<T>(acc, glob_offset),
                      byte_offsets, mask);
#else
  if constexpr (sizeof(T) > 4 || !((N == 1 || N == 8 || N == 16 || N == 32))) {
    // Requires DG2 or PVC.
    simd<T, N> PassThru; // Intentionally undefined
    byte_offsets += glob_offset;
    return detail::gather_impl<T, N, 1, cache_hint::none, cache_hint::none,
                               detail::lsc_data_size::default_size>(
        acc, byte_offsets, mask, PassThru);
  } else {
    return detail::gather_impl<T, N>(acc, byte_offsets, glob_offset, mask);
  }
#endif // __ESIMD_FORCE_STATELESS_MEM
}

/// Loads and broadcasts the element located at \p acc and byte offset
/// \p glob_offset to a vector and returns it as a \ref simd object.
///
/// Supported platforms: DG2/PVC if sizeof(T) > 4 or the number of elements to
/// load is not equal to 1, 8, 16, 32. Otherwise, it is supported on ALL
/// platforms.
///
/// @tparam T Element type.
/// @tparam N The number of vector elements.
/// @tparam AccessorT The accessor type.
/// @param acc The accessor to gather from.
/// @param glob_offset Offset in bytes added to each individual element's offset
///   to compute actual memory access offset for that element.
template <typename T, int N, typename AccessorT>
__ESIMD_API
    std::enable_if_t<detail::is_device_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_read>,
                     simd<T, N>>
    gather(AccessorT acc, detail::DeviceAccessorOffsetT glob_offset) {
  simd<detail::DeviceAccessorOffsetT, N> ByteOffsets = 0;
  return gather<T, N>(acc, ByteOffsets, glob_offset);
}

#ifdef __ESIMD_FORCE_STATELESS_MEM
template <typename T, int N, typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_read> &&
        std::is_integral_v<Toffset> && !std::is_same_v<Toffset, uint64_t>,
    simd<T, N>>
gather(AccessorTy acc, simd<Toffset, N> offsets, uint64_t glob_offset,
       simd_mask<N> mask = 1) {
  return gather<T, N>(acc, convert<uint64_t>(offsets), glob_offset, mask);
}
#endif

/// template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (acc-ga-1)
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (acc-ga-2)
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (acc-ga-3)
///
/// The next 3 functions are similar to (acc-ga-1,2,3), but they don't have
/// the template parameter 'VS'. These functions are added for convenience and
/// to make it possible for user to omit the template parameters T and N,
/// e.g. 'auto res = gather(acc, byte_offsets);
/// template <typename T, int N, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (acc-ga-4)
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask, PropertyListT props = {});// (acc-ga-5)
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
///                   PropertyListT props = {});                   // (acc-ga-6)
///
/// The next 3 functions are similar to (acc-ga-1,2,3), but accept the
/// \p byte_offsets as a \c simd_view argument:
/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename OffsetSimdViewT,
//            typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (acc-ga-7)
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (acc-ga-8)
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   PropertyListT props = {});                   // (acc-ga-9)

/// template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (acc-ga-1)
/// Supported platforms: DG2, PVC only - Temporary restriction for the variant
/// with pass_thru operand. The only exception: DG2/PVC is not required if
/// stateless memory mode is enforced via -fsycl-esimd-force-stateless-mem and
/// VS == 1 and no L1/L2 cache hints passed and the
/// __ESIMD_GATHER_SCATTER_LLVM_IR macro is used.
///
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the load from
/// (acc + byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to load.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param pass_thru The vector pass through values.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
       simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return gather<T, N, VS>(detail::accessorToPointer<T>(acc), byte_offsets, mask,
                          pass_thru, props);
#else
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  return detail::gather_impl<T, N, VS, L1Hint, L2Hint,
                             detail::lsc_data_size::default_size>(
      acc, byte_offsets, mask, pass_thru);
#endif // __ESIMD_FORCE_STATELESS_MEM
}

/// template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (acc-ga-2)
/// Supported platforms: DG2, PVC in most cases. The DG2/PVC is not required if
/// VS == 1 and no L1/L2 cache hints used and sizeof(T) <= 4 and N = {1,8,16,32}
///
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the load from
/// (acc + byte_offsets[i]) is skipped and the corresponding i-th element of
/// the returned vector is undefined.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to load.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
       simd_mask<N / VS> mask, PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return gather<T, N, VS>(detail::accessorToPointer<T>(acc), byte_offsets, mask,
                          props);
#else
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(Alignment >= sizeof(T),
                "gather() requires at least element-size alignment");
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                VS > 1 || sizeof(T) > 4 ||
                !((N == 1 || N == 8 || N == 16 || N == 32))) {
    simd<T, N> PassThru; // Intentionally undefined
    return detail::gather_impl<T, N, VS, L1Hint, L2Hint,
                               detail::lsc_data_size::default_size>(
        acc, byte_offsets, mask, PassThru);
  } else {
    return detail::gather_impl<T, N>(acc, byte_offsets, 0, mask);
  }
#endif // __ESIMD_FORCE_STATELESS_MEM
}

/// template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (acc-ga-3)
/// Supported platforms: DG2, PVC in most cases. The DG2/PVC is not required if
/// VS == 1 and no L1/L2 cache hints used and sizeof(T) <= 4 and N = {1,8,16,32}
///
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to load.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
       PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  return gather<T, N, VS>(acc, byte_offsets, Mask, props);
}

/// template <typename T, int N, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (acc-ga-4)
/// This function is identical to (acc-ga-1) except that vector size is fixed
/// to 1. This variant is added for convenience and let user omit the template
/// arguments and call the function as
/// 'gather(acc, byte_offsets, mask, pass_thru);'.
template <typename T, int N, typename AccessorT, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<OffsetT, N> byte_offsets, simd_mask<N> mask,
       simd<T, N> pass_thru, PropertyListT props = {}) {
  return gather<T, N, 1>(acc, byte_offsets, mask, pass_thru, props);
}

/// template <typename T, int N, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask, PropertyListT props       // (acc-ga-5)
/// This function is identical to (acc-ga-2) except that vector size is fixed
/// to 1. This variant is added for convenience and let user omit the template
/// arguments and call the function as 'gather(acc, byte_offsets, mask);'.
template <typename T, int N, typename AccessorT, typename OffsetT,
          typename MaskT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
     std::is_same_v<MaskT, simd_mask<N>>),
    simd<T, N>>
gather(AccessorT acc, simd<OffsetT, N> byte_offsets, MaskT mask,
       PropertyListT props = {}) {
  return gather<T, N, 1>(acc, byte_offsets, mask, props);
}

/// template <typename T, int N, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N> byte_offsets,
///                   PropertyListT props = {});                   // (acc-ga-6)
/// This function is identical to (acc-ga-3) except that vector size is fixed
/// to 1. This variant is added for convenience and let user omit the template
/// arguments and call the function as 'gather(acc, byte_offsets);'.
template <typename T, int N, typename AccessorT, typename OffsetT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<OffsetT, N> byte_offsets, PropertyListT props = {}) {
  return gather<T, N, 1>(acc, byte_offsets, props);
}

/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename OffsetSimdViewT,
//            typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (acc-ga-7)
/// This function is identical to (acc-ga-1) except that the \p byte_offsets
/// is represented as \c simd_view.
template <typename T, int N, int VS = 1, typename AccessorT,
          typename OffsetSimdViewT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<OffsetSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       simd<T, N> pass_thru, PropertyListT props = {}) {
  return gather<T, N, VS>(acc, byte_offsets.read(), mask, pass_thru, props);
}

/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename OffsetSimdViewT,
//            typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (acc-ga-8)
/// This function is identical to (acc-ga-2) except that the \p byte_offsets
/// is represented as \c simd_view.
template <typename T, int N, int VS = 1, typename AccessorT,
          typename OffsetSimdViewT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<OffsetSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       PropertyListT props = {}) {
  return gather<T, N, VS>(acc, byte_offsets.read(), mask, props);
}

/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename OffsetSimdViewT,
//            typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   PropertyListT props = {});                   // (acc-ga-9)
/// This function is identical to (acc-ga-3) except that the \p byte_offsets
/// is represented as \c simd_view.
template <typename T, int N, int VS = 1, typename AccessorT,
          typename OffsetSimdViewT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<OffsetSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, OffsetSimdViewT byte_offsets, PropertyListT props = {}) {
  return gather<T, N, VS>(acc, byte_offsets.read(), props);
}

/// @anchor accessor_scatter
/// Accessor-based scatter.
///
/// Writes elements of a \ref simd object into an accessor at given offsets.
/// An element can be a 1, 2 or 4-byte value.
///
/// @tparam T Element type; can only be a 1,2,4-byte integer, \c sycl::half or
///   \c float.
/// @tparam N The number of vector elements. Can be \c 1, \c 8, \c 16 or \c 32.
/// @tparam AccessorTy The accessor type.
/// @param acc The accessor to scatter to.
/// @param offsets Per-element offsets in bytes.
/// @param vals Values to write.
/// @param glob_offset Offset in bytes added to each individual element's offset
///   to compute actual memory access offset for that element.
/// @param mask Memory access mask. Elements with zero corresponding mask's
///   predicate are not accessed.
///
///
template <typename T, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<
    (sizeof(T) <= 4) && (N == 1 || N == 8 || N == 16 || N == 32) &&
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write>>
scatter(AccessorTy acc, simd<detail::DeviceAccessorOffsetT, N> offsets,
        simd<T, N> vals, detail::DeviceAccessorOffsetT glob_offset = 0,
        simd_mask<N> mask = 1) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  scatter<T, N>(__ESIMD_DNS::accessorToPointer<T>(acc, glob_offset), offsets,
                vals, mask);
#else
  detail::scatter_impl<T, N, AccessorTy>(acc, vals, offsets, glob_offset, mask);
#endif
}

#ifdef __ESIMD_FORCE_STATELESS_MEM
template <typename T, int N, typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    (sizeof(T) <= 4) && (N == 1 || N == 8 || N == 16 || N == 32) &&
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    std::is_integral_v<Toffset> && !std::is_same_v<Toffset, uint64_t>>
scatter(AccessorTy acc, simd<Toffset, N> offsets, simd<T, N> vals,
        uint64_t glob_offset = 0, simd_mask<N> mask = 1) {
  scatter<T, N, AccessorTy>(acc, convert<uint64_t>(offsets), vals, glob_offset,
                            mask);
}
#endif

/// Load a scalar value from an accessor.
/// @tparam T Type of the value.
/// @tparam AccessorTy Type of the accessor.
/// @param acc Accessor to load from.
/// @param offset Offset in bytes.
/// @return The loaded value.
///
template <typename T, typename AccessorTy>
__ESIMD_API T scalar_load(AccessorTy acc,
                          detail::DeviceAccessorOffsetT offset) {
  const simd<T, 1> Res =
      gather<T, 1, AccessorTy>(acc, simd<decltype(offset), 1>(offset));
  return Res[0];
}

/// Store a scalar value into an accessor.
/// @tparam T Type of the value.
/// @tparam AccessorTy Type of the accessor.
/// @param acc Accessor to store to.
/// @param offset Offset in bytes.
/// @param val The stored value.
///
template <typename T, typename AccessorTy>
__ESIMD_API void scalar_store(AccessorTy acc,
                              detail::DeviceAccessorOffsetT offset, T val) {
  scatter<T, 1, AccessorTy>(acc, simd<decltype(offset), 1>(offset),
                            simd<T, 1>(val));
}

/// @anchor usm_gather_rgba
/// Gather and transpose pixels from given memory locations defined by the base
/// pointer \c p and \c offsets. Up to 4 32-bit data elements may be accessed at
/// each address depending on the channel mask \c Mask template parameter. Each
/// pixel's address must be 4 byte aligned. As an example, let's assume we want
/// to read \c n pixels at address \c addr, skipping \c G and \c B channels.
/// Each channel is a 32-bit float and the pixel data at given address in memory
/// is:
/// @code{.cpp}
/// R1 G1 B1 A1 R2 G2 B2 A2 ... Rn Gn Bn An
/// @endcode
/// Then this can be achieved by using
/// @code{.cpp}
/// simd<uint32_t, n> byte_offsets(0, 4*4 /* byte size of a single pixel */);
/// auto x = gather_rgba<float, n, rgba_channel_mask::AR>(addr, byte_offsets);
/// @endcode
/// Returned \c x will contain \c 2*n \c float elements:
/// @code{.cpp}
/// R1 R2 ... Rn A1 A2 ... An
/// @endcode
///
/// @tparam T Element type of the returned vector. Must be 4 bytes in size.
/// @tparam N Number of pixels to access (matches the size of the \c offsets
///   vector). Must be 8, 16 or 32.
/// @tparam Mask A pixel's channel mask.
/// @param p The USM base pointer representing memory address of the access.
/// @param offsets vector of byte offsets of the pixels relative to the base
/// pointer.
/// @param mask Memory access mask. Pixels with zero corresponding mask's
///   predicate are not accessed. Their values in the resulting vector are
///   undefined.
/// @return Read data - up to N*4 values of type \c Tx.
///
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR, typename T,
          int N, typename Toffset>
__ESIMD_API simd<T, N * get_num_channels_enabled(RGBAMask)>
gather_rgba(const T *p, simd<Toffset, N> offsets, simd_mask<N> mask = 1) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert((N == 8 || N == 16 || N == 32), "Unsupported value of N");
  static_assert(sizeof(T) == 4, "Unsupported size of type T");
  simd<uint64_t, N> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  return __esimd_svm_gather4_scaled<detail::__raw_t<T>, N, RGBAMask>(
      addrs.data(), mask.data());
}

/// A variation of \c gather_rgba API with \c offsets represented as
/// \c simd_view object.
///
/// @tparam T Element type of the returned vector. Must be 4 bytes in size.
/// @tparam N Number of pixels to access (matches the size of the \c offsets
///   vector). Must be 8, 16 or 32.
/// @tparam Mask A pixel's channel mask.
/// @param p The USM base pointer representing memory address of the access.
/// @param offsets simd_view of byte offsets of the pixels relative to the base
/// pointer.
/// @param mask Memory access mask. Pixels with zero corresponding mask's
///   predicate are not accessed. Their values in the resulting vector are
///   undefined.
/// @return Read data - up to N*4 values of type \c Tx.
///
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR, typename T,
          int N, typename OffsetObjT, typename RegionTy>
__ESIMD_API simd<T, N * get_num_channels_enabled(RGBAMask)>
gather_rgba(const T *p, simd_view<OffsetObjT, RegionTy> offsets,
            simd_mask<N> mask = 1) {
  return gather_rgba<RGBAMask, T, N>(p, offsets.read(), mask);
}

/// A variation of \c gather_rgba API with \c offsets represented as
/// scalar.
///
/// @tparam T Element type of the returned vector. Must be 4 bytes in size.
/// @tparam N Number of pixels to access (matches the size of the \c offsets
///   vector). Must be 8, 16 or 32.
/// @tparam Mask A pixel's channel mask.
/// @param p The USM base pointer representing memory address of the access.
/// @param offset scalar byte offsets of the pixels relative to the base
/// pointer.
/// @param mask Memory access mask. Pixels with zero corresponding mask's
///   predicate are not accessed. Their values in the resulting vector are
///   undefined.
/// @return Read data - up to N*4 values of type \c Tx.
///
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR, typename T,
          int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset>,
                             simd<T, N * get_num_channels_enabled(RGBAMask)>>
gather_rgba(const T *p, Toffset offset, simd_mask<N> mask = 1) {
  return gather_rgba<RGBAMask, T, N>(p, simd<Toffset, N>(offset), mask);
}

namespace detail {
template <rgba_channel_mask M> static void validate_rgba_write_channel_mask() {
  using CM = rgba_channel_mask;
  static_assert(
      (M == CM::ABGR || M == CM::BGR || M == CM::GR || M == CM::R) &&
      "Only ABGR, BGR, GR, R channel masks are valid in write operations");
}
} // namespace detail

/// @anchor usm_scatter_rgba
/// Transpose and scatter pixels to given memory locations defined by the base
/// pointer \c p and \c offsets. Up to 4 32-bit data elements may be accessed at
/// each address depending on the channel mask \c RGBAMask. Each
/// pixel's address must be 4 byte aligned. This is basically an inverse
/// operation for gather_rgba. Unlike \c gather_rgba, this function imposes
/// restrictions on possible \c Mask template argument values. It can only be
/// one of the following: \c ABGR, \c BGR, \c GR, \c R.
///
/// @tparam T Element type of the returned vector. Must be 4 bytes in size.
/// @tparam N Number of pixels to access (matches the size of the \c offsets
///   vector). Must be 8, 16 or 32.
/// @tparam RGBAMask A pixel's channel mask.
/// @param p The USM base pointer representing memory address of the access.
/// @param vals values to be written.
/// @param offsets vector of byte offsets of the pixels relative to the base
/// pointer.
/// @param mask Memory access mask. Pixels with zero corresponding mask's
///   predicate are not accessed. Their values in the resulting vector are
///   undefined.
///
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR, typename T,
          int N, typename Toffset>
__ESIMD_API void
scatter_rgba(T *p, simd<Toffset, N> offsets,
             simd<T, N * get_num_channels_enabled(RGBAMask)> vals,
             simd_mask<N> mask = 1) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert((N == 8 || N == 16 || N == 32), "Unsupported value of N");
  static_assert(sizeof(T) == 4, "Unsupported size of type T");
  detail::validate_rgba_write_channel_mask<RGBAMask>();
  simd<uint64_t, N> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  __esimd_svm_scatter4_scaled<detail::__raw_t<T>, N, RGBAMask>(
      addrs.data(), vals.data(), mask.data());
}

/// A variation of \c scatter_rgba API with \c offsets represented as
/// \c simd_view object
///
/// @tparam T Element type of the returned vector. Must be 4 bytes in size.
/// @tparam N Number of pixels to access (matches the size of the \c offsets
///   vector). Must be 8, 16 or 32.
/// @tparam RGBAMask A pixel's channel mask.
/// @param p The USM base pointer representing memory address of the access.
/// @param vals values to be written.
/// @param offsets simd_view of byte offsets of the pixels relative to the base
/// pointer.
/// @param mask Memory access mask. Pixels with zero corresponding mask's
///   predicate are not accessed. Their values in the resulting vector are
///   undefined.
///
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR, typename T,
          int N, typename OffsetObjT, typename RegionTy>
__ESIMD_API void
scatter_rgba(T *p, simd_view<OffsetObjT, RegionTy> offsets,
             simd<T, N * get_num_channels_enabled(RGBAMask)> vals,
             simd_mask<N> mask = 1) {
  scatter_rgba<RGBAMask, T, N>(p, offsets.read(), vals, mask);
}

/// A variation of \c scatter_rgba API with \c offsets represented as
/// scalar
///
/// @tparam T Element type of the returned vector. Must be 4 bytes in size.
/// @tparam N Number of pixels to access (matches the size of the \c offsets
///   vector). Must be 8, 16 or 32.
/// @tparam RGBAMask A pixel's channel mask.
/// @param p The USM base pointer representing memory address of the access.
/// @param vals values to be written.
/// @param offset scalar byte offset of the pixels relative to the base
/// pointer.
/// @param mask Memory access mask. Pixels with zero corresponding mask's
///   predicate are not accessed. Their values in the resulting vector are
///   undefined.
///
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR, typename T,
          int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> && N == 1>
scatter_rgba(T *p, Toffset offset,
             simd<T, N * get_num_channels_enabled(RGBAMask)> vals,
             simd_mask<N> mask = 1) {
  scatter_rgba<RGBAMask, T, N>(p, simd<Toffset, N>(offset), vals, mask);
}

template <typename T, int N, rgba_channel_mask RGBAMask>
__SYCL_DEPRECATED("use scatter_rgba<rgba_channel_mask>()")
__ESIMD_API std::
    enable_if_t<(N == 8 || N == 16 || N == 32) && sizeof(T) == 4> scatter_rgba(
        T *p, simd<uint32_t, N> offsets,
        simd<T, N * get_num_channels_enabled(RGBAMask)> vals,
        simd_mask<N> mask = 1) {
  scatter_rgba<RGBAMask>(p, offsets, vals, mask);
}

/// Gather and transpose pixels from the given memory locations defined by the
/// base specified by \c acc, the global offset \c global_offset and a vector of
/// offsets \c offsets. Up to 4 32-bit data elements may be accessed at each
/// address depending on the channel mask \c RGBAMask. Each pixel's address must
/// be 4-byte aligned.
/// For usage examples, see \ref usm_gather_rgba above, the only difference
/// would be the usage of an accessor instead of a usm pointer.
///
/// @tparam RGBAMask A pixel's channel mask.
/// @tparam AccessorT The accessor type for the memory to be loaded/gathered.
/// The returned vector elements must match the accessor data type. The loaded
/// elements must be 4 bytes in size.
/// @tparam N Number of pixels to access (matches the size of the \c offsets
///   vector). Must be 8, 16 or 32.
/// @param acc The accessor representing memory address of the access.
/// @param offsets Byte offsets of the pixels relative to the base pointer.
/// @param global_offset Byte offset of the pixels relative to the base pointer.
/// @param mask Memory access mask. Pixels with zero corresponding mask's
///   predicate are not accessed. Their values in the resulting vector are
///   undefined.
/// @return Read data - up to N*4 values of type \c Tx.
///
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR,
          typename AccessorT, int N,
          typename T = typename AccessorT::value_type>
__ESIMD_API
    std::enable_if_t<((N == 8 || N == 16 || N == 32) && sizeof(T) == 4 &&
                      detail::is_device_accessor_with_v<
                          AccessorT, detail::accessor_mode_cap::can_read>),
                     simd<T, N * get_num_channels_enabled(RGBAMask)>>
    gather_rgba(AccessorT acc, simd<detail::DeviceAccessorOffsetT, N> offsets,
                detail::DeviceAccessorOffsetT global_offset = 0,
                simd_mask<N> mask = 1) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return gather_rgba<RGBAMask>(
      __ESIMD_DNS::accessorToPointer<T>(acc, global_offset), offsets, mask);
#else
  // TODO (performance) use hardware-supported scale once BE supports it
  constexpr uint32_t Scale = 0;
  const auto SI = get_surface_index(acc);
  return __esimd_gather4_masked_scaled2<detail::__raw_t<T>, N, RGBAMask,
                                        decltype(SI), Scale>(
      SI, global_offset, offsets.data(), mask.data());
#endif
}

#ifdef __ESIMD_FORCE_STATELESS_MEM
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR,
          typename AccessorT, int N,
          typename T = typename AccessorT::value_type, typename Toffset>
__ESIMD_API std::enable_if_t<
    ((N == 8 || N == 16 || N == 32) && sizeof(T) == 4 &&
     detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     std::is_integral_v<Toffset> && !std::is_same_v<Toffset, uint64_t>),
    simd<T, N * get_num_channels_enabled(RGBAMask)>>
gather_rgba(AccessorT acc, simd<Toffset, N> offsets, uint64_t global_offset = 0,
            simd_mask<N> mask = 1) {
  return gather_rgba<RGBAMask, AccessorT, N, T>(acc, convert<uint64_t>(offsets),
                                                global_offset, mask);
}
#endif

/// Gather data from the memory addressed by accessor \c acc, offset common
/// for all loaded elements \c global_offset and per-element offsets \c offsets,
/// and return it as simd vector. See @ref usm_gather_rgba for information about
/// the operation semantics and parameter restrictions/interdependencies.
/// @tparam RGBAMask Pixel's channel mask.
/// @tparam AccessorT The accessor type for the memory to be stored/scattered.
/// The returned vector elements must match the accessor data type. The loaded
/// elements must be 4 bytes in size.
/// @tparam N The number of elements to access.
/// @param offsets Byte offsets of each element.
/// @param vals values to be written.
/// @param global_offset Byte offset of the pixels relative to the base pointer.
/// @param mask Operation mask. All-1 by default.
///
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR,
          typename AccessorT, int N,
          typename T = typename AccessorT::value_type>
__ESIMD_API
    std::enable_if_t<(N == 8 || N == 16 || N == 32) && sizeof(T) == 4 &&
                     detail::is_device_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_write>>
    scatter_rgba(AccessorT acc, simd<detail::DeviceAccessorOffsetT, N> offsets,
                 simd<T, N * get_num_channels_enabled(RGBAMask)> vals,
                 detail::DeviceAccessorOffsetT global_offset = 0,
                 simd_mask<N> mask = 1) {
  detail::validate_rgba_write_channel_mask<RGBAMask>();
#ifdef __ESIMD_FORCE_STATELESS_MEM
  scatter_rgba<RGBAMask>(__ESIMD_DNS::accessorToPointer<T>(acc, global_offset),
                         offsets, vals, mask);
#else
  // TODO (performance) use hardware-supported scale once BE supports it
  constexpr uint32_t Scale = 0;
  const auto SI = get_surface_index(acc);
  __esimd_scatter4_scaled<T, N, decltype(SI), RGBAMask, Scale>(
      mask.data(), SI, global_offset, offsets.data(), vals.data());
#endif
}

#ifdef __ESIMD_FORCE_STATELESS_MEM
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR,
          typename AccessorT, int N,
          typename T = typename AccessorT::value_type, typename Toffset>
__ESIMD_API std::enable_if_t<
    (N == 8 || N == 16 || N == 32) && sizeof(T) == 4 &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write> &&
    std::is_integral_v<Toffset> && !std::is_same_v<Toffset, uint64_t>>
scatter_rgba(AccessorT acc, simd<Toffset, N> offsets,
             simd<T, N * get_num_channels_enabled(RGBAMask)> vals,
             uint64_t global_offset = 0, simd_mask<N> mask = 1) {
  scatter_rgba<RGBAMask, AccessorT, N, T>(acc, convert<uint64_t>(offsets), vals,
                                          global_offset, mask);
}
#endif
/// @} sycl_esimd_memory

namespace detail {

#ifndef __ESIMD_FP_ATOMIC_OP_TYPE_CHECK
#define __ESIMD_FP_ATOMIC_OP_TYPE_CHECK(T)                                     \
  static_assert(is_type<T, float, sycl::half, double>(),                       \
                "float, double or sycl::half type is expected");
#endif // __ESIMD_FP_ATOMIC_OP_TYPE_CHECK

/// Check the legality of an atomic call in terms of size and type.
///
template <__ESIMD_NS::atomic_op Op, typename T, int N, unsigned NumSrc,
          bool IsLSC = false>
constexpr void check_atomic() {

  static_assert(sizeof(T) > 1, "Unsupported data type");

  // LSC atomic operation is supported for any width.
  if constexpr (!IsLSC)
    static_assert((detail::isPowerOf2(N, 32)),
                  "Execution size 1, 2, 4, 8, 16, 32 are supported");

  static_assert(NumSrc == __ESIMD_DNS::get_num_args<Op>(),
                "Wrong number of operands");
  constexpr bool IsInt2BytePlus =
      std::is_integral_v<T> && (sizeof(T) >= sizeof(uint16_t));

  if constexpr (Op == __ESIMD_NS::atomic_op::xchg ||
                Op == __ESIMD_NS::atomic_op::cmpxchg ||
                Op == __ESIMD_NS::atomic_op::predec ||
                Op == __ESIMD_NS::atomic_op::inc ||
                Op == __ESIMD_NS::atomic_op::dec) {

    static_assert(IsInt2BytePlus, "Integral 16-bit or wider type is expected");
  }
  // FP ops (are always delegated to native::lsc::<Op>)
  if constexpr (Op == __ESIMD_NS::atomic_op::fmax ||
                Op == __ESIMD_NS::atomic_op::fmin ||
                Op == __ESIMD_NS::atomic_op::fadd ||
                Op == __ESIMD_NS::atomic_op::fsub ||
                Op == __ESIMD_NS::atomic_op::fcmpxchg) {
    __ESIMD_FP_ATOMIC_OP_TYPE_CHECK(T);
  }
  if constexpr (Op == __ESIMD_NS::atomic_op::add ||
                Op == __ESIMD_NS::atomic_op::sub ||
                Op == __ESIMD_NS::atomic_op::umin ||
                Op == __ESIMD_NS::atomic_op::umax ||
                Op == __ESIMD_NS::atomic_op::bit_and ||
                Op == __ESIMD_NS::atomic_op::bit_or ||
                Op == __ESIMD_NS::atomic_op::bit_xor ||
                Op == __ESIMD_NS::atomic_op::smin ||
                Op == __ESIMD_NS::atomic_op::smax) {
    static_assert(IsInt2BytePlus, "Integral 16-bit or wider type is expected");
    constexpr bool IsSignedMinmax = (Op == __ESIMD_NS::atomic_op::smin) ||
                                    (Op == __ESIMD_NS::atomic_op::smax);
    constexpr bool IsUnsignedMinmax = (Op == __ESIMD_NS::atomic_op::umin) ||
                                      (Op == __ESIMD_NS::atomic_op::umax);

    if constexpr (IsSignedMinmax || IsUnsignedMinmax) {
      constexpr bool SignOK = std::is_signed_v<T> == IsSignedMinmax;
      static_assert(SignOK, "Signed/unsigned integer type expected for "
                            "signed/unsigned min/max operation");
    }
  }
}
#undef __ESIMD_FP_ATOMIC_OP_TYPE_CHECK
} // namespace detail

/// @addtogroup sycl_esimd_memory_slm
/// @{

/// Declare per-work-group slm size.
/// GPU RT/driver requires this function to be called in the beginning
/// of the kernel using SLM. There must be only 1 call site of slm_init()
/// per kernel.
/// If slm_init is called from some function F called from the kernel,
/// then inlining of F into the kernel must be managed/guaranteed.
/// slm_init<SLMSize> can also be used together with slm_allocator() class.
/// In such cases slm_allocator<AdditionalMem> allocates extra chunk of SLM
/// memory and the final amount of allocated SLM may be bigger
/// than what is requested by slm_init. See more details on
/// slm_allocator class usage at it's declaration and ESIMD extension SPEC.
/// @tparam SLMSize  Shared Local Memory (SLM) size
template <uint32_t SLMSize> __ESIMD_API void slm_init() {
  __esimd_slm_init(SLMSize);
}

/// Declare per-work-group slm size. Non-constant argument version to be used
/// with specialization constants only.
/// Same restrictions are applied to this function as to it's template variant
/// slm_init<SLMSize>().
/// This version has an additional restriction - it cannot be used together
//  with esimd::slm_allocator() class.
/// @param size  Shared Local Memory (SLM) size to be allocated for each
/// work-group of ESIMD kernel.
__ESIMD_API void slm_init(uint32_t size) { __esimd_slm_init(size); }

/// template <typename T, int N, int VS,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> slm_gather(simd<uint32_t, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (slm-ga-1)
/// simd<T, N> slm_gather(simd<uint32_t, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (slm-ga-2)
/// simd<T, N> slm_gather(simd<uint32_t, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (slm-ga-3)
///
/// The next 3 functions are similar to the above and were added for
/// convenience. They assume the VS parameter is set to 1 and do not require
/// specifying the template parameters <T, N, VS> at function calls.
/// template <typename T, int N,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> slm_gather(simd<uint32_t, N> byte_offsets,
///                   simd_mask<N> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (slm-ga-4)
/// simd<T, N> slm_gather(simd<uint32_t, N> byte_offsets,
///                   simd_mask<N> mask, PropertyListT props = {});// (slm-ga-5)
/// simd<T, N> slm_gather(simd<uint32_t, N> byte_offsets,
///                   PropertyListT props = {});                   // (slm-ga-6)
///
/// The next 3 functions are variations of the first 3 above (slm-ga-1,2,3)
/// and were added only to support simd_view instead of simd for byte_offsets
/// and/or pass_thru operands.
/// template <typename T, int N, int VS = 1, typename OffsetObjT,
///           typename OffsetRegionT, typename PropertyListT = empty_props_t>
/// simd <T, N> slm_gather(simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             simd_mask<N / VS> mask, simd<T, N> pass_thru,
///             PropertyListT props = {});                         // (slm-ga-7)
/// simd <T, N> slm_gather(simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (slm-ga-8)
/// simd <T, N> slm_gather(simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             PropertyListT props = {});                         // (slm-ga-9)

/// template <typename T, int N, int VS,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> slm_gather(simd<uint32_t, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (slm-ga-1)
#ifndef __ESIMD_GATHER_SCATTER_LLVM_IR
/// Supported platforms: DG2, PVC only - Temporary restriction for the variant
/// with pass_thru operand.
#endif // __ESIMD_GATHER_SCATTER_LLVM_IR
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements. Access to any element's memory location can be disabled via the
/// input vector of predicates \p mask. If mask[i] is unset, then the load from
/// (byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask, defaults to all 1s.
/// @param pass_thru The vector pass through values.
/// @param props The optional compile-time properties. Only 'alignment' property
/// is used.
/// @return A vector of elements read.
template <typename T, int N, int VS,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_gather(simd<uint32_t, N / VS> byte_offsets, simd_mask<N / VS> mask,
           simd<T, N> pass_thru, PropertyListT props = {}) {
  static_assert(N / VS >= 1 && N % VS == 0, "N must be divisible by VS");

  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(Alignment >= sizeof(T),
                "slm_gather() requires at least element-size alignment");

  // Use LSC lowering if VS > 1. Also, if masked gather is
  // not available, then LSC is the only lowering option.
  if constexpr (VS > 1 || !detail::isMaskedGatherScatterLLVMAvailable()) {
    return __ESIMD_DNS::slm_gather_impl<T, VS,
                                        detail::lsc_data_size::default_size>(
        byte_offsets, mask, pass_thru);
  } else {
    using MsgT = detail::__raw_t<T>;
    return __esimd_slm_gather_ld<MsgT, N, Alignment>(
        byte_offsets.data(), mask.data(), pass_thru.data());
  }
}

/// template <typename T, int N, int VS,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> slm_gather(simd<uint32_t, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (slm-ga-2)
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements. Access to any element's memory location can be disabled via the
/// input vector of predicates \p mask. If mask[i] is unset, then the load from
/// (byte_offsets[i]) is skipped and the corresponding i-th element of the
/// returned vector is undefined.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param mask The access mask, defaults to all 1s.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
template <typename T, int N, int VS,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_gather(simd<uint32_t, N / VS> byte_offsets, simd_mask<N / VS> mask,
           PropertyListT props = {}) {
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(Alignment >= sizeof(T),
                "slm_gather() requires at least element-size alignment");

  if constexpr (VS > 1) {
    return detail::slm_gather_impl<T, VS, detail::lsc_data_size::default_size>(
        byte_offsets, mask);
  } else if constexpr (detail::isMaskedGatherScatterLLVMAvailable()) {
    using MsgT = detail::__raw_t<T>;
    simd<MsgT, N> PassThru; // it is intentionally undefined
    return __esimd_slm_gather_ld<MsgT, N, Alignment>(
        byte_offsets.data(), mask.data(), PassThru.data());
  } else {
    static_assert(detail::isPowerOf2(N, 32), "Unsupported vector length");
    detail::LocalAccessorMarker acc;
    return detail::gather_impl<T, N>(acc, byte_offsets, 0, mask);
  }
}

/// template <typename T, int N, int VS,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> slm_gather(simd<uint32_t, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (slm-ga-3)
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
/// @return A vector of elements read.
template <typename T, int N, int VS,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_gather(simd<uint32_t, N / VS> byte_offsets, PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  return slm_gather<T, N, VS>(byte_offsets, Mask, props);
}

/// template <typename T, int N,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> slm_gather(simd<uint32_t, N> byte_offsets,
///                   simd_mask<N> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                   // (slm-ga-4)
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements. Access to any element's memory location can be disabled via the
/// input vector of predicates \p mask. If mask[i] is unset, then the load from
/// (byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask, defaults to all 1s.
/// @param pass_thru The vector pass through values.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
/// @return A vector of elements read.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_gather(simd<uint32_t, N> byte_offsets, simd_mask<N> mask,
           simd<T, N> pass_thru, PropertyListT props = {}) {
  constexpr int VS = 1;
  return slm_gather<T, N, VS>(byte_offsets, mask, pass_thru, props);
}

/// template <typename T, int N,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> slm_gather(simd<uint32_t, N> byte_offsets,
///                   simd_mask<N> mask, PropertyListT props = {});// (slm-ga-5)
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements. Access to any element's memory location can be disabled via the
/// input vector of predicates \p mask. If mask[i] is unset, then the load from
/// (byte_offsets[i]) is skipped and the corresponding i-th element of the
/// returned vector is undefined.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param mask The access mask, defaults to all 1s.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_gather(simd<uint32_t, N> byte_offsets, simd_mask<N> mask,
           PropertyListT props = {}) {
  constexpr int VS = 1;
  return slm_gather<T, N, VS>(byte_offsets, mask, props);
}

/// template <typename T, int N,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> slm_gather(simd<uint32_t, N> byte_offsets,
///                   PropertyListT props = {});                   // (slm-ga-6)
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
/// @return A vector of elements read.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_gather(simd<uint32_t, N> byte_offsets, PropertyListT props = {}) {
  constexpr int VS = 1;
  return slm_gather<T, N, VS>(byte_offsets, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetObjT,
///           typename OffsetRegionT, typename PropertyListT = empty_props_t>
/// simd <T, N> slm_gather(
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             simd_mask<N / VS> mask, simd<T, N> pass_thru,
///             PropertyListT props = {});                         // (slm-ga-7)
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements. Access to any element's memory location can be disabled via the
/// input vector of predicates \p mask. If mask[i] is unset, then the load from
/// (byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask, defaults to all 1s.
/// @param pass_thru The vector pass through values.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
/// @return A vector of elements read.
template <typename T, int N, int VS = 1, typename OffsetObjT,
          typename OffsetRegionT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetObjT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
slm_gather(simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
           simd_mask<N / VS> mask, simd<T, N> pass_thru,
           PropertyListT props = {}) {
  return slm_gather<T, N, VS>(byte_offsets.read(), mask, pass_thru, props);
}

/// simd <T, N> slm_gather(
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (slm-ga-8)
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements. Access to any element's memory location can be disabled via the
/// input vector of predicates \p mask. If mask[i] is unset, then the load from
/// (byte_offsets[i]) is skipped and the corresponding i-th element of the
/// returned vector is undefined.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param mask The access mask, defaults to all 1s.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
template <typename T, int N, int VS = 1, typename OffsetObjT,
          typename OffsetRegionT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetObjT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
slm_gather(simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
           simd_mask<N / VS> mask, PropertyListT props = {}) {
  return slm_gather<T, N, VS>(byte_offsets.read(), mask, props);
}

/// simd <T, N> slm_gather(
///             simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
///             PropertyListT props = {});                         // (slm-ga-9)
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <typename T, int N, int VS = 1, typename OffsetObjT,
          typename OffsetRegionT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetObjT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
slm_gather(simd_view<OffsetObjT, OffsetRegionT> byte_offsets,
           PropertyListT props = {}) {
  return slm_gather<T, N, VS>(byte_offsets.read(), props);
}

/// Load a scalar value from the Shared Local Memory.
/// @tparam T type of the value
/// @param offset SLM offset in bytes
/// @return the loaded value
///
template <typename T> __ESIMD_API T slm_scalar_load(uint32_t offset) {
  const simd<T, 1> Res = slm_gather<T, 1>(simd<uint32_t, 1>(offset));
  return Res[0];
}

/// Scatter operation over the Shared Local Memory.
/// This API has almost the same interface as the @ref accessor_scatter
/// "accessor-based scatter", except that it does not have the accessor and
/// the global offset parameters.
///
template <typename T, int N>
__ESIMD_API std::enable_if_t<(N == 1 || N == 8 || N == 16 || N == 32) &&
                             (sizeof(T) <= 4)>
slm_scatter(simd<uint32_t, N> offsets, simd<T, N> vals, simd_mask<N> mask = 1) {
  detail::LocalAccessorMarker acc;
  detail::scatter_impl<T, N>(acc, vals, offsets, 0, mask);
}

/// Store a scalar value into the Shared Local Memory.
/// @tparam T type of the value
/// @param offset SLM offset in bytes
/// @param val value to store
///
template <typename T>
__ESIMD_API void slm_scalar_store(uint32_t offset, T val) {
  slm_scatter<T, 1>(simd<uint32_t, 1>(offset), simd<T, 1>(val), 1);
}

/// Gather data from the Shared Local Memory at specified \c offsets and
/// return it as simd vector. See @ref usm_gather_rgba for information about
/// the operation semantics and parameter restrictions/interdependencies.
/// @tparam T The element type of the returned vector.
/// @tparam N The number of elements to access.
/// @tparam RGBAMask Pixel's channel mask.
/// @param offsets Byte offsets within the SLM of each element.
/// @param mask Operation mask. All-1 by default.
/// @return Gathered data as an \c N - element vector.
///
template <typename T, int N, rgba_channel_mask RGBAMask>
__ESIMD_API std::enable_if_t<(N == 8 || N == 16 || N == 32) && (sizeof(T) == 4),
                             simd<T, N * get_num_channels_enabled(RGBAMask)>>
slm_gather_rgba(simd<uint32_t, N> offsets, simd_mask<N> mask = 1) {
  const auto SI = __ESIMD_NS::get_surface_index(detail::LocalAccessorMarker());
  return __esimd_gather4_masked_scaled2<T, N, RGBAMask>(
      SI, 0 /*global_offset*/, offsets.data(), mask.data());
}

/// Gather data from the Shared Local Memory at specified \c offsets and
/// return it as simd vector. See @ref usm_scatter_rgba for information about
/// the operation semantics and parameter restrictions/interdependencies.
/// @tparam T The element type of the returned vector.
/// @tparam N The number of elements to access.
/// @tparam Mask Pixel's channel mask.
/// @param offsets Byte offsets within the SLM of each element.
/// @param vals values to be written.
/// @param mask Operation mask. All-1 by default.
///
template <typename T, int N, rgba_channel_mask Mask>
__ESIMD_API std::enable_if_t<(N == 8 || N == 16 || N == 32) && (sizeof(T) == 4)>
slm_scatter_rgba(simd<uint32_t, N> offsets,
                 simd<T, N * get_num_channels_enabled(Mask)> vals,
                 simd_mask<N> mask = 1) {
  detail::validate_rgba_write_channel_mask<Mask>();
  const auto si = __ESIMD_NS::get_surface_index(detail::LocalAccessorMarker());
  constexpr int16_t Scale = 0;
  constexpr int global_offset = 0;
  __esimd_scatter4_scaled<T, N, decltype(si), Mask, Scale>(
      mask.data(), si, global_offset, offsets.data(), vals.data());
}

/// Loads a contiguous block of SLM memory referenced by the given byte-offset
/// \p offset, then returns the loaded data as a simd object.
/// The generated code depends on the combination {T, N, Flags}.
/// Providing flags specifying the alignment of 16-bytes or more produces more
/// efficient code. If the alignment is smaller than 16-bytes, then less
/// efficient gather is generated. If the loaded vector is too long
/// for 1 flat-load GPU instruction, then a series of flat-loads and/or gathers
/// may be generated.
/// @tparam T Element type.
/// @tparam N Number of elements to load.
/// @tparam Flags The alignment specifier type tag.
/// @param byte_offset The byte-offset to load from.
/// @param Flags Specifies the alignment.
/// @return A vector of loaded elements.
///
template <typename T, int N,
          typename Flags = overaligned_tag<detail::OperandSize::OWORD>>
__ESIMD_API std::enable_if_t<is_simd_flag_type_v<Flags>, simd<T, N>>
slm_block_load(uint32_t byte_offset, Flags) {
  constexpr size_t Align = Flags::template alignment<simd<T, N>>;
  return __esimd_slm_block_ld<detail::__raw_t<T>, N, Align>(byte_offset);
}

/// Each of the following slm_block_load functions loads a contiguous memory
/// block from SLM (Shared Local Memory) and the \p byte_offset.
/// The parameter 'pred' is the one element predicate. If it is set to 1, then
/// all 'N' elements are loaded. Otherwise, the block load operation is a NO-OP.
/// The parameter 'pass_thru' specifies the values being copied to the returned
/// result if 'pred' is set to 0.
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.

/// simd<T, N> slm_block_load(uint32_t byte_offset, props={});     // (slm-bl-1)
/// simd<T, N> slm_block_load(uint32_t byte_offset,
///                           simd_mask<1> pred, props={});        // (slm-bl-2)
/// simd<T, N> slm_block_load(uint32_t byte_offset,
///                           simd_mask<1> pred,
///                           simd<T, N> pass_thru, props={});     // (slm-bl-3)

/// The following functions do the same work as slm_block_load(). They accept
/// a local accessor \p lacc and the load is done from SLM associated
/// with \p lacc plus \p byte_offset applied to it. If \p byte_offset
/// is omitted, then zero offset is used.
/// simd<T, N> block_load(local_accessor lacc, uint32_t byte_offset,
///                       props={});                              // (lacc-bl-1)
/// simd<T, N> block_load(local_accessor lacc, props={});         // (lacc-bl-2)
/// simd<T, N> block_load(local_accessor lacc, uint32_t byte_offset,
///                       simd_mask<1> pred, props={});           // (lacc-bl-3)
/// simd<T, N> block_load(local_accessor lacc,
///                       simd_mask<1> pred, props={});           // (lacc-bl-4)
/// simd<T, N> block_load(local_accessor lacc, uint32_t byte_offset,
///                       simd_mask<1> pred, simd<T, N> pass_thru,
///                       props={});                              // (lacc-bl-5)
/// simd<T, N> block_load(local_accessor lacc,
///                       simd_mask<1> pred, simd<T, N> pass_thru,
///                       props={});                              // (lacc-bl-6)

/// simd<T, N> slm_block_load(uint32_t byte_offset, props = {});   // (slm-bl-1)
/// Loads a contiguous memory block from SLM (Shared Local Memory) at
/// the given \p byte_offset. The parameter 'props' specifies the optional
/// compile-time properties list. Only esimd::alignment property is used. Other
/// properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is 16-bytes to generate block_load
/// instruction on all known target devices (Gen12, DG2, PVC, etc).
/// On Gen12 (opposing to DG2 and PVC) the alignment smaller than 8-bytes
/// is valid, but requires JIT compiler generating a slower GATHER instead
/// of faster BLOCK_LOAD.
/// !!! Passing \p byte_offset not aligned by 16-bytes and not specifying
/// the actual alignment in \p props produces incorrect load results on Gen12.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_block_load(uint32_t byte_offset, PropertyListT props = {}) {
  constexpr size_t DefaultAlignment = detail::OperandSize::OWORD;
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);
  return __esimd_slm_block_ld<detail::__raw_t<T>, N, Alignment>(byte_offset);
}

/// simd<T, N> slm_block_load(uint32_t byte_offset, simd_mask<N> pred,
///                           props = {});                        // (slm-bl-2)
/// Loads a contiguous memory block from SLM (Shared Local Memory) at the
/// given \p byte_offset.
/// The parameter \p pred is the one-element predicate. If it is set to 1,
/// then all 'N' elements are loaded. Otherwise, the block load operation
/// is a NO-OP.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is the minimally required (see (R1) below).
///
/// Restrictions - predicate imposed - temporary:
/// R1: The \p byte_offset must be at least 4-byte aligned for 4-byte or smaller
///     elements and 8-byte aligned for 8-byte elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_block_load(uint32_t byte_offset, simd_mask<1> pred,
               PropertyListT props = {}) {
  // Verify input template arguments.
  constexpr size_t DefaultAlignment = sizeof(T) <= 4 ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor64Bit = sizeof(uint64_t) / sizeof(T);
  constexpr int SmallIntFactor32Bit =
      sizeof(uint32_t) / sizeof(T) > 1 ? sizeof(uint32_t) / sizeof(T) : 1;
  static_assert(N > 0 && N % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");

  // If alignment >= 8 and (N * sizeof(T)) % 8 == 0) we can load QWORDs.
  // Don't do it for 4-byte vectors (unless it is greater than 256-bytes),
  // because it would require a bit-cast, which is supposed to be NO-OP, but
  // might confuse GPU BE sometimes. 1- and 2-byte vectors are casted anyways.
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (N * sizeof(T)) % sizeof(uint64_t) == 0 &&
      (sizeof(T) != sizeof(uint32_t) || N * sizeof(T) > 256);
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredN = N / SmallIntFactor;
  detail::check_lsc_vector_size<FactoredN>();

  // Prepare template arguments for the call of intrinsic.
  using LoadElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;

  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr detail::lsc_data_size DS =
      Use64BitData ? detail::lsc_data_size::u64 : detail::lsc_data_size::u32;
  constexpr auto VS = detail::to_lsc_vector_size<FactoredN>();
  constexpr auto Transposed = detail::lsc_data_order::transpose;
  constexpr int NLanes = 1;

  // Prepare non-template arguments and call the intrinsic.
  simd<uint32_t, NLanes> Offsets = byte_offset;
  simd<LoadElemT, FactoredN> Result =
      __esimd_lsc_load_slm<LoadElemT, cache_hint::none, cache_hint::none,
                           AddressScale, ImmOffset, DS, VS, Transposed, NLanes>(
          pred.data(), Offsets.data());
  return Result.template bit_cast_view<T>();
}

/// simd<T, N> slm_block_load(uint32_t byte_offset,
///                           simd_mask<1> pred,
///                           simd<T, N> pass_thru, props={});     // (slm-bl-3)
/// Loads a contiguous memory block from SLM (Shared Local Memory) at the
/// given \p byte_offset.
/// The parameter \p pred is the one-element predicate. If it is set to 1,
/// then all 'N' elements are loaded. Otherwise, the block load operation
/// is a NO-OP.
/// The parameter 'pass_thru' specifies the values being copied to the returned
/// result if 'pred' is set to 0.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is the minimally required (see (R1) below).
///
/// Restrictions - predicate imposed - temporary:
/// R1: The \p byte_offset must be at least 4-byte aligned for 4-byte or smaller
///     elements and 8-byte aligned for 8-byte elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_block_load(uint32_t offset, simd_mask<1> pred, simd<T, N> pass_thru,
               PropertyListT props = {}) {
  // Verify input template arguments.
  constexpr size_t DefaultAlignment = sizeof(T) <= 4 ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor64Bit = sizeof(uint64_t) / sizeof(T);
  constexpr int SmallIntFactor32Bit =
      sizeof(uint32_t) / sizeof(T) > 1 ? sizeof(uint32_t) / sizeof(T) : 1;
  static_assert(N > 0 && N % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");

  // If alignment >= 8 and (N * sizeof(T)) % 8 == 0) we can load QWORDs.
  // Don't do it for 4-byte vectors (unless it is greater than 256-bytes),
  // because it would require a bit-cast, which is supposed to be NO-OP, but
  // might confuse GPU BE sometimes. 1- and 2-byte vectors are casted anyways.
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (N * sizeof(T)) % sizeof(uint64_t) == 0 &&
      (sizeof(T) != sizeof(uint32_t) || N * sizeof(T) > 256);
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredN = N / SmallIntFactor;
  detail::check_lsc_vector_size<FactoredN>();

  // Prepare template arguments for the call of intrinsic.
  using LoadElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;

  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr detail::lsc_data_size DS =
      Use64BitData ? detail::lsc_data_size::u64 : detail::lsc_data_size::u32;
  constexpr auto VS = detail::to_lsc_vector_size<FactoredN>();
  constexpr auto Transposed = detail::lsc_data_order::transpose;
  constexpr int NLanes = 1;

  // Prepare non-template arguments and call the intrinsic.
  simd<uint32_t, NLanes> Offsets = offset;
  simd<LoadElemT, FactoredN> PassThru =
      pass_thru.template bit_cast_view<LoadElemT>();
  simd<LoadElemT, FactoredN> Result =
      __esimd_lsc_load_merge_slm<LoadElemT, cache_hint::none, cache_hint::none,
                                 AddressScale, ImmOffset, DS, VS, Transposed,
                                 NLanes>(pred.data(), Offsets.data(),
                                         PassThru.data());
  return Result.template bit_cast_view<T>();
}

/// simd<T, N> block_load(local_accessor lacc, uint32_t byte_offset,
///                       props={});                              // (lacc-bl-1)
/// Loads a contiguous memory block from SLM (Shared Local Memory) associated
/// with the local accessor \p lacc at the given \p byte_offset.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is 16-bytes to generate block_load
/// instruction on all known target devices (Gen12, DG2, PVC, etc).
/// On Gen12 (opposing to DG2 and PVC) the alignment smaller than 8-bytes
/// is valid, but requires JIT compiler generating a slower GATHER instead
/// of faster BLOCK_LOAD.
/// !!! Passing local accessor associated with SLM starting from offset that
/// is NOT aligned by 16-bytes and NOT specifying the actual alignment in
/// \p props produces incorrect load results on Gen12.
///
/// Note: if two or more local accessors are used in the same kernel, then
/// 16-byte alignment is guaranteed only for one of them.
/// Other local accessors may or may not get 16-byte alignment. N-th local
/// accessor's alignment depends on N-1 local accessor sizes, and their
/// element-alignment/padding. Only element-alignment is guaranteed for them.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_read> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
block_load(AccessorT lacc, uint32_t byte_offset, PropertyListT props = {}) {
  byte_offset += detail::localAccessorToOffset(lacc);
  return slm_block_load<T, N>(byte_offset, props);
}

/// simd<T, N> block_load(local_accessor lacc, props={});         // (lacc-bl-2)
/// Loads a contiguous memory block from SLM (Shared Local Memory) associated
/// with the local accessor \p lacc at zero offset.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is 16-bytes to generate block_load
/// instruction on all known target devices (Gen12, DG2, PVC, etc).
/// On Gen12 (opposing to DG2 and PVC) the alignment smaller than 8-bytes
/// is valid, but requires JIT compiler generating a slower GATHER instead
/// of faster BLOCK_LOAD.
/// !!! Passing local accessor associated with SLM starting from offset that
/// is NOT aligned by 16-bytes and NOT specifying the actual alignment in
/// \p props produces incorrect load results on Gen12.
///
/// Note: if two or more local accessors are used in the same kernel, then
/// 16-byte alignment is guaranteed only for one of them.
/// Other local accessors may or may not get 16-byte alignment. N-th local
/// accessor's alignment depends on N-1 local accessor sizes, and their
/// element-alignment/padding. Only element-alignment is guaranteed for them.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_read> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
block_load(AccessorT lacc, PropertyListT props = {}) {
  return slm_block_load<T, N>(detail::localAccessorToOffset(lacc), props);
}

/// simd<T, N> block_load(local_accessor lacc, uint32_t byte_offset,
///                       simd_mask<1> pred, props={});           // (lacc-bl-3)
/// Loads a contiguous memory block from SLM (Shared Local Memory) associated
/// the local accessor \p lacc at the given \p byte_offset.
///
/// The parameter \p pred is the one-element predicate. If it is set to 1,
/// then all 'N' elements are loaded. Otherwise, the block load operation
/// is a NO-OP, and some undefined value is returned.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is the minimally required (see (R1) below).
///
/// Restrictions - predicate imposed - temporary:
/// R1: The \p lacc + \p byte_offset must be at least 4-byte aligned for 4-byte
///     or smaller elements and 8-byte aligned for 8-byte elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_read> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
block_load(AccessorT lacc, uint32_t byte_offset, simd_mask<1> pred,
           PropertyListT props = {}) {
  byte_offset += detail::localAccessorToOffset(lacc);
  return slm_block_load<T, N>(byte_offset, pred, props);
}

/// simd<T, N> block_load(local_accessor lacc,
///                       simd_mask<1> pred, props={});           // (lacc-bl-4)
/// Loads a contiguous memory block from SLM (Shared Local Memory) associated
/// with the local accessor \p lacc at zero offset.
///
/// The parameter \p pred is the one-element predicate. If it is set to 1,
/// then all 'N' elements are loaded. Otherwise, the block load operation
/// is a NO-OP, and some undefined value is returned.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is the minimally required (see (R1) below).
///
/// Restrictions - predicate imposed - temporary:
/// R1: The local accessor \p lacc must point to memory at least 4-byte aligned
///     for elements of 4-bytes or smaller and 8-byte aligned for 8-byte
///     elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2), or 128;
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2), or 256;
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2), or 512.
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_read> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
block_load(AccessorT lacc, simd_mask<1> pred, PropertyListT props = {}) {
  return slm_block_load<T, N>(detail::localAccessorToOffset(lacc), pred, props);
}

/// simd<T, N> block_load(local_accessor lacc, uint32_t byte_offset,
///                       simd_mask<1> pred, simd<T, N> pass_thru,
///                       props={});                              // (lacc-bl-5)
/// Loads a contiguous memory block from SLM (Shared Local Memory) associated
/// the local accessor \p lacc at the given \p byte_offset.
/// The parameter \p pred is the one-element predicate. If it is set to 1,
/// then all 'N' elements are loaded. Otherwise, the block load operation
/// is a NO-OP, and \p pass_thru value is returned.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is the minimally required (see (R1) below).
///
/// Restrictions - predicate imposed - temporary:
/// R1: The \p lacc + \p byte_offset must be at least 4-byte aligned for 4-byte
///     or smaller elements and 8-byte aligned for 8-byte elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_read> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
block_load(AccessorT lacc, uint32_t byte_offset, simd_mask<1> pred,
           simd<T, N> pass_thru, PropertyListT props = {}) {
  byte_offset += __ESIMD_DNS::localAccessorToOffset(lacc);
  return slm_block_load<T, N>(byte_offset, pred, pass_thru, props);
}

/// simd<T, N> block_load(local_accessor lacc,
///                       simd_mask<1> pred, simd<T, N> pass_thru,
///                       props={});                              // (lacc-bl-6)
/// Loads a contiguous memory block from SLM (Shared Local Memory) associated
/// with the local accessor \p lacc at zero offset.
///
/// The parameter \p pred is the one-element predicate. If it is set to 1,
/// then all 'N' elements are loaded. Otherwise, the block load operation
/// is a NO-OP, and \p pass_thru value is returned.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is the minimally required (see (R1) below).
///
/// Restrictions - predicate imposed - temporary:
/// R1: The local accessor \p lacc must point to memory at least 4-byte aligned
///     for elements of 4-bytes or smaller and 8-byte aligned for 8-byte
///     elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2), or 128;
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2), or 256;
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2), or 512.
/// R2: The target device must be DG2, PVC or newer GPU.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_read> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
block_load(AccessorT lacc, simd_mask<1> pred, simd<T, N> pass_thru,
           PropertyListT props = {}) {
  return slm_block_load<T, N>(__ESIMD_DNS::localAccessorToOffset(lacc), pred,
                              pass_thru, props);
}

/// Stores elements of the vector \p vals to a contiguous block of SLM memory
/// at the given byte-offset \p offset.
/// The generated code depends on the combination {T, N, Flags}.
/// Providing flags specifying the alignment of 16-bytes or more produces more
/// efficient code. If the alignment is smaller than 16-bytes, then less
/// efficient scatter is generated. If the stored vector is too long
/// for 1 flat-store GPU instruction, then a series of flat-store and/or
/// scatters may be generated.
/// @tparam T Element type.
/// @tparam N Number of elements to store.
/// @tparam Flags The alignment specifier type tag.
/// @param offset The byte-offset to store at.
/// @param vals The vector to store.
/// @param Flags Specifies the alignment.
///
template <typename T, int N, typename Flags>
__ESIMD_API std::enable_if_t<is_simd_flag_type_v<Flags>>
slm_block_store(uint32_t offset, simd<T, N> vals, Flags) {
  constexpr size_t Align = Flags::template alignment<simd<T, N>>;
  __esimd_slm_block_st<detail::__raw_t<T>, N, Align>(offset, vals.data());
}

/// Each of the following slm_block_store functions stores the vector \p vals to
/// a contiguous memory block in SLM (Shared Local Memory) at the \p
/// byte_offset. The parameter 'pred' is the one element predicate. If it is set
/// to 1, then all 'N' elements are stored. Otherwise, the block store operation
/// is a NO-OP. The parameter 'props' specifies the optional compile-time
/// properties list. Only esimd::alignment property is used. Other properties
/// are ignored.

/// void slm_block_store(uint32_t byte_offset, simd<T, N> vals, // (slm-bs-1)
///                      simd_mask<1> pred, props={});
/// void slm_block_store(uint32_t byte_offset, simd<T, N> vals, // (slm-bs-2)
///                      props={});
///
/// The following functions do the same work as slm_block_store(). They accept
/// a local accessor \p lacc and the store of \p vals is done to SLM associated
/// with \p lacc plus \p byte_offset applied to it. If \p byte_offset
/// is omitted, then zero offset is used.
/// void block_store(local_accessor lacc, uint32_t byte_offset, // (lacc-bs-1)
///                  simd<T, N> vals, props={});
///
/// void block_store(local_accessor lacc, simd<T, N> vals,      // (lacc-bs-2)
///                  props={});
///
/// void block_store(local_accessor lacc, uint32_t byte_offset, // (lacc-bs-3)
///                  simd<T, N> vals,
///                  simd_mask<1> pred, props={});
///
/// void block_store(local_accessor lacc, simd<T, N> vals,      // (lacc-bs-4)
///                  simd_mask<1> pred, props={});
///
/// void slm_block_store(uint32_t byte_offset, simd<T, N> vals, // (slm-bs-1)
///                      simd_mask<1> pred, props={});
/// Stores the vector \p vals to a contiguous memory block in SLM (Shared Local
/// Memory) at the given \p byte_offset. The parameter \p pred is the
/// one-element predicate. If it is set to 1, then all 'N' elements are stored.
/// Otherwise, the block stored operation is a NO-OP.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is the minimally required (see (R1) below).
///
/// Restrictions - predicate imposed - temporary:
/// R1: The \p byte_offset must be at least 4-byte aligned for 4-byte or smaller
///     elements and 8-byte aligned for 8-byte elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_block_store(uint32_t byte_offset, simd<T, N> vals, simd_mask<1> pred,
                PropertyListT props = {}) {
  // Verify input template arguments.
  constexpr size_t DefaultAlignment = sizeof(T) <= 4 ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor64Bit = sizeof(uint64_t) / sizeof(T);
  constexpr int SmallIntFactor32Bit =
      sizeof(uint32_t) / sizeof(T) > 1 ? sizeof(uint32_t) / sizeof(T) : 1;

  static_assert(N > 0 && N % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed store");

  // If alignment >= 8 and (N * sizeof(T)) % 8 == 0) we can store QWORDs.
  // Don't do it for 4-byte vectors (unless it is greater than 256-bytes),
  // because it would require a bit-cast, which is supposed to be NO-OP, but
  // might confuse GPU BE sometimes. 1- and 2-byte vectors are casted anyways.
  constexpr bool Use64BitData =
      Alignment >= __ESIMD_DNS::OperandSize::QWORD &&
      (N * sizeof(T)) % sizeof(uint64_t) == 0 &&
      (sizeof(T) != sizeof(uint32_t) || N * sizeof(T) > 256);
  constexpr int SmallIntFactor =
      Use64BitData ? SmallIntFactor64Bit : SmallIntFactor32Bit;
  constexpr int FactoredN = N / SmallIntFactor;
  detail::check_lsc_vector_size<FactoredN>();

  // Prepare template arguments for the call of intrinsic.
  using StoreElemT = __ESIMD_DNS::__raw_t<
      std::conditional_t<SmallIntFactor == 1, T,
                         std::conditional_t<Use64BitData, uint64_t, uint32_t>>>;

  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr detail::lsc_data_size DS =
      Use64BitData ? detail::lsc_data_size::u64 : detail::lsc_data_size::u32;
  constexpr auto VS = detail::to_lsc_vector_size<FactoredN>();
  constexpr auto Transposed = detail::lsc_data_order::transpose;
  constexpr int NLanes = 1;

  // Prepare non-template arguments and call the intrinsic.
  simd<uint32_t, NLanes> Offsets = byte_offset;
  __esimd_lsc_store_slm<StoreElemT, cache_hint::none, cache_hint::none,
                        AddressScale, ImmOffset, DS, VS, Transposed, NLanes>(
      pred.data(), Offsets.data(),
      sycl::bit_cast<__ESIMD_DNS::vector_type_t<StoreElemT, FactoredN>>(
          vals.data()));
}

/// void slm_block_store(uint32_t byte_offset, simd<T, N> vals, // (slm-bs-2)
///                      props = {});
/// Stores the vector \p vals to a contiguous memory block in SLM
/// (Shared Local Memory) at the given \p byte_offset. The parameter 'props'
/// specifies the optional compile-time properties list. Only esimd::alignment
/// property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is 16-bytes to generate block_store
/// instruction on all known target devices (Gen12, DG2, PVC, etc).
/// On Gen12 (opposing to DG2 and PVC) the alignment smaller than 8-bytes
/// is valid, but requires JIT compiler generating a slower SCATTER instead
/// of faster BLOCK_STORE.
/// !!! Passing \p byte_offset not aligned by 16-bytes and not specifying
/// the actual alignment in \p props produces incorrect store results on Gen12.
template <typename T, int N,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_block_store(uint32_t byte_offset, simd<T, N> vals,
                PropertyListT props = {}) {
  constexpr size_t DefaultAlignment = detail::OperandSize::OWORD;
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);
  using StoreElemT = detail::__raw_t<T>;
  __esimd_slm_block_st<StoreElemT, N, Alignment>(
      byte_offset,
      sycl::bit_cast<__ESIMD_DNS::vector_type_t<StoreElemT, N>>(vals.data()));
}

/// void block_store(local_accessor lacc, uint32_t byte_offset, // (lacc-bs-1)
///                  simd<T, N> vals, props={});
/// Stores the vector \p vals to a contiguous memory block in SLM (Shared Local
/// Memory) associated with the local accessor \p lacc at the given \p
/// byte_offset. The parameter 'props' specifies the optional compile-time
/// properties list. Only esimd::alignment property is used. Other properties
/// are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is 16-bytes to generate block_store
/// instruction on all known target devices (Gen12, DG2, PVC, etc).
/// On Gen12 (opposing to DG2 and PVC) the alignment smaller than 8-bytes
/// is valid, but requires JIT compiler generating a slower SCATTER instead
/// of faster BLOCK_STORE.
/// !!! Passing \p byte_offset not aligned by 16-bytes and not specifying
/// the actual alignment in \p props produces incorrect store results on Gen12.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(AccessorT lacc, uint32_t byte_offset, simd<T, N> vals,
            PropertyListT props = {}) {
  byte_offset += detail::localAccessorToOffset(lacc);
  slm_block_store<T, N>(byte_offset, vals, props);
}

/// void block_store(local_accessor lacc, simd<T, N> vals, // (lacc-bs-2)
///                  props={});
/// Stores the vector \p vals to a contiguous memory block in SLM
/// (Shared Local Memory) associated with the local accessor \p lacc. The
/// parameter 'props' specifies the optional compile-time properties list. Only
/// esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is 16-bytes to generate block_store
/// instruction on all known target devices (Gen12, DG2, PVC, etc).
/// On Gen12 (opposing to DG2 and PVC) the alignment smaller than 8-bytes
/// is valid, but requires JIT compiler generating a slower SCATTER instead
/// of faster BLOCK_STORE.
/// !!! Passing \p byte_offset not aligned by 16-bytes and not specifying
/// the actual alignment in \p props produces incorrect store results on Gen12.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(AccessorT lacc, simd<T, N> vals, PropertyListT props = {}) {
  slm_block_store<T, N>(detail::localAccessorToOffset(lacc), vals, props);
}

/// void block_store(local_accessor lacc, uint32_t byte_offset, // (lacc-bs-3)
///                  simd<T, N> vals, simd_mask<1> pred, props={});
///
/// Stores the vector \p vals to a contiguous memory block in SLM (Shared Local
/// Memory) associated with the local accessor \p lacc at the given \p
/// byte_offset. The parameter \p pred is the one-element predicate. If it is
/// set to 1, then all 'N' elements are stored. Otherwise, the block store
/// operation is a NO-OP.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is the minimally required (see (R1) below).
///
/// Restrictions - predicate imposed - temporary:
/// R1: The \p byte_offset must be at least 4-byte aligned for 4-byte or smaller
///     elements and 8-byte aligned for 8-byte elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(AccessorT lacc, uint32_t byte_offset, simd<T, N> vals,
            simd_mask<1> pred, PropertyListT props = {}) {
  byte_offset += detail::localAccessorToOffset(lacc);
  slm_block_store<T, N>(byte_offset, vals, pred, props);
}

/// void block_store(local_accessor lacc, simd<T, N> vals, // (lacc-bs-4)
///                  simd_mask<1> pred, props={});
/// Stores the vector \p vals to a contiguous memory block in SLM (Shared Local
/// Memory) associated with the local accessor \p lacc. The parameter \p pred is
/// the one-element predicate. If it is set to 1, then all 'N' elements are
/// stored. Otherwise, the block store operation is a NO-OP.
///
/// The parameter 'props' specifies the optional compile-time properties
/// list. Only esimd::alignment property is used. Other properties are ignored.
///
/// Alignment: If \p props does not specify the 'alignment' property, then
/// the default expected alignment is the minimally required (see (R1) below).
///
/// Restrictions - predicate imposed - temporary:
/// R1: The \p byte_offset must be at least 4-byte aligned for 4-byte or smaller
///     elements and 8-byte aligned for 8-byte elements.
/// R2: The number of elements must be:
///     for 8-byte data: 1, 2, 3, 4, 8, 16, 32(max for DG2), 64;
///     for 4-byte data: 1, 2, 3, 4, 8, 16, 32, 64(max for DG2),
///                      or 128(only if alignment is 8-bytes or more);
///     for 2-byte data: 2, 4, 6, 8, 16, 32, 64, 128(max for DG2),
///                      or 256(only if alignment is 8-bytes or more);
///     for 1-byte data: 4, 8, 12, 16, 32, 64, 128, 256(max for DG2),
///                      or 512(only if alignment is 8-bytes or more).
/// R3: The target device must be DG2, PVC or newer GPU.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(AccessorT lacc, simd<T, N> vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  slm_block_store<T, N>(detail::localAccessorToOffset(lacc), vals, pred, props);
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

/// SLM atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @param offsets is the zero-based offsets.
/// @param pred is predicate.
///
/// @return A vector of the old values at the memory locations before the
///   update.

template <atomic_op Op, typename T, int N, lsc_data_size DS>
__ESIMD_API std::enable_if_t<get_num_args<Op>() == 0, simd<T, N>>
slm_atomic_update_impl(simd<uint32_t, N> offsets, simd_mask<N> pred) {
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 0, /*IsLSC*/ true>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<1>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  constexpr int IOp = lsc_to_internal_atomic_op<T, Op>();
  simd<MsgT, N> Tmp =
      __esimd_lsc_xatomic_slm_0<MsgT, IOp, cache_hint::none, cache_hint::none,
                                AddressScale, ImmOffset, EDS, VS, Transposed,
                                N>(pred.data(), offsets.data());
  return lsc_format_ret<T>(Tmp);
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
/// @param pred is predicate.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, lsc_data_size DS>
__ESIMD_API std::enable_if_t<get_num_args<Op>() == 1, simd<T, N>>
slm_atomic_update_impl(simd<uint32_t, N> offsets, simd<T, N> src0,
                       simd_mask<N> pred) {
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 1, /*IsLSC*/ true>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<1>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  constexpr int IOp = lsc_to_internal_atomic_op<T, Op>();
  if constexpr (std::is_same_v<T, double>) {
    return __esimd_lsc_xatomic_slm_1<T, IOp, cache_hint::none, cache_hint::none,
                                     AddressScale, ImmOffset, EDS, VS,
                                     Transposed, N>(pred.data(), offsets.data(),
                                                    src0.data());
  } else {
    using MsgT = typename lsc_expand_type<T>::type;
    simd<MsgT, N> Msg_data = lsc_format_input<MsgT>(src0);
    simd<MsgT, N> Tmp =
        __esimd_lsc_xatomic_slm_1<MsgT, IOp, cache_hint::none, cache_hint::none,
                                  AddressScale, ImmOffset, EDS, VS, Transposed,
                                  N>(pred.data(), offsets.data(),
                                     Msg_data.data());
    return lsc_format_ret<T>(Tmp);
  }
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
template <atomic_op Op, typename T, int N, lsc_data_size DS>
__ESIMD_API simd<T, N> slm_atomic_update_impl(simd<uint32_t, N> offsets,
                                              simd<T, N> src0, simd<T, N> src1,
                                              simd_mask<N> pred) {
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 2, /*IsLSC*/ true>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<1>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  constexpr int IOp = lsc_to_internal_atomic_op<T, Op>();
  if constexpr (std::is_same_v<T, double>) {
    return __esimd_lsc_xatomic_slm_2<T, IOp, cache_hint::none, cache_hint::none,
                                     AddressScale, ImmOffset, EDS, VS,
                                     Transposed, N>(pred.data(), offsets.data(),
                                                    src0.data(), src1.data());
  } else {
    using MsgT = typename lsc_expand_type<T>::type;
    simd<MsgT, N> Msg_data0 = lsc_format_input<MsgT>(src0);
    simd<MsgT, N> Msg_data1 = lsc_format_input<MsgT>(src1);
    simd<MsgT, N> Tmp =
        __esimd_lsc_xatomic_slm_2<MsgT, IOp, cache_hint::none, cache_hint::none,
                                  AddressScale, ImmOffset, EDS, VS, Transposed,
                                  N>(pred.data(), offsets.data(),
                                     Msg_data0.data(), Msg_data1.data());
    return lsc_format_ret<T>(Tmp);
  }
}

} // namespace detail

/// @anchor slm_atomic_update0
/// @brief Atomic update operation performed on SLM.
/// No-argument variant of the atomic update operation.

/// simd<T, N>
/// slm_atomic_update(simd<uint32_t, N> byte_offset,
///                   simd_mask<N> mask = 1);                   /// (slm-au0-1)

/// The following functions do the same work as slm_atomic_update(). They accept
/// a local accessor \p lacc and the atomic update is done from SLM associated
/// with \p lacc plus \p byte_offset applied to it. If \p byte_offset
/// is omitted, then zero offset is used.

/// simd<T, N> atomic_update(local_accessor lacc,
///                          simd<uint32_t, N> byte_offset,
///                          simd_mask<1> pred = 1);
///                                                             // (lacc-au0-1)

/// Usage of cache hints or non-standard operation width N requires DG2 or PVC.

/// simd<T, N>
/// slm_atomic_update(simd<uint32_t, N> byte_offset,
///                   simd_mask<N> mask = 1);                   /// (slm-au0-1)
///
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation - can be \c atomic_op::inc or
/// \c atomic_op::dec, \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param byte_offset The vector of 32-bit offsets.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 0, simd<T, N>>
slm_atomic_update(simd<uint32_t, N> byte_offset, simd_mask<N> mask = 1) {
  // 2 byte, 8 byte types, non-power of two, and operations wider than 32 are
  // supported only by LSC.
  if constexpr (sizeof(T) == 2 || sizeof(T) == 8 ||
                !__ESIMD_DNS::isPowerOf2(N, 32)) {
    return slm_atomic_update_impl<Op, T, N,
                                  detail::lsc_data_size::default_size>(
        byte_offset, mask);
  } else if constexpr (Op == atomic_op::load) {
    if constexpr (std::is_integral_v<T>) {
      return slm_atomic_update<atomic_op::bit_or, T, N>(byte_offset,
                                                        simd<T, N>(0), mask);
    } else {
      using Tint = detail::uint_type_t<sizeof(T)>;
      simd<Tint, N> Res = slm_atomic_update<atomic_op::bit_or, Tint, N>(
          byte_offset, simd<Tint, N>(0), mask);
      return Res.template bit_cast_view<T>();
    }
  } else {
    detail::check_atomic<Op, T, N, 0>();
    const auto si = get_surface_index(detail::LocalAccessorMarker());
    return __esimd_dword_atomic0<Op, T, N>(mask.data(), si, byte_offset.data());
  }
}

/// simd<T, N> atomic_update(local_accessor lacc,
///                          simd<uint32_t, N> byte_offset,
///                          simd_mask<N> pred = 1);
///                                                             // (lacc-au0-1)
/// Atomically updates \c N memory locations in SLM ssociated
/// with the local accessor \p lacc at the given \p byte_offset,
/// and returns a vector of old values found at the memory locations before
/// update.
template <atomic_op Op, typename T, int N, typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 0 &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset,
              simd_mask<N> mask = 1) {
  byte_offset += detail::localAccessorToOffset(lacc);
  return slm_atomic_update<Op, T, N>(byte_offset, mask);
}

/// One argument variant of the atomic update operation.

/// simd<T, N>
/// slm_atomic_update(simd<uint32_t, N> byte_offset,
///                   simd<T, N> src0,
///                   simd_mask<N> mask = 1);                   /// (slm-au1-1)
///

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               simd<uint32_t, N> byte_offset,
///               simd<T, N> src0,
///               simd_mask<1> pred = 1);                       // (lacc-au1-1)
///

/// Usage of cache hints or non-standard operation width N requires DG2 or PVC.

/// simd<T, N>
/// slm_atomic_update(simd<uint32_t, N> byte_offset,
///                   simd<T, N> src0,
///                   simd_mask<N> mask = 1)                    /// (slm-au1-1)
///
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1, simd<T, N>>
slm_atomic_update(simd<uint32_t, N> byte_offset, simd<T, N> src0,
                  simd_mask<N> mask = 1) {
  // 2 byte, 8 byte types, non-power of two, and operations wider than 32 are
  // supported only by LSC.
  if constexpr (sizeof(T) == 2 || sizeof(T) == 8 ||
                !__ESIMD_DNS::isPowerOf2(N, 32)) {
    // half and short are supported in LSC.
    return slm_atomic_update_impl<Op, T, N,
                                  detail::lsc_data_size::default_size>(
        byte_offset, src0, mask);
  } else if constexpr (Op == atomic_op::store) {
    if constexpr (std::is_integral_v<T>) {
      return slm_atomic_update<atomic_op::xchg, T, N>(byte_offset, src0, mask);
    } else {
      using Tint = detail::uint_type_t<sizeof(T)>;
      simd<Tint, N> Res = slm_atomic_update<atomic_op::xchg, Tint, N>(
          byte_offset, src0.template bit_cast_view<Tint>(), mask);
      return Res.template bit_cast_view<T>();
    }
  } else {
    detail::check_atomic<Op, T, N, 1>();
    const auto si = get_surface_index(detail::LocalAccessorMarker());
    return __esimd_dword_atomic1<Op, T, N>(mask.data(), si, byte_offset.data(),
                                           src0.data());
  }
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               simd<uint32_t, N> byte_offset,
///               simd<T, N> src0,
///               simd_mask<1> pred = 1);                       // (lacc-au1-1)
///
/// Atomically updates \c N memory locations in SLM indicated by
/// local accessor \p lacc and a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1 &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset, simd<T, N> src0,
              simd_mask<N> mask = 1) {
  byte_offset += detail::localAccessorToOffset(lacc);
  return slm_atomic_update<Op, T, N>(byte_offset, src0, mask);
}

/// Two argument variant of the atomic update operation.

/// simd<T, N>
/// slm_atomic_update(simd<uint32_t, N> byte_offset,
///                   simd<T, N> src0, simd<T, N> src1,
///                   simd_mask<N> mask = 1);                   /// (slm-au2-1)

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               simd<uint32_t, N> byte_offset,
///               simd<T, N> src0,
///               simd<T, N> src1,
///               simd_mask<1> pred = 1);                      // (lacc-au2-1)
///

/// simd<T, N>
/// slm_atomic_update(simd<uint32_t, N> byte_offset,
///                   simd<T, N> src0, simd<T, N> src1,
///                   simd_mask<N> mask = 1);                   /// (slm-au2-1)
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand (new value).
/// @param src1 is the second atomic operand (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2, simd<T, N>>
slm_atomic_update(simd<uint32_t, N> byte_offset, simd<T, N> src0,
                  simd<T, N> src1, simd_mask<N> mask = 1) {
  // 2 byte, 8 byte types, non-power of two, and operations wider than 32 are
  // supported only by LSC.
  if constexpr (sizeof(T) == 2 || sizeof(T) == 8 ||
                !__ESIMD_DNS::isPowerOf2(N, 32)) {
    // 2-argument lsc_atomic_update arguments order matches the standard one -
    // expected value first, then new value. But atomic_update uses reverse
    // order, hence the src1/src0 swap.
    return detail::slm_atomic_update_impl<Op, T, N,
                                          detail::lsc_data_size::default_size>(
        byte_offset, src1, src0, mask);
  } else {
    detail::check_atomic<Op, T, N, 2>();
    const auto si = get_surface_index(detail::LocalAccessorMarker());
    return __esimd_dword_atomic2<Op, T, N>(mask.data(), si, byte_offset.data(),
                                           src0.data(), src1.data());
  }
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               simd<uint32_t, N> byte_offset,
///               simd<T, N> src0,
///               simd<T, N> src1,
///               simd_mask<1> pred = 1);                      // (lacc-au2-1)
template <atomic_op Op, typename T, int N, typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset, simd<T, N> src0,
              simd<T, N> src1, simd_mask<N> mask = 1) {
  byte_offset += detail::localAccessorToOffset(lacc);
  return slm_atomic_update<Op, T, N>(byte_offset, src0, src1, mask);
}

/// @} sycl_esimd_memory_slm

namespace detail {

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam Op is operation type.
/// @tparam T is element type.
/// @tparam N is the number of channels (platform dependent).
/// @tparam DS is the data size.
/// @tparam L1H is L1 cache hint.
/// @tparam L2H is L2 cache hint.
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param pred is predicates.
///
template <atomic_op Op, typename T, int N, lsc_data_size DS, cache_hint L1H,
          cache_hint L2H, typename Toffset>
__ESIMD_API std::enable_if_t<get_num_args<Op>() == 0, simd<T, N>>
atomic_update_impl(T *p, simd<Toffset, N> offsets, simd_mask<N> pred) {
  static_assert(sizeof(T) > 1, "Unsupported data type");
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  check_atomic<Op, T, N, 0, /*IsLSC*/ true>();
  check_lsc_data_size<T, DS>();
  check_cache_hint<cache_action::atomic, L1H, L2H>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<1>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  constexpr int IOp = lsc_to_internal_atomic_op<T, Op>();
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  simd<MsgT, N> Tmp =
      __esimd_lsc_xatomic_stateless_0<MsgT, IOp, L1H, L2H, AddressScale,
                                      ImmOffset, EDS, VS, Transposed, N>(
          pred.data(), addrs.data());
  return lsc_format_ret<T>(Tmp);
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
/// @tparam L2H is L2 cache hint.
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
template <atomic_op Op, typename T, int N, lsc_data_size DS, cache_hint L1H,
          cache_hint L2H, typename Toffset>
__ESIMD_API std::enable_if_t<get_num_args<Op>() == 1, simd<T, N>>
atomic_update_impl(T *p, simd<Toffset, N> offsets, simd<T, N> src0,
                   simd_mask<N> pred) {
  static_assert(sizeof(T) > 1, "Unsupported data type");
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 1, /*IsLSC*/ true>();
  check_cache_hint<cache_action::atomic, L1H, L2H>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<1>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  constexpr int IOp = lsc_to_internal_atomic_op<T, Op>();
  simd<MsgT, N> Msg_data = lsc_format_input<MsgT>(src0);
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  simd<MsgT, N> Tmp =
      __esimd_lsc_xatomic_stateless_1<MsgT, IOp, L1H, L2H, AddressScale,
                                      ImmOffset, EDS, VS, Transposed, N>(
          pred.data(), addrs.data(), Msg_data.data());
  return lsc_format_ret<T>(Tmp);
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
/// @tparam L2H is L2 cache hint.
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand (expected value).
/// @param src1 is the second atomic operand (new value).
/// @param pred predicates.
///
template <atomic_op Op, typename T, int N, lsc_data_size DS, cache_hint L1H,
          cache_hint L2H, typename Toffset>
__ESIMD_API std::enable_if_t<get_num_args<Op>() == 2, simd<T, N>>
atomic_update_impl(T *p, simd<Toffset, N> offsets, simd<T, N> src0,
                   simd<T, N> src1, simd_mask<N> pred) {
  static_assert(sizeof(T) > 1, "Unsupported data type");
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 2, /*IsLSC*/ true>();
  check_cache_hint<cache_action::atomic, L1H, L2H>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<1>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  constexpr int IOp = lsc_to_internal_atomic_op<T, Op>();
  simd<MsgT, N> Msg_data0 = lsc_format_input<MsgT>(src0);
  simd<MsgT, N> Msg_data1 = lsc_format_input<MsgT>(src1);
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  simd<MsgT, N> Tmp =
      __esimd_lsc_xatomic_stateless_2<MsgT, IOp, L1H, L2H, AddressScale,
                                      ImmOffset, EDS, VS, Transposed, N>(
          pred.data(), addrs.data(), Msg_data0.data(), Msg_data1.data());
  return lsc_format_ret<T>(Tmp);
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
/// @tparam L2H is L2 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param byte_offsets is the zero-based offsets.
/// @param pred is predicates.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L2H = cache_hint::none,
          typename AccessorTy, typename Toffset>
__ESIMD_API
    std::enable_if_t<get_num_args<Op>() == 0 &&
                         __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy>,
                     simd<T, N>>
    atomic_update_impl(AccessorTy acc, simd<Toffset, N> byte_offsets,
                       simd_mask<N> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update_impl<Op, T, N, DS, L1H, L2H>(accessorToPointer<T>(acc),
                                                    byte_offsets, pred);
#else
  static_assert(sizeof(T) > 1, "Unsupported data type");
  static_assert(std::is_integral_v<Toffset> && sizeof(Toffset) == 4,
                "Unsupported offset type");
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 0, /*IsLSC*/ true>();
  check_cache_hint<cache_action::atomic, L1H, L2H>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<1>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  constexpr int IOp = lsc_to_internal_atomic_op<T, Op>();
  auto si = get_surface_index(acc);
  simd<MsgT, N> Tmp =
      __esimd_lsc_xatomic_bti_0<MsgT, IOp, L1H, L2H, AddressScale, ImmOffset,
                                EDS, VS, Transposed, N>(
          pred.data(), byte_offsets.data(), si);
  return lsc_format_ret<T>(Tmp);
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
/// @tparam L2H is L2 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param byte_offset is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, lsc_data_size DS, cache_hint L1H,
          cache_hint L2H, typename AccessorTy, typename Toffset>
__ESIMD_API
    std::enable_if_t<get_num_args<Op>() == 1 &&
                         __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy>,
                     simd<T, N>>
    atomic_update_impl(AccessorTy acc, simd<Toffset, N> byte_offset,
                       simd<T, N> src0, simd_mask<N> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update_impl<Op, T, N, DS, L1H, L2H>(accessorToPointer<T>(acc),
                                                    byte_offset, src0, pred);
#else
  static_assert(sizeof(T) > 1, "Unsupported data type");
  static_assert(std::is_integral_v<Toffset> && sizeof(Toffset) == 4,
                "Unsupported offset type");
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 1, /*IsLSC*/ true>();
  check_cache_hint<cache_action::atomic, L1H, L2H>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<1>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  constexpr int IOp = lsc_to_internal_atomic_op<T, Op>();
  simd<MsgT, N> Src0Msg = lsc_format_input<MsgT>(src0);
  auto si = get_surface_index(acc);
  simd<MsgT, N> Tmp =
      __esimd_lsc_xatomic_bti_1<MsgT, IOp, L1H, L2H, AddressScale, ImmOffset,
                                EDS, VS, Transposed, N>(
          pred.data(), byte_offset.data(), Src0Msg.data(), si);
  return lsc_format_ret<T>(Tmp);
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
/// @tparam L2H is L2 cache hint.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param byte_offset is the zero-based offsets.
/// @param src0 is the first atomic operand (expected value).
/// @param src1 is the second atomic operand (new value).
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, lsc_data_size DS, cache_hint L1H,
          cache_hint L2H, typename AccessorTy, typename Toffset>
__ESIMD_API
    std::enable_if_t<get_num_args<Op>() == 2 &&
                         __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy>,
                     simd<T, N>>
    atomic_update_impl(AccessorTy acc, simd<Toffset, N> byte_offset,
                       simd<T, N> src0, simd<T, N> src1, simd_mask<N> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update_impl<Op, T, N, DS, L1H, L2H>(
      __ESIMD_DNS::accessorToPointer<T>(acc), byte_offset, src0, src1, pred);
#else
  static_assert(std::is_integral_v<Toffset> && sizeof(Toffset) == 4,
                "Unsupported offset type");
  check_lsc_vector_size<1>();
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 2, /*IsLSC*/ true>();
  check_cache_hint<cache_action::atomic, L1H, L2H>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<1>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  constexpr int IOp = lsc_to_internal_atomic_op<T, Op>();
  simd<MsgT, N> Msg_data0 = lsc_format_input<MsgT>(src0);
  simd<MsgT, N> Msg_data1 = lsc_format_input<MsgT>(src1);
  auto si = get_surface_index(acc);
  simd<MsgT, N> Tmp =
      __esimd_lsc_xatomic_bti_2<MsgT, IOp, L1H, L2H, AddressScale, ImmOffset,
                                EDS, VS, Transposed, N>(
          pred.data(), byte_offset.data(), Msg_data0.data(), Msg_data1.data(),
          si);
  return lsc_format_ret<T>(Tmp);
#endif
}
} // namespace detail

/// @addtogroup sycl_esimd_memory_atomics
/// @{

/// @anchor usm_atomic_update0
/// @brief No-argument variant of the atomic update operation.
///
/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd_mask<N> mask, props = {});               /// (usm-au0-1)
/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               props = {});                                  /// (usm-au0-2)
/// simd<T, N>
///
/// atomic_update(T *p, simd_view<OffsetObjT, RegionTy> byte_offset,
///               simd_mask<N> mask, props = {});               /// (usm-au0-3)
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, RegionTy> byte_offset,
///               props = {});                                  /// (usm-au0-4)
///
/// Usage of cache hints or non-standard operation width N requires DG2 or PVC.
///
/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd_mask<N> mask, props = {});               /// (usm-au0-1)
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has no arguments in addition to the value at the memory location.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
///   \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes
///  (zero-based).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, simd_mask<N> mask,
              PropertyListT props = {}) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");

  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);

  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);

  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                !__ESIMD_DNS::isPowerOf2(N, 32)) {
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, L1Hint, L2Hint, Toffset>(
        p, byte_offset, mask);
  } else {
    if constexpr (Op == atomic_op::load) {
      if constexpr (std::is_integral_v<T>) {
        return atomic_update<atomic_op::bit_or, T, N>(
            p, byte_offset, simd<T, N>(0), mask, props);
      } else {
        using Tint = detail::uint_type_t<sizeof(T)>;
        simd<Tint, N> Res = atomic_update<atomic_op::bit_or, Tint, N>(
            reinterpret_cast<Tint *>(p), byte_offset, simd<Tint, N>(0), mask,
            props);
        return Res.template bit_cast_view<T>();
      }
    } else {
      detail::check_atomic<Op, T, N, 0>();

      simd<uintptr_t, N> vAddr(reinterpret_cast<uintptr_t>(p));
      simd<uintptr_t, N> offset_i1 = convert<uintptr_t>(byte_offset);
      vAddr += offset_i1;
      using Tx = typename detail::__raw_t<T>;
      return __esimd_svm_atomic0<Op, Tx, N>(vAddr.data(), mask.data());
    }
  }
}

/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               props = {});                                  /// (usm-au0-2)
///
/// A variation of \c atomic_update API without mask operand.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
///   \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes
///  (zero-based).
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, typename Toffset,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(p, byte_offset, mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, RegionTy> byte_offset,
///               simd_mask<N> mask, props = {});               /// (usm-au0-3)
///
/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
/// \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
///   Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename RegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd_view<OffsetObjT, RegionTy> offsets, simd_mask<N> mask,
              PropertyListT props = {}) {
  return atomic_update<Op, T, N>(p, offsets.read(), mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, RegionTy> byte_offset,
///               props = {});                                  /// (usm-au0-4)
///
/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object without mask operand.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
///   \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update
/// @param p The USM pointer.
/// @param byte_offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
///   Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename RegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd_view<OffsetObjT, RegionTy> byte_offset,
              PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(p, byte_offset.read(), mask, props);
}

/// A variation of \c atomic_update API with \c offset represented as
/// scalar.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
/// \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The scalar 32-bit or 64-bit offset in bytes.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset>, simd<T, N>>
atomic_update(T *p, Toffset byte_offset, simd_mask<N> mask = 1) {
  return atomic_update<Op, T, N>(p, simd<Toffset, N>(byte_offset), mask);
}

/// @anchor usm_atomic_update1
/// @brief Single-argument variant of the atomic update operation.
///
/// simd<T, N>
/// atomic_update(T *ptr, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd_mask<N> mask, props = {});//(usm-au1-1)
/// simd<T, N>
/// atomic_update(T *ptr, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, props = {});                  // (usm-au1-2)
///
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0,
///               simd_mask<N> mask, props = {});                // (usm-au1-3)
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0,
///               props = {});                                   // (usm-au1-4)
///

/// simd<T, N>
/// atomic_update(T *ptr, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd_mask<N> mask, props = {});//(usm-au1-1)
///
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 1 additional argument.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin, \c atomic_op::fadd, \c
/// atomic_op::fsub, \c atomic_op::store.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, simd<T, N> src0,
              simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");

  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);

  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);

  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  // Auto-convert FP atomics to LSC version.
  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                (Op == atomic_op::fmin) || (Op == atomic_op::fmax) ||
                (Op == atomic_op::fadd) || (Op == atomic_op::fsub) ||
                !__ESIMD_DNS::isPowerOf2(N, 32)) {
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, L1Hint, L2Hint, Toffset>(
        p, byte_offset, src0, mask);
  } else {
    if constexpr (Op == atomic_op::store) {
      if constexpr (std::is_integral_v<T>) {
        return atomic_update<atomic_op::xchg, T, N>(p, byte_offset, src0, mask,
                                                    props);
      } else {
        using Tint = detail::uint_type_t<sizeof(T)>;
        simd<Tint, N> Res = atomic_update<atomic_op::xchg, Tint, N>(
            reinterpret_cast<Tint *>(p), byte_offset,
            src0.template bit_cast_view<Tint>(), mask, props);
        return Res.template bit_cast_view<T>();
      }
    } else {
      detail::check_atomic<Op, T, N, 1>();
      simd<uintptr_t, N> vAddr(reinterpret_cast<uintptr_t>(p));
      simd<uintptr_t, N> offset_i1 = convert<uintptr_t>(byte_offset);
      vAddr += offset_i1;

      using Tx = typename detail::__raw_t<T>;
      return __esimd_svm_atomic1<Op, Tx, N>(vAddr.data(), src0.data(),
                                            mask.data());
    }
  }
}

/// simd<T, N>
/// atomic_update(T *ptr, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, props = {});                  // (usm-au1-2)

/// A variation of \c atomic_update API without mask operand.

/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin, \c atomic_op::fadd, \c
/// atomic_op::fsub, \c atomic_op::store.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The additional argument.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, simd<T, N> src0,
              PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(p, byte_offset, src0, mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0,
///               simd_mask<N> mask, props = {});                // (usm-au1-3)
///
/// A variation of \c atomic_update API with \c byte_offset represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename RegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd_view<OffsetObjT, RegionTy> offsets, simd<T, N> src0,
              simd_mask<N> mask, PropertyListT props = {}) {
  return atomic_update<Op, T, N>(p, offsets.read(), src0, mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0,
///               props = {});                                   // (usm-au1-4)
///
/// A variation of \c atomic_update API with \c byte_offset represented as
/// \c simd_view object and no mask operand.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// @param src0 The additional argument.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename RegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd_view<OffsetObjT, RegionTy> offsets, simd<T, N> src0,
              PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(p, offsets.read(), src0, mask, props);
}

/// A variation of \c atomic_update API with \c offset represented as
/// scalar object.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin \c atomic_op::store.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The scalar 32-bit or 64-bit offsets in bytes.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> &&
        ((Op != atomic_op::store && Op != atomic_op::xchg) || N == 1),
    simd<Tx, N>>
atomic_update(Tx *p, Toffset byte_offset, simd<Tx, N> src0, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(p, simd<Toffset, N>(byte_offset), src0, mask);
}

/// @anchor usm_atomic_update2
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {});               // (usm-au2-1)
/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               props = {});                                  // (usm-au2-2)
///
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {})                // (usm-au2-3)
/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               props = {})                                   // (usm-au2-4)
///

/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {});               // (usm-au2-1)
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, simd<T, N> src0,
              simd<T, N> src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");

  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);

  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);

  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  // Use LSC atomic when cache hints are present, FP atomics is used,
  // non-power of two length is used, or operation width greater than 32.
  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                Op == atomic_op::fcmpxchg || !__ESIMD_DNS::isPowerOf2(N, 32)) {
    // 2-argument lsc_atomic_update arguments order matches the standard one -
    // expected value first, then new value. But atomic_update uses reverse
    // order, hence the src1/src0 swap.
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, L1Hint, L2Hint, Toffset>(
        p, byte_offset, src1, src0, mask);
  } else {
    detail::check_atomic<Op, T, N, 2>();
    simd<uintptr_t, N> vAddr(reinterpret_cast<uintptr_t>(p));
    simd<uintptr_t, N> offset_i1 = convert<uintptr_t>(byte_offset);
    vAddr += offset_i1;
    using Tx = typename detail::__raw_t<T>;
    return __esimd_svm_atomic2<Op, Tx, N>(vAddr.data(), src0.data(),
                                          src1.data(), mask.data());
  }
}

/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               props = {});                                  // (usm-au2-2)
//
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, simd<T, N> src0,
              simd<T, N> src1, PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(p, byte_offset, src0, src1, mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {})                // (usm-au2-3)
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename OffsetRegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
              simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask,
              PropertyListT props = {}) {
  return atomic_update<Op, T, N>(p, byte_offset.read(), src0, src1, mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               props = {})                                   // (usm-au2-4)
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename OffsetRegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
              simd<T, N> src0, simd<T, N> src1, PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(p, byte_offset.read(), src0, src1, mask,
                                 props);
}

/// A variation of \c atomic_update API with \c byte_offset represented as
/// scalar.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param byte_offset The scalar 32-bit or 64-bit offset in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset>, simd<Tx, N>>
atomic_update(Tx *p, Toffset byte_offset, simd<Tx, N> src0, simd<Tx, N> src1,
              simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(p, simd<Toffset, N>(byte_offset), src0, src1,
                                  mask);
}

/// @anchor accessor_atomic_update0
/// @brief No-argument variant of the atomic update operation.
///
/// simd<T, N>
/// atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
///               simd_mask<N> mask, props = {});               /// (acc-au0-1)
/// simd<T, N>
/// atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
///               props = {});                                  /// (acc-au0-2)
/// simd<T, N>
/// atomic_update(AccessorT acc, simd_view<OffsetObjT, RegionTy> byte_offset,
///               simd_mask<N> mask, props = {});               /// (acc-au0-3)
/// simd<T, N>
/// atomic_update(AccessorT acc, simd_view<OffsetObjT, RegionTy> byte_offset,
///               props = {});                                  /// (acc-au0-4)
///

/// Usage of cache hints or non-standard operation width N requires DG2 or PVC.
///
/// simd<T, N>
/// atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
///               simd_mask<N> mask, props = {});               /// (acc-au0-1)
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets, and returns a vector of old values found at the
/// memory locations before update. The update operation has no arguments
/// in addition to the value at the memory location.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
/// \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, simd_mask<N> mask,
              PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update<Op, T, N>(__ESIMD_DNS::accessorToPointer<T>(acc),
                                 byte_offset, mask, props);
#else
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");

  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                !detail::isPowerOf2(N, 32)) {
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, L1Hint, L2Hint>(
        acc, byte_offset, mask);
  } else {
    if constexpr (Op == atomic_op::load) {
      if constexpr (std::is_integral_v<T>) {
        return atomic_update<atomic_op::bit_or, T, N>(
            acc, byte_offset, simd<T, N>(0), mask, props);
      } else {
        using Tint = detail::uint_type_t<sizeof(T)>;
        simd<Tint, N> Res = atomic_update<atomic_op::bit_or, Tint, N>(
            acc, byte_offset, simd<Tint, N>(0), mask, props);
        return Res.template bit_cast_view<T>();
      }
    } else {
      detail::check_atomic<Op, T, N, 0>();
      static_assert(sizeof(Toffset) == 4, "Only 32 bit offset is supported");

      static_assert(sizeof(T) == 4, "Only 32 bit data is supported");
      const auto si = get_surface_index(acc);
      using Tx = typename detail::__raw_t<T>;
      return __esimd_dword_atomic0<Op, Tx, N>(mask.data(), si,
                                              byte_offset.data());
    }
  }
#endif
}

/// simd<T, N>
/// atomic_update(AccessorT acc, simd<OffsetObjT, N> byte_offset,
///               props = {});                                  /// (acc-au0-2)
/// A variation of \c atomic_update API without mask operand
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
/// \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset,
              PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(acc, byte_offset, mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorT acc, simd_view<OffsetObjT, RegionTy> byte_offset,
///               simd_mask<N> mask, props = {});               /// (acc-au0-3)
/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
/// \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// 64-bit offsets are supported only when stateless memory accesses are
/// enforced, i.e. accessor based accesses are automatically converted to
/// stateless accesses.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename AccessorTy, typename RegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd_view<OffsetObjT, RegionTy> byte_offset,
              simd_mask<N> mask, PropertyListT props = {}) {
  return atomic_update<Op, T, N>(acc, byte_offset.read(), mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorT acc, simd_view<OffsetObjT, RegionTy> byte_offset,
///               props = {});                                  /// (acc-au0-4)
/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object and no mask operand.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
/// \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// 64-bit offsets are supported only when stateless memory accesses are
/// enforced, i.e. accessor based accesses are automatically converted to
/// stateless accesses.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename AccessorTy, typename RegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd_view<OffsetObjT, RegionTy> byte_offset,
              PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(acc, byte_offset.read(), mask, props);
}

/// A variation of \c atomic_update API with \c offset represented as
/// scalar.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
/// \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The scalar 32-bit or 64-bit offset in bytes. 64-bit
/// offset are supported only when stateless memory accesses are enforced,
/// i.e. accessor based accesses are automatically converted to stateless
/// accesses.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 0 &&
                         __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy>,
                     simd<T, N>>
    atomic_update(AccessorTy acc, Toffset byte_offset, simd_mask<N> mask) {
  return atomic_update<Op, T, N>(acc, simd<Toffset, N>(byte_offset), mask);
}

/// A variation of \c atomic_update API with \p byte_offset represented as
/// scalar using \c local_accessor.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
/// \c atomic_op::dec, or \c atomic_op::load.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The scalar 32-bit or 64-bit offset in bytes. 64-bit
/// offset are supported only when stateless memory accesses are enforced,
/// i.e. accessor based accesses are automatically converted to stateless
/// accesses.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename AccessorTy>
__ESIMD_API
    std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 0 &&
                         __ESIMD_DNS::is_rw_local_accessor_v<AccessorTy>,
                     simd<T, N>>
    atomic_update(AccessorTy acc, uint32_t byte_offset, simd_mask<N> mask) {
  return atomic_update<Op, T, N>(acc, simd<uint32_t, N>(byte_offset), mask);
}

/// @anchor accessor_atomic_update1
/// @brief Single-argument variant of the atomic update operation.
///
/// simd<T, N>
/// atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd_mask<N> mask, props = {});//(acc-au1-1)
/// simd<T, N>
/// atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, props = {});                  // (acc-au1-2)
///
/// simd<T, N>
/// atomic_update(AccessorT acc,
////              simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0,
///               simd_mask<N> mask, props = {});                // (acc-au1-3)
/// simd<T, N>
/// atomic_update(AccessorT acc,
///               simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0,
///               props = {});                                   // (acc-au1-4)
///

/// simd<T, N>
/// atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd_mask<N> mask, props = {});//(acc-au1-1)
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets, and returns a vector of old values found at the
/// memory locations before update. The update operation has 1 additional
/// argument.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin, \c atomic_op::fadd, \c
/// atomic_op::fsub, \c atomic_op::store.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///

template <atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, simd<T, N> src0,
              simd_mask<N> mask, PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update<Op, T, N>(__ESIMD_DNS::accessorToPointer<T>(acc),
                                 byte_offset, src0, mask, props);
#else
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);

  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);

  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert(sizeof(Toffset) == 4, "Only 32 bit offset is supported");
  // Auto-convert FP atomics to LSC version.
  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                Op == atomic_op::fmin || Op == atomic_op::fmax ||
                Op == atomic_op::fadd || Op == atomic_op::fsub ||
                !__ESIMD_DNS::isPowerOf2(N, 32)) {
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, L1Hint, L2Hint>(
        acc, byte_offset, src0, mask);
  } else if constexpr (Op == atomic_op::store) {
    if constexpr (std::is_integral_v<T>) {
      return atomic_update<atomic_op::xchg, T, N>(acc, byte_offset, src0, mask,
                                                  props);
    } else {
      using Tint = detail::uint_type_t<sizeof(T)>;
      simd<Tint, N> Res = atomic_update<atomic_op::xchg, Tint, N>(
          acc, byte_offset, src0.template bit_cast_view<Tint>(), mask, props);
      return Res.template bit_cast_view<T>();
    }
  } else {
    detail::check_atomic<Op, T, N, 1>();
    static_assert(sizeof(T) == 4, "Only 32 bit data is supported");
    const auto si = __ESIMD_NS::get_surface_index(acc);
    using Tx = typename detail::__raw_t<T>;
    return __esimd_dword_atomic1<Op, Tx, N>(
        mask.data(), si, byte_offset.data(),
        sycl::bit_cast<__ESIMD_DNS::vector_type_t<Tx, N>>(src0.data()));
  }
#endif
}

/// simd<T, N>
/// atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, props = {});                  // (acc-au1-2)
///
/// A variation of \c atomic_update API with no mask operand.
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets, and returns a vector of old values found at the
/// memory locations before update. The update operation has 1 additional
/// argument.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin, \c atomic_op::fadd, \c
/// atomic_op::fsub, \c atomic_op::store.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param src0 The additional argument.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, simd<T, N> src0,
              PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(acc, byte_offset, src0, mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorT acc,
///               simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0,
///               simd_mask<N> mask, props = {});                // (acc-au1-3)
///
/// A variation of \c atomic_update API with \c byte_offset represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// 64-bit offsets are supported only when stateless memory accesses are
/// enforced, i.e. accessor based accesses are automatically converted to
/// stateless accesses.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename AccessorTy, typename RegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd_view<OffsetObjT, RegionTy> byte_offset,
              simd<T, N> src0, simd_mask<N> mask, PropertyListT props = {}) {
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0, mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorT acc,
///               simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
///               simd<T, N> src0,
///               props = {});                                   // (acc-au1-4)
///
/// A variation of \c atomic_update API with \c byte_offset represented as
/// \c simd_view object and no mask operand.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// 64-bit offsets are supported only when stateless memory accesses are
/// enforced, i.e. accessor based accesses are automatically converted to
/// stateless accesses.
/// @param src0 The additional argument.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename AccessorTy, typename RegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd_view<OffsetObjT, RegionTy> byte_offset,
              simd<T, N> src0, PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0, mask, props);
}

/// A variation of \c atomic_update API with \c offset represented as
/// scalar object.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin \c atomic_op::store.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The scalar 32-bit or 64-bit offset in bytes. 64-bit
/// offset are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ((Op != atomic_op::store && Op != atomic_op::xchg) || N == 1),
    simd<T, N>>
atomic_update(AccessorTy acc, Toffset offset, simd<T, N> src0,
              simd_mask<N> mask) {
  return atomic_update<Op, T, N>(acc, simd<Toffset, N>(offset), src0, mask);
}

/// A variation of \c atomic_update API with \c offset represented as
/// scalar object and uses \c local_accessor.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin \c atomic_op::store.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The scalar 32-bit offset in bytes.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::is_rw_local_accessor_v<AccessorTy> &&
        ((Op != atomic_op::store && Op != atomic_op::xchg) || N == 1),
    simd<Tx, N>>
atomic_update(AccessorTy acc, uint32_t offset, simd<Tx, N> src0,
              simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(acc, simd<uint32_t, N>(offset), src0, mask);
}

/// @anchor accessor_atomic_update2
/// @brief Two-argument variant of the atomic update operation.
///
/// simd<T, N>
/// atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
//                simd_mask<N> mask,props = {});                 // (acc-au2-1)
///
/// simd<T, N>
/// atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               props = {});                                   // (acc-au2-2)
/// simd<T, N>
/// atomic_update(AccessorTy acc, simd_view<OffsetObjT, OffsetRegionTy>
///               byte_offset, simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {});                // (acc-au2-3)
///
/// simd<T, N>
/// atomic_update(AccessorTy acc,
///               simd_view<OffsetObjT, OffsetRegionTy>, byte_offset,
///               simd<T, N> src0, simd<T, N> src1, props = {}); // (acc-au2-4)
///

/// simd<T, N>
/// atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
//                simd_mask<N> mask,props = {});                 // (acc-au2-1)
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced,
/// i.e. accessor based accesses are automatically converted to stateless
/// accesses.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 && std::is_integral_v<Toffset> &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, simd<T, N> src0,
              simd<T, N> src1, simd_mask<N> mask, PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update<Op, T, N>(__ESIMD_DNS::accessorToPointer<T>(acc),
                                 byte_offset, src0, src1, mask, props);
#else
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);

  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);

  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert(sizeof(Toffset) == 4, "Only 32 bit offset is supported");
  // Use LSC atomic when cache hints are present, FP atomics is used,
  // non-power of two length is used, or operation width greater than 32.
  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none ||
                Op == atomic_op::fcmpxchg || !__ESIMD_DNS::isPowerOf2(N, 32)) {
    // 2-argument lsc_atomic_update arguments order matches the standard one -
    // expected value first, then new value. But atomic_update uses reverse
    // order, hence the src1/src0 swap.
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, L1Hint, L2Hint>(
        acc, byte_offset, src1, src0, mask);
  } else {
    detail::check_atomic<Op, T, N, 2>();
    static_assert(sizeof(T) == 4, "Only 32 bit data is supported");
    const auto si = __ESIMD_NS::get_surface_index(acc);
    using Tx = typename detail::__raw_t<T>;
    return __esimd_dword_atomic2<Op, Tx, N>(
        mask.data(), si, byte_offset.data(),
        sycl::bit_cast<__ESIMD_DNS::vector_type_t<Tx, N>>(src0.data()),
        sycl::bit_cast<__ESIMD_DNS::vector_type_t<Tx, N>>(src1.data()));
  }
#endif
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               props = {});                                   // (acc-au2-2)
///
/// A variation of \c atomic_update API with no mask operand.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param acc The SYCL accessor.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename T, int N, typename Toffset,
          typename AccessorTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, simd<T, N> src0,
              simd<T, N> src1, PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(acc, byte_offset, src0, src1, mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, simd_view<OffsetObjT, OffsetRegionTy>
///               byte_offset, simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {});              // (acc-au2-3)
///
/// A variation of \c atomic_update API with \c byte_offset represented as
/// a \c simd_view object.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param acc The SYCL accessor.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename AccessorTy, typename OffsetRegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
              simd<T, N> src0, simd<T, N> src1, simd_mask<N> mask,
              PropertyListT props = {}) {
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0, src1, mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc,
///               simd_view<OffsetObjT, OffsetRegionTy>, byte_offset,
///               simd<T, N> src0, simd<T, N> src1, props = {}); // (acc-au2-4)
///
/// A variation of \c atomic_update API with \c byte_offset represented as
/// a \c simd_view object and no mask operand.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @param acc The SYCL accessor.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, typename OffsetObjT,
          typename AccessorTy, typename OffsetRegionTy,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd_view<OffsetObjT, OffsetRegionTy> byte_offset,
              simd<T, N> src0, simd<T, N> src1, PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0, src1, mask,
                                 props);
}

/// A variation of \c atomic_update API with \c offsets represented as
/// scalar.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The scalar 32-bit or 64-bit offset in bytes. 64-bit
/// offset are supported only when stateless memory accesses are enforced,
/// i.e. accessor based accesses are automatically converted to stateless
/// accesses.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_rw_device_accessor_v<AccessorTy>,
                             simd<Tx, N>>
atomic_update(AccessorTy acc, Toffset offset, simd<Tx, N> src0,
              simd<Tx, N> src1, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(acc, simd<Toffset, N>(offset), src0, src1,
                                  mask);
}

/// A variation of \c atomic_update API with \c offsets represented as
/// scalar and \c local_accessor is used.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The scalar 32-bit offset in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::is_rw_local_accessor_v<AccessorTy>,
                             simd<Tx, N>>
atomic_update(AccessorTy acc, uint32_t offset, simd<Tx, N> src0,
              simd<Tx, N> src1, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(acc, simd<uint32_t, N>(offset), src0, src1,
                                  mask);
}

/// @} sycl_esimd_memory_atomics

/// @addtogroup sycl_esimd_memory
/// @{

/// Represetns a bit mask to control behavior of esimd::fence.
/// Enum elements define semantics of the bits in the mask.
enum fence_mask : uint8_t {
  /// “Commit enable” - wait for fence to complete before continuing.
  global_coherent_fence = 0x1,
  /// Flush the instruction cache.
  l3_flush_instructions = 0x2,
  /// Flush sampler (texture) cache.
  l3_flush_texture_data = 0x4,
  /// Flush constant cache.
  l3_flush_constant_data = 0x8,
  /// Flush constant cache.
  l3_flush_rw_data = 0x10,
  /// Issue SLM memory barrier only. If not set, the memory barrier is global.
  local_barrier = 0x20,
  /// Flush L1 read - only data cache.
  l1_flush_ro_data = 0x40,
  /// Creates a software (compiler) barrier, which does not generate
  /// any instruction and only prevents instruction scheduler from
  /// reordering instructions across this barrier at compile time.
  sw_barrier = 0x80
};

/// esimd::fence sets the memory read/write order.
/// @tparam cntl A bitmask composed from \c fence_mask bits.
///
template <uint8_t cntl> __ESIMD_API void fence() { __esimd_fence(cntl); }

__SYCL_DEPRECATED("use fence<fence_mask>()")
__ESIMD_API void fence(fence_mask cntl) { __esimd_fence(cntl); }

/// Memory fence.
/// Supported platforms: DG2, PVC
///
/// @tparam Kind is the memory kind.
/// @tparam FenceOp is the fence cache flush operation to apply after fence.
/// @tparam Scope is the fence operation scope.
template <memory_kind Kind = memory_kind::global,
          fence_flush_op FenceOp = fence_flush_op::none,
          fence_scope Scope = fence_scope::group>
__ESIMD_API void fence() {
  static_assert(
      Kind != memory_kind::local ||
          (FenceOp == fence_flush_op::none && Scope == fence_scope::group),
      "SLM fence must have 'none' lsc_fence_op and 'group' scope");
  constexpr int N = 16;
  simd_mask<N> Mask = 1;
  __esimd_lsc_fence<static_cast<uint8_t>(Kind), static_cast<uint8_t>(FenceOp),
                    static_cast<uint8_t>(Scope), N>(Mask.data());
}

/// Generic work-group barrier.
/// Performs barrier synchronization for all threads within the same thread
/// group. The barrier instruction causes the executing thread to wait until
/// all threads in the same thread group have executed the barrier
/// instruction. Memory ordering is also guaranteed by this instruction. The
/// behavior is undefined if this instruction is executed in divergent control
/// flow.
///
__ESIMD_API void barrier() {
  __esimd_fence(fence_mask::global_coherent_fence | fence_mask::local_barrier);
  __esimd_barrier();
}
/// @} sycl_esimd_memory

/// @addtogroup sycl_esimd_memory
/// @{

/// Media block load.
///
/// @tparam T is the element data type.
/// @tparam m is the height of the 2D block.
/// @tparam N is the width of the 2D block.
/// @tparam AccessorTy is type of the SYCL accessor.
/// @tparam plane is planar surface index.
/// @param acc is the SYCL accessor.
/// @param x is X-coordinate of the left upper rectangle corner in BYTES.
/// @param y is Y-coordinate of the left upper rectangle corner in ROWS.
/// @return the linearized 2D block data read from surface.
///
template <typename T, int m, int N, typename AccessorTy, unsigned plane = 0>
__ESIMD_API simd<T, m * N> media_block_load(AccessorTy acc, unsigned x,
                                            unsigned y) {
  constexpr unsigned Width = N * sizeof(T);
  static_assert(Width * m <= 256u,
                "data does not fit into a single dataport transaction");
  static_assert(Width <= 64u, "valid block width is in range [1, 64]");
  static_assert(m <= 64u, "valid block height is in range [1, 64]");
  static_assert(plane <= 3u, "valid plane index is in range [0, 3]");

  const auto si = __ESIMD_NS::get_surface_index(acc);
  using SurfIndTy = decltype(si);
  constexpr unsigned int RoundedWidth =
      Width < 4 ? 4 : detail::getNextPowerOf2<Width>();
  constexpr int BlockWidth = sizeof(T) * N;
  constexpr int Mod = 0;

  if constexpr (Width < RoundedWidth) {
    constexpr unsigned int n1 = RoundedWidth / sizeof(T);
    simd<T, m * n1> temp =
        __esimd_media_ld<T, m, n1, Mod, SurfIndTy, (int)plane, BlockWidth>(
            si, x, y);
    return temp.template select<m, 1, N, 1>(0, 0);
  } else {
    return __esimd_media_ld<T, m, N, Mod, SurfIndTy, (int)plane, BlockWidth>(
        si, x, y);
  }
}

/// Media block store.
///
/// @tparam T is the element data type.
/// @tparam m is the height of the 2D block.
/// @tparam N is the width of the 2D block.
/// @tparam is AccessorTy type of the SYCL accessor.
/// @tparam plane is planar surface index.
/// @param acc is the SYCL accessor.
/// @param x is X-coordinate of the left upper rectangle corner in BYTES.
/// @param y is Y-coordinate of the left upper rectangle corner in ROWS.
/// @param vals is the linearized 2D block data to be written to surface.
///
template <typename T, int m, int N, typename AccessorTy, unsigned plane = 0>
__ESIMD_API void media_block_store(AccessorTy acc, unsigned x, unsigned y,
                                   simd<T, m * N> vals) {
  constexpr unsigned Width = N * sizeof(T);
  static_assert(Width * m <= 256u,
                "data does not fit into a single dataport transaction");
  static_assert(Width <= 64u, "valid block width is in range [1, 64]");
  static_assert(m <= 64u, "valid block height is in range [1, 64]");
  static_assert(plane <= 3u, "valid plane index is in range [0, 3]");
  const auto si = __ESIMD_NS::get_surface_index(acc);
  using SurfIndTy = decltype(si);
  constexpr unsigned int RoundedWidth =
      Width < 4 ? 4 : detail::getNextPowerOf2<Width>();
  constexpr unsigned int n1 = RoundedWidth / sizeof(T);
  constexpr int BlockWidth = sizeof(T) * N;
  constexpr int Mod = 0;

  if constexpr (Width < RoundedWidth) {
    simd<T, m * n1> temp;
    auto temp_ref = temp.template bit_cast_view<T, m, n1>();
    auto vals_ref = vals.template bit_cast_view<T, m, N>();
    temp_ref.template select<m, 1, N, 1>() = vals_ref;
    __esimd_media_st<T, m, n1, Mod, SurfIndTy, plane, BlockWidth>(si, x, y,
                                                                  temp.data());
  } else {
    __esimd_media_st<T, m, N, Mod, SurfIndTy, plane, BlockWidth>(si, x, y,
                                                                 vals.data());
  }
}

/// Loads a contiguous block of SLM memory referenced by the given
/// local-accessor \p acc and \p byte_offset, then returns the loaded
/// data as a simd object.
/// The generated code depends on the combination {T, N, Flags}.
/// Providing flags specifying the alignment of 16-bytes or more produces more
/// efficient code. If the alignment is smaller than 16-bytes, then less
/// efficient gather is generated. If the loaded vector is too long
/// for 1 flat-load GPU instruction, then a series of flat-loads and/or gathers
/// may be generated.
/// @tparam T Element type.
/// @tparam N Number of elements to load.
/// @tparam AccessorTy Accessor type (auto-deduced).
/// @tparam Flags The alignment specifier type tag.
/// @param acc The local accessor.
/// @param byte_offset The offset to load from in bytes.
/// @param Flags Specifies the alignment.
/// @return A vector of loaded elements.
///
template <typename T, int N, typename AccessorTy,
          typename Flags = overaligned_tag<detail::OperandSize::OWORD>>
__ESIMD_API
    std::enable_if_t<detail::is_local_accessor_with_v<
                         AccessorTy, detail::accessor_mode_cap::can_read> &&
                         is_simd_flag_type_v<Flags>,
                     simd<T, N>>
    block_load(AccessorTy acc, uint32_t byte_offset, Flags flags) {
  return slm_block_load<T, N>(byte_offset + detail::localAccessorToOffset(acc),
                              flags);
}

/// Variant of block_store that uses local accessor as a parameter.
/// Stores elements of the vector \p vals to a contiguous block of SLM memory
/// represented by the given local accessor and the byte-offset \p offset.
/// The generated code depends on the combination {T, N, Flags}.
/// Providing flags specifying the alignment of 16-bytes or more produces more
/// efficient code. If the alignment is smaller than 16-bytes, then less
/// efficient scatter is generated. If the stored vector is too long
/// for 1 flat-store GPU instruction, then a series of flat-store and/or
/// scatters may be generated.
/// @tparam T Element type.
/// @tparam N Number of elements to store.
/// @tparam AccessorT Accessor type (auto-deduced).
/// @param acc The local accessor to store to.
/// @param offset The byte-offset to store at.
/// @param vals The vector to store.
/// @param Flags Specifies the alignment.
///
template <typename T, int N, typename AccessorT, typename Flags>
__ESIMD_API
    std::enable_if_t<detail::is_local_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_write> &&
                     is_simd_flag_type_v<Flags>>
    block_store(AccessorT acc, uint32_t offset, simd<T, N> vals, Flags flags) {
  slm_block_store<T, N>(offset + __ESIMD_DNS::localAccessorToOffset(acc), vals,
                        flags);
}

/// Variant of gather that uses local accessor as a parameter
///
/// Collects elements located at given offsets in an accessor and returns them
/// as a single \ref simd object. An element can be a 1, 2 or 4-byte value.
///
/// @tparam T Element type; can only be a 1,2,4-byte integer, \c sycl::half or
///   \c float.
/// @tparam N The number of vector elements. Can be \c 1, \c 8, \c 16 or \c 32.
/// @tparam AccessorTy The accessor type.
/// @param acc The accessor to gather from.
/// @param offsets Per-element offsets in bytes.
/// @param glob_offset Offset in bytes added to each individual element's offset
///   to compute actual memory access offset for that element.
/// @param mask Memory access mask. Elements with zero corresponding mask's
///   predicate are not accessed, their values in the resulting vector are
///   undefined.
///
template <typename T, int N, typename AccessorTy>
__ESIMD_API
    std::enable_if_t<detail::is_local_accessor_with_v<
                         AccessorTy, detail::accessor_mode_cap::can_read>,
                     simd<T, N>>
    gather(AccessorTy acc, simd<uint32_t, N> offsets, uint32_t glob_offset = 0,
           simd_mask<N> mask = 1) {
  return slm_gather<T, N>(
      offsets + glob_offset + __ESIMD_DNS::localAccessorToOffset(acc), mask);
}

/// Variant of scatter that uses local accessor as a parameter
///
/// Writes elements of a \ref simd object into an accessor at given offsets.
/// An element can be a 1, 2 or 4-byte value.
///
/// @tparam T Element type; can only be a 1,2,4-byte integer, \c sycl::half or
///   \c float.
/// @tparam N The number of vector elements. Can be \c 1, \c 8, \c 16 or \c 32.
/// @tparam AccessorTy The accessor type.
/// @param acc The accessor to scatter to.
/// @param offsets Per-element offsets in bytes.
/// @param vals Values to write.
/// @param glob_offset Offset in bytes added to each individual element's offset
///   to compute actual memory access offset for that element.
/// @param mask Memory access mask. Elements with zero corresponding mask's
///   predicate are not accessed.
///
///
template <typename T, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<detail::is_local_accessor_with_v<
    AccessorTy, detail::accessor_mode_cap::can_write>>
scatter(AccessorTy acc, simd<uint32_t, N> offsets, simd<T, N> vals,
        uint32_t glob_offset = 0, simd_mask<N> mask = 1) {
  slm_scatter<T, N>(offsets + glob_offset +
                        __ESIMD_DNS::localAccessorToOffset(acc),
                    vals, mask);
}

/// Variant of gather_rgba that uses local accessor as a parameter
///
/// Gather and transpose pixels from the given memory locations defined by the
/// base specified by \c acc, the global offset \c global_offset and a vector of
/// offsets \c offsets. Up to 4 32-bit data elements may be accessed at each
/// address depending on the channel mask \c RGBAMask. Each pixel's address must
/// be 4-byte aligned.
/// For usage examples, see \ref usm_gather_rgba above, the only difference
/// would be the usage of an accessor instead of a usm pointer.
///
/// @tparam RGBAMask A pixel's channel mask.
/// @tparam AccessorT The accessor type for the memory to be loaded/gathered.
/// The returned vector elements must match the accessor data type. The loaded
/// elements must be 4 bytes in size.
/// @tparam N Number of pixels to access (matches the size of the \c offsets
///   vector). Must be 8, 16 or 32.
/// @param acc The accessor representing memory address of the access.
/// @param offsets Byte offsets of the pixels relative to the base pointer.
/// @param global_offset Byte offset of the pixels relative to the base pointer.
/// @param mask Memory access mask. Pixels with zero corresponding mask's
///   predicate are not accessed. Their values in the resulting vector are
///   undefined.
/// @return Read data - up to N*4 values of type \c Tx.
///
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR,
          typename AccessorT, int N,
          typename T = typename AccessorT::value_type>
__ESIMD_API
    std::enable_if_t<detail::is_local_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_read>,
                     simd<T, N * get_num_channels_enabled(RGBAMask)>>
    gather_rgba(AccessorT acc, simd<uint32_t, N> offsets,
                uint32_t global_offset = 0, simd_mask<N> mask = 1) {
  return slm_gather_rgba<T, N, RGBAMask>(
      offsets + global_offset + __ESIMD_DNS::localAccessorToOffset(acc), mask);
}

/// Variant of scatter_rgba that uses local accessor as a parameter
/// Gather data from the memory addressed by accessor \c acc, offset common
/// for all loaded elements \c global_offset and per-element offsets \c offsets,
/// and return it as simd vector. See @ref usm_gather_rgba for information about
/// the operation semantics and parameter restrictions/interdependencies.
///
/// @tparam RGBAMask Pixel's channel mask.
/// @tparam AccessorT The accessor type for the memory to be stored/scattered.
/// The returned vector elements must match the accessor data type. The loaded
/// elements must be 4 bytes in size.
/// @tparam N The number of elements to access.
/// @param offsets Byte offsets of each element.
/// @param vals values to be written.
/// @param global_offset Byte offset of the pixels relative to the base pointer.
/// @param mask Operation mask. All-1 by default.
///
template <rgba_channel_mask RGBAMask = rgba_channel_mask::ABGR,
          typename AccessorT, int N,
          typename T = typename AccessorT::value_type>
__ESIMD_API std::enable_if_t<detail::is_local_accessor_with_v<
    AccessorT, detail::accessor_mode_cap::can_write>>
scatter_rgba(AccessorT acc, simd<uint32_t, N> offsets,
             simd<T, N * get_num_channels_enabled(RGBAMask)> vals,
             uint32_t global_offset = 0, simd_mask<N> mask = 1) {
  detail::validate_rgba_write_channel_mask<RGBAMask>();
  slm_scatter_rgba<T, N, RGBAMask>(offsets + global_offset +
                                       __ESIMD_DNS::localAccessorToOffset(acc),
                                   vals, mask);
}

/// @addtogroup sycl_esimd_raw_send
/// @{

/// Raw sends. "s" suffix designates "split" variant - i.e. two sources.
///  This is a low-level API not recommended for general usage.
///
/// @tparam exec_size is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam num_src0 is the number of GRFs for source-0.
/// @tparam num_src1 is the number of GRFs for source-1.
/// @tparam num_dst is the number of GRFs for destination.
/// @tparam eot is the flag that indicates whether this is an EOT message
/// (optional - default to off).
/// @tparam sendc is the flag that indicates whether sendc should be used
/// (optional - default to off).
/// @param msg_dst is the old value of the destination operand.
/// @param msg_src0 is the first source operand of send message.
/// @param msg_src1 is the second source operand of send message.
/// @param ex_desc is the extended message descriptor.
/// @param msg_desc is the message descriptor.
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
/// @return the vector value read from memory.
template <uint8_t exec_size, uint8_t sfid, uint8_t num_src0, uint8_t num_src1,
          uint8_t num_dst, raw_send_eot eot = raw_send_eot::not_eot,
          raw_send_sendc sendc = raw_send_sendc::not_sendc, typename T1, int n1,
          typename T2, int n2, typename T3, int n3>
__ESIMD_API __ESIMD_NS::simd<T1, n1>
raw_sends(__ESIMD_NS::simd<T1, n1> msg_dst, __ESIMD_NS::simd<T2, n2> msg_src0,
          __ESIMD_NS::simd<T3, n3> msg_src1, uint32_t ex_desc,
          uint32_t msg_desc, __ESIMD_NS::simd_mask<exec_size> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send rspVar");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msg_src0");
  constexpr unsigned _Width3 = n3 * sizeof(T3);
  static_assert(_Width3 % 32 == 0, "Invalid size for raw send msg_src1");

  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  using ElemT2 = __ESIMD_DNS::__raw_t<T2>;
  using ElemT3 = __ESIMD_DNS::__raw_t<T3>;

  constexpr uint8_t modifier =
      ((eot == raw_send_eot::eot) << 1) | (sendc == raw_send_sendc::sendc);

  return __esimd_raw_sends2<ElemT1, n1, ElemT2, n2, ElemT3, n3, exec_size>(
      modifier, exec_size, mask.data(), num_src0, num_src1, num_dst, sfid,
      ex_desc, msg_desc, msg_src0.data(), msg_src1.data(), msg_dst.data());
}

/// Raw send. This is a low-level API not recommended for general usage.
///
/// @tparam exec_size is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam num_src0 is the number of GRFs for source-0.
/// @tparam num_dst is the number of GRFs for destination.
/// @tparam eot is the flag that indicates whether this is an EOT message
/// (optional - default to off).
/// @tparam sendc is the flag that indicates whether sendc should be used
/// (optional - default to off).
/// @param msg_dst is the old value of the destination operand.
/// @param msg_src0 is the first source operand of send message.
/// @param ex_desc is the extended message descriptor.
/// @param msg_desc is the message descriptor.
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
/// @return the vector value read from memory
template <uint8_t exec_size, uint8_t sfid, uint8_t num_src0, uint8_t num_dst,
          raw_send_eot eot = raw_send_eot::not_eot,
          raw_send_sendc sendc = raw_send_sendc::not_sendc, typename T1, int n1,
          typename T2, int n2>
__ESIMD_API __ESIMD_NS::simd<T1, n1>
raw_send(__ESIMD_NS::simd<T1, n1> msg_dst, __ESIMD_NS::simd<T2, n2> msg_src0,
         uint32_t ex_desc, uint32_t msg_desc,
         __ESIMD_NS::simd_mask<exec_size> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send rspVar");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msg_src0");

  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  using ElemT2 = __ESIMD_DNS::__raw_t<T2>;

  constexpr uint8_t modifier =
      ((eot == raw_send_eot::eot) << 1) | (sendc == raw_send_sendc::sendc);
  return __esimd_raw_send2<ElemT1, n1, ElemT2, n2, exec_size>(
      modifier, exec_size, mask.data(), num_src0, num_dst, sfid, ex_desc,
      msg_desc, msg_src0.data(), msg_dst.data());
}

/// Raw sends. "s" suffix designates "split" variant - i.e. two sources.
///  This is a low-level API not recommended for general usage.
///
/// @tparam exec_size is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam num_src0 is the number of GRFs for source-0.
/// @tparam num_src1 is the number of GRFs for source-1.
/// @tparam eot is the flag that indicates whether this is an EOT message
/// (optional - default to off).
/// @tparam sendc is the flag that indicates whether sendc should be used
/// (optional - default to off).
/// @param msg_src0 is the first source operand of send message.
/// @param msg_src1 is the second source operand of send message.
/// @param ex_desc is the extended message descriptor.
/// @param msg_desc is the message descriptor.
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
template <uint8_t exec_size, uint8_t sfid, uint8_t num_src0, uint8_t num_src1,
          raw_send_eot eot = raw_send_eot::not_eot,
          raw_send_sendc sendc = raw_send_sendc::not_sendc, typename T1, int n1,
          typename T2, int n2>
__ESIMD_API void raw_sends(__ESIMD_NS::simd<T1, n1> msg_src0,
                           __ESIMD_NS::simd<T2, n2> msg_src1, uint32_t ex_desc,
                           uint32_t msg_desc,
                           __ESIMD_NS::simd_mask<exec_size> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send msg_src0");
  constexpr unsigned _Width2 = n2 * sizeof(T2);
  static_assert(_Width2 % 32 == 0, "Invalid size for raw send msg_src1");

  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  using ElemT2 = __ESIMD_DNS::__raw_t<T2>;

  constexpr uint8_t modifier =
      ((eot == raw_send_eot::eot) << 1) | (sendc == raw_send_sendc::sendc);
  __esimd_raw_sends2_noresult<ElemT1, n1, ElemT2, n2, exec_size>(
      modifier, exec_size, mask.data(), num_src0, num_src1, sfid, ex_desc,
      msg_desc, msg_src0.data(), msg_src1.data());
}

/// Raw send. Generates a \c send or \c sendc instruction for the message
/// gateway. This is a low-level API not recommended for general usage.
///
/// @tparam exec_size is the execution size.
/// @tparam sfid is the shared function ID.
/// @tparam num_src0 is the number of GRFs for source-0.
/// @tparam eot is the flag that indicates whether this is an EOT message
/// (optional - default to off).
/// @tparam sendc is the flag that indicates whether sendc should be used
/// (optional - default to off).
/// @param msg_src0 is the first source operand of send message.
/// @param ex_desc is the extended message descriptor.
/// @param msg_desc is the message descriptor.
/// @param mask is the predicate to specify enabled channels (optional - default
/// to on).
template <uint8_t exec_size, uint8_t sfid, uint8_t num_src0,
          raw_send_eot eot = raw_send_eot::not_eot,
          raw_send_sendc sendc = raw_send_sendc::not_sendc, typename T1, int n1>
__ESIMD_API void raw_send(__ESIMD_NS::simd<T1, n1> msg_src0, uint32_t ex_desc,
                          uint32_t msg_desc,
                          __ESIMD_NS::simd_mask<exec_size> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send msg_src0");
  using ElemT1 = __ESIMD_DNS::__raw_t<T1>;
  constexpr uint8_t modifier =
      ((eot == raw_send_eot::eot) << 1) | (sendc == raw_send_sendc::sendc);
  __esimd_raw_send2_noresult<ElemT1, n1, exec_size>(
      modifier, exec_size, mask.data(), num_src0, sfid, ex_desc, msg_desc,
      msg_src0.data());
}

/// @} sycl_esimd_raw_send

/// @} sycl_esimd_memory

/// @cond EXCLUDE

namespace detail {
// -- Outlined implementations of simd_obj_impl class memory access APIs.

template <typename T, int N, class T1, class SFINAE>
template <typename Flags, int ChunkSize, typename>
void simd_obj_impl<T, N, T1, SFINAE>::copy_from(
    const simd_obj_impl<T, N, T1, SFINAE>::element_type *Addr,
    Flags) SYCL_ESIMD_FUNCTION {
  using UT = simd_obj_impl<T, N, T1, SFINAE>::element_type;
  constexpr unsigned Size = sizeof(T) * N;
  constexpr unsigned Align = Flags::template alignment<T1>;

  constexpr unsigned BlockSize = OperandSize::OWORD * 8;
  constexpr unsigned NumBlocks = Size / BlockSize;
  constexpr unsigned RemSize = Size % BlockSize;

  if constexpr (Align >= OperandSize::DWORD && Size % OperandSize::OWORD == 0 &&
                detail::isPowerOf2(RemSize / OperandSize::OWORD)) {
    if constexpr (NumBlocks > 0) {
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      ForHelper<NumBlocks>::unroll([BlockN, Addr, this](unsigned Block) {
        select<BlockN, 1>(Block * BlockN) =
            block_load<UT, BlockN, Flags>(Addr + (Block * BlockN), Flags{});
      });
    }
    if constexpr (RemSize > 0) {
      constexpr unsigned RemN = RemSize / sizeof(T);
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      select<RemN, 1>(NumBlocks * BlockN) =
          block_load<UT, RemN, Flags>(Addr + (NumBlocks * BlockN), Flags{});
    }
  } else if constexpr (sizeof(T) == 8) {
    simd<int32_t, N * 2> BC(reinterpret_cast<const int32_t *>(Addr), Flags{});
    bit_cast_view<int32_t>() = BC;
  } else {
    constexpr unsigned NumChunks = N / ChunkSize;
    if constexpr (NumChunks > 0) {
      simd<uint32_t, ChunkSize> Offsets(0u, sizeof(T));
      ForHelper<NumChunks>::unroll([Addr, &Offsets, this](unsigned Block) {
        select<ChunkSize, 1>(Block * ChunkSize) =
            gather<UT, ChunkSize>(Addr + (Block * ChunkSize), Offsets);
      });
    }
    constexpr unsigned RemN = N % ChunkSize;
    if constexpr (RemN > 0) {
      if constexpr (RemN == 1) {
        select<1, 1>(NumChunks * ChunkSize) = Addr[NumChunks * ChunkSize];
      } else if constexpr (RemN == 8 || RemN == 16) {
        simd<uint32_t, RemN> Offsets(0u, sizeof(T));
        select<RemN, 1>(NumChunks * ChunkSize) =
            gather<UT, RemN>(Addr + (NumChunks * ChunkSize), Offsets);
      } else {
        constexpr int N1 = RemN < 8 ? 8 : RemN < 16 ? 16 : 32;
        simd_mask_type<N1> Pred(0);
        Pred.template select<RemN, 1>() = 1;
        simd<uint32_t, N1> Offsets(0u, sizeof(T));
        simd<UT, N1> Vals =
            gather<UT, N1>(Addr + (NumChunks * ChunkSize), Offsets, Pred);
        select<RemN, 1>(NumChunks * ChunkSize) =
            Vals.template select<RemN, 1>();
      }
    }
  }
}

template <typename T, int N, class T1, class SFINAE>
template <int ChunkSize, typename Flags, typename AccessorT, typename TOffset>
ESIMD_INLINE void simd_obj_impl<T, N, T1, SFINAE>::copy_to_impl(
    AccessorT acc, TOffset offset) const SYCL_ESIMD_FUNCTION {
  using UT = simd_obj_impl<T, N, T1, SFINAE>::element_type;
  constexpr unsigned Size = sizeof(T) * N;
  constexpr unsigned Align = Flags::template alignment<T1>;

  constexpr unsigned BlockSize = OperandSize::OWORD * 8;
  constexpr unsigned NumBlocks = Size / BlockSize;
  constexpr unsigned RemSize = Size % BlockSize;

  simd<UT, N> Tmp{data()};
  if constexpr (Align >= OperandSize::OWORD && Size % OperandSize::OWORD == 0 &&
                detail::isPowerOf2(RemSize / OperandSize::OWORD)) {
    if constexpr (NumBlocks > 0) {
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      ForHelper<NumBlocks>::unroll([BlockN, acc, offset, &Tmp](unsigned Block) {
        block_store<UT, BlockN, AccessorT>(
            acc, offset + (Block * BlockSize),
            Tmp.template select<BlockN, 1>(Block * BlockN));
      });
    }
    if constexpr (RemSize > 0) {
      constexpr unsigned RemN = RemSize / sizeof(T);
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      block_store<UT, RemN, AccessorT>(
          acc, offset + (NumBlocks * BlockSize),
          Tmp.template select<RemN, 1>(NumBlocks * BlockN));
    }
  } else if constexpr (sizeof(T) == 8) {
    simd<int32_t, N * 2> BC = Tmp.template bit_cast_view<int32_t>();
    BC.copy_to(acc, offset, Flags{});
  } else {
    constexpr unsigned NumChunks = N / ChunkSize;
    if constexpr (NumChunks > 0) {
      simd<TOffset, ChunkSize> Offsets(0u, sizeof(T));
      ForHelper<NumChunks>::unroll([acc, offset, &Offsets,
                                    &Tmp](unsigned Block) {
        scatter<UT, ChunkSize, AccessorT>(
            acc, Offsets, Tmp.template select<ChunkSize, 1>(Block * ChunkSize),
            offset + (Block * ChunkSize * sizeof(T)));
      });
    }
    constexpr unsigned RemN = N % ChunkSize;
    if constexpr (RemN > 0) {
      if constexpr (RemN == 1 || RemN == 8 || RemN == 16) {
        simd<TOffset, RemN> Offsets(0u, sizeof(T));
        scatter<UT, RemN, AccessorT>(
            acc, Offsets, Tmp.template select<RemN, 1>(NumChunks * ChunkSize),
            offset + (NumChunks * ChunkSize * sizeof(T)));
      } else {
        constexpr int N1 = RemN < 8 ? 8 : RemN < 16 ? 16 : 32;
        simd_mask_type<N1> Pred(0);
        Pred.template select<RemN, 1>() = 1;
        simd<UT, N1> Vals;
        Vals.template select<RemN, 1>() =
            Tmp.template select<RemN, 1>(NumChunks * ChunkSize);
        simd<TOffset, N1> Offsets(0u, sizeof(T));
        scatter<UT, N1, AccessorT>(acc, Offsets, Vals,
                                   offset + (NumChunks * ChunkSize * sizeof(T)),
                                   Pred);
      }
    }
  }
}

template <typename T, int N, class T1, class SFINAE>
template <int ChunkSize, typename Flags, typename AccessorT, typename TOffset>
ESIMD_INLINE void simd_obj_impl<T, N, T1, SFINAE>::copy_from_impl(
    AccessorT acc, TOffset offset) SYCL_ESIMD_FUNCTION {
  using UT = simd_obj_impl<T, N, T1, SFINAE>::element_type;
  static_assert(sizeof(UT) == sizeof(T));
  constexpr unsigned Size = sizeof(T) * N;
  constexpr unsigned Align = Flags::template alignment<T1>;

  constexpr unsigned BlockSize = OperandSize::OWORD * 8;
  constexpr unsigned NumBlocks = Size / BlockSize;
  constexpr unsigned RemSize = Size % BlockSize;

  if constexpr (Align >= OperandSize::DWORD && Size % OperandSize::OWORD == 0 &&
                detail::isPowerOf2(RemSize / OperandSize::OWORD)) {
    if constexpr (NumBlocks > 0) {
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      ForHelper<NumBlocks>::unroll([BlockN, acc, offset, this](unsigned Block) {
        select<BlockN, 1>(Block * BlockN) =
            block_load<UT, BlockN, AccessorT, Flags>(
                acc, offset + (Block * BlockSize), Flags{});
      });
    }
    if constexpr (RemSize > 0) {
      constexpr unsigned RemN = RemSize / sizeof(T);
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      select<RemN, 1>(NumBlocks * BlockN) =
          block_load<UT, RemN, AccessorT, Flags>(
              acc, offset + (NumBlocks * BlockSize), Flags{});
    }
  } else if constexpr (sizeof(T) == 8) {
    simd<int32_t, N * 2> BC(acc, offset, Flags{});
    bit_cast_view<int32_t>() = BC;
  } else {
    constexpr unsigned NumChunks = N / ChunkSize;
    if constexpr (NumChunks > 0) {
      simd<TOffset, ChunkSize> Offsets(0u, sizeof(T));
      ForHelper<NumChunks>::unroll(
          [acc, offset, &Offsets, this](unsigned Block) {
            select<ChunkSize, 1>(Block * ChunkSize) =
                gather<UT, ChunkSize, AccessorT>(
                    acc, Offsets, offset + (Block * ChunkSize * sizeof(T)));
          });
    }
    constexpr unsigned RemN = N % ChunkSize;
    if constexpr (RemN > 0) {
      if constexpr (RemN == 1 || RemN == 8 || RemN == 16) {
        simd<TOffset, RemN> Offsets(0u, sizeof(T));
        select<RemN, 1>(NumChunks * ChunkSize) = gather<UT, RemN, AccessorT>(
            acc, Offsets, offset + (NumChunks * ChunkSize * sizeof(T)));
      } else {
        constexpr int N1 = RemN < 8 ? 8 : RemN < 16 ? 16 : 32;
        simd_mask_type<N1> Pred(0);
        Pred.template select<RemN, 1>() = 1;
        simd<TOffset, N1> Offsets(0u, sizeof(T));
        simd<UT, N1> Vals = gather<UT, N1>(
            acc, Offsets, offset + (NumChunks * ChunkSize * sizeof(T)), Pred);
        select<RemN, 1>(NumChunks * ChunkSize) =
            Vals.template select<RemN, 1>();
      }
    }
  }
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, typename Flags, int ChunkSize, typename>
ESIMD_INLINE EnableIfAccessor<AccessorT, accessor_mode_cap::can_read, void>
simd_obj_impl<T, N, T1, SFINAE>::copy_from(AccessorT acc,
                                           detail::DeviceAccessorOffsetT offset,
                                           Flags) SYCL_ESIMD_FUNCTION {

  copy_from_impl<ChunkSize, Flags>(acc, offset);
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, typename Flags, int ChunkSize, typename>
ESIMD_INLINE std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT, accessor_mode_cap::can_read>,
    void>
simd_obj_impl<T, N, T1, SFINAE>::copy_from(AccessorT acc, uint32_t offset,
                                           Flags) SYCL_ESIMD_FUNCTION {

  copy_from_impl<ChunkSize, Flags>(acc, offset);
}

template <typename T, int N, class T1, class SFINAE>
template <typename Flags, int ChunkSize, typename>
void simd_obj_impl<T, N, T1, SFINAE>::copy_to(
    simd_obj_impl<T, N, T1, SFINAE>::element_type *Addr,
    Flags) const SYCL_ESIMD_FUNCTION {
  using UT = simd_obj_impl<T, N, T1, SFINAE>::element_type;
  constexpr unsigned Size = sizeof(T) * N;
  constexpr unsigned Align = Flags::template alignment<T1>;

  constexpr unsigned BlockSize = OperandSize::OWORD * 8;
  constexpr unsigned NumBlocks = Size / BlockSize;
  constexpr unsigned RemSize = Size % BlockSize;

  simd<UT, N> Tmp{data()};
  if constexpr (Align >= OperandSize::OWORD && Size % OperandSize::OWORD == 0 &&
                detail::isPowerOf2(RemSize / OperandSize::OWORD)) {
    if constexpr (NumBlocks > 0) {
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      ForHelper<NumBlocks>::unroll([BlockN, Addr, &Tmp](unsigned Block) {
        block_store<UT, BlockN>(Addr + (Block * BlockN),
                                Tmp.template select<BlockN, 1>(Block * BlockN));
      });
    }
    if constexpr (RemSize > 0) {
      constexpr unsigned RemN = RemSize / sizeof(T);
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      block_store<UT, RemN>(Addr + (NumBlocks * BlockN),
                            Tmp.template select<RemN, 1>(NumBlocks * BlockN));
    }
  } else if constexpr (sizeof(T) == 8) {
    simd<int32_t, N * 2> BC = Tmp.template bit_cast_view<int32_t>();
    BC.copy_to(reinterpret_cast<int32_t *>(Addr), Flags{});
  } else {
    constexpr unsigned NumChunks = N / ChunkSize;
    if constexpr (NumChunks > 0) {
      simd<uint32_t, ChunkSize> Offsets(0u, sizeof(T));
      ForHelper<NumChunks>::unroll([Addr, &Offsets, &Tmp](unsigned Block) {
        scatter<UT, ChunkSize>(
            Addr + (Block * ChunkSize), Offsets,
            Tmp.template select<ChunkSize, 1>(Block * ChunkSize));
      });
    }
    constexpr unsigned RemN = N % ChunkSize;
    if constexpr (RemN > 0) {
      if constexpr (RemN == 1) {
        Addr[NumChunks * ChunkSize] = Tmp[NumChunks * ChunkSize];
      } else if constexpr (RemN == 8 || RemN == 16) {
        // TODO: GPU runtime may handle scatter of 16 byte elements
        // incorrectly. The code below is a workaround which must be deleted
        // once GPU runtime is fixed.
        if constexpr (sizeof(T) == 1 && RemN == 16) {
          if constexpr (Align % OperandSize::DWORD > 0) {
            ForHelper<RemN>::unroll([Addr, &Tmp](unsigned Index) {
              Addr[Index + NumChunks * ChunkSize] =
                  Tmp[Index + NumChunks * ChunkSize];
            });
          } else {
            simd_mask_type<8> Pred(0);
            simd<int32_t, 8> Vals;
            Pred.template select<4, 1>() = 1;
            Vals.template select<4, 1>() =
                Tmp.template bit_cast_view<int32_t>().template select<4, 1>(
                    NumChunks * ChunkSize);

            simd<uint32_t, 8> Offsets(0u, sizeof(int32_t));
            scatter<int32_t, 8>(
                reinterpret_cast<int32_t *>(Addr + (NumChunks * ChunkSize)),
                Offsets, Vals, Pred);
          }
        } else {
          simd<uint32_t, RemN> Offsets(0u, sizeof(T));
          scatter<UT, RemN>(
              Addr + (NumChunks * ChunkSize), Offsets,
              Tmp.template select<RemN, 1>(NumChunks * ChunkSize));
        }
      } else {
        constexpr int N1 = RemN < 8 ? 8 : RemN < 16 ? 16 : 32;
        simd_mask_type<N1> Pred(0);
        Pred.template select<RemN, 1>() = 1;
        simd<UT, N1> Vals;
        Vals.template select<RemN, 1>() =
            Tmp.template select<RemN, 1>(NumChunks * ChunkSize);
        simd<uint32_t, N1> Offsets(0u, sizeof(T));
        scatter<UT, N1>(Addr + (NumChunks * ChunkSize), Offsets, Vals, Pred);
      }
    }
  }
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, typename Flags, int ChunkSize, typename>
ESIMD_INLINE EnableIfAccessor<AccessorT, accessor_mode_cap::can_write, void>
simd_obj_impl<T, N, T1, SFINAE>::copy_to(AccessorT acc,
                                         detail::DeviceAccessorOffsetT offset,
                                         Flags) const SYCL_ESIMD_FUNCTION {
  copy_to_impl<ChunkSize, Flags>(acc, offset);
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, typename Flags, int ChunkSize, typename>
ESIMD_INLINE std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT, accessor_mode_cap::can_write>,
    void>
simd_obj_impl<T, N, T1, SFINAE>::copy_to(AccessorT acc, uint32_t offset,
                                         Flags) const SYCL_ESIMD_FUNCTION {
  copy_to_impl<ChunkSize, Flags>(acc, offset);
}

} // namespace detail
/// @endcond EXCLUDE

} // namespace ext::intel::esimd
} // namespace _V1
} // namespace sycl
