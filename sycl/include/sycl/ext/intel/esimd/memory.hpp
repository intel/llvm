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

/// Extracts a cache hint with the given 'Level' to pass it to
/// ESIMD/GENX intrinsics. If `PropertyListT` does not have the requested
/// cache-hint, then 'cache_hint::none' is returned.
template <typename PropertyListT, cache_level Level>
constexpr cache_hint getCacheHintForIntrin() {
  static_assert(Level == cache_level::L1 || Level == cache_level::L2,
                "ESIMD/GENX intrinsics accept only L1/L2 cache hints");
  if constexpr (Level == cache_level::L1) {
    return getPropertyValue<PropertyListT, cache_hint_L1_key>(cache_hint::none);
  } else {
    return getPropertyValue<PropertyListT, cache_hint_L2_key>(cache_hint::none);
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
/// @tparam PropertyListT is the properties with optional cache hints.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
/// @param pass_thru contains the vector which elements are copied
/// to the returned result when the corresponding element of \p pred is 0.
/// @return is a vector of type T and size N * NElts
///
template <typename T, int NElts, lsc_data_size DS, typename PropertyListT,
          int N, typename OffsetT>
__ESIMD_API simd<T, N * NElts> gather_impl(const T *p, simd<OffsetT, N> offsets,
                                           simd_mask<N> pred,
                                           simd<T, N * NElts> pass_thru) {
  static_assert(std::is_integral_v<OffsetT>, "Unsupported offset type");
  check_lsc_vector_size<NElts>();
  check_lsc_data_size<T, DS>();
  check_cache_hints<cache_action::load, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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

/// USM pointer scatter.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_store.ugm
///
/// Scatters elements to specific address.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam PropertyListT is the properties with optional cache hints.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets in bytes.
/// @param vals is values to store.
/// @param pred is predicates.
///
template <typename T, int NElts, lsc_data_size DS, typename PropertyListT,
          int N, typename Toffset>
__ESIMD_API void scatter_impl(T *p, simd<Toffset, N> offsets,
                              simd<T, N * NElts> vals, simd_mask<N> pred) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  check_lsc_vector_size<NElts>();
  check_lsc_data_size<T, DS>();
  check_cache_hints<cache_action::store, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size VS = to_lsc_vector_size<NElts>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  simd<MsgT, N * NElts> Tmp = lsc_format_input<MsgT, T>(vals);
  __esimd_lsc_store_stateless<MsgT, L1H, L2H, AddressScale, ImmOffset, EDS, VS,
                              Transposed, N>(pred.data(), addrs.data(),
                                             Tmp.data());
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
/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///            typename PropertyListT = empty_props_t>
/// simd <T, N> gather(const T *p,
///             OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, simd<T, N> pass_thru,
///             PropertyListT props = {});                         // (usm-ga-7)
/// simd <T, N> gather(const T *p,
///             OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-ga-8)
/// simd <T, N> gather(const T *p,
///             OffsetSimdViewT byte_offsets,
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
template <
    typename T, int N, int VS, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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

  // Use LSC lowering if cache-hints are used or VS > 1. Also, if
  // llvm.masked.gather is not available, then LSC is the only lowering option.
  if constexpr (detail::has_cache_hints<PropertyListT>() || VS > 1 ||
                !detail::isMaskedGatherScatterLLVMAvailable()) {
    static_assert(VS == 1 || sizeof(T) >= 4,
                  "VS > 1 is supprted only for 4- and 8-byte elements");
    return detail::gather_impl<T, VS, detail::lsc_data_size::default_size,
                               PropertyListT>(p, byte_offsets, mask, pass_thru);
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
template <
    typename T, int N, int VS, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
       PropertyListT props = {}) {
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(Alignment >= sizeof(T),
                "gather() requires at least element-size alignment");

  if constexpr (detail::has_cache_hints<PropertyListT>() || VS > 1 ||
                detail::isMaskedGatherScatterLLVMAvailable() ||
                !detail::isPowerOf2(N, 32)) {
    simd<T, N> PassThru; // it is intentionally undefined
    return gather<T, N, VS>(p, byte_offsets, mask, PassThru, props);
  } else {
    simd<uintptr_t, N> Addrs = reinterpret_cast<uintptr_t>(p);
    Addrs += convert<uintptr_t>(byte_offsets);
    using MsgT = detail::__raw_t<T>;
    if constexpr (sizeof(T) == 1) {
      auto Ret = __esimd_svm_gather<MsgT, N, detail::ElemsPerAddrEncoding<4>(),
                                    detail::ElemsPerAddrEncoding<1>()>(
          Addrs.data(), mask.data());
      detail::check_rdregion_params<N * 4, N, /*VS*/ 0, N, 4>();
      return __esimd_rdregion<MsgT, N * 4, N, /*VS*/ 0, N, 4>(Ret, 0);
    } else if constexpr (sizeof(T) == 2) {
      auto Ret = __esimd_svm_gather<MsgT, N, detail::ElemsPerAddrEncoding<2>(),
                                    detail::ElemsPerAddrEncoding<2>()>(
          Addrs.data(), mask.data());
      detail::check_rdregion_params<N * 2, N, /*VS*/ 0, N, 2>();
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
template <
    typename T, int N, int VS, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
gather(const T *p, simd<OffsetT, N> byte_offsets, PropertyListT props = {}) {
  constexpr int VS = 1;
  return gather<T, N, VS>(p, byte_offsets, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///            typename PropertyListT = empty_props_t>
/// simd <T, N> gather(const T *p,
///             OffsetSimdViewT byte_offsets,
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
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
gather(const T *p, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       simd<T, N> pass_thru, PropertyListT props = {}) {
  return gather<T, N, VS>(p, byte_offsets.read(), mask, pass_thru, props);
}

/// template <int VS = 1, typename OffsetT, typename T, typename
/// PassThruSimdViewT, int N = PassThruSimdViewT::getSizeX() *
/// PassThruSimdViewT::getSizeY(),
///            typename PropertyListT = empty_props_t>
/// simd <T, N> gather(const T *p,
///             simd<OffsetT, N / VS> byte_offsets,
///             simd_mask<N / VS> mask, PassThruSimdViewT pass_thru,
///             PropertyListT props = {});
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters. Loads ("gathers") elements of the type 'T'
/// from memory locations addressed by the base pointer \p p and byte offsets \p
/// byte_offsets, and returns the loaded elements. Access to any element's
/// memory location can be disabled via the input vector of predicates \p mask.
/// If mask[i] is unset, then the load from (p + byte_offsets[i]) is skipped and
/// the corresponding i-th element from \p pass_thru operand is returned.
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
template <
    int VS = 1, typename OffsetT, typename T, typename PassThruSimdViewT,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<PassThruSimdViewT>,
    simd<T, N>>
gather(const T *p, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
       PassThruSimdViewT pass_thru, PropertyListT props = {}) {
  return gather<T, N, VS>(p, byte_offsets, mask, pass_thru.read(), props);
}

/// template <int VS = 1, typename OffsetSimdViewT, typename T, typename
/// PassThruSimdViewT, int N = PassThruSimdViewT::getSizeX() *
/// PassThruSimdViewT::getSizeY(),
///            typename PropertyListT = empty_props_t>
/// simd <T, N> gather(const T *p,
///             OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, PassThruSimdViewT pass_thru,
///             PropertyListT props = {});
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters. Loads ("gathers") elements of the type 'T'
/// from memory locations addressed by the base pointer \p p and byte offsets \p
/// byte_offsets, and returns the loaded elements. Access to any element's
/// memory location can be disabled via the input vector of predicates \p mask.
/// If mask[i] is unset, then the load from (p + byte_offsets[i]) is skipped and
/// the corresponding i-th element from \p pass_thru operand is returned.
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
template <
    int VS = 1, typename OffsetSimdViewT, typename T,
    typename PassThruSimdViewT,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<PassThruSimdViewT>,
    simd<T, N>>
gather(const T *p, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       PassThruSimdViewT pass_thru, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of pass_thru parameter must correspond to the size of "
                "byte_offsets parameter.");
  return gather<T, N, VS>(p, byte_offsets.read(), mask, pass_thru.read(),
                          props);
}

/// template <int VS = 1, typename OffsetSimdViewT, typename T, int N,
///            typename PropertyListT = empty_props_t>
/// simd <T, N> gather(const T *p,
///             OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, simd<T, N> pass_thru,
///             PropertyListT props = {});
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters. Loads ("gathers") elements of the type 'T'
/// from memory locations addressed by the base pointer \p p and byte offsets \p
/// byte_offsets, and returns the loaded elements. Access to any element's
/// memory location can be disabled via the input vector of predicates \p mask.
/// If mask[i] is unset, then the load from (p + byte_offsets[i]) is skipped and
/// the corresponding i-th element from \p pass_thru operand is returned.
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
template <
    int VS, typename OffsetSimdViewT, typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
gather(const T *p, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       simd<T, N> pass_thru, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of pass_thru parameter must correspond to the size of "
                "byte_offsets parameter.");
  return gather<T, N, VS>(p, byte_offsets.read(), mask, pass_thru, props);
}

/// simd <T, N> gather(const T *p,
///             OffsetSimdViewT byte_offsets,
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
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
gather(const T *p, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       PropertyListT props = {}) {
  return gather<T, N, VS>(p, byte_offsets.read(), mask, props);
}

/// simd <T, N> gather(const T *p,
///             OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {});
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters. Loads ("gathers") elements of the type 'T'
/// from memory locations addressed by the base pointer \p p and byte offsets \p
/// byte_offsets, and returns the loaded elements. Access to any element's
/// memory location can be disabled via the input vector of predicates \p mask.
/// If mask[i] is unset, then the load from (p + byte_offsets[i]) is skipped and
/// the corresponding i-th element of the returned vector is undefined.
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
template <
    int VS = 1, typename OffsetSimdViewT, typename T,
    int N = OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY() * VS,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
gather(const T *p, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       PropertyListT props = {}) {
  return gather<T, N, VS>(p, byte_offsets.read(), mask, props);
}

/// simd <T, N> gather(const T *p,
///             OffsetSimdViewT byte_offsets,
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
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
gather(const T *p, OffsetSimdViewT byte_offsets, PropertyListT props = {}) {
  return gather<T, N, VS>(p, byte_offsets.read(), props);
}
/// simd <T, N> gather(const T *p,
///             OffsetSimdViewT byte_offsets,
///             PropertyListT props = {});
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters.  Loads ("gathers") elements of the type 'T'
/// from memory locations addressed by the base pointer \p p and byte offsets \p
/// byte_offsets, and returns the loaded elements.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <
    int VS = 1, typename OffsetSimdViewT, typename T,
    int N = OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY() * VS,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
gather(const T *p, OffsetSimdViewT byte_offsets, PropertyListT props = {}) {
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

/// template <typename T, int N, int VS = 1, typename OffsetT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
/// 	simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-sc-1)

/// template <typename T, int N, int VS = 1, typename OffsetT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
/// 	PropertyListT props = {});                         // (usm-sc-2)

/// The next two functions are similar to usm-sc-{1,2} with the 'byte_offsets'
/// parameter represented as 'simd_view'.

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-sc-3)

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals,
///      PropertyListT props = {});                         // (usm-sc-4)

/// template <typename T, int N, int VS = 1, typename OffsetT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
/// 	simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-sc-1)
///
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector. Access to
/// any element's memory location can be disabled via the input mask.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    typename T, int N, int VS = 1, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  static_assert(std::is_integral_v<OffsetT>, "Unsupported offset type");
  static_assert(N / VS >= 1 && N % VS == 0, "N must be divisible by VS");

  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(Alignment >= sizeof(T),
                "scatter() requires at least element-size alignment");

  // Use LSC lowering if cache-hints are used or VS > 1.
  if constexpr (detail::has_cache_hints<PropertyListT>() || VS > 1 ||
                (!__ESIMD_DNS::isPowerOf2(N, 32) &&
                 !detail::isMaskedGatherScatterLLVMAvailable())) {
    static_assert(VS == 1 || sizeof(T) >= 4,
                  "VS > 1 is supprted only for 4- and 8-byte elements");
    return detail::scatter_impl<T, VS, detail::lsc_data_size::default_size,
                                PropertyListT>(p, byte_offsets, vals, mask);
  } else if constexpr (detail::isMaskedGatherScatterLLVMAvailable()) {
    simd<uint64_t, N> Addrs(reinterpret_cast<uint64_t>(p));
    Addrs = Addrs + convert<uint64_t>(byte_offsets);
    using MsgT = detail::__raw_t<T>;
    __esimd_scatter_st<MsgT, N, Alignment>(
        sycl::bit_cast<__ESIMD_DNS::vector_type_t<MsgT, N>>(vals.data()),
        Addrs.data(), mask.data());
  } else {
    using Tx = detail::__raw_t<T>;
    simd<uint64_t, N> byte_offsets_i = convert<uint64_t>(byte_offsets);
    simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
    addrs = addrs + byte_offsets_i;
    if constexpr (sizeof(T) == 1) {
      detail::check_wrregion_params<N * 4, N, /*VS*/ 0, N, 4>();
      simd<T, N * 4> D; // Intentionally undefined.
      D = __esimd_wrregion<Tx, N * 4, N, /*VS*/ 0, N, 4>(D.data(), vals.data(),
                                                         0);
      __esimd_svm_scatter<Tx, N, detail::ElemsPerAddrEncoding<4>(),
                          detail::ElemsPerAddrEncoding<1>()>(
          addrs.data(), D.data(), mask.data());
    } else if constexpr (sizeof(T) == 2) {
      detail::check_wrregion_params<N * 2, N, /*VS*/ 0, N, 2>();
      simd<Tx, N * 2> D; // Intentionally undefined.
      D = __esimd_wrregion<Tx, N * 2, N, /*VS*/ 0, N, 2>(D.data(), vals.data(),
                                                         0);
      __esimd_svm_scatter<Tx, N, detail::ElemsPerAddrEncoding<2>(),
                          detail::ElemsPerAddrEncoding<2>()>(
          addrs.data(), D.data(), mask.data());
    } else
      __esimd_svm_scatter<Tx, N, detail::ElemsPerAddrEncoding<1>(),
                          detail::ElemsPerAddrEncoding<1>()>(
          addrs.data(), vals.data(), mask.data());
  }
}

/// template <int VS = 1, typename OffsetT, typename ValuesSimdViewT, typename
/// T, int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename PropertyListT = empty_properties_t>
/// void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, ValuesSimdViewT vals,
/// simd_mask<N / VS> mask, PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters.
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector. Access to
/// any element's memory location can be disabled via the input mask.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename OffsetT, typename ValuesSimdViewT, typename T,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(T *p, simd<OffsetT, N / VS> byte_offsets, ValuesSimdViewT vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  scatter<T, N, VS>(p, byte_offsets, vals.read(), mask, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
/// 	PropertyListT props = {});                               // (usm-sc-2)
///
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    typename T, int N, int VS = 1, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(T *p, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
        PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  scatter<T, N, VS>(p, byte_offsets, vals, Mask, props);
}

/// template <int VS = 1, typename OffsetSimdViewT, typename ValuesSimdViewT,
/// typename T, int N = ValuesSimdViewT::getSizeX() *
/// ValuesSimdViewT::getSizeY(), typename PropertyListT = empty_properties_t>
/// void scatter(T *p, OffsetSimdViewT byte_offsets, ValuesSimdViewT vals,
/// 	simd_mask<N / VS> mask, PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters.
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename OffsetSimdViewT, typename ValuesSimdViewT, typename T,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(T *p, OffsetSimdViewT byte_offsets, ValuesSimdViewT vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(p, byte_offsets.read(), vals.read(), mask, props);
}

/// template <int VS = 1, typename OffsetT, typename ValuesSimdViewT, typename
/// T, int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename PropertyListT = empty_properties_t>
/// void scatter(T *p, simd<OffsetT, N / VS> byte_offsets, ValuesSimdViewT vals,
/// 	PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters.
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename OffsetT, typename ValuesSimdViewT, typename T,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(T *p, simd<OffsetT, N / VS> byte_offsets, ValuesSimdViewT vals,
        PropertyListT props = {}) {
  scatter<T, N, VS>(p, byte_offsets, vals.read(), props);
}

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-sc-3)
///
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector. Access to
/// any element's memory location can be disabled via the input mask.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  scatter<T, N, VS>(p, byte_offsets.read(), vals, mask, props);
}

/// template <int VS, typename OffsetSimdViewT, typename T, int N, typename
/// PropertyListT = empty_properties_t>
/// void scatter(T *p, OffsetSimdViewT byte_offsets, simd<T,N> vals,
/// 	simd_mask<N / VS> mask, PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters.
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector. Access to
/// any element's memory location can be disabled via the input mask.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS, typename OffsetSimdViewT, typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(p, byte_offsets.read(), vals, mask, props);
}

/// template <int VS, typename OffsetSimdViewT, typename T, int N, typename
/// PropertyListT = empty_properties_t>
/// void scatter(T *p, OffsetSimdViewT byte_offsets, simd<T,N> vals,
/// 	PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters.
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector. Access to
/// any element's memory location can be disabled via the input mask.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS, typename OffsetSimdViewT, typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(p, byte_offsets.read(), vals, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
/// 	  typename PropertyListT = empty_properties_t>
/// void scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals,
///      PropertyListT props = {});                         // (usm-sc-4)
///
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(T *p, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  scatter<T, N, VS>(p, byte_offsets.read(), vals, Mask, props);
}

/// template <int VS = 1, typename OffsetSimdViewT, typename
/// ValuesSimdViewT, typename T, int N = ValuesSimdViewT::getSizeX() *
/// ValuesSimdViewT::getSizeY(), typename PropertyListT =
/// empty_properties_t>
/// void scatter(T *p, OffsetSimdViewT byte_offsets, ValuesSimdViewT vals,
///      PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters.
/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename OffsetSimdViewT, typename ValuesSimdViewT, typename T,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(T *p, OffsetSimdViewT byte_offsets, ValuesSimdViewT vals,
        PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(p, byte_offsets.read(), vals.read(), props);
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
  scatter<Tx, N, 1>(p, simd<Toffset, N>(offset), vals, mask);
}

namespace detail {
// Accessors may get either 32-bit offset or 64-bit depending on
// the -fsycl-esimd-force-stateles-mem mode setting.
#ifdef __ESIMD_FORCE_STATELESS_MEM
using DeviceAccessorOffsetT = uint64_t;
#else
using DeviceAccessorOffsetT = uint32_t;
#endif

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
/// @tparam PropertyListT is the list of optional cache-hint properties and
/// the required alignment property.
/// @param p is the base pointer.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed.
/// @param pass_thru contains the vector which elements are copied
/// to the returned result when the corresponding element of \p pred is 0.
/// @return is a vector of type T and size NElts.
///
template <typename T, int NElts, typename PropertyListT>
__ESIMD_API std::enable_if_t<is_property_list_v<PropertyListT>, simd<T, NElts>>
block_load_impl(const T *p, simd_mask<1> pred, simd<T, NElts> pass_thru) {
  // Verify input template arguments.
  check_cache_hints<cache_action::load, PropertyListT>();
  constexpr size_t Alignment =
      PropertyListT::template get_property<alignment_key>().value;
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
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();

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
/// @tparam PropertyListT is the list of optional cache-hint properties and
/// the required alignment property.
/// @tparam AccessorT is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed. The default is '1' - perform
/// the operation.
/// @return is a vector of type T and size NElts. The elements of the returned
/// vector for which the corresponding element in \p pred is 0 are undefined.
///
template <typename T, int NElts, typename PropertyListT, typename AccessorT>
__ESIMD_API
    std::enable_if_t<detail::is_device_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_read> &&
                         is_property_list_v<PropertyListT>,
                     simd<T, NElts>>
    block_load_impl(AccessorT acc, DeviceAccessorOffsetT offset,
                    simd_mask<1> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  simd<T, NElts> PassThru; // Intentionally undefined.
  return block_load_impl<T, NElts, PropertyListT>(
      accessorToPointer<T>(acc, offset), pred, PassThru);
#else  // !__ESIMD_FORCE_STATELESS_MEM
  // Verify input template arguments.
  check_cache_hints<cache_action::load, PropertyListT>();
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
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
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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
/// @tparam PropertyListT is the list of optional cache-hint properties and
/// the required alignment property.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offset is the zero-based offset in bytes.
/// @param pred is operation predicate. Operation is skipped for index 'i'
/// if pred[0] == 0 the result element is taken from \p pass_thru[i].
/// Otherwise, the operation is performed and the result if it copied to
/// the result.
/// @param pass_thru contains the values copied to the result if \p pred is 0.
/// @return is a vector of type T and size NElts
///
template <typename T, int NElts, typename PropertyListT, typename AccessorT>
__ESIMD_API
    std::enable_if_t<detail::is_device_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_read> &&
                         is_property_list_v<PropertyListT>,
                     simd<T, NElts>>
    block_load_impl(AccessorT acc, DeviceAccessorOffsetT offset,
                    simd_mask<1> pred, simd<T, NElts> pass_thru) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return block_load_impl<T, NElts, PropertyListT>(
      accessorToPointer<T>(acc, offset), pred, pass_thru);
#else  // !__ESIMD_FORCE_STATELESS_MEM
  // Verify input template arguments.
  check_cache_hints<cache_action::load, PropertyListT>();
  constexpr size_t Alignment =
      PropertyListT::template get_property<alignment_key>().value;
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
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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

template <typename T, int NElts, typename PropertyListT>
__ESIMD_API std::enable_if_t<detail::is_property_list_v<PropertyListT>>
block_store_impl(T *p, simd<T, NElts> vals, simd_mask<1> pred) {
  detail::check_cache_hints<cache_action::store, PropertyListT>();
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
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
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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

template <typename T, int NElts, typename PropertyListT, typename AccessorT>
__ESIMD_API
    std::enable_if_t<detail::is_device_accessor_with_v<
                         AccessorT, detail::accessor_mode_cap::can_write> &&
                     detail::is_property_list_v<PropertyListT>>
    block_store_impl(AccessorT acc, DeviceAccessorOffsetT offset,
                     simd<T, NElts> vals, simd_mask<1> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  block_store_impl<T, NElts, PropertyListT>(accessorToPointer<T>(acc, offset),
                                            vals, pred);
#else
  // Verify input template arguments.
  check_cache_hints<cache_action::store, PropertyListT>();
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
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
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
block_load(const T *ptr, PropertyListT props = {}) {
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  using NewPropertyListT =
      detail::add_alignment_property_t<PropertyListT, DefaultAlignment>;
  if constexpr (detail::has_cache_hints<PropertyListT>()) {
    simd<T, N> PassThru; // Intentionally undefined.
    simd_mask<1> Mask = 1;
    return detail::block_load_impl<T, N, NewPropertyListT>(ptr, Mask, PassThru);
  } else {
    constexpr size_t Alignment =
        NewPropertyListT::template get_property<alignment_key>().value;
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API
    std::enable_if_t<detail::is_property_list_v<PropertyListT>, simd<T, N>>
    block_load(const T *ptr, simd_mask<1> pred, PropertyListT props = {}) {
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  using NewPropertyListT =
      detail::add_alignment_property_t<PropertyListT, DefaultAlignment>;
  simd<T, N> PassThru; // Intentionally uninitialized.
  return detail::block_load_impl<T, N, NewPropertyListT>(ptr, pred, PassThru);
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
block_load(const T *ptr, simd_mask<1> pred, simd<T, N> pass_thru,
           PropertyListT props = {}) {
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  using NewPropertyListT =
      detail::add_alignment_property_t<PropertyListT, DefaultAlignment>;
  return detail::block_load_impl<T, N, NewPropertyListT>(ptr, pred, pass_thru);
}

/// simd<T, N> block_load(const T* ptr, simd_mask<1> pred,
///                       PassThruSimdViewT pass_thru, props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// This function loads a contiguous memory block from USM pointer \p ptr. If
/// the predicate \p pred is set to 0, then the load is omitted and the vector
/// \p pass_thru is returned.
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
template <
    typename PassThruSimdViewT, typename T,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<PassThruSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
block_load(const T *ptr, simd_mask<1> pred, PassThruSimdViewT pass_thru,
           PropertyListT props = {}) {
  return block_load<T, N>(ptr, pred, pass_thru.read(), props);
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
block_load(const T *ptr, size_t byte_offset, simd_mask<1> pred,
           simd<T, N> pass_thru, PropertyListT props = {}) {
  const T *AdjustedPtr = reinterpret_cast<const T *>(
      reinterpret_cast<const int8_t *>(ptr) + byte_offset);
  return block_load<T, N>(AdjustedPtr, pred, pass_thru, props);
}

/// simd<T, N> block_load(const T* ptr, size_t byte_offset,
///                       simd_mask<1> pred, PassThruSimdViewT pass_thru,
///                       props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename PassThruSimdViewT, typename T,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<PassThruSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
block_load(const T *ptr, size_t byte_offset, simd_mask<1> pred,
           PassThruSimdViewT pass_thru, PropertyListT props = {}) {
  return block_load<T, N>(ptr, byte_offset, pred, pass_thru.read(), props);
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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

  using NewPropertyListT =
      detail::add_alignment_property_t<PropertyListT, DefaultAlignment>;
  if constexpr (detail::has_cache_hints<PropertyListT>() || !IsLegacySize) {
    return detail::block_load_impl<T, N, NewPropertyListT>(acc, byte_offset,
                                                           simd_mask<1>(1));
  } else {
    constexpr size_t Alignment =
        NewPropertyListT::template get_property<alignment_key>().value;
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, PropertyListT /* props */ = {}) {
  // Create new properties without the alignment property passed in 'props',
  // and add alignment<16> as it is usable and most favourable in this case.
  using NewPropertyListT =
      detail::add_or_replace_alignment_property_t<PropertyListT, 16>;
  return block_load<T, N>(acc, 0, NewPropertyListT{});
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, detail::DeviceAccessorOffsetT byte_offset,
           simd_mask<1> pred, simd<T, N> pass_thru,
           PropertyListT /* props */ = {}) {
  // If the alignment property is not passed, then assume the byte_offset
  // is element-aligned and is at least 4-bytes.
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  using NewPropertyListT =
      detail::add_alignment_property_t<PropertyListT, DefaultAlignment>;
  return detail::block_load_impl<T, N, NewPropertyListT>(acc, byte_offset, pred,
                                                         pass_thru);
}

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT byte_offset, simd_mask<1> pred,
///            PassThruSimdViewT pass_thru, props = {});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename PassThruSimdViewT,
    typename T = PassThruSimdViewT::value_type::element_type,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<PassThruSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, detail::DeviceAccessorOffsetT byte_offset,
           simd_mask<1> pred, PassThruSimdViewT pass_thru,
           PropertyListT props = {}) {
  return block_load<T, N>(acc, byte_offset, pred, pass_thru.read(), props);
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, simd_mask<1> pred, simd<T, N> pass_thru,
           PropertyListT /* props */ = {}) {
  // Create new properties without the alignment property passed in 'props',
  // and add alignment<16> as it is usable and most favourable in this case.
  using NewPropertyListT =
      detail::add_or_replace_alignment_property_t<PropertyListT, 16>;
  return block_load<T, N>(acc, 0, pred, pass_thru, NewPropertyListT{});
}

/// block_load(AccessorT acc, simd_mask<1> pred,
///            PassThruSimdViewT pass_thru, props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// This function loads a contiguous memory block referenced
/// by accessor \p acc and implied offset=0.
/// If the predicate \p pred is set to 0, then the load is omitted and the
/// \p pass_thru value is returned.
///
/// The parameter \p props specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2. Other properties are ignored. If \p props
/// specifies the alignment property, then it is ignored because this
/// variant implies zero offset, which means the most favourable 16-byte
/// alignment is used.
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
template <
    typename PassThruSimdViewT,
    typename T = PassThruSimdViewT::value_type::element_type,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<PassThruSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, simd_mask<1> pred, PassThruSimdViewT pass_thru,
           PropertyListT props = {}) {
  return block_load<T, N>(acc, pred, pass_thru.read(), props);
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, simd_mask<1> pred, PropertyListT /* props */ = {}) {
  // Create new properties without the alignment property passed in 'props',
  // and add alignment<16> as it is usable and most favourable in this case.
  using NewPropertyListT =
      detail::add_or_replace_alignment_property_t<PropertyListT, 16>;
  simd<T, N> PassThru; // Intentionally uninitialized.
  return block_load<T, N>(acc, 0, pred, PassThru, NewPropertyListT{});
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<detail::is_property_list_v<PropertyListT>>
block_store(T *ptr, simd<T, N> vals, PropertyListT /* props */ = {}) {
  if constexpr (detail::has_cache_hints<PropertyListT>()) {
    constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
    using NewPropertyListT =
        detail::add_alignment_property_t<PropertyListT, DefaultAlignment>;
    simd_mask<1> Mask = 1;
    detail::block_store_impl<T, N, NewPropertyListT>(ptr, vals, Mask);
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<detail::is_property_list_v<PropertyListT>>
block_store(T *ptr, simd<T, N> vals, simd_mask<1> pred,
            PropertyListT /* props */ = {}) {
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  using NewPropertyListT =
      detail::add_alignment_property_t<PropertyListT, DefaultAlignment>;
  detail::block_store_impl<T, N, NewPropertyListT>(ptr, vals, pred);
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(T *ptr, size_t byte_offset, simd<T, N> vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  T *AdjustedPtr =
      reinterpret_cast<T *>(reinterpret_cast<int8_t *>(ptr) + byte_offset);
  block_store<T, N>(AdjustedPtr, vals, pred, props);
}

/// void block_store(T* ptr, ValuesSimdViewT vals, props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT, typename T,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<detail::is_simd_view_type_v<ValuesSimdViewT> &&
                             detail::is_property_list_v<PropertyListT>>
block_store(T *ptr, ValuesSimdViewT vals, PropertyListT props = {}) {
  block_store<T, N>(ptr, vals.read(), props);
}

/// void block_store(T* ptr, size_t byte_offset,
///                          ValuesSimdViewT vals, props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT, typename T,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(T *ptr, size_t byte_offset, ValuesSimdViewT vals,
            PropertyListT props = {}) {
  block_store<T, N>(ptr, byte_offset, vals.read(), props);
}

/// void block_store(T* ptr, ValuesSimdViewT vals,
///             simd_mask<1> pred, props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT, typename T,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<detail::is_simd_view_type_v<ValuesSimdViewT> &&
                             detail::is_property_list_v<PropertyListT>>
block_store(T *ptr, ValuesSimdViewT vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  block_store<T, N>(ptr, vals.read(), pred, props);
}

/// void block_store(T* ptr, size_t byte_offset,
/// ValuesSimdViewT vals, simd_mask<1> pred, props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT, typename T,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(T *ptr, size_t byte_offset, ValuesSimdViewT vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  block_store<T, N>(ptr, byte_offset, vals.read(), pred, props);
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
  constexpr int DefaultLSCAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(
          DefaultLSCAlignment);
  constexpr bool AlignmentRequiresLSC =
      PropertyListT::template has_property<alignment_key>() && Alignment < 16;
  using Tx = detail::__raw_t<T>;
  constexpr unsigned Sz = sizeof(Tx) * N;
  constexpr bool SzRequiresLSC =
      Sz < detail::OperandSize::OWORD || Sz % detail::OperandSize::OWORD != 0 ||
      !detail::isPowerOf2(Sz / detail::OperandSize::OWORD) ||
      Sz > 8 * detail::OperandSize::OWORD;
  if constexpr (detail::has_cache_hints<PropertyListT>() ||
                AlignmentRequiresLSC || SzRequiresLSC) {
    using NewPropertyListT =
        detail::add_alignment_property_t<PropertyListT, DefaultLSCAlignment>;
    simd_mask<1> Mask = 1;
    detail::block_store_impl<T, N, NewPropertyListT>(acc, byte_offset, vals,
                                                     Mask);
  } else {
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, simd<T, N> vals, PropertyListT props = {}) {
  // Create new properties without the alignment property passed in 'props',
  // and add alignment<16> as it is usable and most favourable in this case.
  using NewPropertyListT =
      detail::add_or_replace_alignment_property_t<PropertyListT, 16>;
  block_store<T, N>(acc, 0, vals, NewPropertyListT{});
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, detail::DeviceAccessorOffsetT byte_offset,
            simd<T, N> vals, simd_mask<1> pred, PropertyListT props = {}) {
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  using NewPropertyListT =
      detail::add_alignment_property_t<PropertyListT, DefaultAlignment>;
  detail::block_store_impl<T, N, NewPropertyListT>(acc, byte_offset, vals,
                                                   pred);
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, simd<T, N> vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  // Create new properties without the alignment property passed in 'props',
  // and add alignment<16> as it is usable and most favourable in this case.
  using NewPropertyListT =
      detail::add_or_replace_alignment_property_t<PropertyListT, 16>;
  block_store<T, N>(acc, 0, vals, pred, NewPropertyListT{});
}

/// void block_store(AccessorT acc, OffsetT byte_offset,
///                   ValuesSimdViewT vals, props = {});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT,
    typename T = ValuesSimdViewT::value_type::element_type,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, detail::DeviceAccessorOffsetT byte_offset,
            ValuesSimdViewT vals, PropertyListT props = {}) {
  block_store<T, N>(acc, byte_offset, vals.read(), props);
}

/// void block_store(AccessorT acc, ValuesSimdViewT vals, props = {});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT,
    typename T = ValuesSimdViewT::value_type::element_type,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, ValuesSimdViewT vals, PropertyListT props = {}) {
  block_store<T, N>(acc, vals.read(), props);
}

/// void block_store(AccessorT acc, OffsetT byte_offset,
///     ValuesSimdViewT vals, simd_mask<1> pred, props = {});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT,
    typename T = ValuesSimdViewT::value_type::element_type,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, detail::DeviceAccessorOffsetT byte_offset,
            ValuesSimdViewT vals, simd_mask<1> pred, PropertyListT props = {}) {
  block_store<T, N>(acc, byte_offset, vals.read(), pred, props);
}

/// void block_store(AccessorT acc, ValuesSimdViewT vals,
///                  simd_mask<1> pred, props = {});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT,
    typename T = ValuesSimdViewT::value_type::element_type,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_write>>
block_store(AccessorT acc, ValuesSimdViewT vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  block_store<T, N>(acc, vals.read(), pred, props);
}

/// @} sycl_esimd_memory_block

/// @} sycl_esimd_memory

/// @cond ESIMD_DETAIL

// Implementations of accessor-based gather and scatter functions
namespace detail {
template <typename T, int N, typename AccessorTy>
ESIMD_INLINE ESIMD_NODEBUG std::enable_if_t<
    std::is_same_v<detail::LocalAccessorMarker, AccessorTy> ||
    is_accessor_with_v<AccessorTy, detail::accessor_mode_cap::can_write>>
scatter_impl(AccessorTy acc, simd<T, N> vals, simd<uint32_t, N> offsets,
             uint32_t glob_offset, simd_mask<N> mask) {

  static_assert(detail::isPowerOf2(N, 32), "Unexpected vector length");
  if constexpr (sizeof(T) == 8) {
    scatter_impl<uint32_t, N>(
        acc, vals.template bit_cast_view<uint32_t>().template select<N, 2>(0),
        offsets, glob_offset, mask);
    scatter_impl<uint32_t, N>(
        acc, vals.template bit_cast_view<uint32_t>().template select<N, 2>(1),
        offsets, glob_offset + sizeof(uint32_t), mask);
  } else {
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
}

#ifndef __ESIMD_FORCE_STATELESS_MEM
/// Accessor-based scatter.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_store.ugm
///
/// Scatters elements to surface.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to store per address.
/// @tparam DS is the data size.
/// @tparam PropertyListT is the properties with optional cache hints.
/// @tparam N is the number of channels (platform dependent).
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets in bytes.
/// @param vals is values to store.
/// @param pred is predicates.
///
template <typename T, int NElts, lsc_data_size DS, typename PropertyListT,
          int N, typename AccessorTy, typename OffsetT>
__ESIMD_API std::enable_if_t<
    is_device_accessor_with_v<AccessorTy, accessor_mode_cap::can_write>>
scatter_impl(AccessorTy acc, simd<OffsetT, N> offsets, simd<T, N * NElts> vals,
             simd_mask<N> pred) {
  static_assert(std::is_integral_v<OffsetT>,
                "Scatter must have integral byte_offset type");
  static_assert(sizeof(OffsetT) <= 4,
                "Implicit truncation of 64-bit byte_offset to 32-bit is "
                "disabled. Use -fsycl-esimd-force-stateless-mem or explicitly "
                "convert offsets to a 32-bit vector");
  check_lsc_vector_size<NElts>();
  check_lsc_data_size<T, DS>();
  check_cache_hints<cache_action::store, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size LSCNElts = to_lsc_vector_size<NElts>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  simd<MsgT, N * NElts> Tmp = lsc_format_input<MsgT, T>(vals);
  simd<uint32_t, N> ByteOffsets32 = convert<uint32_t>(offsets);
  auto si = get_surface_index(acc);
  __esimd_lsc_store_bti<MsgT, L1H, L2H, AddressScale, ImmOffset, EDS, LSCNElts,
                        Transposed, N>(pred.data(), ByteOffsets32.data(),
                                       Tmp.data(), si);
}
#endif // __ESIMD_FORCE_STATELESS_MEM

template <typename T, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<
    (std::is_same_v<detail::LocalAccessorMarker, AccessorTy> ||
     is_accessor_with_v<AccessorTy, detail::accessor_mode_cap::can_read>),
    simd<T, N>>
gather_impl(AccessorTy acc, simd<uint32_t, N> offsets, uint32_t glob_offset,
            simd_mask<N> mask) {
  static_assert(detail::isPowerOf2(N, 32), "Unexpected vector length");

  if constexpr (sizeof(T) == 8) {
    simd<T, N> Res;
    Res.template bit_cast_view<uint32_t>().template select<N, 2>(0) =
        gather_impl<uint32_t, N>(acc, offsets, glob_offset, mask);
    Res.template bit_cast_view<uint32_t>().template select<N, 2>(1) =
        gather_impl<uint32_t, N>(acc, offsets, glob_offset + sizeof(uint32_t),
                                 mask);
    return Res;
  } else {
    using Treal = __raw_t<T>;
    constexpr int TypeSizeLog2 = detail::ElemsPerAddrEncoding<sizeof(T)>();
    // TODO (performance) use hardware-supported scale once BE supports it
    constexpr uint32_t scale = 0;
    const auto si = get_surface_index(acc);
    if constexpr (sizeof(T) < 4) {
      using Tint = std::conditional_t<std::is_integral_v<T>, T,
                                      detail::uint_type_t<sizeof(T)>>;

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
}

#ifndef __ESIMD_FORCE_STATELESS_MEM
template <typename T, int N, int VS, typename PropertyListT, lsc_data_size DS,
          typename OffsetT, typename AccessorT>
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
  check_cache_hints<cache_action::load, PropertyListT>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size LSCVS = to_lsc_vector_size<VS>();
  constexpr auto Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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
/// element of \p pred is zero.
/// @return is a vector of type T and size N * NElts.
///
template <typename T, int NElts, lsc_data_size DS, int N>
__ESIMD_API simd<T, N * NElts> slm_gather_impl(simd<uint32_t, N> offsets,
                                               simd_mask<N> pred,
                                               simd<T, N * NElts> pass_thru) {
  check_lsc_vector_size<NElts>();
  check_lsc_data_size<T, DS>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size LSCVS = to_lsc_vector_size<NElts>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  simd<MsgT, N * NElts> PassThruExpanded = lsc_format_input<MsgT>(pass_thru);
  simd<MsgT, N * NElts> Result =
      __esimd_lsc_load_merge_slm<MsgT, cache_hint::none, cache_hint::none,
                                 AddressScale, ImmOffset, EDS, LSCVS,
                                 Transposed, N>(pred.data(), offsets.data(),
                                                PassThruExpanded.data());
  return lsc_format_ret<T>(Result);
}

/// SLM scatter implementation.
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
template <typename T, int NElts, lsc_data_size DS, int N>
__ESIMD_API void slm_scatter_impl(simd<uint32_t, N> offsets,
                                  simd<T, N * NElts> vals, simd_mask<N> pred) {
  check_lsc_vector_size<NElts>();
  check_lsc_data_size<T, DS>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size LSCVS = to_lsc_vector_size<NElts>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  simd<MsgT, N * NElts> Tmp = lsc_format_input<MsgT, T>(vals);
  __esimd_lsc_store_slm<MsgT, cache_hint::none, cache_hint::none, AddressScale,
                        ImmOffset, EDS, LSCVS, Transposed, N>(
      pred.data(), offsets.data(), Tmp.data());
}

/// USM pointer prefetch implementation.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Prefetches elements located at specified address.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam PropertyListT is the properties with optional cache hints.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param byte_offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
///
template <typename T, int NElts, lsc_data_size DS, typename PropertyListT,
          int N, typename Toffset>
__ESIMD_API void prefetch_impl(const T *p, simd<Toffset, N> byte_offsets,
                               simd_mask<N> pred) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  check_lsc_vector_size<NElts>();
  check_lsc_data_size<T, DS>();
  check_cache_hints<cache_action::prefetch, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size LSCVS = to_lsc_vector_size<NElts>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(byte_offsets);
  __esimd_lsc_prefetch_stateless<MsgT, L1H, L2H, AddressScale, ImmOffset, EDS,
                                 LSCVS, Transposed, N>(pred.data(),
                                                       addrs.data());
}

template <typename T, int NElts, lsc_data_size DS, typename PropertyListT,
          typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset>>
prefetch_impl(const T *p, Toffset offset, simd_mask<1> pred) {
  check_lsc_data_size<T, DS>();
  check_cache_hints<cache_action::prefetch, PropertyListT>();

  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(
      (Alignment >= __ESIMD_DNS::OperandSize::DWORD && sizeof(T) <= 4) ||
          (Alignment >= __ESIMD_DNS::OperandSize::QWORD && sizeof(T) > 4),
      "Incorrect alignment for the data type");

  constexpr int SmallIntFactor64Bit = sizeof(uint64_t) / sizeof(T);
  constexpr int SmallIntFactor32Bit =
      sizeof(uint32_t) / sizeof(T) > 1 ? sizeof(uint32_t) / sizeof(T) : 1;
  static_assert(NElts > 0 && NElts % SmallIntFactor32Bit == 0,
                "Number of elements is not supported by Transposed load");

  // If alignment >= 8 and (NElts * sizeof(T)) % 8 == 0) we can prefetch QWORDs.
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

  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = finalize_data_size<LoadElemT, DS>();

  static_assert(
      EDS == lsc_data_size::u32 || EDS == lsc_data_size::u64,
      "Transposed prefetch is supported only for data size u32 or u64");
  constexpr lsc_vector_size LSCVS = to_lsc_vector_size<FactoredNElts>();
  constexpr lsc_data_order Transposed = lsc_data_order::transpose;
  constexpr int N = 1;

  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p) + offset;
  __esimd_lsc_prefetch_stateless<LoadElemT, L1H, L2H, AddressScale, ImmOffset,
                                 EDS, LSCVS, Transposed, N>(pred.data(),
                                                            addrs.data());
}

#ifndef __ESIMD_FORCE_STATELESS_MEM
/// Accessor-based prefetch gather.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_load.ugm
///
/// Prefetches elements located at surface.
///
/// @tparam T is element type.
/// @tparam NElts is the number of elements to load per address.
/// @tparam DS is the data size.
/// @tparam PropertyListT is the properties with optional cache hints.
/// @tparam N is the number of channels (platform dependent).
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @tparam OffsetT is the type of \c byte_offsets.
/// @param acc is the SYCL accessor.
/// @param byte_offsets is the zero-based offsets in bytes.
/// @param pred is predicates.
///

template <typename T, int NElts, lsc_data_size DS, typename PropertyListT,
          int N, typename AccessorTy, typename OffsetT>
__ESIMD_API std::enable_if_t<
    is_device_accessor_with_v<AccessorTy, accessor_mode_cap::can_read>>
prefetch_impl(AccessorTy acc, simd<OffsetT, N> byte_offsets,
              simd_mask<N> pred) {
  static_assert(std::is_integral_v<OffsetT>,
                "Prefetch must have integral byte_offset type");
  static_assert(sizeof(OffsetT) <= 4,
                "Implicit truncation of 64-bit byte_offset to 32-bit is "
                "disabled. Use -fsycl-esimd-force-stateless-mem or explicitly "
                "convert offsets to a 32-bit vector");
  check_lsc_vector_size<NElts>();
  check_lsc_data_size<T, DS>();
  check_cache_hints<cache_action::prefetch, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = expand_data_size(finalize_data_size<T, DS>());
  constexpr lsc_vector_size LSCVS = to_lsc_vector_size<NElts>();
  constexpr lsc_data_order Transposed = lsc_data_order::nontranspose;
  using MsgT = typename lsc_expand_type<T>::type;
  simd<uint32_t, N> ByteOffsets32 = convert<uint32_t>(byte_offsets);
  auto SI = get_surface_index(acc);
  __esimd_lsc_prefetch_bti<MsgT, L1H, L2H, AddressScale, ImmOffset, EDS, LSCVS,
                           Transposed, N>(pred.data(), ByteOffsets32.data(),
                                          SI);
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
/// @tparam PropertyListT is the properties with optional cache hints.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @tparam OffsetT is the type of \c byte_offset.
/// @param acc is the SYCL accessor.
/// @param byte_offset is the zero-based offset in bytes.
/// @param pred is operation predicate. Zero means operation is skipped
/// entirely, non-zero - operation is performed.
///
template <typename T, int NElts, lsc_data_size DS, typename PropertyListT,
          typename AccessorTy, typename OffsetT>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<OffsetT> &&
    is_device_accessor_with_v<AccessorTy, accessor_mode_cap::can_read>>
prefetch_impl(AccessorTy acc, OffsetT byte_offset, simd_mask<1> pred) {
  static_assert(sizeof(OffsetT) <= 4,
                "Implicit truncation of 64-bit byte_offset to 32-bit is "
                "disabled. Use -fsycl-esimd-force-stateless-mem or explicitly "
                "convert offsets to a 32-bit vector");
  check_lsc_data_size<T, DS>();
  check_cache_hints<cache_action::prefetch, PropertyListT>();

  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));

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

  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
  constexpr uint16_t AddressScale = 1;
  constexpr int ImmOffset = 0;
  constexpr lsc_data_size EDS = finalize_data_size<LoadElemT, DS>();

  static_assert(
      EDS == lsc_data_size::u32 || EDS == lsc_data_size::u64,
      "Transposed prefetch is supported only for data size u32 or u64");
  constexpr lsc_vector_size LSCVS = to_lsc_vector_size<FactoredNElts>();
  constexpr lsc_data_order Transposed = lsc_data_order::transpose;
  constexpr int N = 1;

  simd<uint32_t, N> offsets = byte_offset;
  auto SI = get_surface_index(acc);
  __esimd_lsc_prefetch_bti<LoadElemT, L1H, L2H, AddressScale, ImmOffset, EDS,
                           LSCVS, Transposed, N>(pred.data(), offsets.data(),
                                                 SI);
}
#endif // __ESIMD_FORCE_STATELESS_MEM

// Compute the data size for 2d block load or store.
template <typename T, int NBlocks, int Height, int Width, bool Transposed,
          bool Transformed>
constexpr int get_lsc_block_2d_data_size() {
  if constexpr (Transformed)
    return roundUpNextMultiple<Height, 4 / sizeof(T)>() *
           getNextPowerOf2<Width>() * NBlocks;
  return Width * Height * NBlocks;
}

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
/// @tparam PropertyListT The compile-time properties. Only cache hint
/// properties are used.
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
template <
    typename T, int BlockWidth, int BlockHeight, int NBlocks, bool Transposed,
    bool Transformed, typename PropertyListT,
    int N = get_lsc_block_2d_data_size<__raw_t<T>, NBlocks, BlockHeight,
                                       BlockWidth, Transposed, Transformed>()>
__ESIMD_API simd<T, N> load_2d_impl(const T *Ptr, unsigned SurfaceWidth,
                                    unsigned SurfaceHeight,
                                    unsigned SurfacePitch, int X, int Y) {

  check_cache_hints<cache_action::load, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
  using RawT = __raw_t<T>;
  check_lsc_block_2d_restrictions<RawT, BlockWidth, BlockHeight, NBlocks,
                                  Transposed, Transformed, block_2d_op::load>();
  // For Load BlockWidth is padded up to the next power-of-two value.
  // For Load with Transpose the pre-operation BlockHeight is padded up
  // to the next power-of-two value.
  // For Load with Transform pre-operation BlockHeight is padded up to
  // multiple of K, where K = 4B / sizeof(T).
  constexpr int ElemsPerDword = 4 / sizeof(RawT);
  constexpr int GRFRowSize = Transposed    ? BlockHeight
                             : Transformed ? BlockWidth * ElemsPerDword
                                           : BlockWidth;
  constexpr int GRFRowPitch = getNextPowerOf2<GRFRowSize>();
  constexpr int GRFColSize =
      Transposed
          ? BlockWidth
          : (Transformed ? (BlockHeight + ElemsPerDword - 1) / ElemsPerDword
                         : BlockHeight);
  constexpr int GRFBlockSize = GRFRowPitch * GRFColSize;
  constexpr int GRFBlockPitch =
      roundUpNextMultiple<64 / sizeof(RawT), GRFBlockSize>();
  constexpr int ActualN = NBlocks * GRFBlockPitch;

  constexpr int DstBlockElements = GRFColSize * GRFRowSize;
  constexpr int DstElements = DstBlockElements * NBlocks;

  static_assert(N == ActualN || N == DstElements, "Incorrect element count");
  simd_mask<1> Mask = 1;
  constexpr lsc_data_size DS =
      finalize_data_size<RawT, lsc_data_size::default_size>();
  uintptr_t Addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr lsc_data_order Transpose =
      Transposed ? lsc_data_order::transpose : lsc_data_order::nontranspose;
  simd<RawT, ActualN> Raw =
      __esimd_lsc_load2d_stateless<RawT, L1H, L2H, DS, Transpose, NBlocks,
                                   BlockWidth, BlockHeight, Transformed,
                                   ActualN>(Mask.data(), Addr, SurfaceWidth,
                                            SurfaceHeight, SurfacePitch, X, Y);

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

    simd<RawT, DstElements> Dst;

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
/// @tparam PropertyListT The compile-time properties. Only cache hint
/// properties are used.
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
template <typename T, int BlockWidth, int BlockHeight, int NBlocks,
          typename PropertyListT,
          int N = get_lsc_block_2d_data_size<__raw_t<T>, NBlocks, BlockHeight,
                                             BlockWidth, false /*Transposed*/,
                                             false /*Transformed*/>()>
__ESIMD_API void prefetch_2d_impl(const T *Ptr, unsigned SurfaceWidth,
                                  unsigned SurfaceHeight, unsigned SurfacePitch,
                                  int X, int Y) {
  using RawT = __raw_t<T>;
  check_cache_hints<cache_action::prefetch, PropertyListT>();
  check_lsc_block_2d_restrictions<RawT, BlockWidth, BlockHeight, NBlocks, false,
                                  false, block_2d_op::prefetch>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
  constexpr lsc_data_size DS =
      finalize_data_size<RawT, lsc_data_size::default_size>();
  uintptr_t Addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr lsc_data_order Transpose = lsc_data_order::nontranspose;
  simd_mask<1> Mask = 1;
  __esimd_lsc_prefetch2d_stateless<RawT, L1H, L2H, DS, Transpose, NBlocks,
                                   BlockWidth, BlockHeight, false, N>(
      Mask.data(), Addr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y);
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
/// @tparam PropertyListT The compile-time properties. Only cache hint
/// properties are used.
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
template <typename T, int BlockWidth, int BlockHeight, typename PropertyListT,
          int N = detail::get_lsc_block_2d_data_size<
              __raw_t<T>, 1u, BlockHeight, BlockWidth, false /*Transposed*/,
              false /*Transformed*/>()>
__ESIMD_API void store_2d_impl(T *Ptr, unsigned SurfaceWidth,
                               unsigned SurfaceHeight, unsigned SurfacePitch,
                               int X, int Y, simd<T, N> Vals) {
  using RawT = __raw_t<T>;
  __ESIMD_DNS::check_cache_hints<__ESIMD_DNS::cache_action::store,
                                 PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
  check_lsc_block_2d_restrictions<RawT, BlockWidth, BlockHeight, 1, false,
                                  false, block_2d_op::store>();
  constexpr lsc_data_size DS =
      finalize_data_size<RawT, lsc_data_size::default_size>();
  uintptr_t Addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr lsc_data_order Transpose = lsc_data_order::nontranspose;

  constexpr int Pitch = getNextPowerOf2<BlockWidth>();
  constexpr int NElts = BlockHeight * Pitch;
  simd<RawT, NElts> Raw;
  simd_mask<1> Mask = 1;

  if constexpr (NElts == N) {
    Raw = Vals;
  } else {
    // For store with padding, allocate the block with padding, and place
    // original data there.
    auto Data2D = Vals.template bit_cast_view<RawT, BlockHeight, BlockWidth>();
    auto Raw2D = Raw.template bit_cast_view<RawT, BlockHeight, Pitch>();
    Raw2D.template select<BlockHeight, 1, BlockWidth, 1>(0, 0) = Data2D;
  }

  __esimd_lsc_store2d_stateless<RawT, L1H, L2H, DS, Transpose, 1u, BlockWidth,
                                BlockHeight, false, NElts>(
      Mask.data(), Addr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y,
      Raw.data());
}

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
  if constexpr (!detail::isPowerOf2(N, 32)) {
    // Requires DG2 or PVC.
    simd<T, N> PassThru; // Intentionally undefined
    byte_offsets += glob_offset;
    return detail::gather_impl<T, N, 1,
                               oneapi::experimental::empty_properties_t,
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
template <
    typename T, int N, int VS, typename AccessorT, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
  return detail::gather_impl<T, N, VS, PropertyListT,
                             detail::lsc_data_size::default_size>(
      acc, byte_offsets, mask, pass_thru);
#endif // __ESIMD_FORCE_STATELESS_MEM
}

/// template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (acc-ga-2)
/// Supported platforms: DG2, PVC in most cases. DG2/PVC is not required if
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
template <
    typename T, int N, int VS, typename AccessorT, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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

  if constexpr (detail::has_cache_hints<PropertyListT>() || VS > 1 ||
                !(detail::isPowerOf2(N, 32))) {
    simd<T, N> PassThru; // Intentionally undefined
    return detail::gather_impl<T, N, VS, PropertyListT,
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
/// Supported platforms: DG2, PVC in most cases. DG2/PVC is not required if
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
template <
    typename T, int N, int VS, typename AccessorT, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
// Dev note: the mask type was turned into template parameter `MaskT` to
// avoid the conflicts of this prototype with the old gather() function
// accepting a 'global_offset' parameter and avoid 'ambiguous call' errors
// for calls like this: gather(acc, byte_offsets_simd, 0, mask);
template <
    typename T, int N, typename AccessorT, typename OffsetT, typename MaskT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     std::is_same_v<MaskT, simd_mask<N>> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<OffsetT, N> byte_offsets, MaskT mask,
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
// Dev note: the mask type was turned into template parameter `MaskT` to
// avoid the conflicts of this prototype with the old gather() function
// accepting a 'global_offset' parameter and avoid 'ambiguous call' errors
// for calls like this: gather(acc, byte_offsets_simd, 0);
template <
    typename T, int N, typename AccessorT, typename OffsetT, typename MaskT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     std::is_same_v<MaskT, simd_mask<N>> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
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
template <
    typename T, int N, typename AccessorT, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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

/// template <int VS, typename T, int N, typename OffsetSimdViewT,
//            typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});
/// This function is identical to (lacc-ga-1) except that the \p byte_offsets
/// is represented as \c simd_view.
template <
    int VS, typename T, int N, typename AccessorT, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<OffsetSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       simd<T, N> pass_thru, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of pass_thru parameter must correspond to the size of "
                "byte_offsets parameter.");
  return gather<T, N, VS>(acc, byte_offsets.read(), mask, pass_thru, props);
}

/// template <int VS = 1, typename AccessorT,
///    typename OffsetSimdViewT, typename PassThruSimdViewT,
///    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
///    typename T = PassThruSimdViewT::value_type::element_type,
///    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask, PassThruSimdViewT pass_thru,
///                   PropertyListT props = {});
/// This function is identical to (lacc-ga-1) except that the \p byte_offsets
/// and \p pass_thru are represented as \c simd_view.
template <
    int VS = 1, typename AccessorT, typename OffsetSimdViewT,
    typename PassThruSimdViewT,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename T = PassThruSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<OffsetSimdViewT> &&
     detail::is_simd_view_type_v<PassThruSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       PassThruSimdViewT pass_thru, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of pass_thru parameter must correspond to the size of "
                "byte_offsets parameter.");
  return gather<T, N, VS>(acc, byte_offsets.read(), mask, pass_thru.read(),
                          props);
}

/// template <int VS = 1, typename AccessorT,
///    typename OffsetT, typename PassThruSimdViewT,
///    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
///    typename T = PassThruSimdViewT::value_type::element_type,
///    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});
/// This function is identical to (lacc-ga-1) except that the \p byte_offsets
/// is represented as \c simd_view.
template <
    int VS = 1, typename AccessorT, typename OffsetT,
    typename PassThruSimdViewT,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename T = PassThruSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_device_accessor_with_v<AccessorT,
                                       detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<PassThruSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
       simd_mask<N / VS> mask, PassThruSimdViewT pass_thru,
       PropertyListT props = {}) {
  return gather<T, N, VS>(acc, byte_offsets, mask, pass_thru.read(), props);
}

/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename OffsetSimdViewT,
//            typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (acc-ga-8)
/// This function is identical to (acc-ga-2) except that the \p byte_offsets
/// is represented as \c simd_view.
template <
    typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
/// template <typename T, int N, int VS = 1, typename AccessorTy,
/// typename OffsetT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
///              simd<T, N> vals, simd_mask<N / VS> mask,
///              PropertyListT props = {});                        // (acc-sc-1)
///
/// template <typename T, int N, int VS = 1, typename AccessorTy,
/// typename OffsetT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
///              simd<T, N> vals, PropertyListT props = {});      // (acc-sc-2)

/// The following two functions are similar to acc-sc-{1,2} with the
/// 'byte_offsets' parameter represented as 'simd_view'.

/// template <typename T, int N, int VS = 1, typename AccessorTy,
/// typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	         simd_mask<N / VS> mask, PropertyListT props = {});// (acc-sc-3)
///
/// template <typename T, int N, int VS = 1, typename AccessorTy,
/// typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	         PropertyListT props = {});                       // (acc-sc-4)
///
/// template <typename T, int N, int VS = 1, typename AccessorTy,
/// typename OffsetT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets, simd<T, N>
///              simd<T, N> vals, simd_mask<N / VS> mask,
///              PropertyListT props = {});                      // (acc-sc-1)
///
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the store to
/// (acc + byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    typename T, int N, int VS = 1, typename AccessorTy, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  scatter<T, N, VS>(__ESIMD_DNS::accessorToPointer<T>(acc), byte_offsets, vals,
                    mask, props);
#else
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(Alignment >= sizeof(T),
                "gather() requires at least element-size alignment");

  if constexpr (detail::has_cache_hints<PropertyListT>() || VS > 1 ||
                !detail::isPowerOf2(N, 32)) {
    detail::scatter_impl<T, VS, detail::lsc_data_size::default_size,
                         PropertyListT>(acc, byte_offsets, vals, mask);
  } else {
    detail::scatter_impl<T, N, AccessorTy>(acc, vals, byte_offsets, 0, mask);
  }

#endif // __ESIMD_FORCE_STATELESS_MEM
}
/// template <typename T, int N, int VS = 1, typename AccessorTy,
/// typename OffsetT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
///              simd<T, N> vals, PropertyListT props = {});   // (acc-sc-2)
///
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    typename T, int N, int VS = 1, typename AccessorTy, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets, simd<T, N> vals,
        PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  scatter<T, N, VS>(acc, byte_offsets, vals, Mask, props);
}

/// template <typename T, int N, int VS = 1, typename AccessorTy,
/// typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	         simd_mask<N / VS> mask,
///              PropertyListT props = {});                       // (acc-sc-3)
///
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the store to
/// (acc + byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    typename T, int N, int VS = 1, typename AccessorTy,
    typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  scatter<T, N, VS>(acc, byte_offsets.read(), vals, mask, props);
}

/// template <int VS, typename AccessorTy, typename T, int N,
/// typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	         simd_mask<N / VS> mask,
///              PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets. Access to any
/// element's memory location can be disabled via the input vector of predicates
/// \p mask. If mask[i] is unset, then the store to (acc + byte_offsets[i]) is
/// skipped.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS, typename AccessorTy, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(acc, byte_offsets.read(), vals, mask, props);
}

/// template <int VS, typename AccessorTy, typename T, int N,
/// typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	         PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS, typename AccessorTy, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(acc, byte_offsets.read(), vals, props);
}

/// template <int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
/// typename OffsetSimdViewT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets,
///              ValuesSimdViewT vals, simd_mask<N / VS> mask,
///              PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets. Access to any
/// element's memory location can be disabled via the input vector of predicates
/// \p mask. If mask[i] is unset, then the store to (acc + byte_offsets[i]) is
/// skipped.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
    typename OffsetSimdViewT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, ValuesSimdViewT vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(acc, byte_offsets.read(), vals.read(), mask, props);
}

/// template <int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
/// typename OffsetSimdViewT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets,
///              ValuesSimdViewT vals, PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
    typename OffsetSimdViewT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, ValuesSimdViewT vals,
        PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(acc, byte_offsets.read(), vals.read(), props);
}

/// template <int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
/// typename OffsetT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
///              ValuesSimdViewT vals, simd_mask<N / VS> mask,
///              PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets. Access to any
/// element's memory location can be disabled via the input vector of predicates
/// \p mask. If mask[i] is unset, then the store to (acc + byte_offsets[i]) is
/// skipped.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename AccessorTy, typename ValuesSimdViewT, typename OffsetT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
        ValuesSimdViewT vals, simd_mask<N / VS> mask,
        PropertyListT props = {}) {
  scatter<T, N, VS>(acc, byte_offsets, vals.read(), mask, props);
}

/// template <int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
/// typename OffsetT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
///              ValuesSimdViewT vals, PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename AccessorTy, typename ValuesSimdViewT, typename OffsetT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
        ValuesSimdViewT vals, PropertyListT props = {}) {
  scatter<T, N, VS>(acc, byte_offsets, vals.read(), props);
}

/// template <typename T, int N, int VS = 1, typename AccessorTy,
/// typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	         PropertyListT props = {});                        // (acc-sc-4)
///
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    typename T, int N, int VS = 1, typename AccessorTy,
    typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  scatter<T, N, VS>(acc, byte_offsets.read(), vals, Mask, props);
}

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
__ESIMD_API
    std::enable_if_t<(detail::isPowerOf2(N, 32)) &&
                     detail::is_device_accessor_with_v<
                         AccessorTy, detail::accessor_mode_cap::can_write>>
    scatter(AccessorTy acc, simd<detail::DeviceAccessorOffsetT, N> offsets,
            simd<T, N> vals, detail::DeviceAccessorOffsetT glob_offset,
            simd_mask<N> mask = 1) {
  offsets += glob_offset;
  scatter<T, N>(acc, offsets, vals, mask);
}

template <typename T, int N, typename AccessorTy>
__ESIMD_API
    std::enable_if_t<(detail::isPowerOf2(N, 32)) &&
                     detail::is_device_accessor_with_v<
                         AccessorTy, detail::accessor_mode_cap::can_write>>
    scatter(AccessorTy acc, detail::DeviceAccessorOffsetT glob_offset,
            simd<T, N> vals, simd_mask<N> mask = 1) {
  simd<detail::DeviceAccessorOffsetT, N> ByteOffsets = 0;
  scatter<T, N>(acc, ByteOffsets, vals, glob_offset, mask);
}

#ifdef __ESIMD_FORCE_STATELESS_MEM
template <typename T, int N, typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorTy,
                                      detail::accessor_mode_cap::can_write> &&
    std::is_integral_v<Toffset> && !std::is_same_v<Toffset, uint64_t>>
scatter(AccessorTy acc, simd<Toffset, N> offsets, simd<T, N> vals,
        uint64_t glob_offset, simd_mask<N> mask = 1) {
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
          int N, typename OffsetSimdViewT, typename RegionTy>
__ESIMD_API std::enable_if_t<detail::is_simd_view_type_v<OffsetSimdViewT>,
                             simd<T, N * get_num_channels_enabled(RGBAMask)>>
gather_rgba(const T *p, OffsetSimdViewT offsets, simd_mask<N> mask = 1) {
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
          int N, typename OffsetSimdViewT, typename RegionTy>
__ESIMD_API std::enable_if_t<detail::is_simd_view_type_v<OffsetSimdViewT>>
scatter_rgba(T *p, OffsetSimdViewT offsets,
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
/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT
///           typename PropertyListT = empty_props_t>
/// simd <T, N> slm_gather(OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
///                        simd<T, N> pass_thru
///                        PropertyListT props = {});              // (slm-ga-7)
/// simd <T, N> slm_gather(OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
///                        PropertyListT props = {});              // (slm-ga-8)
/// simd <T, N> slm_gather(OffsetSimdViewT byte_offsets,
///                        PropertyListT props = {});              // (slm-ga-9)

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
template <
    typename T, int N, int VS,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
    if constexpr (sizeof(T) == 8) {
      simd<T, N> Res;
      Res.template bit_cast_view<uint32_t>().template select<N, 2>(0) =
          __esimd_slm_gather_ld<uint32_t, N, Alignment>(
              byte_offsets.data(), mask.data(),
              (pass_thru.template bit_cast_view<uint32_t>()
                   .template select<N, 2>(0))
                  .data());
      simd<uint32_t, N / VS> Offset = byte_offsets + sizeof(uint32_t);
      Res.template bit_cast_view<uint32_t>().template select<N, 2>(1) =
          __esimd_slm_gather_ld<uint32_t, N, sizeof(uint32_t)>(
              Offset.data(), mask.data(),
              (pass_thru.template bit_cast_view<uint32_t>()
                   .template select<N, 2>(1))
                  .data());
      return Res;
    } else {
      using MsgT = detail::__raw_t<T>;
      return __esimd_slm_gather_ld<MsgT, N, Alignment>(
          byte_offsets.data(), mask.data(), pass_thru.data());
    }
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
template <
    typename T, int N, int VS,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_gather(simd<uint32_t, N / VS> byte_offsets, simd_mask<N / VS> mask,
           PropertyListT props = {}) {
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(Alignment >= sizeof(T),
                "slm_gather() requires at least element-size alignment");

  if constexpr (VS > 1 || (!detail::isPowerOf2(N, 32) &&
                           !detail::isMaskedGatherScatterLLVMAvailable())) {
    simd<T, N> PassThru; // Intentionally undefined
    return detail::slm_gather_impl<T, VS, detail::lsc_data_size::default_size>(
        byte_offsets, mask, PassThru);
  } else if constexpr (detail::isMaskedGatherScatterLLVMAvailable()) {
    if constexpr (sizeof(T) == 8) {
      simd<T, N> Res;
      simd<uint32_t, N> PassThru; // it is intentionally undefined

      Res.template bit_cast_view<uint32_t>().template select<N, 2>(0) =
          __esimd_slm_gather_ld<uint32_t, N, Alignment>(
              byte_offsets.data(), mask.data(), PassThru.data());
      simd<uint32_t, N / VS> Offset = byte_offsets + sizeof(uint32_t);
      Res.template bit_cast_view<uint32_t>().template select<N, 2>(1) =
          __esimd_slm_gather_ld<uint32_t, N, sizeof(uint32_t)>(
              Offset.data(), mask.data(), PassThru.data());
      return Res;
    } else {
      using MsgT = detail::__raw_t<T>;
      simd<MsgT, N> PassThru; // it is intentionally undefined
      return __esimd_slm_gather_ld<MsgT, N, Alignment>(
          byte_offsets.data(), mask.data(), PassThru.data());
    }
  } else {
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
template <
    typename T, int N, int VS,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
slm_gather(simd<uint32_t, N> byte_offsets, PropertyListT props = {}) {
  constexpr int VS = 1;
  return slm_gather<T, N, VS>(byte_offsets, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename PropertyListT = empty_props_t>
/// simd <T, N> slm_gather(
///             OffsetSimdViewT byte_offsets,
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
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
slm_gather(OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
           simd<T, N> pass_thru, PropertyListT props = {}) {
  return slm_gather<T, N, VS>(byte_offsets.read(), mask, pass_thru, props);
}

/// template <int VS, typename T, int N, typename OffsetSimdViewT,
///           typename PropertyListT = empty_props_t>
/// simd <T, N> slm_gather(
///             OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, simd<T, N> pass_thru,
///             PropertyListT props = {});
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters.
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements. Access to any element's memory location can be disabled via the
/// input vector of predicates \p mask. If mask[i] is unset, then the load from
/// (byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
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
template <
    int VS, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_simd_view_type_v<OffsetSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
slm_gather(OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
           simd<T, N> pass_thru, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of pass_thru parameter must correspond to the size of "
                "byte_offsets parameter.");
  return slm_gather<T, N, VS>(byte_offsets.read(), mask, pass_thru, props);
}

/// template <int VS = 1,
///    typename OffsetSimdViewT, typename PassThruSimdViewT,
///    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
///     typename T = PassThruSimdViewT::value_type::element_type,
///    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
/// simd <T, N> slm_gather(
///             OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, PassThruSimdViewT pass_thru,
///             PropertyListT props = {});
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters.
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements. Access to any element's memory location can be disabled via the
/// input vector of predicates \p mask. If mask[i] is unset, then the load from
/// (byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
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
template <
    int VS = 1, typename OffsetSimdViewT, typename PassThruSimdViewT,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename T = PassThruSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_simd_view_type_v<OffsetSimdViewT> &&
     detail::is_simd_view_type_v<PassThruSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
slm_gather(OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
           PassThruSimdViewT pass_thru, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of pass_thru parameter must correspond to the size of "
                "byte_offsets parameter.");
  return slm_gather<T, N, VS>(byte_offsets.read(), mask, pass_thru.read(),
                              props);
}

/// template <int VS = 1,
///    typename PassThruSimdViewT,
///    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
///    typename T = PassThruSimdViewT::value_type::element_type,
///    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
/// simd <T, N> slm_gather(
///             OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, PassThruSimdViewT pass_thru,
///             PropertyListT props = {});
/// Variation of the API that allows using \c simd_view without specifying \c T
/// and \c N template parameters.
/// Loads ("gathers") elements of the type 'T' from Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets, and returns the loaded
/// elements. Access to any element's memory location can be disabled via the
/// input vector of predicates \p mask. If mask[i] is unset, then the load from
/// (byte_offsets[i]) is skipped and the corresponding i-th element from
/// \p pass_thru operand is returned.
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
template <
    int VS = 1, typename PassThruSimdViewT,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename T = PassThruSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_simd_view_type_v<PassThruSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
slm_gather(simd<uint32_t, N / VS> byte_offsets, simd_mask<N / VS> mask,
           PassThruSimdViewT pass_thru, PropertyListT props = {}) {
  return slm_gather<T, N, VS>(byte_offsets, mask, pass_thru.read(), props);
}

/// simd <T, N> slm_gather(
///             OffsetSimdViewT byte_offsets,
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
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
slm_gather(OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
           PropertyListT props = {}) {
  return slm_gather<T, N, VS>(byte_offsets.read(), mask, props);
}

/// simd <T, N> slm_gather(
///             OffsetSimdViewT byte_offsets,
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
/// property is used.
/// @return A vector of elements read.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
slm_gather(OffsetSimdViewT byte_offsets, PropertyListT props = {}) {
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

/// template <typename T, int N, int VS = 1,
///           typename PropertyListT = empty_properties_t>
/// void slm_scatter(simd<uint32_t, N / VS> byte_offsets,
///                  simd<T, N> vals, simd_mask<N / VS> mask,
///                  PropertyListT props = {});                   // (slm-sc-1)
/// void slm_scatter(simd<uint32_t, N / VS> byte_offsets,
///                   simd<T, N> vals, PropertyListT props = {});  // (slm-sc-2)
///
/// The next 2 functions are variations of the first 2 above (slm-sc-1,2)
/// and were added only to support simd_view instead of simd for byte_offsets.
/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///            typename PropertyListT = empty_props_t>
/// void slm_scatter(OffsetSimdViewT byte_offsets,
///             simd<T, N> vals, simd_mask<N / VS> mask,
///             PropertyListT props = {});                         // (slm-sc-3)
/// void slm_scatter(OffsetSimdViewT byte_offsets,
///             simd<T, N> vals, PropertyListT props = {});        // (slm-sc-4)

/// template <typename T, int N, int VS = 1,
///           typename PropertyListT = empty_properties_t>
/// void slm_scatter(simd<uint32_t, N / VS> byte_offsets,
///                   simd<T, N> vals, simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (slm-sc-1)
/// Stores ("scatters") elements of the type 'T' to Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets. Storage of any element
/// can be disabled via the input vector of predicates \p mask.
/// If mask[i] is unset, then the storage to (byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector of values to store.
/// @param mask The access mask, defaults to all 1s.
/// @param props The optional compile-time properties. Only 'alignment' property
/// is used.
template <
    typename T, int N, int VS = 1,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_scatter(simd<uint32_t, N / VS> byte_offsets, simd<T, N> vals,
            simd_mask<N / VS> mask, PropertyListT props = {}) {
  static_assert(N / VS >= 1 && N % VS == 0, "N must be divisible by VS");

  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
  static_assert(Alignment >= sizeof(T),
                "slm_scatter() requires at least element-size alignment");

  // Use LSC lowering if VS > 1.
  if constexpr (VS > 1 || (!detail::isPowerOf2(N, 32) &&
                           !detail::isMaskedGatherScatterLLVMAvailable())) {
    __ESIMD_DNS::slm_scatter_impl<T, VS, detail::lsc_data_size::default_size>(
        byte_offsets, vals, mask);
  } else if constexpr (detail::isMaskedGatherScatterLLVMAvailable()) {
    if constexpr (sizeof(T) == 8) {
      __esimd_slm_scatter_st<uint32_t, N, Alignment>(
          vals.template bit_cast_view<uint32_t>()
              .template select<N, 2>(0)
              .data(),
          byte_offsets.data(), mask.data());
      simd<uint32_t, N / VS> Offset = byte_offsets + sizeof(uint32_t);
      __esimd_slm_scatter_st<uint32_t, N, sizeof(uint32_t)>(
          vals.template bit_cast_view<uint32_t>()
              .template select<N, 2>(1)
              .data(),
          Offset.data(), mask.data());

    } else {
      using MsgT = detail::__raw_t<T>;
      __esimd_slm_scatter_st<MsgT, N, Alignment>(
          sycl::bit_cast<__ESIMD_DNS::vector_type_t<MsgT, N>>(vals.data()),
          byte_offsets.data(), mask.data());
    }
  } else {
    detail::LocalAccessorMarker acc;
    detail::scatter_impl<T, N>(acc, vals, byte_offsets, 0, mask);
  }
}

/// template <typename T, int N, int VS = 1,
///           typename PropertyListT = empty_properties_t>
/// void slm_scatter(simd<uint32_t, N / VS> byte_offsets, simd<T, N> vals,
///                   PropertyListT props = {});                   // (slm-sc-2)
/// Stores ("scatters") elements of the type 'T' to Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors..
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param vals The vector of values to store.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    typename T, int N, int VS = 1,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_scatter(simd<uint32_t, N / VS> byte_offsets, simd<T, N> vals,
            PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  slm_scatter<T, N, VS>(byte_offsets, vals, Mask, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename PropertyListT = empty_props_t>
/// void slm_scatter(
///             OffsetSimdViewT byte_offsets, simd<T, N> vals,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (slm-sc-3)
/// Stores ("scatters") elements of the type 'T' to Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets.
/// Storage to any element's memory location can be disabled via the
/// input vector of predicates \p mask. If mask[i] is unset, then the storage to
/// (byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors..
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector of values to store.
/// @param mask The access mask, defaults to all 1s.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_scatter(OffsetSimdViewT byte_offsets, simd<T, N> vals,
            simd_mask<N / VS> mask, PropertyListT props = {}) {
  slm_scatter<T, N, VS>(byte_offsets.read(), vals, mask, props);
}

/// void slm_scatter(
///             OffsetSimdViewT byte_offsets, simd<T, N> vals,
///             PropertyListT props = {});                         // (slm-sc-4)
/// Stores ("scatters") elements of the type 'T' to Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param vals The vector of values to store.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_scatter(OffsetSimdViewT byte_offsets, simd<T, N> vals,
            PropertyListT props = {}) {
  return slm_scatter<T, N, VS>(byte_offsets.read(), vals, props);
}

/// template <int VS, typename T, int N, typename OffsetSimdViewT,
/// typename PropertyListT = empty_properties_t>
/// void slm_scatter(OffsetSimdViewT byte_offsets, simd<T, N> vals,
///	         simd_mask<N / VS> mask, PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param vals The vector of values to store.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    int VS, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_scatter(OffsetSimdViewT byte_offsets, simd<T, N> vals,
            simd_mask<N / VS> mask, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  slm_scatter<T, N, VS>(byte_offsets.read(), vals, mask, props);
}

/// template <int VS, typename T, int N, typename OffsetSimdViewT,
/// typename PropertyListT = empty_properties_t>
/// void slm_scatter(OffsetSimdViewT byte_offsets, simd<T, N> vals,
///	         PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param vals The vector of values to store.
/// @param props The optional comspile-time properties. Only 'alignment'
/// property is used.
template <
    int VS, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_scatter(OffsetSimdViewT byte_offsets, simd<T, N> vals,
            PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  slm_scatter<T, N, VS>(byte_offsets.read(), vals, props);
}

/// template <int VS = 1, typename ValuesSimdViewT, typename OffsetSimdViewT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void slm_scatter(OffsetSimdViewT byte_offsets,
///              ValuesSimdViewT vals, simd_mask<N / VS> mask,
///              PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param vals The vector of values to store.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    int VS = 1, typename ValuesSimdViewT, typename OffsetSimdViewT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_scatter(OffsetSimdViewT byte_offsets, ValuesSimdViewT vals,
            simd_mask<N / VS> mask, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  slm_scatter<T, N, VS>(byte_offsets.read(), vals.read(), mask, props);
}

/// template <int VS = 1, typename ValuesSimdViewT, typename OffsetSimdViewT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void slm_scatter(OffsetSimdViewT byte_offsets,
///              ValuesSimdViewT vals, PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param vals The vector of values to store.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    int VS = 1, typename ValuesSimdViewT, typename OffsetSimdViewT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_scatter(OffsetSimdViewT byte_offsets, ValuesSimdViewT vals,
            PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  slm_scatter<T, N, VS>(byte_offsets.read(), vals.read(), props);
}

/// template <int VS = 1, typename ValuesSimdViewT, typename OffsetT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void slm_scatter(simd<OffsetT, N / VS> byte_offsets,
///              ValuesSimdViewT vals, simd_mask<N / VS> mask,
///              PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param vals The vector of values to store.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    int VS = 1, typename ValuesSimdViewT, typename OffsetT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_scatter(simd<OffsetT, N / VS> byte_offsets, ValuesSimdViewT vals,
            simd_mask<N / VS> mask, PropertyListT props = {}) {
  slm_scatter<T, N, VS>(byte_offsets, vals.read(), mask, props);
}

/// template <int VS = 1, typename ValuesSimdViewT, typename OffsetT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void slm_scatter(simd<OffsetT, N / VS> byte_offsets,
///              ValuesSimdViewT vals, PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to Shared Local Memory
/// locations addressed by byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, (byte_offsets[i]) must be element size aligned.
/// @param vals The vector of values to store.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    int VS = 1, typename ValuesSimdViewT, typename OffsetT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_scatter(simd<OffsetT, N / VS> byte_offsets, ValuesSimdViewT vals,
            PropertyListT props = {}) {
  slm_scatter<T, N, VS>(byte_offsets, vals.read(), props);
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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

/// simd<T, N> slm_block_load(uint32_t byte_offset,
///                           simd_mask<1> pred,
///                           PassThruSimdViewT pass_thru, props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename PassThruSimdViewT,
    typename T = PassThruSimdViewT::value_type::element_type,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<PassThruSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
slm_block_load(uint32_t offset, simd_mask<1> pred, PassThruSimdViewT pass_thru,
               PropertyListT props = {}) {
  return slm_block_load<T, N>(offset, pred, pass_thru.read(), props);
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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

/// simd<T, N> block_load(local_accessor lacc, uint32_t byte_offset,
///                       simd_mask<1> pred, PassThruSimdViewT pass_thru,
///                       props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename PassThruSimdViewT,
    typename T = PassThruSimdViewT::value_type::element_type,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<PassThruSimdViewT> &&
        detail::is_local_accessor_with_v<AccessorT,
                                         detail::accessor_mode_cap::can_read> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
block_load(AccessorT lacc, uint32_t byte_offset, simd_mask<1> pred,
           PassThruSimdViewT pass_thru, PropertyListT props = {}) {
  return block_load<T, N>(lacc, byte_offset, pred, pass_thru.read(), props);
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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

/// simd<T, N> block_load(local_accessor lacc,
///                       simd_mask<1> pred, PassThruSimdViewT pass_thru,
///                       props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename PassThruSimdViewT,
    typename T = PassThruSimdViewT::value_type::element_type,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<PassThruSimdViewT> &&
        detail::is_local_accessor_with_v<AccessorT,
                                         detail::accessor_mode_cap::can_read> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
block_load(AccessorT lacc, simd_mask<1> pred, PassThruSimdViewT pass_thru,
           PropertyListT props = {}) {
  return block_load<T, N>(lacc, pred, pass_thru.read(), props);
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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

/// void slm_block_store(uint32_t byte_offset, ValuesSimdViewT vals,
///                      simd_mask<1> pred, props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT,
    typename T = ValuesSimdViewT::value_type::element_type,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_block_store(uint32_t byte_offset, ValuesSimdViewT vals, simd_mask<1> pred,
                PropertyListT props = {}) {
  slm_block_store<T, N>(byte_offset, vals.read(), pred, props);
}

/// void slm_block_store(uint32_t byte_offset, ValuesSimdViewT vals,
///                      props = {});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT,
    typename T = ValuesSimdViewT::value_type::element_type,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
slm_block_store(uint32_t byte_offset, ValuesSimdViewT vals,
                PropertyListT props = {}) {
  slm_block_store<T, N>(byte_offset, vals.read(), props);
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(AccessorT lacc, simd<T, N> vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  slm_block_store<T, N>(detail::localAccessorToOffset(lacc), vals, pred, props);
}

/// void block_store(local_accessor lacc, uint32_t byte_offset,
///                  ValuesSimdViewT vals, props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT,
    typename T = ValuesSimdViewT::value_type::element_type,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(AccessorT lacc, uint32_t byte_offset, ValuesSimdViewT vals,
            PropertyListT props = {}) {
  block_store<T, N>(lacc, byte_offset, vals.read(), props);
}

/// void block_store(local_accessor lacc, ValuesSimdViewT vals,
///                  props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT,
    typename T = ValuesSimdViewT::value_type::element_type,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(AccessorT lacc, ValuesSimdViewT vals, PropertyListT props = {}) {
  block_store<T, N>(lacc, vals.read(), props);
}

/// void block_store(local_accessor lacc, uint32_t byte_offset,
///                  ValuesSimdViewT vals, simd_mask<1> pred, props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT,
    typename T = ValuesSimdViewT::value_type::element_type,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(AccessorT lacc, uint32_t byte_offset, ValuesSimdViewT vals,
            simd_mask<1> pred, PropertyListT props = {}) {
  block_store<T, N>(lacc, byte_offset, vals.read(), pred, props);
}

/// void block_store(local_accessor lacc, ValuesSimdViewT vals,
///                  simd_mask<1> pred, props={});
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
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
template <
    typename ValuesSimdViewT,
    typename T = ValuesSimdViewT::value_type::element_type,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
block_store(AccessorT lacc, ValuesSimdViewT vals, simd_mask<1> pred,
            PropertyListT props = {}) {
  block_store<T, N>(lacc, vals.read(), pred, props);
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
  if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
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
  if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
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
  // 2 byte, 8 byte types, non-power of two, and operations wider than
  // 32 are supported only by LSC.
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
///               simd_mask<N> mask = 1);                       // (lacc-au1-1)
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
  // Non-LSC atomic_update supports only 4-byte int vector operations with
  // 1,2,4,8,16,32 vector length. Non-LSC supports only 'store' for FP types.
  if constexpr (Op == atomic_op::fmin || Op == atomic_op::fmax ||
                Op == atomic_op::fadd || Op == atomic_op::fsub ||
                sizeof(T) != 4 || !__ESIMD_DNS::isPowerOf2(N, 32)) {
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
/// slm_atomic_update(simd<uint32_t, N> byte_offset,
///                   SrcSimdViewT src0,
///                   simd_mask<N> mask = 1)
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename SrcSimdViewT,
          typename T = SrcSimdViewT::value_type::element_type, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT>,
                             simd<T, N>>
slm_atomic_update(simd<uint32_t, N> byte_offset, SrcSimdViewT src0,
                  simd_mask<N> mask = 1) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return slm_atomic_update<Op, T, N>(byte_offset, src0.read(), mask);
}

/// simd<T, N>
/// slm_atomic_update(OffsetSimdViewT byte_offset,
///                   simd<T, N> src0,
///                   simd_mask<N> mask = 1)
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename OffsetSimdViewT, typename T, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1 &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT>,
                             simd<T, N>>
slm_atomic_update(OffsetSimdViewT byte_offset, simd<T, N> src0,
                  simd_mask<N> mask = 1) {
  static_assert(N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return slm_atomic_update<Op, T, N>(byte_offset.read(), src0, mask);
}

/// simd<T, N>
/// slm_atomic_update(OffsetSimdViewT byte_offset,
///                   SrcSimdViewT src0,
///                   simd_mask<N> mask = 1)
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename OffsetSimdViewT, typename SrcSimdViewT,
          typename T = SrcSimdViewT::value_type::element_type,
          int N = SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY()>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1 &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT> &&
                                 detail::is_simd_view_type_v<SrcSimdViewT>,
                             simd<T, N>>
slm_atomic_update(OffsetSimdViewT byte_offset, SrcSimdViewT src0,
                  simd_mask<N> mask = 1) {
  static_assert(N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return slm_atomic_update<Op, T, N>(byte_offset.read(), src0.read(), mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               simd<uint32_t, N> byte_offset,
///               simd<T, N> src0,
///               simd_mask<1> mask = 1);                       // (lacc-au1-1)
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

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               OffsetSimdViewT byte_offset,
///               simd<T, N> src0,
///               simd_mask<1> mask = 1);
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// local accessor \p lacc and a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename OffsetSimdViewT, typename T, int N,
          typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1 &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT> &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, OffsetSimdViewT byte_offset, simd<T, N> src0,
              simd_mask<N> mask = 1) {
  static_assert(N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(lacc, byte_offset.read(), src0, mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               simd<uint32_t, N> byte_offset,
///               SrcSimdViewT src0,
///               simd_mask<1> mask = 1);
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// local accessor \p lacc and a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename SrcSimdViewT,
          typename T = SrcSimdViewT::value_type::element_type, int N,
          typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset, SrcSimdViewT src0,
              simd_mask<N> mask = 1) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(lacc, byte_offset, src0.read(), mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0,
///               simd_mask<1> mask = 1);
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// local accessor \p lacc and a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT,
          typename T = SrcSimdViewT::value_type::element_type,
          int N = SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
          typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 1 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT> &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              simd_mask<N> mask = 1) {
  static_assert(N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(lacc, byte_offset.read(), src0.read(), mask);
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
///               simd_mask<1> mask = 1);                      // (lacc-au2-1)
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
  // Non-LSC atomic_update supports only 4-byte int vector operations with
  // 1,2,4,8,16,32 vector length.
  if constexpr (sizeof(T) != 4 || Op == atomic_op::fcmpxchg ||
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
/// slm_atomic_update(simd<uint32_t, N> byte_offset,
///                   SrcSimdViewT src0, simd<T, N> src1,
///                   simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand (new value).
/// @param src1 is the second atomic operand (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename SrcSimdViewT, typename T, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT>,
                             simd<T, N>>
slm_atomic_update(simd<uint32_t, N> byte_offset, SrcSimdViewT src0,
                  simd<T, N> src1, simd_mask<N> mask = 1) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset and src1 parameters.");
  return slm_atomic_update<Op, T, N>(byte_offset, src0.read(), src1, mask);
}

/// simd<T, N>
/// slm_atomic_update(simd<uint32_t, N> byte_offset,
///                   simd<T, N> src0, SrcSimdViewT src1,
///                   simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand (new value).
/// @param src1 is the second atomic operand (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename SrcSimdViewT, typename T, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT>,
                             simd<T, N>>
slm_atomic_update(simd<uint32_t, N> byte_offset, simd<T, N> src0,
                  SrcSimdViewT src1, simd_mask<N> mask = 1) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src1 parameter must correspond to the size of "
                "byte_offset and src0 parameters.");
  return slm_atomic_update<Op, T, N>(byte_offset, src0, src1.read(), mask);
}

/// simd<T, N>
/// slm_atomic_update(simd<uint32_t, N> byte_offset,
///                   SrcSimdViewT src0, SrcSimdViewT src1,
///                   simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand (new value).
/// @param src1 is the second atomic operand (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename SrcSimdViewT,
          typename T = SrcSimdViewT::value_type::element_type, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT>,
                             simd<T, N>>
slm_atomic_update(simd<uint32_t, N> byte_offset, SrcSimdViewT src0,
                  SrcSimdViewT src1, simd_mask<N> mask = 1) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
      "Size of src1 and src0 parameters must correspond to the size of "
      "byte_offset parameter.");
  return slm_atomic_update<Op, T, N>(byte_offset, src0.read(), src1.read(),
                                     mask);
}

/// simd<T, N>
/// slm_atomic_update(OffsetSimdViewT byte_offset,
///                   simd<T, N> src0, simd<T, N> src1,
///                   simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand (new value).
/// @param src1 is the second atomic operand (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename OffsetSimdViewT, typename T, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT>,
                             simd<T, N>>
slm_atomic_update(OffsetSimdViewT byte_offset, simd<T, N> src0, simd<T, N> src1,
                  simd_mask<N> mask = 1) {
  static_assert(
      N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src1 and src0 parameters must correspond to the size of "
      "byte_offset parameter.");
  return slm_atomic_update<Op, T, N>(byte_offset.read(), src0, src1, mask);
}

/// simd<T, N>
/// slm_atomic_update(OffsetSimdViewT byte_offset,
///                   SrcSimdViewT src0, simd<T, N> src1,
///                   simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand (new value).
/// @param src1 is the second atomic operand (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename OffsetSimdViewT, typename SrcSimdViewT,
          typename T, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT>,
                             simd<T, N>>
slm_atomic_update(OffsetSimdViewT byte_offset, SrcSimdViewT src0,
                  simd<T, N> src1, simd_mask<N> mask = 1) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
                    N == OffsetSimdViewT::getSizeX() *
                             OffsetSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset and src1 parameters.");
  return slm_atomic_update<Op, T, N>(byte_offset.read(), src0.read(), src1,
                                     mask);
}

/// simd<T, N>
/// slm_atomic_update(OffsetSimdViewT byte_offset,
///                   simd<T, N> src0, SrcSimdViewT src1,
///                   simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand (new value).
/// @param src1 is the second atomic operand (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename OffsetSimdViewT, typename SrcSimdViewT,
          typename T, int N>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT>,
                             simd<T, N>>
slm_atomic_update(OffsetSimdViewT byte_offset, simd<T, N> src0,
                  SrcSimdViewT src1, simd_mask<N> mask = 1) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
                    N == OffsetSimdViewT::getSizeX() *
                             OffsetSimdViewT::getSizeY(),
                "Size of src1 parameter must correspond to the size of "
                "byte_offset and src0 parameters.");
  return slm_atomic_update<Op, T, N>(byte_offset.read(), src0, src1.read(),
                                     mask);
}

/// simd<T, N>
/// slm_atomic_update(OffsetSimdViewT byte_offset,
///                   SrcSimdViewT src0, SrcSimdViewT src1,
///                   simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Atomically updates \c N memory locations in SLM indicated by
/// a vector of offsets, and returns a vector of old
/// values found at the memory locations before update.
/// @tparam Op The atomic operation.
/// @param byte_offset The vector of 32-bit offsets.
/// @param src0 is the first atomic operand (new value).
/// @param src1 is the second atomic operand (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename OffsetSimdViewT, typename SrcSimdViewT,
          typename T = SrcSimdViewT::value_type::element_type,
          int N = SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY()>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT>,
                             simd<T, N>>
slm_atomic_update(OffsetSimdViewT byte_offset, SrcSimdViewT src0,
                  SrcSimdViewT src1, simd_mask<N> mask = 1) {
  static_assert(
      N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src1 and src0 parameters must correspond to the size of "
      "byte_offset parameter.");
  return slm_atomic_update<Op, T, N>(byte_offset.read(), src0, src1, mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               simd<uint32_t, N> byte_offset,
///               simd<T, N> src0,
///               simd<T, N> src1,
///               simd_mask<N> mask = 1);                      // (lacc-au2-1)
template <atomic_op Op, typename T, int N, typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset, simd<T, N> src0,
              simd<T, N> src1, simd_mask<N> mask = 1) {
  byte_offset += detail::localAccessorToOffset(lacc);
  return slm_atomic_update<Op, T, N>(byte_offset, src0, src1, mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               simd<uint32_t, N> byte_offset,
///               SrcSimdViewT src0,
///               simd<T, N> src1,
///               simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
template <atomic_op Op, typename SrcSimdViewT, typename T, int N,
          typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset, SrcSimdViewT src0,
              simd<T, N> src1, simd_mask<N> mask = 1) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset and src1 parameters.");
  return atomic_update<Op, T, N>(lacc, byte_offset, src0.read(), src1, mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               simd<uint32_t, N> byte_offset,
///               simd<T, N> src0,
///               SrcSimdViewT src1,
///               simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
template <atomic_op Op, typename SrcSimdViewT, typename T, int N,
          typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset, simd<T, N> src0,
              SrcSimdViewT src1, simd_mask<N> mask = 1) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src1 parameter must correspond to the size of "
                "byte_offset and src0 parameters.");
  return atomic_update<Op, T, N>(lacc, byte_offset, src0, src1.read(), mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               simd<uint32_t, N> byte_offset,
///               SrcSimdViewT src0,
///               SrcSimdViewT src1,
///               simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
template <atomic_op Op, typename SrcSimdViewT,
          typename T = SrcSimdViewT::value_type::element_type, int N,
          typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, simd<uint32_t, N> byte_offset, SrcSimdViewT src0,
              SrcSimdViewT src1, simd_mask<N> mask = 1) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
      "Size of src1 and src0 parameters must correspond to the size of "
      "byte_offset parameter.");
  return atomic_update<Op, T, N>(lacc, byte_offset, src0.read(), src1.read(),
                                 mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               OffsetSimdViewT byte_offset,
///               simd<T, N> src0,
///               simd<T, N> src1,
///               simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
template <atomic_op Op, typename OffsetSimdViewT, typename T, int N,
          typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT> &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, OffsetSimdViewT byte_offset, simd<T, N> src0,
              simd<T, N> src1, simd_mask<N> mask = 1) {
  static_assert(
      N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src1 and src0 parameters must correspond to the size of "
      "byte_offset parameter.");
  return atomic_update<Op, T, N>(lacc, byte_offset.read(), src0, src1, mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0,
///               simd<T, N> src1,
///               simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
template <atomic_op Op, typename OffsetSimdViewT, typename SrcSimdViewT,
          typename T, int N, typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT> &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              simd<T, N> src1, simd_mask<N> mask = 1) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset and src1 parameters.");
  return atomic_update<Op, T, N>(lacc, byte_offset.read(), src0.read(), src1,
                                 mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               OffsetSimdViewT byte_offset,
///               simd<T, N> src0,
///               SrcSimdViewT src1,
///               simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
template <atomic_op Op, typename OffsetSimdViewT, typename SrcSimdViewT,
          typename T, int N, typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT> &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, OffsetSimdViewT byte_offset, simd<T, N> src0,
              SrcSimdViewT src1, simd_mask<N> mask = 1) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
                    N == OffsetSimdViewT::getSizeX() *
                             OffsetSimdViewT::getSizeY(),
                "Size of src1 parameter must correspond to the size of "
                "byte_offset and src0 parameters.");
  return atomic_update<Op, T, N>(lacc, byte_offset.read(), src0, src1.read(),
                                 mask);
}

/// simd<T, N>
/// atomic_update(local_accessor lacc,
///               OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0,
///               SrcSimdViewT src1,
///               simd_mask<N> mask = 1);
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
template <atomic_op Op, typename OffsetSimdViewT, typename SrcSimdViewT,
          typename T = SrcSimdViewT::value_type::element_type,
          int N = SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
          typename AccessorT>
__ESIMD_API std::enable_if_t<__ESIMD_DNS::get_num_args<Op>() == 2 &&
                                 detail::is_simd_view_type_v<SrcSimdViewT> &&
                                 detail::is_simd_view_type_v<OffsetSimdViewT> &&
                                 __ESIMD_DNS::is_rw_local_accessor_v<AccessorT>,
                             simd<T, N>>
atomic_update(AccessorT lacc, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              SrcSimdViewT src1, simd_mask<N> mask = 1) {
  static_assert(
      N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src1 and src0 parameters must correspond to the size of "
      "byte_offset parameter.");
  return atomic_update<Op, T, N>(lacc, byte_offset.read(), src0.read(),
                                 src1.read(), mask);
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
/// @tparam PropertyListT is the properties with optional cache hints.
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param pred is predicates.
///
template <atomic_op Op, typename T, int N, lsc_data_size DS,
          typename PropertyListT, typename Toffset>
__ESIMD_API std::enable_if_t<get_num_args<Op>() == 0, simd<T, N>>
atomic_update_impl(T *p, simd<Toffset, N> offsets, simd_mask<N> pred) {
  static_assert(sizeof(T) > 1, "Unsupported data type");
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  check_atomic<Op, T, N, 0, /*IsLSC*/ true>();
  check_lsc_data_size<T, DS>();
  check_cache_hints<cache_action::atomic, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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
/// @tparam PropertyListT is the properties with optional cache hints.
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
template <atomic_op Op, typename T, int N, lsc_data_size DS,
          typename PropertyListT, typename Toffset>
__ESIMD_API std::enable_if_t<get_num_args<Op>() == 1, simd<T, N>>
atomic_update_impl(T *p, simd<Toffset, N> offsets, simd<T, N> src0,
                   simd_mask<N> pred) {
  static_assert(sizeof(T) > 1, "Unsupported data type");
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 1, /*IsLSC*/ true>();
  check_cache_hints<cache_action::atomic, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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
/// @tparam PropertyListT is the properties with optional cache hints.
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand (expected value).
/// @param src1 is the second atomic operand (new value).
/// @param pred predicates.
///
template <atomic_op Op, typename T, int N, lsc_data_size DS,
          typename PropertyListT, typename Toffset>
__ESIMD_API std::enable_if_t<get_num_args<Op>() == 2, simd<T, N>>
atomic_update_impl(T *p, simd<Toffset, N> offsets, simd<T, N> src0,
                   simd<T, N> src1, simd_mask<N> pred) {
  static_assert(sizeof(T) > 1, "Unsupported data type");
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 2, /*IsLSC*/ true>();
  check_cache_hints<cache_action::atomic, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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
/// @tparam PropertyListT is the properties with optional cache hints.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param byte_offsets is the zero-based offsets.
/// @param pred is predicates.
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N,
          lsc_data_size DS = lsc_data_size::default_size,
          typename PropertyListT, typename AccessorTy, typename Toffset>
__ESIMD_API
    std::enable_if_t<get_num_args<Op>() == 0 &&
                         __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy>,
                     simd<T, N>>
    atomic_update_impl(AccessorTy acc, simd<Toffset, N> byte_offsets,
                       simd_mask<N> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update_impl<Op, T, N, DS, PropertyListT>(
      accessorToPointer<T>(acc), byte_offsets, pred);
#else
  static_assert(sizeof(T) > 1, "Unsupported data type");
  static_assert(std::is_integral_v<Toffset> && sizeof(Toffset) == 4,
                "Unsupported offset type");
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 0, /*IsLSC*/ true>();
  check_cache_hints<cache_action::atomic, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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
/// @tparam PropertyListT is the properties with optional cache hints.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param byte_offset is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, lsc_data_size DS,
          typename PropertyListT, typename AccessorTy, typename Toffset>
__ESIMD_API
    std::enable_if_t<get_num_args<Op>() == 1 &&
                         __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy>,
                     simd<T, N>>
    atomic_update_impl(AccessorTy acc, simd<Toffset, N> byte_offset,
                       simd<T, N> src0, simd_mask<N> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update_impl<Op, T, N, DS, PropertyListT>(
      accessorToPointer<T>(acc), byte_offset, src0, pred);
#else
  static_assert(sizeof(T) > 1, "Unsupported data type");
  static_assert(std::is_integral_v<Toffset> && sizeof(Toffset) == 4,
                "Unsupported offset type");
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 1, /*IsLSC*/ true>();
  check_cache_hints<cache_action::atomic, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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
/// @tparam PropertyListT is the properties with optional cache hints.
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param byte_offset is the zero-based offsets.
/// @param src0 is the first atomic operand (expected value).
/// @param src1 is the second atomic operand (new value).
/// @param pred is predicates.
///
/// @return A vector of the old values at the memory locations before the
///   update.
template <atomic_op Op, typename T, int N, lsc_data_size DS,
          typename PropertyListT, typename AccessorTy, typename Toffset>
__ESIMD_API
    std::enable_if_t<get_num_args<Op>() == 2 &&
                         __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy>,
                     simd<T, N>>
    atomic_update_impl(AccessorTy acc, simd<Toffset, N> byte_offset,
                       simd<T, N> src0, simd<T, N> src1, simd_mask<N> pred) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update_impl<Op, T, N, DS, PropertyListT>(
      __ESIMD_DNS::accessorToPointer<T>(acc), byte_offset, src0, src1, pred);
#else
  static_assert(std::is_integral_v<Toffset> && sizeof(Toffset) == 4,
                "Unsupported offset type");
  check_lsc_vector_size<1>();
  check_lsc_data_size<T, DS>();
  check_atomic<Op, T, N, 2, /*IsLSC*/ true>();
  check_cache_hints<cache_action::atomic, PropertyListT>();
  constexpr auto L1H = getCacheHintForIntrin<PropertyListT, cache_level::L1>();
  constexpr auto L2H = getCacheHintForIntrin<PropertyListT, cache_level::L2>();
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
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               simd_mask<N> mask, props = {});               /// (usm-au0-3)
/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, simd_mask<N> mask,
              PropertyListT props = {}) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");

  if constexpr (detail::has_cache_hints<PropertyListT>() ||
                !__ESIMD_DNS::isPowerOf2(N, 32) || sizeof(T) < 4) {
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, PropertyListT, Toffset>(
        p, byte_offset, mask);
  } else if constexpr (N == 16 || N == 32) {
    // TODO: In fact GPU BE supports legalization for any N, even for
    // non-power-of-2, but it is implemented with an error now. For example,
    // N=17 is emulated as 2 calls (N=16 and N=1), while it must be 3 calls:
    // (N=8, N=8, N=1). I.e. Gen12 atomic instruction supports only N up to 8
    // and GPU thinks now it is up to 16.
    // Thus we emulate N=16 with 2 calls with N=8 each.
    // N=32 is emulated with 4 calls with N=8 each.
    // Task1: Remove the special-case emulation for N=16 and N=32 below when
    // GPU driver fixes the error.
    // Task2: remove the condition "!__ESIMD_DNS::isPowerOf2(N, 32)" above
    // and let svm.atomic for any N.

    simd<T, N> Res;
    for (int I = 0; I < N; I += 8) {
      simd_mask<8> Mask8 = mask.template select<8, 1>(I);
      simd<Toffset, 8> ByteOffset8 = byte_offset.template select<8, 1>(I);
      Res.template select<8, 1>(I) =
          atomic_update<Op, T, 8>(p, ByteOffset8, Mask8, props);
    }
    return Res;
  } else if constexpr (Op == atomic_op::load) {
    if constexpr (std::is_integral_v<T>) {
      return atomic_update<atomic_op::bit_or, T, N>(p, byte_offset,
                                                    simd<T, N>(0), mask, props);
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
template <
    atomic_op Op, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(p, byte_offset, mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT offsets, simd_mask<N> mask,
              PropertyListT props = {}) {
  return atomic_update<Op, T, N>(p, offsets.read(), mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT byte_offset, PropertyListT props = {}) {
  return atomic_update<Op, T, N>(p, byte_offset.read(), props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               props = {});
///
/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object without mask operand and allows the use without
/// specifying \c T and \c N template parameters.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc,
///   \c atomic_op::dec, or \c atomic_op::load.
/// @param p The USM pointer.
/// @param byte_offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
///   Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <
    atomic_op Op, typename OffsetSimdViewT, typename T,
    int N = OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT byte_offset, PropertyListT props = {}) {
  return atomic_update<Op, T, N>(p, byte_offset.read(), props);
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
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               simd<T, N> src0,
///               simd_mask<N> mask, props = {});                // (usm-au1-3)
/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, simd<T, N> src0,
              simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");

  // Auto-convert FP atomics to LSC version.
  if constexpr (detail::has_cache_hints<PropertyListT>() ||
                (Op == atomic_op::fmin) || (Op == atomic_op::fmax) ||
                (Op == atomic_op::fadd) || (Op == atomic_op::fsub) ||
                !__ESIMD_DNS::isPowerOf2(N, 32) || sizeof(T) < 4) {
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, PropertyListT, Toffset>(
        p, byte_offset, src0, mask);
  } else if constexpr (N == 16 || N == 32) {
    // TODO: In fact GPU BE supports legalization for any N, even for
    // non-power-of-2, but it is implemented with an error now. For example,
    // N=17 is emulated as 2 calls (N=16 and N=1), while it must be 3 calls:
    // (N=8, N=8, N=1). I.e. Gen12 atomic instruction supports only N up to 8
    // and GPU thinks now it is up to 16.
    // Thus we emulate N=16 with 2 calls with N=8 each.
    // N=32 is emulated with 4 calls with N=8 each.
    // Task1: Remove the special-case emulation for N=16 and N=32 below when
    // GPU driver fixes the error.
    // Task2: remove the condition "!__ESIMD_DNS::isPowerOf2(N, 32)" above
    // and let svm.atomic for any N.
    simd<T, N> Res;
    for (int I = 0; I < N; I += 8) {
      simd_mask<8> Mask8 = mask.template select<8, 1>(I);
      simd<Toffset, 8> ByteOffset8 = byte_offset.template select<8, 1>(I);
      simd<T, 8> Src08 = src0.template select<8, 1>(I);
      Res.template select<8, 1>(I) =
          atomic_update<Op, T, 8>(p, ByteOffset8, Src08, Mask8, props);
    }
    return Res;
  } else if constexpr (Op == atomic_op::store) {
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

/// simd<T, N>
/// atomic_update(T *ptr, simd<Toffset, N> byte_offset,
///               SrcSimdViewT src0, simd_mask<N> mask, props = {});
///
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 1 additional argument.
/// A variation of \c atomic_update API with \c src0 represented as
/// \c simd_view object and allows the use without specifying \c T and \c N
/// template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin, \c atomic_op::fadd, \c
/// atomic_op::fsub, \c atomic_op::store.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(p, byte_offset, src0.read(), mask, props);
}

/// simd<T, N>
/// atomic_update(T *ptr, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, props = {});                  // (usm-au1-2)

/// A variation of \c atomic_update API without mask operand.

/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
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
template <
    atomic_op Op, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
/// atomic_update(T *ptr, simd<Toffset, N> byte_offset,
///               SrcSimdViewT src0, props = {});

/// A variation of \c atomic_update API with \c src0 represented as
/// \c simd_view object and no mask operand and allows the use without
/// specifying \c T and \c N template parameters.

/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(p, byte_offset, src0.read(), props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT offsets, simd<T, N> src0, simd_mask<N> mask,
              PropertyListT props = {}) {
  return atomic_update<Op, T, N>(p, offsets.read(), src0, mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0,
///               simd_mask<N> mask, props = {});
///
/// A variation of \c atomic_update API with \c byte_offset and \c src0
/// represented as \c simd_view object and allows the use without specifying \c
/// T and \c N template parameters.
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
template <
    atomic_op Op, typename OffsetSimdViewT, typename SrcSimdViewT, typename T,
    int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT offsets, SrcSimdViewT src0,
              simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(
      N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY() &&
          N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
      "Size of src0 and offsets parameters must correspond to the size of "
      "mask parameter.");
  return atomic_update<Op, T, N>(p, offsets.read(), src0.read(), mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT offsets, simd<T, N> src0,
              PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(p, offsets.read(), src0, mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0,
///               props = {});
///
/// A variation of \c atomic_update API with \c byte_offset represented as
/// \c simd_view object and no mask operand and allows the use without
/// specifying \c T and \c N template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
/// @param p The USM pointer.
/// @param byte_offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// @param src0 The additional argument.
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used. Other properties are
///   ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <
    atomic_op Op, typename OffsetSimdViewT, typename SrcSimdViewT, typename T,
    int N = SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT offsets, SrcSimdViewT src0,
              PropertyListT props = {}) {
  static_assert(N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "offsets parameter.");
  return atomic_update<Op, T, N>(p, offsets.read(), src0.read(), props);
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
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {})                // (usm-au2-3)
/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, simd<T, N> src0,
              simd<T, N> src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");

  // Use LSC atomic when cache hints are present, FP atomics is used,
  // non-power of two length is used, or operation width greater than 32, or the
  // data size is less than 4 bytes.
  if constexpr (detail::has_cache_hints<PropertyListT>() ||
                Op == atomic_op::fcmpxchg || !__ESIMD_DNS::isPowerOf2(N, 32) ||
                sizeof(T) < 4) {
    // 2-argument lsc_atomic_update arguments order matches the standard one -
    // expected value first, then new value. But atomic_update uses reverse
    // order, hence the src1/src0 swap.
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, PropertyListT, Toffset>(
        p, byte_offset, src1, src0, mask);
  } else if constexpr (N == 16 || N == 32) {
    // TODO: In fact GPU BE supports legalization for any N, even for
    // non-power-of-2, but it is implemented with an error now. For example,
    // N=17 is emulated as 2 calls (N=16 and N=1), while it must be 3 calls:
    // (N=8, N=8, N=1). I.e. Gen12 atomic instruction supports only N up to 8
    // and GPU thinks now it is up to 16.
    // Thus we emulate N=16 with 2 calls with N=8 each.
    // N=32 is emulated with 4 calls with N=8 each.
    // Task1: Remove the special-case emulation for N=16 and N=32 below when
    // GPU driver fixes the error.
    // Task2: remove the condition "!__ESIMD_DNS::isPowerOf2(N, 32)" above
    // and let svm.atomic for any N.
    simd<T, N> Res;
    for (int I = 0; I < N; I += 8) {
      simd_mask<8> Mask8 = mask.template select<8, 1>(I);
      simd<Toffset, 8> ByteOffset8 = byte_offset.template select<8, 1>(I);
      simd<T, 8> Src08 = src0.template select<8, 1>(I);
      simd<T, 8> Src18 = src1.template select<8, 1>(I);
      Res.template select<8, 1>(I) =
          atomic_update<Op, T, 8>(p, ByteOffset8, Src08, Src18, Mask8, props);
    }
    return Res;
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
///               SrcSimdViewT src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {});
///
/// A variation of \c atomic_update API with \c src0 represented as
/// \c simd_view object and allows the use without specifying \c T and \c N
/// template parameters.

/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              simd<T, N> src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(p, byte_offset, src0.read(), src1, mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, SrcSimdViewT src1,
///               simd_mask<N> mask, props = {});
///
/// A variation of \c atomic_update API with \c src1 represented as
/// \c simd_view object and allows the use without specifying \c T and \c N
/// template parameters.

/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, simd<T, N> src0,
              SrcSimdViewT src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src1 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(p, byte_offset, src0, src1.read(), mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               SrcSimdViewT src0, SrcSimdViewT src1,
///               simd_mask<N> mask, props = {});
///
/// A variation of \c atomic_update API with \c src0 and \c src1 represented as
/// \c simd_view object and allows the use without specifying \c T and \c N
/// template parameters.

/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              SrcSimdViewT src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
      "Size of src1 and src0 parameters must correspond to the size of "
      "byte_offset parameter.");
  return atomic_update<Op, T, N>(p, byte_offset, src0.read(), src1.read(), mask,
                                 props);
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
template <
    atomic_op Op, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               SrcSimdViewT src0, simd<T, N> src1,
///               props = {});
///
/// A variation of \c atomic_update API with \c src0 represented as
/// \c simd_view object without \c mask operand and allows the use without
/// specifying \c T and \c N template parameters.

/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              simd<T, N> src1, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(p, byte_offset, src0.read(), src1, props);
}

/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, SrcSimdViewT src1,
///               props = {});
///
/// A variation of \c atomic_update API with \c src1 represented as
/// \c simd_view object without \c mask operand and allows the use without
/// specifying \c T and \c N template parameters.

/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, simd<T, N> src0,
              SrcSimdViewT src1, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src1 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(p, byte_offset, src0, src1.read(), props);
}

/// simd<T, N>
/// atomic_update(T *p, simd<Toffset, N> byte_offset,
///               SrcSimdViewT src0, SrcSimdViewT src1,
///               props = {});
///
/// A variation of \c atomic_update API with \c src0 and \c src1 represented as
/// \c simd_view object without \c mask operand and allows the use without
/// specifying \c T and \c N template parameters.

/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(T *p, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              SrcSimdViewT src1, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
      "Size of src1 and src0 parameters must correspond to the size of "
      "byte_offset parameter.");
  return atomic_update<Op, T, N>(p, byte_offset, src0.read(), src1.read(),
                                 props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT byte_offset, simd<T, N> src0,
              simd<T, N> src1, simd_mask<N> mask, PropertyListT props = {}) {
  return atomic_update<Op, T, N>(p, byte_offset.read(), src0, src1, mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {})
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT, typename T,
    int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              simd<T, N> src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
          N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src0 and byte_offset parameters must correspond to the size of "
      "mask parameter.");
  return atomic_update<Op, T, N>(p, byte_offset.read(), src0.read(), src1, mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               simd<T, N> src0, SrcSimdViewT src1,
///               simd_mask<N> mask, props = {})
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT, typename T,
    int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT byte_offset, simd<T, N> src0,
              SrcSimdViewT src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
          N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src1 and byte_offset parameters must correspond to the size of "
      "mask parameter.");
  return atomic_update<Op, T, N>(p, byte_offset.read(), src0, src1.read(), mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0, SrcSimdViewT src1,
///               simd_mask<N> mask, props = {})
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT, typename T,
    int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              SrcSimdViewT src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
                    N == OffsetSimdViewT::getSizeX() *
                             OffsetSimdViewT::getSizeY(),
                "Size of src0, src1 and byte_offset parameters must correspond "
                "to the size of "
                "mask parameter.");
  return atomic_update<Op, T, N>(p, byte_offset.read(), src0.read(),
                                 src1.read(), mask, props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT byte_offset, simd<T, N> src0,
              simd<T, N> src1, PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(p, byte_offset.read(), src0, src1, mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0, simd<T, N> src1,
///               props = {})
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT, typename T,
    int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              simd<T, N> src1, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
          N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src0 and byte_offset parameters must correspond to the size of "
      "src1 parameter.");
  return atomic_update<Op, T, N>(p, byte_offset.read(), src0.read(), src1,
                                 props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               simd<T, N> src0, SrcSimdViewT src1,
///               props = {})
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT, typename T,
    int N,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT byte_offset, simd<T, N> src0,
              SrcSimdViewT src1, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
          N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src1 and byte_offset parameters must correspond to the size of "
      "src0 parameter.");
  return atomic_update<Op, T, N>(p, byte_offset.read(), src0, src1.read(),
                                 props);
}

/// simd<T, N>
/// atomic_update(T *p, OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0, SrcSimdViewT src1,
///               props = {})
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @param p The USM pointer.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param props The parameter 'props' specifies the optional compile-time
///   properties list. Only L1/L2 properties are used.
//    Other properties are ignored.
/// @return A vector of the old values at the memory locations before the
///   update.
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT, typename T,
    int N = SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(T *p, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              SrcSimdViewT src1, PropertyListT props = {}) {
  static_assert(N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of src0, src1 and byte_offset parameters must be equal.");
  return atomic_update<Op, T, N>(p, byte_offset.read(), src0.read(),
                                 src1.read(), props);
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
/// atomic_update(AccessorT acc, OffsetSimdViewT byte_offset,
///               simd_mask<N> mask, props = {});               /// (acc-au0-3)
/// simd<T, N>
/// atomic_update(AccessorT acc, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename Toffset, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");

  if constexpr (detail::has_cache_hints<PropertyListT>() ||
                !detail::isPowerOf2(N, 32) || sizeof(T) < 4) {
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, PropertyListT>(
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
/// atomic_update(AccessorT acc, simd<Toffset, N> byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename Toffset, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
/// atomic_update(AccessorT acc, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, simd_mask<N> mask,
              PropertyListT props = {}) {
  return atomic_update<Op, T, N>(acc, byte_offset.read(), mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorT acc, OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 0 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset,
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
////              OffsetSimdViewT byte_offset,
///               simd<T, N> src0,
///               simd_mask<N> mask, props = {});                // (acc-au1-3)
/// simd<T, N>
/// atomic_update(AccessorT acc,
///               OffsetSimdViewT byte_offset,
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

template <
    atomic_op Op, typename T, int N, typename Toffset, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert(sizeof(Toffset) == 4, "Only 32 bit offset is supported");
  // Auto-convert FP atomics to LSC version.
  if constexpr (detail::has_cache_hints<PropertyListT>() ||
                Op == atomic_op::fmin || Op == atomic_op::fmax ||
                Op == atomic_op::fadd || Op == atomic_op::fsub ||
                !__ESIMD_DNS::isPowerOf2(N, 32) || sizeof(T) < 4) {
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, PropertyListT>(
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
///               SrcSimdViewT src0, simd_mask<N> mask, props = {});
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets, and returns a vector of old values found at the
/// memory locations before update. The update operation has 1 additional
/// argument.
/// A variation of \c atomic_update API with \c src0 represented as
/// \c simd_view object and allows the use without
/// specifying \c T and \c N template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin, \c atomic_op::fadd, \c
/// atomic_op::fsub, \c atomic_op::store.
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

template <
    atomic_op Op, typename SrcSimdViewT, typename Toffset,
    typename T = SrcSimdViewT::value_type::element_type, int N,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset, src0.read(), mask, props);
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
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
/// @tparam T The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
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
template <
    atomic_op Op, typename T, int N, typename Toffset, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
/// atomic_update(AccessorT acc, SrcSimdViewT byte_offset,
///               simd<T, N> src0, props = {});
///
/// A variation of \c atomic_update API with no mask operand and \c src0
/// represented as \c simd_view object that allows the use without specifying
/// \c T and \c N template parameters.
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets, and returns a vector of old values found at the
/// memory locations before update. The update operation has 1 additional
/// argument.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param byte_offset The vector of 32-bit or 64-bit offsets in bytes.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename Toffset,
    typename T = SrcSimdViewT::value_type::element_type, int N,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset, src0.read(), props);
}

/// simd<T, N>
/// atomic_update(AccessorT acc,
///               OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, simd<T, N> src0,
              simd_mask<N> mask, PropertyListT props = {}) {
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0, mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorT acc,
///               OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0,
///               simd_mask<N> mask, props = {});
///
/// A variation of \c atomic_update API with \c byte_offset and \c src0
/// represented as \c simd_view object that allows the use without specifying
/// \c T and \c N template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT,
    typename T = SrcSimdViewT::value_type::element_type,
    int N = SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0.read(), mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(AccessorT acc,
///               OffsetSimdViewT byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, simd<T, N> src0,
              PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0, mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorT acc,
///               OffsetSimdViewT byte_offset,
///               SrcSimdViewT src0,
///               props = {});
///
/// A variation of \c atomic_update API with \c byte_offset and \c src0
/// represented as \c simd_view object and no \c mask operand that allows the
/// use without specifying \c T and \c N template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::fadd, \c atomic_op::fsub, \c atomic_op::store.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT,
    typename T = SrcSimdViewT::value_type::element_type,
    int N = SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 1 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              PropertyListT props = {}) {
  static_assert(N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0.read(), props);
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
/// atomic_update(AccessorTy acc, OffsetSimdViewT
///               byte_offset, simd<T, N> src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {});                // (acc-au2-3)
///
/// simd<T, N>
/// atomic_update(AccessorTy acc,
///               OffsetSimdViewT, byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename Toffset, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert(sizeof(Toffset) == 4, "Only 32 bit offset is supported");
  // Use LSC atomic when cache hints are present, FP atomics is used,
  // non-power of two length is used, operation width greater than 32, or the
  // data size is less than 4 bytes,
  if constexpr (detail::has_cache_hints<PropertyListT>() ||
                Op == atomic_op::fcmpxchg || !__ESIMD_DNS::isPowerOf2(N, 32) ||
                sizeof(T) < 4) {
    // 2-argument lsc_atomic_update arguments order matches the standard one -
    // expected value first, then new value. But atomic_update uses reverse
    // order, hence the src1/src0 swap.
    return detail::atomic_update_impl<
        Op, T, N, detail::lsc_data_size::default_size, PropertyListT>(
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
///               SrcSimdViewT src0, simd<T, N> src1,
//                simd_mask<N> mask,props = {});
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// A variation of \c atomic_update API with \c src0 represented as
/// \c simd_view object and allows the use without specifying \c T and \c N
/// template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 && std::is_integral_v<Toffset> &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              simd<T, N> src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset, src0.read(), src1, mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, SrcSimdViewT src1,
//                simd_mask<N> mask,props = {});
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// A variation of \c atomic_update API with \c src1 represented as
/// \c simd_view object and allows the use without specifying \c T and \c N
/// template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 && std::is_integral_v<Toffset> &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, simd<T, N> src0,
              SrcSimdViewT src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src1 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset, src0, src1.read(), mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset,
///               SrcSimdViewT src0, SrcSimdViewT src1,
//                simd_mask<N> mask,props = {});
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// A variation of \c atomic_update API with \c src0 and \c src1 represented as
/// \c simd_view object and allows the use without specifying \c T and \c N
/// template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT,
    typename T = SrcSimdViewT::value_type::element_type, int N,
    typename Toffset, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 && std::is_integral_v<Toffset> &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              SrcSimdViewT src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
      "Size of src0 and src1 parameters must correspond to the size of "
      "byte_offset parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset, src0.read(), src1.read(),
                                 mask, props);
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
template <
    atomic_op Op, typename T, int N, typename Toffset, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
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
/// atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset,
///               SrcSimdViewT src0, simd<T, N> src1,
//                props = {});
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// A variation of \c atomic_update API with no \c mask operand and with \c src0
/// represented as \c simd_view object and allows the use without specifying \c
/// T and \c N template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 && std::is_integral_v<Toffset> &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              simd<T, N> src1, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src0 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset, src0.read(), src1, props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset,
///               simd<T, N> src0, SrcSimdViewT src1,
//                props = {});
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// A variation of \c atomic_update API with no \c mask operand with \c src1
/// represented as \c simd_view object and allows the use without specifying \c
/// T and \c N template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename T, int N, typename Toffset,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 && std::is_integral_v<Toffset> &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, simd<T, N> src0,
              SrcSimdViewT src1, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
                "Size of src1 parameter must correspond to the size of "
                "byte_offset parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset, src0, src1.read(), props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset,
///               SrcSimdViewT src0, SrcSimdViewT src1,
//                props = {});
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// A variation of \c atomic_update API with no \c mask operand with \c src0 and
/// \c src1 represented as \c simd_view object and allows the use without
/// specifying \c T and \c N template parameters.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <
    atomic_op Op, typename SrcSimdViewT,
    typename T = SrcSimdViewT::value_type::element_type, int N,
    typename Toffset, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 && std::is_integral_v<Toffset> &&
        detail::is_simd_view_type_v<SrcSimdViewT> &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    simd<T, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> byte_offset, SrcSimdViewT src0,
              SrcSimdViewT src1, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
      "Size of src0 and src1 parameters must correspond to the size of "
      "byte_offset parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset, src0.read(), src1.read(),
                                 props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, OffsetSimdViewT
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, simd<T, N> src0,
              simd<T, N> src1, simd_mask<N> mask, PropertyListT props = {}) {
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0, src1, mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, OffsetSimdViewT
///               byte_offset, SrcSimdViewT src0, simd<T, N> src1,
///               simd_mask<N> mask, props = {});
///
/// A variation of \c atomic_update API with \c byte_offset and \c src0
/// represented as \c simd_view object and allows the use without specifying \c
/// T and \c N template parameters.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT, typename T,
    int N, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              simd<T, N> src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
          N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src0 and byte_offset parameters must correspond to the size of "
      "src1 parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0.read(), src1,
                                 mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, OffsetSimdViewT
///               byte_offset, simd<T, N> src0, SrcSimdViewT src1,
///               simd_mask<N> mask, props = {});
///
/// A variation of \c atomic_update API with \c byte_offset and \c src1
/// represented as \c simd_view object and allows the use without specifying \c
/// T and \c N template parameters.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT, typename T,
    int N, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, simd<T, N> src0,
              SrcSimdViewT src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
          N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src1 and byte_offset parameters must correspond to the size of "
      "src0 parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0, src1.read(),
                                 mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, OffsetSimdViewT
///               byte_offset, SrcSimdViewT src0, SrcSimdViewT src1,
///               simd_mask<N> mask, props = {});
///
/// A variation of \c atomic_update API with \c byte_offset, \c src0 and
/// \c src1 represented as \c simd_view object and allows the use without
/// specifying \c T and \c N template parameters.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT,
    typename T = SrcSimdViewT::value_type::element_type, int N,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              SrcSimdViewT src1, simd_mask<N> mask, PropertyListT props = {}) {
  static_assert(N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
                    N == OffsetSimdViewT::getSizeX() *
                             OffsetSimdViewT::getSizeY(),
                "Size of src0, src1 and byte_offset parameters must correspond "
                "to the size of "
                "mask parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0.read(),
                                 src1.read(), mask, props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc,
///               OffsetSimdViewT, byte_offset,
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
template <
    atomic_op Op, typename T, int N, typename OffsetSimdViewT,
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, simd<T, N> src0,
              simd<T, N> src1, PropertyListT props = {}) {
  simd_mask<N> mask = 1;
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0, src1, mask,
                                 props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, OffsetSimdViewT
///               byte_offset, SrcSimdViewT src0, simd<T, N> src1,
///               props = {});
///
/// A variation of \c atomic_update API with with no mask operand and \c
/// byte_offset and \c src0 represented as \c simd_view object and allows the
/// use without specifying \c T and \c N template parameters.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT, typename T,
    int N, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              simd<T, N> src1, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
          N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src0 and byte_offset parameters must correspond to the size of "
      "src1 parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0.read(), src1,
                                 props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, OffsetSimdViewT
///               byte_offset, simd<T, N> src0, SrcSimdViewT src1,
///               props = {});
///
/// A variation of \c atomic_update API with no mask operand and \c byte_offset
/// and \c src1 represented as \c simd_view object and allows the use without
/// specifying \c T and \c N template parameters.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT, typename T,
    int N, typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, simd<T, N> src0,
              SrcSimdViewT src1, PropertyListT props = {}) {
  static_assert(
      N == SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY() &&
          N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src1 and byte_offset parameters must correspond to the size of "
      "src0 parameter.");
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0, src1.read(),
                                 props);
}

/// simd<T, N>
/// atomic_update(AccessorTy acc, OffsetSimdViewT
///               byte_offset, SrcSimdViewT src0, SrcSimdViewT src1,
///               props = {});
///
/// A variation of \c atomic_update API with no mask operand and \c byte_offset,
/// \c src0 and \c src1 represented as \c simd_view object and allows the use
/// without specifying \c T and \c N template parameters.
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
template <
    atomic_op Op, typename SrcSimdViewT, typename OffsetSimdViewT,
    typename T = SrcSimdViewT::value_type::element_type,
    int N = SrcSimdViewT::getSizeX() * SrcSimdViewT::getSizeY(),
    typename AccessorTy,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    __ESIMD_DNS::get_num_args<Op>() == 2 &&
        __ESIMD_DNS::is_rw_device_accessor_v<AccessorTy> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_simd_view_type_v<OffsetSimdViewT> &&
        detail::is_simd_view_type_v<SrcSimdViewT>,
    simd<T, N>>
atomic_update(AccessorTy acc, OffsetSimdViewT byte_offset, SrcSimdViewT src0,
              SrcSimdViewT src1, PropertyListT props = {}) {
  static_assert(
      N == OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
      "Size of src0, src1 and byte_offset parameters must correspond.");
  return atomic_update<Op, T, N>(acc, byte_offset.read(), src0.read(),
                                 src1.read(), props);
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
  l2_flush_instructions = 0x2,
  /// Flush sampler (texture) cache.
  l2_flush_texture_data = 0x4,
  /// Flush constant cache.
  l2_flush_constant_data = 0x8,
  /// Flush constant cache.
  l2_flush_rw_data = 0x10,
  /// Issue SLM memory barrier only. If not set, the memory barrier is global.
  local_barrier = 0x20,
  /// Flush L1 read - only data cache.
  l1_flush_ro_data = 0x40,
};

/// esimd::fence sets the memory read/write order.
/// @tparam cntl A bitmask composed from \c fence_mask bits.
///
template <uint8_t cntl> __ESIMD_API void fence() { __esimd_fence(cntl); }

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
  static_assert(detail::isPowerOf2(N), "N must be a power of 2");

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
/// template <typename T, int N, int VS, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                 // (lacc-ga-1)
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                 // (lacc-ga-2)
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
///                   PropertyListT props = {});                 // (lacc-ga-3)
///
/// The next 3 functions are similar to (lacc-ga-1,2,3), but they don't have
/// the template parameter 'VS'. These functions are added for convenience and
/// to make it possible for the user to omit the template parameters T and N,
/// e.g. 'auto res = gather(acc, byte_offsets);
/// template <typename T, int N, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N> byte_offsets,
///                   simd_mask<N> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                 // (lacc-ga-4)
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N> byte_offsets,
///                   simd_mask<N> mask, PropertyListT props = {});//(lacc-ga-5)
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N> byte_offsets,
///                   PropertyListT props = {});                // (lacc-ga-6)
///
/// The next 3 functions are similar to (lacc-ga-1,2,3), but accept the
/// \p byte_offsets as a \c simd_view argument:
/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename OffsetSimdViewT,
//            typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                 // (lacc-ga-7)
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                 // (lacc-ga-8)
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   PropertyListT props = {});                 // (lacc-ga-9)

/// template <typename T, int N, int VS, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                 // (lacc-ga-1)
/// Supported platforms: DG2, PVC only - Temporary restriction for the variant
/// with pass_thru operand. The only exception: DG2/PVC is not required if
/// the __ESIMD_GATHER_SCATTER_LLVM_IR macro is used.
///
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the local accessor \p acc and byte offsets \p byte_offsets, and returns
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
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param pass_thru The vector pass through values.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
/// @return A vector of elements read.
template <
    typename T, int N, int VS, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
       simd_mask<N / VS> mask, simd<T, N> pass_thru, PropertyListT props = {}) {
  return slm_gather<T, N, VS>(byte_offsets +
                                  __ESIMD_DNS::localAccessorToOffset(acc),
                              mask, pass_thru, props);
}

/// template <typename T, int N, int VS, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                  // (lacc-ga-2)
/// Supported platforms: DG2, PVC in most cases. DG2/PVC is not required if
/// VS == 1 and the __ESIMD_GATHER_SCATTER_LLVM_IR macro is used or sizeof(T) <=
/// 4 and N = {1,2,4,8,16,32}
///
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the local accessor \p acc and byte offsets \p byte_offsets, and returns
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
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
/// @return A vector of elements read.
template <
    typename T, int N, int VS, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
       simd_mask<N / VS> mask, PropertyListT props = {}) {
  return slm_gather<T, N, VS>(
      byte_offsets + __ESIMD_DNS::localAccessorToOffset(acc), mask, props);
}

/// template <typename T, int N, int VS, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
///                   PropertyListT props = {});                  // (lacc-ga-3)
/// Supported platforms: DG2, PVC in most cases. DG2/PVC is not required if
/// VS == 1 and the __ESIMD_GATHER_SCATTER_LLVM_IR macro is used or sizeof(T) <=
/// 4 and N = {1,2,4,8,16,32}
///
/// Loads ("gathers") elements of the type 'T' from memory locations addressed
/// by the local accessor \p acc and byte offsets \p byte_offsets, and returns
/// the loaded elements.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to load.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
/// @return A vector of elements read.
template <
    typename T, int N, int VS, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
       PropertyListT props = {}) {
  return slm_gather<T, N, VS>(
      byte_offsets + __ESIMD_DNS::localAccessorToOffset(acc), props);
}

/// template <typename T, int N, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N> byte_offsets,
///                   simd_mask<N> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                  // (lacc-ga-4)
/// This function is identical to (lacc-ga-1) except that vector size is fixed
/// to 1. This variant is added for convenience and lets the user omit the
/// template arguments and call the function as 'gather(acc, byte_offsets, mask,
/// pass_thru);'.
// Dev note: the mask type was turned into template parameter `MaskT` to
// avoid the conflicts of this prototype with the old gather() function
// accepting a 'global_offset' parameter and avoid 'ambiguous call' errors
// for calls like this: gather(acc, byte_offsets_simd, 0, mask);
template <
    typename T, int N, typename AccessorT, typename MaskT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     std::is_same_v<MaskT, simd_mask<N>> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<uint32_t, N> byte_offsets, MaskT mask,
       simd<T, N> pass_thru, PropertyListT props = {}) {
  return slm_gather<T, N>(byte_offsets +
                              __ESIMD_DNS::localAccessorToOffset(acc),
                          mask, pass_thru, props);
}

/// template <typename T, int N, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N> byte_offsets,
///                   simd_mask<N> mask, PropertyListT props      // (lacc-ga-5)
/// This function is identical to (lacc-ga-2) except that vector size is fixed
/// to 1. This variant is added for convenience and let user omit the template
/// arguments and call the function as 'gather(acc, byte_offsets, mask);'.
// Dev note: the mask type was turned into template parameter `MaskT` to
// avoid the conflicts of this prototype with the old gather() function
// accepting a 'global_offset' parameter and avoid 'ambiguous call' errors
// for calls like this: gather(acc, byte_offsets_simd, 0);
template <
    typename T, int N, typename AccessorT, typename MaskT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     std::is_same_v<MaskT, simd_mask<N>> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<uint32_t, N> byte_offsets, MaskT mask,
       PropertyListT props = {}) {
  return slm_gather<T, N>(
      byte_offsets + __ESIMD_DNS::localAccessorToOffset(acc), mask, props);
}

/// template <typename T, int N, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N> byte_offsets,
///                   PropertyListT props = {});                  // (lacc-ga-6)
/// This function is identical to (lacc-ga-3) except that vector size is fixed
/// to 1. This variant is added for convenience and let user omit the template
/// arguments and call the function as 'gather(acc, byte_offsets);'.
template <
    typename T, int N, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<uint32_t, N> byte_offsets,
       PropertyListT props = {}) {
  return slm_gather<T, N>(
      byte_offsets + __ESIMD_DNS::localAccessorToOffset(acc), props);
}

/// template <typename T, int N, int VS = 1,
///           typename OffsetSimdViewT,
//            typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});                  // (lacc-ga-7)
/// This function is identical to (lacc-ga-1) except that the \p byte_offsets
/// is represented as \c simd_view.
template <
    typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<OffsetSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       simd<T, N> pass_thru, PropertyListT props = {}) {
  return gather<T, N, VS>(acc, byte_offsets.read(), mask, pass_thru, props);
}

/// template <int VS, typename T, int N, typename OffsetSimdViewT,
//            typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});
/// This function is identical to (lacc-ga-1) except that the \p byte_offsets
/// is represented as \c simd_view.
template <
    int VS, typename T, int N, typename AccessorT, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<OffsetSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       simd<T, N> pass_thru, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of pass_thru parameter must correspond to the size of "
                "byte_offsets parameter.");
  return gather<T, N, VS>(acc, byte_offsets.read(), mask, pass_thru, props);
}

/// template <int VS = 1, typename AccessorT,
///    typename OffsetSimdViewT, typename PassThruSimdViewT,
///    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
///    typename T = PassThruSimdViewT::value_type::element_type,
///    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask, PassThruSimdViewT pass_thru,
///                   PropertyListT props = {});
/// This function is identical to (lacc-ga-1) except that the \p byte_offsets
/// and \p pass_thru are represented as \c simd_view.
template <
    int VS = 1, typename AccessorT, typename OffsetSimdViewT,
    typename PassThruSimdViewT,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename T = PassThruSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<OffsetSimdViewT> &&
     detail::is_simd_view_type_v<PassThruSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
       PassThruSimdViewT pass_thru, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of pass_thru parameter must correspond to the size of "
                "byte_offsets parameter.");
  return gather<T, N, VS>(acc, byte_offsets.read(), mask, pass_thru.read(),
                          props);
}

/// template <int VS = 1, typename AccessorT,
///    typename PassThruSimdViewT,
///    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
///    typename T = PassThruSimdViewT::value_type::element_type,
///    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>>
/// simd<T, N> gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask, simd<T, N> pass_thru,
///                   PropertyListT props = {});
/// This function is identical to (lacc-ga-1) except that the \p pass_thru
/// is represented as \c simd_view.
template <
    int VS = 1, typename AccessorT, typename PassThruSimdViewT,
    int N = PassThruSimdViewT::getSizeX() * PassThruSimdViewT::getSizeY(),
    typename T = PassThruSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<PassThruSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
       simd_mask<N / VS> mask, PassThruSimdViewT pass_thru,
       PropertyListT props = {}) {
  return gather<T, N, VS>(acc, byte_offsets, mask, pass_thru.read(), props);
}

/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename OffsetSimdViewT,
//            typename PropertyListT = empty_properties_t>
/// simd<T, N> gather(AccessorT acc, OffsetSimdViewT byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                  // (lacc-ga-8)
/// This function is identical to (lacc-ga-2) except that the \p byte_offsets
/// is represented as \c simd_view.
template <
    typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
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
///                   PropertyListT props = {});                  // (lacc-ga-9)
/// This function is identical to (lacc-ga-3) except that the \p byte_offsets
/// is represented as \c simd_view.
template <
    typename T, int N, int VS = 1, typename AccessorT, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    (detail::is_local_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
     detail::is_simd_view_type_v<OffsetSimdViewT> &&
     ext::oneapi::experimental::is_property_list_v<PropertyListT>),
    simd<T, N>>
gather(AccessorT acc, OffsetSimdViewT byte_offsets, PropertyListT props = {}) {
  return gather<T, N, VS>(acc, byte_offsets.read(), props);
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
    gather(AccessorTy acc, simd<uint32_t, N> offsets, uint32_t glob_offset,
           simd_mask<N> mask = 1) {
  return slm_gather<T, N>(
      offsets + glob_offset + __ESIMD_DNS::localAccessorToOffset(acc), mask);
}

/// Variant of scatter that uses local accessor as a parameter
/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// void scatter(AccessorT acc,
///              simd<uint32_t, N / VS> byte_offsets,
///              simd<T, N> vals,
///              simd_mask<N / VS> mask,
///              PropertyListT props = {});                  // (lacc-sc-1)

/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// void scatter(AccessorT acc,
///              simd<uint32_t, N / VS> byte_offsets,
///              simd<T, N> vals,
///              PropertyListT props = {});                 // (lacc-sc-2)

/// The next two functions are similar to lacc-sc-{1,2} with the 'byte_offsets'
/// parameter represerented as 'simd_view'.

/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename OffsetSimdViewT,
///           typename PropertyListT = empty_properties_t>
/// void scatter(AccessorT acc,
///              OffsetSimdViewT byte_offsets,
///              simd<T, N> vals,
///              simd_mask<N / VS> mask,
///              PropertyListT props = {});                 // (lacc-sc-3)

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// void scatter(AccessorT acc,
///              OffsetSimdViewT byte_offsets,
///              simd<T, N> vals,
///              PropertyListT props = {});                // (lacc-sc-4)

/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// void scatter(AccessorT acc,
///              simd<uint32_t, N / VS> byte_offsets,
///              simd<T, N> vals,
///              simd_mask<N / VS> mask,
///              PropertyListT props = {});               // (lacc-sc-1)
///
/// Writes ("scatters") elements of the input vector to memory locations
/// addressed by the local accessor \p acc and byte offsets \p byte_offsets.
/// Access to any element's memory location can be disabled via
/// the input mask.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc The accessor to scatter to.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    typename T, int N, int VS = 1, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorT acc, simd<uint32_t, N / VS> byte_offsets, simd<T, N> vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  slm_scatter<T, N, VS>(byte_offsets + __ESIMD_DNS::localAccessorToOffset(acc),
                        vals, mask, props);
}

/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// void scatter(AccessorT acc,
///              simd<uint32_t, N / VS> byte_offsets,
///              simd<T, N> vals,
///              PropertyListT props = {});                 // (lacc-sc-2)
///
/// Writes ("scatters") elements of the input vector to memory locations
/// addressed by the local accessor \p acc and byte offsets \p byte_offsets.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc The accessor to scatter to.
/// @param byte_offsets the vector of 32-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    typename T, int N, int VS = 1, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorT acc, simd<uint32_t, N / VS> byte_offsets, simd<T, N> vals,
        PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  scatter<T, N, VS>(acc, byte_offsets, vals, Mask, props);
}

/// template <typename T, int N, int VS = 1, typename AccessorT,
///           typename OffsetSimdViewT,
///           typename PropertyListT = empty_properties_t>
/// void scatter(AccessorT acc,
///              OffsetSimdViewT byte_offsets,
///              simd<T, N> vals,
///              simd_mask<N / VS> mask,
///              PropertyListT props = {});                 // (lacc-sc-3)
///
/// Writes ("scatters") elements of the input vector to memory locations
/// addressed by the local accessor \p acc and byte offsets \p byte_offsets.
/// Access to any element's memory location can be disabled via the input mask.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc The accessor to scatter to.
/// @param byte_offsets the vector of 32-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorT acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  scatter<T, N, VS>(acc, byte_offsets.read(), vals, mask, props);
}

/// template <int VS, typename AccessorTy, typename T, int N,
/// typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	         simd_mask<N / VS> mask,
///              PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the local accessor \p acc and byte offsets \p byte_offsets. Access to any
/// element's memory location can be disabled via the input vector of predicates
/// \p mask. If mask[i] is unset, then the store to (acc + byte_offsets[i]) is
/// skipped.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS, typename AccessorTy, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorTy,
                                     detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(acc, byte_offsets.read(), vals, mask, props);
}

/// template <int VS, typename AccessorTy, typename T, int N,
/// typename OffsetSimdViewT, typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
/// 	         PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the local accessor \p acc and byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS, typename AccessorTy, typename T, int N, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorTy,
                                     detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(acc, byte_offsets.read(), vals, props);
}

/// template <int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
/// typename OffsetSimdViewT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets,
///              ValuesSimdViewT vals, simd_mask<N / VS> mask,
///              PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the local accessor \p acc and byte offsets \p byte_offsets. Access to any
/// element's memory location can be disabled via the input vector of predicates
/// \p mask. If mask[i] is unset, then the store to (acc + byte_offsets[i]) is
/// skipped.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
    typename OffsetSimdViewT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorTy,
                                     detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, ValuesSimdViewT vals,
        simd_mask<N / VS> mask, PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(acc, byte_offsets.read(), vals.read(), mask, props);
}

/// template <int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
/// typename OffsetSimdViewT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, OffsetSimdViewT byte_offsets,
///              ValuesSimdViewT vals, PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the local accessor \p acc and byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
    typename OffsetSimdViewT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorTy,
                                     detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, OffsetSimdViewT byte_offsets, ValuesSimdViewT vals,
        PropertyListT props = {}) {
  static_assert(N / VS ==
                    OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY(),
                "Size of vals parameter must correspond to the size of "
                "byte_offsets parameter.");
  scatter<T, N, VS>(acc, byte_offsets.read(), vals.read(), props);
}

/// template <int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
/// typename OffsetT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
///              ValuesSimdViewT vals, simd_mask<N / VS> mask,
///              PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the local accessor \p acc and byte offsets \p byte_offsets. Access to any
/// element's memory location can be disabled via the input vector of predicates
/// \p mask. If mask[i] is unset, then the store to (acc + byte_offsets[i]) is
/// skipped.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename AccessorTy, typename ValuesSimdViewT, typename OffsetT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorTy,
                                     detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
        ValuesSimdViewT vals, simd_mask<N / VS> mask,
        PropertyListT props = {}) {
  scatter<T, N, VS>(acc, byte_offsets, vals.read(), mask, props);
}

/// template <int VS = 1, typename AccessorTy, typename ValuesSimdViewT,
/// typename OffsetT,
/// int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
/// typename T = ValuesSimdViewT::value_type::element_type,
/// typename PropertyListT = empty_properties_t>
/// void scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
///              ValuesSimdViewT vals, PropertyListT props = {});
///
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Stores ("scatters") elements of the type 'T' to memory locations addressed
/// by the local accessor \p acc and byte offsets \p byte_offsets.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc Accessor referencing the data to store.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, (acc + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param props The optional compile-time properties. Only 'alignment'
/// and cache hint properties are used.
template <
    int VS = 1, typename AccessorTy, typename ValuesSimdViewT, typename OffsetT,
    int N = ValuesSimdViewT::getSizeX() * ValuesSimdViewT::getSizeY(),
    typename T = ValuesSimdViewT::value_type::element_type,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorTy,
                                     detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<ValuesSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorTy acc, simd<OffsetT, N / VS> byte_offsets,
        ValuesSimdViewT vals, PropertyListT props = {}) {
  scatter<T, N, VS>(acc, byte_offsets, vals.read(), props);
}

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// void scatter(AccessorT acc,
///              OffsetSimdViewT byte_offsets,
///              simd<T, N> vals,
///              PropertyListT props = {});                // (lacc-sc-4)
///
/// Writes ("scatters") elements of the input vector to memory locations
/// addressed by the local accessor \p acc and byte offsets \p byte_offsets.
/// @tparam T Element type.
/// @tparam N Number of elements to write.
/// @tparam VS Vector size. It can also be read as the number of writes per each
/// address. The parameter 'N' must be divisible by 'VS'. (VS > 1) is supported
/// only on DG2 and PVC and only for 4- and 8-byte element vectors.
/// @param acc The accessor to scatter to.
/// @param byte_offsets the vector of 32-bit offsets in bytes
/// represented as a 'simd_view' object.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// If the alignment property is not passed, then it is assumed that each
/// accessed address is aligned by element-size.
/// @param vals The vector to scatter.
/// @param props The optional compile-time properties. Only 'alignment'
/// property is used.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT,
                                     detail::accessor_mode_cap::can_write> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
scatter(AccessorT acc, OffsetSimdViewT byte_offsets, simd<T, N> vals,
        PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  scatter<T, N, VS>(acc, byte_offsets.read(), vals, Mask, props);
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
        uint32_t glob_offset, simd_mask<N> mask = 1) {
  slm_scatter<T, N>(offsets + glob_offset +
                        __ESIMD_DNS::localAccessorToOffset(acc),
                    vals, mask);
}

/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (usm-pf-1)
/// void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (usm-pf-2)
///
/// The next 2 functions are similar to the above and were added for
/// convenience. They assume the VS parameter is set to 1 and do not require
/// specifying the template parameters <T, N, VS> at function calls.
/// template <typename T, int N, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask,
///                   PropertyListT props = {});                   // (usm-pf-3)
/// void prefetch(const T *p, simd<OffsetT, N> byte_offsets,
///                   PropertyListT props = {});                   // (usm-pf-4)
/// The next 2 functions are variations of the first 2 above (usm-pf-1,2)
/// and were added only to support simd_view instead of simd for byte_offsets
/// operand.
/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-pf-5)
/// void prefetch(const T *p, OffsetSimdViewT byte_offsets,
///             PropertyListT props = {});                        // (usm-pf-6)
///
/// The next functions perform transposed 1-channel prefetch.
/// template <typename T, int VS = 1, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetT byte_offset, simd_mask<1> mask,
///                   PropertyListT props = {});                   // (usm-pf-7)
/// void prefetch(const T *p, OffsetT byte_offset,
///                   PropertyListT props = {});                   // (usm-pf-8)
/// template <typename T, int VS = 1,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd_mask<1> mask,
///                   PropertyListT props = {});                   // (usm-pf-9)
/// void prefetch(const T *p, PropertyListT props = {});           //(usm-pf-10)
///

/// template <typename T, int N, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   simd_mask<N / VS> mask,
///                   PropertyListT props = {});                   // (usm-pf-1)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, to the cache.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the prefetch from
/// (p + byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, int VS, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets, simd_mask<N / VS> mask,
         PropertyListT props = {}) {
  static_assert(N / VS >= 1 && N % VS == 0, "N must be divisible by VS");
  detail::prefetch_impl<T, VS, detail::lsc_data_size::default_size,
                        PropertyListT>(p, byte_offsets, mask);
}

/// template <typename T, int VS, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (usm-pf-2)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, into the cache.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, int VS, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, simd<OffsetT, N / VS> byte_offsets,
         PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  prefetch<T, N, VS>(p, byte_offsets, Mask, props);
}

/// template <typename T, int N, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd<OffsetT, N> byte_offsets,
///                   simd_mask<N> mask,
///                   PropertyListT props = {});                   // (usm-pf-3)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, to the cache.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the prefetch from
/// (p + byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, simd<OffsetT, N> byte_offsets, simd_mask<N> mask,
         PropertyListT props = {}) {
  constexpr int VS = 1;
  prefetch<T, N, VS>(p, byte_offsets, mask, props);
}

/// template <typename T, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd<OffsetT, N> byte_offsets,
///                   PropertyListT props = {});                   // (usm-pf-4)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations addressed
/// by the base pointer \p p and byte offsets \p byte_offsets, into the cache.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, simd<OffsetT, N> byte_offsets, PropertyListT props = {}) {
  constexpr int VS = 1;
  prefetch<T, N, VS>(p, byte_offsets, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (usm-pf-5)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations
/// addressed by the base pointer \p p and byte offsets \p byte_offsets to the
/// cache. Access to any element's memory location can be disabled via the input
/// vector of predicates \p mask. If mask[i] is unset, then the load from (p +
/// byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per
/// each address. The parameter 'N' must be divisible by 'VS'.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
         PropertyListT props = {}) {
  prefetch<T, N, VS>(p, byte_offsets.read(), mask, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetSimdViewT byte_offsets,
///             PropertyListT props = {});                      // (usm-pf-6)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations
/// addressed by the base pointer \p p and byte offsets \p byte_offsets to the
/// cache.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per
/// each address. The parameter 'N' must be divisible by 'VS'.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, OffsetSimdViewT byte_offsets, PropertyListT props = {}) {
  prefetch<T, N, VS>(p, byte_offsets.read(), props);
}

/// template <int VS = 1, typename T, int N, typename OffsetSimdViewT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {});
/// Supported platforms: DG2, PVC only.
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Prefetches elements of the type 'T' from memory locations
/// addressed by the base pointer \p p and byte offsets \p byte_offsets to the
/// cache. Access to any element's memory location can be disabled via the input
/// vector of predicates \p mask. If mask[i] is unset, then the load from (p +
/// byte_offsets[i]) is skipped.
/// @tparam VS Vector size. It can also be read as the number of reads per
/// each address. The parameter 'N' must be divisible by 'VS'.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    int VS = 1, typename OffsetSimdViewT, typename T,
    int N = OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY() * VS,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
         PropertyListT props = {}) {
  prefetch<T, N, VS>(p, byte_offsets.read(), mask, props);
}

/// template <int VS = 1, typename T, int N, typename OffsetSimdViewT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetSimdViewT byte_offsets,
///             PropertyListT props = {});
/// Supported platforms: DG2, PVC only.
/// Variation of the API that allows using \c simd_view without specifying
/// \c T and \c N template parameters.
/// Prefetches elements of the type 'T' from memory locations
/// addressed by the base pointer \p p and byte offsets \p byte_offsets to the
/// cache.
/// @tparam VS Vector size. It can also be read as the number of reads per
/// each address. The parameter 'N' must be divisible by 'VS'.
/// @param p The base address.
/// @param byte_offsets the vector of 32-bit or 64-bit offsets in bytes.
/// For each i, ((byte*)p + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    int VS = 1, typename OffsetSimdViewT, typename T,
    int N = OffsetSimdViewT::getSizeX() * OffsetSimdViewT::getSizeY() * VS,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, OffsetSimdViewT byte_offsets, PropertyListT props = {}) {
  prefetch<T, N, VS>(p, byte_offsets.read(), props);
}

/// template <typename T, int VS = 1, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetT byte_offset, simd_mask<1> mask,
///                   PropertyListT props = {});                   // (usm-pf-7)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from continuous memory location
/// addressed by the base pointer \p p, and offset \p byte_offset and the length
/// \p VS elements into the cache.
/// The maximum size of a prefetched block is 512 bytes for PVC and 256 bytes
/// for ACM (DG2). When sizeof(T) is equal to 8 the address must be 8-byte
/// aligned. Also, 8-byte alignment is required when the function has to load
/// more than 256-bytes. In all other cases 4-byte alignment is required. When T
/// is 1- or 2-byte type the data is treated as 4-byte data. Allowed \c VS
/// values for 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64. Allowed \c VS values
/// for 32 bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128. Allowed \c VS values for
/// 16 bit data are 2, 4, 8, 16, 32, 64, 128, 256. Allowed \c VS values for 8
/// bit data are 4, 8, 12, 16, 32, 64, 128, 256, 512.

/// @tparam T Element type.
/// @tparam VS Vector size. It specifies the number of consequent elements to
/// prefetch.
/// @param p The base address.
/// @param byte_offset offset from the base address.
/// @param mask The access mask. If it is set to 0, then the prefetch is
/// omitted.
/// @param props The optional compile-time properties.
template <
    typename T, int VS = 1, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<OffsetT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, OffsetT byte_offset, simd_mask<1> mask,
         PropertyListT props = {}) {
  detail::prefetch_impl<T, VS, detail::lsc_data_size::default_size,
                        PropertyListT>(p, byte_offset, mask);
}

/// template <typename T, int VS = 1, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, OffsetT byte_offset,
///                   PropertyListT props = {});                   // (usm-pf-8)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from continuous memory location
/// addressed by the base pointer \p p, and offset \p byte_offset and the length
/// \p VS elements into the cache.
/// @tparam T Element type.
/// @tparam VS Vector size. It specifies the number of consequent elements to
/// prefetch.
/// @param p The base address.
/// @param byte_offset offset from the base address
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int VS = 1, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<OffsetT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, OffsetT byte_offset, PropertyListT props = {}) {
  simd_mask<1> Mask = 1;
  prefetch<T, VS>(p, byte_offset, Mask, props);
}

/// template <typename T, int VS = 1,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, simd_mask<1> mask,
///                   PropertyListT props = {});                 //(usm-pf-9)
/// Supported platforms: DG2, PVC only.
///  Prefetches elements of the type 'T' from continuous memory location
///  addressed by the base pointer \p p
/// and the length \p VS elements into the cache.
/// @tparam T Element type.
/// @tparam VS Vector size. It specifies the number of consequent elements to
/// prefetch.
/// @param p The base address.
/// @param mask The access mask. If it is set to 0, then the prefetch is
/// omitted.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int VS = 1,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, simd_mask<1> mask, PropertyListT props = {}) {
  prefetch<T, VS>(p, 0, mask, props);
}

/// template <typename T, int VS = 1,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(const T *p, PropertyListT props = {});      // (usm-pf-10)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from continuous memory location
/// addressed by the base pointer \p p and the length \p VS into the cache.
/// @tparam T Element type.
/// @tparam VS Vector size. It specifies the number of consequent elements to
/// prefetch.
/// @param p The base address.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int VS = 1,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(const T *p, PropertyListT props = {}) {
  simd_mask<1> Mask = 1;
  prefetch<T, VS>(p, 0, Mask, props);
}

/// template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
///               simd_mask<N / VS> mask,
///               PropertyListT props = {});                   // (acc-pf-1)
/// void prefetch(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
///               PropertyListT props = {});                   // (acc-pf-2)
///
/// The next 2 functions are similar to the above and were added for
/// convenience. They assume the VS parameter is set to 1 and do not require
/// specifying the template parameters <T, N, VS> at function calls.
/// template <typename T, int N, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, simd<OffsetT, N> byte_offsets,
///               simd_mask<N> mask,
///               PropertyListT props = {});                   // (acc-pf-3)
/// void prefetch(AccessorT acc, simd<OffsetT, N> byte_offsets,
///               PropertyListT props = {});                   // (acc-pf-4)
/// The next 2 functions are variations of the first 2 above (acc-pf-1,2)
/// and were added only to support simd_view instead of simd for byte_offsets
/// operand.
/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename AccessorT, typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, OffsetSimdViewT byte_offsets,
///               simd_mask<N / VS> mask, PropertyListT props = {});//(acc-pf-5)
/// void prefetch(AccessorT acc, OffsetSimdViewT byte_offsets,
///               PropertyListT props = {});                        //(acc-pf-6)
///
/// The next functions perform transposed 1-channel prefetch.
/// template <typename T, int VS = 1, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, OffsetT byte_offset, simd_mask<1> mask,
///               PropertyListT props = {});                   // (acc-pf-7)
/// void prefetch(AccessorT acc, OffsetT byte_offset,
///               PropertyListT props = {});                   // (acc-pf-8)
/// template <typename T, int VS = 1, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, simd_mask<1> mask,
///               PropertyListT props = {});                   // (acc-pf-9)
/// void prefetch(AccessorT acc, PropertyListT props = {});       // (acc-pf-10)
///

/// template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
///               simd_mask<N / VS> mask,
///               PropertyListT props = {});                   // (acc-pf-1)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets, to the cache.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the prefetch from
/// (acc + byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'.
/// @param acc Accessor referencing the data to load.
/// @param byte_offsets the vector of offsets in bytes. If force stateless
/// memory is used the offsets can be up to 64 bit size, otherwise up to 32 bit
/// size. For each i, (acc + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, int VS, typename AccessorT, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
         simd_mask<N / VS> mask, PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  prefetch<T, N, VS>(detail::accessorToPointer<T>(acc), byte_offsets, mask,
                     props);
#else
  static_assert(N / VS >= 1 && N % VS == 0, "N must be divisible by VS");
  detail::prefetch_impl<T, VS, detail::lsc_data_size::default_size,
                        PropertyListT>(acc, byte_offsets, mask);
#endif // __ESIMD_FORCE_STATELESS_MEM
}

/// template <typename T, int N, int VS, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, simd<uint32_t, N / VS> byte_offsets,
///                   PropertyListT props = {});                   // (acc-pf-2)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets, into the cache.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per each
/// address. The parameter 'N' must be divisible by 'VS'.
/// @param acc Accessor referencing the data to load.
/// @param byte_offsets the vector of offsets in bytes. If force stateless
/// memory is used the offsets can be up to 64 bit size, otherwise up to 32 bit
/// size. For each i, (acc + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, int VS, typename AccessorT, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(AccessorT acc, simd<OffsetT, N / VS> byte_offsets,
         PropertyListT props = {}) {
  simd_mask<N / VS> Mask = 1;
  prefetch<T, N, VS>(acc, byte_offsets, Mask, props);
}

/// template <typename T, int N, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, simd<uint32_t, N> byte_offsets,
///                   simd_mask<N> mask,
///                   PropertyListT props = {});                   // (acc-pf-3)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets, to the cache.
/// Access to any element's memory location can be disabled via the input vector
/// of predicates \p mask. If mask[i] is unset, then the prefetch from
/// (acc + byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @param acc Accessor referencing the data to load.
/// @param byte_offsets the vector of offsets in bytes. If force stateless
/// memory is used the offsets can be up to 64 bit size, otherwise up to 32 bit
/// size. For each i, (acc + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, typename AccessorT, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(AccessorT acc, simd<OffsetT, N> byte_offsets, simd_mask<N> mask,
         PropertyListT props = {}) {
  constexpr int VS = 1;
  prefetch<T, N, VS>(acc, byte_offsets, mask, props);
}

/// template <typename T, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, simd<uint32_t, N> byte_offsets,
///                   PropertyListT props = {});                   // (acc-pf-4)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations addressed
/// by the accessor \p acc and byte offsets \p byte_offsets, into the cache.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @param acc Accessor referencing the data to load.
/// @param byte_offsets the vector of offsets in bytes. If force stateless
/// memory is used the offsets can be up to 64 bit size, otherwise up to 32 bit
/// size. For each i, (acc + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, typename AccessorT, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(AccessorT acc, simd<OffsetT, N> byte_offsets,
         PropertyListT props = {}) {
  constexpr int VS = 1;
  prefetch<T, N, VS>(acc, byte_offsets, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename AccessorT, typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, OffsetSimdViewT byte_offsets,
///             simd_mask<N / VS> mask, PropertyListT props = {}); // (acc-pf-5)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations
/// addressed by the accessor \p acc and byte offsets \p byte_offsets to the
/// cache. Access to any element's memory location can be disabled via the input
/// vector of predicates \p mask. If mask[i] is unset, then the load from (acc +
/// byte_offsets[i]) is skipped.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per
/// each address. The parameter 'N' must be divisible by 'VS'.
/// @param acc Accessor referencing the data to load.
/// @param byte_offsets the vector of offsets in bytes. If force stateless
/// memory is used the offsets can be up to 64 bit size, otherwise up to 32 bit
/// size. For each i, (acc + byte_offsets[i]) must be element size aligned.
/// @param mask The access mask.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(AccessorT acc, OffsetSimdViewT byte_offsets, simd_mask<N / VS> mask,
         PropertyListT props = {}) {
  prefetch<T, N, VS>(acc, byte_offsets.read(), mask, props);
}

/// template <typename T, int N, int VS = 1, typename OffsetSimdViewT,
///           typename AccessorT, typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, OffsetSimdViewT byte_offsets,
///             PropertyListT props = {});                      // (acc-pf-6)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from memory locations
/// addressed by the accessor \p acc and byte offsets \p byte_offsets to the
/// cache.
/// @tparam T Element type.
/// @tparam N Number of elements to read.
/// @tparam VS Vector size. It can also be read as the number of reads per
/// each address. The parameter 'N' must be divisible by 'VS'.
/// @param acc Accessor referencing the data to load.
/// @param byte_offsets the vector of offsets in bytes. If force stateless
/// memory is used the offsets can be up to 64 bit size, otherwise up to 32 bit
/// size. For each i, (acc + byte_offsets[i]) must be element size aligned.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int N, int VS = 1, typename OffsetSimdViewT, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
    detail::is_simd_view_type_v<OffsetSimdViewT> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(AccessorT acc, OffsetSimdViewT byte_offsets,
         PropertyListT props = {}) {
  prefetch<T, N, VS>(acc, byte_offsets.read(), props);
}

/// template <typename T, int VS = 1, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, uint32_t byte_offset, simd_mask<1> mask,
///                   PropertyListT props = {});                   // (acc-pf-7)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from continuous memory location
/// addressed by the accessor \p acc, and offset \p byte_offset and the length
/// \p VS elements into the cache.
/// The maximum size of prefetched block is 512 bytes for PVC and 256 bytes for
/// ACM (DG2). When sizeof(T) equal to 8 the address must be 8-byte aligned.
/// Also, 8-bytes alignment is required when the function has to load more than
/// 256-bytes. In all other cases 4-byte alignment is required. When T is 1- or
/// 2-byte type the data is treated as 4-byte data. Allowed \c VS values for
/// 64 bit data are 1, 2, 3, 4, 8, 16, 32, 64. Allowed \c VS values for 32
/// bit data are 1, 2, 3, 4, 8, 16, 32, 64, 128. Allowed \c VS values for 16
/// bit data are 2, 4, 8, 16, 32, 64, 128, 256. Allowed \c VS values for 8
/// bit data are 4, 8, 12, 16, 32, 64, 128, 256, 512.
/// @tparam T Element type.
/// @tparam VS Vector size. It specifies the number of consequent elements to
/// prefetch.
/// @param acc Accessor referencing the data to load.
/// @param byte_offset offset from the base address.
/// @param mask The access mask. If it is set to 0, then the prefetch is
/// omitted.
/// @param props The optional compile-time properties.
template <
    typename T, int VS = 1, typename AccessorT, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<OffsetT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(AccessorT acc, OffsetT byte_offset, simd_mask<1> mask,
         PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  prefetch<T, VS>(detail::accessorToPointer<T>(acc), byte_offset, mask, props);
#else
  detail::prefetch_impl<T, VS, detail::lsc_data_size::default_size,
                        PropertyListT>(acc, byte_offset, mask);
#endif // __ESIMD_FORCE_STATELESS_MEM
}

/// template <typename T, int VS = 1, typename AccessorT, typename OffsetT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, uint32_t byte_offset,
///                   PropertyListT props = {});                   // (acc-pf-8)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from continuous memory location
/// addressed by the accessor \p acc, and offset \p byte_offset and the length
/// \p VS elements into the cache.
/// @tparam T Element type.
/// @tparam VS Vector size. It specifies the number of consequent elements to
/// prefetch.
/// @param acc Accessor referencing the data to load.
/// @param byte_offset offset from the base address
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int VS = 1, typename AccessorT, typename OffsetT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<OffsetT> &&
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(AccessorT acc, OffsetT byte_offset, PropertyListT props = {}) {
  simd_mask<1> Mask = 1;
  prefetch<T, VS>(acc, byte_offset, Mask, props);
}

/// template <typename T, int VS = 1, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, simd_mask<1> mask,
///                   PropertyListT props = {});                 //(acc-pf-9)
/// Supported platforms: DG2, PVC only.
///  Prefetches elements of the type 'T' from continuous memory location
///  addressed by the accessor \p acc
/// and the length \p VS elements into the cache.
/// @tparam T Element type.
/// @tparam VS Vector size. It specifies the number of consequent elements to
/// prefetch.
/// @param acc Accessor referencing the data to load.
/// @param mask The access mask. If it is set to 0, then the prefetch is
/// omitted.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int VS = 1, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(AccessorT acc, simd_mask<1> mask, PropertyListT props = {}) {
  prefetch<T, VS>(acc, 0, mask, props);
}

/// template <typename T, int VS = 1, typename AccessorT,
///           typename PropertyListT = empty_properties_t>
/// void prefetch(AccessorT acc, PropertyListT props = {});      // (acc-pf-10)
/// Supported platforms: DG2, PVC only.
/// Prefetches elements of the type 'T' from continuous memory location
/// addressed by the accessor \p acc and the length \p VS into the cache.
/// @tparam T Element type.
/// @tparam VS Vector size. It specifies the number of consequent elements to
/// prefetch.
/// @param acc Accessor referencing the data to load.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
template <
    typename T, int VS = 1, typename AccessorT,
    typename PropertyListT = ext::oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT,
                                      detail::accessor_mode_cap::can_read> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch(AccessorT acc, PropertyListT props = {}) {
  simd_mask<1> Mask = 1;
  prefetch<T, VS>(acc, 0, Mask, props);
}

/// template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
///          bool Transposed = false, bool Transformed = false,
///          int N = detail::get_lsc_block_2d_data_size<
///              T, NBlocks, BlockHeight, BlockWidth, Transposed,
///              Transformed>(),
///          typename PropertyListT = empty_properties_t>
/// simd<T, N>
/// load_2d(const T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
///             unsigned SurfacePitch, int X, int Y,
///             PropertyListT props = {});
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
/// @tparam N is the data size
/// @param Ptr is the surface base address for this operation.
/// @param SurfaceWidth is the surface width minus 1 in bytes
/// @param SurfaceHeight is the surface height minus 1 in rows
/// @param SurfacePitch is the surface pitch minus 1 in bytes
/// @param X is zero based X-coordinate of the left upper rectangle corner in
/// number of elements.
/// @param Y is zero based Y-coordinate of the left upper rectangle corner in
/// rows.
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
/// @return is a vector of type T and size N, where N is
///  BlockWidth * BlockHeight * NBlocks, if not transformed;
///  otherwise,
///  N = roundUpNextMultiple(BlockHeight, 4 / sizeof(T)) *
///   getNextPowerOf2(BlockWidth) * NBlocks
///
template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          bool Transposed = false, bool Transformed = false,
          int N = detail::get_lsc_block_2d_data_size<
              T, NBlocks, BlockHeight, BlockWidth, Transposed, Transformed>(),
          typename PropertyListT = oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
load_2d(const T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
        unsigned SurfacePitch, int X, int Y, PropertyListT props = {}) {
  return detail::load_2d_impl<T, BlockWidth, BlockHeight, NBlocks, Transposed,
                              Transformed, PropertyListT>(
      Ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y);
}

/// template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
///          int N = detail::get_lsc_block_2d_data_size<
///              T, NBlocks, BlockHeight, BlockWidth, false, false>(),
///          typename PropertyListT = empty_properties_t>
/// void
/// prefetch_2d(const T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
///            unsigned SurfacePitch, int X, int Y, PropertyListT props = {});
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
/// @tparam N is the data size
/// @param Ptr is the surface base address for this operation.
/// @param SurfaceWidth is the surface width minus 1 in bytes
/// @param SurfaceHeight is the surface height minus 1 in rows
/// @param SurfacePitch is the surface pitch minus 1 in bytes
/// @param X is zero based X-coordinate of the left upper rectangle corner
/// in number of elements.
/// @param Y is zero based Y-coordinate of the left upper rectangle corner
/// in rows.
/// @param props The compile-time properties. Only cache hint
/// properties are used.
///
template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          int N = detail::get_lsc_block_2d_data_size<
              T, NBlocks, BlockHeight, BlockWidth, false /*Transposed*/,
              false /*Transformed*/>(),
          typename PropertyListT = oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
prefetch_2d(const T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
            unsigned SurfacePitch, int X, int Y, PropertyListT props = {}) {
  detail::prefetch_2d_impl<T, BlockWidth, BlockHeight, NBlocks, PropertyListT>(
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
///  N = BlockWidth * BlockHeight
/// @param props The optional compile-time properties. Only cache hint
/// properties are used.
///
template <typename T, int BlockWidth, int BlockHeight = 1,
          int N = detail::get_lsc_block_2d_data_size<
              T, 1u, BlockHeight, BlockWidth, false /*Transposed*/,
              false /*Transformed*/>(),
          typename PropertyListT = oneapi::experimental::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
store_2d(T *Ptr, unsigned SurfaceWidth, unsigned SurfaceHeight,
         unsigned SurfacePitch, int X, int Y, simd<T, N> Vals,
         PropertyListT props = {}) {
  detail::store_2d_impl<T, BlockWidth, BlockHeight, PropertyListT>(
      Ptr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y, Vals);
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
/// @tparam Fence - fence before signaling
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
template <bool Fence = true>
__ESIMD_API void
named_barrier_signal(uint8_t barrier_id, uint8_t producer_consumer_mode,
                     uint32_t num_producers, uint32_t num_consumers) {
  if constexpr (Fence)
    __esimd_fence(fence_mask::global_coherent_fence |
                  fence_mask::local_barrier);
  __esimd_nbarrier_arrive(barrier_id, producer_consumer_mode, num_producers,
                          num_consumers);
}

/// @} sycl_esimd_memory_nbarrier

/// @} sycl_esimd_memory

/// @cond EXCLUDE

namespace detail {
// -- Outlined implementations of simd_obj_impl class memory access APIs.

template <typename T, int N, class T1, class SFINAE>
template <int ChunkSize, typename PropertyListT>
std::enable_if_t<ext::oneapi::experimental::is_property_list_v<PropertyListT>>
simd_obj_impl<T, N, T1, SFINAE>::copy_from(
    const simd_obj_impl<T, N, T1, SFINAE>::element_type *Addr,
    PropertyListT) SYCL_ESIMD_FUNCTION {
  using UT = simd_obj_impl<T, N, T1, SFINAE>::element_type;
  constexpr unsigned Size = sizeof(T) * N;
  constexpr size_t Align =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(UT));

  constexpr unsigned BlockSize = OperandSize::OWORD * 8;
  constexpr unsigned NumBlocks = Size / BlockSize;
  constexpr unsigned RemSize = Size % BlockSize;

  if constexpr (Align >= OperandSize::DWORD && Size % OperandSize::OWORD == 0 &&
                detail::isPowerOf2(RemSize / OperandSize::OWORD)) {
    if constexpr (NumBlocks > 0) {
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      ForHelper<NumBlocks>::unroll([BlockN, Addr, this](unsigned Block) {
        select<BlockN, 1>(Block * BlockN) =
            block_load<UT, BlockN>(Addr + (Block * BlockN), PropertyListT{});
      });
    }
    if constexpr (RemSize > 0) {
      constexpr unsigned RemN = RemSize / sizeof(T);
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      select<RemN, 1>(NumBlocks * BlockN) =
          block_load<UT, RemN>(Addr + (NumBlocks * BlockN), PropertyListT{});
    }
  } else if constexpr (sizeof(T) == 8) {
    simd<int32_t, N * 2> BC(reinterpret_cast<const int32_t *>(Addr),
                            PropertyListT{});
    bit_cast_view<int32_t>() = BC;
  } else {
    constexpr unsigned NumChunks = N / ChunkSize;
    if constexpr (NumChunks > 0) {
      simd<uint32_t, ChunkSize> Offsets(0u, sizeof(T));
      ForHelper<NumChunks>::unroll([Addr, &Offsets, this](unsigned Block) {
        select<ChunkSize, 1>(Block * ChunkSize) = gather<UT, ChunkSize>(
            Addr + (Block * ChunkSize), Offsets, PropertyListT{});
      });
    }
    constexpr unsigned RemN = N % ChunkSize;
    if constexpr (RemN > 0) {
      if constexpr (RemN == 1) {
        select<1, 1>(NumChunks * ChunkSize) = Addr[NumChunks * ChunkSize];
      } else if constexpr (RemN == 8 || RemN == 16) {
        simd<uint32_t, RemN> Offsets(0u, sizeof(T));
        select<RemN, 1>(NumChunks * ChunkSize) = gather<UT, RemN>(
            Addr + (NumChunks * ChunkSize), Offsets, PropertyListT{});
      } else {
        constexpr int N1 = RemN < 8 ? 8 : RemN < 16 ? 16 : 32;
        simd_mask_type<N1> Pred(0);
        Pred.template select<RemN, 1>() = 1;
        simd<uint32_t, N1> Offsets(0u, sizeof(T));
        simd<UT, N1> Vals = gather<UT, N1>(Addr + (NumChunks * ChunkSize),
                                           Offsets, Pred, PropertyListT{});
        select<RemN, 1>(NumChunks * ChunkSize) =
            Vals.template select<RemN, 1>();
      }
    }
  }
}

template <typename T, int N, class T1, class SFINAE>
template <typename Flags, int ChunkSize>
std::enable_if_t<is_simd_flag_type_v<Flags>>
simd_obj_impl<T, N, T1, SFINAE>::copy_from(
    const simd_obj_impl<T, N, T1, SFINAE>::element_type *Addr,
    Flags) SYCL_ESIMD_FUNCTION {
  constexpr unsigned Align = Flags::template alignment<T1>;
  copy_from<ChunkSize>(Addr, properties{alignment<Align>});
}

template <typename T, int N, class T1, class SFINAE>
template <int ChunkSize, typename PropertyListT, typename AccessorT,
          typename TOffset>
ESIMD_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
simd_obj_impl<T, N, T1, SFINAE>::copy_to_impl(
    AccessorT acc, TOffset offset, PropertyListT) const SYCL_ESIMD_FUNCTION {
  using UT = simd_obj_impl<T, N, T1, SFINAE>::element_type;
  constexpr unsigned Size = sizeof(T) * N;
  constexpr size_t Align =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(UT));

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
            Tmp.template select<BlockN, 1>(Block * BlockN), PropertyListT{});
      });
    }
    if constexpr (RemSize > 0) {
      constexpr unsigned RemN = RemSize / sizeof(T);
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      block_store<UT, RemN, AccessorT>(
          acc, offset + (NumBlocks * BlockSize),
          Tmp.template select<RemN, 1>(NumBlocks * BlockN), PropertyListT{});
    }
  } else if constexpr (sizeof(T) == 8) {
    simd<int32_t, N * 2> BC = Tmp.template bit_cast_view<int32_t>();
    BC.copy_to(acc, offset, PropertyListT{});
  } else {
    constexpr unsigned NumChunks = N / ChunkSize;
    if constexpr (NumChunks > 0) {
      simd<TOffset, ChunkSize> Offsets(0u, sizeof(T));
      ForHelper<NumChunks>::unroll(
          [acc, offset, &Offsets, &Tmp](unsigned Block) {
            scatter<UT, ChunkSize>(
                acc, Offsets + (offset + (Block * ChunkSize * sizeof(T))),
                Tmp.template select<ChunkSize, 1>(Block * ChunkSize),
                PropertyListT{});
          });
    }
    constexpr unsigned RemN = N % ChunkSize;
    if constexpr (RemN > 0) {
      if constexpr (RemN == 1 || RemN == 8 || RemN == 16) {
        simd<TOffset, RemN> Offsets(0u, sizeof(T));
        scatter<UT, RemN>(
            acc, Offsets + (offset + (NumChunks * ChunkSize * sizeof(T))),
            Tmp.template select<RemN, 1>(NumChunks * ChunkSize),
            PropertyListT{});
      } else {
        constexpr int N1 = RemN < 8 ? 8 : RemN < 16 ? 16 : 32;
        simd_mask_type<N1> Pred(0);
        Pred.template select<RemN, 1>() = 1;
        simd<UT, N1> Vals;
        Vals.template select<RemN, 1>() =
            Tmp.template select<RemN, 1>(NumChunks * ChunkSize);
        simd<TOffset, N1> Offsets(0u, sizeof(T));
        scatter<UT, N1>(
            acc, Offsets + (offset + (NumChunks * ChunkSize * sizeof(T))), Vals,
            Pred, PropertyListT{});
      }
    }
  }
}

template <typename T, int N, class T1, class SFINAE>
template <int ChunkSize, typename Flags, typename AccessorT, typename TOffset>
ESIMD_INLINE std::enable_if_t<is_simd_flag_type_v<Flags>>
simd_obj_impl<T, N, T1, SFINAE>::copy_to_impl(
    AccessorT acc, TOffset offset) const SYCL_ESIMD_FUNCTION {
  constexpr unsigned Align = Flags::template alignment<T1>;
  copy_to_impl<ChunkSize>(acc, offset, properties{alignment<Align>});
}

template <typename T, int N, class T1, class SFINAE>
template <int ChunkSize, typename PropertyListT, typename AccessorT,
          typename TOffset>
ESIMD_INLINE std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
simd_obj_impl<T, N, T1, SFINAE>::copy_from_impl(
    AccessorT acc, TOffset offset, PropertyListT) SYCL_ESIMD_FUNCTION {
  using UT = simd_obj_impl<T, N, T1, SFINAE>::element_type;
  static_assert(sizeof(UT) == sizeof(T));
  constexpr unsigned Size = sizeof(T) * N;
  constexpr size_t Align =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(UT));

  constexpr unsigned BlockSize = OperandSize::OWORD * 8;
  constexpr unsigned NumBlocks = Size / BlockSize;
  constexpr unsigned RemSize = Size % BlockSize;

  if constexpr (Align >= OperandSize::DWORD && Size % OperandSize::OWORD == 0 &&
                detail::isPowerOf2(RemSize / OperandSize::OWORD)) {
    if constexpr (NumBlocks > 0) {
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      ForHelper<NumBlocks>::unroll([BlockN, acc, offset, this](unsigned Block) {
        select<BlockN, 1>(Block * BlockN) = block_load<UT, BlockN, AccessorT>(
            acc, offset + (Block * BlockSize), PropertyListT{});
      });
    }
    if constexpr (RemSize > 0) {
      constexpr unsigned RemN = RemSize / sizeof(T);
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      select<RemN, 1>(NumBlocks * BlockN) = block_load<UT, RemN, AccessorT>(
          acc, offset + (NumBlocks * BlockSize), PropertyListT{});
    }
  } else if constexpr (sizeof(T) == 8) {
    simd<int32_t, N * 2> BC(acc, offset, PropertyListT{});
    bit_cast_view<int32_t>() = BC;
  } else {
    constexpr unsigned NumChunks = N / ChunkSize;
    if constexpr (NumChunks > 0) {
      simd<TOffset, ChunkSize> Offsets(0u, sizeof(T));
      ForHelper<NumChunks>::unroll(
          [acc, offset, &Offsets, this](unsigned Block) {
            select<ChunkSize, 1>(Block * ChunkSize) =
                gather<UT, ChunkSize, AccessorT>(
                    acc, Offsets + (offset + (Block * ChunkSize * sizeof(T))),
                    PropertyListT{});
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
            acc, Offsets + (offset + (NumChunks * ChunkSize * sizeof(T))), Pred,
            PropertyListT{});
        select<RemN, 1>(NumChunks * ChunkSize) =
            Vals.template select<RemN, 1>();
      }
    }
  }
}

template <typename T, int N, class T1, class SFINAE>
template <int ChunkSize, typename Flags, typename AccessorT, typename TOffset>
ESIMD_INLINE std::enable_if_t<is_simd_flag_type_v<Flags>>
simd_obj_impl<T, N, T1, SFINAE>::copy_from_impl(AccessorT acc, TOffset offset)
    SYCL_ESIMD_FUNCTION {
  constexpr unsigned Align = Flags::template alignment<T1>;
  copy_from_impl<ChunkSize>(acc, offset, properties{alignment<Align>});
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, typename Flags, int ChunkSize>
ESIMD_INLINE std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT, accessor_mode_cap::can_read> &&
    is_simd_flag_type_v<Flags>>
simd_obj_impl<T, N, T1, SFINAE>::copy_from(AccessorT acc,
                                           detail::DeviceAccessorOffsetT offset,
                                           Flags) SYCL_ESIMD_FUNCTION {

  copy_from_impl<ChunkSize, Flags>(acc, offset);
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, int ChunkSize, typename PropertyListT>
ESIMD_INLINE std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT, accessor_mode_cap::can_read> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
simd_obj_impl<T, N, T1, SFINAE>::copy_from(AccessorT acc,
                                           detail::DeviceAccessorOffsetT offset,
                                           PropertyListT) SYCL_ESIMD_FUNCTION {

  copy_from_impl<ChunkSize, PropertyListT>(acc, offset);
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, typename Flags, int ChunkSize>
ESIMD_INLINE std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT, accessor_mode_cap::can_read> &&
        is_simd_flag_type_v<Flags>,
    void>
simd_obj_impl<T, N, T1, SFINAE>::copy_from(AccessorT acc, uint32_t offset,
                                           Flags) SYCL_ESIMD_FUNCTION {

  copy_from_impl<ChunkSize, Flags>(acc, offset);
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, int ChunkSize, typename PropertyListT>
ESIMD_INLINE std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT, accessor_mode_cap::can_read> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    void>
simd_obj_impl<T, N, T1, SFINAE>::copy_from(AccessorT acc, uint32_t offset,
                                           PropertyListT) SYCL_ESIMD_FUNCTION {

  copy_from_impl<ChunkSize, PropertyListT>(acc, offset);
}

template <typename T, int N, class T1, class SFINAE>
template <int ChunkSize, typename PropertyListT>
std::enable_if_t<ext::oneapi::experimental::is_property_list_v<PropertyListT>>
simd_obj_impl<T, N, T1, SFINAE>::copy_to(
    simd_obj_impl<T, N, T1, SFINAE>::element_type *Addr,
    PropertyListT) const SYCL_ESIMD_FUNCTION {
  using UT = simd_obj_impl<T, N, T1, SFINAE>::element_type;
  constexpr unsigned Size = sizeof(T) * N;
  constexpr size_t Align =
      detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(UT));

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
                                Tmp.template select<BlockN, 1>(Block * BlockN),
                                PropertyListT{});
      });
    }
    if constexpr (RemSize > 0) {
      constexpr unsigned RemN = RemSize / sizeof(T);
      constexpr unsigned BlockN = BlockSize / sizeof(T);
      block_store<UT, RemN>(Addr + (NumBlocks * BlockN),
                            Tmp.template select<RemN, 1>(NumBlocks * BlockN),
                            PropertyListT{});
    }
  } else if constexpr (sizeof(T) == 8) {
    simd<int32_t, N * 2> BC = Tmp.template bit_cast_view<int32_t>();
    BC.copy_to(reinterpret_cast<int32_t *>(Addr), PropertyListT{});
  } else {
    constexpr unsigned NumChunks = N / ChunkSize;
    if constexpr (NumChunks > 0) {
      simd<uint32_t, ChunkSize> Offsets(0u, sizeof(T));
      ForHelper<NumChunks>::unroll([Addr, &Offsets, &Tmp](unsigned Block) {
        scatter<UT, ChunkSize>(
            Addr + (Block * ChunkSize), Offsets,
            Tmp.template select<ChunkSize, 1>(Block * ChunkSize),
            PropertyListT{});
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
                Offsets, Vals, Pred, PropertyListT{});
          }
        } else {
          simd<uint32_t, RemN> Offsets(0u, sizeof(T));
          scatter<UT, RemN>(Addr + (NumChunks * ChunkSize), Offsets,
                            Tmp.template select<RemN, 1>(NumChunks * ChunkSize),
                            PropertyListT{});
        }
      } else {
        constexpr int N1 = RemN < 8 ? 8 : RemN < 16 ? 16 : 32;
        simd_mask_type<N1> Pred(0);
        Pred.template select<RemN, 1>() = 1;
        simd<UT, N1> Vals;
        Vals.template select<RemN, 1>() =
            Tmp.template select<RemN, 1>(NumChunks * ChunkSize);
        simd<uint32_t, N1> Offsets(0u, sizeof(T));
        scatter<UT, N1>(Addr + (NumChunks * ChunkSize), Offsets, Vals, Pred,
                        PropertyListT{});
      }
    }
  }
}

template <typename T, int N, class T1, class SFINAE>
template <typename Flags, int ChunkSize>
std::enable_if_t<is_simd_flag_type_v<Flags>>
simd_obj_impl<T, N, T1, SFINAE>::copy_to(
    simd_obj_impl<T, N, T1, SFINAE>::element_type *Addr,
    Flags) const SYCL_ESIMD_FUNCTION {
  constexpr unsigned Align = Flags::template alignment<T1>;
  copy_to<ChunkSize>(Addr, properties{alignment<Align>});
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, typename Flags, int ChunkSize>
ESIMD_INLINE std::enable_if_t<detail::is_device_accessor_with_v<
                                  AccessorT, accessor_mode_cap::can_write> &&
                              is_simd_flag_type_v<Flags>>
simd_obj_impl<T, N, T1, SFINAE>::copy_to(AccessorT acc,
                                         detail::DeviceAccessorOffsetT offset,
                                         Flags) const SYCL_ESIMD_FUNCTION {
  copy_to_impl<ChunkSize, Flags>(acc, offset);
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, int ChunkSize, typename PropertyListT>
ESIMD_INLINE std::enable_if_t<
    detail::is_device_accessor_with_v<AccessorT,
                                      accessor_mode_cap::can_write> &&
    ext::oneapi::experimental::is_property_list_v<PropertyListT>>
simd_obj_impl<T, N, T1, SFINAE>::copy_to(
    AccessorT acc, detail::DeviceAccessorOffsetT offset,
    PropertyListT) const SYCL_ESIMD_FUNCTION {
  copy_to_impl<ChunkSize, PropertyListT>(acc, offset);
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, typename Flags, int ChunkSize>
ESIMD_INLINE std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT, accessor_mode_cap::can_write> &&
        is_simd_flag_type_v<Flags>,
    void>
simd_obj_impl<T, N, T1, SFINAE>::copy_to(AccessorT acc, uint32_t offset,
                                         Flags) const SYCL_ESIMD_FUNCTION {
  copy_to_impl<ChunkSize, Flags>(acc, offset);
}

template <typename T, int N, class T1, class SFINAE>
template <typename AccessorT, int ChunkSize, typename PropertyListT>
ESIMD_INLINE std::enable_if_t<
    detail::is_local_accessor_with_v<AccessorT, accessor_mode_cap::can_write> &&
        ext::oneapi::experimental::is_property_list_v<PropertyListT>,
    void>
simd_obj_impl<T, N, T1, SFINAE>::copy_to(
    AccessorT acc, uint32_t offset, PropertyListT) const SYCL_ESIMD_FUNCTION {
  copy_to_impl<ChunkSize, PropertyListT>(acc, offset);
}

} // namespace detail
/// @endcond EXCLUDE

} // namespace ext::intel::esimd
} // namespace _V1
} // namespace sycl
