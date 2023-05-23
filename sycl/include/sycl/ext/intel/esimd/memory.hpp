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
#include <sycl/ext/intel/esimd/simd.hpp>
#include <sycl/ext/intel/esimd/simd_view.hpp>
#include <sycl/half_type.hpp>

#include <cstdint>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::intel::esimd {

/// @addtogroup sycl_esimd_memory
/// @{

/// @defgroup sycl_esimd_memory_atomics Atomic memory access.
/// Memory access functions which perform per-lane atomic update using given
/// operation. "Per-lane" means that the atomicity guarantees of a vector atomic
/// operation are the same as of N independent scalar atomic operations per
/// lane (N is number of lanes).

/// @defgroup sycl_esimd_memory_slm Shared local memory access functions.

/// @} sycl_esimd_memory

/// @cond ESIMD_DETAIL

namespace detail {
// Type used in internal functions to designate SLM access by
// providing dummy accessor of this type. Used to make it possible to delegate
// implemenations of SLM memory accesses to general surface-based memory
// accesses and thus reuse validity checks etc.
struct LocalAccessorMarker {};

} // namespace detail

/// @endcond ESIMD_DETAIL

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
    return __esimd_get_surface_index(
        detail::AccessorPrivateProxy::getQualifiedPtrOrImageObj(acc));
  }
}

// TODO @Pennycook
// {quote}
// ...I'd like us to think more about what we can do to make these interfaces
// more user - friendly. A user providing cache hints has to provide a lot more
// template arguments than required.Could we make this nicer by providing the
// hints as tag - type arguments ?
// ...
//   // Without cache hints, type and length can be deduced from offsets
//   float* p;
//   simd<uint32_t, 16> offsets;
//   auto result = flat_load(p, offsets);
//
//   // With cache hints as templates, verbosity increases significantly:
//   // - Providing any cache hint forces the user to specify the type and
//   length float* p; simd<uint32_t, 16> offsets; auto result =
//   flat_load<uint32_t, 16, 1, CacheHint::Foo, CacheHint::Bar>(p, offsets);
//
//   // With cache hints as tag types, verbosity is reduced:
//   // - Providing a cache hint does not prevent deduction of type and length
//   float* p;
//   simd <uint32_t, 16> offsets;
//   auto result = flat_load(p, offsets, CacheHint::Foo{});
//
// Note also that the templated form prevents a developer from specifying an L3
// hint without also explicitly specifying an L1 hint. If flat_load accepted a
// list of hints, it might be possible to refactor the hints to specify them in
// any order, and it may be more extensible to future cache hints:
// {/quote}
//
// TODO @keryell
// {quote}
// An approach a la https ://github.com/chriskohlhoff/propria from
// @chriskohlhoff would be to add a property to the pointer, such as
//
//    auto result = flat_load(p, offsets);
//    auto result = flat_load(decorate<CacheHint::Foo, CacheHint::Bar>(p),
//    offsets);
// The advantage is that you do not have to change all tour API and all the uses
// of this decorated pointer will benefit from this. decorate is to be bikeshed
// accordingly.
// {/quote}
//

/// Loads ("gathers") elements from different memory locations and returns a
/// vector of them. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector. Access to
/// any element's memory location can be disabled via the input vector of
/// predicates (mask).
/// @tparam Tx Element type, must be of size 4 or less.
/// @tparam N Number of elements to read; can be \c 1, \c 2, \c 4, \c 8, \c 16
///   or \c 32.
/// @param p The base address.
/// @param offsets the vector of 32-bit or 64-bit offsets in bytes. For each
/// lane \c i,   ((byte*)p + offsets[i]) must be element size aligned.
/// @param mask The access mask, defaults to all 1s.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
///
template <typename Tx, int N, typename Toffset>
__ESIMD_API simd<Tx, N> gather(const Tx *p, simd<Toffset, N> offsets,
                               simd_mask<N> mask = 1) {
  using T = detail::__raw_t<Tx>;
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert(detail::isPowerOf2(N, 32), "Unsupported value of N");
  simd<uint64_t, N> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;

  if constexpr (sizeof(T) == 1) {
    auto Ret = __esimd_svm_gather<T, N, detail::ElemsPerAddrEncoding<4>(),
                                  detail::ElemsPerAddrEncoding<1>()>(
        addrs.data(), mask.data());
    return __esimd_rdregion<T, N * 4, N, /*VS*/ 0, N, 4>(Ret, 0);
  } else if constexpr (sizeof(T) == 2) {
    auto Ret = __esimd_svm_gather<T, N, detail::ElemsPerAddrEncoding<2>(),
                                  detail::ElemsPerAddrEncoding<2>()>(
        addrs.data(), mask.data());
    return __esimd_rdregion<T, N * 2, N, /*VS*/ 0, N, 2>(Ret, 0);
  } else
    return __esimd_svm_gather<T, N, detail::ElemsPerAddrEncoding<1>(),
                              detail::ElemsPerAddrEncoding<1>()>(addrs.data(),
                                                                 mask.data());
}

/// A variation of \c gather API with \c offsets represented as \c simd_view
/// object.
///
/// @tparam Tx Element type, must be of size 4 or less.
/// @tparam N Number of elements to read; can be \c 1, \c 2, \c 4, \c 8, \c 16
///   or \c 32.
/// @param p The base address.
/// @param offsets the simd_view of 32-bit or 64-bit offsets in bytes. For each
/// lane \c i,   ((byte*)p + offsets[i]) must be element size aligned.
/// @param mask The access mask, defaults to all 1s.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
///
template <typename Tx, int N, typename Toffset,
          typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API simd<Tx, N> gather(const Tx *p,
                               simd_view<Toffset, RegionTy> offsets,
                               simd_mask<N> mask = 1) {
  return gather<Tx, N>(p, offsets.read(), mask);
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
template <typename Tx, int N, typename Toffset,
          typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API void scatter(Tx *p, simd_view<Toffset, RegionTy> offsets,
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

/// Loads a contiguous block of memory from given memory address and returns
/// the loaded data as a vector. Actual code generated depends on the
/// alignment parameter.
/// @tparam Tx Element type.
/// @tparam N Number of elements to load, <code>N * sizeof(Tx)</code> must be
///    1, 2, 4 or 8 owords long.
/// @tparam Flags The alignment specifier type tag. Auto-deduced from the
///    \c Flags parameter. If it is less than \c 16, then slower unaligned
///    access is generated, othewise the access is aligned.
/// @param addr The address to load from.
/// @param Flags Specifies the alignment.
/// @return A vector of loaded elements.
///
template <typename Tx, int N, typename Flags = vector_aligned_tag,
          class T = detail::__raw_t<Tx>,
          typename = std::enable_if_t<is_simd_flag_type_v<Flags>>>
__ESIMD_API simd<Tx, N> block_load(const Tx *addr, Flags = {}) {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");

  uintptr_t Addr = reinterpret_cast<uintptr_t>(addr);
  if constexpr (Flags::template alignment<simd<T, N>> >=
                detail::OperandSize::OWORD) {
    return __esimd_svm_block_ld<T, N>(Addr);
  } else {
    return __esimd_svm_block_ld_unaligned<T, N>(Addr);
  }
}

/// Loads a contiguous block of memory from given accessor and offset and
/// returns the loaded data as a vector. Actual code generated depends on the
/// alignment parameter.
/// @tparam Tx Element type.
/// @tparam N Number of elements to load, <code>N * sizeof(Tx)</code> must be
///    1, 2, 4 or 8 owords long.
/// @tparam AccessorTy Accessor type (auto-deduced).
/// @tparam Flags The alignment specifier type tag. Auto-deduced from the
///    \c Flags parameter. If it is less than \c 16, then slower unaligned
///    access is generated, othewise the access is aligned.
/// @param acc The accessor.
/// @param offset The offset to load from in bytes.
/// @param Flags Specifies the alignment.
/// @return A vector of loaded elements.
///
template <typename Tx, int N, typename AccessorTy,
          typename Flags = vector_aligned_tag,
          typename = std::enable_if_t<is_simd_flag_type_v<Flags>>,
          class T = detail::__raw_t<Tx>>
__ESIMD_API simd<Tx, N> block_load(AccessorTy acc,
#ifdef __ESIMD_FORCE_STATELESS_MEM
                                   uint64_t offset,
#else
                                   uint32_t offset,
#endif
                                   Flags = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return block_load<Tx, N>(__ESIMD_DNS::accessorToPointer<Tx>(acc, offset));
#else
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
    return __esimd_oword_ld<T, N>(surf_ind, offset >> 4);
  } else {
    return __esimd_oword_ld_unaligned<T, N>(surf_ind, offset);
  }
#endif
}

/// Stores elements of a vector to a contiguous block of memory at given
/// address. The address must be at least \c 16 bytes-aligned.
/// @tparam Tx Element type.
/// @tparam N Number of elements to store, <code>N * sizeof(Tx)</code> must be
///    1, 2, 4 or 8 owords long.
/// @param p The memory address to store at.
/// @param vals The vector to store.
///
template <typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API void block_store(Tx *p, simd<Tx, N> vals) {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");

  uintptr_t Addr = reinterpret_cast<uintptr_t>(p);
  __esimd_svm_block_st<T, N>(Addr, vals.data());
}

/// Stores elements of a vector to a contiguous block of memory represented by
/// an accessor and an offset within this accessor.
/// @tparam Tx Element type.
/// @tparam N Number of elements to store, <code>N * sizeof(Tx)</code> must be
///    1, 2, 4 or 8 owords long.
/// @tparam AccessorTy Accessor type (auto-deduced).
/// @param acc The accessor to store to.
/// @param offset The offset to store at. It is in bytes and must be a multiple
///   of \c 16.
/// @param vals The vector to store.
///
template <typename Tx, int N, typename AccessorTy,
          class T = detail::__raw_t<Tx>>
__ESIMD_API void block_store(AccessorTy acc,
#ifdef __ESIMD_FORCE_STATELESS_MEM
                             uint64_t offset,
#else
                             uint32_t offset,
#endif
                             simd<Tx, N> vals) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  block_store<Tx, N>(__ESIMD_DNS::accessorToPointer<Tx>(acc, offset), vals);
#else
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
  __esimd_oword_st<T, N>(surf_ind, offset >> 4, vals.data());
#endif
}

/// @} sycl_esimd_memory

/// @cond ESIMD_DETAIL

// Implementations of accessor-based gather and scatter functions
namespace detail {
template <typename T, int N, typename AccessorTy>
ESIMD_INLINE
    ESIMD_NODEBUG std::enable_if_t<(sizeof(T) <= 4) &&
                                   (N == 1 || N == 8 || N == 16 || N == 32) &&
                                   !std::is_pointer<AccessorTy>::value>
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

template <typename T, int N, typename AccessorTy>
ESIMD_INLINE ESIMD_NODEBUG std::enable_if_t<
    (sizeof(T) <= 4) && (N == 1 || N == 8 || N == 16 || N == 32) &&
        !std::is_pointer<AccessorTy>::value,
    simd<T, N>>
gather_impl(AccessorTy acc, simd<uint32_t, N> offsets, uint32_t glob_offset,
            simd_mask<N> mask) {

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
    const simd<PromoT, N> promo_vals =
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

} // namespace detail

/// @endcond ESIMD_DETAIL

/// @addtogroup sycl_esimd_memory
/// @{

/// @anchor accessor_gather Accessor-based gather.
///
/// Collects elements located at given offsets in an accessor and returns them
/// as a single \ref simd object. An element can be 1, 2 or 4-byte value.
///
/// @tparam T Element type; can only be a 1,2,4-byte integer, \c sycl::half or
///   \c float.
/// @tparam N The number of vector elements. Can be \c 1, \c 8, \c 16 or \c 32.
/// @tparam AccessorTy The accessor type.
/// @tparam Toffset The offset type.
/// @param acc The accessor to gather from.
/// @param offsets Per-element offsets in bytes.
/// @param glob_offset Offset in bytes added to each individual element's offset
///   to compute actual memory access offset for that element.
/// @param mask Memory access mask. Elements with zero corresponding mask's
///   predicate are not accessed, their values in the resulting vector are
///   undefined.
///
template <typename T, int N, typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    (sizeof(T) <= 4) && (N == 1 || N == 8 || N == 16 || N == 32) &&
        !std::is_pointer<AccessorTy>::value && std::is_integral_v<Toffset>,
    simd<T, N>>
gather(AccessorTy acc, simd<Toffset, N> offsets,
#ifdef __ESIMD_FORCE_STATELESS_MEM
       uint64_t glob_offset = 0,
#else
       uint32_t glob_offset = 0,
#endif
       simd_mask<N> mask = 1) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return gather<T, N>(__ESIMD_DNS::accessorToPointer<T>(acc, glob_offset),
                      offsets, mask);
#else
  return detail::gather_impl<T, N, AccessorTy>(acc, offsets, glob_offset, mask);
#endif
}

/// @anchor accessor_scatter
/// Accessor-based scatter.
///
/// Writes elements of a \ref simd object into an accessor at given offsets.
/// An element can be 1, 2 or 4-byte value.
///
/// @tparam T Element type; can only be a 1,2,4-byte integer, \c sycl::half or
///   \c float.
/// @tparam N The number of vector elements. Can be \c 1, \c 8, \c 16 or \c 32.
/// @tparam AccessorTy The accessor type.
/// @tparam Toffset The offset type.
/// @param acc The accessor to scatter to.
/// @param offsets Per-element offsets in bytes.
/// @param vals Values to write.
/// @param glob_offset Offset in bytes added to each individual element's offset
///   to compute actual memory access offset for that element.
/// @param mask Memory access mask. Elements with zero corresponding mask's
///   predicate are not accessed.
///
///
template <typename T, int N, typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    (sizeof(T) <= 4) && (N == 1 || N == 8 || N == 16 || N == 32) &&
    !std::is_pointer<AccessorTy>::value && std::is_integral_v<Toffset>>
scatter(AccessorTy acc, simd<Toffset, N> offsets, simd<T, N> vals,
#ifdef __ESIMD_FORCE_STATELESS_MEM
        uint64_t glob_offset = 0,
#else
        uint32_t glob_offset = 0,
#endif
        simd_mask<N> mask = 1) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  scatter<T, N>(__ESIMD_DNS::accessorToPointer<T>(acc, glob_offset), offsets,
                vals, mask);
#else
  detail::scatter_impl<T, N, AccessorTy>(acc, vals, offsets, glob_offset, mask);
#endif
}

/// Load a scalar value from an accessor.
/// @tparam T Type of the value.
/// @tparam AccessorTy Type of the accessor.
/// @param acc Accessor to load from.
/// @param offset Offset in bytes.
/// @return The loaded value.
///
template <typename T, typename AccessorTy>
__ESIMD_API T scalar_load(AccessorTy acc, uint32_t offset) {
  const simd<T, 1> Res =
      gather<T, 1, AccessorTy>(acc, simd<uint32_t, 1>(offset));
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
__ESIMD_API void scalar_store(AccessorTy acc, uint32_t offset, T val) {
  scatter<T, 1, AccessorTy>(acc, simd<uint32_t, 1>(offset), simd<T, 1>(val));
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
          int N, typename Toffset,
          typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API simd<T, N * get_num_channels_enabled(RGBAMask)>
gather_rgba(const T *p, simd_view<Toffset, RegionTy> offsets,
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

template <typename T, int N, rgba_channel_mask RGBAMask>
__SYCL_DEPRECATED("use gather_rgba<rgba_channel_mask>()")
__ESIMD_API std::enable_if_t<
    (N == 8 || N == 16 || N == 32) && sizeof(T) == 4,
    simd<T, N * get_num_channels_enabled(
                    RGBAMask)>> gather_rgba(const T *p,
                                            simd<uint32_t, N> offsets,
                                            simd_mask<N> mask = 1) {
  return gather_rgba<RGBAMask>(p, offsets, mask);
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
          int N, typename Toffset,
          typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API void
scatter_rgba(T *p, simd_view<Toffset, RegionTy> offsets,
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
/// The returned vector elements mutch the accessor data type. The loaded
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
__ESIMD_API std::enable_if_t<((N == 8 || N == 16 || N == 32) &&
                              sizeof(T) == 4 && !std::is_pointer_v<AccessorT>),
                             simd<T, N * get_num_channels_enabled(RGBAMask)>>
gather_rgba(AccessorT acc, simd<uint32_t, N> offsets,
            uint32_t global_offset = 0, simd_mask<N> mask = 1) {
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

/// Gather data from the memory addressed by accessor \c acc, offset common
/// for all loaded elements \c global_offset and per-element offsets \c offsets,
/// and return it as simd vector. See @ref usm_gather_rgba for information about
/// the operation semantics and parameter restrictions/interdependencies.
/// @tparam RGBAMask Pixel's channel mask.
/// @tparam AccessorT The accessor type for the memory to be stored/scattered.
/// The returned vector elements mast match the accessor data type. The loaded
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
__ESIMD_API std::enable_if_t<(N == 8 || N == 16 || N == 32) && sizeof(T) == 4 &&
                             !std::is_pointer_v<AccessorT>>
scatter_rgba(AccessorT acc, simd<uint32_t, N> offsets,
             simd<T, N * get_num_channels_enabled(RGBAMask)> vals,
             uint32_t global_offset = 0, simd_mask<N> mask = 1) {
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

/// @} sycl_esimd_memory

namespace detail {
/// Check the legality of an atomic call in terms of size and type.
///
template <__ESIMD_NS::atomic_op Op, typename T, int N, unsigned NumSrc>
constexpr void check_atomic() {

  static_assert((detail::isPowerOf2(N, 32)),
                "Execution size 1, 2, 4, 8, 16, 32 are supported");

  static_assert(NumSrc == __ESIMD_DNS::get_num_args<Op>(),
                "wrong number of operands");
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
                Op == __ESIMD_NS::atomic_op::fsub) {
    static_assert((is_type<T, float, sycl::half, double>()),
                  "float, double or sycl::half type is expected");
  }
  if constexpr (Op == __ESIMD_NS::atomic_op::add ||
                Op == __ESIMD_NS::atomic_op::sub ||
                Op == __ESIMD_NS::atomic_op::min ||
                Op == __ESIMD_NS::atomic_op::max ||
                Op == __ESIMD_NS::atomic_op::bit_and ||
                Op == __ESIMD_NS::atomic_op::bit_or ||
                Op == __ESIMD_NS::atomic_op::bit_xor ||
                Op == __ESIMD_NS::atomic_op::minsint ||
                Op == __ESIMD_NS::atomic_op::maxsint) {
    static_assert(IsInt2BytePlus, "Integral 16-bit or wider type is expected");
    constexpr bool IsSignedMinmax = (Op == __ESIMD_NS::atomic_op::minsint) ||
                                    (Op == __ESIMD_NS::atomic_op::maxsint);
    constexpr bool IsUnsignedMinmax = (Op == __ESIMD_NS::atomic_op::min) ||
                                      (Op == __ESIMD_NS::atomic_op::max);

    if constexpr (IsSignedMinmax || IsUnsignedMinmax) {
      constexpr bool SignOK = std::is_signed_v<T> == IsSignedMinmax;
      static_assert(SignOK, "Signed/unsigned integer type expected for "
                            "signed/unsigned min/max operation");
    }
  }
}
} // namespace detail

/// @addtogroup sycl_esimd_memory_atomics
/// @{

/// @anchor usm_atomic_update1
/// @brief Single-argument variant of the atomic update operation.
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
/// \c atomic_op::fmax, \c atomic_op::fmin, \c atomic_op::store.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset>
__ESIMD_API simd<Tx, N> atomic_update(Tx *p, simd<Toffset, N> offset,
                                      simd<Tx, N> src0, simd_mask<N> mask) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  if constexpr ((Op == atomic_op::fmin) || (Op == atomic_op::fmax) ||
                (Op == atomic_op::fadd) || (Op == atomic_op::fsub)) {
    // Auto-convert FP atomics to LSC version. Warning is given - see enum.
    return atomic_update<detail::to_lsc_atomic_op<Op>(), Tx, N>(p, offset, src0,
                                                                mask);
  } else if constexpr (Op == atomic_op::store) {
    if constexpr (std::is_integral_v<Tx>) {
      return atomic_update<atomic_op::xchg, Tx, N>(p, offset, src0, mask);
    } else {
      using Tint = detail::uint_type_t<sizeof(Tx)>;
      simd<Tint, N> Res = atomic_update<atomic_op::xchg, Tint, N>(
          reinterpret_cast<Tint *>(p), offset,
          src0.template bit_cast_view<Tint>(), mask);
      return Res.template bit_cast_view<Tx>();
    }
  } else {
    detail::check_atomic<Op, Tx, N, 1>();
    simd<uintptr_t, N> vAddr(reinterpret_cast<uintptr_t>(p));
    simd<uintptr_t, N> offset_i1 = convert<uintptr_t>(offset);
    vAddr += offset_i1;

    using T = typename detail::__raw_t<Tx>;
    return __esimd_svm_atomic1<Op, T, N>(vAddr.data(), src0.data(),
                                         mask.data());
  }
}

/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::store.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API simd<Tx, N> atomic_update(Tx *p,
                                      simd_view<Toffset, RegionTy> offsets,
                                      simd<Tx, N> src0, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(p, offsets.read(), src0, mask);
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
/// @param offset The scalar 32-bit or 64-bit offsets in bytes.
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
atomic_update(Tx *p, Toffset offset, simd<Tx, N> src0, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(p, simd<Toffset, N>(offset), src0, mask);
}

/// @anchor usm_atomic_update0
/// @brief No-argument variant of the atomic update operation.
///
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has no arguments in addition to the value at the memory location.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc or
/// \c atomic_op::dec, \c atomic_op::load.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset>
__ESIMD_API simd<Tx, N> atomic_update(Tx *p, simd<Toffset, N> offset,
                                      simd_mask<N> mask) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  if constexpr (Op == atomic_op::load) {
    if constexpr (std::is_integral_v<Tx>) {
      return atomic_update<atomic_op::bit_or, Tx, N>(p, offset, simd<Tx, N>(0),
                                                     mask);
    } else {
      using Tint = detail::uint_type_t<sizeof(Tx)>;
      simd<Tint, N> Res = atomic_update<atomic_op::bit_or, Tint, N>(
          reinterpret_cast<Tint *>(p), offset, simd<Tint, N>(0), mask);
      return Res.template bit_cast_view<Tx>();
    }
  } else {
    detail::check_atomic<Op, Tx, N, 0>();

    simd<uintptr_t, N> vAddr(reinterpret_cast<uintptr_t>(p));
    simd<uintptr_t, N> offset_i1 = convert<uintptr_t>(offset);
    vAddr += offset_i1;
    using T = typename detail::__raw_t<Tx>;
    return __esimd_svm_atomic0<Op, T, N>(vAddr.data(), mask.data());
  }
}

/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc or
/// \c atomic_op::dec, \c atomic_op::load.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API simd<Tx, N> atomic_update(Tx *p,
                                      simd_view<Toffset, RegionTy> offsets,
                                      simd_mask<N> mask = 1) {
  return atomic_update<Op, Tx, N>(p, offsets.read(), mask);
}

/// A variation of \c atomic_update API with \c offset represented as
/// scalar.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc or
/// \c atomic_op::dec, \c atomic_op::load.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param offset The scalar 32-bit or 64-bit offset in bytes.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset>, simd<Tx, N>>
atomic_update(Tx *p, Toffset offset, simd_mask<N> mask = 1) {
  return atomic_update<Op, Tx, N>(p, simd<Toffset, N>(offset), mask);
}

/// @anchor usm_atomic_update2
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpwr.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param offset The vector of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset>
__ESIMD_API simd<Tx, N> atomic_update(Tx *p, simd<Toffset, N> offset,
                                      simd<Tx, N> src0, simd<Tx, N> src1,
                                      simd_mask<N> mask) {
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  if constexpr (Op == atomic_op::fcmpwr) {
    // Auto-convert FP atomics to LSC version. Warning is given - see enum.
    return atomic_update<detail::to_lsc_atomic_op<Op>(), Tx, N>(p, offset, src0,
                                                                src1, mask);
  } else {
    detail::check_atomic<Op, Tx, N, 2>();
    simd<uintptr_t, N> vAddr(reinterpret_cast<uintptr_t>(p));
    simd<uintptr_t, N> offset_i1 = convert<uintptr_t>(offset);
    vAddr += offset_i1;
    using T = typename detail::__raw_t<Tx>;
    return __esimd_svm_atomic2<Op, T, N>(vAddr.data(), src0.data(), src1.data(),
                                         mask.data());
  }
}

/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpwr.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param offset The simd_view of 32-bit or 64-bit offsets in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API simd<Tx, N>
atomic_update(Tx *p, simd_view<Toffset, RegionTy> offsets, simd<Tx, N> src0,
              simd<Tx, N> src1, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(p, offsets.read(), src0, src1, mask);
}

/// A variation of \c atomic_update API with \c offsets represented as
/// scalar.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpwr.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param offset The scalar 32-bit or 64-bit offset in bytes.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset>, simd<Tx, N>>
atomic_update(Tx *p, Toffset offset, simd<Tx, N> src0, simd<Tx, N> src1,
              simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(p, simd<Toffset, N>(offset), src0, src1,
                                  mask);
}

/// @anchor accessor_atomic_update1
/// @brief Single-argument variant of the atomic update operation.
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
/// \c atomic_op::fmax, \c atomic_op::fmin, \c atomic_op::store.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The vector of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 !std::is_pointer<AccessorTy>::value,
                             simd<Tx, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> offset, simd<Tx, N> src0,
              simd_mask<N> mask) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update<Op, Tx, N>(__ESIMD_DNS::accessorToPointer<Tx>(acc),
                                  offset, src0, mask);
#else
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert(sizeof(Toffset) == 4, "Only 32 bit offset is supported");
  if constexpr ((Op == atomic_op::fmin) || (Op == atomic_op::fmax) ||
                (Op == atomic_op::fadd) || (Op == atomic_op::fsub)) {
    // Auto-convert FP atomics to LSC version. Warning is given - see enum.
    return atomic_update<detail::to_lsc_atomic_op<Op>(), Tx, N>(acc, offset,
                                                                src0, mask);
  } else if constexpr (Op == atomic_op::store) {
    if constexpr (std::is_integral_v<Tx>) {
      return atomic_update<atomic_op::xchg, Tx, N>(acc, offset, src0, mask);
    } else {
      using Tint = detail::uint_type_t<sizeof(Tx)>;
      simd<Tint, N> Res = atomic_update<atomic_op::xchg, Tint, N>(
          acc, offset, src0.template bit_cast_view<Tint>(), mask);
      return Res.template bit_cast_view<Tx>();
    }
  } else {
    detail::check_atomic<Op, Tx, N, 1>();
    static_assert(sizeof(Tx) == 4, "Only 32 bit data is supported");
    const auto si = __ESIMD_NS::get_surface_index(acc);
    using T = typename detail::__raw_t<Tx>;
    return __esimd_dword_atomic1<Op, T, N>(mask.data(), si, offset.data(),
                                           src0.data());
  }
#endif
}

/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c
/// atomic_op::max, \c atomic_op::xchg, \c atomic_op::bit_and, \c
/// atomic_op::bit_or, \c atomic_op::bit_xor, \c atomic_op::minsint, \c
/// atomic_op::maxsint, \c atomic_op::fmax, \c atomic_op::fmin, \c
/// atomic_op::store.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offsets The simd_view of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy, typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 !std::is_pointer<AccessorTy>::value,
                             simd<Tx, N>>
atomic_update(AccessorTy acc, simd_view<Toffset, RegionTy> offsets,
              simd<Tx, N> src0, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(acc, offsets.read(), src0, mask);
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
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> && !std::is_pointer<AccessorTy>::value &&
        ((Op != atomic_op::store && Op != atomic_op::xchg) || N == 1),
    simd<Tx, N>>
atomic_update(AccessorTy acc, Toffset offset, simd<Tx, N> src0,
              simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(acc, simd<Toffset, N>(offset), src0, mask);
}

/// @anchor accessor_atomic_update0
/// @brief No-argument variant of the atomic update operation.
///
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets, and returns a vector of old values found at the
/// memory locations before update. The update operation has no arguments
/// in addition to the value at the memory location.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc or
/// \c atomic_op::dec, \c atomic_op::load.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The vector of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API
    __ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                     !std::is_pointer<AccessorTy>::value,
                                 simd<Tx, N>>
    atomic_update(AccessorTy acc, simd<Toffset, N> offset, simd_mask<N> mask) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update<Op, Tx, N>(__ESIMD_DNS::accessorToPointer<Tx>(acc),
                                  offset, mask);
#else
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  if constexpr (Op == atomic_op::load) {
    if constexpr (std::is_integral_v<Tx>) {
      return atomic_update<atomic_op::bit_or, Tx, N>(acc, offset,
                                                     simd<Tx, N>(0), mask);
    } else {
      using Tint = detail::uint_type_t<sizeof(Tx)>;
      simd<Tint, N> Res = atomic_update<atomic_op::bit_or, Tint, N>(
          acc, offset, simd<Tint, N>(0), mask);
      return Res.template bit_cast_view<Tx>();
    }
  } else {
    detail::check_atomic<Op, Tx, N, 0>();
    static_assert(sizeof(Toffset) == 4, "Only 32 bit offset is supported");

    static_assert(sizeof(Tx) == 4, "Only 32 bit data is supported");
    const auto si = __ESIMD_NS::get_surface_index(acc);
    using T = typename detail::__raw_t<Tx>;
    return __esimd_dword_atomic0<Op, T, N>(mask.data(), si, offset.data());
  }
#endif
}

/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc or
/// \c atomic_op::dec, \c atomic_op::load.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The simd_view of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy, typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 !std::is_pointer<AccessorTy>::value,
                             simd<Tx, N>>
atomic_update(AccessorTy acc, simd_view<Toffset, RegionTy> offsets,
              simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(acc, offsets.read(), mask);
}

/// A variation of \c atomic_update API with \c offset represented as
/// scalar.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc or
/// \c atomic_op::dec, \c atomic_op::load.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The scalar 32-bit or 64-bit offset in bytes. 64-bit
/// offset are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 !std::is_pointer<AccessorTy>::value,
                             simd<Tx, N>>
atomic_update(AccessorTy acc, Toffset offset, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(acc, simd<Toffset, N>(offset), mask);
}

/// @anchor accessor_atomic_update2
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpwr.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The vector of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 !std::is_pointer<AccessorTy>::value,
                             simd<Tx, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> offset, simd<Tx, N> src0,
              simd<Tx, N> src1, simd_mask<N> mask) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update<Op, Tx, N>(__ESIMD_DNS::accessorToPointer<Tx>(acc),
                                  offset, src0, src1, mask);
#else
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert(sizeof(Toffset) == 4, "Only 32 bit offset is supported");
  if constexpr (Op == atomic_op::fcmpwr) {
    // Auto-convert FP atomics to LSC version. Warning is given - see enum.
    return atomic_update<detail::to_lsc_atomic_op<Op>(), Tx, N>(
        acc, offset, src0, src1, mask);
  } else {
    detail::check_atomic<Op, Tx, N, 2>();
    static_assert(sizeof(Tx) == 4, "Only 32 bit data is supported");
    const auto si = __ESIMD_NS::get_surface_index(acc);
    using T = typename detail::__raw_t<Tx>;
    return __esimd_dword_atomic2<Op, T, N>(mask.data(), si, offset.data(),
                                           src0.data(), src1.data());
  }
#endif
}

/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpwr.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The simd_view of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy, typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 !std::is_pointer<AccessorTy>::value,
                             simd<Tx, N>>
atomic_update(AccessorTy acc, simd_view<Toffset, RegionTy> offsets,
              simd<Tx, N> src0, simd<Tx, N> src1, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(acc, offsets.read(), src0, src1, mask);
}

/// A variation of \c atomic_update API with \c offsets represented as
/// scalar.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpwr.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The scalar 32-bit or 64-bit offset in bytes. 64-bit
/// offset are supported only when stateless memory accesses are enforced, i.e.
/// accessor based accesses are automatically converted to stateless accesses.
/// @param src0 The first additional argument (new value).
/// @param src1 The second additional argument (expected value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<std::is_integral_v<Toffset> &&
                                 !std::is_pointer<AccessorTy>::value,
                             simd<Tx, N>>
atomic_update(AccessorTy acc, Toffset offset, simd<Tx, N> src0,
              simd<Tx, N> src1, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(acc, simd<Toffset, N>(offset), src0, src1,
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

/// @addtogroup sycl_esimd_memory_slm
/// @{

/// Declare per-work-group slm size.
/// @tparam SLMSize  Shared Local Memory (SLM) size
template <uint32_t SLMSize> __ESIMD_API void slm_init() {
  __esimd_slm_init(SLMSize);
}

/// Declare per-work-group slm size. Non-constant argument version to be used
/// with specialization constants only.
/// @param size  Shared Local Memory (SLM) size
__ESIMD_API void slm_init(uint32_t size) { __esimd_slm_init(size); }

/// Gather operation over the Shared Local Memory.
/// This API has almost the same interface as the @ref accessor_gather
/// "accessor-based gather", except that it does not have the accessor and the
/// global offset parameters.
///
template <typename T, int N>
__ESIMD_API
    std::enable_if_t<(N == 1 || N == 8 || N == 16 || N == 32), simd<T, N>>
    slm_gather(simd<uint32_t, N> offsets, simd_mask<N> mask = 1) {
  detail::LocalAccessorMarker acc;
  return detail::gather_impl<T, N>(acc, offsets, 0, mask);
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

/// Loads a contiguous block of memory from the SLM at given offset and
/// returns the loaded data as a vector.
/// @tparam T Element type.
/// @tparam N Number of elements to load, <code>N * sizeof(Tx)</code> must be
///    1, 2, 4 or 8 owords long.
/// @param offset The offset to load from in bytes. Must be oword-aligned.
/// @return A vector of loaded elements.
///
template <typename T, int N>
__ESIMD_API simd<T, N> slm_block_load(uint32_t offset) {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 16 * detail::OperandSize::OWORD,
                "block size must be at most 16 owords");

  const auto si = __ESIMD_NS::get_surface_index(detail::LocalAccessorMarker());
  return __esimd_oword_ld<detail::__raw_t<T>, N>(si, offset >> 4);
}

/// Stores elements of a vector to a contiguous block of SLM at given
/// offset.
/// @tparam T Element type.
/// @tparam N Number of elements to store, <code>N * sizeof(Tx)</code> must be
///    1, 2, 4 or 8 owords long.
/// @param offset The offset in bytes to store at. Must be oword-aligned.
/// @param vals The vector to store.
///
template <typename T, int N>
__ESIMD_API void slm_block_store(uint32_t offset, simd<T, N> vals) {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");
  const auto si = __ESIMD_NS::get_surface_index(detail::LocalAccessorMarker());
  // offset in genx.oword.st is in owords
  __esimd_oword_st<detail::__raw_t<T>, N>(si, offset >> 4, vals.data());
}

/// Atomic update operation performed on SLM. No source operands version.
/// See description of template and function parameters in @ref
/// usm_atomic_update0 "atomic update" operation docs.
template <atomic_op Op, typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API simd<Tx, N> slm_atomic_update(simd<uint32_t, N> offsets,
                                          simd_mask<N> mask) {
  detail::check_atomic<Op, T, N, 0>();
  const auto si = __ESIMD_NS::get_surface_index(detail::LocalAccessorMarker());
  return __esimd_dword_atomic0<Op, T, N>(mask.data(), si, offsets.data());
}

/// Atomic update operation performed on SLM. One source operands version.
/// See description of template and function parameters in @ref
/// usm_atomic_update1 "atomic update" operation docs.
template <atomic_op Op, typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API simd<Tx, N> slm_atomic_update(simd<uint32_t, N> offsets,
                                          simd<Tx, N> src0, simd_mask<N> mask) {
  detail::check_atomic<Op, T, N, 1>();
  const auto si = __ESIMD_NS::get_surface_index(detail::LocalAccessorMarker());
  return __esimd_dword_atomic1<Op, T, N>(mask.data(), si, offsets.data(),
                                         src0.data());
}

/// Atomic update operation performed on SLM. Two source operands version.
/// See description of template and function parameters in @ref
/// usm_atomic_update2 "atomic update" operation docs.
template <atomic_op Op, typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API simd<Tx, N> slm_atomic_update(simd<uint32_t, N> offsets,
                                          simd<Tx, N> src0, simd<Tx, N> src1,
                                          simd_mask<N> mask) {
  detail::check_atomic<Op, T, N, 2>();
  const auto si = __ESIMD_NS::get_surface_index(detail::LocalAccessorMarker());
  return __esimd_dword_atomic2<Op, T, N>(mask.data(), si, offsets.data(),
                                         src0.data(), src1.data());
}

/// @} sycl_esimd_memory_slm

#ifndef __ESIMD_FORCE_STATELESS_MEM
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
    simd<T, m *n1> temp =
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
#endif // !__ESIMD_FORCE_STATELESS_MEM

/// @} sycl_esimd_memory

/// @cond EXCLUDE

namespace detail {
// ----- Outlined implementations of simd_obj_impl class memory access APIs.

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
template <typename AccessorT, typename Flags, int ChunkSize, typename>
ESIMD_INLINE EnableIfAccessor<AccessorT, accessor_mode_cap::can_read,
                              sycl::access::target::device, void>
simd_obj_impl<T, N, T1, SFINAE>::copy_from(AccessorT acc, uint32_t offset,
                                           Flags) SYCL_ESIMD_FUNCTION {
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
      simd<uint32_t, ChunkSize> Offsets(0u, sizeof(T));
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
        simd<uint32_t, RemN> Offsets(0u, sizeof(T));
        select<RemN, 1>(NumChunks * ChunkSize) = gather<UT, RemN, AccessorT>(
            acc, Offsets, offset + (NumChunks * ChunkSize * sizeof(T)));
      } else {
        constexpr int N1 = RemN < 8 ? 8 : RemN < 16 ? 16 : 32;
        simd_mask_type<N1> Pred(0);
        Pred.template select<RemN, 1>() = 1;
        simd<uint32_t, N1> Offsets(0u, sizeof(T));
        simd<UT, N1> Vals = gather<UT, N1>(
            acc, Offsets, offset + (NumChunks * ChunkSize * sizeof(T)), Pred);
        select<RemN, 1>(NumChunks * ChunkSize) =
            Vals.template select<RemN, 1>();
      }
    }
  }
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
ESIMD_INLINE EnableIfAccessor<AccessorT, accessor_mode_cap::can_write,
                              sycl::access::target::device, void>
simd_obj_impl<T, N, T1, SFINAE>::copy_to(AccessorT acc, uint32_t offset,
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
      simd<uint32_t, ChunkSize> Offsets(0u, sizeof(T));
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
        simd<uint32_t, RemN> Offsets(0u, sizeof(T));
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
        simd<uint32_t, N1> Offsets(0u, sizeof(T));
        scatter<UT, N1, AccessorT>(acc, Offsets, Vals,
                                   offset + (NumChunks * ChunkSize * sizeof(T)),
                                   Pred);
      }
    }
  }
}

} // namespace detail
/// @endcond EXCLUDE

} // namespace ext::intel::esimd
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
