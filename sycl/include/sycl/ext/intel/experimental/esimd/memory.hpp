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

#include <CL/sycl/half_type.hpp>
#include <sycl/ext/intel/experimental/esimd/common.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/memory_intrin.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/types.hpp>
#include <sycl/ext/intel/experimental/esimd/detail/util.hpp>
#include <sycl/ext/intel/experimental/esimd/simd.hpp>

#include <cstdint>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

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

// Shared Local Memory Binding Table Index (aka surface index).
static inline constexpr SurfaceIndex SLM_BTI = 254;
static inline constexpr SurfaceIndex INVALID_BTI =
    static_cast<SurfaceIndex>(-1);
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
#ifdef __SYCL_DEVICE_ONLY__
  if constexpr (std::is_same_v<detail::LocalAccessorMarker, AccessorTy>) {
    return detail::SLM_BTI;
  } else {
    const auto mem_obj = detail::AccessorPrivateProxy::getNativeImageObj(acc);
    return __esimd_get_surface_index(mem_obj);
  }
#else
  throw sycl::feature_not_supported();
#endif
}

#ifdef __SYCL_DEVICE_ONLY__
#define __ESIMD_GET_SURF_HANDLE(acc) get_surface_index(acc)
#else
#define __ESIMD_GET_SURF_HANDLE(acc) acc
#endif // __SYCL_DEVICE_ONLY__

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
/// @tparam N Number of elements to read; can be \c 8, \c 16 or \c 32.
/// @param p The base address.
/// @param offsets the vector of 32-bit offsets in bytes. For each lane \c i,
///   ((byte*)p + offsets[i]) must be element size aligned.
/// @param mask The access mask, defaults to all 1s.
/// @return A vector of elements read. Elements in masked out lanes are
///   undefined.
///
template <typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API std::enable_if_t<N == 8 || N == 16 || N == 32, simd<Tx, N>>
gather(const Tx *p, simd<uint32_t, N> offsets, simd_mask<N> mask = 1) {
  simd<uint64_t, N> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;

  if constexpr (sizeof(T) == 1) {
    auto Ret = __esimd_svm_gather<T, N, detail::ElemsPerAddrEncoding<4>()>(
        addrs.data(), detail::ElemsPerAddrEncoding<1>(), mask.data());
    return __esimd_rdregion<T, N * 4, N, /*VS*/ 0, N, 4>(Ret, 0);
  } else if constexpr (sizeof(T) == 2) {
    auto Ret = __esimd_svm_gather<T, N, detail::ElemsPerAddrEncoding<2>()>(
        addrs.data(), detail::ElemsPerAddrEncoding<2>(), mask.data());
    return __esimd_rdregion<T, N * 2, N, /*VS*/ 0, N, 2>(Ret, 0);
  } else
    return __esimd_svm_gather<T, N, detail::ElemsPerAddrEncoding<1>()>(
        addrs.data(), detail::ElemsPerAddrEncoding<1>(), mask.data());
}

/// Writes ("scatters") elements of the input vector to different memory
/// locations. Each memory location is base address plus an offset - a
/// value of the corresponding element in the input offset vector. Access to
/// any element's memory location can be disabled via the input mask.
/// @tparam Tx Element type, must be of size 4 or less.
/// @tparam N Number of elements to write; can be \c 8, \c 16 or \c 32.
/// @param p The base address.
/// @param offsets A vector of 32-bit offsets in bytes. For each lane \c i,
///   ((byte*)p + offsets[i]) must be element size aligned.
/// @param vals The vector to scatter.
/// @param mask The access mask, defaults to all 1s.
///
template <typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API std::enable_if_t<N == 8 || N == 16 || N == 32>
scatter(Tx *p, simd<uint32_t, N> offsets, simd<Tx, N> vals,
        simd_mask<N> mask = 1) {
  simd<uint64_t, N> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  if constexpr (sizeof(T) == 1) {
    simd<T, N * 4> D;
    D = __esimd_wrregion<T, N * 4, N, /*VS*/ 0, N, 4>(D.data(), vals.data(), 0);
    __esimd_svm_scatter<T, N, detail::ElemsPerAddrEncoding<4>()>(
        addrs.data(), D.data(), detail::ElemsPerAddrEncoding<1>(), mask.data());
  } else if constexpr (sizeof(T) == 2) {
    simd<T, N * 2> D;
    D = __esimd_wrregion<T, N * 2, N, /*VS*/ 0, N, 2>(D.data(), vals.data(), 0);
    __esimd_svm_scatter<T, N, detail::ElemsPerAddrEncoding<2>()>(
        addrs.data(), D.data(), detail::ElemsPerAddrEncoding<2>(), mask.data());
  } else
    __esimd_svm_scatter<T, N, detail::ElemsPerAddrEncoding<1>()>(
        addrs.data(), vals.data(), detail::ElemsPerAddrEncoding<1>(),
        mask.data());
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
__ESIMD_API simd<Tx, N> block_load(AccessorTy acc, uint32_t offset,
                                   Flags = {}) {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");

#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = __esimd_get_surface_index(
      detail::AccessorPrivateProxy::getNativeImageObj(acc));
#endif // __SYCL_DEVICE_ONLY__

  if constexpr (Flags::template alignment<simd<T, N>> >=
                detail::OperandSize::OWORD) {
#if defined(__SYCL_DEVICE_ONLY__)
    return __esimd_oword_ld<T, N>(surf_ind, offset >> 4);
#else
    return __esimd_oword_ld<T, N>(acc, offset >> 4);
#endif // __SYCL_DEVICE_ONLY__
  } else {
#if defined(__SYCL_DEVICE_ONLY__)
    return __esimd_oword_ld_unaligned<T, N>(surf_ind, offset);
#else
    return __esimd_oword_ld_unaligned<T, N>(acc, offset);
#endif // __SYCL_DEVICE_ONLY__
  }
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
__ESIMD_API void block_store(AccessorTy acc, uint32_t offset,
                             simd<Tx, N> vals) {
  constexpr unsigned Sz = sizeof(T) * N;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");

#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = __esimd_get_surface_index(
      detail::AccessorPrivateProxy::getNativeImageObj(acc));
  __esimd_oword_st<T, N>(surf_ind, offset >> 4, vals.data());
#else
  __esimd_oword_st<T, N>(acc, offset >> 4, vals.data());
#endif // __SYCL_DEVICE_ONLY__
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
  const auto si = __ESIMD_GET_SURF_HANDLE(acc);

  if constexpr (sizeof(T) < 4) {
    using Tint = std::conditional_t<std::is_integral_v<T>, T,
                                    detail::uint_type_t<sizeof(T)>>;
    using Treal = __raw_t<T>;
    simd<Tint, N> vals_int = bitcast<Tint, Treal, N>(std::move(vals).data());
    using PromoT =
        typename sycl::detail::conditional_t<std::is_signed<Tint>::value,
                                             int32_t, uint32_t>;
    const simd<PromoT, N> promo_vals = convert<PromoT>(std::move(vals_int));
    __esimd_scatter_scaled<PromoT, N, decltype(si), TypeSizeLog2, scale>(
        mask.data(), si, glob_offset, offsets.data(), promo_vals.data());
  } else {
    __esimd_scatter_scaled<T, N, decltype(si), TypeSizeLog2, scale>(
        mask.data(), si, glob_offset, offsets.data(), vals.data());
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
    using PromoT =
        typename sycl::detail::conditional_t<std::is_signed<Tint>::value,
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
    return __esimd_gather_masked_scaled2<T, N, decltype(si), TypeSizeLog2,
                                         scale>(si, glob_offset, offsets.data(),
                                                mask.data());
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
/// @param acc The accessor to gather from.
/// @param offsets Per-element offsets in bytes.
/// @param glob_offset Offset in bytes added to each individual element's offset
///   to compute actual memory access offset for that element.
/// @param mask Memory access mask. Elements with zero corresponding mask's
///   predicate are not accessed, their values in the resulting vector are
///   undefined.
///
template <typename T, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<(sizeof(T) <= 4) &&
                                 (N == 1 || N == 8 || N == 16 || N == 32) &&
                                 !std::is_pointer<AccessorTy>::value,
                             simd<T, N>>
gather(AccessorTy acc, simd<uint32_t, N> offsets, uint32_t glob_offset = 0,
       simd_mask<N> mask = 1) {

  return detail::gather_impl<T, N, AccessorTy>(acc, offsets, glob_offset, mask);
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
__ESIMD_API std::enable_if_t<(sizeof(T) <= 4) &&
                             (N == 1 || N == 8 || N == 16 || N == 32) &&
                             !std::is_pointer<AccessorTy>::value>
scatter(AccessorTy acc, simd<uint32_t, N> offsets, simd<T, N> vals,
        uint32_t glob_offset = 0, simd_mask<N> mask = 1) {

  detail::scatter_impl<T, N, AccessorTy>(acc, vals, offsets, glob_offset, mask);
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
/// @tparam Tx Element type of the returned vector. Must be 4 bytes in size.
/// @tparam N Number of pixels to access (matches the size of the \c offsets
///   vector). Must be 8, 16 or 32.
/// @tparam Mask A pixel's channel mask.
/// @param p The USM base pointer representing memory address of the access.
/// @param offsets Byte offsets of the pixels relative to the base pointer.
/// @param mask Memory access mask. Pixels with zero corresponding mask's
///   predicate are not accessed. Their values in the resulting vector are
///   undefined.
/// @return Read data - up to N*4 values of type \c Tx.
///
template <typename Tx, int N, rgba_channel_mask Mask,
          class T = detail::__raw_t<Tx>>
__ESIMD_API std::enable_if_t<(N == 8 || N == 16 || N == 32) && (sizeof(T) == 4),
                             simd<Tx, N * get_num_channels_enabled(Mask)>>
gather_rgba(const Tx *p, simd<uint32_t, N> offsets, simd_mask<N> mask = 1) {

  simd<uint64_t, N> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  return __esimd_svm_gather4_scaled<T, N, Mask>(addrs.data(), mask.data());
}

/// @anchor usm_scatter_rgba
/// Transpose and scatter pixels to given memory locations defined by the base
/// pointer \c p and \c offsets. Up to 4 32-bit data elements may be accessed at
/// each address depending on the channel mask \c Mask template parameter. Each
/// pixel's address must be 4 byte aligned. This is basically an inverse
/// operation for gather_rgba.
///
/// @tparam Tx Element type of the returned vector. Must be 4 bytes in size.
/// @tparam N Number of pixels to access (matches the size of the \c offsets
///   vector). Must be 8, 16 or 32.
/// @tparam Mask A pixel's channel mask.
/// @param p The USM base pointer representing memory address of the access.
/// @param vals values to be written.
/// @param offsets Byte offsets of the pixels relative to the base pointer.
/// @param mask Memory access mask. Pixels with zero corresponding mask's
///   predicate are not accessed. Their values in the resulting vector are
///   undefined.
///
template <typename Tx, int N, rgba_channel_mask Mask,
          class T = detail::__raw_t<Tx>>
__ESIMD_API std::enable_if_t<(N == 8 || N == 16 || N == 32) && (sizeof(T) == 4)>
scatter_rgba(Tx *p, simd<uint32_t, N> offsets,
             simd<Tx, N * get_num_channels_enabled(Mask)> vals,
             simd_mask<N> mask = 1) {
  simd<uint64_t, N> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  __esimd_svm_scatter4_scaled<T, N, Mask>(addrs.data(), vals.data(),
                                          mask.data());
}

/// @} sycl_esimd_memory

/// @cond ESIMD_DETAIL

namespace detail {
/// Check the legality of an atomic call in terms of size and type.
///
template <atomic_op Op, typename T, int N, unsigned NumSrc>
constexpr bool check_atomic() {
  if constexpr (!detail::isPowerOf2(N, 32)) {
    static_assert((detail::isPowerOf2(N, 32)),
                  "Execution size 1, 2, 4, 8, 16, 32 are supported");
    return false;
  }

  // No source operands.
  if constexpr (Op == atomic_op::inc || Op == atomic_op::dec) {
    if constexpr (NumSrc != 0) {
      static_assert(NumSrc == 0, "No source operands are expected");
      return false;
    }
    if constexpr (!is_type<T, uint16_t, uint32_t, uint64_t>()) {
      static_assert((is_type<T, uint16_t, uint32_t, uint64_t>()),
                    "Type UW, UD or UQ is expected");
      return false;
    }
    return true;
  }

  // One source integer operand.
  if constexpr (Op == atomic_op::add || Op == atomic_op::sub ||
                Op == atomic_op::min || Op == atomic_op::max ||
                Op == atomic_op::xchg || Op == atomic_op::bit_and ||
                Op == atomic_op::bit_or || Op == atomic_op::bit_xor ||
                Op == atomic_op::minsint || Op == atomic_op::maxsint) {
    if constexpr (NumSrc != 1) {
      static_assert(NumSrc == 1, "One source operand is expected");
      return false;
    }
    if constexpr ((Op != atomic_op::minsint && Op != atomic_op::maxsint) &&
                  !is_type<T, uint16_t, uint32_t, uint64_t>()) {
      static_assert((is_type<T, uint16_t, uint32_t, uint64_t>()),
                    "Type UW, UD or UQ is expected");
      return false;
    }
    if constexpr ((Op == atomic_op::minsint || Op == atomic_op::maxsint) &&
                  !is_type<T, int16_t, int32_t, int64_t>()) {
      static_assert((is_type<T, int16_t, int32_t, int64_t>()),
                    "Type W, D or Q is expected");
      return false;
    }
    return true;
  }

  // One source float operand.
  if constexpr (Op == atomic_op::fmax || Op == atomic_op::fmin ||
                Op == atomic_op::fadd || Op == atomic_op::fsub) {
    if constexpr (NumSrc != 1) {
      static_assert(NumSrc == 1, "One source operand is expected");
      return false;
    }
    if constexpr (!is_type<T, float, sycl::half>()) {
      static_assert((is_type<T, float, sycl::half>()),
                    "Type F or HF is expected");
      return false;
    }
    return true;
  }

  // Two source operands.
  if constexpr (Op == atomic_op::cmpxchg || Op == atomic_op::fcmpwr) {
    if constexpr (NumSrc != 2) {
      static_assert(NumSrc == 2, "Two source operands are expected");
      return false;
    }
    if constexpr (Op == atomic_op::cmpxchg &&
                  !is_type<T, uint16_t, uint32_t, uint64_t>()) {
      static_assert((is_type<T, uint16_t, uint32_t, uint64_t>()),
                    "Type UW, UD or UQ is expected");
      return false;
    }
    if constexpr (Op == atomic_op::fcmpwr && !is_type<T, float, sycl::half>()) {
      static_assert((is_type<T, float, sycl::half>()),
                    "Type F or HF is expected");
      return false;
    }
    return true;
  }
  // Unsupported svm atomic Op.
  return false;
}
} // namespace detail

/// @endcond ESIMD_DETAIL

/// @addtogroup sycl_esimd_memory_atomics
/// @{

/// @anchor usm_atomic_update0
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has no arguments in addition to the value at the memory location.
///
/// @tparam Op The atomic operation - can be \c atomic_op::inc or
///   atomic_op::dec.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param offset The vector of 32-bit offsets in bytes.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, Tx, N, 0>(), simd<Tx, N>>
atomic_update(Tx *p, simd<unsigned, N> offset, simd_mask<N> mask) {
  simd<uintptr_t, N> vAddr(reinterpret_cast<uintptr_t>(p));
  simd<uintptr_t, N> offset_i1 = convert<uintptr_t>(offset);
  vAddr += offset_i1;
  return __esimd_svm_atomic0<Op, T, N>(vAddr.data(), mask.data());
}

/// @anchor usm_atomic_update1
/// Atomically updates \c N memory locations represented by a USM pointer and
/// a vector of offsets relative to the pointer, and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 1 additional argument.
///
/// @tparam Op The atomic operation - can be one of the following:
/// \c atomic_op::add, \c atomic_op::sub, \c atomic_op::min, \c atomic_op::max,
/// \c atomic_op::xchg, \c atomic_op::bit_and, \c atomic_op::bit_or,
/// \c atomic_op::bit_xor, \c atomic_op::minsint, \c atomic_op::maxsint,
/// \c atomic_op::fmax, \c atomic_op::fmin.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @param p The USM pointer.
/// @param offset The vector of 32-bit offsets in bytes.
/// @param src0 The additional argument.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, Tx, N, 1>(), simd<Tx, N>>
atomic_update(Tx *p, simd<unsigned, N> offset, simd<Tx, N> src0,
              simd_mask<N> mask) {
  simd<uintptr_t, N> vAddr(reinterpret_cast<uintptr_t>(p));
  simd<uintptr_t, N> offset_i1 = convert<uintptr_t>(offset);
  vAddr += offset_i1;
  return __esimd_svm_atomic1<Op, T, N>(vAddr.data(), src0.data(), mask.data());
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
/// @param offset The vector of 32-bit offsets in bytes.
/// @param src0 The first additional argument (expected value).
/// @param src1 The second additional argument (new value).
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, Tx, N, 2>(), simd<Tx, N>>
atomic_update(Tx *p, simd<unsigned, N> offset, simd<Tx, N> src0,
              simd<Tx, N> src1, simd_mask<N> mask) {
  simd<uintptr_t, N> vAddr(reinterpret_cast<uintptr_t>(p));
  simd<uintptr_t, N> offset_i1 = convert<uintptr_t>(offset);
  vAddr += offset_i1;
  return __esimd_svm_atomic2<Op, T, N>(vAddr.data(), src0.data(), src1.data(),
                                       mask.data());
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
  /// Enable thread scheduling barrier.
  sw_barrier = 0x80
};

/// esimd::fence sets the memory read/write order.
/// @tparam cntl A bitmask composed from \c fence_mask bits.
///
__ESIMD_API void fence(fence_mask cntl) { __esimd_fence(cntl); }

/// Generic work-group barrier.
/// Performs barrier synchronization for all threads within the same thread
/// group. The barrier instruction causes the executing thread to wait until
/// all threads in the same thread group have executed the barrier instruction.
/// Memory ordering is also guaranteed by this instruction.
/// The behavior is undefined if this instruction is executed in divergent
/// control flow.
///
__ESIMD_API void barrier() {
  __esimd_fence(fence_mask::global_coherent_fence | fence_mask::local_barrier);
  __esimd_barrier();
}

/// Generic work-group split barrier
__ESIMD_API void sbarrier(split_barrier_action flag) { __esimd_sbarrier(flag); }

/// @} sycl_esimd_memory

/// @addtogroup sycl_esimd_memory_slm
/// @{

/// Declare per-work-group slm size.
/// @param size the requested size of the shared local memory for current work
/// group. Must be compile-time constant.
#ifdef __SYCL_DEVICE_ONLY__
// TODO slm_init should call __esimd_slm_init (TBD) and declared as __ESIMD_API
// on both host and device. Currently __ESIMD_API on device leads to:
// "... cannot call an undefined function without SYCL_EXTERNAL attribute"
__ESIMD_INTRIN
#else
__ESIMD_API
#endif
void slm_init(uint32_t size)
#ifdef __SYCL_DEVICE_ONLY__
    ;
#else
{
  throw sycl::feature_not_supported();
}
#endif // __SYCL_DEVICE_ONLY__

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
/// "accessor-based scatter", except that it does not have the accessor and the
/// global offset parameters.
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

/// Gather data from the Shared Local Memory at specified \c offsets and return
/// it as simd vector. See @ref usm_gather_rgba for information about the
/// operation semantics and parameter restrictions/interdependencies.
/// @tparam T The element type of the returned vector.
/// @tparam N The number of elements to access.
/// @tparam Mask Pixel's channel mask.
/// @param offsets Byte offsets within the SLM of each element.
/// @param mask Operation mask. All-1 by default.
/// @return Gathered data as an \c N - element vector.
///
template <typename T, int N, rgba_channel_mask Mask>
__ESIMD_API std::enable_if_t<(N == 8 || N == 16 || N == 32) && (sizeof(T) == 4),
                             simd<T, N * get_num_channels_enabled(Mask)>>
slm_gather_rgba(simd<uint32_t, N> offsets, simd_mask<N> mask = 1) {

  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  return __esimd_gather4_scaled<T, N, decltype(si), Mask>(
      mask.data(), si, 0 /*global_offset*/, offsets.data());
}

/// Gather data from the Shared Local Memory at specified \c offsets and return
/// it as simd vector. See @ref usm_gather_rgba for information about the
/// operation semantics and parameter restrictions/interdependencies.
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
  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
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

  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
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
  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  // offset in genx.oword.st is in owords
  __esimd_oword_st<detail::__raw_t<T>, N>(si, offset >> 4, vals.data());
}

/// Atomic update operation performed on SLM. No source operands version.
/// See description of template and function parameters in @ref
/// usm_atomic_update0 "atomic update" operation docs.
template <atomic_op Op, typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, N, 0>(), simd<Tx, N>>
slm_atomic_update(simd<uint32_t, N> offsets, simd_mask<N> mask) {
  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  return __esimd_dword_atomic0<Op, T, N>(mask.data(), si, offsets.data());
}

/// Atomic update operation performed on SLM. One source operands version.
/// See description of template and function parameters in @ref
/// usm_atomic_update1 "atomic update" operation docs.
template <atomic_op Op, typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, N, 1>(), simd<Tx, N>>
slm_atomic_update(simd<uint32_t, N> offsets, simd<Tx, N> src0,
                  simd_mask<N> mask) {
  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  return __esimd_dword_atomic1<Op, T, N>(mask.data(), si, offsets.data(),
                                         src0.data());
}

/// Atomic update operation performed on SLM. Two source operands version.
/// See description of template and function parameters in @ref
/// usm_atomic_update2 "atomic update" operation docs.
template <atomic_op Op, typename Tx, int N, class T = detail::__raw_t<Tx>>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, N, 2>(), simd<Tx, N>>
slm_atomic_update(simd<uint32_t, N> offsets, simd<Tx, N> src0, simd<Tx, N> src1,
                  simd_mask<N> mask) {
  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  return __esimd_dword_atomic2<Op, T, N>(mask.data(), si, offsets.data(),
                                         src0.data(), src1.data());
}

/// @} sycl_esimd_memory_slm

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

  const auto si = __ESIMD_GET_SURF_HANDLE(acc);
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
  const auto si = __ESIMD_GET_SURF_HANDLE(acc);
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
__ESIMD_API simd<T1, n1>
raw_sends_load(simd<T1, n1> msgDst, simd<T2, n2> msgSrc0, simd<T3, n3> msgSrc1,
               uint32_t exDesc, uint32_t msgDesc, uint8_t execSize,
               uint8_t sfid, uint8_t numSrc0, uint8_t numSrc1, uint8_t numDst,
               uint8_t isEOT = 0, uint8_t isSendc = 0, simd_mask<N> mask = 1) {
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
__ESIMD_API simd<T1, n1>
raw_send_load(simd<T1, n1> msgDst, simd<T2, n2> msgSrc0, uint32_t exDesc,
              uint32_t msgDesc, uint8_t execSize, uint8_t sfid, uint8_t numSrc0,
              uint8_t numDst, uint8_t isEOT = 0, uint8_t isSendc = 0,
              simd_mask<N> mask = 1) {
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
raw_sends_store(simd<T1, n1> msgSrc0, simd<T2, n2> msgSrc1, uint32_t exDesc,
                uint32_t msgDesc, uint8_t execSize, uint8_t sfid,
                uint8_t numSrc0, uint8_t numSrc1, uint8_t isEOT = 0,
                uint8_t isSendc = 0, simd_mask<N> mask = 1) {
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
__ESIMD_API void
raw_send_store(simd<T1, n1> msgSrc0, uint32_t exDesc, uint32_t msgDesc,
               uint8_t execSize, uint8_t sfid, uint8_t numSrc0,
               uint8_t isEOT = 0, uint8_t isSendc = 0, simd_mask<N> mask = 1) {
  constexpr unsigned _Width1 = n1 * sizeof(T1);
  static_assert(_Width1 % 32 == 0, "Invalid size for raw send msgSrc0");

  uint8_t modifier = ((isEOT & 0x1) << 1) | (isSendc & 0x1);
  __esimd_raw_send2_noresult<T1, n1, N>(modifier, execSize, mask.data(),
                                        numSrc0, sfid, exDesc, msgDesc,
                                        msgSrc0.data());
}
/// @} sycl_esimd_raw_send

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
           detail::getNextPowerOf2<Width>() * NBlocks;
  return Width * Height * NBlocks;
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
__ESIMD_API simd<T, N * NElts> lsc_slm_gather(simd<uint32_t, N> offsets,
                                              simd_mask<N> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  return __esimd_lsc_load_slm<T, cache_hint::none, cache_hint::none,
                              _AddressScale, _ImmOffset, _DS, _VS, _Transposed,
                              N>(pred.data(), offsets.data());
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
__ESIMD_API simd<T, NElts> lsc_slm_block_load(uint32_t offset) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  simd_mask<N> pred = 1;
  simd<uint32_t, N> offsets = offset;
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
__ESIMD_API
    std::enable_if_t<!std::is_pointer<AccessorTy>::value, simd<T, N * NElts>>
    lsc_gather(AccessorTy acc, simd<uint32_t, N> offsets,
               simd_mask<N> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = get_surface_index(acc);
  return __esimd_lsc_load_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                              _Transposed, N>(pred.data(), offsets.data(),
                                              surf_ind);
#else
  return __esimd_lsc_load_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                              _Transposed, N>(pred.data(), offsets.data(), acc);
#endif // __SYCL_DEVICE_ONLY__
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
__ESIMD_API
    std::enable_if_t<!std::is_pointer<AccessorTy>::value, simd<T, NElts>>
    lsc_block_load(AccessorTy acc, uint32_t offset) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  simd_mask<N> pred = 1;
  simd<uint32_t, N> offsets = offset;
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = get_surface_index(acc);
  return __esimd_lsc_load_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                              _Transposed, N>(pred.data(), offsets.data(),
                                              surf_ind);
#else
  return __esimd_lsc_load_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                              _Transposed, N>(pred.data(), offsets.data(), acc);
#endif // __SYCL_DEVICE_ONLY__
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
__ESIMD_API simd<T, N * NElts> lsc_gather(const T *p, simd<uint32_t, N> offsets,
                                          simd_mask<N> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  return __esimd_lsc_load_stateless<T, L1H, L3H, _AddressScale, _ImmOffset, _DS,
                                    _VS, _Transposed, N>(pred.data(),
                                                         addrs.data());
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
__ESIMD_API simd<T, NElts> lsc_block_load(const T *p) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  simd_mask<N> pred = 1;
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
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
lsc_prefetch(AccessorTy acc, simd<uint32_t, N> offsets, simd_mask<N> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = get_surface_index(acc);
  __esimd_lsc_prefetch_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(),
                                           surf_ind);
#else
  __esimd_lsc_prefetch_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(), acc);
#endif // __SYCL_DEVICE_ONLY__
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
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  simd_mask<N> pred = 1;
  simd<uint32_t, N> offsets = offset;
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = get_surface_index(acc);
  __esimd_lsc_prefetch_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(),
                                           surf_ind);
#else
  __esimd_lsc_prefetch_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                           _Transposed, N>(pred.data(), offsets.data(), acc);
#endif // __SYCL_DEVICE_ONLY__
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
__ESIMD_API void lsc_prefetch(const T *p, simd<uint32_t, N> offsets,
                              simd_mask<N> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __esimd_lsc_prefetch_stateless<T, L1H, L3H, _AddressScale, _ImmOffset, _DS,
                                 _VS, _Transposed, N>(pred.data(),
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
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  simd_mask<N> pred = 1;
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
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
__ESIMD_API void lsc_slm_scatter(simd<uint32_t, N> offsets,
                                 simd<T, N * NElts> vals,
                                 simd_mask<N> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  __esimd_lsc_store_slm<T, cache_hint::none, cache_hint::none, _AddressScale,
                        _ImmOffset, _DS, _VS, _Transposed, N>(
      pred.data(), offsets.data(), vals.data());
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
__ESIMD_API void lsc_slm_block_store(uint32_t offset, simd<T, NElts> vals) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  simd_mask<N> pred = 1;
  simd<uint32_t, N> offsets = offset;
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
lsc_scatter(AccessorTy acc, simd<uint32_t, N> offsets, simd<T, N * NElts> vals,
            simd_mask<N> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = get_surface_index(acc);
  __esimd_lsc_store_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                        _Transposed, N>(pred.data(), offsets.data(),
                                        vals.data(), surf_ind);
#else
  __esimd_lsc_store_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                        _Transposed, N>(pred.data(), offsets.data(),
                                        vals.data(), acc);
#endif // __SYCL_DEVICE_ONLY__
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
lsc_block_store(AccessorTy acc, uint32_t offset, simd<T, NElts> vals) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  simd_mask<N> pred = 1;
  simd<uint32_t, N> offsets = offset;
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = get_surface_index(acc);
  __esimd_lsc_store_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                        _Transposed, N>(pred.data(), offsets.data(),
                                        vals.data(), surf_ind);
#else
  __esimd_lsc_store_bti<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                        _Transposed, N>(pred.data(), offsets.data(),
                                        vals.data(), acc);
#endif // __SYCL_DEVICE_ONLY__
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
__ESIMD_API void lsc_scatter(T *p, simd<uint32_t, N> offsets,
                             simd<T, N * NElts> vals, simd_mask<N> pred = 1) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  __esimd_lsc_store_stateless<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                              _Transposed, N>(pred.data(), addrs.data(),
                                              vals.data());
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
__ESIMD_API void lsc_block_store(T *p, simd<T, NElts> vals) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  simd_mask<N> pred = 1;
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
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
__ESIMD_API simd<T, N> lsc_load2d(const T *Ptr, unsigned SurfaceWidth,
                                  unsigned SurfaceHeight, unsigned SurfacePitch,
                                  int X, int Y) {
  static_assert(!Transposed || !Transformed,
                "Transposed and transformed is not supported");
  static_assert(!Transposed || (Transposed && NBlocks == 1),
                "Transposed expected to be 1 block only");
  constexpr int ElemsPerDword = 4 / sizeof(T);
  constexpr int GRFRowSize = Transposed ? BlockHeight : BlockWidth;
  constexpr int GRFRowPitch = detail::getNextPowerOf2<GRFRowSize>();
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
  simd_mask<N> pred = 1;
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
///
template <typename T, int BlockWidth, int BlockHeight = 1, int NBlocks = 1,
          bool Transposed = false, bool Transformed = false,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none,
          int N = detail::get_lsc_block_2d_data_size<
              T, NBlocks, BlockHeight, BlockWidth, Transposed, Transformed>()>
__ESIMD_API void lsc_prefetch2d(const T *Ptr, unsigned SurfaceWidth,
                                unsigned SurfaceHeight, unsigned SurfacePitch,
                                int X, int Y) {
  constexpr lsc_data_size DS =
      detail::finalize_data_size<T, lsc_data_size::default_size>();
  simd_mask<N> pred = 1;
  uintptr_t surf_addr = reinterpret_cast<uintptr_t>(Ptr);
  constexpr detail::lsc_data_order _Transposed =
      Transposed ? detail::lsc_data_order::transpose
                 : detail::lsc_data_order::nontranspose;
  __esimd_lsc_prefetch2d_stateless<T, L1H, L3H, DS, _Transposed, NBlocks,
                                   BlockWidth, BlockHeight, Transformed, N>(
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
                             int X, int Y, simd<T, N> Vals) {
  constexpr lsc_data_size DS =
      detail::finalize_data_size<T, lsc_data_size::default_size>();
  simd_mask<N> pred = 1;
  uintptr_t surf_addr = reinterpret_cast<uintptr_t>(Ptr);
  __esimd_lsc_store2d_stateless<T, L1H, L3H, DS,
                                detail::lsc_data_order::nontranspose, 1u,
                                BlockWidth, BlockHeight, false, N>(
      pred.data(), surf_addr, SurfaceWidth, SurfaceHeight, SurfacePitch, X, Y,
      Vals.data());
}

/// SLM atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam T is element type.
/// @tparam Op is operation type.
/// @tparam DS is the data size.
/// @tparam N is the number of channels (platform dependent).
/// @param offsets is the zero-based offsets.
/// @param pred is predicates.
///
template <typename T, atomic_op Op,
          lsc_data_size DS = lsc_data_size::default_size, int N>
__ESIMD_API simd<T, N> lsc_slm_atomic_update(simd<uint32_t, N> offsets,
                                             simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 0>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  return __esimd_lsc_xatomic_slm_0<T, _Op, cache_hint::none, cache_hint::none,
                                   _AddressScale, _ImmOffset, _DS, _VS,
                                   _Transposed, N>(pred.data(), offsets.data());
}

/// SLM atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam T is element type.
/// @tparam Op is operation type.
/// @tparam DS is the data size.
/// @tparam N is the number of channels (platform dependent).
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
template <typename T, atomic_op Op,
          lsc_data_size DS = lsc_data_size::default_size, int N>
__ESIMD_API simd<T, N> lsc_slm_atomic_update(simd<uint32_t, N> offsets,
                                             simd<T, N> src0,
                                             simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 1>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  return __esimd_lsc_xatomic_slm_1<T, _Op, cache_hint::none, cache_hint::none,
                                   _AddressScale, _ImmOffset, _DS, _VS,
                                   _Transposed, N>(pred.data(), offsets.data(),
                                                   src0.data());
}

/// SLM atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.slm
///
/// @tparam T is element type.
/// @tparam Op is operation type.
/// @tparam DS is the data size.
/// @tparam N is the number of channels (platform dependent).
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
/// @param pred is predicates.
///
template <typename T, atomic_op Op,
          lsc_data_size DS = lsc_data_size::default_size, int N>
__ESIMD_API simd<T, N> lsc_slm_atomic_update(simd<uint32_t, N> offsets,
                                             simd<T, N> src0, simd<T, N> src1,
                                             simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 2>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  return __esimd_lsc_xatomic_slm_2<T, _Op, cache_hint::none, cache_hint::none,
                                   _AddressScale, _ImmOffset, _DS, _VS,
                                   _Transposed, N>(pred.data(), offsets.data(),
                                                   src0.data(), src1.data());
}

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam T is element type.
/// @tparam Op is operation type.
/// @tparam DS is the data size.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets.
/// @param pred is predicates.
///
template <typename T, atomic_op Op,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L3H = cache_hint::none, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value, simd<T, N>>
lsc_atomic_update(AccessorTy acc, simd<uint32_t, N> offsets,
                  simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 0>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = get_surface_index(acc);
  return __esimd_lsc_xatomic_bti_0<T, _Op, cache_hint::none, L3H, _AddressScale,
                                   _ImmOffset, _DS, _VS, _Transposed, N>(
      pred.data(), offsets.data(), surf_ind);
#else
  return __esimd_lsc_xatomic_bti_0<T, _Op, cache_hint::none, L3H, _AddressScale,
                                   _ImmOffset, _DS, _VS, _Transposed, N>(
      pred.data(), offsets.data(), acc);
#endif // __SYCL_DEVICE_ONLY__
}

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam T is element type.
/// @tparam Op is operation type.
/// @tparam DS is the data size.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
template <typename T, atomic_op Op,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L3H = cache_hint::none, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value, simd<T, N>>
lsc_atomic_update(AccessorTy acc, simd<uint32_t, N> offsets, simd<T, N> src0,
                  simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 1>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = get_surface_index(acc);
  return __esimd_lsc_xatomic_bti_1<T, _Op, cache_hint::none, L3H, _AddressScale,
                                   _ImmOffset, _DS, _VS, _Transposed, N>(
      pred.data(), offsets.data(), src0.data(), surf_ind);
#else
  return __esimd_lsc_xatomic_bti_1<T, _Op, cache_hint::none, L3H, _AddressScale,
                                   _ImmOffset, _DS, _VS, _Transposed, N>(
      pred.data(), offsets.data(), src0.data(), acc);
#endif // __SYCL_DEVICE_ONLY__
}

/// Accessor-based atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam T is element type.
/// @tparam Op is operation type.
/// @tparam DS is the data size.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @tparam AccessorTy is the \ref sycl::accessor type.
/// @param acc is the SYCL accessor.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
/// @param pred is predicates.
///
template <typename T, atomic_op Op,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L3H = cache_hint::none, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<!std::is_pointer<AccessorTy>::value, simd<T, N>>
lsc_atomic_update(AccessorTy acc, simd<uint32_t, N> offsets, simd<T, N> src0,
                  simd<T, N> src1, simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 2>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
#if defined(__SYCL_DEVICE_ONLY__)
  auto surf_ind = get_surface_index(acc);
  return __esimd_lsc_xatomic_bti_2<T, _Op, cache_hint::none, L3H, _AddressScale,
                                   _ImmOffset, _DS, _VS, _Transposed, N>(
      pred.data(), offsets.data(), src0.data(), src1.data(), surf_ind);
#else
  return __esimd_lsc_xatomic_bti_2<T, _Op, cache_hint::none, L3H, _AddressScale,
                                   _ImmOffset, _DS, _VS, _Transposed, N>(
      pred.data(), offsets.data(), src0.data(), src1.data(), acc);
#endif // __SYCL_DEVICE_ONLY__
}

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam T is element type.
/// @tparam Op is operation type.
/// @tparam DS is the data size.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param pred is predicates.
///
template <typename T, atomic_op Op,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L3H = cache_hint::none, int N>
__ESIMD_API simd<T, N> lsc_atomic_update(T *p, simd<uint32_t, N> offsets,
                                         simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 0>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  return __esimd_lsc_xatomic_stateless_0<T, _Op, cache_hint::none, L3H,
                                         _AddressScale, _ImmOffset, _DS, _VS,
                                         _Transposed, N>(pred.data(),
                                                         addrs.data());
}

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam T is element type.
/// @tparam Op is operation type.
/// @tparam DS is the data size.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param pred is predicates.
///
template <typename T, atomic_op Op,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L3H = cache_hint::none, int N>
__ESIMD_API simd<T, N> lsc_atomic_update(T *p, simd<uint32_t, N> offsets,
                                         simd<T, N> src0, simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 1>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  return __esimd_lsc_xatomic_stateless_1<T, _Op, cache_hint::none, L3H,
                                         _AddressScale, _ImmOffset, _DS, _VS,
                                         _Transposed, N>(
      pred.data(), addrs.data(), src0.data());
}

/// USM pointer atomic.
/// Supported platforms: DG2, PVC
/// VISA instruction: lsc_atomic_<OP>.ugm
///
/// @tparam T is element type.
/// @tparam Op is operation type.
/// @tparam DS is the data size.
/// @tparam L3H is L3 cache hint.
/// @tparam N is the number of channels (platform dependent).
/// @param p is the base pointer.
/// @param offsets is the zero-based offsets.
/// @param src0 is the first atomic operand.
/// @param src1 is the second atomic operand.
/// @param pred is predicates.
///
template <typename T, atomic_op Op,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L3H = cache_hint::none, int N>
__ESIMD_API simd<T, N> lsc_atomic_update(T *p, simd<uint32_t, N> offsets,
                                         simd<T, N> src0, simd<T, N> src1,
                                         simd_mask<N> pred) {
  detail::check_lsc_vector_size<1>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_atomic<Op, 2>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<1>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::nontranspose;
  constexpr detail::lsc_atomic_op _Op = detail::to_lsc_atomic_op<Op>();
  simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  addrs += convert<uintptr_t>(offsets);
  return __esimd_lsc_xatomic_stateless_2<T, _Op, cache_hint::none, L3H,
                                         _AddressScale, _ImmOffset, _DS, _VS,
                                         _Transposed, N>(
      pred.data(), addrs.data(), src0.data(), src1.data());
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
__ESIMD_API void lsc_fence(simd_mask<N> pred = 1) {
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
