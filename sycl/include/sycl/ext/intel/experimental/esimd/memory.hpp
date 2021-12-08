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

/** @file
 * Memory access intrinsics.
 */

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace intel {
namespace experimental {
namespace esimd {

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

// Cache hint usage in most memory APIs is deprecated, as is not really
// supported by the underlying dataport messages.
// This is auxiliary code to warn if APIs use cache hints different from
// CacheHint::None.
template <CacheHint L1H, CacheHint L3H, class T = void> struct IfNotNone;

template <CacheHint L1H, CacheHint L3H>
struct IfNotNone<
    L1H, L3H,
    std::enable_if_t<L1H == CacheHint::None && L3H == CacheHint::None>> {
  static inline constexpr bool warn() { return false; }
};

template <CacheHint L1H, CacheHint L3H>
struct IfNotNone<
    L1H, L3H,
    std::enable_if_t<L1H != CacheHint::None || L3H != CacheHint::None>> {
  [[deprecated("cache hints are deprecated in this API and are "
               "ignored")]] static inline constexpr bool
  warn() {
    return true;
  }
};
} // namespace detail

/// Get surface index corresponding to a SYCL accessor.
///
/// \param acc a SYCL buffer or image accessor.
/// \return the index of the corresponding surface (aka "binding table index").
///
/// \ingroup sycl_esimd
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
/// Flat-address gather.
/// \ingroup sycl_esimd
template <typename T, int n, int ElemsPerAddr = 1,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
__ESIMD_API std::enable_if_t<((n == 8 || n == 16 || n == 32) &&
                              (ElemsPerAddr == 1 || ElemsPerAddr == 2 ||
                               ElemsPerAddr == 4)),
                             simd<T, n * ElemsPerAddr>>
gather(const T *p, simd<uint32_t, n> offsets, simd_mask<n> pred = 1) {
  detail::IfNotNone<L1H, L3H>::warn();
  simd<uint64_t, n> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, n> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;

  if constexpr (sizeof(T) == 1 && ElemsPerAddr == 2) {
    auto Ret = __esimd_svm_gather<T, n, detail::ElemsPerAddrEncoding<4>()>(
        addrs.data(), detail::ElemsPerAddrEncoding<ElemsPerAddr>(),
        pred.data());
    return __esimd_rdregion<T, n * 4, n * ElemsPerAddr, /*VS*/ 4, 2, 1>(Ret, 0);
  } else if constexpr (sizeof(T) == 1 && ElemsPerAddr == 1) {
    auto Ret = __esimd_svm_gather<T, n, detail::ElemsPerAddrEncoding<4>()>(
        addrs.data(), detail::ElemsPerAddrEncoding<ElemsPerAddr>(),
        pred.data());
    return __esimd_rdregion<T, n * 4, n * ElemsPerAddr, /*VS*/ 0, n, 4>(Ret, 0);
  } else if constexpr (sizeof(T) == 2 && ElemsPerAddr == 1) {
    auto Ret = __esimd_svm_gather<T, n, detail::ElemsPerAddrEncoding<2>()>(
        addrs.data(), detail::ElemsPerAddrEncoding<2>(), pred.data());
    return __esimd_rdregion<T, n * 2, n, /*VS*/ 0, n, 2>(Ret, 0);
  } else if constexpr (sizeof(T) == 2)
    return __esimd_svm_gather<T, n,
                              detail::ElemsPerAddrEncoding<ElemsPerAddr>()>(
        addrs.data(), detail::ElemsPerAddrEncoding<2 * ElemsPerAddr>(),
        pred.data());
  else
    return __esimd_svm_gather<T, n,
                              detail::ElemsPerAddrEncoding<ElemsPerAddr>()>(
        addrs.data(), detail::ElemsPerAddrEncoding<ElemsPerAddr>(),
        pred.data());
}

/// Flat-address scatter.
/// \ingroup sycl_esimd
template <typename T, int n, int ElemsPerAddr = 1>
__ESIMD_API std::enable_if_t<((n == 8 || n == 16 || n == 32) &&
                              (ElemsPerAddr == 1 || ElemsPerAddr == 2 ||
                               ElemsPerAddr == 4))>
scatter(T *p, simd<uint32_t, n> offsets, simd<T, n * ElemsPerAddr> vals,
        simd_mask<n> pred = 1) {
  simd<uint64_t, n> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, n> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  if constexpr (sizeof(T) == 1 && ElemsPerAddr == 2) {
    simd<T, n * 4> D;
    D = __esimd_wrregion<T, n * 4, n * ElemsPerAddr, /*VS*/ 4, 2, 1>(
        D.data(), vals.data(), 0);
    __esimd_svm_scatter<T, n, detail::ElemsPerAddrEncoding<4>()>(
        addrs.data(), D.data(), detail::ElemsPerAddrEncoding<ElemsPerAddr>(),
        pred.data());
  } else if constexpr (sizeof(T) == 1 && ElemsPerAddr == 1) {
    simd<T, n * 4> D;
    D = __esimd_wrregion<T, n * 4, n * ElemsPerAddr, /*VS*/ 0, n, 4>(
        D.data(), vals.data(), 0);
    __esimd_svm_scatter<T, n, detail::ElemsPerAddrEncoding<4>()>(
        addrs.data(), D.data(), detail::ElemsPerAddrEncoding<ElemsPerAddr>(),
        pred.data());
  } else if constexpr (sizeof(T) == 2 && ElemsPerAddr == 1) {
    simd<T, n * 2> D;
    D = __esimd_wrregion<T, n * 2, n, /*VS*/ 0, n, 2>(D.data(), vals.data(), 0);
    __esimd_svm_scatter<T, n, detail::ElemsPerAddrEncoding<2>()>(
        addrs.data(), D.data(), detail::ElemsPerAddrEncoding<2>(), pred.data());
  } else if constexpr (sizeof(T) == 2)
    __esimd_svm_scatter<T, n, detail::ElemsPerAddrEncoding<ElemsPerAddr>()>(
        addrs.data(), vals.data(),
        detail::ElemsPerAddrEncoding<2 * ElemsPerAddr>(), pred.data());
  else
    __esimd_svm_scatter<T, n, detail::ElemsPerAddrEncoding<ElemsPerAddr>()>(
        addrs.data(), vals.data(), detail::ElemsPerAddrEncoding<ElemsPerAddr>(),
        pred.data());
}

template <typename T, int n, int ElemsPerAddr = 1,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
__SYCL_DEPRECATED("use scatter.")
__ESIMD_API std::enable_if_t<((n == 8 || n == 16 || n == 32) &&
                              (ElemsPerAddr == 1 || ElemsPerAddr == 2 ||
                               ElemsPerAddr ==
                                   4))> scatter1(T *p,
                                                 simd<T, n * ElemsPerAddr> vals,
                                                 simd<uint32_t, n> offsets,
                                                 simd_mask<n> pred = 1) {
  detail::IfNotNone<L1H, L3H>::warn();
  scatter<T, n, ElemsPerAddr>(p, offsets, vals, pred);
}

/// Flat-address block-load.
/// \ingroup sycl_esimd
template <typename T, int n, typename Flags = vector_aligned_tag,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None,
          typename = std::enable_if_t<is_simd_flag_type_v<Flags>>>
__ESIMD_API simd<T, n> block_load(const T *addr, Flags = {}) {
  detail::IfNotNone<L1H, L3H>::warn();
  constexpr unsigned Sz = sizeof(T) * n;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");

  uintptr_t Addr = reinterpret_cast<uintptr_t>(addr);
  if constexpr (Flags::template alignment<simd<T, n>> >=
                detail::OperandSize::OWORD) {
    return __esimd_svm_block_ld<T, n>(Addr);
  } else {
    return __esimd_svm_block_ld_unaligned<T, n>(Addr);
  }
}

/// Accessor-based block-load.
/// \ingroup sycl_esimd
template <typename T, int n, typename AccessorTy,
          typename Flags = vector_aligned_tag,
          typename = std::enable_if_t<is_simd_flag_type_v<Flags>>>
__ESIMD_API simd<T, n> block_load(AccessorTy acc, uint32_t offset, Flags = {}) {
  constexpr unsigned Sz = sizeof(T) * n;
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

  if constexpr (Flags::template alignment<simd<T, n>> >=
                detail::OperandSize::OWORD) {
#if defined(__SYCL_DEVICE_ONLY__)
    return __esimd_oword_ld<T, n>(surf_ind, offset >> 4);
#else
    return __esimd_oword_ld<T, n>(acc, offset >> 4);
#endif // __SYCL_DEVICE_ONLY__
  } else {
#if defined(__SYCL_DEVICE_ONLY__)
    return __esimd_oword_ld_unaligned<T, n>(surf_ind, offset);
#else
    return __esimd_oword_ld_unaligned<T, n>(acc, offset);
#endif // __SYCL_DEVICE_ONLY__
  }
}

/// Flat-address block-store.
/// \ingroup sycl_esimd
// TODO the above note about cache hints applies to this API as well.
template <typename T, int n, CacheHint L1H = CacheHint::None,
          CacheHint L3H = CacheHint::None>
__ESIMD_API void block_store(T *p, simd<T, n> vals) {
  detail::IfNotNone<L1H, L3H>::warn();
  constexpr unsigned Sz = sizeof(T) * n;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 8 * detail::OperandSize::OWORD,
                "block size must be at most 8 owords");

  uintptr_t Addr = reinterpret_cast<uintptr_t>(p);
  __esimd_svm_block_st<T, n>(Addr, vals.data());
}

/// Accessor-based block-store.
/// \ingroup sycl_esimd
template <typename T, int n, typename AccessorTy>
__ESIMD_API void block_store(AccessorTy acc, uint32_t offset, simd<T, n> vals) {
  constexpr unsigned Sz = sizeof(T) * n;
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
  __esimd_oword_st<T, n>(surf_ind, offset >> 4, vals.data());
#else
  __esimd_oword_st<T, n>(acc, offset >> 4, vals.data());
#endif // __SYCL_DEVICE_ONLY__
}

// Implementations of accessor-based gather and scatter functions
namespace detail {
template <typename T, int N, typename AccessorTy>
ESIMD_INLINE
    ESIMD_NODEBUG std::enable_if_t<(sizeof(T) <= 4) &&
                                   (N == 1 || N == 8 || N == 16 || N == 32) &&
                                   !std::is_pointer<AccessorTy>::value>
    scatter_impl(AccessorTy acc, simd<T, N> vals, simd<uint32_t, N> offsets,
                 uint32_t glob_offset, simd_mask<N> pred) {

  constexpr int TypeSizeLog2 = detail::ElemsPerAddrEncoding<sizeof(T)>();
  // TODO (performance) use hardware-supported scale once BE supports it
  constexpr int16_t scale = 0;
  const auto si = __ESIMD_GET_SURF_HANDLE(acc);

  if constexpr (sizeof(T) < 4) {
    static_assert(std::is_integral<T>::value,
                  "only integral 1- & 2-byte types are supported");
    using PromoT =
        typename sycl::detail::conditional_t<std::is_signed<T>::value, int32_t,
                                             uint32_t>;
    const simd<PromoT, N> promo_vals = convert<PromoT>(vals);
    __esimd_scatter_scaled<PromoT, N, decltype(si), TypeSizeLog2, scale>(
        pred.data(), si, glob_offset, offsets.data(), promo_vals.data());
  } else {
    __esimd_scatter_scaled<T, N, decltype(si), TypeSizeLog2, scale>(
        pred.data(), si, glob_offset, offsets.data(), vals.data());
  }
}

template <typename T, int N, typename AccessorTy>
ESIMD_INLINE ESIMD_NODEBUG std::enable_if_t<
    (sizeof(T) <= 4) && (N == 1 || N == 8 || N == 16 || N == 32) &&
        !std::is_pointer<AccessorTy>::value,
    simd<T, N>>
gather_impl(AccessorTy acc, simd<uint32_t, N> offsets, uint32_t glob_offset,
            simd_mask<N> pred) {

  constexpr int TypeSizeLog2 = detail::ElemsPerAddrEncoding<sizeof(T)>();
  // TODO (performance) use hardware-supported scale once BE supports it
  constexpr uint32_t scale = 0;
  const auto si = get_surface_index(acc);

  if constexpr (sizeof(T) < 4) {
    static_assert(std::is_integral<T>::value,
                  "only integral 1- & 2-byte types are supported");
    using PromoT =
        typename sycl::detail::conditional_t<std::is_signed<T>::value, int32_t,
                                             uint32_t>;
    const simd<PromoT, N> promo_vals =
        __esimd_gather_masked_scaled2<PromoT, N, decltype(si), TypeSizeLog2,
                                      scale>(si, glob_offset, offsets.data(),
                                             pred.data());
    return convert<T>(promo_vals);
  } else {
    return __esimd_gather_masked_scaled2<T, N, decltype(si), TypeSizeLog2,
                                         scale>(si, glob_offset, offsets.data(),
                                                pred.data());
  }
}

} // namespace detail

/// Accessor-based gather.
///
/// Collects elements located at given offsets in an accessor and returns them
/// as a single \ref simd object. An element can be 1, 2 or 4-byte value.
///
/// \tparam T is element type; can only be a 1,2,4-byte integer or \c float.
/// \tparam N is the number of elements.
/// \tparam AccessorTy is \ref sycl::accessor type.
/// \tparam L1H is L1 cache hint.
/// \tparam L3H is L3 cache hint.
/// \param acc is the accessor to gather from.
/// \param offsets is per-element offsets.
/// \param glob_offset is offset added to each individual element's offset to
/// compute actual memory access offset for that element.
///
/// \ingroup sycl_esimd
template <typename T, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<(sizeof(T) <= 4) &&
                                 (N == 1 || N == 8 || N == 16 || N == 32) &&
                                 !std::is_pointer<AccessorTy>::value,
                             simd<T, N>>
gather(AccessorTy acc, simd<uint32_t, N> offsets, uint32_t glob_offset = 0,
       simd_mask<N> pred = 1) {

  return detail::gather_impl<T, N, AccessorTy>(acc, offsets, glob_offset, pred);
}

template <typename T, int N, typename AccessorTy,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
__ESIMD_API std::enable_if_t<(sizeof(T) <= 4) &&
                                 (N == 1 || N == 8 || N == 16 || N == 32) &&
                                 !std::is_pointer<AccessorTy>::value,
                             simd<T, N>>
gather1(AccessorTy acc, simd<uint32_t, N> offsets, uint32_t glob_offset = 0,
        simd_mask<N> pred = 1) {

  if constexpr (sizeof(T) != 1) {
    glob_offset *= sizeof(T);
    offsets *= sizeof(T);
  }
  detail::IfNotNone<L1H, L3H>::warn();
  return gather<T, N, AccessorTy>(acc, offsets, glob_offset, pred);
}

/// Accessor-based scatter.
///
/// Writes elements of a \ref simd object into an accessor at given offsets.
/// An element can be 1, 2 or 4-byte value.
///
/// \tparam T is element type; can only be a 1,2,4-byte integer or \c float.
/// \tparam N is the number of elements.
/// \tparam AccessorTy is \ref sycl::accessor type.
/// \param acc is the accessor to scatter to.
/// \param offsets is per-element offsets.
/// \param vals is values to write.
/// \param glob_offset is offset added to each individual element's offset to
/// compute actual memory access offset for that element.
/// \param pred is per-element predicates; elements with zero corresponding
/// predicates are not written.
///
/// \ingroup sycl_esimd
template <typename T, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<(sizeof(T) <= 4) &&
                             (N == 1 || N == 8 || N == 16 || N == 32) &&
                             !std::is_pointer<AccessorTy>::value>
scatter(AccessorTy acc, simd<uint32_t, N> offsets, simd<T, N> vals,
        uint32_t glob_offset = 0, simd_mask<N> pred = 1) {

  detail::scatter_impl<T, N, AccessorTy>(acc, vals, offsets, glob_offset, pred);
}

template <typename T, int N, typename AccessorTy,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
__SYCL_DEPRECATED("use scatter.")
__ESIMD_API std::enable_if_t<
    (sizeof(T) <= 4) && (N == 1 || N == 8 || N == 16 || N == 32) &&
    !std::is_pointer<AccessorTy>::value> scatter1(AccessorTy acc,
                                                  simd<T, N> vals,
                                                  simd<uint32_t, N> offsets,
                                                  uint32_t glob_offset = 0,
                                                  simd_mask<N> pred = 1) {

  if constexpr (sizeof(T) != 1) {
    glob_offset *= sizeof(T);
    offsets *= sizeof(T);
  }
  detail::IfNotNone<L1H, L3H>::warn();
  scatter<T, N, AccessorTy>(acc, offsets, vals, glob_offset, pred);
}

/// Load a scalar value from an accessor.
/// @tparam T type of the value
/// @tparam AccessorTy type of the accessor
/// @param offset offset in bytes
/// @return the loaded value
/// \ingroup sycl_esimd
template <typename T, typename AccessorTy>
__ESIMD_API T scalar_load(AccessorTy acc, uint32_t offset) {
  const simd<T, 1> Res =
      gather<T, 1, AccessorTy>(acc, simd<uint32_t, 1>(offset));
  return Res[0];
}

template <typename T, typename AccessorTy, CacheHint L1H = CacheHint::None,
          CacheHint L3H = CacheHint::None>
__SYCL_DEPRECATED("use scalar_load.")
__ESIMD_API T scalar_load1(AccessorTy acc, uint32_t offset) {
  detail::IfNotNone<L1H, L3H>::warn();
  const simd<T, 1> Res =
      gather1<T, 1, AccessorTy>(acc, simd<uint32_t, 1>(offset));
  return Res[0];
}

/// Store a scalar value into an accessor.
/// \ingroup sycl_esimd
template <typename T, typename AccessorTy>
__ESIMD_API void scalar_store(AccessorTy acc, uint32_t offset, T val) {
  scatter<T, 1, AccessorTy>(acc, simd<uint32_t, 1>(offset), simd<T, 1>(val));
}

template <typename T, typename AccessorTy, CacheHint L1H = CacheHint::None,
          CacheHint L3H = CacheHint::None>
__SYCL_DEPRECATED("use scalar_load.")
__ESIMD_API void scalar_store1(AccessorTy acc, uint32_t offset, T val) {
  detail::IfNotNone<L1H, L3H>::warn();
  scatter1<T, 1, AccessorTy>(acc, simd<T, 1>(val), simd<uint32_t, 1>(offset));
}

/// Gathering read for the given starting pointer \p p and \p offsets.
/// Up to 4 data elements may be accessed at each address depending on the
/// enabled channel \p Mask.
/// \tparam T element type of the returned vector. Must be 4-byte.
/// \tparam N size of the \p offsets vector. Must be 16 or 32.
/// \tparam Mask represents a pixel's channel mask.
/// @param p the USM pointer.
/// @param offsets byte-offsets within the \p buffer to be gathered.
/// @param pred predication control used for masking lanes.
/// \ingroup sycl_esimd
template <typename T, int N, rgba_channel_mask Mask>
__ESIMD_API std::enable_if_t<(N == 16 || N == 32) && (sizeof(T) == 4),
                             simd<T, N * get_num_channels_enabled(Mask)>>
gather_rgba(const T *p, simd<uint32_t, N> offsets, simd_mask<N> pred = 1) {

  simd<uint64_t, N> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  return __esimd_svm_gather4_scaled<T, N, Mask>(addrs.data(), pred.data());
}

/// Flat-address gather4.
/// Only allow simd-16 and simd-32.
/// \ingroup sycl_esimd
template <typename T, int n, rgba_channel_mask Mask,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
__SYCL_DEPRECATED("use gather_rgba.")
__ESIMD_API std::enable_if_t<
    (n == 16 || n == 32) && (sizeof(T) == 4),
    simd<T, n * get_num_channels_enabled(Mask)>> gather4(const T *p,
                                                         simd<uint32_t, n>
                                                             offsets,
                                                         simd_mask<n> pred =
                                                             1) {
  detail::IfNotNone<L1H, L3H>::warn();
  return gather_rgba<T, n, Mask>(p, offsets, pred);
}

/// Scatter write for the given starting pointer \p p and \p offsets.
/// Up to 4 data elements may be written at each address depending on the
/// enabled channel \p Mask.
/// \tparam T element type of the input vector. Must be 4-byte.
/// \tparam N size of the \p offsets vector. Must be 16 or 32.
/// \tparam Mask represents a pixel's channel mask.
/// @param p the USM pointer.
/// @param vals values to be written.
/// @param offsets byte-offsets within the \p buffer to be written.
/// @param pred predication control used for masking lanes.
/// \ingroup sycl_esimd
template <typename T, int N, rgba_channel_mask Mask>
__ESIMD_API std::enable_if_t<(N == 16 || N == 32) && (sizeof(T) == 4)>
scatter_rgba(T *p, simd<uint32_t, N> offsets,
             simd<T, N * get_num_channels_enabled(Mask)> vals,
             simd_mask<N> pred = 1) {
  simd<uint64_t, N> offsets_i = convert<uint64_t>(offsets);
  simd<uint64_t, N> addrs(reinterpret_cast<uint64_t>(p));
  addrs = addrs + offsets_i;
  __esimd_svm_scatter4_scaled<T, N, Mask>(addrs.data(), vals.data(),
                                          pred.data());
}

/// Flat-address scatter4.
/// \ingroup sycl_esimd
template <typename T, int n, rgba_channel_mask Mask,
          CacheHint L1H = CacheHint::None, CacheHint L3H = CacheHint::None>
__SYCL_DEPRECATED("use scatter_rgba.")
__ESIMD_API std::enable_if_t<(n == 16 || n == 32) && sizeof(T) == 4> scatter4(
    T *p, simd<T, n * get_num_channels_enabled(Mask)> vals,
    simd<uint32_t, n> offsets, simd_mask<n> pred = 1) {
  detail::IfNotNone<L1H, L3H>::warn();
  scatter_rgba<T, n, Mask>(p, offsets, vals, pred);
}

namespace detail {
/// Check the legality of an atomic call in terms of size and type.
/// \ingroup sycl_esimd
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
  if constexpr (Op == atomic_op::fmax || Op == atomic_op::fmin) {
    if constexpr (NumSrc != 1) {
      static_assert(NumSrc == 1, "One source operand is expected");
      return false;
    }
    if constexpr (!is_type<T, float, sycl::detail::half_impl::StorageT>()) {
      static_assert((is_type<T, float, sycl::detail::half_impl::StorageT>()),
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
    if constexpr (Op == atomic_op::fcmpwr &&
                  !is_type<T, float, sycl::detail::half_impl::StorageT>()) {
      static_assert((is_type<T, float, sycl::detail::half_impl::StorageT>()),
                    "Type F or HF is expected");
      return false;
    }
    return true;
  }
  // Unsupported svm atomic Op.
  return false;
}
} // namespace detail

// TODO @Pennycook
// {quote}
// We should look into what can be done to simplify these atomic functions and
// align their design with the other new atomic features.That is perhaps out of
// scope for this PR(the direction is less clear than for the reduce changes,
// for example) but we should open an issue to track it.
// {/quote}

/// USM address atomic update, version with no source operands: \c inc and \c
/// dec. \ingroup sycl_esimd
template <atomic_op Op, typename T, int n>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 0>(), simd<T, n>>
atomic_update(T *p, simd<unsigned, n> offset, simd_mask<n> pred) {
  simd<uintptr_t, n> vAddr(reinterpret_cast<uintptr_t>(p));
  simd<uintptr_t, n> offset_i1 = convert<uintptr_t>(offset);
  vAddr += offset_i1;
  return __esimd_svm_atomic0<Op, T, n>(vAddr.data(), pred.data());
}

template <atomic_op Op, typename T, int n, CacheHint L1H = CacheHint::None,
          CacheHint L3H = CacheHint::None>
__SYCL_DEPRECATED("use simd::atomic_update.")
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 0>(),
                             simd<T, n>> flat_atomic(T *p,
                                                     simd<unsigned, n> offset,
                                                     simd_mask<n> pred) {
  detail::IfNotNone<L1H, L3H>::warn();
  return atomic_update<Op>(p, offset, pred);
}

/// USM address atomic update, version with one source operand: e.g. \c add, \c
/// sub. \ingroup sycl_esimd
template <atomic_op Op, typename T, int n>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 1>(), simd<T, n>>
atomic_update(T *p, simd<unsigned, n> offset, simd<T, n> src0,
              simd_mask<n> pred) {
  simd<uintptr_t, n> vAddr(reinterpret_cast<uintptr_t>(p));
  simd<uintptr_t, n> offset_i1 = convert<uintptr_t>(offset);
  vAddr += offset_i1;
  return __esimd_svm_atomic1<Op, T, n>(vAddr.data(), src0.data(), pred.data());
}

template <atomic_op Op, typename T, int n, CacheHint L1H = CacheHint::None,
          CacheHint L3H = CacheHint::None>
__SYCL_DEPRECATED("use simd::atomic_update.")
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 1>(),
                             simd<T, n>> flat_atomic(T *p,
                                                     simd<unsigned, n> offset,
                                                     simd<T, n> src0,
                                                     simd_mask<n> pred) {
  detail::IfNotNone<L1H, L3H>::warn();
  return atomic_update<Op>(p, offset, src0, pred);
}

/// USM address atomic update, version with two source operands: e.g. \c
/// cmpxchg. \ingroup sycl_esimd
template <atomic_op Op, typename T, int n>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 2>(), simd<T, n>>
atomic_update(T *p, simd<unsigned, n> offset, simd<T, n> src0, simd<T, n> src1,
              simd_mask<n> pred) {
  simd<uintptr_t, n> vAddr(reinterpret_cast<uintptr_t>(p));
  simd<uintptr_t, n> offset_i1 = convert<uintptr_t>(offset);
  vAddr += offset_i1;
  return __esimd_svm_atomic2<Op, T, n>(vAddr.data(), src0.data(), src1.data(),
                                       pred.data());
}

template <atomic_op Op, typename T, int n, CacheHint L1H = CacheHint::None,
          CacheHint L3H = CacheHint::None>
__SYCL_DEPRECATED("use simd::atomic_update.")
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 2>(),
                             simd<T, n>> flat_atomic(T *p,
                                                     simd<unsigned, n> offset,
                                                     simd<T, n> src0,
                                                     simd<T, n> src1,
                                                     simd_mask<n> pred) {
  detail::IfNotNone<L1H, L3H>::warn();
  return atomic_update<Op>(p, offset, src0, src1, pred);
}

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

enum __SYCL_DEPRECATED("use esimd::fence_mask.") EsimdFenceMask {
  __ESIMD_DEPR_ENUM_V(ESIMD_GLOBAL_COHERENT_FENCE,
                      fence_mask::global_coherent_fence, fence_mask),
  __ESIMD_DEPR_ENUM_V(ESIMD_L3_FLUSH_INSTRUCTIONS,
                      fence_mask::l3_flush_instructions, fence_mask),
  __ESIMD_DEPR_ENUM_V(ESIMD_L3_FLUSH_TEXTURE_DATA,
                      fence_mask::l3_flush_texture_data, fence_mask),
  __ESIMD_DEPR_ENUM_V(ESIMD_L3_FLUSH_CONSTANT_DATA,
                      fence_mask::l3_flush_constant_data, fence_mask),
  __ESIMD_DEPR_ENUM_V(ESIMD_L3_FLUSH_RW_DATA, fence_mask::l3_flush_rw_data,
                      fence_mask),
  __ESIMD_DEPR_ENUM_V(ESIMD_LOCAL_BARRIER, fence_mask::local_barrier,
                      fence_mask),
  __ESIMD_DEPR_ENUM_V(ESIMD_L1_FLUSH_RO_DATA, fence_mask::l1_flush_ro_data,
                      fence_mask),
  __ESIMD_DEPR_ENUM_V(ESIMD_SW_BARRIER, fence_mask::sw_barrier, fence_mask)
};

/// esimd::fence sets the memory read/write order.
/// \tparam cntl is a bitmask composed from \c fence_mask bits.
/// \ingroup sycl_esimd
__ESIMD_API void fence(fence_mask cntl) { __esimd_fence(cntl); }

/// Generic work-group barrier.
/// Performs barrier synchronization for all threads within the same thread
/// group. The barrier instruction causes the executing thread to wait until
/// all threads in the same thread group have executed the barrier instruction.
/// Memory ordering is also guaranteed by this instruction.
/// The behavior is undefined if this instruction is executed in divergent
/// control flow.
/// \ingroup sycl_esimd
__ESIMD_API void barrier() {
  __esimd_fence(fence_mask::global_coherent_fence | fence_mask::local_barrier);
  __esimd_barrier();
}

/// Generic work-group split barrier
__ESIMD_API void sbarrier(split_barrier_action flag) { __esimd_sbarrier(flag); }

/// @defgroup sycl_esimd_slm SLM functions
/// \ingroup sycl_esimd
/// @{

/// Declare per-work-group slm size.
SYCL_EXTERNAL SYCL_ESIMD_FUNCTION void slm_init(uint32_t size);

/// SLM gather.
///
/// Only allow simd-16 and simd-32.
template <typename T, int n>
__ESIMD_API
    std::enable_if_t<(n == 1 || n == 8 || n == 16 || n == 32), simd<T, n>>
    slm_gather(simd<uint32_t, n> offsets, simd_mask<n> pred = 1) {
  detail::LocalAccessorMarker acc;
  return detail::gather_impl<T, n>(acc, offsets, 0, pred);
}

/// SLM gather (deprecated version).
template <typename T, int n>
__SYCL_DEPRECATED("use slm_gather.")
__ESIMD_API std::enable_if_t<(n == 1 || n == 8 || n == 16 || n == 32),
                             simd<T, n>> slm_load(simd<uint32_t, n> offsets,
                                                  simd_mask<n> pred = 1) {
  return slm_gather<T, n>(offsets, pred);
}

/// Load a scalar value from the Shared Local Memory.
/// @tparam T type of the value
/// @param offset SLM offset in bytes
/// @return the loaded value
/// \ingroup sycl_esimd
template <typename T> __ESIMD_API T slm_scalar_load(uint32_t offset) {
  const simd<T, 1> Res = slm_gather<T, 1>(simd<uint32_t, 1>(offset));
  return Res[0];
}

/// SLM scatter.
template <typename T, int n>
__ESIMD_API std::enable_if_t<(n == 1 || n == 8 || n == 16 || n == 32) &&
                             (sizeof(T) <= 4)>
slm_scatter(simd<uint32_t, n> offsets, simd<T, n> vals, simd_mask<n> pred = 1) {
  detail::LocalAccessorMarker acc;
  detail::scatter_impl<T, n>(acc, vals, offsets, 0, pred);
}

/// SLM scatter (deprecated version).
template <typename T, int n>
__SYCL_DEPRECATED("use slm_scatter.")
__ESIMD_API std::enable_if_t<(n == 16 || n == 32)> slm_store(
    simd<T, n> vals, simd<uint32_t, n> offsets, simd_mask<n> pred = 1) {
  slm_scatter<T, n>(offsets, vals, pred);
}

/// Store a scalar value into the Shared Local Memory.
/// @tparam T type of the value
/// @param offset SLM offset in bytes
/// @param val value to store
/// \ingroup sycl_esimd
template <typename T>
__ESIMD_API void slm_scalar_store(uint32_t offset, T val) {
  slm_scatter<T, 1>(simd<uint32_t, 1>(offset), simd<T, 1>(val), 1);
}

/// Gathering read from the SLM given specified \p offsets.
/// Up to 4 data elements may be accessed at each address depending on the
/// enabled channel \p Mask.
/// \tparam T element type of the returned vector. Must be 4-byte.
/// \tparam N size of the \p offsets vector. Must be 8, 16 or 32.
/// \tparam Mask represents a pixel's channel mask.
/// @param offsets byte-offsets within the SLM.
/// @param pred predication control used for masking lanes.
/// \ingroup sycl_esimd
template <typename T, int N, rgba_channel_mask Mask>
__ESIMD_API std::enable_if_t<(N == 8 || N == 16 || N == 32) && (sizeof(T) == 4),
                             simd<T, N * get_num_channels_enabled(Mask)>>
slm_gather_rgba(simd<uint32_t, N> offsets, simd_mask<N> pred = 1) {

  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  return __esimd_gather4_scaled<T, N, decltype(si), Mask>(
      pred.data(), si, 0 /*global_offset*/, offsets.data());
}

/// SLM gather4.
///
/// Only allow simd-8, simd-16 and simd-32.
template <typename T, int n, rgba_channel_mask Mask>
__SYCL_DEPRECATED("use slm_gather_rgba.")
__ESIMD_API std::enable_if_t<
    (n == 8 || n == 16 || n == 32) && (sizeof(T) == 4),
    simd<T, n * get_num_channels_enabled(Mask)>> slm_load4(simd<uint32_t, n>
                                                               offsets,
                                                           simd_mask<n> pred =
                                                               1) {
  return slm_gather_rgba<T, n, Mask>(offsets, pred);
}

/// Scatter write to the SLM given specified \p offsets.
/// Up to 4 data elements may be written at each address depending on the
/// enabled channel \p Mask.
/// \tparam T element type of the input vector. Must be 4-byte.
/// \tparam N size of the \p offsets vector. Must be 8, 16 or 32.
/// \tparam Mask represents a pixel's channel mask.
/// @param offsets byte-offsets within the SLM.
/// @param vals values to be written.
/// @param pred predication control used for masking lanes.
/// \ingroup sycl_esimd
template <typename T, int N, rgba_channel_mask Mask>
__ESIMD_API std::enable_if_t<(N == 8 || N == 16 || N == 32) && (sizeof(T) == 4)>
slm_scatter_rgba(simd<uint32_t, N> offsets,
                 simd<T, N * get_num_channels_enabled(Mask)> vals,
                 simd_mask<N> pred = 1) {
  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  constexpr int16_t Scale = 0;
  constexpr int global_offset = 0;
  __esimd_scatter4_scaled<T, N, decltype(si), Mask, Scale>(
      pred.data(), si, global_offset, offsets.data(), vals.data());
}

/// SLM scatter4.
template <typename T, int n, rgba_channel_mask Mask>
__SYCL_DEPRECATED("use slm_scatter_rgba.")
__ESIMD_API std::
    enable_if_t<(n == 8 || n == 16 || n == 32) && (sizeof(T) == 4)> slm_store4(
        simd<T, n * get_num_channels_enabled(Mask)> vals,
        simd<uint32_t, n> offsets, simd_mask<n> pred = 1) {
  slm_scatter_rgba<T, n, Mask>(offsets, vals, pred);
}

/// SLM block-load.
template <typename T, int n>
__ESIMD_API simd<T, n> slm_block_load(uint32_t offset) {
  constexpr unsigned Sz = sizeof(T) * n;
  static_assert(Sz >= detail::OperandSize::OWORD,
                "block size must be at least 1 oword");
  static_assert(Sz % detail::OperandSize::OWORD == 0,
                "block size must be whole number of owords");
  static_assert(detail::isPowerOf2(Sz / detail::OperandSize::OWORD),
                "block must be 1, 2, 4 or 8 owords long");
  static_assert(Sz <= 16 * detail::OperandSize::OWORD,
                "block size must be at most 16 owords");

  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  return __esimd_oword_ld<T, n>(si, offset >> 4);
}

/// SLM block-store.
template <typename T, int n>
__ESIMD_API void slm_block_store(uint32_t offset, simd<T, n> vals) {
  constexpr unsigned Sz = sizeof(T) * n;
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
  __esimd_oword_st<T, n>(si, offset >> 4, vals.data());
}

/// SLM atomic update operation, no source operands: \c inc and \c dec.
template <atomic_op Op, typename T, int n>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 0>(), simd<T, n>>
slm_atomic_update(simd<uint32_t, n> offsets, simd_mask<n> pred) {
  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  return __esimd_dword_atomic0<Op, T, n>(pred.data(), si, offsets.data());
}

template <atomic_op Op, typename T, int n>
__SYCL_DEPRECATED("use slm_atomic_update.")
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 0>(),
                             simd<T, n>> slm_atomic(simd<uint32_t, n> offsets,
                                                    simd_mask<n> pred) {
  return slm_atomic_update<Op, T, n>(offsets, pred);
}

/// SLM atomic update operation, one source operand: e.g. \c add, \c sub.
template <atomic_op Op, typename T, int n>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 1>(), simd<T, n>>
slm_atomic_update(simd<uint32_t, n> offsets, simd<T, n> src0,
                  simd_mask<n> pred) {
  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  return __esimd_dword_atomic1<Op, T, n>(pred.data(), si, offsets.data(),
                                         src0.data());
}

template <atomic_op Op, typename T, int n>
__SYCL_DEPRECATED("use slm_atomic_update.")
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 1>(),
                             simd<T, n>> slm_atomic(simd<uint32_t, n> offsets,
                                                    simd<T, n> src0,
                                                    simd_mask<n> pred) {
  return slm_atomic_update<Op, T, n>(offsets, src0, pred);
}

/// SLM atomic, two source operands.
template <atomic_op Op, typename T, int n>
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 2>(), simd<T, n>>
slm_atomic_update(simd<uint32_t, n> offsets, simd<T, n> src0, simd<T, n> src1,
                  simd_mask<n> pred) {
  const auto si = __ESIMD_GET_SURF_HANDLE(detail::LocalAccessorMarker());
  return __esimd_dword_atomic2<Op, T, n>(pred.data(), si, offsets.data(),
                                         src0.data(), src1.data());
}

template <atomic_op Op, typename T, int n>
__SYCL_DEPRECATED("use slm_atomic_update.")
__ESIMD_API std::enable_if_t<detail::check_atomic<Op, T, n, 2>(),
                             simd<T, n>> slm_atomic(simd<uint32_t, n> offsets,
                                                    simd<T, n> src0,
                                                    simd<T, n> src1,
                                                    simd_mask<n> pred) {
  return slm_atomic_update<Op, T, n>(offsets, src0, src1, pred);
}
/// @}

/// Media block load.
///
/// \tparam T is the element data type.
/// \tparam m is the height of the 2D block.
/// \tparam n is the width of the 2D block.
/// \tparam AccessorTy is type of the SYCL accessor.
/// \tparam plane is planar surface index.
/// \param acc is the SYCL accessor.
/// \param x is X-coordinate of the left upper rectangle corner in BYTES.
/// \param y is Y-coordinate of the left upper rectangle corner in ROWS.
/// \return the linearized 2D block data read from surface.
///
/// \ingroup sycl_esimd
template <typename T, int m, int n, typename AccessorTy, unsigned plane = 0>
__ESIMD_API simd<T, m * n> media_block_load(AccessorTy acc, unsigned x,
                                            unsigned y) {
  constexpr unsigned Width = n * sizeof(T);
  static_assert(Width * m <= 256u,
                "data does not fit into a single dataport transaction");
  static_assert(Width <= 64u, "valid block width is in range [1, 64]");
  static_assert(m <= 64u, "valid block height is in range [1, 64]");
  static_assert(plane <= 3u, "valid plane index is in range [0, 3]");

  const auto si = __ESIMD_GET_SURF_HANDLE(acc);
  using SurfIndTy = decltype(si);
  constexpr unsigned int RoundedWidth =
      Width < 4 ? 4 : detail::getNextPowerOf2<Width>();
  constexpr int BlockWidth = sizeof(T) * n;
  constexpr int Mod = 0;

  if constexpr (Width < RoundedWidth) {
    constexpr unsigned int n1 = RoundedWidth / sizeof(T);
    simd<T, m *n1> temp =
        __esimd_media_ld<T, m, n1, Mod, SurfIndTy, (int)plane, BlockWidth>(
            si, x, y);
    return temp.template select<m, 1, n, 1>(0, 0);
  } else {
    return __esimd_media_ld<T, m, n, Mod, SurfIndTy, (int)plane, BlockWidth>(
        si, x, y);
  }
}

/// Media block store.
///
/// \tparam T is the element data type.
/// \tparam m is the height of the 2D block.
/// \tparam n is the width of the 2D block.
/// \tparam is AccessorTy type of the SYCL accessor.
/// \tparam plane is planar surface index.
/// \param acc is the SYCL accessor.
/// \param x is X-coordinate of the left upper rectangle corner in BYTES.
/// \param y is Y-coordinate of the left upper rectangle corner in ROWS.
/// \param vals is the linearized 2D block data to be written to surface.
///
/// \ingroup sycl_esimd
template <typename T, int m, int n, typename AccessorTy, unsigned plane = 0>
__ESIMD_API void media_block_store(AccessorTy acc, unsigned x, unsigned y,
                                   simd<T, m * n> vals) {
  constexpr unsigned Width = n * sizeof(T);
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
  constexpr int BlockWidth = sizeof(T) * n;
  constexpr int Mod = 0;

  if constexpr (Width < RoundedWidth) {
    simd<T, m * n1> temp;
    auto temp_ref = temp.template bit_cast_view<T, m, n1>();
    auto vals_ref = vals.template bit_cast_view<T, m, n>();
    temp_ref.template select<m, 1, n, 1>() = vals_ref;
    __esimd_media_st<T, m, n1, Mod, SurfIndTy, plane, BlockWidth>(si, x, y,
                                                                  temp.data());
  } else {
    __esimd_media_st<T, m, n, Mod, SurfIndTy, plane, BlockWidth>(si, x, y,
                                                                 vals.data());
  }
}

#ifndef __SYCL_DEVICE_ONLY__

inline void slm_init(uint32_t size) {}

#endif

/// get_value
///
/// \param acc is the SYCL accessor.
/// \return the binding table index value.
///
/// \ingroup sycl_esimd
template <typename AccessorTy>
__SYCL_DEPRECATED("use get_surface_index")
__ESIMD_API uint32_t get_value(AccessorTy acc) {
  return static_cast<uint32_t>(get_surface_index(acc));
}

/// \defgroup sycl_esimd_raw_send_api Raw send APIs
/// APIs below are used to implement the send messages on Intel(R) processor
/// graphics, as defined in the documentation at
/// https://01.org/sites/default/files/documentation/intel-gfx-prm-osrc-icllp-vol02a-commandreference-instructions_2.pdf
///
/// \ingroup sycl_esimd
/// @{

/// Raw sends load.
///
/// \param msgDst is the old value of the destination operand.
/// \param msgSrc0 is the first source operand of send message.
/// \param msgSrc1 is the second source operand of send message.
/// \param exDesc is the extended message descriptor.
/// \param msgDesc is the message descriptor.
/// \param execSize is the execution size, which must be a compile time
/// constant.
/// \param sfid is the shared function ID, which must be a compile time
/// constant.
/// \param numSrc0 is the number of GRFs for source-0, which must be a compile
/// time constant.
/// \param numSrc1 is the number of GRFs for source-1, which must be a compile
/// constant.
/// \param numDst is the number of GRFs for destination, which must be a compile
/// time constant.
/// \param isEOT is the flag that indicates whether this is an EOT message,
/// which must be a compile time constant (optional - default to 0).
/// \param isSendc is the flag that indicates whether sendc should be used,
/// which must be a compile time constant (optional - default to 0).
/// \param mask is the predicate to specify enabled channels (optional - default
/// to on).
/// \return the vector value read from memory.
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
/// \param msgDst is the old value of the destination operand.
/// \param msgSrc0 is the first source operand of send message.
/// \param exDesc is the extended message descriptor.
/// \param msgDesc is the message descriptor.
/// \param execSize is the execution size, which must be a compile time
/// constant.
/// \param sfid is the shared function ID, which must be a compile time
/// constant.
/// \param numSrc0 is the number of GRFs for source-0, which must be a compile
/// time constant.
/// \param numDst is the number of GRFs for destination, which must be a compile
/// time constant.
/// \param isEOT is the flag that indicates whether this is an EOT message,
/// which must be a compile time constant (optional - default to 0).
/// \param isSendc is the flag that indicates whether sendc should be used,
/// which must be a compile time constant (optional - default to 0).
/// \param mask is the predicate to specify enabled channels (optional - default
/// to on).
/// \return the vector value read from memory.
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

/// Raw sends store.
///
/// \param msgSrc0 is the first source operand of send message.
/// \param msgSrc1 is the second source operand of send message.
/// \param exDesc is the extended message descriptor.
/// \param msgDesc is the message descriptor.
/// \param execSize is the execution size, which must be a compile time
/// constant.
/// \param sfid is the shared function ID, which must be a compile time
/// constant.
/// \param numSrc0 is the number of GRFs for source-0, which must be a compile
/// time constant.
/// \param numSrc1 is the number of GRFs for source-1, which must be a compile
/// time constant.
/// \param isEOT is the flag that indicates whether this is an EOT message,
/// which must be a compile time constant (optional - default to 0).
/// \param isSendc is the flag that indicates whether sendc should be used,
/// which must be a compile time constant (optional - default to 0).
/// \param mask is the predicate to specify enabled channels (optional - default
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

/// Raw send store.
///
/// \param msgSrc0 is the first source operand of send message.
/// \param exDesc is the extended message descriptor.
/// \param msgDesc is the message descriptor.
/// \param execSize is the execution size, which must be a compile time
/// constant.
/// \param sfid is the shared function ID, which must be a compile time
/// constant.
/// \param numSrc0 is the number of GRFs for source-0, which must be a compile
/// time constant.
/// \param isEOT is the flag that indicates whether this is an EOT message,
/// which must be a compile time constant (optional - default to 0).
/// \param isSendc is the flag that indicates whether sendc should be used,
/// which must be a compile time constant (optional - default to 0).
/// \param mask is the predicate to specify enabled channels (optional - default
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
/// @}

#undef __ESIMD_GET_SURF_HANDLE

} // namespace esimd
} // namespace experimental
} // namespace intel
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
