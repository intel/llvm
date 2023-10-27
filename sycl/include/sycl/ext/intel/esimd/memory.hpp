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

namespace detail {
// Accessors may get either 32-bit offset or 64-bit depending on
// the -fsycl-esimd-force-stateles-mem mode setting.
#ifdef __ESIMD_FORCE_STATELESS_MEM
using DeviceAccessorOffsetT = uint64_t;
#else
using DeviceAccessorOffsetT = uint32_t;
#endif

template <typename T, int NElts, cache_hint L1H = cache_hint::none,
          cache_hint L2H = cache_hint::none, typename FlagsT>
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
      std::max(static_cast<size_t>(1), sizeof(uint32_t) / sizeof(T));
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
template <typename T, int NElts, cache_hint L1H = cache_hint::none,
          cache_hint L2H = cache_hint::none, typename FlagsT>
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
      std::max(static_cast<size_t>(1), sizeof(uint32_t) / sizeof(T));
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
template <typename T, int NElts, cache_hint L1H = cache_hint::none,
          cache_hint L2H = cache_hint::none, typename AccessorT,
          typename FlagsT = __ESIMD_DNS::dqword_element_aligned_tag>
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
      std::max(static_cast<size_t>(1), sizeof(uint32_t) / sizeof(T));
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
template <typename T, int NElts, cache_hint L1H = cache_hint::none,
          cache_hint L2H = cache_hint::none, typename AccessorT,
          typename FlagsT = dqword_element_aligned_tag>
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
      std::max(static_cast<size_t>(1), sizeof(uint32_t) / sizeof(T));
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
block_store(Tx *addr, simd<Tx, N> vals, Flags = {}) {
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

/// simd<T, N> block_load(const T* ptr, props={});                      // (1)
/// simd<T, N> block_load(const T* ptr, size_t offset, props={});       // (2)

/// simd<T, N> block_load(const T* ptr, simd_mask<1> pred, props={});   // (3)
/// simd<T, N> block_load(const T* ptr, size_t offset, simd_mask<1> pred,
///                       props={});                                    // (4)

/// simd<T, N> block_load(const T* ptr, simd_mask<1> pred,
///                       simd<T, N> pass_thru, props={});              // (5)
/// simd<T, N> block_load(const T* ptr, size_t offset, simd_mask<1> pred,
///                       simd<T, N> pass_thru, props={});              // (6)

/// simd<T, N> block_load(const T* ptr, props={}); // (1)
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
/// the default assumed alignment is the minimally required element-size
/// alignment. Note that additional/temporary restrictions may apply
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

  if constexpr (L1Hint != cache_hint::none || L2Hint != cache_hint::none) {
    detail::check_cache_hint<detail::cache_action::load, L1Hint, L2Hint>();
    constexpr int DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
    constexpr size_t Alignment =
        detail::getPropertyValue<PropertyListT, alignment_key>(
            DefaultAlignment);

    simd_mask<1> Mask = 1;
    return detail::block_load_impl<T, N, L1Hint, L2Hint>(
        ptr, Mask, overaligned_tag<Alignment>{});
  } else {
    // If the alignment property is not passed, then assume the pointer
    // is element-aligned.
    constexpr size_t Alignment =
        detail::getPropertyValue<PropertyListT, alignment_key>(sizeof(T));
    return block_load<T, N>(ptr, overaligned_tag<Alignment>{});
  }
}

/// simd<T, N> block_load(const T* ptr, size_t byte_offset, props={}); // (2)
/// This function loads a contiguous memory block from address referenced
/// by USM pointer \p ptr and byte-offset \p byte_offset.
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
/// the default assumed alignment is the minimally required element-size
/// alignment. Note that additional/temporary restrictions may apply
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
    ext::oneapi::experimental::is_property_list_v<PropertyListT>, simd<T, N>>
block_load(const T *ptr, size_t byte_offset, PropertyListT props = {}) {
  const T *AdjustedPtr = reinterpret_cast<const T *>(
      reinterpret_cast<const int8_t *>(ptr) + byte_offset);
  return block_load<T, N>(AdjustedPtr, props);
}

/// simd<T, N> block_load(const T* ptr, simd_mask<1> pred, props={}); // (3)
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

/// simd<T, N> block_load(const T* ptr, size_t byte_offset, simd_mask<1> pred,
///                       props={}); // (4)
/// This function loads a contiguous memory block from address referenced
/// by USM pointer \p ptr and byte-offset \p byte_offset.
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
///                       simd<T, N> pass_thru, props={}); // (5)
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

/// simd<T, N> block_load(const T* ptr, size_t byte_offset, simd_mask<1> pred,
///                       simd<T, N> pass_thru, props={}); // (6)
/// This function loads a contiguous memory block from address referenced
/// by USM pointer \p ptr and byte-offset \p byte_offset.
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

/// Loads a contiguous block of memory from given accessor and offset and
/// returns the loaded data as a vector. Actual code generated depends on the
/// alignment parameter.
/// @tparam Tx Element type.
/// @tparam N Number of elements to load, <code>N * sizeof(Tx)</code> must be
///    1, 2, 4 or 8 owords long.
/// @tparam AccessorTy Accessor type (auto-deduced).
/// @tparam Flags The alignment specifier type tag. Auto-deduced from the
///    \c Flags parameter. If it is less than \c 16, then slower unaligned
///    access is generated, otherwise the access is aligned.
/// @param acc The accessor.
/// @param offset The offset to load from in bytes.
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
__ESIMD_API simd<Tx, N>
block_load(AccessorTy acc, detail::DeviceAccessorOffsetT offset, Flags flags) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return block_load<Tx, N>(__ESIMD_DNS::accessorToPointer<Tx>(acc, offset),
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
    return __esimd_oword_ld<T, N>(surf_ind, offset >> 4);
  } else {
    return __esimd_oword_ld_unaligned<T, N>(surf_ind, offset);
  }
#endif
}

/// Each of the following block load functions loads a contiguous memory block
/// from the address referenced by accessor 'acc', or from 'acc +
/// offset', where 'offset' is the offset in bytes (not in elements!). The
/// parameter 'pred' is the one element predicate. If it is set to 1, then all
/// 'N' elements are loaded. Otherwise, the block load operation is a NO-OP.
/// The parameter 'pass_thru' specifies the values being copied to the returned
/// result if 'pred' is set to 0.
/// The parameter 'props' specifies the optional compile-time properties
/// of the type esimd::properties and may include esimd::cache_hint_L1,
/// esimd::cache_hint_L2, esimd::cache_hint_L3, esimd::alignment.

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT offset, props = {});         // (acc-1)
/// simd<T, N> block_load(AccessorT acc, props);                   // (acc-2)

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT offset, simd_mask<1> pred,
///            simd<T, N> pass_thru, flags = {});                  // (acc-3)
/// simd<T, N>
/// block_load(AccessorT acc, OffsetT offset, simd_mask<1> pred,
///            flags = {});                                        // (acc-4)

/// simd<T, N>
/// block_load(AccessorT acc, simd_mask<1> pred,
///            simd<T, N> pass_thru, flags = {});                  // (acc-5)
/// simd<T, N>
/// block_load(AccessorT acc, simd_mask<1> pred, flags = {});      // (acc-6)

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT offset, props = {});         // (acc-1)
/// This function loads a contiguous memory block referenced
/// by accessor \p acc and byte-offset \p offset.
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
/// (R1), (R2) and (R3) are not applied if there are no cache hints.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, detail::DeviceAccessorOffsetT offset,
           PropertyListT props = {}) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return block_load<T, N>(detail::accessorToPointer<T>(acc, offset), props);
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
        acc, offset, simd_mask<1>(1), overaligned_tag<Alignment>{});
  } else {
    return block_load<T, N>(acc, offset, overaligned_tag<Alignment>{});
  }
#endif // !__ESIMD_FORCE_STATELESS_MEM
}

/// simd<T, N> block_load(AccessorT acc, props);                   // (acc-2)
/// This function loads a contiguous memory block referenced
/// by accessor \p acc using implied offset=0.
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
/// (R1), (R2) and (R3) are not applied if there are no cache hints.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, PropertyListT props = {}) {
  return block_load<T, N>(acc, 0, props);
}

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT offset, simd_mask<1> pred,
///            simd<T, N> pass_thru, flags = {});                  // (acc-3)
/// This function loads a contiguous memory block referenced
/// by accessor \p acc using the byte-offset \p offset.
/// If the predicate \p pred is set to 0, then the load is omitted and the
/// returned \p pass_thru is returned.
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
/// (R1), (R2) and (R3) are not applied if there are no cache hints.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, detail::DeviceAccessorOffsetT offset,
           simd_mask<1> pred, simd<T, N> pass_thru, PropertyListT props = {}) {
  constexpr auto L1Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L1_key>(
          cache_hint::none);
  constexpr auto L2Hint =
      detail::getPropertyValue<PropertyListT, cache_hint_L2_key>(
          cache_hint::none);
  static_assert(!PropertyListT::template has_property<cache_hint_L3_key>(),
                "L3 cache hint is reserved. The old/experimental L3 LSC cache "
                "hint is cache_level::L2 now.");

  // If the alignment property is not passed, then assume the offset
  // is element-aligned and is at leat 4-bytes.
  constexpr size_t DefaultAlignment = (sizeof(T) <= 4) ? 4 : sizeof(T);
  constexpr size_t Alignment =
      detail::getPropertyValue<PropertyListT, alignment_key>(DefaultAlignment);
  return detail::block_load_impl<T, N, L1Hint, L2Hint>(
      acc, offset, pred, pass_thru, overaligned_tag<Alignment>{});
}

/// simd<T, N>
/// block_load(AccessorT acc, OffsetT offset, simd_mask<1> pred,
///            flags = {});                                        // (acc-4)
/// This function loads a contiguous memory block referenced
/// by accessor \p acc using the byte-offset \p offset.
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
/// (R1), (R2) and (R3) are not applied if there are no cache hints.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, detail::DeviceAccessorOffsetT offset,
           simd_mask<1> pred, PropertyListT props = {}) {
  simd<T, N> PassThru; // Intentionally uninitialized.
  return block_load<T, N>(acc, offset, pred, PassThru, props);
}

/// simd<T, N>
/// block_load(AccessorT acc, simd_mask<1> pred,
///            simd<T, N> pass_thru, flags = {});                // (acc-5)
/// Same as (acc-3) variant except that the byte-offset is not passed
/// and is implied to be 0.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, simd_mask<1> pred, simd<T, N> pass_thru,
           PropertyListT props = {}) {
  return block_load<T, N>(acc, 0, pred, pass_thru, props);
}

/// simd<T, N>
/// block_load(AccessorT acc, simd_mask<1> pred, flags = {});    // (acc-6)
/// Same as (acc-4) variant except that the byte-offset is not passed
/// and is implied to be 0.
template <typename T, int N, typename AccessorT,
          typename PropertyListT =
              ext::oneapi::experimental::detail::empty_properties_t>
__ESIMD_API std::enable_if_t<
    ext::oneapi::experimental::is_property_list_v<PropertyListT> &&
        detail::is_device_accessor_with_v<AccessorT,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
block_load(AccessorT acc, simd_mask<1> pred, PropertyListT props = {}) {
  simd<T, N> PassThru; // Intentionally uninitialized.
  return block_load<T, N>(acc, 0, pred, PassThru, props);
}

/// @} sycl_esimd_memory_block

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
__ESIMD_API std::enable_if_t<detail::is_device_accessor_with_v<
    AccessorTy, detail::accessor_mode_cap::can_write>>
block_store(AccessorTy acc, detail::DeviceAccessorOffsetT offset,
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
                                   !std::is_pointer_v<AccessorTy>>
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
        !std::is_pointer_v<AccessorTy>,
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
__ESIMD_API std::enable_if_t<
    (sizeof(T) <= 4) && (N == 1 || N == 8 || N == 16 || N == 32) &&
        detail::is_device_accessor_with_v<AccessorTy,
                                          detail::accessor_mode_cap::can_read>,
    simd<T, N>>
gather(AccessorTy acc, simd<detail::DeviceAccessorOffsetT, N> offsets,
       detail::DeviceAccessorOffsetT glob_offset = 0, simd_mask<N> mask = 1) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return gather<T, N>(__ESIMD_DNS::accessorToPointer<T>(acc, glob_offset),
                      offsets, mask);
#else
  return detail::gather_impl<T, N, AccessorTy>(acc, offsets, glob_offset, mask);
#endif
}

#ifdef __ESIMD_FORCE_STATELESS_MEM
template <typename T, int N, typename AccessorTy, typename Toffset>
__ESIMD_API std::enable_if_t<
    (sizeof(T) <= 4) && (N == 1 || N == 8 || N == 16 || N == 32) &&
        detail::is_device_accessor_with_v<
            AccessorTy, detail::accessor_mode_cap::can_read> &&
        std::is_integral_v<Toffset> && !std::is_same_v<Toffset, uint64_t>,
    simd<T, N>>
gather(AccessorTy acc, simd<Toffset, N> offsets, uint64_t glob_offset = 0,
       simd_mask<N> mask = 1) {
  return gather<T, N, AccessorTy>(acc, convert<uint64_t>(offsets), glob_offset,
                                  mask);
}
#endif

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
template <__ESIMD_NS::atomic_op Op, typename T, int N, unsigned NumSrc>
constexpr void check_atomic() {

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
/// @param offset The byte-offset to load from.
/// @param Flags Specifies the alignment.
/// @return A vector of loaded elements.
///
template <typename T, int N,
          typename Flags = overaligned_tag<detail::OperandSize::OWORD>>
__ESIMD_API std::enable_if_t<is_simd_flag_type_v<Flags>, simd<T, N>>
slm_block_load(uint32_t offset, Flags = {}) {
  constexpr size_t Align = Flags::template alignment<simd<T, N>>;
  return __esimd_slm_block_ld<detail::__raw_t<T>, N, Align>(offset);
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
template <typename T, int N,
          typename Flags = overaligned_tag<detail::OperandSize::OWORD>>
__ESIMD_API std::enable_if_t<is_simd_flag_type_v<Flags>>
slm_block_store(uint32_t offset, simd<T, N> vals, Flags = {}) {
  constexpr size_t Align = Flags::template alignment<simd<T, N>>;
  __esimd_slm_block_st<detail::__raw_t<T>, N, Align>(offset, vals.data());
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
    // Auto-convert FP atomics to LSC version.
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
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
  if constexpr (Op == atomic_op::fcmpxchg) {
    // Auto-convert FP atomics to LSC version.
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
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> &&
        sycl::detail::acc_properties::is_accessor_v<AccessorTy> &&
        !sycl::detail::acc_properties::is_local_accessor_v<AccessorTy>,
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
    // Auto-convert FP atomics to LSC version.
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

/// Variant of \c atomic_update that uses \c local_accessor as a parameter.
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
template <atomic_op Op, typename Tx, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<
    sycl::detail::acc_properties::is_local_accessor_v<AccessorTy>, simd<Tx, N>>
atomic_update(AccessorTy acc, simd<uint32_t, N> offset, simd<Tx, N> src0,
              simd_mask<N> mask) {
  if constexpr ((Op == atomic_op::fmin) || (Op == atomic_op::fmax) ||
                (Op == atomic_op::fadd) || (Op == atomic_op::fsub)) {
    // Auto-convert FP atomics to LSC version.
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
    return slm_atomic_update<Op, Tx, N>(
        offset + __ESIMD_DNS::localAccessorToOffset(acc), src0, mask);
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
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> && !std::is_pointer_v<AccessorTy>, simd<Tx, N>>
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
    std::is_integral_v<Toffset> && !std::is_pointer_v<AccessorTy> &&
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
__ESIMD_API __ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> &&
        sycl::detail::acc_properties::is_accessor_v<AccessorTy> &&
        !sycl::detail::acc_properties::is_local_accessor_v<AccessorTy>,
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

/// Variant of \c atomic_update that uses \c local_accessor as a parameter.
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
template <atomic_op Op, typename Tx, int N, typename AccessorTy>
__ESIMD_API __ESIMD_API std::enable_if_t<
    sycl::detail::acc_properties::is_local_accessor_v<AccessorTy>, simd<Tx, N>>
atomic_update(AccessorTy acc, simd<uint32_t, N> offset, simd_mask<N> mask) {
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
    return slm_atomic_update<Op, Tx, N>(
        offset + __ESIMD_DNS::localAccessorToOffset(acc), mask);
  }
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
/// offsets are supported only when stateless memory accesses are enforced,
/// i.e. accessor based accesses are automatically converted to stateless
/// accesses.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy, typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> && !std::is_pointer_v<AccessorTy>, simd<Tx, N>>
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
/// offset are supported only when stateless memory accesses are enforced,
/// i.e. accessor based accesses are automatically converted to stateless
/// accesses.
/// @param mask Operation mask, only locations with non-zero in the
///   corresponding mask element are updated.
/// @return A vector of the old values at the memory locations before the
///   update.
///
template <atomic_op Op, typename Tx, int N, typename Toffset,
          typename AccessorTy>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> && !std::is_pointer_v<AccessorTy>, simd<Tx, N>>
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
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The vector of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced,
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
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> &&
        sycl::detail::acc_properties::is_accessor_v<AccessorTy> &&
        !sycl::detail::acc_properties::is_local_accessor_v<AccessorTy>,
    simd<Tx, N>>
atomic_update(AccessorTy acc, simd<Toffset, N> offset, simd<Tx, N> src0,
              simd<Tx, N> src1, simd_mask<N> mask) {
#ifdef __ESIMD_FORCE_STATELESS_MEM
  return atomic_update<Op, Tx, N>(__ESIMD_DNS::accessorToPointer<Tx>(acc),
                                  offset, src0, src1, mask);
#else
  static_assert(std::is_integral_v<Toffset>, "Unsupported offset type");
  static_assert(sizeof(Toffset) == 4, "Only 32 bit offset is supported");
  if constexpr (Op == atomic_op::fcmpxchg) {
    // Auto-convert FP atomics to LSC version.
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

/// Variant of \c atomic_update that uses \c local_accessor as a parameter.
/// Atomically updates \c N memory locations represented by an accessor and
/// a vector of offsets and returns a vector of old
/// values found at the memory locations before update. The update operation
/// has 2 additional arguments.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
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
template <atomic_op Op, typename Tx, int N, typename AccessorTy>
__ESIMD_API std::enable_if_t<
    sycl::detail::acc_properties::is_local_accessor_v<AccessorTy>, simd<Tx, N>>
atomic_update(AccessorTy acc, simd<uint32_t, N> offset, simd<Tx, N> src0,
              simd<Tx, N> src1, simd_mask<N> mask) {
  if constexpr (Op == atomic_op::fcmpxchg) {
    // Auto-convert FP atomics to LSC version.
    return atomic_update<detail::to_lsc_atomic_op<Op>(), Tx, N>(
        acc, offset, src0, src1, mask);
  } else {
    return slm_atomic_update<Op, Tx, N>(
        offset + __ESIMD_DNS::localAccessorToOffset(acc), src0, src1, mask);
  }
}

/// A variation of \c atomic_update API with \c offsets represented as
/// \c simd_view object.
///
/// @tparam Op The atomic operation - can be one of the following:
///   \c atomic_op::cmpxchg, \c atomic_op::fcmpxchg.
/// @tparam Tx The vector element type.
/// @tparam N The number of memory locations to update.
/// @tparam AccessorTy type of the SYCL accessor.
/// @param acc The SYCL accessor.
/// @param offset The simd_view of 32-bit or 64-bit offsets in bytes. 64-bit
/// offsets are supported only when stateless memory accesses are enforced,
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
          typename AccessorTy, typename RegionTy = region1d_t<Toffset, N, 1>>
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> && !std::is_pointer_v<AccessorTy>, simd<Tx, N>>
atomic_update(AccessorTy acc, simd_view<Toffset, RegionTy> offsets,
              simd<Tx, N> src0, simd<Tx, N> src1, simd_mask<N> mask) {
  return atomic_update<Op, Tx, N>(acc, offsets.read(), src0, src1, mask);
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
__ESIMD_API std::enable_if_t<
    std::is_integral_v<Toffset> && !std::is_pointer_v<AccessorTy>, simd<Tx, N>>
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
/// local-accessor \p acc and byte-offset \p offset, then returns the loaded
/// data as a simd object.
/// The generated code depends on the combination {T, N, Flags}.
/// Providing flags specifying the alignment of 16-bytes or more produces more
/// efficient code. If the alignment is smaller than 16-bytes, then less
/// efficient gather is generated. If the loaded vector is too long
/// for 1 flat-load GPU instruction, then a series of flat-loads and/or gathers
/// may be generated.
/// @tparam Tx Element type.
/// @tparam N Number of elements to load.
/// @tparam AccessorTy Accessor type (auto-deduced).
/// @tparam Flags The alignment specifier type tag.
/// @param acc The local accessor.
/// @param offset The offset to load from in bytes.
/// @param Flags Specifies the alignment.
/// @return A vector of loaded elements.
///
template <typename Tx, int N, typename AccessorTy,
          typename Flags = overaligned_tag<detail::OperandSize::OWORD>>
__ESIMD_API
    std::enable_if_t<detail::is_local_accessor_with_v<
                         AccessorTy, detail::accessor_mode_cap::can_read> &&
                         is_simd_flag_type_v<Flags>,
                     simd<Tx, N>>
    block_load(AccessorTy acc, uint32_t offset, Flags = {}) {
  return slm_block_load<Tx, N, Flags>(offset +
                                      __ESIMD_DNS::localAccessorToOffset(acc));
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
/// @tparam Tx Element type.
/// @tparam N Number of elements to store.
/// @tparam AccessorTy Accessor type (auto-deduced).
/// @param acc The local accessor to store to.
/// @param offset The byte-offset to store at.
/// @param vals The vector to store.
/// @param Flags Specifies the alignment.
///
template <typename Tx, int N, typename AccessorTy,
          typename Flags = overaligned_tag<detail::OperandSize::OWORD>>
__ESIMD_API
    std::enable_if_t<detail::is_local_accessor_with_v<
                         AccessorTy, detail::accessor_mode_cap::can_write> &&
                     is_simd_flag_type_v<Flags>>
    block_store(AccessorTy acc, uint32_t offset, simd<Tx, N> vals, Flags = {}) {
  slm_block_store<Tx, N, Flags>(
      offset + __ESIMD_DNS::localAccessorToOffset(acc), vals);
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

  using OffsetTy = decltype(offset);

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
      simd<OffsetTy, ChunkSize> Offsets(0u, sizeof(T));
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
        simd<OffsetTy, RemN> Offsets(0u, sizeof(T));
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
        simd<OffsetTy, N1> Offsets(0u, sizeof(T));
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

  using OffsetTy = decltype(offset);

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
      simd<OffsetTy, ChunkSize> Offsets(0u, sizeof(T));
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
        simd<OffsetTy, RemN> Offsets(0u, sizeof(T));
        select<RemN, 1>(NumChunks * ChunkSize) = gather<UT, RemN, AccessorT>(
            acc, Offsets, offset + (NumChunks * ChunkSize * sizeof(T)));
      } else {
        constexpr int N1 = RemN < 8 ? 8 : RemN < 16 ? 16 : 32;
        simd_mask_type<N1> Pred(0);
        Pred.template select<RemN, 1>() = 1;
        simd<OffsetTy, N1> Offsets(0u, sizeof(T));
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
