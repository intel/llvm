//==----------------- util.hpp - DPC++ Explicit SIMD API  ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Utility functions used for implementing Explicit SIMD APIs.
//===----------------------------------------------------------------------===//

#pragma once

/// @cond ESIMD_DETAIL

#include <sycl/detail/type_traits.hpp>
#include <sycl/ext/intel/esimd/detail/types.hpp>

#include <type_traits>

#ifdef __SYCL_DEVICE_ONLY__
#define __ESIMD_INTRIN __DPCPP_SYCL_EXTERNAL SYCL_ESIMD_FUNCTION
#elif !defined(__ESIMD_BUILD_HOST_CODE)
#define __ESIMD_INTRIN ESIMD_NOINLINE __attribute__((internal_linkage))
#else
#define __ESIMD_INTRIN inline
#endif // __SYCL_DEVICE_ONLY__

#ifdef __SYCL_DEVICE_ONLY__
#define __ESIMD_INTRIN_END ;
#else
#define __ESIMD_INTRIN_END                                                     \
  { __ESIMD_UNSUPPORTED_ON_HOST; }
#endif // __SYCL_DEVICE_ONLY__

namespace sycl {
inline namespace _V1 {
namespace ext::intel::esimd::detail {

/// ESIMD intrinsic operand size in bytes.
struct OperandSize {
  enum { BYTE = 1, WORD = 2, DWORD = 4, QWORD = 8, OWORD = 16, GRF = 32 };
};

/// Compute next power of 2 of a constexpr with guaranteed compile-time
/// evaluation.
template <unsigned int N, unsigned int K, bool K_gt_eq_N> struct NextPowerOf2;

template <unsigned int N, unsigned int K> struct NextPowerOf2<N, K, true> {
  static constexpr unsigned int get() { return K; }
};

template <unsigned int N, unsigned int K> struct NextPowerOf2<N, K, false> {
  static constexpr unsigned int get() {
    return NextPowerOf2<N, K * 2, K * 2 >= N>::get();
  }
};

template <unsigned int N> constexpr unsigned int getNextPowerOf2() {
  return NextPowerOf2<N, 1, (1 >= N)>::get();
}

template <> constexpr unsigned int getNextPowerOf2<0>() { return 0; }

template <unsigned int N, unsigned int M>
constexpr unsigned int roundUpNextMultiple() {
  return ((N + M - 1) / M) * M;
}

/// Compute binary logarithm of a constexpr with guaranteed compile-time
/// evaluation.
template <unsigned int N, bool N_gt_1> struct Log2;

template <unsigned int N> struct Log2<N, false> {
  static constexpr unsigned int get() { return 0; }
};

template <unsigned int N> struct Log2<N, true> {
  static constexpr unsigned int get() {
    return 1 + Log2<(N >> 1), ((N >> 1) > 1)>::get();
  }
};

template <unsigned int N> constexpr unsigned int log2() {
  return Log2<N, (N > 1)>::get();
}

/// type traits
template <typename T> struct is_esimd_vector : public std::false_type {};

template <typename T, int N>
struct is_esimd_vector<simd<T, N>> : public std::true_type {};

template <typename T, int N>
using is_hw_int_type =
    typename std::bool_constant<std::is_integral_v<T> && (sizeof(T) == N)>;

template <typename T> using is_qword_type = is_hw_int_type<T, 8>;
template <typename T> using is_dword_type = is_hw_int_type<T, 4>;
template <typename T> using is_word_type = is_hw_int_type<T, 2>;
template <typename T> using is_byte_type = is_hw_int_type<T, 1>;

template <typename T, int N>
using is_hw_fp_type = typename std::bool_constant<std::is_floating_point_v<T> &&
                                                  (sizeof(T) == N)>;

template <typename T> using is_fp_type = is_hw_fp_type<T, 4>;
template <typename T> using is_df_type = is_hw_fp_type<T, 8>;

template <typename T>
using is_fp_or_dword_type =
    typename std::bool_constant<is_fp_type<T>::value ||
                                is_dword_type<T>::value>;

/// Convert types into vector types
template <typename T> struct simd_type {
  using type = simd<T, 1>;
};
template <typename T, int N> struct simd_type<raw_vector_type<T, N>> {
  using type = simd<T, N>;
};

template <typename T> struct simd_type<T &> {
  using type = typename simd_type<T>::type;
};
template <typename T> struct simd_type<T &&> {
  using type = typename simd_type<T>::type;
};
template <typename T> struct simd_type<const T> {
  using type = typename simd_type<T>::type;
};

template <typename T> struct dword_type {
  using type = T;
};
template <> struct dword_type<char> {
  using type = int;
};
template <> struct dword_type<short> {
  using type = int;
};
template <> struct dword_type<uchar> {
  using type = uint;
};
template <> struct dword_type<ushort> {
  using type = uint;
};

template <typename T> struct byte_type {
  using type = T;
};
template <> struct byte_type<short> {
  using type = char;
};
template <> struct byte_type<int> {
  using type = char;
};
template <> struct byte_type<ushort> {
  using type = uchar;
};
template <> struct byte_type<uint> {
  using type = uchar;
};

template <typename T> struct word_type {
  using type = T;
};
template <> struct word_type<char> {
  using type = short;
};
template <> struct word_type<int> {
  using type = short;
};
template <> struct word_type<uchar> {
  using type = ushort;
};
template <> struct word_type<uint> {
  using type = ushort;
};

// Utility for compile time loop unrolling.
template <unsigned N> class ForHelper {
  template <unsigned I, typename Action> static void repeat(Action A) {
    if constexpr (I < N)
      A(I);
    if constexpr (I + 1 < N)
      repeat<I + 1, Action>(A);
  }

public:
  template <typename Action> static void unroll(Action A) {
    ForHelper::template repeat<0, Action>(A);
  }
};

#ifdef __ESIMD_FORCE_STATELESS_MEM
/// Returns the address referenced by the accessor \p Acc and
/// the byte offset \p Offset.
template <typename T, typename AccessorTy, typename OffsetTy = uint32_t>
auto accessorToPointer(AccessorTy Acc, OffsetTy Offset = 0) {
  using QualCharPtrType =
      std::conditional_t<std::is_const_v<typename AccessorTy::value_type>,
                         const char *, char *>;
  using QualTPtrType =
      std::conditional_t<std::is_const_v<typename AccessorTy::value_type>,
                         const T *, T *>;
  auto BytePtr =
      reinterpret_cast<QualCharPtrType>(
          Acc.template get_multi_ptr<access::decorated::yes>().get()) +
      Offset;
  return reinterpret_cast<QualTPtrType>(BytePtr);
}
#endif // __ESIMD_FORCE_STATELESS_MEM

/// @brief Checks parameters for read region intrinsic API. The checks were
/// refactored from simd_obj API.
/// @tparam N the input vector size.
/// @tparam M the return vector size.
/// @tparam VStride the vertical stride in elements between rows.
/// @tparam Width the size or each row, non-zero and even divides `M`.
/// @tparam Stride horizontal stride in elements within each row.
// The rdregion intrinsics computes a result vector using following algorithm:
//
// \code{.cpp}
// uint16_t EltOffset = Offset / sizeof(T);
// assert(Offset % sizeof(T) == 0);
//
// int NumRows = M / Width;
// assert(M % Width == 0);
//
// int Index = 0;
// for (int i = 0; i < NumRows; ++i) {
//   for (int j = 0; j < Width; ++j) {
//     Result[Index++] = Input[i * VStride +  j * Stride +
//     EltOffset];
//   }
// }
// \endcode
// Hence the checks are to prevent reading beyond the input vector.
template <int N, int M, int VStride, int Width, int Stride>
constexpr void check_rdregion_params() {
  static_assert(Width > 0 && M % Width == 0, "Malformed RHS region.");
  static_assert(Width == M ||
                    ((M / Width) - 1) * VStride + (Width - 1) * Stride < N,
                "Malformed RHS region - too big vertical and/or "
                "horizontal stride.");
}

/// @brief Checks parameters for write region intrinsic API. The checks were
/// refactored from simd_obj API.
/// @tparam N the input vector size.
/// @tparam M the return vector size.
/// @tparam VStride the vertical stride in elements between rows.
/// @tparam Width the size or each row, non-zero and even divides `M`.
/// @tparam Stride horizontal stride in elements within each row.
// The wrregion intrinsics computes a result vector using following algorithm:
//
// \code{.cpp}
// uint16_t EltOffset = Offset / sizeof(T);
// assert(Offset % sizeof(T) == 0);
//
// int NumRows = M / Width;
// assert(M % Width == 0);
//
// Result = OldValue;
// int Index = 0;
// for (int i = 0; i < NumRows; ++i) {
//   for (int j = 0; j < Width; ++j) {
//       if (Mask[Index])
//           Result[i * VStride +  j * Stride + EltOffset] = NewVal[Index];
//       ++Index;
//   }
// }
// \endcode
// Hence the checks are to prevent reading beyond the input array and prevent
// writing beyond destination vector.
template <int N, int M, int VStride, int Width, int Stride>
constexpr void check_wrregion_params() {
  static_assert(M <= N, "Attempt to access beyond viewed area: The "
                        "viewed object in LHS does not fit RHS.");
  static_assert((M - 1) * Stride < N, "Malformed RHS region - too big stride.");
  check_rdregion_params<N, M, VStride, Width, Stride>();
}

// Generate an array of bitmasks for compressed load/store -- all 1 bits
// strictly less than i -- [0 1 3 7 15 31 63 127 255 511 1023 2047 ... ]
template <uint32_t... args> struct CompressedBitmask {
  static const uint32_t value[sizeof...(args)];
};

template <uint32_t... args>
const uint32_t CompressedBitmask<args...>::value[sizeof...(args)] = {args...};

template <int N, unsigned... args> struct GenerateCompressedBitmaskImpl {
  using value =
      typename GenerateCompressedBitmaskImpl<N - 1, ~(((uint32_t)(~0)) << N),
                                             args...>::value;
};

template <unsigned... args> struct GenerateCompressedBitmaskImpl<0, args...> {
  using value = CompressedBitmask<0, args...>;
};

template <int N> struct GenerateCompressedBitmask {
  using value = typename GenerateCompressedBitmaskImpl<N - 1>::value;
};

} // namespace ext::intel::esimd::detail
} // namespace _V1
} // namespace sycl

/// @endcond ESIMD_DETAIL
