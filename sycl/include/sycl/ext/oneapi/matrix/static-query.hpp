//===-------------- static-query.hpp - SYCL matrix ------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
// This file implements the static query interface for the joint_matrix
// experimental extension. AMX, DPAS and different other TPUs support different
// logical sizes and types. The query interface is used to validate user code
// and inform them about supported types, sizes, scope, and layouts by the
// current implementation. Note that this query interface is a compile-time
// query, so there will be no runtime errors. The query interface provides
// three functionalities:
// 1- At compile time, inform the user whether a specific
// combination is valid or not.
// 2- Construct the matrices using a default shape
// if user does not provide a combination
// 3- General query interface for sizes, types,
// static/dynamic, scope. This is needed to void padding by the user,
// for tuning, and efficient code generation if used by a library.

#pragma once

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace ext {
namespace oneapi {
namespace experimental::matrix {

enum class tpu {
  dpas,
  amx,
};
enum class matrix_type {
  bf8,
  bf16,
  fp16,
  fp19, // tfloat32
  fp32,
  fp64,
  sint2,
  sint4,
  sint8,
  sint16,
  sint32,
  sint64,
  uint2,
  uint4,
  uint8,
  uint16,
  uint32,
  uint64
};

enum class scope_t { sub_group, work_group };

template <tpu u, typename Ta = void, typename Tb = void, typename Tc = void,
          int M = 0, int N = 0, int K = 0, typename Enabled = void>
struct tpu_params;

#if __cplusplus >= 201703L
template <typename Ta, typename Tb, typename Tc>
constexpr bool is_combination_valid_amx(int M, int N, int K) {
  // is_same_v is a C++17 feature
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && M <= 16 && N <= 16 && K <= 64) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && M <= 16 && N <= 16 && K <= 64) ||
      (std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && M <= 16 && N <= 16 && K <= 64) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && M <= 16 && N <= 16 && K <= 64) ||
      // bf16
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float> &&
       M <= 16 && N <= 16 && K <= 32))
    return true;
  else
    return false;
}

template <typename Ta, typename Tb, typename Tc>
constexpr bool are_types_valid_amx() {
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float>))
    return true;
  else
    return false;
}
#endif

// General query:
// types are not given, no default sizes and no implicit matrix construction
template <int M, int N, int K>
struct tpu_params<tpu::amx, void, void, void, M, N, K> {
  static constexpr std::size_t defaultM = -1; // depends on the type
  static constexpr std::size_t defaultN = -1;
  static constexpr std::size_t defaultK = -1;

  bool dynamic_p = false; // should be true in future implementations because
                          // AMX hardware supports dynamic sizes
  uint32_t numtiles = 8;
  scope_t scope = scope_t::sub_group;
  struct combination {
    uint32_t max_msize;
    uint32_t max_nsize;
    uint32_t max_ksize;
    matrix_type atype;
    matrix_type btype;
    matrix_type ctype;
    uint32_t msize;
    uint32_t nsize;
    uint32_t ksize;
  };
  using mt = matrix_type;
  static constexpr combination combinations[] = {
      {16, 16, 64, mt::sint8, mt::sint8, mt::sint32},
      {16, 16, 64, mt::sint8, mt::uint8, mt::sint32},
      {16, 16, 64, mt::uint8, mt::sint8, mt::sint32},
      {16, 16, 64, mt::uint8, mt::uint8, mt::sint32},
      {16, 16, 32, mt::bf16, mt::bf16, mt::fp32}};
  static constexpr int num_combinations =
      sizeof(combinations) / sizeof(combination);
};

#if __cplusplus >= 201703L
// Sizes-only query
// Specialization for when only types are given, need to query only sizes
template <typename Ta, typename Tb, typename Tc>
struct tpu_params<tpu::amx, Ta, Tb, Tc, 0, 0, 0,
                  typename std::enable_if<(!std::is_same_v<Ta, void> &&
                                           !std::is_same_v<Tb, void> &&
                                           !std::is_same_v<Tc, void>)>::type> {
  static_assert((are_types_valid_amx<Ta, Tb, Tc>()),
                "Invalid types for AMX, supported types are int8_t, uint8_t, "
                "and bf16 (Note that unsigned short should be used in the"
                "DPC++ code to implement bf16) ");

  // construct the matrices using the default sizes
  static constexpr std::size_t defaultM = 16;
  static constexpr std::size_t defaultN = 16;
  static constexpr std::size_t defaultK = ((sizeof(Ta) == 1) ? 64 : 32);

  template <typename Group>
  using joint_matrix_a =
      joint_matrix<Ta, defaultM, defaultK, matrix_layout::row_major,
                   matrix_use::unnecessary, Group>;
  template <typename Group>
  using joint_matrix_b =
      joint_matrix<Tb, defaultK, defaultN, matrix_layout::packed_b,
                   matrix_use::unnecessary, Group>;
  template <typename Group>
  using joint_matrix_c =
      joint_matrix<Tc, defaultM, defaultN, matrix_layout::row_major,
                   matrix_use::unnecessary, Group>;

  bool dynamic_p = false; // should be true in future implementations because
                          // AMX hardware supports dynamic sizes
  uint32_t numtiles = 8;
  scope_t scope = scope_t::sub_group;
  struct combination {
    uint32_t max_msize;
    uint32_t max_nsize;
    uint32_t max_ksize;
    matrix_type atype;
    matrix_type btype;
    matrix_type ctype;
    uint32_t msize;
    uint32_t nsize;
    uint32_t ksize;
  };
  static constexpr combination combinations[] = {
      {16, 16, (sizeof(Ta) == 1) ? 64 : 32}};
  static constexpr int num_combinations =
      sizeof(combinations) / sizeof(combination);
};

// Valid or not:
// Specialization when both types and sizes are given
template <typename Ta, typename Tb, typename Tc, int M, int N, int K>
struct tpu_params<
    tpu::amx, Ta, Tb, Tc, M, N, K,
    typename std::enable_if<(
        !std::is_same_v<Ta, void> && !std::is_same_v<Tb, void> &&
        !std::is_same_v<Tc, void> && M != 0 && N != 0 && K != 0)>::type> {
  // Validate that parameters are supported
  static_assert(
      (M == 0 && N == 0 && K == 0) ||
          (is_combination_valid_amx<Ta, Tb, Tc>(M, N, K)),
      "Invalid parameters for AMX, query valid types and maximum sizes "
      "using: tpu_params<tpu::amx> myparams; and then check out "
      "myparams.combinations array");

  // if combination is valid, construct the matrices

  static constexpr std::size_t defaultM = (M != 0) ? M : 16;
  static constexpr std::size_t defaultN = (N != 0) ? N : 16;
  static constexpr std::size_t defaultK =
      (K != 0) ? K : ((sizeof(Ta) == 1) ? 64 : 32);

  template <typename Group>
  using joint_matrix_a =
      joint_matrix<Ta, defaultM, defaultK, matrix_layout::row_major,
                   matrix_use::unnecessary, Group>;
  template <typename Group>
  using joint_matrix_b =
      joint_matrix<Tb, defaultK, defaultN, matrix_layout::packed_b,
                   matrix_use::unnecessary, Group>;
  template <typename Group>
  using joint_matrix_c =
      joint_matrix<Tc, defaultM, defaultN, matrix_layout::row_major,
                   matrix_use::unnecessary, Group>;

  bool dynamic_p = false; // should be true in future implementations
                          // because AMX hardware supports dynamic sizes
  uint32_t numtiles = 8;
  scope_t scope = scope_t::sub_group;
};

// DPAS case
// The DPAS implementation supports the logical capability support of the HW
// So in this case, M, N, K sizes returned by the query represent the logical
// capabilities of the DPAS hardware.

template <typename Ta, typename Tb, typename Tc>
constexpr bool is_combination_valid_dpas(int M, int N, int K) {
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && (M == 1 || M == 2 || M == 4 || M == 8) &&
       N == 8 && K == 32) ||
      (std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && (M == 1 || M == 2 || M == 4 || M == 8) &&
       N == 8 && K == 32) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && (M == 1 || M == 2 || M == 4 || M == 8) &&
       N == 8 && K == 32) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && (M == 1 || M == 2 || M == 4 || M == 8) &&
       N == 8 && K == 32) ||
      (std::is_same_v<Ta, half> && std::is_same_v<Tb, half> &&
       std::is_same_v<Tc, float> && (M == 1 || M == 2 || M == 4 || M == 8) &&
       N == 8 && K == 16) ||
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float> &&
       (M == 1 || M == 2 || M == 4 || M == 8) && N == 8 && K == 16))
    return true;
  else
    return false;
}

template <typename Ta, typename Tb, typename Tc>
constexpr bool are_types_valid_dpas() {
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, half> && std::is_same_v<Tb, half> &&
       std::is_same_v<Tc, float>) ||
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float>))
    return true;
  else
    return false;
}
#endif

// General Query
// specialization for when types are not given --> no default values
template <int M, int N, int K>
struct tpu_params<tpu::dpas, void, void, void, M, N, K> {
  static constexpr std::size_t defaultM = -1; // depends on the type
  static constexpr std::size_t defaultN = -1;
  static constexpr std::size_t defaultK = -1;

  bool dynamic_p = false; // no dynamic allocation on the GPU
  uint32_t numtiles = -1; // does not apply for DPAS
  scope_t scope = scope_t::sub_group;

  struct combination {
    uint32_t max_msize;
    uint32_t max_nsize;
    uint32_t max_ksize;
    matrix_type atype;
    matrix_type btype;
    matrix_type ctype;
    uint32_t msize;
    uint32_t nsize;
    uint32_t ksize;
  };
  using mt = matrix_type;
  static constexpr combination combinations[] = {
      {0, 0, 0, mt::sint8, mt::sint8, mt::sint32, 1, 8, 32},
      {0, 0, 0, mt::sint8, mt::sint8, mt::sint32, 2, 8, 32},
      {0, 0, 0, mt::sint8, mt::sint8, mt::sint32, 4, 8, 32},
      {0, 0, 0, mt::sint8, mt::sint8, mt::sint32, 8, 8, 32},
      {0, 0, 0, mt::sint8, mt::uint8, mt::sint32, 1, 8, 32},
      {0, 0, 0, mt::sint8, mt::uint8, mt::sint32, 2, 8, 32},
      {0, 0, 0, mt::sint8, mt::uint8, mt::sint32, 4, 8, 32},
      {0, 0, 0, mt::sint8, mt::uint8, mt::sint32, 8, 8, 32},
      {0, 0, 0, mt::uint8, mt::sint8, mt::sint32, 1, 8, 32},
      {0, 0, 0, mt::uint8, mt::sint8, mt::sint32, 2, 8, 32},
      {0, 0, 0, mt::uint8, mt::sint8, mt::sint32, 4, 8, 32},
      {0, 0, 0, mt::uint8, mt::sint8, mt::sint32, 8, 8, 32},
      {0, 0, 0, mt::uint8, mt::uint8, mt::sint32, 1, 8, 32},
      {0, 0, 0, mt::uint8, mt::uint8, mt::sint32, 2, 8, 32},
      {0, 0, 0, mt::uint8, mt::uint8, mt::sint32, 4, 8, 32},
      {0, 0, 0, mt::uint8, mt::uint8, mt::sint32, 8, 8, 32},
      {0, 0, 0, mt::fp16, mt::fp16, mt::fp32, 1, 8, 16},
      {0, 0, 0, mt::fp16, mt::fp16, mt::fp32, 2, 8, 16},
      {0, 0, 0, mt::fp16, mt::fp16, mt::fp32, 4, 8, 16},
      {0, 0, 0, mt::fp16, mt::fp16, mt::fp32, 8, 8, 16},
      {0, 0, 0, mt::bf16, mt::bf16, mt::fp32, 1, 8, 16},
      {0, 0, 0, mt::bf16, mt::bf16, mt::fp32, 2, 8, 16},
      {0, 0, 0, mt::bf16, mt::bf16, mt::fp32, 4, 8, 16},
      {0, 0, 0, mt::bf16, mt::bf16, mt::fp32, 8, 8, 16},
  };
  static constexpr int num_combinations =
      sizeof(combinations) / sizeof(combination);
};

// Sizes-only query:
// Specialization for when only types are given, need to query only sizes

#if __cplusplus >= 201703L
template <typename Ta, typename Tb, typename Tc>
struct tpu_params<tpu::dpas, Ta, Tb, Tc, 0, 0, 0,
                  typename std::enable_if<(!std::is_same_v<Ta, void> &&
                                           !std::is_same_v<Tb, void> &&
                                           !std::is_same_v<Tc, void>)>::type> {
  static_assert((are_types_valid_dpas<Ta, Tb, Tc>()),
                "Invalid types for DPAS, supported types are int8_t, uint8_t, "
                "half, and bf16 (Note that unsigned short should be used in the"
                "DPC++ code to implement bf16)");

  // construct the matrices using the default sizes

  static constexpr std::size_t defaultM = 8;
  static constexpr std::size_t defaultN = 8;
  static constexpr std::size_t defaultK = ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group>
  using joint_matrix_a =
      joint_matrix<Ta, defaultM, defaultK, matrix_layout::row_major,
                   matrix_use::unnecessary, Group>;
  template <typename Group>
  using joint_matrix_b =
      joint_matrix<Tb, defaultK, defaultN, matrix_layout::packed_b,
                   matrix_use::unnecessary, Group>;
  template <typename Group>
  using joint_matrix_c =
      joint_matrix<Tc, defaultM, defaultN, matrix_layout::row_major,
                   matrix_use::unnecessary, Group>;

  bool dynamic_p = false; // no dynamic allocation on the GPU
  uint32_t numtiles = -1; // does not apply for DPAS
  scope_t scope = scope_t::sub_group;
  struct combination {
    uint32_t max_msize;
    uint32_t max_nsize;
    uint32_t max_ksize;
    matrix_type atype;
    matrix_type btype;
    matrix_type ctype;
    uint32_t msize;
    uint32_t nsize;
    uint32_t ksize;
  };
  using mt = matrix_type;
  static constexpr combination combinations[] = {
      // The types used in the initialization below are fake and not used. In
      // this case, users already chose the types, they are only looking for the
      // sizes
      {0, 0, 0, mt::bf8, mt::bf8, mt::bf8, 1, 8, (sizeof(Ta) == 1) ? 32 : 16},
      {0, 0, 0, mt::bf8, mt::bf8, mt::bf8, 2, 8, (sizeof(Ta) == 1) ? 32 : 16},
      {0, 0, 0, mt::bf8, mt::bf8, mt::bf8, 4, 8, (sizeof(Ta) == 1) ? 32 : 16},
      {0, 0, 0, mt::bf8, mt::bf8, mt::bf8, 8, 8, (sizeof(Ta) == 1) ? 32 : 16},
  };
  static constexpr int num_combinations =
      sizeof(combinations) / sizeof(combination);
};

// Valid or not:
// Specialization when both types and sizes are given
template <typename Ta, typename Tb, typename Tc, int M, int N, int K>
struct tpu_params<
    tpu::dpas, Ta, Tb, Tc, M, N, K,
    typename std::enable_if<((!std::is_same_v<Ta, void> && M != 0))>::type> {
  // Validate that parameters are supported
  static_assert((M == 0 && N == 0 && K == 0) ||
                    (is_combination_valid_dpas<Ta, Tb, Tc>(M, N, K)),
                "Invalid parameters for DPAS, query valid combinations "
                "using: tpu_params<tpu::dpas> myparams; and then check out "
                "myparams.combinations array");

  // if combination is valid, construct the matrices
  static constexpr std::size_t defaultM = (M != 0) ? M : 8;
  static constexpr std::size_t defaultN = (N != 0) ? N : 8;
  static constexpr std::size_t defaultK =
      (K != 0) ? K : ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group>
  using joint_matrix_a =
      joint_matrix<Ta, defaultM, defaultK, matrix_layout::row_major,
                   matrix_use::unnecessary, Group>;
  template <typename Group>
  using joint_matrix_b =
      joint_matrix<Tb, defaultK, defaultN, matrix_layout::packed_b,
                   matrix_use::unnecessary, Group>;
  template <typename Group>
  using joint_matrix_c =
      joint_matrix<Tc, defaultM, defaultN, matrix_layout::row_major,
                   matrix_use::unnecessary, Group>;

  bool dynamic_p = false; // no dynamic allocation on the GPU
  uint32_t numtiles = -1; // does not apply for DPAS
  scope_t scope = scope_t::sub_group;
};
#endif
} // namespace experimental::matrix
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
