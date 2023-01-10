//===---------- static-query-use.hpp - SYCL matrix ------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
// This file implements the static query interface for the joint_matrix
// experimental extension. Intel AMX, Intel XMX, and Nvidia Tensor Cores support
// different logical sizes and types. The query interface is used to validate
// user code and inform them about supported types, sizes, scopes, and layouts
// by the current implementation. Note that this query interface is a
// compile-time query, so there will be no runtime errors. The query interface
// provides three functionalities: 1- At compile time, inform the user whether a
// specific combination is valid or not. 2- Construct the matrices using a
// default shape if user does not provide a combination 3- General query
// interface for sizes, types, scopes. This is needed to void padding by the
// user, for tuning, and efficient code generation if used by a library.

#pragma once

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental::matrix {

enum class tpu {
  xmx8,
  xmx16,
  amx,
};
enum class matrix_type {
  bf8,
  bf16,
  fp16,
  tf32,
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
          int sM = 0, int sN = 0, int sK = 0, typename Enabled = void>
struct tpu_params;

template <typename Ta, typename Tb, typename Tc>
constexpr bool is_combination_valid_amx(int sM, int sN, int sK) {
  // is_same_v is a C++17 feature
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && sM <= 16 && sN <= 16 && sK <= 64) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && sM <= 16 && sN <= 16 && sK <= 64) ||
      (std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && sM <= 16 && sN <= 16 && sK <= 64) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && sM <= 16 && sN <= 16 && sK <= 64) ||
      // bf16
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float> &&
       sM <= 16 && sN <= 16 && sK <= 32))
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

// General query:
// types are not given, no default sizes and no implicit matrix construction
template <int sM, int sN, int sK>
struct tpu_params<tpu::amx, void, void, void, sM, sN, sK> {
  static constexpr std::size_t M = -1; // depends on the type
  static constexpr std::size_t N = -1;
  static constexpr std::size_t K = -1;

  uint32_t numtiles = 8;
  static constexpr scope_t scopes[] = {scope_t::sub_group};
  static constexpr int num_scopes = sizeof(scopes) / sizeof(scope_t);
  struct combination {
    uint32_t max_msize;
    uint32_t max_nsize;
    uint32_t max_ksize;
    matrix_type atype;
    matrix_type btype;
    matrix_type accumulatortype;
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
  static constexpr std::size_t M = 16;
  static constexpr std::size_t N = 16;
  static constexpr std::size_t K = ((sizeof(Ta) == 1) ? 64 : 32);

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_accumulator =
      joint_matrix<Group, Tc, use::accumulator, M, N>;

  uint32_t numtiles = 8;
  static constexpr scope_t scopes[] = {scope_t::sub_group};
  static constexpr int num_scopes = sizeof(scopes) / sizeof(scope_t);
  struct combination {
    uint32_t max_msize;
    uint32_t max_nsize;
    uint32_t max_ksize;
    matrix_type atype;
    matrix_type btype;
    matrix_type accumulatortype;
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
template <typename Ta, typename Tb, typename Tc, int sM, int sN, int sK>
struct tpu_params<
    tpu::amx, Ta, Tb, Tc, sM, sN, sK,
    typename std::enable_if<(
        !std::is_same_v<Ta, void> && !std::is_same_v<Tb, void> &&
        !std::is_same_v<Tc, void> && sM != 0 && sN != 0 && sK != 0)>::type> {
  // Validate that parameters are supported
  static_assert(
      (sM == 0 && sN == 0 && sK == 0) ||
          (is_combination_valid_amx<Ta, Tb, Tc>(sM, sN, sK)),
      "Invalid parameters for AMX, query valid types and maximum sizes "
      "using: tpu_params<tpu::amx> myparams; and then check out "
      "myparams.combinations array");

  // if combination is valid, construct the matrices

  static constexpr std::size_t M = (sM != 0) ? sM : 16;
  static constexpr std::size_t N = (sN != 0) ? sN : 16;
  static constexpr std::size_t K =
      (sK != 0) ? sK : ((sizeof(Ta) == 1) ? 64 : 32);

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_accumulator =
      joint_matrix<Group, Tc, use::accumulator, M, N>;

  uint32_t numtiles = 8;
  static constexpr scope_t scopes[] = {scope_t::sub_group};
  static constexpr int num_scopes = sizeof(scopes) / sizeof(scope_t);
};

// Intel XMX with SIMD8 capability
// The Intel XMX implementation supports the logical capability support of the
// HW So in this case, M, N, K sizes returned by the query represent the logical
// capabilities of the Intel XMX hardware.

template <typename Ta, typename Tb, typename Tc>
constexpr bool is_combination_valid_xmx8(int sM, int sN, int sK) {
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && (sM == 1 || sM == 2 || sM == 4 || sM == 8) &&
       sN == 8 && sK == 32) ||
      (std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && (sM == 1 || sM == 2 || sM == 4 || sM == 8) &&
       sN == 8 && sK == 32) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && (sM == 1 || sM == 2 || sM == 4 || sM == 8) &&
       sN == 8 && sK == 32) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && (sM == 1 || sM == 2 || sM == 4 || sM == 8) &&
       sN == 8 && sK == 32) ||
      (std::is_same_v<Ta, half> && std::is_same_v<Tb, half> &&
       std::is_same_v<Tc, float> &&
       (sM == 1 || sM == 2 || sM == 4 || sM == 8) && sN == 8 && sK == 16) ||
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float> &&
       (sM == 1 || sM == 2 || sM == 4 || sM == 8) && sN == 8 && sK == 16))
    return true;
  else
    return false;
}

template <typename Ta, typename Tb, typename Tc>
constexpr bool are_types_valid_xmx8() {
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

// General Query
// specialization for when types are not given --> no default values
template <int sM, int sN, int sK>
struct tpu_params<tpu::xmx8, void, void, void, sM, sN, sK> {
  static constexpr std::size_t M = -1; // depends on the type
  static constexpr std::size_t N = -1;
  static constexpr std::size_t K = -1;

  uint32_t numtiles = -1; // does not apply for XMX8
  static constexpr scope_t scopes[] = {scope_t::sub_group};
  static constexpr int num_scopes = sizeof(scopes) / sizeof(scope_t);

  struct combination {
    uint32_t max_msize;
    uint32_t max_nsize;
    uint32_t max_ksize;
    matrix_type atype;
    matrix_type btype;
    matrix_type accumulatortype;
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

template <typename Ta, typename Tb, typename Tc>
struct tpu_params<tpu::xmx8, Ta, Tb, Tc, 0, 0, 0,
                  typename std::enable_if<(!std::is_same_v<Ta, void> &&
                                           !std::is_same_v<Tb, void> &&
                                           !std::is_same_v<Tc, void>)>::type> {
  static_assert((are_types_valid_xmx8<Ta, Tb, Tc>()),
                "Invalid types for XMX8, supported types are int8_t, uint8_t, "
                "half, and bf16 (Note that unsigned short should be used in the"
                "DPC++ code to implement bf16)");

  // construct the matrices using the default sizes

  static constexpr std::size_t M = 8;
  static constexpr std::size_t N = 8;
  static constexpr std::size_t K = ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_accumulator =
      joint_matrix<Group, Tc, use::accumulator, M, N>;

  uint32_t numtiles = -1; // does not apply for XMX8
  static constexpr scope_t scopes[] = {scope_t::sub_group};
  static constexpr int num_scopes = sizeof(scopes) / sizeof(scope_t);
  struct combination {
    uint32_t max_msize;
    uint32_t max_nsize;
    uint32_t max_ksize;
    matrix_type atype;
    matrix_type btype;
    matrix_type accumulatortype;
    uint32_t msize;
    uint32_t nsize;
    uint32_t ksize;
  };
  using mt = matrix_type;
  static constexpr combination combinations[] = {
      // The types used in the initialization below are fake and not used. In
      // this case, users already chose the types, they are only looking for
      // the
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
template <typename Ta, typename Tb, typename Tc, int sM, int sN, int sK>
struct tpu_params<
    tpu::xmx8, Ta, Tb, Tc, sM, sN, sK,
    typename std::enable_if<((!std::is_same_v<Ta, void> && sM != 0))>::type> {
  // Validate that parameters are supported
  static_assert((sM == 0 && sN == 0 && sK == 0) ||
                    (is_combination_valid_xmx8<Ta, Tb, Tc>(sM, sN, sK)),
                "Invalid parameters for XMX8, query valid combinations "
                "using: tpu_params<tpu::xmx8> myparams; and then check out "
                "myparams.combinations array");

  // if combination is valid, construct the matrices
  static constexpr std::size_t M = (sM != 0) ? sM : 8;
  static constexpr std::size_t N = (sN != 0) ? sN : 8;
  static constexpr std::size_t K =
      (sK != 0) ? sK : ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_accumulator =
      joint_matrix<Group, Tc, use::accumulator, M, N>;

  uint32_t numtiles = -1; // does not apply for XMX8
  static constexpr scope_t scopes[] = {scope_t::sub_group};
  static constexpr int num_scopes = sizeof(scopes) / sizeof(scope_t);
};

// Intel XMX with SIMD16 capability
// The Intel XMX implementation supports the logical capability support of the
// HW So in this case, M, N, K sizes returned by the query represent the logical
// capabilities of the Intel XMX hardware.

template <typename Ta, typename Tb, typename Tc>
constexpr bool is_combination_valid_xmx16(int sM, int sN, int sK) {
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && (sM == 1 || sM == 2 || sM == 4 || sM == 8) &&
       sN == 16 && sK == 32) ||
      (std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && (sM == 1 || sM == 2 || sM == 4 || sM == 8) &&
       sN == 16 && sK == 32) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && (sM == 1 || sM == 2 || sM == 4 || sM == 8) &&
       sN == 16 && sK == 32) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && (sM == 1 || sM == 2 || sM == 4 || sM == 8) &&
       sN == 16 && sK == 32) ||
      (std::is_same_v<Ta, half> && std::is_same_v<Tb, half> &&
       std::is_same_v<Tc, float> &&
       (sM == 1 || sM == 2 || sM == 4 || sM == 8) && sN == 16 && sK == 16) ||
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float> &&
       (sM == 1 || sM == 2 || sM == 4 || sM == 8) && sN == 16 && sK == 16))
    return true;
  else
    return false;
}

template <typename Ta, typename Tb, typename Tc>
constexpr bool are_types_valid_xmx16() {
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

// General Query
// specialization for when types are not given --> no default values
template <int sM, int sN, int sK>
struct tpu_params<tpu::xmx16, void, void, void, sM, sN, sK> {
  static constexpr std::size_t M = -1; // depends on the type
  static constexpr std::size_t N = -1;
  static constexpr std::size_t K = -1;

  uint32_t numtiles = -1; // does not apply for XMX
  static constexpr scope_t scopes[] = {scope_t::sub_group};
  static constexpr int num_scopes = sizeof(scopes) / sizeof(scope_t);

  struct combination {
    uint32_t max_msize;
    uint32_t max_nsize;
    uint32_t max_ksize;
    matrix_type atype;
    matrix_type btype;
    matrix_type accumulatortype;
    uint32_t msize;
    uint32_t nsize;
    uint32_t ksize;
  };
  using mt = matrix_type;
  static constexpr combination combinations[] = {
      {0, 0, 0, mt::sint8, mt::sint8, mt::sint32, 1, 16, 32},
      {0, 0, 0, mt::sint8, mt::sint8, mt::sint32, 2, 16, 32},
      {0, 0, 0, mt::sint8, mt::sint8, mt::sint32, 4, 16, 32},
      {0, 0, 0, mt::sint8, mt::sint8, mt::sint32, 8, 16, 32},
      {0, 0, 0, mt::sint8, mt::uint8, mt::sint32, 1, 16, 32},
      {0, 0, 0, mt::sint8, mt::uint8, mt::sint32, 2, 16, 32},
      {0, 0, 0, mt::sint8, mt::uint8, mt::sint32, 4, 16, 32},
      {0, 0, 0, mt::sint8, mt::uint8, mt::sint32, 8, 16, 32},
      {0, 0, 0, mt::uint8, mt::sint8, mt::sint32, 1, 16, 32},
      {0, 0, 0, mt::uint8, mt::sint8, mt::sint32, 2, 16, 32},
      {0, 0, 0, mt::uint8, mt::sint8, mt::sint32, 4, 16, 32},
      {0, 0, 0, mt::uint8, mt::sint8, mt::sint32, 8, 16, 32},
      {0, 0, 0, mt::uint8, mt::uint8, mt::sint32, 1, 16, 32},
      {0, 0, 0, mt::uint8, mt::uint8, mt::sint32, 2, 16, 32},
      {0, 0, 0, mt::uint8, mt::uint8, mt::sint32, 4, 16, 32},
      {0, 0, 0, mt::uint8, mt::uint8, mt::sint32, 8, 16, 32},
      {0, 0, 0, mt::fp16, mt::fp16, mt::fp32, 1, 16, 16},
      {0, 0, 0, mt::fp16, mt::fp16, mt::fp32, 2, 16, 16},
      {0, 0, 0, mt::fp16, mt::fp16, mt::fp32, 4, 16, 16},
      {0, 0, 0, mt::fp16, mt::fp16, mt::fp32, 8, 16, 16},
      {0, 0, 0, mt::bf16, mt::bf16, mt::fp32, 1, 16, 16},
      {0, 0, 0, mt::bf16, mt::bf16, mt::fp32, 2, 16, 16},
      {0, 0, 0, mt::bf16, mt::bf16, mt::fp32, 4, 16, 16},
      {0, 0, 0, mt::bf16, mt::bf16, mt::fp32, 8, 16, 16},
  };
  static constexpr int num_combinations =
      sizeof(combinations) / sizeof(combination);
};

// Sizes-only query:
// Specialization for when only types are given, need to query only sizes

template <typename Ta, typename Tb, typename Tc>
struct tpu_params<tpu::xmx16, Ta, Tb, Tc, 0, 0, 0,
                  typename std::enable_if<(!std::is_same_v<Ta, void> &&
                                           !std::is_same_v<Tb, void> &&
                                           !std::is_same_v<Tc, void>)>::type> {
  static_assert((are_types_valid_xmx16<Ta, Tb, Tc>()),
                "Invalid types for XMX16, supported types are int8_t, uint8_t, "
                "half, and bf16 (Note that unsigned short should be used in the"
                "DPC++ code to implement bf16)");

  // construct the matrices using the default sizes

  static constexpr std::size_t M = 8;
  static constexpr std::size_t N = 16;
  static constexpr std::size_t K = ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_accumulator =
      joint_matrix<Group, Tc, use::accumulator, M, N>;

  uint32_t numtiles = -1; // does not apply for XMX
  static constexpr scope_t scopes[] = {scope_t::sub_group};
  static constexpr int num_scopes = sizeof(scopes) / sizeof(scope_t);
  struct combination {
    uint32_t max_msize;
    uint32_t max_nsize;
    uint32_t max_ksize;
    matrix_type atype;
    matrix_type btype;
    matrix_type accumulatortype;
    uint32_t msize;
    uint32_t nsize;
    uint32_t ksize;
  };
  using mt = matrix_type;
  static constexpr combination combinations[] = {
      // The types used in the initialization below are fake and not used. In
      // this case, users already chose the types, they are only looking for
      // the
      // sizes
      {0, 0, 0, mt::bf8, mt::bf8, mt::bf8, 1, 16, (sizeof(Ta) == 1) ? 32 : 16},
      {0, 0, 0, mt::bf8, mt::bf8, mt::bf8, 2, 16, (sizeof(Ta) == 1) ? 32 : 16},
      {0, 0, 0, mt::bf8, mt::bf8, mt::bf8, 4, 16, (sizeof(Ta) == 1) ? 32 : 16},
      {0, 0, 0, mt::bf8, mt::bf8, mt::bf8, 8, 16, (sizeof(Ta) == 1) ? 32 : 16},
  };
  static constexpr int num_combinations =
      sizeof(combinations) / sizeof(combination);
};

// Valid or not:
// Specialization when both types and sizes are given
template <typename Ta, typename Tb, typename Tc, int sM, int sN, int sK>
struct tpu_params<
    tpu::xmx16, Ta, Tb, Tc, sM, sN, sK,
    typename std::enable_if<((!std::is_same_v<Ta, void> && sM != 0))>::type> {
  // Validate that parameters are supported
  static_assert((sM == 0 && sN == 0 && sK == 0) ||
                    (is_combination_valid_xmx16<Ta, Tb, Tc>(sM, sN, sK)),
                "Invalid parameters for XMX16, query valid combinations "
                "using: tpu_params<tpu::xmx16> myparams; and then check out "
                "myparams.combinations array");

  // if combination is valid, construct the matrices
  static constexpr std::size_t M = (sM != 0) ? sM : 8;
  static constexpr std::size_t N = (sN != 0) ? sN : 8;
  static constexpr std::size_t K =
      (sK != 0) ? sK : ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_accumulator =
      joint_matrix<Group, Tc, use::accumulator, M, N>;

  uint32_t numtiles = -1; // does not apply for XMX16
  static constexpr scope_t scopes[] = {scope_t::sub_group};
  static constexpr int num_scopes = sizeof(scopes) / sizeof(scope_t);
};
} // namespace experimental::matrix
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
