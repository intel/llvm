//===---------- static-query-use.hpp - SYCL matrix ------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
// This file implements the static query interface for the joint_matrix
// experimental extension. Intel(R) Advanced Matrix Extensions (Intel(R) AMX),
// and Intel(R) Xe Matrix Extensions (Intel(R) XMX) support different logical
// sizes and types. The query interface is used to validate user code and inform
// them about supported types, sizes, scopes, and layouts by the current
// implementation. Note that this query interface is a compile-time query, so
// there will be no runtime errors. The query interface provides three
// functionalities: 1- At compile time, inform the user whether a specific
// combination is valid or not. 2- Construct the matrices using a default shape
// if user does not provide a combination 3- General query interface for sizes,
// types, scopes. This is needed to void padding by the user, for tuning, and
// efficient code generation if used by a library.

#pragma once

#include <sycl/aliases.hpp> // for half
#include <sycl/ext/oneapi/matrix/matrix-tensorcores.hpp>
#include <sycl/ext/oneapi/matrix/matrix-unified-utils.hpp> // for use, layout
#include <sycl/ext/oneapi/matrix/matrix-unified.hpp>       // for joint_matrix

#include <cstddef>     // for size_t
#include <stdint.h>    // for uint32_t, int8_t
#include <type_traits> // for enable_if

namespace sycl {
inline namespace _V1 {
namespace ext {
namespace oneapi {
namespace experimental::matrix {

template <architecture u, typename Ta, typename Tb, typename Tc,
          typename Td = Tc, size_t sM = 0, size_t sN = 0, size_t sK = 0,
          typename Enabled = void>
struct matrix_params;

template <typename Ta, typename Tb, typename Tc>
constexpr bool is_combination_valid_amx(size_t sM, size_t sN, size_t sK) {
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

// Default values query
// Specialization for when only types are given, need to query only sizes
template <typename Ta, typename Tb, typename Tc, typename Td>
struct matrix_params<
    architecture::intel_cpu_spr, Ta, Tb, Tc, Td, 0, 0, 0,
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
  using joint_matrix_c = joint_matrix<Group, Tc, use::accumulator, M, N>;
  template <typename Group>
  using joint_matrix_d = joint_matrix<Group, Td, use::accumulator, M, N>;
};

// Validation query
// Specialization when both types and sizes are given
template <typename Ta, typename Tb, typename Tc, typename Td, size_t sM,
          size_t sN, size_t sK>
struct matrix_params<
    architecture::intel_cpu_spr, Ta, Tb, Tc, Td, sM, sN, sK,
    typename std::enable_if<(
        !std::is_same_v<Ta, void> && !std::is_same_v<Tb, void> &&
        !std::is_same_v<Tc, void> && sM != 0 && sN != 0 && sK != 0)>::type> {
  // Validate that parameters are supported
  static_assert(
      (sM == 0 && sN == 0 && sK == 0) ||
          (is_combination_valid_amx<Ta, Tb, Tc>(sM, sN, sK)),
      "Invalid parameters for AMX, query valid types and maximum sizes "
      "using: matrix_params<architecture::intel_cpu_spr> myparams; and then "
      "check out "
      "myparams.combinations array");

  // if combination is valid, construct the matrices

  static constexpr std::size_t M = sM;
  static constexpr std::size_t N = sN;
  static constexpr std::size_t K = sK;

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_c = joint_matrix<Group, Tc, use::accumulator, M, N>;
  template <typename Group>
  using joint_matrix_d = joint_matrix<Group, Td, use::accumulator, M, N>;
};

// Intel XMX with SIMD8 capability
// The Intel XMX implementation supports the logical capability support of the
// HW So in this case, M, N, K sizes returned by the query represent the logical
// capabilities of the Intel XMX hardware.

template <typename Ta, typename Tb, typename Tc>
constexpr bool is_combination_valid_xmx8(size_t sM, size_t sN, size_t sK) {
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && (sM >= 1 && sM <= 8) && sN == 8 &&
       sK == 32) ||
      (std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && (sM >= 1 && sM <= 8) && sN == 8 &&
       sK == 32) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && (sM >= 1 && sM <= 8) && sN == 8 &&
       sK == 32) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && (sM >= 1 && sM <= 8) && sN == 8 &&
       sK == 32) ||
      (std::is_same_v<Ta, half> && std::is_same_v<Tb, half> &&
       std::is_same_v<Tc, float> && (sM >= 1 && sM <= 8) && sN == 8 &&
       sK == 16) ||
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float> &&
       (sM >= 1 && sM <= 8) && sN == 8 && sK == 16))
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

// Default-values query:
// Specialization for when only types are given, need to query only sizes

template <typename Ta, typename Tb, typename Tc, typename Td>
struct matrix_params<
    architecture::intel_gpu_dg2_g10, Ta, Tb, Tc, Td, 0, 0, 0,
    typename std::enable_if<(!std::is_same_v<Ta, void> &&
                             !std::is_same_v<Tb, void> &&
                             !std::is_same_v<Tc, void>)>::type> {
  static_assert((are_types_valid_xmx8<Ta, Tb, Tc>()),
                "Invalid types for architecture::intel_gpu_dg2_g10, supported "
                "types are int8_t, uint8_t, half, and bf16");

  // construct the matrices using the default sizes

  static constexpr std::size_t M = 8;
  static constexpr std::size_t N = 8;
  static constexpr std::size_t K = ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_c = joint_matrix<Group, Tc, use::accumulator, M, N>;
  template <typename Group>
  using joint_matrix_d = joint_matrix<Group, Td, use::accumulator, M, N>;
};

// Validation query:
// Specialization when both types and sizes are given
template <typename Ta, typename Tb, typename Tc, typename Td, size_t sM,
          size_t sN, size_t sK>
struct matrix_params<
    architecture::intel_gpu_dg2_g10, Ta, Tb, Tc, Td, sM, sN, sK,
    typename std::enable_if<(
        !std::is_same_v<Ta, void> && !std::is_same_v<Tb, void> &&
        !std::is_same_v<Tc, void> && sM != 0 && sN != 0 && sK != 0)>::type> {
  // Validate that parameters are supported
  static_assert(
      (sM == 0 && sN == 0 && sK == 0) ||
          (is_combination_valid_xmx8<Ta, Tb, Tc>(sM, sN, sK)),
      "Invalid parameters for XMX8, query valid combinations "
      "using: "
      "q.get_device().get_info<sycl::info::device::matrix::combinations>()");

  // if combination is valid, construct the matrices
  static constexpr std::size_t M = sM;
  static constexpr std::size_t N = sN;
  static constexpr std::size_t K = sK;

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_c = joint_matrix<Group, Tc, use::accumulator, M, N>;
  template <typename Group>
  using joint_matrix_d = joint_matrix<Group, Td, use::accumulator, M, N>;
};

// Default-values query:
// Specialization for when only types are given, need to query only sizes

template <typename Ta, typename Tb, typename Tc, typename Td>
struct matrix_params<
    architecture::intel_gpu_dg2_g11, Ta, Tb, Tc, Td, 0, 0, 0,
    typename std::enable_if<(!std::is_same_v<Ta, void> &&
                             !std::is_same_v<Tb, void> &&
                             !std::is_same_v<Tc, void>)>::type> {
  static_assert((are_types_valid_xmx8<Ta, Tb, Tc>()),
                "Invalid types for architecture::intel_gpu_dg2_g11, supported"
                "types are int8_t, uint8_t, half, and bf16");

  // construct the matrices using the default sizes

  static constexpr std::size_t M = 8;
  static constexpr std::size_t N = 8;
  static constexpr std::size_t K = ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_c = joint_matrix<Group, Tc, use::accumulator, M, N>;
  template <typename Group>
  using joint_matrix_d = joint_matrix<Group, Td, use::accumulator, M, N>;
};

// Validation query:
// Specialization when both types and sizes are given
template <typename Ta, typename Tb, typename Tc, typename Td, size_t sM,
          size_t sN, size_t sK>
struct matrix_params<
    architecture::intel_gpu_dg2_g11, Ta, Tb, Tc, Td, sM, sN, sK,
    typename std::enable_if<(
        !std::is_same_v<Ta, void> && !std::is_same_v<Tb, void> &&
        !std::is_same_v<Tc, void> && sM != 0 && sN != 0 && sK != 0)>::type> {
  // Validate that parameters are supported
  static_assert(
      (sM == 0 && sN == 0 && sK == 0) ||
          (is_combination_valid_xmx8<Ta, Tb, Tc>(sM, sN, sK)),
      "Invalid parameters for XMX8, query valid combinations "
      "using: "
      "q.get_device().get_info<sycl::info::device::matrix::combinations>()");

  // if combination is valid, construct the matrices
  static constexpr std::size_t M = sM;
  static constexpr std::size_t N = sN;
  static constexpr std::size_t K = sK;

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_c = joint_matrix<Group, Tc, use::accumulator, M, N>;
  template <typename Group>
  using joint_matrix_d = joint_matrix<Group, Td, use::accumulator, M, N>;
};

// Default-values query:
// Specialization for when only types are given, need to query only sizes

template <typename Ta, typename Tb, typename Tc, typename Td>
struct matrix_params<
    architecture::intel_gpu_dg2_g12, Ta, Tb, Tc, Td, 0, 0, 0,
    typename std::enable_if<(!std::is_same_v<Ta, void> &&
                             !std::is_same_v<Tb, void> &&
                             !std::is_same_v<Tc, void>)>::type> {
  static_assert((are_types_valid_xmx8<Ta, Tb, Tc>()),
                "Invalid types for architecture::intel_gpu_dg2_g12, supported "
                "types are int8_t, uint8_t, half, and bf16");

  // construct the matrices using the default sizes

  static constexpr std::size_t M = 8;
  static constexpr std::size_t N = 8;
  static constexpr std::size_t K = ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_c = joint_matrix<Group, Tc, use::accumulator, M, N>;
  template <typename Group>
  using joint_matrix_d = joint_matrix<Group, Td, use::accumulator, M, N>;
};

// Validation query:
// Specialization when both types and sizes are given
template <typename Ta, typename Tb, typename Tc, typename Td, size_t sM,
          size_t sN, size_t sK>
struct matrix_params<
    architecture::intel_gpu_dg2_g12, Ta, Tb, Tc, Td, sM, sN, sK,
    typename std::enable_if<(
        !std::is_same_v<Ta, void> && !std::is_same_v<Tb, void> &&
        !std::is_same_v<Tc, void> && sM != 0 && sN != 0 && sK != 0)>::type> {
  // Validate that parameters are supported
  static_assert(
      (sM == 0 && sN == 0 && sK == 0) ||
          (is_combination_valid_xmx8<Ta, Tb, Tc>(sM, sN, sK)),
      "Invalid parameters for XMX8, query valid combinations "
      "using: "
      "q.get_device().get_info<sycl::info::device::matrix::combinations>()");

  // if combination is valid, construct the matrices
  static constexpr std::size_t M = sM;
  static constexpr std::size_t N = sN;
  static constexpr std::size_t K = sK;

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_c = joint_matrix<Group, Tc, use::accumulator, M, N>;
  template <typename Group>
  using joint_matrix_d = joint_matrix<Group, Td, use::accumulator, M, N>;
};

// Intel XMX with SIMD16 capability
// The Intel XMX implementation supports the logical capability support of the
// HW So in this case, M, N, K sizes returned by the query represent the logical
// capabilities of the Intel XMX hardware.

template <typename Ta, typename Tb, typename Tc>
constexpr bool is_combination_valid_xmx16(size_t sM, size_t sN, size_t sK) {
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && (sM >= 1 && sM <= 8) && sN == 16 &&
       sK == 32) ||
      (std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && (sM >= 1 && sM <= 8) && sN == 16 &&
       sK == 32) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int> && (sM >= 1 && sM <= 8) && sN == 16 &&
       sK == 32) ||
      (std::is_same_v<Ta, uint8_t> && std::is_same_v<Tb, uint8_t> &&
       std::is_same_v<Tc, int> && (sM >= 1 && sM <= 8) && sN == 16 &&
       sK == 32) ||
      (std::is_same_v<Ta, half> && std::is_same_v<Tb, half> &&
       std::is_same_v<Tc, float> && (sM >= 1 && sM <= 8) && sN == 16 &&
       sK == 16) ||
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float> &&
       (sM >= 1 && sM <= 8) && sN == 16 && sK == 16))
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

// Default values query:
// Specialization for when only types are given, need to query only sizes

template <typename Ta, typename Tb, typename Tc, typename Td>
struct matrix_params<
    architecture::intel_gpu_pvc, Ta, Tb, Tc, Td, 0, 0, 0,
    typename std::enable_if<(!std::is_same_v<Ta, void> &&
                             !std::is_same_v<Tb, void> &&
                             !std::is_same_v<Tc, void>)>::type> {
  static_assert((are_types_valid_xmx16<Ta, Tb, Tc>()),
                "Invalid types for architecture::intel_gpu_pvc, supported "
                "types are int8_t, uint8_t, "
                "half, and bf16");

  // construct the matrices using the default sizes

  static constexpr std::size_t M = 8;
  static constexpr std::size_t N = 16;
  static constexpr std::size_t K = ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_c = joint_matrix<Group, Tc, use::accumulator, M, N>;
  template <typename Group>
  using joint_matrix_d = joint_matrix<Group, Td, use::accumulator, M, N>;
};

// Validation query:
// Specialization when both types and sizes are given
template <typename Ta, typename Tb, typename Tc, typename Td, size_t sM,
          size_t sN, size_t sK>
struct matrix_params<
    architecture::intel_gpu_pvc, Ta, Tb, Tc, Td, sM, sN, sK,
    typename std::enable_if<(
        !std::is_same_v<Ta, void> && !std::is_same_v<Tb, void> &&
        !std::is_same_v<Tc, void> && sM != 0 && sN != 0 && sK != 0)>::type> {
  // Validate that parameters are supported
  static_assert(
      (sM == 0 && sN == 0 && sK == 0) ||
          (is_combination_valid_xmx16<Ta, Tb, Tc>(sM, sN, sK)),
      "Invalid parameters for architecture::intel_gpu_pvc, query valid "
      "combinations "
      "using: "
      "q.get_device().get_info<sycl::info::device::matrix::combinations>()");

  // if combination is valid, construct the matrices
  static constexpr std::size_t M = sM;
  static constexpr std::size_t N = sN;
  static constexpr std::size_t K = sK;

  template <typename Group, layout Layout>
  using joint_matrix_a = joint_matrix<Group, Ta, use::a, M, K, Layout>;
  template <typename Group, layout Layout>
  using joint_matrix_b = joint_matrix<Group, Tb, use::b, K, N, Layout>;
  template <typename Group>
  using joint_matrix_c = joint_matrix<Group, Tc, use::accumulator, M, N>;
  template <typename Group>
  using joint_matrix_d = joint_matrix<Group, Td, use::accumulator, M, N>;
};
} // namespace experimental::matrix
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl
