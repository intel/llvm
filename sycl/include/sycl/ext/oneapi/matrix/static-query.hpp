//===-------------- static-query.hpp - SYCL matrix ------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

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
  // Note that unsigned variants are not implemented yet
  // is_same_v is a C++17 feature
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
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
  // Note that unsigned variants are not implemented yet
  if ((std::is_same_v<Ta, int8_t> && std::is_same_v<Tb, int8_t> &&
       std::is_same_v<Tc, int>) ||
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float>))
    return true;
  else
    return false;
}
#endif

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
  static constexpr int num_combinations = 2;
  combination combinations[num_combinations];
  constexpr tpu_params() : combinations() {
    // Note that unsigned int8 variants are not implemented yet
    combinations[0].atype = matrix_type::sint8;
    combinations[0].btype = matrix_type::sint8;
    combinations[0].ctype = matrix_type::sint32;
    combinations[0].max_msize = 16;
    combinations[0].max_nsize = 16;
    combinations[0].max_ksize = 64;
    combinations[1].atype = matrix_type::bf16;
    combinations[1].btype = matrix_type::bf16;
    combinations[1].ctype = matrix_type::fp32;
    combinations[1].max_msize = 16;
    combinations[1].max_nsize = 16;
    combinations[1].max_ksize = 32;
  }
};

#if __cplusplus >= 201703L
// Specialization for when only types are given, need to query only sizes
template <typename Ta, typename Tb, typename Tc>
struct tpu_params<tpu::amx, Ta, Tb, Tc, 0, 0, 0,
                  typename std::enable_if<(!std::is_same_v<Ta, void> &&
                                           !std::is_same_v<Tb, void> &&
                                           !std::is_same_v<Tc, void>)>::type> {
  static_assert((are_types_valid_amx<Ta, Tb, Tc>()),
                "Invalid types for AMX, supported types are int8_t,"
                "and bf16 (Note that unsigned short should be used in the"
                "DPC++ code to implement bf16) ");

  // construct the matrices using the default sizes
  static constexpr std::size_t defaultM = 16;
  static constexpr std::size_t defaultN = 16;
  static constexpr std::size_t defaultK = ((sizeof(Ta) == 1) ? 64 : 32);

  template <typename Group>
  using joint_matrix_a =
      joint_matrix<Ta, defaultM, defaultK, matrix_layout::row_major, Group>;
  template <typename Group>
  using joint_matrix_b =
      joint_matrix<Tb, defaultK, defaultN, matrix_layout::packed_b, Group>;
  template <typename Group>
  using joint_matrix_c =
      joint_matrix<Tc, defaultM, defaultN, matrix_layout::row_major, Group>;

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
  static constexpr int num_combinations = 1;
  combination combinations[num_combinations];
  constexpr tpu_params() : combinations() {
    combinations[0].max_msize = 16;
    combinations[0].max_nsize = 16;
    combinations[0].max_ksize = (sizeof(Ta) == 1) ? 64 : 32;
  }
};

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
      "using: "
      "constexpr tpu_params<tpu::amx> myparams = tpu_params<tpu::amx>();");

  // if combination is valid, construct the matrices

  static constexpr std::size_t defaultM = (M != 0) ? M : 16;
  static constexpr std::size_t defaultN = (N != 0) ? N : 16;
  static constexpr std::size_t defaultK =
      (K != 0) ? K : ((sizeof(Ta) == 1) ? 64 : 32);

  template <typename Group>
  using joint_matrix_a =
      joint_matrix<Ta, defaultM, defaultK, matrix_layout::row_major, Group>;
  template <typename Group>
  using joint_matrix_b =
      joint_matrix<Tb, defaultK, defaultN, matrix_layout::packed_b, Group>;
  template <typename Group>
  using joint_matrix_c =
      joint_matrix<Tc, defaultM, defaultN, matrix_layout::row_major, Group>;

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
  static constexpr int num_combinations = 2;
  combination combinations[num_combinations];
  constexpr tpu_params() : combinations() {
    // Note that unsigned int8 variants are not implemented yet
    combinations[0].atype = matrix_type::sint8;
    combinations[0].btype = matrix_type::sint8;
    combinations[0].ctype = matrix_type::sint32;
    combinations[0].max_msize = 16;
    combinations[0].max_nsize = 16;
    combinations[0].max_ksize = 64;
    combinations[1].atype = matrix_type::bf16;
    combinations[1].btype = matrix_type::bf16;
    combinations[1].ctype = matrix_type::fp32;
    combinations[1].max_msize = 16;
    combinations[1].max_nsize = 16;
    combinations[1].max_ksize = 32;
  }
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
      (std::is_same_v<Ta, half> && std::is_same_v<Tb, half> &&
       std::is_same_v<Tc, float>) ||
      (std::is_same_v<Ta, unsigned short> &&
       std::is_same_v<Tb, unsigned short> && std::is_same_v<Tc, float>))
    return true;
  else
    return false;
}
#endif

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
  static constexpr int num_combinations = 12;
  combination combinations[num_combinations];
  constexpr tpu_params() : combinations() {
    int i = 0;
    combinations[i].atype = matrix_type::sint8;
    combinations[i].btype = matrix_type::sint8;
    combinations[i].ctype = matrix_type::sint32;
    combinations[i].msize = 1;
    combinations[i + 1].msize = 2;
    combinations[i + 2].msize = 4;
    combinations[i + 3].msize = 8;
    for (int ii = 0; ii < 4; ii++) {
      combinations[i + ii].ksize = 32;
      combinations[i + ii].nsize = 8;
    }
    i = 4;
    combinations[i].atype = matrix_type::fp16;
    combinations[i].btype = matrix_type::fp16;
    combinations[i].ctype = matrix_type::fp64;
    combinations[i].msize = 1;
    combinations[i + 1].msize = 2;
    combinations[i + 2].msize = 4;
    combinations[i + 3].msize = 8;
    for (int ii = 0; ii < 4; ii++) {
      combinations[i + ii].ksize = 16;
      combinations[i + ii].nsize = 8;
    }
    i = 8;
    combinations[i].atype = matrix_type::bf16;
    combinations[i].btype = matrix_type::bf16;
    combinations[i].ctype = matrix_type::fp64;
    combinations[i].msize = 1;
    combinations[i + 1].msize = 2;
    combinations[i + 2].msize = 4;
    combinations[i + 3].msize = 8;
    for (int ii = 0; ii < 4; ii++) {
      combinations[i + ii].ksize = 16;
      combinations[i + ii].nsize = 8;
    }
  }
};

// Specialization for when only types are given, need to query only sizes

#if __cplusplus >= 201703L
template <typename Ta, typename Tb, typename Tc>
struct tpu_params<tpu::dpas, Ta, Tb, Tc, 0, 0, 0,
                  typename std::enable_if<(!std::is_same_v<Ta, void> &&
                                           !std::is_same_v<Tb, void> &&
                                           !std::is_same_v<Tc, void>)>::type> {
  static_assert((are_types_valid_dpas<Ta, Tb, Tc>()),
                "Invalid types for DPAS, supported types are int8_t, "
                "half, and bf16");

  // construct the matrices using the default sizes

  static constexpr std::size_t defaultM = 8;
  static constexpr std::size_t defaultN = 8;
  static constexpr std::size_t defaultK = ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group>
  using joint_matrix_a =
      joint_matrix<Ta, defaultM, defaultK, matrix_layout::row_major, Group>;
  template <typename Group>
  using joint_matrix_b =
      joint_matrix<Tb, defaultK, defaultN, matrix_layout::packed_b, Group>;
  template <typename Group>
  using joint_matrix_c =
      joint_matrix<Tc, defaultM, defaultN, matrix_layout::row_major, Group>;

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
  static constexpr int num_combinations = 4;
  combination combinations[num_combinations];
  constexpr tpu_params() : combinations() {
    int i = 0;
    combinations[i].msize = 1;
    combinations[i + 1].msize = 2;
    combinations[i + 2].msize = 4;
    combinations[i + 3].msize = 8;
    for (int ii = 0; ii < 4; ii++) {
      combinations[i + ii].ksize = ((sizeof(Ta) == 1) ? 32 : 16);
      combinations[i + ii].nsize = 8;
    }
  }
};

// Specialization when both types and sizes are given
template <typename Ta, typename Tb, typename Tc, int M, int N, int K>
struct tpu_params<
    tpu::dpas, Ta, Tb, Tc, M, N, K,
    typename std::enable_if<((!std::is_same_v<Ta, void> && M != 0))>::type> {
  // Validate that parameters are supported
  static_assert((M == 0 && N == 0 && K == 0) ||
                    (is_combination_valid_dpas<Ta, Tb, Tc>(M, N, K)),
                "Invalid parameters for DPAS, query valid combinations "
                "using: constexpr "
                "tpu_params<tpu::dpas> myparams = tpu_params<tpu::dpas>(); ");

  // if combination is valid, construct the matrices
  static constexpr std::size_t defaultM = (M != 0) ? M : 8;
  static constexpr std::size_t defaultN = (N != 0) ? N : 8;
  static constexpr std::size_t defaultK =
      (K != 0) ? K : ((sizeof(Ta) == 1) ? 32 : 16);

  template <typename Group>
  using joint_matrix_a =
      joint_matrix<Ta, defaultM, defaultK, matrix_layout::row_major, Group>;
  template <typename Group>
  using joint_matrix_b =
      joint_matrix<Tb, defaultK, defaultN, matrix_layout::packed_b, Group>;
  template <typename Group>
  using joint_matrix_c =
      joint_matrix<Tc, defaultM, defaultN, matrix_layout::row_major, Group>;

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
  static constexpr int num_combinations = 12;
  combination combinations[num_combinations];
  constexpr tpu_params() : combinations() {
    int i = 0;
    combinations[i].atype = matrix_type::sint8;
    combinations[i].btype = matrix_type::sint8;
    combinations[i].ctype = matrix_type::sint32;
    combinations[i].msize = 1;
    combinations[i + 1].msize = 2;
    combinations[i + 2].msize = 4;
    combinations[i + 3].msize = 8;
    for (int ii = 0; ii < 4; ii++) {
      combinations[i + ii].ksize = 32;
      combinations[i + ii].nsize = 8;
    }
    i = 4;
    combinations[i].atype = matrix_type::fp16;
    combinations[i].btype = matrix_type::fp16;
    combinations[i].ctype = matrix_type::fp64;
    combinations[i].msize = 1;
    combinations[i + 1].msize = 2;
    combinations[i + 2].msize = 4;
    combinations[i + 3].msize = 8;
    for (int ii = 0; ii < 4; ii++) {
      combinations[i + ii].ksize = 16;
      combinations[i + ii].nsize = 8;
    }
    i = 8;
    combinations[i].atype = matrix_type::bf16;
    combinations[i].btype = matrix_type::bf16;
    combinations[i].ctype = matrix_type::fp64;
    combinations[i].msize = 1;
    combinations[i + 1].msize = 2;
    combinations[i + 2].msize = 4;
    combinations[i + 3].msize = 8;
    for (int ii = 0; ii < 4; ii++) {
      combinations[i + ii].ksize = 16;
      combinations[i + ii].nsize = 8;
    }
  }
};
#endif
} // namespace experimental::matrix
} // namespace oneapi
} // namespace ext
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
