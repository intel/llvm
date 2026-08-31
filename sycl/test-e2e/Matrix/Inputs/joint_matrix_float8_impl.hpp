//==---------- joint_matrix_float8_impl.hpp  - DPC++ joint_matrix-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <sycl/usm.hpp>

constexpr size_t TN = 16;
constexpr size_t TK = 32;

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          size_t TM, layout B_layout, unsigned int vnniFactor>
void joint_matrix_gemm_vnni(
    sub_group sg, size_t sg_startx, size_t sg_starty, size_t sg_size,
    multi_ptr<TA, sycl::access::address_space::global_space,
              access::decorated::no>
        pA,
    multi_ptr<TB, sycl::access::address_space::global_space,
              access::decorated::no>
        pB,
    multi_ptr<TC, sycl::access::address_space::global_space,
              access::decorated::no>
        pC) {
  joint_matrix<sub_group, TA, use::a, TM, TK, layout::row_major> sub_a;
  joint_matrix<sub_group, TB, use::b, TK, TN, B_layout> sub_b;
  joint_matrix<sub_group, TC, use::accumulator, TM, TN> sub_c;
  joint_matrix_load(sg, sub_c,
                    pC + (sg_startx * TM) * N + sg_starty / sg_size * TN, N,
                    layout::row_major);
  for (int k = 0; k < K; k += TK) {
    joint_matrix_load(sg, sub_a, pA + (sg_startx * TM) * K + k, K);
    joint_matrix_load(sg, sub_b,
                      pB + k * N + sg_starty / sg_size * TN * vnniFactor,
                      N * vnniFactor);
    joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  }
  joint_matrix_store(sg, sub_c,
                     pC + (sg_startx * TM) * N + sg_starty / sg_size * TN, N,
                     layout::row_major);
}

template <typename TA, typename TB, typename TC, size_t TM,
          unsigned int vnniFactor, layout B_layout>
class fp8matrix;

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          size_t TM, layout B_layout, unsigned int vnniFactor>
void matrix_multiply(TC *C, TA *A, TB *B, queue q) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;

  auto pA = address_space_cast<sycl::access::address_space::global_space,
                               access::decorated::no>(A);
  auto pB = address_space_cast<sycl::access::address_space::global_space,
                               access::decorated::no>(B);
  auto pC = address_space_cast<sycl::access::address_space::global_space,
                               access::decorated::no>(C);
  size_t sg_size =
      get_sg_size<fp8matrix<TA, TB, TC, TM, vnniFactor, B_layout>>(q);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<fp8matrix<TA, TB, TC, TM, vnniFactor, B_layout>>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix_gemm_vnni<TA, TB, TC, M, N, K, TM, B_layout,
                                  vnniFactor>(sg, sg_startx, sg_starty, sg_size,
                                              pA, pB, pC);
         }); // parallel for
   }).wait();
}

template <typename TA, typename TB, typename TC, size_t M, size_t N, size_t K,
          size_t TM, unsigned int vnniFactor>
void joint_matrix_verify(queue q) {
  sycl::half *Ah = malloc_shared<sycl::half>(M * K, q);
  sycl::half *Bh = malloc_shared<sycl::half>(K * N, q);
  TA *A = malloc_shared<TA>(M * K, q);
  TB *B = malloc_shared<TB>(K * N, q);
  TC *C = malloc_shared<TC>(M * N, q);
  TC *D = malloc_shared<TC>(M * N, q);

  matrix_rand<sycl::half>(M, K, Ah, 5);
  matrix_truncate_fraction_bits<TA>(M, K, Ah);
  matrix_rand<sycl::half>(K, N, Bh, 5);
  matrix_truncate_fraction_bits<TB>(K, N, Bh);
  matrix_fill(M, N, C, (TC)1);
  matrix_fill(M, N, D, (TC)1);
  // Assign Ah and Bh values to A and B
  matrix_copy(q, M, K, Ah, A);
  matrix_copy(q, K, N, Bh, B);

  if (vnniFactor > 1) {
    TB *vnniB = malloc_shared<TB>(K * N, q);
    matrix_vnni<TB>(K, N, B, vnniB, vnniFactor);
    matrix_multiply<TA, TB, TC, M, N, K, TM, layout::ext_intel_packed,
                    vnniFactor>(C, A, vnniB, q);
  } else {
    matrix_multiply<TA, TB, TC, M, N, K, TM, layout::row_major, vnniFactor>(
        C, A, B, q);
  }
  matrix_multiply_ref<sycl::half, sycl::half, TC>(Ah, Bh, D, M, N, K);
  assert(matrix_compare(M, N, C, D));
  free(A, q);
  free(B, q);
  free(Ah, q);
  free(Bh, q);
  free(C, q);
  free(D, q);
}

template <size_t TM> void bf8_hf8_combinations(queue q) {
  // scale could be increased to 8 once these tests run on native hardware
  static constexpr size_t SCALE = 2;
  static constexpr size_t MATRIX_M = TM * SCALE;
  // satisfy 64B stride requirement in 2D block load
  static constexpr size_t MATRIX_N = std::max(TN * SCALE, 64ul);
  static constexpr size_t MATRIX_K = TK * SCALE;

  joint_matrix_verify<syclex::fp8_e5m2, syclex::fp8_e5m2, float, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 4>(q);
  joint_matrix_verify<syclex::fp8_e5m2, syclex::fp8_e5m2, float, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 1>(q);
  joint_matrix_verify<syclex::fp8_e4m3, syclex::fp8_e4m3, float, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 4>(q);
  joint_matrix_verify<syclex::fp8_e4m3, syclex::fp8_e4m3, float, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 1>(q);
  joint_matrix_verify<syclex::fp8_e5m2, syclex::fp8_e4m3, float, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 4>(q);
  joint_matrix_verify<syclex::fp8_e5m2, syclex::fp8_e4m3, float, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 1>(q);
  joint_matrix_verify<syclex::fp8_e4m3, syclex::fp8_e5m2, float, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 4>(q);
  joint_matrix_verify<syclex::fp8_e4m3, syclex::fp8_e5m2, float, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 1>(q);
#if 0
  // These combinations are disabled by lack of bfloat16 accumulator support in IGC
  // Adding these is tracked by Jira GSD-10112
  joint_matrix_verify<syclex::fp8_e5m2, syclex::fp8_e5m2, bfloat16, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 4>(q);
  joint_matrix_verify<syclex::fp8_e5m2, syclex::fp8_e5m2, bfloat16, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 1>(q);
  joint_matrix_verify<syclex::fp8_e4m3, syclex::fp8_e4m3, bfloat16, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 4>(q);
  joint_matrix_verify<syclex::fp8_e4m3, syclex::fp8_e4m3, bfloat16, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 1>(q);
  joint_matrix_verify<syclex::fp8_e5m2, syclex::fp8_e4m3, bfloat16, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 4>(q);
  joint_matrix_verify<syclex::fp8_e5m2, syclex::fp8_e4m3, bfloat16, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 1>(q);
  joint_matrix_verify<syclex::fp8_e4m3, syclex::fp8_e5m2, bfloat16, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 4>(q);
  joint_matrix_verify<syclex::fp8_e4m3, syclex::fp8_e5m2, bfloat16, MATRIX_M,
                      MATRIX_N, MATRIX_K, TM, 1>(q);
#endif
}

int main() {
  sycl::queue q;
  if (!is_type_supported_by_device(q, matrix_type::fp8_e5m2)) {
    std::cout << "bf8 and hf8 types not supported on this device" << std::endl;
    return 0;
  }
  bf8_hf8_combinations<1 /*TM*/>(q);
  bf8_hf8_combinations<2 /*TM*/>(q);
  bf8_hf8_combinations<3 /*TM*/>(q);
  bf8_hf8_combinations<4 /*TM*/>(q);
  bf8_hf8_combinations<5 /*TM*/>(q);
  bf8_hf8_combinations<6 /*TM*/>(q);
  bf8_hf8_combinations<7 /*TM*/>(q);
  bf8_hf8_combinations<8 /*TM*/>(q);
  std::cout << "Passed\n";
  return 0;
}
