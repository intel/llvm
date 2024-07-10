//===---joint_matrix_colA_rowB_colC_impl.hpp - DPC++ joint_matrix----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <random>
#include <sycl/usm.hpp>

constexpr size_t TM = 8;
constexpr size_t TK = 16;

template <typename T1, typename T2, size_t NUM_ROWS_A, size_t NUM_COLS_A,
          size_t NUM_ROWS_B, size_t NUM_COLS_B, size_t NUM_ROWS_C,
          size_t NUM_COLS_C>
void matrix_multiply(T1 *C, T2 *A, T2 *B, queue q) {
  size_t M = NUM_ROWS_C;
  size_t N = NUM_COLS_C;
  size_t K = NUM_COLS_A;

  assert(NUM_ROWS_C == NUM_ROWS_A && NUM_COLS_A == NUM_ROWS_B);
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  size_t sg_size = get_sg_size<class mult>(q);

  q.submit([&](handler &cgh) {
     cgh.parallel_for<class mult>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           auto pA =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(A);
           auto pB =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(B);
           auto pC =
               address_space_cast<sycl::access::address_space::global_space,
                                  sycl::access::decorated::no>(C);

           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::col_major>
               sub_a;
           joint_matrix<sub_group, bfloat16, use::b, TK, TN, layout::row_major>
               sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;
           joint_matrix_fill(sg, sub_c, 1);
           for (int k = 0; k < K; k += TK) {
             joint_matrix_load(sg, sub_a, pA + (sg_startx * TM) * K + k, K);
             joint_matrix_load(sg, sub_b, pB + k * N + sg_starty / sg_size * TN,
                               N);
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(
               sg, sub_c, pC + (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::col_major);
         }); // parallel for
   }).wait();
}

int main() {
  static constexpr size_t MATRIX_M = 1024;
  static constexpr size_t MATRIX_N = 1024;
  static constexpr size_t MATRIX_K = 1024;
  queue q;
  bfloat16 *A = malloc_shared<bfloat16>(MATRIX_M * MATRIX_K, q);
  bfloat16 *B = malloc_shared<bfloat16>(MATRIX_K * MATRIX_N, q);
  float *C = malloc_shared<float>(MATRIX_M * MATRIX_N, q);
  float *D = malloc_shared<float>(MATRIX_M * MATRIX_N, q);

  matrix_rand(MATRIX_M, MATRIX_K, A, (bfloat16)5);
  matrix_rand(MATRIX_K, MATRIX_N, B, (bfloat16)5);
  matrix_fill(MATRIX_M, MATRIX_N, C, (float)1.0);
  matrix_fill(MATRIX_M, MATRIX_N, D, (float)1.0);

  matrix_multiply<float, bfloat16, MATRIX_M, MATRIX_K, MATRIX_K, MATRIX_N,
                  MATRIX_M, MATRIX_N>(C, A, B, q);
  matrix_multiply_ref(A, B, D, MATRIX_M, MATRIX_N, MATRIX_K,
                      true /*transposed c*/);

  bool res = matrix_compare(MATRIX_M, MATRIX_N, C, D);

  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
