//===---joint_matrix_out_bounds_impl.hpp - DPC++ joint_matrix--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/usm.hpp>

template <typename Tab, size_t TM, size_t TN, size_t TK, layout B_layout>
class mult;

template <typename T1, typename T2, size_t M, size_t N, size_t K, size_t TM,
          size_t TN, size_t TK, layout A_layout, layout B_layout>
void matrix_multiply(T1 *C, T2 *A, T2 *B, queue q) {

  // Add one iteration for the out of bounds dpas instruction
  size_t NDRangeM = M / TM + (((M % TM) != 0) ? 1 : 0);
  size_t NDRangeN = N / TN + (((N % TN) != 0) ? 1 : 0);
  size_t sg_size = get_sg_size<mult<T2, TM, TN, TK, B_layout>>(q);
  std::cout << "SG size: " << sg_size << " ";

  q.submit([&](handler &cgh) {
     cgh.parallel_for<mult<T2, TM, TN, TK, B_layout>>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
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
           joint_matrix<sub_group, T2, use::a, TM, TK, A_layout> sub_a;
           joint_matrix<sub_group, T2, use::b, TK, TN, B_layout> sub_b;
           joint_matrix<sub_group, T1, use::accumulator, TM, TN> sub_c;

           // bounds-checked fill where width and height are added
           ext::intel::experimental::matrix::joint_matrix_fill_checked(
               sg, sub_c, 1, M, N, sg_startx * TM, sg_starty / sg_size * TN);

           for (int k = 0; k < K; k += TK) {
             // bounds-checked load where width and height are added
             // params order: Stride, Height, Width, CoordX, CoordY
             if constexpr (A_layout == layout::row_major) {
               ext::intel::experimental::matrix::joint_matrix_load_checked(
                   sg, sub_a, pA, K, M, K, sg_startx * TM, k);
             } else {
               ext::intel::experimental::matrix::joint_matrix_load_checked(
                   sg, sub_a, pA, M, K, M, k, sg_startx * TM);
             }

             // bounds-checked load where width and height are added
             // params order: Stride, Height, Width, CoordX, CoordY
             if constexpr (B_layout != layout::col_major) {
               constexpr unsigned int vnniFactor = vnni_factor<T2, B_layout>();
               ext::intel::experimental::matrix::joint_matrix_load_checked(
                   sg, sub_b, pB, N * vnniFactor, K / vnniFactor,
                   N * vnniFactor, k / vnniFactor,
                   sg_starty / sg_size * TN * vnniFactor);
             } else {
               ext::intel::experimental::matrix::joint_matrix_load_checked(
                   sg, sub_b, pB, K, N, K, sg_starty / sg_size * TN, k);
             }

             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }

           // bounds-checked store where width and height are added
           ext::intel::experimental::matrix::joint_matrix_store_checked(
               sg, sub_c, pC, N, layout::row_major, M, N, sg_startx * TM,
               sg_starty / sg_size * TN);
         }); // parallel for
   }).wait();
}

template <typename Tab, typename Tc, size_t MATRIX_M, size_t MATRIX_N,
          size_t MATRIX_K, size_t TM, size_t TN, size_t TK, layout A_layout,
          layout B_layout>
void test() {
  std::cout << MATRIX_M << "x" << MATRIX_N << "x" << MATRIX_K << ", " << TM
            << "x" << TN << "x" << TK << ": ";
  queue q;

  // reference data
  Tab *A = malloc_shared<Tab>(MATRIX_M * MATRIX_K, q);
  Tab *B = malloc_shared<Tab>(MATRIX_K * MATRIX_N, q);
  Tc *C = malloc_shared<Tc>(MATRIX_M * MATRIX_N, q);
  Tc *D = malloc_shared<Tc>(MATRIX_M * MATRIX_N, q);
  matrix_rand(MATRIX_M, MATRIX_K, A, (Tab)5);
  matrix_rand(MATRIX_K, MATRIX_N, B, (Tab)5);
  matrix_fill(MATRIX_M, MATRIX_N, D, (Tc)1);
  matrix_multiply_ref(A, B, D, MATRIX_M, MATRIX_N, MATRIX_K);

  // test data
  if constexpr (A_layout == layout::col_major) {
    Tab *colA = malloc_shared<Tab>(MATRIX_K * MATRIX_M, q);
    matrix_transpose(MATRIX_M, MATRIX_K, colA, A);
    Tab *tmp = A;
    A = colA;
    free(tmp, q);
  }

  if constexpr (B_layout == layout::col_major) {
    Tab *colB = malloc_shared<Tab>(MATRIX_N * MATRIX_K, q);
    matrix_transpose(MATRIX_K, MATRIX_N, colB, B);
    Tab *tmp = B;
    B = colB;
    free(tmp, q);
  }

  if constexpr (B_layout == layout::ext_intel_packed) {
    Tab *vnniB = malloc_shared<Tab>(MATRIX_K * MATRIX_N, q);
    matrix_vnni(MATRIX_K, MATRIX_N, B, vnniB, vnni_factor<Tab, B_layout>());
    Tab *tmp = B;
    B = vnniB;
    free(tmp, q);
  }

  matrix_multiply<Tc, Tab, MATRIX_M, MATRIX_N, MATRIX_K, TM, TN, TK, A_layout,
                  B_layout>(C, A, B, q);
  assert(matrix_compare(MATRIX_M, MATRIX_N, C, D));
  std::cout << "passed" << std::endl;

  free(A, q);
  free(B, q);
  free(C, q);
  free(D, q);
}

template <layout A_layout, layout B_layout> void test_all() {
  std::cout << "bf16: ";
  test<bfloat16, float, /*MATRIX_M*/ 1024 + 24, /*MATRIX_N*/ 1024 + 24,
       /*MATRIX_K*/ 1024 + 24, /*TM*/ 8, /*TN*/ 16, /*TK*/ 16, A_layout,
       B_layout>();
  std::cout << "half: ";
  test<half, float, 1024 + 24, 1024 + 24, 1024 + 24, 8, 16, 16, A_layout,
       B_layout>();
  std::cout << "int8: ";
  test<int8_t, int32_t, 1024, 1024, 1024 + 16, 8, 16, 32, A_layout, B_layout>();
}

template <layout A_layout, layout B_layout> void test_all_big_shapes() {
  std::cout << "bf16: ";
  test<bfloat16, float, 1024 + 24, 1024 + 24, 1024 + 24, 16, 16, 16, A_layout,
       B_layout>();
  test<bfloat16, float, 1024 + 24, 1024 + 24, 1024 + 24, 1, 64, 16, A_layout,
       B_layout>();
  test<bfloat16, float, 1024 + 24, 1024 + 24, 1024 + 24, 1, 64, 32, A_layout,
       B_layout>();
  test<bfloat16, float, 1024 + 24, 1024 + 24, 1024 + 24, 32, 64, 16, A_layout,
       B_layout>();
  test<bfloat16, float, 1024 + 24, 1024 + 24, 1024 + 24, 32, 64, 32, A_layout,
       B_layout>();

  std::cout << "half: ";
  test<half, float, 1024 + 24, 1024 + 24, 1024 + 24, 16, 16, 16, A_layout,
       B_layout>();
  test<half, float, 1024 + 24, 1024 + 24, 1024 + 24, 1, 64, 16, A_layout,
       B_layout>();
  test<half, float, 1024 + 24, 1024 + 24, 1024 + 24, 1, 64, 32, A_layout,
       B_layout>();
  test<half, float, 1024 + 24, 1024 + 24, 1024 + 24, 32, 64, 16, A_layout,
       B_layout>();
  test<half, float, 1024 + 24, 1024 + 24, 1024 + 24, 32, 64, 32, A_layout,
       B_layout>();
}
