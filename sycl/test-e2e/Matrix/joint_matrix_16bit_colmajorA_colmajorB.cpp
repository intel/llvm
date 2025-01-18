//==-joint_matrix_16bit_colmajorA_colmajorB.cpp- DPC++ joint_matrix-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This tests support of col major layout for matrix B which does transpose and
// then VNNI transform. This is currently only available on AMX

// REQUIRES: aspect-ext_intel_matrix

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %{build} -o %t32.out -DSG_SZ=32
// RUN: %{run} %t32.out

// XFAIL: gpu
// XFAIL-TRACKER: GSD-5768

#include "common.hpp"

constexpr size_t TM = 8;
constexpr size_t TN = 16;
constexpr size_t TK = 16;

template <typename T> class imatrix;

template <typename T1, typename T2, size_t M, size_t N, size_t K>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K, N> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<T2, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<T2, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<float, 2> bufC((float *)C.get_data(), range<2>(M, N));

  queue q;
  size_t sg_size = get_sg_size<class imatrix<T2>>(q);
  std::cout << "subgroup size " << sg_size << " ";

  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.template get_access<access::mode::read_write>(cgh);
     auto accB = bufB.template get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class imatrix<T2>>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T2, use::a, TM, TK, layout::col_major> sub_a;
           joint_matrix<sub_group, T2, use::b, TK, TN, layout::col_major> sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(
                 sg, sub_a,
                 accA.template get_multi_ptr<access::decorated::no>() +
                     (k * TK) * M + sg_startx * TM,
                 M);
             joint_matrix_load(
                 sg, sub_b,
                 accB.template get_multi_ptr<access::decorated::no>() +
                     (sg_starty / sg_size * TN) * K + k * TK,
                 K);
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
}

template <typename T> void test() {
  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  T A[MATRIX_K][MATRIX_M];
  T B[MATRIX_N][MATRIX_K];
  float C[MATRIX_M][MATRIX_N];
  float D[MATRIX_M][MATRIX_N];

  matrix_fill(MATRIX_K, MATRIX_M, (T *)A,
              [](int i, int j) { return 1.0f * (i + j); });
  matrix_fill(MATRIX_N, MATRIX_K, (T *)B,
              [](int i, int j) { return 2.0f * i + 3.0f * j; });
  matrix_fill(MATRIX_M, MATRIX_N, (float *)C, 1.0f);
  matrix_fill(MATRIX_M, MATRIX_N, (float *)D, 1.0f);

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<T, MATRIX_M, MATRIX_K> MA((T *)&A);
  big_matrix<T, MATRIX_K, MATRIX_N> MB((T *)&B);
  matrix_multiply(MC, MA, MB);
  matrix_multiply_ref((T *)A, (T *)B, (float *)D, MATRIX_M, MATRIX_N, MATRIX_K,
                      false, true, true);

  assert(matrix_compare(MATRIX_M, MATRIX_N, (float *)C, (float *)D));
  std::cout << "passed" << std::endl;
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device().get_info<syclex::info::device::matrix_combinations>();
  bool bf16_run = false;
  bool half_run = false;

  for (auto &combination : combinations) {
    if (!bf16_run && combination.atype == matrix_type::bf16) {
      std::cout << "bf16 ";
      test<bfloat16>();
      bf16_run = true;
    }

    if (!half_run && combination.atype == matrix_type::fp16) {
      std::cout << "half ";
      test<half>();
      half_run = true;
    }
  }

  return 0;
}
