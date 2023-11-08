//==----------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

// TODO: add this test to XMX8 and SG32 folders
#include "common.hpp"
#define SG_SZ 16

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

template <typename T1, typename T2, size_t TM, size_t TN, size_t TK>
void matrix_fill_store(big_matrix<T1, TM, TN> &C, big_matrix<T2, TM, TK> &A,
                       big_matrix<T2, TK / 2, TN * 2> &B) {
  buffer<bfloat16, 2> bufA(A.get_data(), range<2>(TM, TK));
  buffer<bfloat16, 2> bufB(B.get_data(), range<2>(TK / 2, TN * 2));
  buffer<float, 2> bufC((float *)C.get_data(), range<2>(TM, TN));

  queue q;
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for(
         nd_range<2>({1, 1 * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major>
               sub_a;

           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, bfloat16, use::b, TK, TN,
                        layout::ext_intel_packed>
               sub_b;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;

           // TODO: uncomment these calls to add testing for other types of
           // matrices
           // joint_matrix_fill(sg, sub_a, 5.0);
           // joint_matrix_fill(sg, sub_b, 5.0);
           joint_matrix_fill(sg, sub_c, 5.0);

           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a, accA.template get_multi_ptr<access::decorated::no>(),
               TK);

           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_b, accB.template get_multi_ptr<access::decorated::no>(),
               TN * 2);

           joint_matrix_store(
               sg, sub_c, accC.template get_multi_ptr<access::decorated::no>(),
               TN, layout::row_major);
         }); // parallel for
   }).wait();
}

template <size_t TM, size_t TN, size_t TK> bool run_test() {

  bfloat16 A[TM][TK];
  bfloat16 A_ref[TM][TK];
  bfloat16 B[TK / 2][TN * 2];
  bfloat16 B_ref[TK / 2][TN * 2];
  float C[TM][TN];
  float C_ref[TM][TN];

  matrix_fill(TM, TK, (bfloat16 *)A, (bfloat16)0);
  matrix_fill(TK / 2, TN * 2, (bfloat16 *)B, (bfloat16)0);
  matrix_fill(TM, TN, (float *)C, 0.0f);

  matrix_fill(TM, TK, (bfloat16 *)A_ref, (bfloat16)5);
  matrix_fill(TK / 2, TN * 2, (bfloat16 *)B_ref, (bfloat16)5);
  matrix_fill(TM, TN, (float *)C_ref, 5.0f);

  big_matrix<float, TM, TN> MC((float *)&C);
  big_matrix<bfloat16, TM, TK> MA((bfloat16 *)&A);
  big_matrix<bfloat16, TK / 2, TN * 2> MB((bfloat16 *)&B);

  matrix_fill_store(MC, MA, MB);

  // TODO: uncomment these calls to verify other types of matrices
  // bool res = matrix_compare(TM, TK, (bfloat16 *)A, (bfloat16 *)A_ref);
  // res &= matrix_compare(TK / 2, TN * 2, (bfloat16 *)B, (bfloat16 *)B_ref);
  bool res = matrix_compare(TM, TN, (float *)C, (float *)C_ref);

  return res;
}

int main() {
  // TODO: add all supported size and types combinations
  bool res = run_test<8, 16, 16>();
  res &= run_test<32, 64, 16>();
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
