//==----------- element_wise_abc_impl.hpp  - DPC++ joint_matrix-------------
//----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define TM 8
#define TN SG_SZ
#define TK 32

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T1, typename T2, size_t NUM_ROWS_A, size_t NUM_COLS_A,
          size_t NUM_ROWS_B, size_t NUM_COLS_B, size_t NUM_ROWS_C,
          size_t NUM_COLS_C>
void matrix_elem_wise_ops(big_matrix<T1, NUM_ROWS_C, NUM_COLS_C> &C,
                          big_matrix<T2, NUM_ROWS_A, NUM_COLS_A> &A,
                          big_matrix<T2, NUM_ROWS_B, NUM_COLS_B> &B) {
  size_t M = NUM_ROWS_C;
  size_t N = NUM_COLS_C;
  size_t K = NUM_COLS_A;

  // B => K/4 x N*4, A => M x K, C => M, N
  // stride should be X's cols, e.g., B's stirde = N*4
  assert(NUM_ROWS_C == NUM_ROWS_A && NUM_COLS_A == NUM_ROWS_B * 4);

  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<int8_t, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<int8_t, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<int32_t, 2> bufC(C.get_data(), range<2>(M, N));

  queue q;
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [accA, accB, accC, M, N,
          K](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, int8_t, use::a, TM, TK, layout::row_major>
               sub_a;

           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, int8_t, use::b, TK, TN,
                        ext::intel::experimental::matrix::layout::packed>
               sub_b;

           joint_matrix<sub_group, int32_t, use::accumulator, TM, TN> sub_c;

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);

           joint_matrix_load(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * K,
               K);
           auto wi_slice_a =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_a);
           for (int i = 0; i < wi_slice_a.length(); i++) {
             wi_slice_a[i] += 1;
           }

           joint_matrix_load(
               sg, sub_b,
               accB.template get_multi_ptr<access::decorated::no>() +
                   +sg_starty / SG_SZ * TN * 4,
               N * 4);
           auto wi_slice_b =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_b);
           for (int i = 0; i < wi_slice_b.length(); i++) {
             wi_slice_b[i] += 1;
           }

           auto wi_slice_c =
               sycl::ext::intel::experimental::matrix::get_wi_data(sg, sub_c);
           for (int i = 0; i < wi_slice_c.length(); i++) {
             wi_slice_c[i] += 1;
           }
         }); // parallel for
   }).wait();
}

int8_t A[TM][TK];
int8_t B[TK / 4][TN * 4];
int32_t C[TM][TN];

int main() {
  big_matrix<int32_t, TM, TN> MC((int32_t *)&C);
  big_matrix<int8_t, TM, TK> MA((int8_t *)&A);
  big_matrix<int8_t, TK / 4, TN * 4> MB((int8_t *)&B);
  matrix_elem_wise_ops(MC, MA, MB);

  return 0;
}
