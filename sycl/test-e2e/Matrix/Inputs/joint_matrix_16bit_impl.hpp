//===---joint_matrix_16bit_impl.hpp - DPC++ joint_matrix----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

template <typename Tab, typename TAcc, typename TResult, size_t TM, size_t TN,
          size_t TK, layout B_layout>
class imatrix;

template <typename Tab, typename TAcc, typename TResult, size_t M, size_t N,
          size_t K, size_t TM, size_t TN, size_t TK, layout B_layout, size_t VF>
void matrix_multiply(big_matrix<TResult, M, N> &D, big_matrix<TAcc, M, N> &C,
                     big_matrix<Tab, M, K> &A,
                     big_matrix<Tab, K / VF, N * VF> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<Tab, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<Tab, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<TAcc, 2> bufC((TAcc *)C.get_data(), range<2>(M, N));
  buffer<TResult, 2> bufD((TResult *)D.get_data(), range<2>(M, N));
  queue q;
  size_t sg_size =
      get_sg_size<imatrix<Tab, TAcc, TResult, TM, TN, TK, B_layout>>(q);

  q.submit([&](handler &cgh) {
     accessor accA{bufA, cgh};
     accessor accB{bufB, cgh};
     accessor accC{bufC, cgh};
     accessor accD{bufD, cgh};

     cgh.parallel_for<imatrix<Tab, TAcc, TResult, TM, TN, TK, B_layout>>(
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
           joint_matrix<sub_group, Tab, use::a, TM, TK, layout::row_major>
               sub_a;
           joint_matrix<sub_group, Tab, use::b, TK, TN, B_layout> sub_b;
           joint_matrix<sub_group, TAcc, use::accumulator, TM, TN> sub_c;
           joint_matrix<sub_group, TResult, use::accumulator, TM, TN> sub_d;

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);

           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(
                 sg, sub_a,
                 accA.template get_multi_ptr<access::decorated::no>() +
                     (sg_startx * TM) * K + k * TK,
                 K);
             joint_matrix_load(
                 sg, sub_b,
                 accB.template get_multi_ptr<access::decorated::no>() +
                     (k * TK / VF) * (N * VF) + sg_starty / sg_size * TN * VF,
                 N * VF);

             joint_matrix_mad(sg, sub_d, sub_a, sub_b, sub_c);
             joint_matrix_copy(sg, sub_d, sub_c);
           }

           joint_matrix_store(
               sg, sub_d,
               accD.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
}

template <typename Tab, typename TAcc, typename TResult, size_t TM, size_t TN,
          size_t TK, layout B_layout, size_t VF>
void test() {
  std::cout << "Testing: " << TM << " x " << TN << " x " << TK
            << " [TM x TN x TK]" << std::endl;

  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  Tab A[MATRIX_M][MATRIX_K];
  Tab B[MATRIX_K / VF][MATRIX_N * VF];
  TAcc C[MATRIX_M][MATRIX_N];
  TResult D[MATRIX_M][MATRIX_N];
  TResult DRef[MATRIX_M][MATRIX_N];

  matrix_rand<Tab>(MATRIX_M, MATRIX_K, (Tab *)A, Tab(1));
  matrix_rand<Tab>(MATRIX_K / VF, MATRIX_N * VF, (Tab *)B, Tab(1));

  matrix_fill(MATRIX_M, MATRIX_N, (TAcc *)C, TAcc(1));
  matrix_fill(MATRIX_M, MATRIX_N, (TResult *)D, TResult(1));
  matrix_fill(MATRIX_M, MATRIX_N, (TResult *)DRef, TResult(1));

  big_matrix<TAcc, MATRIX_M, MATRIX_N> MC((TAcc *)&C);
  big_matrix<TResult, MATRIX_M, MATRIX_N> MD((TResult *)&D);
  big_matrix<Tab, MATRIX_M, MATRIX_K> MA((Tab *)&A);
  big_matrix<Tab, MATRIX_K / VF, MATRIX_N * VF> MB((Tab *)&B);

  matrix_multiply<Tab, TAcc, TResult, MATRIX_M, MATRIX_N, MATRIX_K, TM, TN, TK,
                  B_layout, VF>(MD, MC, MA, MB);
  matrix_multiply_ref<Tab, Tab, TResult, VF>(
      (Tab *)A, (Tab *)B, (TResult *)DRef, MATRIX_M, MATRIX_N, MATRIX_K / VF);
  assert(matrix_compare(MATRIX_M, MATRIX_N, (TResult *)D, (TResult *)DRef));
}

template <typename TLow, typename THigh, size_t TM, size_t TN, size_t TK,
          layout B_layout, size_t VF>
void test_combo() {
  test<TLow, TLow, THigh, TM, TN, TK, B_layout, VF>();
  test<TLow, THigh, TLow, TM, TN, TK, B_layout, VF>();
  test<TLow, TLow, TLow, TM, TN, TK, B_layout, VF>();
  test<TLow, THigh, THigh, TM, TN, TK, B_layout, VF>();
}

template <typename TLow, typename THigh, layout B_layout, size_t VF>
void test_all() {
  test_combo<TLow, THigh, /*TM*/ 8, /*TN*/ 16, /*TK*/ 16, B_layout, VF>();
  test_combo<TLow, THigh, /*TM*/ 16, /*TN*/ 16, /*TK*/ 16, B_layout, VF>();
  test_combo<TLow, THigh, /*TM*/ 1, /*TN*/ 64, /*TK*/ 16, B_layout, VF>();
  test_combo<TLow, THigh, /*TM*/ 1, /*TN*/ 64, /*TK*/ 32, B_layout, VF>();
  test_combo<TLow, THigh, /*TM*/ 32, /*TN*/ 64, /*TK*/ 16, B_layout, VF>();
  test_combo<TLow, THigh, /*TM*/ 32, /*TN*/ 64, /*TK*/ 32, B_layout, VF>();
}
