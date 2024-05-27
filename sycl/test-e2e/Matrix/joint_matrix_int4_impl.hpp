//==----------- element_wise_all_ops_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

typedef int32_t storageT;
typedef precision::sint4 elementT; 

template <size_t TM, size_t TN, size_t TK, class kernel_name, typename T1, typename T2, size_t NUM_ROWS_A, size_t NUM_COLS_A,
          size_t NUM_ROWS_B, size_t NUM_COLS_B, size_t NUM_ROWS_C,
          size_t NUM_COLS_C>
void matrix_multiply(big_matrix<T1, NUM_ROWS_C, NUM_COLS_C> &C,
                     big_matrix<T2, NUM_ROWS_A, NUM_COLS_A> &A,
                     big_matrix<T2, NUM_ROWS_B, NUM_COLS_B> &B) {
  size_t M = NUM_ROWS_C;
  size_t N = NUM_COLS_C;
  size_t K = NUM_COLS_A;

  assert(NUM_ROWS_C == NUM_ROWS_A && NUM_COLS_A == NUM_ROWS_B);
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<storageT, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<storageT, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<storageT, 2> bufC((storageT *)C.get_data(), range<2>(M, N));

  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<kernel_name>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ	 
	 [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           // The matrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, elementT, use::a, TM, TK,
                        layout::row_major>
               sub_a;
           joint_matrix<sub_group, elementT, use::b, TK, TN,
                        layout::row_major>
               sub_b;
           joint_matrix<sub_group, storageT, use::accumulator, TM, TN> sub_c;
           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
           for (int k = 0; k < K; k += TK) {
             joint_matrix_load(
                 sg, sub_a,
                 accA.template get_multi_ptr<access::decorated::no>() +
                     (sg_startx * TM) * K + k,
                 K);
             joint_matrix_load(
                 sg, sub_b,
                 accB.template get_multi_ptr<access::decorated::no>() +
                     (k) * (N) + sg_starty / sg_size * TN,
                 N);
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
	   joint_matrix_apply(sg, sub_a,
			      [=](storageT &x) { x ++; });
             joint_matrix_apply(sg, sub_b,
                                [=](storageT &x) { x --; });
             joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();
  bool support_p = false;
  // joint_matrix_prefetch is not supported on DG2
  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].atype == matrix_type::sint4 || combinations[i].btype == matrix_type::sint4) {
      support_p = true;
      break;
    }
  }
  if (!support_p) {
    std::cout << "int4 data type is not supported on this device" << std::endl;
    // Once the test is not marked as XFAIL, this should change to return 0;
    return 1;
  }
  
  static constexpr size_t TM = 8;
  static constexpr size_t TN = 16;
  static constexpr size_t TK = 64;

  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  storageT A[MATRIX_M][MATRIX_K];
  storageT B[MATRIX_K][MATRIX_N];
  storageT C[MATRIX_M][MATRIX_N];
  storageT D[MATRIX_M][MATRIX_N];

  matrix_fill(MATRIX_M, MATRIX_K, (storageT *)A,
              [](int i, int j) { return 1 * (i + j); });
  matrix_fill(MATRIX_K, MATRIX_N, (storageT *)B,
              [](int i, int j) { return 2 * i + 3 * j; });
  matrix_fill(MATRIX_M, MATRIX_N, (storageT *)C, 1);
  matrix_fill(MATRIX_M, MATRIX_N, (storageT *)D, 1);

  big_matrix<storageT, MATRIX_M, MATRIX_N> MC((storageT *)&C);
  big_matrix<storageT, MATRIX_M, MATRIX_N> MD((storageT *)&D);
  big_matrix<storageT, MATRIX_M, MATRIX_K> MA((storageT *)&A);
  big_matrix<storageT, MATRIX_K, MATRIX_N> MB((storageT *)&B);
  matrix_multiply<TM, TN, TK, class gemm_int4>(MC, MA, MB);
  matrix_multiply_ref((storageT *)A, (storageT *)B, (storageT *)D, MATRIX_M, MATRIX_N,
                      MATRIX_K);

  bool res = matrix_compare(MATRIX_M, MATRIX_N, (storageT *)C, (storageT *)D);
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
