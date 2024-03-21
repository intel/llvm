//==-----joint_matrix_rowmajorA_rowmajorB_impl.hpp - DPC++ joint_matrix----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

template <size_t TM, size_t TN, size_t TK, class kernel_name, typename TA,
          typename TB, typename TC, size_t M, size_t N, size_t K>
void matrix_multiply(big_matrix<TC, M, N> &C, big_matrix<TA, M, K> &A,
                     big_matrix<TB, K, N> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<TA, 2> bufA((TA *)A.get_data(), range<2>(M, K));
  buffer<TB, 2> bufB((TB *)B.get_data(), range<2>(K, N));
  buffer<TC, 2> bufC((TC *)C.get_data(), range<2>(M, N));

  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  q.submit([&](handler &cgh) {
     sycl::accessor accC{bufC, cgh, sycl::read_write};
     sycl::accessor accA{bufA, cgh, sycl::read_only};
     sycl::accessor accB{bufB, cgh, sycl::read_only};

     cgh.parallel_for<kernel_name>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[intel::reqd_sub_group_size(SG_SZ)]]
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
           joint_matrix<sub_group, TA, use::a, TM, TK, layout::row_major> sub_a;
           joint_matrix<sub_group, TB, use::b, TK, TN, layout::row_major> sub_b;
           joint_matrix<sub_group, TC, use::accumulator, TM, TN> sub_c;

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
                     (k * TK) * (N) + sg_starty / sg_size * TN,
                 N);
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

template <size_t TN, size_t TK, class kernel_name, typename TA, typename TB,
          typename TC>
int gemm_row_major() {
  static constexpr size_t TM = 8;

  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  TA A[MATRIX_M][MATRIX_K];
  TB B[MATRIX_K][MATRIX_N];
  TC C[MATRIX_M][MATRIX_N];
  TC D[MATRIX_M][MATRIX_N];

  matrix_fill(MATRIX_M, MATRIX_K, (TA *)A,
              [](int i, int j) { return 1 * (i + j); });
  matrix_fill(MATRIX_K, MATRIX_N, (TB *)B,
              [](int i, int j) { return 2 * i + 3 * j; });
  matrix_fill(MATRIX_M, MATRIX_N, (TC *)C, (TC)1);
  matrix_fill(MATRIX_M, MATRIX_N, (TC *)D, (TC)1);

  big_matrix<TC, MATRIX_M, MATRIX_N> MC((TC *)&C);
  big_matrix<TC, MATRIX_M, MATRIX_N> MD((TC *)&D);
  big_matrix<TA, MATRIX_M, MATRIX_K> MA((TA *)&A);
  big_matrix<TB, MATRIX_K, MATRIX_N> MB((TB *)&B);
  matrix_multiply<TM, TN, TK, kernel_name>(MC, MA, MB);
  matrix_multiply_ref((TA *)A, (TB *)B, (TC *)D, MATRIX_M, MATRIX_N, MATRIX_K);

  bool res = matrix_compare(MATRIX_M, MATRIX_N, (TC *)C, (TC *)D);
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();
  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].atype == matrix_type::bf16) {
      if (combinations[i].nsize == 0 ||
          (combinations[i].nsize == 16 && combinations[i].max_msize == 8 &&
           combinations[i].ksize == 16)) {
        gemm_row_major<16, 16, class gemm_bfloat16_16, bfloat16, bfloat16,
                       float>();
      }
      if (combinations[i].nsize == 8 && combinations[i].max_msize == 8 &&
          combinations[i].ksize == 16) {
        gemm_row_major<8, 16, class gemm_bfloat16_8, bfloat16, bfloat16,
                       float>();
      }
    }
    if (combinations[i].atype == matrix_type::sint8 &&
        combinations[i].btype == matrix_type::sint8) {
      if (combinations[i].nsize == 0 ||
          (combinations[i].nsize == 16 && combinations[i].max_msize == 8 &&
           combinations[i].ksize == 32)) {
        gemm_row_major<16, 32, class gemm_int8_16, int8_t, int8_t, int32_t>();
        gemm_row_major<16, 32, class gemm_us_int8_16, uint8_t, int8_t,
                       int32_t>();
        gemm_row_major<16, 32, class gemm_su_int8_16, int8_t, uint8_t,
                       int32_t>();
        gemm_row_major<16, 32, class gemm_uu_int8_16, uint8_t, uint8_t,
                       int32_t>();
      }
      if (combinations[i].nsize == 8 && combinations[i].max_msize == 8 &&
          combinations[i].ksize == 32) {
        gemm_row_major<8, 32, class gemm_int8_8, int8_t, int8_t, int32_t>();
        gemm_row_major<8, 32, class gemm_us_int8_8, uint8_t, int8_t, int32_t>();
        gemm_row_major<8, 32, class gemm_su_int8_8, int8_t, uint8_t, int32_t>();
        gemm_row_major<8, 32, class gemm_uu_int8_8, uint8_t, uint8_t,
                       int32_t>();
      }
    }
  }
  return 0;
}
