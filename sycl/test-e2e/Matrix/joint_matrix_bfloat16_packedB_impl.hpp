//=----- joint_matrix_bfloat16_packedB_impl.hpp - DPC++ joint_matrix -------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//=-------------------------------------------------------------------------=//

template <size_t TM, size_t TN, size_t TK, class kernel_name, typename T1,
          typename T2, size_t M, size_t N, size_t K>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K / 2, N * 2> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<bfloat16, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<bfloat16, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<float, 2> bufC((float *)C.get_data(), range<2>(M, N));

  queue q;
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<kernel_name>(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [=](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
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

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) { //
             joint_matrix_load(
                 sg, sub_a,
                 accA.template get_multi_ptr<access::decorated::no>() +
                     (sg_startx * TM) * K + k * TK,
                 K);
             // Assuming B data is already in VNNI format.
             joint_matrix_load(
                 sg, sub_b,
                 accB.template get_multi_ptr<access::decorated::no>() +
                     (k * TK / 2) * (N * 2) + sg_starty / SG_SZ * TN * 2,
                 N * 2);
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / SG_SZ * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
}

template <size_t TM, size_t TN, size_t TK, class kernel_name> int test() {
  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  bfloat16 A[MATRIX_M][MATRIX_K];
  bfloat16 B[MATRIX_K / 2][MATRIX_N * 2];
  float C[MATRIX_M][MATRIX_N];
  float D[MATRIX_M][MATRIX_N];

  matrix_fill(MATRIX_M, MATRIX_K, (bfloat16 *)A,
              [](int i, int j) { return 1.0f * (i + j); });
  matrix_fill(MATRIX_K / 2, MATRIX_N * 2, (bfloat16 *)B,
              [](int i, int j) { return 2.0f * i + 3.0f * j; });
  matrix_fill(MATRIX_M, MATRIX_N, (float *)C, 1.0f);
  matrix_fill(MATRIX_M, MATRIX_N, (float *)D, 1.0f);

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<bfloat16, MATRIX_M, MATRIX_K> MA((bfloat16 *)&A);
  big_matrix<bfloat16, MATRIX_K / 2, MATRIX_N * 2> MB((bfloat16 *)&B);
  matrix_multiply<TM, TN, TK, kernel_name>(MC, MA, MB);
  matrix_multiply_ref<bfloat16, bfloat16, float, 2>(
      (bfloat16 *)A, (bfloat16 *)B, (float *)D, MATRIX_M, MATRIX_N,
      MATRIX_K / 2);

  bool res = matrix_compare(MATRIX_M, MATRIX_N, (float *)C, (float *)D);
  std::cout << TM << "x" << TN << "x" << TK << " ";
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  int ret = 0;
  for (auto &combination : combinations) {
    if (combination.nsize == 0) { // Intel AMX
      ret += test<16, 16, 16, class amx16x16x16>();
      break;
    }

    if (combination.nsize == 16) { // architecture::intel_gpu_pvc
      ret += test<16, 16, 16, class pvc16x16x16>();
      ret += test<32, 64, 16, class pvc32x64x16>();
      ret += test<1, 64, 16, class pvc1x64x16>();
      break;
    }
  }

  return ret;
}
