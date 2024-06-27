//==----------------------------------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------===//

template <typename T, size_t TM, size_t TN, size_t TK> class mult;

static constexpr int JM_ARRAY_SZ = 2;

template <typename TResult, typename T, size_t M, size_t N, size_t K, size_t TM,
          size_t TN, size_t TK, size_t VNNI>
void matrix_multiply(big_matrix<TResult, M, N> &C, big_matrix<T, M, K> &A,
                     big_matrix<T, K / VNNI, N * VNNI> &B) {
  size_t NDRangeM = M / (TM * JM_ARRAY_SZ);
  size_t NDRangeN = N / TN;
  buffer<T, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<T, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<TResult, 2> bufC((TResult *)C.get_data(), range<2>(M, N));

  queue q;
  size_t sg_size = get_sg_size<mult<T, TM, TN, TK>>(q);
  q.submit([&](handler &cgh) {
     sycl::accessor accA{bufA, cgh, sycl::read_write};
     sycl::accessor accB{bufB, cgh, sycl::read_write};
     sycl::accessor accC{bufC, cgh, sycl::read_write};

     cgh.parallel_for<mult<T, TM, TN, TK>>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           // Matrix API has to be accessed by all the workitems in a
           // subgroup. These functions will be called once by the subgroup.
           // No code divergence between the workitems.
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, T, use::a, TM, TK, layout::row_major>
               sub_a[JM_ARRAY_SZ];

           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, T, use::b, TK, TN, layout::ext_intel_packed>
               sub_b;
           joint_matrix<sub_group, TResult, use::accumulator, TM, TN>
               sub_c[JM_ARRAY_SZ];

           for (int i = 0; i < JM_ARRAY_SZ; ++i)
             joint_matrix_fill(sg, sub_c[i], TResult(1));

           for (int k = 0; k < K / TK; ++k) {
             joint_matrix_load(
                 sg, sub_b,
                 accB.template get_multi_ptr<access::decorated::no>() +
                     (k * TK / VNNI) * (N * VNNI) +
                     sg_starty / sg_size * TN * VNNI,
                 N * VNNI);

             for (int i = 0; i < JM_ARRAY_SZ; ++i) {
               joint_matrix_load(
                   sg, sub_a[i],
                   accA.template get_multi_ptr<access::decorated::no>() +
                       (sg_startx * TM * JM_ARRAY_SZ + TM * i) * K + k * TK,
                   K);
               joint_matrix_mad(sg, sub_c[i], sub_a[i], sub_b, sub_c[i]);
             }
           }

           for (int i = 0; i < JM_ARRAY_SZ; ++i)
             joint_matrix_store(
                 sg, sub_c[i],
                 accC.template get_multi_ptr<access::decorated::no>() +
                     (sg_startx * TM * JM_ARRAY_SZ + TM * i) * N +
                     sg_starty / sg_size * TN,
                 N, layout::row_major);
         }); // parallel for
   }).wait();
}

template <typename T, typename TResult, size_t VNNI, size_t TM, size_t TN,
          size_t TK>
void test() {
  std::cout << "Testing: " << TM << " x " << TN << " x " << TK
            << " [TM x TN x TK]" << std::endl;
  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;

  T A[MATRIX_M][MATRIX_K];
  T B[MATRIX_K / VNNI][MATRIX_N * VNNI];

  TResult C[MATRIX_M][MATRIX_N];
  TResult D[MATRIX_M][MATRIX_N];

  matrix_fill(MATRIX_M, MATRIX_K, (T *)A,
              [](int i, int j) { return TResult(1) * (i + j); });
  matrix_fill(MATRIX_K / VNNI, MATRIX_N * VNNI, (T *)B,
              [](int i, int j) { return TResult(2) * i + TResult(3) * j; });
  matrix_fill(MATRIX_M, MATRIX_N, (TResult *)C, TResult(1));
  matrix_fill(MATRIX_M, MATRIX_N, (TResult *)D, TResult(1));

  big_matrix<TResult, MATRIX_M, MATRIX_N> MC((TResult *)&C);
  big_matrix<TResult, MATRIX_M, MATRIX_N> MD((TResult *)&D);
  big_matrix<T, MATRIX_M, MATRIX_K> MA((T *)&A);
  big_matrix<T, MATRIX_K / VNNI, MATRIX_N * VNNI> MB((T *)&B);

  matrix_multiply<TResult, T, MATRIX_M, MATRIX_N, MATRIX_K, TM, TN, TK, VNNI>(
      MC, MA, MB);
  matrix_multiply_ref<T, T, TResult, VNNI>((T *)A, (T *)B, (TResult *)D,
                                           MATRIX_M, MATRIX_N, MATRIX_K / VNNI);

  assert(matrix_compare(MATRIX_M, MATRIX_N, (TResult *)C, (TResult *)D));
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      test<bfloat16, float, 2, /*TM*/ 16, /*TN*/ 16, /*TK*/ 32>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<bfloat16, float, 2, /*TM*/ 8, /*TN*/ 16, /*TK*/ 16>();
#if (!defined(SG_SZ) || SG_SZ != 32)
      // These combination are not currently supported for subgroup size = 32 in
      // IGC
      test<bfloat16, float, 2, /*TM*/ 16, /*TN*/ 16, /*TK*/ 16>();
      test<bfloat16, float, 2, /*TM*/ 1, /*TN*/ 64, /*TK*/ 16>();
      test<bfloat16, float, 2, /*TM*/ 32, /*TN*/ 64, /*TK*/ 16>();
      break;
#endif
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<bfloat16, float, 2, /*TM*/ 8, /*TN*/ 8, /*TK*/ 16>();
      break;
    }
  }
  return 0;
}
