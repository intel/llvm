//===---joint_matrix_all_sizes_impl.hpp - DPC++ joint_matrix---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

static constexpr size_t M_MULTIPLIER = 16;

template <typename T, size_t TM, size_t TN, size_t TK> class mult;

template <typename T1, typename T2, size_t M, size_t N, size_t K,
          int vnniFactor, size_t TM, size_t TN, size_t TK, typename kernel_name>
void matrix_multiply(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                     big_matrix<T2, K / vnniFactor, N * vnniFactor> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<T2, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<T2, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<T1, 2> bufC(C.get_data(), range<2>(M, N));

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
           joint_matrix<sub_group, T2, use::a, TM, TK, layout::row_major> sub_a;
           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, T2, use::b, TK, TN, layout::ext_intel_packed>
               sub_b;
           joint_matrix<sub_group, T1, use::accumulator, TM, TN> sub_c;

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
                     (k * TK / vnniFactor) * (N * vnniFactor) +
                     sg_starty / sg_size * TN * vnniFactor,
                 N * vnniFactor);
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

template <typename T, typename TResult, int vnni_factor, size_t tM, size_t tN,
          size_t tK, typename kernel_name>
void init_and_multiply() {
  static constexpr size_t MATRIX_M = tM * M_MULTIPLIER;
  static constexpr size_t MATRIX_N = 128;
  static constexpr size_t MATRIX_K = 128;

  std::cout << "MATRIX_M=" << MATRIX_M << "\n";

  T A[MATRIX_M][MATRIX_K];
  T B[MATRIX_K][MATRIX_N];
  T Bvnni[MATRIX_K / vnni_factor][MATRIX_N * vnni_factor];
  TResult C[MATRIX_M][MATRIX_N];
  TResult D[MATRIX_M][MATRIX_N];

  matrix_rand(MATRIX_M, MATRIX_K, (T *)A, (T)50);
  matrix_rand(MATRIX_K, MATRIX_N, (T *)B, (T)50);
  matrix_fill(MATRIX_M, MATRIX_N, (TResult *)C, (TResult)1);
  matrix_fill(MATRIX_M, MATRIX_N, (TResult *)D, (TResult)1);

  big_matrix<TResult, MATRIX_M, MATRIX_N> MC((TResult *)&C);
  big_matrix<TResult, MATRIX_M, MATRIX_N> MD((TResult *)&D);
  big_matrix<T, MATRIX_M, MATRIX_K> MA((T *)&A);
  matrix_vnni<T>(MATRIX_K, MATRIX_N, (T *)&B, (T *)&Bvnni, vnni_factor);
  big_matrix<T, MATRIX_K / vnni_factor, MATRIX_N * vnni_factor> MBvnni(
      (T *)&Bvnni);

  matrix_multiply<TResult, T, MATRIX_M, MATRIX_N, MATRIX_K, vnni_factor, tM, tN,
                  tK, kernel_name>(MC, MA, MBvnni);
  matrix_multiply_ref((T *)A, (T *)B, (TResult *)D, MATRIX_M, MATRIX_N,
                      MATRIX_K);

  assert(matrix_compare(MATRIX_M, MATRIX_N, (TResult *)C, (TResult *)D));
}

template <typename T, typename TResult, size_t VNNI, size_t TN, size_t TK>
void test() {
  init_and_multiply<T, TResult, VNNI, 1, TN, TK, mult<T, 1, TN, TK>>();
  init_and_multiply<T, TResult, VNNI, 2, TN, TK, mult<T, 2, TN, TK>>();
  init_and_multiply<T, TResult, VNNI, 3, TN, TK, mult<T, 3, TN, TK>>();
  init_and_multiply<T, TResult, VNNI, 4, TN, TK, mult<T, 4, TN, TK>>();
  init_and_multiply<T, TResult, VNNI, 5, TN, TK, mult<T, 5, TN, TK>>();
  init_and_multiply<T, TResult, VNNI, 6, TN, TK, mult<T, 6, TN, TK>>();
  init_and_multiply<T, TResult, VNNI, 7, TN, TK, mult<T, 7, TN, TK>>();
  init_and_multiply<T, TResult, VNNI, 8, TN, TK, mult<T, 8, TN, TK>>();
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      test<bfloat16, float, 2, /*TN*/ 16, /*TK*/ 32>();
      test<int8_t, int32_t, 4, /*TN*/ 16, /*TK*/ 64>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<bfloat16, float, 2, /*TN*/ 16, /*TK*/ 16>();
      test<int8_t, int32_t, 4, /*TN*/ 16, /*TK*/ 32>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<bfloat16, float, 2, /*TN*/ 8, /*TK*/ 16>();
      test<int8_t, int32_t, 4, /*TN*/ 8, /*TK*/ 32>();
      break;
    }
  }

  return 0;
}
