//==----------- element_wise_ops_impl.hpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

template <size_t TM, size_t TN, size_t TK, size_t VF, class kernel_name,
          typename Tc, typename Ta, size_t M, size_t N, size_t K>
void matrix_multiply(big_matrix<Tc, M, N> &C, big_matrix<Ta, M, K> &A,
                     big_matrix<Ta, K / VF, N * VF> &B) {
  // stride should be X's cols, e.g., B's stirde = N*4
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<Ta, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<Ta, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<Tc, 2> bufC(C.get_data(), range<2>(M, N));

  queue q;
  size_t sg_size = get_sg_size<kernel_name>(q);
  q.submit([&](handler &cgh) {
     auto accC = bufC.template get_access<access::mode::read_write>(cgh);
     auto accA = bufA.template get_access<access::mode::read_write>(cgh);
     auto accB = bufB.template get_access<access::mode::read_write>(cgh);

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

           sycl::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, Ta, use::a, TM, TK, layout::row_major> sub_a;
           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, Ta, use::b, TK, TN, layout::ext_intel_packed>
               sub_b;
           joint_matrix<sub_group, Tc, use::accumulator, TM, TN> sub_c;

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
             joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
           }
           joint_matrix_apply(sg, sub_c, [](Tc &x) { x = x * 2; });
           joint_matrix_store(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
         }); // parallel for
   }).wait();
}

template <typename Ta, typename Tc, size_t TM, size_t TN, size_t TK, size_t VF,
          class kernel_name>
bool test() {

  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;

  Ta A[MATRIX_M][MATRIX_K];
  Ta B[MATRIX_K / VF][MATRIX_N * VF];
  Tc C[MATRIX_M][MATRIX_N];
  Tc D[MATRIX_M][MATRIX_N];

  matrix_rand(MATRIX_M, MATRIX_K, (Ta *)A, (Ta)100);
  matrix_rand(MATRIX_K / VF, MATRIX_N * VF, (Ta *)B, (Ta)100);
  matrix_fill(MATRIX_M, MATRIX_N, (Tc *)C, (Tc)1);
  matrix_fill(MATRIX_M, MATRIX_N, (Tc *)D, (Tc)1);

  big_matrix<Tc, MATRIX_M, MATRIX_N> MC((Tc *)&C);
  big_matrix<Tc, MATRIX_M, MATRIX_N> MD((Tc *)&D);
  big_matrix<Ta, MATRIX_M, MATRIX_K> MA((Ta *)&A);
  big_matrix<Ta, MATRIX_K / VF, MATRIX_N * VF> MB((Ta *)&B);

  matrix_multiply<TM, TN, TK, VF, kernel_name>(MC, MA, MB);
  matrix_multiply_ref<Ta, Ta, Tc, VF>((Ta *)A, (Ta *)B, (Tc *)D, MATRIX_M,
                                      MATRIX_N, MATRIX_K / VF, false, false,
                                      false, [](Tc &x) { x = x * 2; });
  bool res = matrix_compare(MATRIX_M, MATRIX_N, (Tc *)C, (Tc *)D);

  std::cout << TM << "x" << TN << "x" << TK << ": "
            << (res ? "passed" : "failed") << std::endl;
  return res;
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  bool passed = true;
  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      passed &= test<uint8_t, int32_t, 8, 16, 32, 4, class amx_uint_8x16x32>();
      passed &= test<int8_t, int32_t, 8, 16, 32, 4, class amx_sint_8x16x32>();
      passed &= test<bfloat16, float, 8, 16, 32, 2, class amx_bf16_8x16x32>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      passed &= test<uint8_t, int32_t, 8, 16, 32, 4, class pvc_uint_8x16x32>();
      passed &= test<int8_t, int32_t, 8, 16, 32, 4, class pvc_sint_8x16x32>();
      passed &= test<bfloat16, float, 8, 16, 16, 2, class pvc_bf16_8x16x16>();
#if (!defined(SG_SZ) || SG_SZ != 32)
      // These combination are not currently supported for subgroup size = 32 in
      // IGC
      passed &= test<bfloat16, float, 16, 16, 16, 2, class pvc_bf16_16x16x16>();
      passed &= test<bfloat16, float, 32, 64, 16, 2, class pvc_bf16_32x64x16>();
#endif
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      passed &= test<uint8_t, int32_t, 8, 8, 32, 4, class dg2_uint_8x8x32>();
      passed &= test<int8_t, int32_t, 8, 8, 32, 4, class dg2_sint_8x8x32>();
      passed &= test<bfloat16, float, 8, 8, 16, 2, class dg2_bf16_8x16x16>();
      break;
    }
  }

  return !passed;
}
