//==----------- element_wise_abc_impl.hpp  - DPC++ joint_matrix-------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

using namespace sycl::ext::oneapi::experimental::matrix;

#define TM 8
#define TK 32

class imatrix;

template <typename T1, typename T2, size_t M, size_t N, size_t K,
          int vnniFactor>
void matrix_elem_wise_ops(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                          big_matrix<T2, K / vnniFactor, N * vnniFactor> &B) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<T2, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<T2, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<T1, 2> bufC(C.get_data(), range<2>(M, N));

  queue q;
  size_t wg_size = get_wg_size<imatrix>(q);

  q.submit([&](handler &cgh) {
     accessor accC{bufC, cgh};
     accessor accA{bufA, cgh};
     accessor accB{bufB, cgh};

     cgh.parallel_for<class imatrix>(
         nd_range<2>({NDRangeM, NDRangeN * wg_size}, {1, 1 * wg_size}),
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
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * K,
               K);
           joint_matrix_apply(sg, sub_a, [](T2 &x) { x += 1; });

           joint_matrix_load(
               sg, sub_b,
               accB.template get_multi_ptr<access::decorated::no>() +
                   sg_starty / wg_size * TN * vnniFactor,
               N * vnniFactor);
           joint_matrix_apply(sg, sub_b, [](T2 &x) { x += 1; });

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / wg_size * TN,
               N, layout::row_major);
           joint_matrix_apply(sg, sub_c, [](T1 &x) { x += 1; });
         }); // parallel for
   }).wait();
}

int main() {
  static constexpr unsigned vnniFactor = 4;

  int8_t A[TM][TK];
  int8_t B[TK / vnniFactor][TN * vnniFactor];
  int32_t C[TM][TN];

  big_matrix<int32_t, TM, TN> MC((int32_t *)&C);
  big_matrix<int8_t, TM, TK> MA((int8_t *)&A);
  big_matrix<int8_t, TK / vnniFactor, TN * vnniFactor> MB((int8_t *)&B);

  matrix_elem_wise_ops<int32_t, int8_t, TM, TN, TK, vnniFactor>(MC, MA, MB);

  return 0;
}
