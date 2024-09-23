//==----------- element_wise_abc_impl.hpp  - DPC++ joint_matrix-------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

template <size_t M, size_t N, size_t K, int vnniFactor> class add;

template <typename T1, typename T2, size_t M, size_t N, size_t K,
          int vnniFactor>
void matrix_elem_wise_ops(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A,
                          big_matrix<T2, K / vnniFactor, N * vnniFactor> &B) {
  size_t NDRangeM = 1;
  size_t NDRangeN = 1;
  buffer<T2, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<T2, 2> bufB(B.get_data(), range<2>(K, N));
  buffer<T1, 2> bufC(C.get_data(), range<2>(M, N));

  queue q;
  size_t sg_size = get_sg_size<add<M, N, K, vnniFactor>>(q);
  q.submit([&](handler &cgh) {
     accessor accC{bufC, cgh};
     accessor accA{bufA, cgh};
     accessor accB{bufB, cgh};

     cgh.parallel_for<add<M, N, K, vnniFactor>>(
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
           joint_matrix<sub_group, T2, use::a, M, K, layout::row_major> sub_a;
           // For B, we assume B has been already VNNIed.
           joint_matrix<sub_group, T2, use::b, K, N, layout::ext_intel_packed>
               sub_b;
           joint_matrix<sub_group, T1, use::accumulator, M, N> sub_c;

           joint_matrix_load(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * M) * K,
               K);
           joint_matrix_apply(sg, sub_a, [](T2 &x) { x += 1; });

           joint_matrix_load(
               sg, sub_b,
               accB.template get_multi_ptr<access::decorated::no>() +
                   sg_starty / sg_size * N * vnniFactor,
               N * vnniFactor);
           joint_matrix_apply(sg, sub_b, [](T2 &x) { x += 1; });

           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * M) * N + sg_starty / sg_size * N,
               N, layout::row_major);
           joint_matrix_apply(sg, sub_c, [](T1 &x) { x += 1; });
         }); // parallel for
   }).wait();
}

template <typename Ta, typename Tc, size_t TM, size_t TN, size_t TK, size_t VF>
void test() {
  Tc A[TM][TK];
  Tc B[TK / VF][TN * VF];
  Ta C[TM][TN];

  big_matrix<Ta, TM, TN> MC((Ta *)&C);
  big_matrix<Tc, TM, TK> MA((Tc *)&A);
  big_matrix<Tc, TK / VF, TN * VF> MB((Tc *)&B);

  return matrix_elem_wise_ops<Ta, int8_t, TM, TN, TK, VF>(MC, MA, MB);
}

int main() {
  queue q;
  std::vector<combination> combinations =
      q.get_device()
          .get_info<sycl::ext::oneapi::experimental::info::device::
                        matrix_combinations>();

  for (unsigned int i = 0; i < combinations.size(); i++) {
    if (combinations[i].nsize == 0) { // Intel AMX
      test<int32_t, int8_t, 16, 16, 64, 4>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      test<int32_t, int8_t, 8, 16, 32, 4>();
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      test<int32_t, int8_t, 8, 8, 32, 4>();
      break;
    }
  }

  return 0;
}
