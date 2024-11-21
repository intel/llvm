//==-------- joint_matrix_down_convert_impl.hpp  - DPC++ joint_matrix-------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr size_t TM = 8;
// TN and TK must be the same for this test.
constexpr size_t TN = 16;
constexpr size_t TK = 16;

template <Activation act, size_t TM, size_t TK, typename Group, typename Tsrc,
          typename Tdest, use UseSrc, use UseDest, layout LayoutSrc,
          layout LayoutDest>
void applyActivation(
    Group &sg, joint_matrix<Group, Tsrc, UseSrc, TM, TK, LayoutSrc> &sub_c,
    joint_matrix<Group, Tdest, UseDest, TM, TN, LayoutDest> &sub_a) {
  if constexpr (act == Activation::None) {
    joint_matrix_copy(sg, sub_c, sub_a);
  } else if constexpr (act == Activation::ReLU) {

    joint_matrix_apply(
        sg, sub_c, [=](float &x) { x = sycl::max(static_cast<float>(0), x); });
    joint_matrix_copy(sg, sub_c, sub_a);

  } else if constexpr (act == Activation::Sigmoid) {
    joint_matrix_apply(sg, sub_c,
                       [=](float &x) { x = 1.0f / (1.0f + sycl::exp(-x)); });
    joint_matrix_copy(sg, sub_c, sub_a);
  }
  return;
}

template <Activation act> class copy;

template <Activation act, typename T1, typename T2, size_t M, size_t N,
          size_t K>
void matrix_activation_copy(big_matrix<T1, M, N> &C, big_matrix<T2, M, K> &A) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<bfloat16, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<float, 2> bufC((float *)C.get_data(), range<2>(M, N));

  queue q;
  size_t sg_size = get_sg_size<copy<act>>(q);
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::write>(cgh);

     cgh.parallel_for<copy<act>>(
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
           joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major>
               sub_a;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;
           joint_matrix_load(
               sg, sub_c,
               accC.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
           applyActivation<act>(sg, sub_c, sub_a);

           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a,
               accA.template get_multi_ptr<access::decorated::no>() +
                   (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N);
         }); // parallel for
   }).wait();
}

int main() {
  static constexpr size_t MATRIX_M = TM * 2;
  static constexpr size_t MATRIX_N = TN * 2;
  static constexpr size_t MATRIX_K = TK * 2;
  bfloat16 A[MATRIX_M][MATRIX_K];
  float C[MATRIX_M][MATRIX_N];

  matrix_rand(MATRIX_M, MATRIX_N, *C, (float)5);

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<bfloat16, MATRIX_M, MATRIX_K> MA((bfloat16 *)&A);

  matrix_activation_copy<Activation::None>(MC, MA);
  bool res0 = matrix_compare(MATRIX_M, MATRIX_N, (bfloat16 *)A, (float *)C);
  bool res = matrix_compare<Activation::None>(MATRIX_M, MATRIX_N, (bfloat16 *)A,
                                              (float *)C);
  std::cout << (res ? "Copy passed" : "Copy failed") << std::endl;

  matrix_activation_copy<Activation::ReLU>(MC, MA);
  res &= matrix_compare<Activation::ReLU>(MATRIX_M, MATRIX_N, (bfloat16 *)A,
                                          (float *)C);
  std::cout << (res ? "ReLU passed" : "ReLU failed") << std::endl;

  matrix_activation_copy<Activation::Sigmoid>(MC, MA);
  res &= matrix_compare<Activation::Sigmoid>(MATRIX_M, MATRIX_N, (bfloat16 *)A,
                                             (float *)C);
  std::cout << (res ? "Sigmoid passed" : "Sigmoid failed") << std::endl;

  return !res;
}
