//===---joint_matrix_annotated_ptr_impl.hpp - DPC++ joint_matrix-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/usm.hpp>

#define TM 8
#define TK 16

template <unsigned int vnniFactor> class mult;

template <typename T1, typename T2, size_t M, size_t N, size_t K,
          unsigned int vnniFactor>
void matrix_multiply(T1 *C, T2 *A, T2 *B, queue &q) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  size_t sg_size = get_sg_size<mult<vnniFactor>>(q);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<mult<vnniFactor>>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[intel::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, bfloat16, use::a, TM, TK, layout::row_major>
               sub_a;
           joint_matrix<sub_group, bfloat16, use::b, TK, TN, layout::row_major>
               sub_b;
           joint_matrix<sub_group, bfloat16, use::b, TK, TN,
                        layout::ext_intel_packed>
               sub_bp;
           joint_matrix<sub_group, float, use::accumulator, TM, TN> sub_c;

           auto C_ptr = syclex::annotated_ptr{
               C, syclex::properties{
                      syclintelex::read_assertion<syclintelex::cache_control<
                          syclintelex::cache_mode::invalidate,
                          syclex::cache_level::L2>>}};
           auto A_ptr = syclex::annotated_ptr{
               A,
               syclex::properties{syclintelex::read_assertion<
                   syclintelex::cache_control<syclintelex::cache_mode::constant,
                                              syclex::cache_level::L2>>}};
           auto B_ptr = syclex::annotated_ptr{
               B,
               syclex::properties{syclintelex::read_hint<
                   syclintelex::cache_control<syclintelex::cache_mode::cached,
                                              syclex::cache_level::L2>>}};
           joint_matrix_load(sg, sub_c,
                             C_ptr + (sg_startx * TM) * N +
                                 sg_starty / sg_size * TN,
                             N, layout::row_major);
           for (int k = 0; k < K / TK; k += 1) {
             joint_matrix_load(sg, sub_a, A_ptr + (sg_startx * TM) * K + k * TK,
                               K);
             if constexpr (vnniFactor == 0) {
               joint_matrix_load(
                   sg, sub_b, B_ptr + (k * TK) * N + sg_starty / sg_size * TN,
                   N);
               joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
             } else {
               joint_matrix_load(sg, sub_bp,
                                 B_ptr +
                                     (k * TK / vnniFactor) * (N * vnniFactor) +
                                     sg_starty / sg_size * TN * vnniFactor,
                                 N * vnniFactor);

               joint_matrix_mad(sg, sub_c, sub_a, sub_bp, sub_c);
             }
           }
           auto C_w_ptr = syclex::annotated_ptr{
               C,
               syclex::properties{syclintelex::write_hint<
                   syclintelex::cache_control<syclintelex::cache_mode::uncached,
                                              syclex::cache_level::L2>>}};
           joint_matrix_store(sg, sub_c,
                              C_w_ptr + (sg_startx * TM) * N +
                                  sg_starty / sg_size * TN,
                              N, layout::row_major);
         }); // parallel for
   }).wait();
}

int main() {
  queue q;
  static constexpr size_t M = TM * 2;
  static constexpr size_t N = TN * 2;
  static constexpr size_t K = TK * 2;
  static constexpr unsigned int vnniFactor = 2;
  bfloat16 *A = malloc_shared<bfloat16>(M * K, q);
  bfloat16 *B = malloc_shared<bfloat16>(K * N, q);
  bfloat16 *vnniB = malloc_shared<bfloat16>(K * N, q);
  float *C = malloc_shared<float>(M * N, q);
  float *D = malloc_shared<float>(M * N, q);

  matrix_fill(M, K, A, [](int i, int j) { return 1.0f * (i + j); });
  matrix_fill(K, N, (bfloat16 *)B,
              [](int i, int j) { return 2.0f * i + 3.0f * j; });
  matrix_fill(M, N, C, 1.0f);
  matrix_fill(M, N, D, 1.0f);

  matrix_vnni<bfloat16>(K, N, B, vnniB, vnniFactor);

  matrix_multiply_ref(A, B, D, M, N, K);

  // Currently row major B fails when annotated_ptr is used
  matrix_multiply<float, bfloat16, M, N, K, 0>(C, A, B, q);
  bool res0 = matrix_compare(M, N, C, D);
  std::cout << (res0 ? "B row major passed" : "failed") << std::endl;

  matrix_fill(M, N, C, 1.0f);
  matrix_multiply<float, bfloat16, M, N, K, vnniFactor>(C, A, vnniB, q);
  bool res1 = matrix_compare(M, N, C, D);
  std::cout << (res1 ? "B VNNI passed" : "failed") << std::endl;

  return !(res0 & res1);
}
