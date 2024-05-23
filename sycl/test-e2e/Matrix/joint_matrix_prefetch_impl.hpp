//===---joint_matrix_prefetch_impl.hpp - DPC++ joint_matrix----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/usm.hpp>

#define TM 8
#define TK 16

template <layout B_layout, layout C_layout, unsigned int vnniFactor> class mult;

template <typename T1, typename T2, typename T, size_t M, size_t N, size_t K,
          layout B_layout, layout C_layout, unsigned int vnniFactor>
void joint_matrix_gemm_vnni(sub_group sg, size_t sg_startx, size_t sg_starty,
                            size_t sg_size, T1 *A, T2 *B, T *C) {
  auto pA = address_space_cast<sycl::access::address_space::global_space,
                               access::decorated::no>(A);
  auto pB = address_space_cast<sycl::access::address_space::global_space,
                               access::decorated::no>(B);
  auto pC = address_space_cast<sycl::access::address_space::global_space,
                               access::decorated::no>(C);

  joint_matrix<sub_group, T1, use::a, TM, TK, layout::row_major> sub_a;
  joint_matrix<sub_group, T2, use::b, TK, TN, B_layout> sub_b;
  joint_matrix<sub_group, T, use::accumulator, TM, TN> sub_c;
  joint_matrix_prefetch<TM, TK>(sg, A + (sg_startx * TM) * K, K,
                                layout::row_major,
                                syclex::properties{syclex::prefetch_hint_L1});
  joint_matrix_prefetch<TK, TN>(sg, B + sg_starty / sg_size * TN * vnniFactor,
                                N * vnniFactor, B_layout,
                                syclex::properties{syclex::prefetch_hint_L1});

  if constexpr (C_layout == layout::col_major)
    joint_matrix_prefetch<TM, TN>(
        sg, C + (sg_starty / sg_size * TN) * M + (sg_startx * TM), M, C_layout,
        syclex::properties{syclex::prefetch_hint_L1});
  else
    joint_matrix_prefetch<TM, TN>(
        sg, C + (sg_startx * TM) * N + sg_starty / sg_size * TN, N, C_layout,
        syclex::properties{syclex::prefetch_hint_L1});

  joint_matrix_fill(sg, sub_c, 1);
  for (int k = 0; k < K; k += TK) {
    joint_matrix_load(sg, sub_a, pA + (sg_startx * TM) * K + k, K);
    joint_matrix_load(sg, sub_b,
                      pB + k * N + sg_starty / sg_size * TN * vnniFactor,
                      N * vnniFactor);
    joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
  }

  if constexpr (C_layout == layout::col_major)
    joint_matrix_store(sg, sub_c,
                       pC + (sg_starty / sg_size * TN) * M + (sg_startx * TM),
                       M, C_layout);
  else
    joint_matrix_store(sg, sub_c,
                       pC + (sg_startx * TM) * N + sg_starty / sg_size * TN, N,
                       C_layout);
}

template <typename T, typename T1, typename T2, size_t M, size_t N, size_t K,
          layout B_layout, layout C_layout, unsigned int vnniFactor>
void matrix_multiply(T *C, T1 *A, T2 *B, queue q) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;

  size_t sg_size = get_sg_size<mult<B_layout, C_layout, vnniFactor>>(q);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<mult<B_layout, C_layout, vnniFactor>>(
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
           joint_matrix_gemm_vnni<T1, T2, T, M, N, K, B_layout, C_layout,
                                  vnniFactor>(sg, sg_startx, sg_starty, sg_size,
                                              A, B, C);
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
    if (combinations[i].nsize == 0 || combinations[i].nsize == 16) {
      support_p = true;
      break;
    }
  }
  if (!support_p) {
    std::cout << "Prefetch not supported on this device" << std::endl;
    // Once the test is not marked as XFAIL, this should change to return 0;
    return 1;
  }
  static constexpr size_t M = TM * 2;
  static constexpr size_t N = TN * 2;
  static constexpr size_t K = TK * 2;
  static constexpr unsigned int vnniFactor = 2;
  bfloat16 *A = malloc_shared<bfloat16>(M * K, q);
  bfloat16 *B = malloc_shared<bfloat16>(K * N, q);
  bfloat16 *vnniB = malloc_shared<bfloat16>(K * N, q);
  float *C = malloc_shared<float>(M * N, q);
  float *D = malloc_shared<float>(M * N, q);
  float *Dt = malloc_shared<float>(M * N, q);

  matrix_fill(M, K, A, [](int i, int j) { return 1.0f * (i + j); });
  matrix_fill(K, N, (bfloat16 *)B,
              [](int i, int j) { return 2.0f * i + 3.0f * j; });
  matrix_fill(M, N, C, 1.0f);
  matrix_fill(M, N, D, 1.0f);
  matrix_fill(M, N, Dt, 1.0f);

  matrix_vnni<bfloat16>(K, N, B, vnniB, vnniFactor);

  matrix_multiply_ref(A, B, D, M, N, K);
  matrix_multiply<float, bfloat16, bfloat16, M, N, K, layout::row_major,
                  layout::row_major, 1>(C, A, B, q);

  bool res = matrix_compare(M, N, C, D);
  matrix_multiply<float, bfloat16, bfloat16, M, N, K, layout::ext_intel_packed,
                  layout::row_major, vnniFactor>(C, A, vnniB, q);

  res = res && matrix_compare(M, N, C, D);

  matrix_multiply_ref(A, B, Dt, M, N, K, true);
  matrix_multiply<float, bfloat16, bfloat16, M, N, K, layout::row_major,
                  layout::col_major, 1>(C, A, B, q);

  res = res && matrix_compare(M, N, C, Dt);
  std::cout << (res ? "passed" : "failed") << std::endl;
  return !res;
}
