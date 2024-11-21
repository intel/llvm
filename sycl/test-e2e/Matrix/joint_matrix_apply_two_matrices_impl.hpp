//==------- joint_matrix_apply_two_matrices_impl.hpp  - DPC++ joint_matrix--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <sycl/usm.hpp>

template <typename Tc, typename Ta, size_t M, size_t N>
bool apply_verify(Tc *C, Tc *D, Ta *A, Ta *Ar, Tc *Cref, Ta *Aref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      Tc diffd = D[i * N + j] - Cref[i * N + j] * 2;
      Tc diffc = C[i * N + j] - sycl::max(static_cast<Tc>(0), Cref[i * N + j]);
      Ta diffar = Ar[i * N + j] - (Aref[i * N + j] + 42);
      Ta diffa = A[i * N + j] - (Aref[i * N + j] + 5);
      if constexpr (std::is_same_v<Ta, bfloat16>) {
        if (std::fabs(diffd) > FLOAT_EPSILON ||
            std::fabs(diffc) > FLOAT_EPSILON ||
            std::fabs(diffar) > FLOAT_EPSILON ||
            std::fabs(diffa) > FLOAT_EPSILON || std::isnan(C[i * N + j]) ||
            std::isnan(A[i * N + j])) {
          return false;
        }
      } else {
        if (std::abs(diffd) > 0 || std::abs(diffc) > 0 ||
            std::abs(diffar) > 0 || std::abs(diffa) > 0) {
          return false;
        }
      }
    }
  return true;
}
template <typename Tc, typename Ta, size_t TM, size_t TN, size_t TK, size_t M,
          size_t N, size_t K, class kernel_name>
bool apply_two_matrices(Tc *C, Tc *D, Ta *A, Ta *Ar, Tc *Cref, Ta *Aref,
                        queue q) {
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;

  size_t sg_size = get_sg_size<kernel_name>(q);
  q.submit([&](handler &cgh) {
     cgh.parallel_for<kernel_name>(
         nd_range<2>({NDRangeM, NDRangeN * sg_size}, {1, 1 * sg_size}),
         [=](nd_item<2> spmd_item)
#ifdef SG_SZ
             [[sycl::reqd_sub_group_size(SG_SZ)]]
#endif
         {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           auto pC =
               address_space_cast<sycl::access::address_space::global_space,
                                  access::decorated::no>(C);
           auto pD =
               address_space_cast<sycl::access::address_space::global_space,
                                  access::decorated::no>(D);
           auto pA =
               address_space_cast<sycl::access::address_space::global_space,
                                  access::decorated::no>(A);
           auto pAr =
               address_space_cast<sycl::access::address_space::global_space,
                                  access::decorated::no>(Ar);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<sub_group, Ta, use::a, TM, TK, layout::row_major> sub_a;
           joint_matrix<sub_group, Ta, use::a, TM, TK, layout::row_major>
               sub_ar;
           joint_matrix<sub_group, Tc, use::accumulator, TM, TN> sub_c;
           joint_matrix<sub_group, Tc, use::accumulator, TM, TN> sub_d;

           joint_matrix_load(
               sg, sub_c, pC + (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
           joint_matrix_apply(sg, sub_c, sub_d, [](Tc &x, Tc &y) {
             y = x * 2;
             x = sycl::max(static_cast<Tc>(0), x);
           });
           joint_matrix_store(
               sg, sub_d, pD + (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
           joint_matrix_store(
               sg, sub_c, pC + (sg_startx * TM) * N + sg_starty / sg_size * TN,
               N, layout::row_major);
           joint_matrix_load(
               sg, sub_a, pA + (sg_startx * TM) * K + sg_starty / sg_size * TK,
               K);
           joint_matrix_apply(sg, sub_a, sub_ar, [](Ta &x, Ta &y) {
             y = x + 42;
             x += 5;
           });
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_ar,
               pAr + (sg_startx * TM) * K + sg_starty / sg_size * TK, K);
           ext::intel::experimental::matrix::joint_matrix_store(
               sg, sub_a, pA + (sg_startx * TM) * K + sg_starty / sg_size * TK,
               K);
         }); // parallel for
   }).wait();
  return apply_verify<Tc, Ta, M, N>(C, D, A, Ar, Cref, Aref);
}

template <typename Ta, typename Tc, size_t TM, size_t TN, size_t TK,
          class kernel_name>
bool test() {
  static constexpr size_t M = TM * 2;
  static constexpr size_t N = TN * 2;
  static constexpr size_t K = TK * 2;
  queue q;

  Tc *Cref = malloc_shared<Tc>(M * N, q);
  Ta *Aref = malloc_shared<Ta>(M * K, q);
  Tc *C = malloc_shared<Tc>(M * N, q);
  Tc *D = malloc_shared<Tc>(M * N, q);
  Ta *A = malloc_shared<Ta>(M * K, q);
  Ta *Ar = malloc_shared<Ta>(M * K, q);

  matrix_rand(M, N, (Tc *)Cref, (Tc)100);
  matrix_rand(M, K, (Ta *)Aref, (Ta)100);
  matrix_copy(M, N, Cref, C);
  matrix_copy(M, K, Aref, A);

  bool res = apply_two_matrices<Tc, Ta, TM, TN, TK, M, N, K, kernel_name>(
      C, D, A, Ar, Cref, Aref, q);

  if constexpr (std::is_same_v<Ta, bfloat16>)
    std::cout << "bfloat16 " << TM << "x" << TN << "x" << TK << ": "
              << (res ? "passed" : "failed") << std::endl;
  else if constexpr (std::is_same_v<Ta, int8_t>)
    std::cout << "int8_t " << TM << "x" << TN << "x" << TK << ": "
              << (res ? "passed" : "failed") << std::endl;
  free(C, q);
  free(D, q);
  free(A, q);
  free(Ar, q);
  free(Cref, q);
  free(Aref, q);
  
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
      passed &= test<int8_t, int32_t, 16, 16, 64, class amx_int_16x16x64>();
      passed &= test<bfloat16, float, 16, 16, 32, class amx_bf16_16x16x32>();
      break;
    }

    if (combinations[i].nsize == 16) { // architecture::intel_gpu_pvc
      passed &= test<int8_t, int32_t, 8, 16, 32, class pvc_int_8x16x32>();
      passed &= test<bfloat16, float, 8, 16, 16, class pvc_bf16_8x16x16>();
// This combination is not currently supported for sub group size = 32 in IGC
#if (!defined(SG_SZ) || SG_SZ != 32)
      passed &= test<bfloat16, float, 16, 16, 16, class pvc_bf16_16x16x16>();
#endif
      break;
    }

    if (combinations[i].nsize == 8) { // architecture::intel_gpu_dg2*
      passed &= test<int8_t, int32_t, 8, 8, 32, class dg2_int_8x8x32>();
      passed &= test<bfloat16, float, 8, 8, 16, class dg2_bf16_8x16x16>();
      break;
    }
  }

  return !passed;
}
