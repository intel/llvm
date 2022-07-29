//==------- simd_view_select_2d.hpp  - DPC++ ESIMD on-device test ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Smoke test for 2D region select API which can be used to represent 2D tiles.

#include "../esimd_test_utils.hpp"

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

template <typename T> struct char_to_int {
  using type = typename std::conditional<
      sizeof(T) == 1,
      typename std::conditional<std::is_signed<T>::value, int, unsigned>::type,
      T>::type;
};

template <class T, int... I> class test_id;

// This function:
// - Creates 3 matrices - A[M x K], B[K x N] and C[M x N].
// - Selects a subregion from each using 2D select:
//   tile_a[Wm x Wn], tile_b[Wk x Wn] and tile_c[Wm x Wn].
//   When selecting along each dimension - M, N and K -
//   offsets off_m, off_n, off_k and strides Sm, Sn and Sk are used.
// - multiplies tile_a x tile_b as matrices and writes result to tile_c.
//
template <
    // element type:
    class T,
    // input/output matrix sizes:
    int M, int N, int K,
    // input/output matrix region sizes (widths) being multiplied:
    // A[Wm x Wk] x B[Wk x Wn] = C[Wm x Wn]
    int Wm, int Wn, int Wk,
    // strides used to select the regions:
    int Sm, int Sn, int Sk>
bool test_impl(queue q, int off_m, int off_n, int off_k) {
  assert((off_m + Wm * Sm <= M) && (off_n + Wn * Sn <= N) &&
         (off_k + Wk * Sk <= K));

  std::cout << "Testing T=" << typeid(T).name() << " [M,N,K]=[" << M << "," << N
            << "," << K << "]"
            << " [Wm,Wn,Wk]=[" << Wm << "," << Wn << "," << Wk << "]"
            << " [Sm,Sn,Sk]=[" << Sm << "," << Sn << "," << Sk << "]"
            << " [off_m,off_n,off_k]=[" << off_m << "," << off_n << "," << off_k
            << "]"
            << "\n";

  T *mat_a = sycl::malloc_shared<T>(M * K, q);
  T *mat_b = sycl::malloc_shared<T>(K * N, q);
  T *mat_c = sycl::malloc_shared<T>(M * N, q);
  T *gold = new T[M * N];

  for (int a = 0; a < M * K; a++) {
    // 1 1 1 ...
    // 2 2 2 ...
    // . . .
    // M M M ...
    mat_a[a] = (T)(a / K + 1);
  }
  for (int b = 0; b < K * N; b++) {
    // 1 1 1 ...
    // 2 2 2 ...
    // . . .
    // N N N ...
    mat_b[b] = (T)(b / N + 1);
  }
  for (int c = 0; c < M * N; c++) {
    mat_c[c] = (T)1;
    gold[c] = (T)1;
  }
  // Create gold data
  for (int m = 0; m < Wm; m++) {
    for (int n = 0; n < Wn; n++) {
      int ind_c = (off_m + m * Sm) * N + off_n + n * Sn;
      T acc = gold[ind_c];

      for (int k = 0; k < Wk; k++) {
        int ind_a = (off_m + m * Sm) * K + off_k + k * Sk;
        int ind_b = (off_k + k * Sk) * N + off_n + n * Sn;
        acc += mat_a[ind_a] * mat_b[ind_b];
      }
      gold[ind_c] = acc;
    }
  }

  try {
    auto e = q.submit([&](handler &cgh) {
      cgh.single_task<test_id<T, M, N, K, Wm, Wn, Wk, Sm, Sn, Sk>>(
          [=]() SYCL_ESIMD_KERNEL {
            simd<T, M * K> a(mat_a);
            simd<T, K * N> b(mat_b);
            simd<T, M * N> c(mat_c);

            auto tile_a = a.template bit_cast_view<T, M, K>()
                              .template select<Wm, Sm, Wk, Sk>(off_m, off_k);
            auto tile_b = b.template bit_cast_view<T, K, N>()
                              .template select<Wk, Sk, Wn, Sn>(off_k, off_n);
            auto tile_c = c.template bit_cast_view<T, M, N>()
                              .template select<Wm, Sm, Wn, Sn>(off_m, off_n);

            for (int m = 0; m < Wm; m++) {
              for (int n = 0; n < Wn; n++) {
                tile_c.template select<1, 1, 1, 1>(m, n) +=
                    reduce<T>(tile_a.row(m) * tile_b.column(n), std::plus<>{});
              }
            }
            c.copy_to(mat_c);
          });
    });
    e.wait_and_throw();
  } catch (sycl::exception const &e) {
    std::cout << "  SYCL exception caught: " << e.what() << '\n';
    sycl::free(mat_a, q);
    sycl::free(mat_b, q);
    sycl::free(mat_c, q);
    delete[] gold;
    return false;
  }
  int err_cnt = 0;

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      T gold_val = gold[m * N + n];
      T val = mat_c[m * N + n];
      using ValT = typename char_to_int<T>::type;

      if ((val != gold_val) && (++err_cnt < 20)) {
        std::cout << " ERROR at [" << m << "," << n << "]: " << (ValT)val
                  << " != " << (ValT)gold_val << "(gold)\n";
      }
    }
  }
  if (err_cnt > 0) {
    int NN = M * N;
    std::cout << "  pass rate: " << ((float)(NN - err_cnt) / (float)NN) * 100.0f
              << "% (" << (NN - err_cnt) << "/" << NN << ")\n";
  }
  std::cout << (err_cnt > 0 ? "  FAILED\n" : "  Passed\n");
  sycl::free(mat_a, q);
  sycl::free(mat_b, q);
  sycl::free(mat_c, q);
  delete[] gold;
  return err_cnt > 0 ? false : true;
}

template <class T> bool test(queue q) {
  bool passed = true;
  passed &= test_impl<T, 8, 8, 8, /**/ 8, 8, 8, /**/ 1, 1, 1>(q, 0, 0, 0);
  passed &= test_impl<T, 8, 16, 8, /**/ 3, 3, 2, /**/ 2, 3, 4>(q, 2, 1, 0);
  passed &= test_impl<T, 8, 16, 4, /**/ 3, 5, 2, /**/ 2, 3, 1>(q, 2, 1, 0);
  if constexpr (sizeof(T) > 1) // TODO w/a vISA builder bug
    passed &= test_impl<T, 9, 17, 5, /**/ 3, 5, 2, /**/ 2, 3, 1>(q, 2, 1, 0);
  return passed;
}
