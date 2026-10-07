//===---joint_matrix_hip_all_layouts.cpp - DPC++ joint_matrix-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa %amd_arch_options %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: target-amd

// Regression test: a 16x16x16 bf16 -> float tile computed for every
// combination of A/B memory layouts (row/col major), each stored in memory
// according to its declared layout, and checked against a standard row-major
// C = A*B reference. Guards against the AMD `use::a` load-path layout bug where
// the row-major A multiplicand was loaded transposed.

#include <cassert>
#include <cmath>
#include <vector>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::bfloat16;

constexpr int M = 16, N = 16, K = 16;

template <layout AL, layout BL> bool run_combo() {
  std::vector<float> Alog(M * K), Blog(K * N), Ref(M * N, 0.f);
  for (int i = 0; i < M * K; ++i)
    Alog[i] = static_cast<float>((i * 3 + 1) % 7) - 3.f;
  for (int i = 0; i < K * N; ++i)
    Blog[i] = static_cast<float>((i * 2 + 1) % 5) - 2.f;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float e = 0.f;
      for (int k = 0; k < K; ++k)
        e += Alog[m * K + k] * Blog[k * N + n];
      Ref[m * N + n] = e;
    }

  const int lda = (AL == layout::row_major) ? K : M;
  const int ldb = (BL == layout::row_major) ? N : K;
  std::vector<bfloat16> Amem(M * K), Bmem(K * N);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      Amem[(AL == layout::row_major) ? m * lda + k : k * lda + m] =
          bfloat16(Alog[m * K + k]);
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < N; ++n)
      Bmem[(BL == layout::row_major) ? k * ldb + n : n * ldb + k] =
          bfloat16(Blog[k * N + n]);
  std::vector<float> C(M * N, 0.f);

  {
    queue q;
    buffer<bfloat16> bA(Amem.data(), range{M * K});
    buffer<bfloat16> bB(Bmem.data(), range{K * N});
    buffer<float> bC(C.data(), range{M * N});
    q.submit([&](handler &h) {
       accessor accA{bA, h, read_only};
       accessor accB{bB, h, read_only};
       accessor accC{bC, h, write_only};
       h.parallel_for(nd_range<2>{{4, 16}, {4, 16}}, [=](nd_item<2> it) {
         auto sg = it.get_sub_group();
         joint_matrix<sub_group, float, use::accumulator, M, N> c;
         joint_matrix<sub_group, bfloat16, use::a, M, K, AL> a;
         joint_matrix<sub_group, bfloat16, use::b, K, N, BL> b;
         joint_matrix_fill(sg, c, 0.f);
         joint_matrix_load(
             sg, a, accA.template get_multi_ptr<access::decorated::yes>(), lda);
         joint_matrix_load(
             sg, b, accB.template get_multi_ptr<access::decorated::yes>(), ldb);
         joint_matrix_mad(sg, c, a, b, c);
         joint_matrix_store(
             sg, c, accC.template get_multi_ptr<access::decorated::yes>(), N,
             layout::row_major);
       });
     }).wait();
  }

  for (int i = 0; i < M * N; ++i)
    if (std::fabs(C[i] - Ref[i]) > 2.f)
      return false;
  return true;
}

int main() {
  bool rr = run_combo<layout::row_major, layout::row_major>();
  bool rc = run_combo<layout::row_major, layout::col_major>();
  bool cr = run_combo<layout::col_major, layout::row_major>();
  bool cc = run_combo<layout::col_major, layout::col_major>();
  assert(rr && "A=row B=row");
  assert(rc && "A=row B=col");
  assert(cr && "A=col B=row");
  assert(cc && "A=col B=col");
  return 0;
}
