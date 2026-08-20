//===---joint_matrix_hip_multi_subgroup.cpp - DPC++ joint_matrix----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa %amd_arch_options %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: target-amd

// Regression test: a tiled GEMM whose work-group contains MORE THAN ONE
// sub-group (a 64x64 output tile computed by a 4x4 grid of wavefronts). Guards
// against the AMD fragment lane-index bug where the per-lane offset was derived
// from the work-group linear id instead of the sub-group local id, which made
// every sub-group beyond the first read/write out of range.

#include <cassert>
#include <cmath>
#include <vector>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::bfloat16;

constexpr int T = 16;      // MFMA tile M=N=K
constexpr int TILE = 64;   // work-group output tile (TILE x TILE)
constexpr int S = 128;     // full square matrix dimension (multiple of TILE)

int main() {
  std::vector<float> Alog(S * S), Blog(S * S), Ref(S * S, 0.f);
  for (int i = 0; i < S * S; ++i)
    Alog[i] = static_cast<float>((i * 3 + 1) % 7) - 3.f;
  for (int i = 0; i < S * S; ++i)
    Blog[i] = static_cast<float>((i * 2 + 1) % 5) - 2.f;
  for (int m = 0; m < S; ++m)
    for (int n = 0; n < S; ++n) {
      float e = 0.f;
      for (int k = 0; k < S; ++k)
        e += Alog[m * S + k] * Blog[k * S + n];
      Ref[m * S + n] = e;
    }

  std::vector<bfloat16> Amem(S * S), Bmem(S * S); // A row-major, B col-major
  for (int m = 0; m < S; ++m)
    for (int k = 0; k < S; ++k)
      Amem[m * S + k] = bfloat16(Alog[m * S + k]);
  for (int k = 0; k < S; ++k)
    for (int n = 0; n < S; ++n)
      Bmem[n * S + k] = bfloat16(Blog[k * S + n]);
  std::vector<float> C(S * S, 0.f);

  {
    queue q;
    buffer<bfloat16> bA(Amem.data(), range{(size_t)S * S});
    buffer<bfloat16> bB(Bmem.data(), range{(size_t)S * S});
    buffer<float> bC(C.data(), range{(size_t)S * S});
    // Work-group: {1, TILE/T, (TILE/T)*WAVE} where WAVE=64 -> 16 sub-groups.
    const int nwarp = TILE / T; // 4
    range<2> lws{(size_t)nwarp, (size_t)nwarp * 64};
    range<2> gws{(size_t)(S / TILE) * nwarp, (size_t)(S / TILE) * nwarp * 64};
    q.submit([&](handler &h) {
       accessor accA{bA, h, read_only};
       accessor accB{bB, h, read_only};
       accessor accC{bC, h, write_only};
       h.parallel_for(
           nd_range<2>{gws, lws}, [=](nd_item<2> it) {
             auto sg = it.get_sub_group();
             const int warpM = it.get_local_id(0);
             const int warpN = it.get_local_id(1) / 64;
             const int row = (it.get_group(0) * (TILE / T) + warpM) * T;
             const int col = (it.get_group(1) * (TILE / T) + warpN) * T;
             auto pa = accA.template get_multi_ptr<access::decorated::yes>();
             auto pb = accB.template get_multi_ptr<access::decorated::yes>();
             auto pc = accC.template get_multi_ptr<access::decorated::yes>();
             joint_matrix<sub_group, float, use::accumulator, T, T> c;
             joint_matrix<sub_group, bfloat16, use::a, T, T, layout::row_major> a;
             joint_matrix<sub_group, bfloat16, use::b, T, T, layout::col_major> b;
             joint_matrix_fill(sg, c, 0.f);
             for (int kk = 0; kk < S; kk += T) {
               joint_matrix_load(sg, a, pa + row * S + kk, S);
               joint_matrix_load(sg, b, pb + col * S + kk, S);
               joint_matrix_mad(sg, c, a, b, c);
             }
             joint_matrix_store(sg, c, pc + row * S + col, S,
                                layout::row_major);
           });
     }).wait();
  }

  for (int i = 0; i < S * S; ++i)
    assert(std::fabs(C[i] - Ref[i]) < 4.f && "multi-subgroup GEMM mismatch");
  return 0;
}
