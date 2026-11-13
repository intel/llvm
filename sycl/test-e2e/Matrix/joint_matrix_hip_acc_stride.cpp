//===---joint_matrix_hip_acc_stride.cpp - DPC++ joint_matrix---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -fsycl-targets=amdgcn-amd-amdhsa %amd_arch_options %s -o %t.out
// RUN: %{run} %t.out

// REQUIRES: target-amd

// Regression test: loads and stores accumulator tiles of a larger matrix, so
// that the stride differs from the tile size, for both layouts. The 32x32
// row-major load used to step between groups of rows with the tile size
// instead of the stride.

#include <cassert>
#include <vector>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

template <size_t T, layout LoadL, layout StoreL> void test() {
  constexpr size_t S = 2 * T + 8;  // Leading dimension of the full matrices.
  constexpr size_t R0 = 4, C0 = T; // Origin of the tile.
  std::vector<float> In(S * S), Out(S * S, -1.f);
  for (size_t i = 0; i < S * S; ++i)
    In[i] = static_cast<float>(i);
  {
    queue Q;
    buffer<float> BufIn(In.data(), range{S * S});
    buffer<float> BufOut(Out.data(), range{S * S});
    Q.submit([&](handler &CGH) {
       accessor AccIn{BufIn, CGH, read_only};
       accessor AccOut{BufOut, CGH, read_write};
       CGH.parallel_for(nd_range<2>{{1, 64}, {1, 64}}, [=](nd_item<2> It) {
         auto SG = It.get_sub_group();
         joint_matrix<sub_group, float, use::accumulator, T, T> C;
         const size_t InOff =
             LoadL == layout::row_major ? R0 * S + C0 : C0 * S + R0;
         const size_t OutOff =
             StoreL == layout::row_major ? R0 * S + C0 : C0 * S + R0;
         joint_matrix_load(
             SG, C,
             AccIn.template get_multi_ptr<access::decorated::yes>() + InOff, S,
             LoadL);
         joint_matrix_store(
             SG, C,
             AccOut.template get_multi_ptr<access::decorated::yes>() + OutOff,
             S, StoreL);
       });
     }).wait();
  }
  auto Idx = [](size_t R, size_t C, layout L) {
    return L == layout::row_major ? R * S + C : C * S + R;
  };
  for (size_t r = 0; r < S; ++r)
    for (size_t c = 0; c < S; ++c) {
      const bool InTile = r >= R0 && r < R0 + T && c >= C0 && c < C0 + T;
      const float Expected = InTile ? In[Idx(r, c, LoadL)] : -1.f;
      assert(Out[Idx(r, c, StoreL)] == Expected &&
             "accumulator load/store with stride mismatch");
    }
}

template <size_t T> void test_layouts() {
  test<T, layout::row_major, layout::row_major>();
  test<T, layout::row_major, layout::col_major>();
  test<T, layout::col_major, layout::row_major>();
  test<T, layout::col_major, layout::col_major>();
}

int main() {
  test_layouts<16>();
  test_layouts<32>();
  return 0;
}
