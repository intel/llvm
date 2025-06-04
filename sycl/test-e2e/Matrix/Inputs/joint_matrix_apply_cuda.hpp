//==------------ joint_matrix_apply_cuda.hpp  - DPC++ joint_matrix----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// cuda backend test for joint_matrix_apply (and joint_matrix_fill)

#include "common.hpp"

#include <cmath>

#define SG_SZ 32

constexpr size_t nWGperDim = 2;
nd_range<2> r({nWGperDim, nWGperDim *SG_SZ}, {1, 1 * SG_SZ});

template <typename T1, typename T2, size_t M, size_t K, size_t N>
class KernelApply;

template <typename T, size_t M, size_t N>
void assert_ref(T *C, const float ref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      auto diff = C[i + j * M] - ref;
      assert(std::fabs(static_cast<float>(diff)) <
             std::numeric_limits<float>::epsilon());
    }
}

template <typename T, typename T2, size_t M, size_t K, size_t N, typename F>
void matrix_verify_lambda(queue q,
                          big_matrix<T2, M * nWGperDim, N * nWGperDim> &C,
                          const float ref, F &&lambda) {
  {
    buffer<T2, 2> bufC(C.get_data(), range<2>(N * nWGperDim, M * nWGperDim));

    q.submit([&](handler &cgh) {
      accessor<T2, 2, access::mode::read_write, target::device> accC(bufC, cgh);

      cgh.parallel_for<KernelApply<T, T2, M, K, N>>(
          r, [accC, lambda](
                 nd_item<2> spmd_item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
            const auto global_idx = spmd_item.get_global_id(0);
            const auto global_idy = spmd_item.get_global_id(1);
            const auto sg_startx = global_idx - spmd_item.get_local_id(0);
            const auto sg_starty = global_idy - spmd_item.get_local_id(1);

            auto sg = spmd_item.get_sub_group();

            joint_matrix<sycl::sub_group, T, use::a, M, K, layout::row_major>
                sub_a;
            joint_matrix<sycl::sub_group, T, use::b, K, N, layout::row_major>
                sub_b;
            joint_matrix<sycl::sub_group, T2, use::accumulator, M, N> sub_c;

            joint_matrix_fill(sg, sub_a, 3);
            joint_matrix_fill(sg, sub_b, 1);
            joint_matrix_fill(sg, sub_c, -80);

            joint_matrix_apply(sg, sub_a, lambda);

            joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);

            joint_matrix_store(
                sg, sub_c,
                accC.template get_multi_ptr<access::decorated::no>() +
                    (sg_startx * M) * (N * nWGperDim) + sg_starty / SG_SZ * N,
                (N * nWGperDim), layout::row_major);
          }); // parallel for
    });
  }
  assert_ref<T2, M * nWGperDim, N * nWGperDim>(C.get_data(), ref);
}

class Logical {};

template <typename T1, typename T2, size_t M, size_t K, size_t N, typename OP>
class KernelWiData;

template <typename T, size_t M, size_t N>
void assert_ops_ref(T *C, const float ref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      auto diff = C[i + j * M] - ref;
      assert(std::fabs(static_cast<float>(diff)) <
             std::numeric_limits<float>::epsilon());
    }
}
