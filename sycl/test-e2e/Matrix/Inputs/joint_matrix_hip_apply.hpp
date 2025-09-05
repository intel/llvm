//===---joint_matrix_hip_apply.hpp - DPC++ joint_matrix--------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cstddef>
#include <cstdint>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::bfloat16;

template <typename InType, typename OutType, size_t M, size_t N, size_t K>
void hip_matrix_apply() {
  InType A[M * K];
  InType B[K * N];
  OutType C[M * N];
  OutType D[M * N];
  OutType E[M * N];

  for (auto i = 0; i < M * K; ++i) {
    A[i] = 1;
  }

  for (auto i = 0; i < K * N; ++i) {
    B[i] = 2;
  }

  for (auto i = 0; i < M * N; ++i) {
    D[i] = 0;
    C[i] = 3;
    E[i] = 3;
  }

  try {
    auto defaultQueue = sycl::queue{};

    auto bufA = sycl::buffer{A, sycl::range{M * K}};
    auto bufB = sycl::buffer{B, sycl::range{K * N}};
    auto bufC = sycl::buffer{C, sycl::range{M * N}};
    auto bufD = sycl::buffer{D, sycl::range{M * N}};

    defaultQueue
        .submit([&](sycl::handler &cgh) {
          sycl::accessor accA{bufA, cgh, sycl::read_write};
          sycl::accessor accB{bufB, cgh, sycl::read_write};
          sycl::accessor accC{bufC, cgh, sycl::read_only};
          sycl::accessor accD{bufD, cgh, sycl::write_only};

          cgh.parallel_for(
              sycl::nd_range<2>{{4, 16}, {4, 16}}, [=](sycl::nd_item<2> idx) {
                auto sg = idx.get_sub_group();
                joint_matrix<sub_group, OutType, use::accumulator, M, N> sub_c;
                joint_matrix<sub_group, InType, use::b, K, N, layout::row_major>
                    sub_b;
                joint_matrix<sub_group, InType, use::a, M, K, layout::col_major>
                    sub_a;

                joint_matrix_load(
                    sg, sub_a,
                    accA.template get_multi_ptr<access::decorated::yes>(), K);
                joint_matrix_load(
                    sg, sub_b,
                    accB.template get_multi_ptr<access::decorated::yes>(), N);
                joint_matrix_load(
                    sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(), N,
                    layout::row_major);

                joint_matrix_apply(sg, sub_a, [=](InType &v) { v *= 2; });
                joint_matrix_apply(sg, sub_b, [=](InType &v) { v *= 3; });
                joint_matrix_apply(sg, sub_c, [=](OutType &v) { v *= 4; });

                joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);

                joint_matrix_store(
                    sg, sub_c,
                    accD.template get_multi_ptr<access::decorated::yes>(), N,
                    layout::row_major);
              });
        })
        .wait();

    defaultQueue.throw_asynchronous();
  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      E[m * N + n] *= 4;
      for (int k = 0; k < K; k++) {
        E[m * N + n] += A[m * K + k] * 2 * B[k * N + n] * 3;
      }
    }
  }

  for (int i = 0; i < M * N; ++i) {
    assert(D[i] == E[i] && "Unexpected difference");
  }
};
