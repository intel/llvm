//===---joint_matrix_hip_mfma.hpp - DPC++ joint_matrix---------------------===//
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
#include <random>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::bfloat16;

template <typename InType, typename OutType, size_t M, size_t N, size_t K,
          size_t KX, layout OutLayout>
void hip_matrix_mfma() {
  InType A[M * K * KX];
  InType B[K * N * KX];
  OutType C[M * N];
  OutType D[M * N];
  OutType E[M * N];

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-10, 10);

  for (auto i = 0; i < M * K * KX; ++i) {
    A[i] = static_cast<InType>(dist(gen));
  }

  for (auto i = 0; i < K * N * KX; ++i) {
    B[i] = static_cast<InType>(dist(gen));
  }

  for (auto i = 0; i < M * N; ++i) {
    D[i] = 0;
    C[i] = static_cast<OutType>(dist(gen));
    if (OutLayout == layout::row_major)
      E[i] = C[i];
    else
      E[(i % N) * M + int(i / M)] = C[i];
  }

  try {
    auto defaultQueue = sycl::queue{};

    auto bufA = sycl::buffer{A, sycl::range{M * K * KX}};
    auto bufB = sycl::buffer{B, sycl::range{K * N * KX}};
    auto bufC = sycl::buffer{C, sycl::range{M * N}};
    auto bufD = sycl::buffer{D, sycl::range{M * N}};

    defaultQueue
        .submit([&](sycl::handler &cgh) {
          sycl::accessor accA{bufA, cgh, sycl::read_only};
          sycl::accessor accB{bufB, cgh, sycl::read_only};
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
                    sg, sub_c,
                    accC.template get_multi_ptr<access::decorated::yes>(), N,
                    layout::row_major);

                for (auto kx = 0; kx < KX; ++kx) {
                  joint_matrix_load(
                      sg, sub_a,
                      accA.template get_multi_ptr<access::decorated::yes>() +
                          kx * K,
                      K * KX);
                  joint_matrix_load(
                      sg, sub_b,
                      accB.template get_multi_ptr<access::decorated::yes>() +
                          kx * K * N,
                      N);
                  joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
                }

                joint_matrix_store(
                    sg, sub_c,
                    accD.template get_multi_ptr<access::decorated::yes>(), N,
                    OutLayout);
              });
        })
        .wait();

    defaultQueue.throw_asynchronous();
  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  constexpr int LDA = K * KX;

  for (auto m = 0; m < M; m++) {
    for (auto n = 0; n < N; n++) {
      OutType e = 0;
      for (auto k = 0; k < LDA; k++) {
        e += A[m * LDA + k] * B[k * N + n];
      }
      if (OutLayout == layout::row_major)
        E[m * N + n] += e;
      else
        E[n * M + m] += e;
    }
  }

  for (auto i = 0; i < M * N; ++i) {
    assert(abs(D[i] - E[i]) < 100 && "Unexpected difference");
  }
};
