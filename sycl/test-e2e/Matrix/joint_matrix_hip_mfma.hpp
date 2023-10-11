
#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::bfloat16;

template <typename, size_t M, size_t N> struct input_limit {
  static constexpr int value = M * N;
};

template <> struct input_limit<int8_t, 16, 16> {
  static constexpr auto value = 128;
};

template <> struct input_limit<int8_t, 32, 32> {
  static constexpr auto value = 128;
};

template <typename InType, typename OutType, size_t M, size_t N, size_t K,
          layout OutLayout>
void hip_matrix_mfma() {
  InType A[M * K];
  InType B[K * N];
  OutType C[M * N];
  OutType D[M * N];
  OutType E[M * N];

  for (auto i = 0; i < M * K; ++i) {
    A[i] = i % input_limit<InType, M, N>::value;
  }

  for (auto i = 0; i < K * N; ++i) {
    B[i] = i % input_limit<InType, M, N>::value;
  }

  for (auto i = 0; i < M * N; ++i) {
    D[i] = 0;
    C[i] = i;
    if (OutLayout == layout::row_major)
      E[i] = i;
    else
      E[(i % N) * M + int(i / M)] = i;
  }

  try {
    auto defaultQueue = sycl::queue{};

    auto bufA = sycl::buffer{A, sycl::range{M * K}};
    auto bufB = sycl::buffer{B, sycl::range{K * N}};
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
                joint_matrix<sub_group, OutType, use::accumulator, M, N>
                    sub_c{};
                joint_matrix<sub_group, InType, use::b, K, N, layout::row_major>
                    sub_b{};
                joint_matrix<sub_group, InType, use::a, M, K, layout::col_major>
                    sub_a{};

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

                sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);

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

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        if (OutLayout == layout::row_major)
          E[m * N + n] += A[m * K + k] * B[k * N + n];
        else
          E[n * M + m] += A[m * K + k] * B[k * N + n];
      }
    }
  }

  for (int i = 0; i < M * N; ++i) {
    assert(abs(D[i] - E[i]) <= D[i] / 100 && "Unexpected difference");
  }
};
