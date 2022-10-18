//==----------- element_wise_all_ops_cuda.cpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-backend --cuda-gpu-arch=sm_80 -DSYCL_EXT_ONEAPI_MATRIX_VERSION=3 %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;
using sycl::ext::oneapi::experimental::bfloat16;

#define SG_SZ 32
constexpr size_t nWGperDim = 2;

class Logical {};

template <typename T1, typename T2, size_t M, size_t K, size_t N, typename OP>
class KernelName;

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T, size_t M, size_t N>
void assert_ops_ref(T *C, const float ref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      auto diff = C[i + j * M] - ref;
      assert(std::fabs(static_cast<float>(diff)) <
             std::numeric_limits<float>::epsilon());
    }
}
template <typename T, typename T2, size_t M, size_t K, size_t N,
          class Operation>
void matrix_verify_op(queue q, big_matrix<T2, M * nWGperDim, N * nWGperDim> &C,
                      nd_range<2> &r, const float ref, Operation Op) {
  {
    buffer<T2, 2> bufC(C.get_data(), range<2>(N * nWGperDim, M * nWGperDim));

    q.submit([&](handler &cgh) {
       accessor<T2, 2, access::mode::read_write, target::device> accC(bufC,
                                                                      cgh);

       cgh.parallel_for<KernelName<T, T2, M, K, N, Operation>>(
           r, [accC,
               Op](nd_item<2> spmd_item) [[sycl::reqd_sub_group_size(SG_SZ)]] {
             const auto global_idx = spmd_item.get_global_id(0);
             const auto global_idy = spmd_item.get_global_id(1);
             const auto sg_startx = global_idx - spmd_item.get_local_id(0);
             const auto sg_starty = global_idy - spmd_item.get_local_id(1);

             auto sg = spmd_item.get_sub_group();

             joint_matrix<T, matrix_use::a, M, K> sub_a;
             joint_matrix<T, matrix_use::b, K, N> sub_b;
             joint_matrix<T2, matrix_use::accumulator, M, N> sub_c;

             joint_matrix_fill(sg, sub_a, 3);
             joint_matrix_fill(sg, sub_b, 1);
             joint_matrix_fill(sg, sub_c, -80);

             auto wi_slice_a = sub_a.get_wi_data();
             for (int i = 0; i < wi_slice_a.length(); i++) {
               if constexpr (std::is_same_v<Operation, Logical>) {
                 if (wi_slice_a[i]) {
                   if (wi_slice_a[i] > 2.0 || wi_slice_a[i] >= 3.0 ||
                       wi_slice_a[i] < 4.0 || wi_slice_a[i] <= 3.0) {
                     T val = (wi_slice_a[i] != (2.0)) ? wi_slice_a[i]
                                                      : static_cast<T>(2.0);
                     val = ((val) - (1));
                     val = ((val) + (1));
                     if (wi_slice_a[i] == (2.0)) {
                       val = ((val) - (2));
                       val = ((val) * (3));
                       val = ((val) / (2));

                     } else {
                       val = ((val) + (2));
                     }
                     wi_slice_a[i] = val;
                   }
                 }
               } else {
                 wi_slice_a[i] = Op(wi_slice_a[i], 2);
               }
             }

             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);

             joint_matrix_store(sg, sub_c,
                                accC.get_pointer() +
                                    (sg_startx * M) * (N * nWGperDim) +
                                    sg_starty / SG_SZ * N,
                                (N * nWGperDim));
           }); // parallel for
     }).wait();
  }
  assert_ops_ref<T2, M * nWGperDim, N * nWGperDim>(C.get_data(), ref);
}

static constexpr size_t MATRIX_M = 16 * nWGperDim;
static constexpr size_t MATRIX_N = 16 * nWGperDim;

int main() {

  float D[MATRIX_M][MATRIX_N];
  big_matrix<float, MATRIX_M, MATRIX_N> MD_f((float *)&D);

  queue q;
  auto computeCapability =
      std::stof(q.get_device().get_info<info::device::backend_version>());
  nd_range<2> r({nWGperDim, nWGperDim * SG_SZ}, {1, 1 * SG_SZ});

  if (computeCapability >= 7.0) {
    matrix_verify_op<half, float, 16, 16, 16>(q, MD_f, r, 0.0,
                                              std::plus<half>{});
    matrix_verify_op<half, float, 16, 16, 16>(q, MD_f, r, 0.0, Logical{});
    matrix_verify_op<half, float, 16, 16, 16>(q, MD_f, r, 16.0,
                                              std::multiplies<half>{});
    matrix_verify_op<half, float, 16, 16, 16>(q, MD_f, r, -56.0,
                                              std::divides<half>{});
    matrix_verify_op<half, float, 16, 16, 16>(q, MD_f, r, -64.0,
                                              std::minus<half>{});
  }

  if (computeCapability >= 7.2) {
    int32_t D_i[MATRIX_M][MATRIX_N];
    big_matrix<int32_t, MATRIX_M, MATRIX_N> MD_i((int32_t *)&D_i);
    matrix_verify_op<uint8_t, int32_t, 16, 16, 16>(q, MD_i, r, 0,
                                                   std::plus<uint8_t>{});
    matrix_verify_op<uint8_t, int32_t, 16, 16, 16>(q, MD_i, r, 16,
                                                   std::multiplies<uint8_t>{});
    matrix_verify_op<uint8_t, int32_t, 16, 16, 16>(q, MD_i, r, -64,
                                                   std::minus<uint8_t>{});
    matrix_verify_op<int8_t, int32_t, 16, 16, 16>(q, MD_i, r, 0,
                                                  std::plus<int8_t>{});
    matrix_verify_op<int8_t, int32_t, 16, 16, 16>(q, MD_i, r, 0.0, Logical{});
    matrix_verify_op<int8_t, int32_t, 16, 16, 16>(q, MD_i, r, 16,
                                                  std::multiplies<int8_t>{});
    matrix_verify_op<int8_t, int32_t, 16, 16, 16>(q, MD_i, r, -64,
                                                  std::minus<int8_t>{});
  }

  if (computeCapability >= 8.0) {

    matrix_verify_op<bfloat16, float, 16, 16, 16>(q, MD_f, r, 0.0,
                                                  std::plus<bfloat16>{});
    matrix_verify_op<bfloat16, float, 16, 16, 16>(q, MD_f, r, 0.0, Logical{});
    matrix_verify_op<bfloat16, float, 16, 16, 16>(q, MD_f, r, 16.0,
                                                  std::multiplies<bfloat16>{});
    matrix_verify_op<bfloat16, float, 16, 16, 16>(q, MD_f, r, -56.0,
                                                  std::divides<bfloat16>{});
    matrix_verify_op<bfloat16, float, 16, 16, 16>(q, MD_f, r, -64.0,
                                                  std::minus<bfloat16>{});

    double D_d[MATRIX_M / 2][MATRIX_N / 2];
    big_matrix<double, 8 * nWGperDim, 8 * nWGperDim> MD_d((double *)&D_d);

    matrix_verify_op<double, double, 8, 4, 8>(q, MD_d, r, -60.0,
                                              std::plus<double>{});
    matrix_verify_op<double, double, 8, 4, 8>(q, MD_d, r, -60.0, Logical{});
    matrix_verify_op<double, double, 8, 4, 8>(q, MD_d, r, -56.0,
                                              std::multiplies<double>{});
    matrix_verify_op<double, double, 8, 4, 8>(q, MD_d, r, -74.0,
                                              std::divides<double>{});
    matrix_verify_op<double, double, 8, 4, 8>(q, MD_d, r, -76.0,
                                              std::minus<double>{});
  }

  return 0;
}
