//==----------- element_wise_all_ops_bf16.cpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <iostream>
#include <random>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 8

#define TM 8
#define TN SG_SZ
#define TK 16

static float make_fp32(uint16_t x) {
  unsigned int y = x;
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

static uint16_t make_bf16(float x) {
  int *res = reinterpret_cast<int *>(&x);
  *res = *res >> 16;
  return (uint16_t)*res;
}

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T, size_t M, size_t N>
void assert_ops_ref(
    accessor<T, 2, access::mode::read, access::target::host_buffer> C,
    const float ref) {
  for (size_t i = 0; i < M; i++)
    for (size_t j = 0; j < N; j++) {
      auto diff = make_fp32(C[i][j]) - ref;
      assert(std::fabs(static_cast<float>(diff)) <
             std::numeric_limits<float>::epsilon());
    }
}
template <typename T, size_t M, size_t N>
void matrix_verify_add(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                       const float ref) {
  buffer<unsigned short, 2> bufA(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accA = bufA.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class add_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<T, TM, TK> sub_a(sg);

           joint_matrix_fill(sg, sub_a, make_bf16(5.0));

           auto wi_slice_a = sub_a.get_wi_data();
           for (int i = 0; i < wi_slice_a.length(); i++) {
             wi_slice_a[i] = wi_slice_a[i] + make_bf16(2);
           }
           joint_matrix_store(sg, sub_a,
                              accA.get_pointer() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, matrix_layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufA.get_access<access::mode::read>(), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_sub(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                       const float ref) {
  buffer<unsigned short, 2> bufA(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accA = bufA.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class sub_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<T, TM, TK> sub_a(sg);

           joint_matrix_fill(sg, sub_a, make_bf16(5.0));

           auto wi_slice_a = sub_a.get_wi_data();
           for (int i = 0; i < wi_slice_a.length(); i++) {
             wi_slice_a[i] = wi_slice_a[i] - make_bf16(2);
           }
           joint_matrix_store(sg, sub_a,
                              accA.get_pointer() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, matrix_layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufA.get_access<access::mode::read>(), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_mul(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                       const float ref) {
  buffer<unsigned short, 2> bufA(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accA = bufA.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class mul_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<T, TM, TK> sub_a(sg);

           joint_matrix_fill(sg, sub_a, make_bf16(5.0));

           auto wi_slice_a = sub_a.get_wi_data();
           for (int i = 0; i < wi_slice_a.length(); i++) {
             wi_slice_a[i] = wi_slice_a[i] * make_bf16(3.0);
           }
           joint_matrix_store(sg, sub_a,
                              accA.get_pointer() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, matrix_layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufA.get_access<access::mode::read>(), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_div(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                       const float ref) {
  buffer<unsigned short, 2> bufA(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accA = bufA.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class div_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<T, TM, TK> sub_a(sg);

           joint_matrix_fill(sg, sub_a, make_bf16(4.0));

           auto wi_slice_a = sub_a.get_wi_data();
           for (int i = 0; i < wi_slice_a.length(); i++) {
             wi_slice_a[i] = wi_slice_a[i] / make_bf16(2.0);
           }
           joint_matrix_store(sg, sub_a,
                              accA.get_pointer() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, matrix_layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufA.get_access<access::mode::read>(), ref);
}

template <typename T, size_t M, size_t N>
void matrix_verify_logic(queue q, big_matrix<T, M, N> &A, nd_range<2> &r,
                         const float ref) {
  buffer<unsigned short, 2> bufA(A.get_data(), range<2>(M, N));

  q.submit([&](handler &cgh) {
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     cgh.parallel_for<class logic_matrix>(
         r, [accA](nd_item<2> spmd_item) [[intel::reqd_sub_group_size(SG_SZ)]] {
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           ext::oneapi::sub_group sg = spmd_item.get_sub_group();
           joint_matrix<T, TM, TK> sub_a(sg);

           joint_matrix_fill(sg, sub_a, make_bf16(5.0));

           auto wi_slice_a = sub_a.get_wi_data();
           for (int i = 0; i < wi_slice_a.length(); i++) {
             if (wi_slice_a[i]) {
               if (wi_slice_a[i] > make_bf16(2.0) ||
                   wi_slice_a[i] >= make_bf16(2.0) ||
                   wi_slice_a[i] < make_bf16(2.0) ||
                   wi_slice_a[i] <= make_bf16(2.0)) {
                 T val = (wi_slice_a[i] != make_bf16(2.0)) ? wi_slice_a[i]
                                                           : make_bf16(2.0);
                 val = make_bf16(make_fp32(val) - static_cast<float>(1));
                 val = make_bf16(make_fp32(val) + static_cast<float>(1));
                 if (wi_slice_a[i] == make_bf16(2.0)) {
                   val = make_bf16(make_fp32(val) - static_cast<float>(2));
                   val = make_bf16(make_fp32(val) * static_cast<float>(3));
                   val = make_bf16(make_fp32(val) / static_cast<float>(2));

                 } else {
                   val = make_bf16(make_fp32(val) + static_cast<float>(2));
                 }
                 wi_slice_a[i] = val;
               }
             }
           }
           joint_matrix_store(sg, sub_a,
                              accA.get_pointer() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, matrix_layout::row_major);
         }); // parallel for
   }).wait();
  assert_ops_ref<T, M, N>(bufA.get_access<access::mode::read>(), ref);
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
unsigned short A[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];

void matrix_ops_ref(float *D, int M, int N) {
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      *(D + m * N + n) = 0;
      *(D + m * N + n) *= 2;
    }
}

int main() {

  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<unsigned short, MATRIX_M, MATRIX_N> MA((unsigned short *)&A);

  size_t NDRangeM = MATRIX_M / TM;
  size_t NDRangeN = MATRIX_N / TN;
  queue q;
  nd_range<2> r({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ});

  matrix_verify_add<unsigned short, MATRIX_M, MATRIX_N>(q, MA, r, 7.0);
  matrix_verify_sub<unsigned short, MATRIX_M, MATRIX_N>(q, MA, r, 3.0);
  matrix_verify_mul<unsigned short, MATRIX_M, MATRIX_N>(q, MA, r, 15.0);
  matrix_verify_div<unsigned short, MATRIX_M, MATRIX_N>(q, MA, r, 2.0);
  matrix_verify_logic<unsigned short, MATRIX_M, MATRIX_N>(q, MA, r, 7.0);

  return 0;
}
