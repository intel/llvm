//==-------- elemwise_irreg_size_ops_bf16.cpp  - DPC++ joint_matrix---- ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %clangxx -fsycl %s -o %t.out
// This test is for element wise operations when matrix size does not multiply
// SG size. This corner case only applies to AMX. Also, it tests bf16 type.
// only run this on AMX
// RUN: %CPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

#define SG_SZ 16

// 10x12 is not multiply the sg size, slicing implementation will have to insert
// padding
#define TM 10
#define TN 12
#define TK 16

template <typename T, size_t NUM_ROWS, size_t NUM_COLS> struct big_matrix {
public:
  T *mat;

public:
  T *get_data() { return mat; }
  void set_data(T *data) { mat = data; }
  big_matrix(T *data) : mat(data) {}
};

template <typename T1, typename T2, size_t NUM_ROWS_A, size_t NUM_COLS_A,
          size_t NUM_ROWS_B, size_t NUM_COLS_B, size_t NUM_ROWS_C,
          size_t NUM_COLS_C>
void matrix_multiply(big_matrix<T1, NUM_ROWS_C, NUM_COLS_C> &C,
                     big_matrix<T2, NUM_ROWS_A, NUM_COLS_A> &A,
                     big_matrix<T2, NUM_ROWS_B, NUM_COLS_B> &B) {
  size_t M = NUM_ROWS_C;
  size_t N = NUM_COLS_C;
  size_t K = NUM_COLS_A;

  assert(NUM_ROWS_C == NUM_ROWS_A && NUM_COLS_A == NUM_ROWS_B * 2);
  size_t NDRangeM = M / TM;
  size_t NDRangeN = N / TN;
  buffer<unsigned short, 2> bufA(A.get_data(), range<2>(M, K));
  buffer<unsigned short, 2> bufB(B.get_data(), range<2>(K / 2, N * 2));
  buffer<float, 2> bufC((float *)C.get_data(), range<2>(M, N));

  queue q;
  q.submit([&](handler &cgh) {
     auto accC = bufC.get_access<access::mode::read_write>(cgh);
     auto accA = bufA.get_access<access::mode::read_write>(cgh);
     auto accB = bufB.get_access<access::mode::read_write>(cgh);

     cgh.parallel_for<class imatrix>(
         nd_range<2>({NDRangeM, NDRangeN * SG_SZ}, {1, 1 * SG_SZ}),
         [accA, accB, accC, M, N, K](nd_item<2> spmd_item)
             [[intel::reqd_sub_group_size(SG_SZ)]]

         {
           // The submatrix API has to be accessed by all the workitems in a
           // subgroup these functions will be called once by the subgroup no
           // code divergence between the workitems
           const auto global_idx = spmd_item.get_global_id(0);
           const auto global_idy = spmd_item.get_global_id(1);
           const auto sg_startx = global_idx - spmd_item.get_local_id(0);
           const auto sg_starty = global_idy - spmd_item.get_local_id(1);

           sub_group sg = spmd_item.get_sub_group();
           joint_matrix<unsigned short, TM, TK> sub_a(sg);
           // For B, since current implementation does not support non-packed
           // layout, users need to specify the packed_b layout.
           // By default, the layout is row_major
           joint_matrix<unsigned short, TK, TN, matrix_layout::packed_b> sub_b(
               sg);
           joint_matrix<float, TM, TN> sub_c(sg);
           joint_matrix_load(sg, sub_c,
                             accC.get_pointer() + (sg_startx * TM) * N +
                                 sg_starty / SG_SZ * TN,
                             N, matrix_layout::row_major);
           for (int k = 0; k < K; k += TK) {
             joint_matrix_load(sg, sub_a,
                               accA.get_pointer() + (sg_startx * TM) * K + k, K,
                               matrix_layout::row_major);
             // Assume we alreay in vnni format.
             joint_matrix_load(sg, sub_b,
                               accB.get_pointer() + (k) * (N) +
                                   sg_starty / SG_SZ * TN * 2,
                               N * 2, matrix_layout::packed_b);
             sub_c = joint_matrix_mad(sg, sub_a, sub_b, sub_c);
           }
           auto wi_slice_c = sub_c.get_wi_data();
           for (int i = 0; i < wi_slice_c.length(); i++) {
             wi_slice_c[i] += 5.0;
           }
           joint_matrix_store(sg, sub_c,
                              accC.get_pointer() + (sg_startx * TM) * N +
                                  sg_starty / SG_SZ * TN,
                              N, matrix_layout::row_major);
         }); // parallel for
   }).wait();
}

static constexpr size_t MATRIX_M = TM * 2;
static constexpr size_t MATRIX_N = TN * 2;
static constexpr size_t MATRIX_K = TK * 2;
unsigned short A[MATRIX_M][MATRIX_K];
unsigned short B[MATRIX_K / 2][MATRIX_N * 2];
float C[MATRIX_M][MATRIX_N];
float D[MATRIX_M][MATRIX_N];

float make_fp32(short x) {
  unsigned int y = x;
  y = y << 16;
  float *res = reinterpret_cast<float *>(&y);
  return *res;
}

unsigned short make_bf16(float x) {
  int *res = reinterpret_cast<int *>(&x);
  *res = *res >> 16;
  return (unsigned short)*res;
}

void matrix_multiply_ref(int *A_mem, int *B_mem, int *C_mem, int M, int N,
                         int K) {
  // tiling
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        short *va = (short *)(A_mem + m * K + k);
        short *vb = (short *)(B_mem + k * N + n);
        float acc = *((float *)(C_mem + m * N + n));
        // FIXME: Should we do reduce-add in another version?
        for (int i = 0; i < 2; i++) {
          acc += (make_fp32(va[i]) * make_fp32(vb[i]));
        }
        *((float *)(C_mem + m * N + n)) = acc;
      }
      *((float *)(C_mem + m * N + n)) += 5.0;
    }
}

int main() {
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_K; j++) {
      A[i][j] = make_bf16(1.0f * (i + j));
    }
  }
  for (int i = 0; i < MATRIX_K / 2; i++) {
    for (int j = 0; j < MATRIX_N * 2; j++) {
      B[i][j] = make_bf16(2.0f * i + 3.0f * j);
    }
  }
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      C[i][j] = 1.0;
      D[i][j] = 1.0;
    }
  }

  big_matrix<float, MATRIX_M, MATRIX_N> MC((float *)&C);
  big_matrix<float, MATRIX_M, MATRIX_N> MD((float *)&D);
  big_matrix<unsigned short, MATRIX_M, MATRIX_K> MA((unsigned short *)&A);
  big_matrix<unsigned short, MATRIX_K / 2, MATRIX_N * 2> MB(
      (unsigned short *)&B);
  matrix_multiply(MC, MA, MB);
  matrix_multiply_ref((int32_t *)A, (int32_t *)B, (int32_t *)D, MATRIX_M,
                      MATRIX_N, MATRIX_K / 2);

  bool res = true;
  for (int i = 0; i < MATRIX_M; i++) {
    for (int j = 0; j < MATRIX_N; j++) {
      if (C[i][j] != D[i][j])
        res = false;
    }
  }
  if (res)
    std::cout << "passed\n";
  else
    std::cout << "failed\n";
}
