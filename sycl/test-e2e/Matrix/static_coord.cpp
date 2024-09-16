//==----------- get_coord_int8_matB.cpp  - DPC++ joint_matrix---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: matrix

// RUN: %{build} -o %t.out -DSYCL_EXT_ONEAPI_MATRIX_VERSION=4
// RUN: %{run} %t.out
// XFAIL: cpu

#include "common.hpp"
#include <iostream>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental::matrix;

constexpr size_t SG_SZ = 16; // try also 8 and 32
constexpr size_t TM = 8;
constexpr size_t TN = 16;
constexpr size_t TK = 16;

int main() {
  static constexpr size_t M = TM * 2;
  static constexpr size_t N = TN * 2;
  static constexpr size_t K = TK * 2;

  queue q;
  bfloat16 *A = malloc_shared<bfloat16>(M * K, q);
  bfloat16 *B = malloc_shared<bfloat16>(K * N, q);
  bfloat16 *vnniB = malloc_shared<bfloat16>(K * N, q);
  float *C = malloc_shared<float>(M * N, q);
  float *D = malloc_shared<float>(M * N, q);

  matrix_rand(M, K, A, (bfloat16)5);
  matrix_rand(K, N, B, (bfloat16)5);
  matrix_fill(M, N, D, (float)0);

  int A_wi_own[M][K];
  int rowd = (SG_SZ >= TK ? SG_SZ / TK : 1);
  int cold = (TK > SG_SZ ? TK / SG_SZ : 1);
  for (int row = 0; row < M; row++)
    for (int col = 0; col < K; col++) {
      unsigned int wi = (row % (rowd)) * TK + (((col % TK) / (cold)));
      A_wi_own[row][col] = wi;
    }
  std::cout << "A[" << M << "][" << K << "] WI id is a SG distribution is\n";
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < K; col++) {
      std::cout << "  " << A_wi_own[row][col];
    }
    std::cout << "\n";
  }

  int B_wi_own[K][N];
  int rowdn = (SG_SZ >= TN ? SG_SZ / TN : 1);
  int coldn = (TN > SG_SZ ? TN / SG_SZ : 1);

  for (int row = 0; row < K; row++)
    for (int col = 0; col < N; col++) {
      unsigned int wi = (row % (rowdn)) * TN + (col % TN) / coldn;
      B_wi_own[row][col] = wi;
    }

  std::cout << "B[" << K << "][" << N << "] WI id is a SG distribution is\n";
  for (int row = 0; row < K; row++) {
    for (int col = 0; col < N; col++) {
      std::cout << "  " << B_wi_own[row][col];
    }
    std::cout << "\n";
  }

  int C_wi_own[M][N];

  for (int row = 0; row < M; row++)
    for (int col = 0; col < N; col++) {
      unsigned int wi = (row % rowdn) * TN + (col % TN) / coldn;
      C_wi_own[row][col] = wi;
    }
  std::cout << "C[" << M << "][" << N << "] WI id is a SG distribution is\n";
  for (int row = 0; row < M; row++) {
    for (int col = 0; col < N; col++) {
      std::cout << "  " << C_wi_own[row][col];
    }
    std::cout << "\n";
  }
}
