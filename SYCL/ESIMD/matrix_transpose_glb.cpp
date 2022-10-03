//==--------------- matrix_transpose_glb.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Temporarily disable while the failure is being investigated.
// UNSUPPORTED: windows

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace std;
using namespace sycl::ext::intel::esimd;

const unsigned int ESIMD_EMULATOR_SIZE_LIMIT = 1U << 10;

void initMatrix(int *M, unsigned N) {
  assert(N >= 8 && (((N - 1) & N) == 0) &&
         "only power of 2 (>= 16) is supported");
  for (unsigned i = 0; i < N * N; ++i)
    M[i] = i;
}

void printMatrix(const char *msg, int *M, unsigned N) {
  cerr << msg << "\n";
  if (N > 64) {
    cerr << "<<<maitrix of size " << N << " x " << N << ">>>\n";
    return;
  }

  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      cerr.width(4);
      cerr << M[i * N + j] << "  ";
    }
    cerr << "\n";
  }
}

bool checkResult(const int *M, unsigned N) {
  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; j < N; ++j) {
      unsigned t = M[j * N + i];
      if (t != i * N + j) {
        cerr << "Error at M(" << i << ", " << j << ") = " << t << "\n";
        return false;
      }
    }
  }
  return true;
}

// This represents the register file that kernels operate on.
// The max size we need is for 3 16x16 matrices
ESIMD_PRIVATE ESIMD_REGISTER(192) simd<int, 3 * 32 * 8> GRF;

// The basic idea of vecotrizing transposition can be illustrated by
// transposing a 2 x 2 matrix as follows:
//
// A B
// C D
// ==>
// merge([A, A, B, B], [C, C, D, D], 0b1010) = [A, C, B, D]
// ==>
// A C
// B D
//
// This function does 8x8 transposition
ESIMD_NOINLINE void transpose_matrix(int InR, int OuR) {
  // mask to control how to merge two vectors.
  simd_mask<16> mask = 0;
  mask.select<8, 2>(0) = 1;
  auto t1 = GRF.template bit_cast_view<int, 48, 16>().select<4, 1, 16, 1>(
      InR >> 1, 0);
  auto t2 = GRF.template bit_cast_view<int, 48, 16>().select<4, 1, 16, 1>(
      OuR >> 1, 0);

  // j = 1
  t2.row(0).merge(t1.template replicate_vs_w_hs<8, 1, 2, 0>(0, 0),
                  t1.template replicate_vs_w_hs<8, 1, 2, 0>(2, 0), mask);
  t2.row(1).merge(t1.template replicate_vs_w_hs<8, 1, 2, 0>(0, 8),
                  t1.template replicate_vs_w_hs<8, 1, 2, 0>(2, 8), mask);
  t2.row(2).merge(t1.template replicate_vs_w_hs<8, 1, 2, 0>(1, 0),
                  t1.template replicate_vs_w_hs<8, 1, 2, 0>(3, 0), mask);
  t2.row(3).merge(t1.template replicate_vs_w_hs<8, 1, 2, 0>(1, 8),
                  t1.template replicate_vs_w_hs<8, 1, 2, 0>(3, 8), mask);

  // j = 2
  t1.row(0).merge(t2.template replicate_vs_w_hs<8, 1, 2, 0>(0, 0),
                  t2.template replicate_vs_w_hs<8, 1, 2, 0>(2, 0), mask);
  t1.row(1).merge(t2.template replicate_vs_w_hs<8, 1, 2, 0>(0, 8),
                  t2.template replicate_vs_w_hs<8, 1, 2, 0>(2, 8), mask);
  t1.row(2).merge(t2.template replicate_vs_w_hs<8, 1, 2, 0>(1, 0),
                  t2.template replicate_vs_w_hs<8, 1, 2, 0>(3, 0), mask);
  t1.row(3).merge(t2.template replicate_vs_w_hs<8, 1, 2, 0>(1, 8),
                  t2.template replicate_vs_w_hs<8, 1, 2, 0>(3, 8), mask);

  // j = 4
  t2.row(0).merge(t1.template replicate_vs_w_hs<8, 1, 2, 0>(0, 0),
                  t1.template replicate_vs_w_hs<8, 1, 2, 0>(2, 0), mask);
  t2.row(1).merge(t1.template replicate_vs_w_hs<8, 1, 2, 0>(0, 8),
                  t1.template replicate_vs_w_hs<8, 1, 2, 0>(2, 8), mask);
  t2.row(2).merge(t1.template replicate_vs_w_hs<8, 1, 2, 0>(1, 0),
                  t1.template replicate_vs_w_hs<8, 1, 2, 0>(3, 0), mask);
  t2.row(3).merge(t1.template replicate_vs_w_hs<8, 1, 2, 0>(1, 8),
                  t1.template replicate_vs_w_hs<8, 1, 2, 0>(3, 8), mask);
}

// read a N-by-N sub-matrix
template <int N>
ESIMD_NOINLINE void read(int *buf, int MZ, int col, int row, int GrfIdx) {
  auto res = GRF.select<N * N, 1>(GrfIdx * 8);
  buf += row * MZ + col;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    simd<int, N> data;
    data.copy_from(buf);
    res.template select<N, 1>(i * N) = data;
    buf += MZ;
  }
}

// write a N-by-N sub-matrix
template <int N>
ESIMD_NOINLINE void write(int *buf, int MZ, int col, int row, int GrfIdx) {
  auto val = GRF.select<N * N, 1>(GrfIdx * 8);
  buf += row * MZ + col;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    simd<int, N> val2 = val.template select<N, 1>(i * N);
    val2.copy_to(buf);
    buf += MZ;
  }
}

// Square matrix transposition on block of size 8x8
// input and output are in the same buffer
ESIMD_INLINE void transpose8(int *buf, int MZ, int block_col, int block_row) {
  static const int N = 8;

  if (block_row == block_col) {
    read<N>(buf, MZ, N * block_col, N * block_row, 0); // to grf line#0

    transpose_matrix(0, 8); // to grf line#8

    write<N>(buf, MZ, N * block_row, N * block_col, 8); // from grf line#8
  } else if (block_row < block_col) {
    // Read two blocks to be swapped. line #0 and line #8
    read<N>(buf, MZ, N * block_col, N * block_row, 0);
    read<N>(buf, MZ, N * block_row, N * block_col, 8);

    // Tranpose them.
    transpose_matrix(0, 16);
    transpose_matrix(8, 24);

    // Write them back to the transposed location.
    write<N>(buf, MZ, N * block_row, N * block_col, 16);
    write<N>(buf, MZ, N * block_col, N * block_row, 24);
  }
}

// Square matrix transposition on block of size 16x16.
// In this version, a thread handle a block of size 16x16 which allows
// to better latency hidding and subsentantially improve overall performance.
//
ESIMD_INLINE void transpose16(int *buf, int MZ, int block_col, int block_row) {
  static const int N = 16;

  if (block_row == block_col) {
    // Read a tile of 4 8x8 sub-blocks to
    //
    //  [ line00-07 line08-15 ]
    //  [ line16-23 line24-32 ]
    //
    read<8>(buf, MZ, (N * block_col + 0), N * block_row + 0, 0);
    read<8>(buf, MZ, (N * block_col + 8), N * block_row + 0, 8);
    read<8>(buf, MZ, (N * block_col + 0), N * block_row + 8, 16);
    read<8>(buf, MZ, (N * block_col + 8), N * block_row + 8, 24);

    // Tranpose sub-blocks.
    transpose_matrix(0, 32);  // to line#32
    transpose_matrix(8, 40);  // to line#40
    transpose_matrix(16, 48); // to line#48
    transpose_matrix(24, 56); // to line#56

    // write out as
    //
    // [line32-39 line48-55]
    // [line40-47 line56-63]
    //
    write<8>(buf, MZ, (N * block_col + 0), N * block_row + 0, 32);
    write<8>(buf, MZ, (N * block_col + 8), N * block_row + 0, 48);
    write<8>(buf, MZ, (N * block_col + 0), N * block_row + 8, 40);
    write<8>(buf, MZ, (N * block_col + 8), N * block_row + 8, 56);
  } else if (block_row < block_col) {
    // Read two tiles of 4 8x8 sub-blocks to be swapped.
    read<8>(buf, MZ, (N * block_col + 0), N * block_row + 0, 0);
    read<8>(buf, MZ, (N * block_col + 8), N * block_row + 0, 8);
    read<8>(buf, MZ, (N * block_col + 0), N * block_row + 8, 16);
    read<8>(buf, MZ, (N * block_col + 8), N * block_row + 8, 24);

    read<8>(buf, MZ, (N * block_row + 0), N * block_col + 0, 32);
    read<8>(buf, MZ, (N * block_row + 8), N * block_col + 0, 40);
    read<8>(buf, MZ, (N * block_row + 0), N * block_col + 8, 48);
    read<8>(buf, MZ, (N * block_row + 8), N * block_col + 8, 56);

    // Tranpose the first tile.
    transpose_matrix(0, 64);  //(a00);
    transpose_matrix(8, 72);  //(a01);
    transpose_matrix(16, 80); //(a10);
    transpose_matrix(24, 88); //(a11);

    // Tranpose the second tile.
    transpose_matrix(32, 0);  //(b00);
    transpose_matrix(40, 8);  //(b01);
    transpose_matrix(48, 16); //(b10);
    transpose_matrix(56, 24); //(b11);

    // Write the first tile to the transposed location.
    write<8>(buf, MZ, (N * block_row + 0), N * block_col + 0, 64);
    write<8>(buf, MZ, (N * block_row + 8), N * block_col + 0, 80);
    write<8>(buf, MZ, (N * block_row + 0), N * block_col + 8, 72);
    write<8>(buf, MZ, (N * block_row + 8), N * block_col + 8, 88);

    // Write the second tile to the transposed location.
    write<8>(buf, MZ, (N * block_col + 0), N * block_row + 0, 0);
    write<8>(buf, MZ, (N * block_col + 8), N * block_row + 0, 16);
    write<8>(buf, MZ, (N * block_col + 0), N * block_row + 8, 8);
    write<8>(buf, MZ, (N * block_col + 8), N * block_row + 8, 24);
  }
}

bool runTest(unsigned MZ, unsigned block_size, unsigned num_iters,
             double &kernel_times, double &total_times) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler(),
          property::queue::enable_profiling{});
  int *M = malloc_shared<int>(MZ * MZ, q);

  initMatrix(M, MZ);
  cerr << "\nTranspose square matrix of size " << MZ << "\n";
  // printMatrix("Initial matrix:", M, MZ);

  if ((q.get_backend() == sycl::backend::ext_intel_esimd_emulator) &&
      (MZ > ESIMD_EMULATOR_SIZE_LIMIT)) {
    cerr << "Matrix Size larger than " << ESIMD_EMULATOR_SIZE_LIMIT
         << " is skipped"
         << "\n";
    cerr << "for esimd_emulator backend due to timeout"
         << "\n";
    return true;
  }

  // Each C-for-Metal thread works on one or two blocks of size 8 x 8.
  int thread_width = MZ / block_size;
  int thread_height = MZ / block_size;

  // create ranges
  // We need that many workitems
  auto GlobalRange = range<2>(thread_width, thread_height);

  // Number of workitems in a workgroup
  range<2> LocalRange{1, 1};
  nd_range<2> Range(GlobalRange, LocalRange);

  // Start timer.
  esimd_test::Timer timer;
  double start;

  // Launches the task on the GPU.

  try {
    // num_iters + 1, iteration#0 is for warmup
    for (int i = 0; i <= num_iters; ++i) {
      double etime = 0;
      if (block_size == 16 && MZ >= 16) {
        auto e = q.submit([&](handler &cgh) {
          cgh.parallel_for<class Transpose16>(
              Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                transpose16(M, MZ, ndi.get_global_id(0), ndi.get_global_id(1));
              });
        });
        e.wait();
        etime = esimd_test::report_time("kernel time", e, e);
      } else if (block_size == 8) {
        auto e = q.submit([&](handler &cgh) {
          cgh.parallel_for<class Transpose08>(
              Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                transpose8(M, MZ, ndi.get_global_id(0), ndi.get_global_id(1));
              });
        });
        e.wait();
        etime = esimd_test::report_time("kernel time", e, e);
      }

      if (i > 0)
        kernel_times += etime;
      else
        start = timer.Elapsed();
    }
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(M, q);
    return false; // not success
  }

  // End timer.
  double end = timer.Elapsed();

  total_times += (end - start) * 1000.0f;

  // printMatrix("\nTransposed matrix:", M, MZ);
  bool success = checkResult(M, MZ);
  free(M, q);
  return success;
}

int main(int argc, char *argv[]) {
  unsigned MZ = 1U << 5;
  if (argc >= 2) {
    unsigned exponent = atoi(argv[1]);
    MZ = (MZ > (1U << exponent)) ? MZ : (1U << exponent);
    MZ = (MZ < (1U << 12)) ? MZ : (1U << 12);
  }

  unsigned num_iters = 10;
  double kernel_times = 0;
  double total_times = 0;

  bool success = true;
  success &= runTest(MZ, 16, num_iters, kernel_times, total_times);
  if (argc == 1) {
    success &= runTest(1U << 10, 8, num_iters, kernel_times, total_times);
    success &= runTest(1U << 11, 8, num_iters, kernel_times, total_times);
    success &= runTest(1U << 12, 8, num_iters, kernel_times, total_times);
    // success &= runTest(1U << 13, 8, num_iters, kernel_times, total_times);
    success &= runTest(1U << 10, 16, num_iters, kernel_times, total_times);
    success &= runTest(1U << 11, 16, num_iters, kernel_times, total_times);
    success &= runTest(1U << 12, 16, num_iters, kernel_times, total_times);
    // success &= runTest(1U << 13, 16, num_iters, kernel_times, total_times);
  }

  esimd_test::display_timing_stats(kernel_times, num_iters, total_times);

  cerr << (success ? "PASSED\n" : "FAILED\n");
  return !success;
}
