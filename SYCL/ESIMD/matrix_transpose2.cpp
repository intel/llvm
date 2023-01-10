//==--------------- matrix_transpose2.cpp  - DPC++ ESIMD on-device test ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// FIXME: Investigate Windows-specific failures
// REQUIRES: TEMPORARY_DISABLED
// UNSUPPORTED: cuda || hip || gpu-intel-pvc
// TODO: esimd_emulator fails due to outdated __esimd_media_ld
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// This test checks matrix transpose implementation with media block read/write

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
template <typename T>
ESIMD_INLINE simd<T, 64> transpose_matrix(simd<T, 64> v1) {
  simd<T, 64> v2;
  // mask to control how to merge two vectors.
  simd_mask<16> mask = 0;
  mask.select<8, 2>(0) = 1;
  auto t1 = v1.template bit_cast_view<T, 4, 16>();
  auto t2 = v2.template bit_cast_view<T, 4, 16>();

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
  return v2;
}

// read a N-by-N sub-matrix
template <typename T, int N, typename AccessorTy>
ESIMD_INLINE simd<T, N * N> read(AccessorTy img, int MZ, int col, int row) {
  static_assert(N == 8, "only 8x8 sub-matrix is supported");

  simd<T, N * N> res;
  auto in = res.template bit_cast_view<unsigned char, 8, 32>();
  in = media_block_load<unsigned char, 8, 32>(img, col * sizeof(T), row);

  return res;
}

// write a N-by-N sub-matrix
template <typename T, int N, typename AccessorTy>
ESIMD_INLINE void write(AccessorTy img, int MZ, int col, int row,
                        simd<T, N * N> val) {
  static_assert(N == 8, "only 8x8 sub-matrix is supported");

  auto out = val.template bit_cast_view<uchar, 8, 32>();
  media_block_store<unsigned char, 8, 32>(img, col * sizeof(T), row, out);
}

// Square matrix transposition on block of size 8x8
// input and output are in the same image
template <typename AccessorInTy, typename AccessorOutTy>
ESIMD_INLINE void transpose8(AccessorInTy in, AccessorOutTy out, int MZ,
                             int block_col, int block_row) {
  static const int N = 8;

  if (block_row == block_col) {
    auto m1 = read<int, N, AccessorInTy>(in, MZ, N * block_col, N * block_row);

    // cerr << m1 << std::endl;

    auto t1 = transpose_matrix(m1);

    // cerr << t1 << std::endl;

    write<int, N, AccessorOutTy>(out, MZ, N * block_row, N * block_col, t1);
  } else if (block_row < block_col) {
    // Read two blocks to be swapped.
    auto m1 = read<int, N, AccessorInTy>(in, MZ, N * block_col, N * block_row);
    auto m2 = read<int, N, AccessorInTy>(in, MZ, N * block_row, N * block_col);

    // Tranpose them.
    auto t1 = transpose_matrix(m1);
    auto t2 = transpose_matrix(m2);

    // Write them back to the transposed location.
    write<int, N, AccessorOutTy>(out, MZ, N * block_row, N * block_col, t1);
    write<int, N, AccessorOutTy>(out, MZ, N * block_col, N * block_row, t2);
  }
}

// Square matrix transposition on block of size 16x16.
// In this version, a thread handle a block of size 16x16 which allows
// to better latency hidding and subsentantially improve overall performance.
//
template <typename AccessorInTy, typename AccessorOutTy>
ESIMD_INLINE void transpose16(AccessorInTy in, AccessorOutTy out, int MZ,
                              int block_col, int block_row) {
  static const int N = 16;

  if (block_row == block_col) {
    // Read a tile of 4 8x8 sub-blocks:
    //
    //  [ m00 m01 ]
    //  [ m10 m11 ]
    //
    // matrix<int, 8, 8> m00, m01, m10, m11, t00, t01, t10, t11;
    auto m00 = read<int, 8, AccessorInTy>(in, MZ, (N * block_col + 0),
                                          N * block_row + 0);
    auto m01 = read<int, 8, AccessorInTy>(in, MZ, (N * block_col + 8),
                                          N * block_row + 0);
    auto m10 = read<int, 8, AccessorInTy>(in, MZ, (N * block_col + 0),
                                          N * block_row + 8);
    auto m11 = read<int, 8, AccessorInTy>(in, MZ, (N * block_col + 8),
                                          N * block_row + 8);

    // Tranpose sub-blocks.
    auto t00 = transpose_matrix(m00);
    auto t01 = transpose_matrix(m01);
    auto t10 = transpose_matrix(m10);
    auto t11 = transpose_matrix(m11);

    // write out as
    //
    // [t00 t10]
    // [t01 t11]
    //
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_col + 0),
                                 N * block_row + 0, t00);
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_col + 8),
                                 N * block_row + 0, t10);
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_col + 0),
                                 N * block_row + 8, t01);
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_col + 8),
                                 N * block_row + 8, t11);
  } else if (block_row < block_col) {
    // Read two tiles of 4 8x8 sub-blocks to be swapped.
    //
    // matrix<int, 8, 8> a00, a01, a10, a11, t00, t01, t10, t11;
    auto a00 = read<int, 8, AccessorInTy>(in, MZ, (N * block_col + 0),
                                          N * block_row + 0);
    auto a01 = read<int, 8, AccessorInTy>(in, MZ, (N * block_col + 8),
                                          N * block_row + 0);
    auto a10 = read<int, 8, AccessorInTy>(in, MZ, (N * block_col + 0),
                                          N * block_row + 8);
    auto a11 = read<int, 8, AccessorInTy>(in, MZ, (N * block_col + 8),
                                          N * block_row + 8);

    // matrix<int, 8, 8> b00, b01, b10, b11, r00, r01, r10, r11;
    auto b00 = read<int, 8, AccessorInTy>(in, MZ, (N * block_row + 0),
                                          N * block_col + 0);
    auto b01 = read<int, 8, AccessorInTy>(in, MZ, (N * block_row + 8),
                                          N * block_col + 0);
    auto b10 = read<int, 8, AccessorInTy>(in, MZ, (N * block_row + 0),
                                          N * block_col + 8);
    auto b11 = read<int, 8, AccessorInTy>(in, MZ, (N * block_row + 8),
                                          N * block_col + 8);

    // Tranpose the first tile.
    auto t00 = transpose_matrix(a00);
    auto t01 = transpose_matrix(a01);
    auto t10 = transpose_matrix(a10);
    auto t11 = transpose_matrix(a11);

    // Tranpose the second tile.
    auto r00 = transpose_matrix(b00);
    auto r01 = transpose_matrix(b01);
    auto r10 = transpose_matrix(b10);
    auto r11 = transpose_matrix(b11);

    // Write the first tile to the transposed location.
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_row + 0),
                                 N * block_col + 0, t00);
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_row + 8),
                                 N * block_col + 0, t10);
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_row + 0),
                                 N * block_col + 8, t01);
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_row + 8),
                                 N * block_col + 8, t11);

    // Write the second tile to the transposed location.
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_col + 0),
                                 N * block_row + 0, r00);
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_col + 8),
                                 N * block_row + 0, r10);
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_col + 0),
                                 N * block_row + 8, r01);
    write<int, 8, AccessorOutTy>(out, MZ, (N * block_col + 8),
                                 N * block_row + 8, r11);
  }
}

bool runTest(unsigned MZ, unsigned block_size, unsigned num_iters,
             double &kernel_times, double &total_times) {
  queue q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler(),
          property::queue::enable_profiling{});
  int *M = new int[MZ * MZ];

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
      // make sure that image object has short live-range
      // than M
      sycl::image<2> imgM((unsigned int *)M, image_channel_order::rgba,
                          image_channel_type::unsigned_int32,
                          range<2>{MZ / 4, MZ});

      double etime = 0;
      if (block_size == 16 && MZ >= 16) {
        auto e = q.submit([&](handler &cgh) {
          auto accInput = imgM.get_access<uint4, access::mode::read>(cgh);
          auto accOutput = imgM.get_access<uint4, access::mode::write>(cgh);
          cgh.parallel_for<class K16>(
              Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                transpose16(accInput, accOutput, MZ, ndi.get_global_id(0),
                            ndi.get_global_id(1));
              });
        });
        e.wait();
        etime = esimd_test::report_time("kernel time", e, e);
      } else if (block_size == 8) {
        auto e = q.submit([&](handler &cgh) {
          auto accInput = imgM.get_access<uint4, access::mode::read>(cgh);
          auto accOutput = imgM.get_access<uint4, access::mode::write>(cgh);
          cgh.parallel_for<class K08>(
              Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                transpose8(accInput, accOutput, MZ, ndi.get_global_id(0),
                           ndi.get_global_id(1));
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
    delete[] M;
    return false; // not success
  }

  // End timer.
  double end = timer.Elapsed();

  total_times += (end - start) * 1000.0f;

  // printMatrix("\nTransposed matrix:", M, MZ);
  bool success = checkResult(M, MZ);
  delete[] M;
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
