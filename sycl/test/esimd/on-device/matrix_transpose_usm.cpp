//==--------------- matrix_transpose_usm.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

#ifdef __linux__
#include <libgen.h>
#include <sys/time.h>
#include <unistd.h>
#elif defined(_WIN32) || defined(WIN32)
#include <Windows.h>
#endif

double getTimeStamp() {
#ifdef __linux__
  {
    struct timeval t;
    if (gettimeofday(&t, 0) != 0) {
      fprintf(stderr, "Linux-specific time measurement counter (gettimeofday) "
                      "is not available.\n");
      std::exit(1);
    }
    return t.tv_sec + t.tv_usec / 1e6;
  }
#elif defined(_WIN32) || defined(WIN32)
  {
    LARGE_INTEGER curclock;
    LARGE_INTEGER freq;
    if (!QueryPerformanceCounter(&curclock) ||
        !QueryPerformanceFrequency(&freq)) {
      fprintf(stderr, "Windows - specific time measurement "
                      "counter(QueryPerformanceCounter, "
                      "QueryPerformanceFrequency) is not available.\n");
      std::exit(1);
    }
    return double(curclock.QuadPart) / freq.QuadPart;
  }
#else
  {
    fprintf(stderr, "Unsupported platform.\n");
    std::exit(1);
  }
#endif
}

using namespace cl::sycl;
using namespace std;
using namespace sycl::INTEL::gpu;

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

// return msecond
static double report_time(const string &msg, event e) {
  cl_ulong time_start =
      e.get_profiling_info<info::event_profiling::command_start>();
  cl_ulong time_end =
      e.get_profiling_info<info::event_profiling::command_end>();
  double elapsed = (time_end - time_start) / 1e6;
  // cerr << msg << elapsed << " msecs" << std::endl;
  return elapsed;
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
  simd<uint16_t, 16> mask = 0;
  mask.select<8, 2>(0) = 1;
  auto t1 = v1.template format<T, 4, 16>();
  auto t2 = v2.template format<T, 4, 16>();

  // j = 1
  t2.row(0).merge(t1.template replicate<8, 1, 2, 0>(0, 0),
                  t1.template replicate<8, 1, 2, 0>(2, 0), mask);
  t2.row(1).merge(t1.template replicate<8, 1, 2, 0>(0, 8),
                  t1.template replicate<8, 1, 2, 0>(2, 8), mask);
  t2.row(2).merge(t1.template replicate<8, 1, 2, 0>(1, 0),
                  t1.template replicate<8, 1, 2, 0>(3, 0), mask);
  t2.row(3).merge(t1.template replicate<8, 1, 2, 0>(1, 8),
                  t1.template replicate<8, 1, 2, 0>(3, 8), mask);

  // j = 2
  t1.row(0).merge(t2.template replicate<8, 1, 2, 0>(0, 0),
                  t2.template replicate<8, 1, 2, 0>(2, 0), mask);
  t1.row(1).merge(t2.template replicate<8, 1, 2, 0>(0, 8),
                  t2.template replicate<8, 1, 2, 0>(2, 8), mask);
  t1.row(2).merge(t2.template replicate<8, 1, 2, 0>(1, 0),
                  t2.template replicate<8, 1, 2, 0>(3, 0), mask);
  t1.row(3).merge(t2.template replicate<8, 1, 2, 0>(1, 8),
                  t2.template replicate<8, 1, 2, 0>(3, 8), mask);

  // j = 4
  t2.row(0).merge(t1.template replicate<8, 1, 2, 0>(0, 0),
                  t1.template replicate<8, 1, 2, 0>(2, 0), mask);
  t2.row(1).merge(t1.template replicate<8, 1, 2, 0>(0, 8),
                  t1.template replicate<8, 1, 2, 0>(2, 8), mask);
  t2.row(2).merge(t1.template replicate<8, 1, 2, 0>(1, 0),
                  t1.template replicate<8, 1, 2, 0>(3, 0), mask);
  t2.row(3).merge(t1.template replicate<8, 1, 2, 0>(1, 8),
                  t1.template replicate<8, 1, 2, 0>(3, 8), mask);
  return v2;
}

// read a N-by-N sub-matrix
template <typename T, int N>
ESIMD_INLINE simd<T, N * N> read(T *buf, int MZ, int col, int row) {
  simd<T, N * N> res;
  buf += row * MZ + col;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    res.template select<N, 1>(i * N) = block_load<T, N>(buf);
    buf += MZ;
  }
  return res;
}

// write a N-by-N sub-matrix
template <typename T, int N>
ESIMD_INLINE void write(T *buf, int MZ, int col, int row, simd<T, N * N> val) {
  buf += row * MZ + col;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    block_store<T, N>(buf, val.template select<N, 1>(i * N));
    buf += MZ;
  }
}

// Square matrix transposition on block of size 8x8
// input and output are in the same buffer
ESIMD_INLINE void transpose8(int *buf, int MZ, int block_col, int block_row) {
  static const int N = 8;

  if (block_row == block_col) {
    auto m1 = read<int, N>(buf, MZ, N * block_col, N * block_row);

    // cerr << m1 << std::endl;

    auto t1 = transpose_matrix(m1);

    // cerr << t1 << std::endl;

    write<int, N>(buf, MZ, N * block_row, N * block_col, t1);
  } else if (block_row < block_col) {
    // Read two blocks to be swapped.
    auto m1 = read<int, N>(buf, MZ, N * block_col, N * block_row);
    auto m2 = read<int, N>(buf, MZ, N * block_row, N * block_col);

    // Tranpose them.
    auto t1 = transpose_matrix(m1);
    auto t2 = transpose_matrix(m2);

    // Write them back to the transposed location.
    write<int, N>(buf, MZ, N * block_row, N * block_col, t1);
    write<int, N>(buf, MZ, N * block_col, N * block_row, t2);
  }
}

// Square matrix transposition on block of size 16x16.
// In this version, a thread handle a block of size 16x16 which allows
// to better latency hidding and subsentantially improve overall performance.
//
ESIMD_INLINE void transpose16(int *buf, int MZ, int block_col, int block_row) {
  static const int N = 16;

  if (block_row == block_col) {
    // Read a tile of 4 8x8 sub-blocks:
    //
    //  [ m00 m01 ]
    //  [ m10 m11 ]
    //
    // matrix<int, 8, 8> m00, m01, m10, m11, t00, t01, t10, t11;
    auto m00 = read<int, 8>(buf, MZ, (N * block_col + 0), N * block_row + 0);
    auto m01 = read<int, 8>(buf, MZ, (N * block_col + 8), N * block_row + 0);
    auto m10 = read<int, 8>(buf, MZ, (N * block_col + 0), N * block_row + 8);
    auto m11 = read<int, 8>(buf, MZ, (N * block_col + 8), N * block_row + 8);

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
    write<int, 8>(buf, MZ, (N * block_col + 0), N * block_row + 0, t00);
    write<int, 8>(buf, MZ, (N * block_col + 8), N * block_row + 0, t10);
    write<int, 8>(buf, MZ, (N * block_col + 0), N * block_row + 8, t01);
    write<int, 8>(buf, MZ, (N * block_col + 8), N * block_row + 8, t11);
  } else if (block_row < block_col) {
    // Read two tiles of 4 8x8 sub-blocks to be swapped.
    //
    // matrix<int, 8, 8> a00, a01, a10, a11, t00, t01, t10, t11;
    auto a00 = read<int, 8>(buf, MZ, (N * block_col + 0), N * block_row + 0);
    auto a01 = read<int, 8>(buf, MZ, (N * block_col + 8), N * block_row + 0);
    auto a10 = read<int, 8>(buf, MZ, (N * block_col + 0), N * block_row + 8);
    auto a11 = read<int, 8>(buf, MZ, (N * block_col + 8), N * block_row + 8);

    // matrix<int, 8, 8> b00, b01, b10, b11, r00, r01, r10, r11;
    auto b00 = read<int, 8>(buf, MZ, (N * block_row + 0), N * block_col + 0);
    auto b01 = read<int, 8>(buf, MZ, (N * block_row + 8), N * block_col + 0);
    auto b10 = read<int, 8>(buf, MZ, (N * block_row + 0), N * block_col + 8);
    auto b11 = read<int, 8>(buf, MZ, (N * block_row + 8), N * block_col + 8);

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
    write<int, 8>(buf, MZ, (N * block_row + 0), N * block_col + 0, t00);
    write<int, 8>(buf, MZ, (N * block_row + 8), N * block_col + 0, t10);
    write<int, 8>(buf, MZ, (N * block_row + 0), N * block_col + 8, t01);
    write<int, 8>(buf, MZ, (N * block_row + 8), N * block_col + 8, t11);

    // Write the second tile to the transposed location.
    write<int, 8>(buf, MZ, (N * block_col + 0), N * block_row + 0, r00);
    write<int, 8>(buf, MZ, (N * block_col + 8), N * block_row + 0, r10);
    write<int, 8>(buf, MZ, (N * block_col + 0), N * block_row + 8, r01);
    write<int, 8>(buf, MZ, (N * block_col + 8), N * block_row + 8, r11);
  }
}

bool runTest(unsigned MZ, unsigned block_size) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(),
          property::queue::enable_profiling{});
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  int *M = static_cast<int *>(malloc_shared(MZ * MZ * sizeof(int), dev, ctxt));

  initMatrix(M, MZ);
  cerr << "\nTranspose square matrix of size " << MZ << "\n";
  // printMatrix("Initial matrix:", M, MZ);

  // Each C-for-Metal thread works on one or two blocks of size 8 x 8.
  int thread_width = MZ / block_size;
  int thread_height = MZ / block_size;

  // create ranges
  // We need that many workitems
  auto GlobalRange = cl::sycl::range<2>(thread_width, thread_height);

  // Number of workitems in a workgroup
  cl::sycl::range<2> LocalRange{1, 1};
  cl::sycl::nd_range<2> Range(GlobalRange, LocalRange);

  // Start timer.
  double start = getTimeStamp();

  // Launches the task on the GPU.
  double kernel_times = 0;
  unsigned num_iters = 10;

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
      etime = report_time("kernel time", e);
    } else if (block_size == 8) {
      auto e = q.submit([&](handler &cgh) {
        cgh.parallel_for<class Transpose08>(
            Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
              transpose8(M, MZ, ndi.get_global_id(0), ndi.get_global_id(1));
            });
      });
      e.wait();
      etime = report_time("kernel time", e);
    }

    if (i > 0)
      kernel_times += etime;
  }

  // End timer.
  double end = getTimeStamp();

  float total_time = (end - start) * 1000.0f / num_iters;
  float kernel_time = kernel_times / num_iters;

  float bandwidth_total =
      2.0f * 1000 * sizeof(int) * MZ * MZ / (1024 * 1024 * 1024) / total_time;
  float bandwidth_kernel =
      2.0f * 1000 * sizeof(int) * MZ * MZ / (1024 * 1024 * 1024) / kernel_time;

  cerr << "GPU transposition time = " << total_time << " msec\n";
  cerr << "GPU transposition bandwidth = " << bandwidth_total << " GB/s\n";
  cerr << "GPU kernel time = " << kernel_time << " msec\n";
  cerr << "GPU kernel bandwidth = " << bandwidth_kernel << " GB/s\n";

  // printMatrix("\nTransposed matrix:", M, MZ);
  bool success = checkResult(M, MZ);
  free(M, ctxt);
  return success;
}

int main(int argc, char *argv[]) {
  unsigned MZ = 1U << 5;
  if (argc >= 2) {
    unsigned exponent = atoi(argv[1]);
    MZ = (MZ > (1U << exponent)) ? MZ : (1U << exponent);
    MZ = (MZ < (1U << 12)) ? MZ : (1U << 12);
  }

  bool success = true;
  success &= runTest(MZ, 16);
  if (argc == 1) {
    success &= runTest(1U << 7, 8);
    success &= runTest(1U << 8, 8);
    success &= runTest(1U << 9, 8);
    // success &= runTest(1U << 13, 8);
    success &= runTest(1U << 7, 16);
    success &= runTest(1U << 8, 16);
    success &= runTest(1U << 9, 16);
    // success &= runTest(1U << 13, 16);
  }

  cerr << (success ? "PASSED\n" : "FAILED\n");
  return !success;
}
