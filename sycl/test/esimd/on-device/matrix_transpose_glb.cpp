//==--------------- matrix_transpose_glb.cpp  - DPC++ ESIMD on-device test -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUN: %ESIMD_RUN_PLACEHOLDER %t.out
// XFAIL: linux
// UNSUPPORTED: cuda

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
  simd<uint16_t, 16> mask = 0;
  mask.select<8, 2>(0) = 1;
  auto t1 = GRF.template format<int, 48, 16>().select<4, 1, 16, 1>(InR >> 1, 0);
  auto t2 = GRF.template format<int, 48, 16>().select<4, 1, 16, 1>(OuR >> 1, 0);

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
}

// read a N-by-N sub-matrix
template <int N>
ESIMD_NOINLINE void read(int *buf, int MZ, int col, int row, int GrfIdx) {
  auto res = GRF.select<N * N, 1>(GrfIdx * 8);
  buf += row * MZ + col;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    res.template select<N, 1>(i * N) = block_load<int, N>(buf);
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
    block_store<int, N>(buf, val.template select<N, 1>(i * N));
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
    success &= runTest(1U << 10, 8);
    success &= runTest(1U << 11, 8);
    success &= runTest(1U << 12, 8);
    // success &= runTest(1U << 13, 8);
    success &= runTest(1U << 10, 16);
    success &= runTest(1U << 11, 16);
    success &= runTest(1U << 12, 16);
    // success &= runTest(1U << 13, 16);
  }

  cerr << (success ? "PASSED\n" : "FAILED\n");
  return !success;
}
