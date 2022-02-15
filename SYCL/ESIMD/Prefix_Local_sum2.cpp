//==------- Prefix_Local_sum2.cpp  - DPC++ ESIMD on-device test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out 20

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <sycl/ext/intel/experimental/esimd.hpp>

#define MAX_TS_WIDTH 1024
// kernel can handle TUPLE_SZ 1, 2, or 4
#define TUPLE_SZ 4

#if TUPLE_SZ == 1
#define GATHER_SCATTER_MASK rgba_channel_mask::R
#elif TUPLE_SZ == 2
#define GATHER_SCATTER_MASK rgba_channel_mask::GR
#elif TUPLE_SZ == 4
#define GATHER_SCATTER_MASK rgba_channel_mask::ABGR
#endif

#define PREFIX_ENTRIES 256
#define PREFIX_ENTRIES_LOW 32
#define ENTRIES_THRESHOLD 2048
// minimum number of threads to launch a kernel (power of 2)
#define MIN_NUM_THREADS 1

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

void compute_local_prefixsum(const unsigned int input[],
                             unsigned int prefixSum[], unsigned int size) {

  memcpy(prefixSum, input, size * TUPLE_SZ * sizeof(unsigned));
  unsigned local_sum[TUPLE_SZ];
  for (unsigned k = 0; k < size; k += PREFIX_ENTRIES) {
    memset(local_sum, 0, TUPLE_SZ * sizeof(unsigned)); // init 0

    for (int i = 0; i < PREFIX_ENTRIES; i++) {
      for (int j = 0; j < TUPLE_SZ; j++) {
        local_sum[j] += input[(k + i) * TUPLE_SZ + j];
      }
    }
    // store local_sum in the last entry
    memcpy(&prefixSum[(k + PREFIX_ENTRIES - 1) * TUPLE_SZ], local_sum,
           TUPLE_SZ * sizeof(unsigned));
  }
}

void cmk_acum_iterative(unsigned *buf, unsigned h_pos,
                        unsigned int stride_elems,
                        unsigned int stride_threads) {

  simd<unsigned int, 32> element_offset(0, 1); // 0, 1, 2, ..., 31

  // global offset for a thread
  unsigned int global_offset = (h_pos * stride_threads * TUPLE_SZ);
  // element offsets for scattered read: [e0,e1,e2,...,e31] where e_i =
  // global_offset + # prefix_entries + prefix_entries - 1;
  element_offset =
      (((element_offset + 1) * stride_elems - 1) * TUPLE_SZ + global_offset) *
      sizeof(unsigned);

  simd<unsigned int, 32 * TUPLE_SZ> S, T;

  S = gather_rgba<unsigned int, 32, GATHER_SCATTER_MASK>(buf, element_offset);

#pragma unroll
  for (int i = 1; i < PREFIX_ENTRIES / 32; i++) {
    element_offset += (stride_elems * 32 * TUPLE_SZ) * sizeof(unsigned);
    // scattered read, each inst reads 16 entries
    T = gather_rgba<unsigned int, 32, GATHER_SCATTER_MASK>(buf, element_offset);
    S += T;
  }

  auto cnt_table = S.bit_cast_view<unsigned int, TUPLE_SZ, 32>();

  simd<unsigned, TUPLE_SZ> sum;
#pragma unroll
  for (int i = 0; i < TUPLE_SZ; i++) {
    simd<unsigned, 32> t = cnt_table.row(i);
    sum.select<1, 1>(i) = reduce<int>(t, std::plus<>());
  }

  simd<unsigned, 8> result = 0;
  result.select<TUPLE_SZ, 1>(0) = sum;
  simd<unsigned, 8> voff(0, 1);        // 0, 1, 2, 3
  simd_mask<8> p = voff < TUPLE_SZ;    // predicate
  voff = (voff + (global_offset + stride_threads * TUPLE_SZ - TUPLE_SZ)) *
         sizeof(unsigned);
  scatter<unsigned, 8>(buf, voff, result, p);
}

//************************************
// Demonstrate summation of arrays both in scalar on CPU and parallel on device
// This is a ULT test variant of PrefixSum kernel with different implementation
// to increase test coverage of different usage cases and help isolate bugs.
// Difference from PrefixSum kernel:
// - Use gather4<>() to read in data
// - Use bit_cast_view<>() to convert a 1D vector to 2D matrix view
// - Use reduce<int>(t, std::plus<>()) to do reduction
//************************************
int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cout << "Usage: prefix [N]. N is 2^N entries x TUPLE_SZ" << std::endl;
    exit(1);
  }
  unsigned log2_element = atoi(argv[1]);
  unsigned int size = 1 << log2_element;

  cl::sycl::range<2> LocalRange{1, 1};

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(),
          property::queue::enable_profiling{});

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  // allocate and initialized input
  unsigned int *pInputs = static_cast<unsigned int *>(
      malloc(size * TUPLE_SZ * sizeof(unsigned int)));
  for (unsigned int i = 0; i < size * TUPLE_SZ; ++i) {
    pInputs[i] = rand() % 128;
  }

  // allocate device buffer
  unsigned int *pDeviceOutputs =
      malloc_shared<unsigned int>(size * TUPLE_SZ, q);

  // allocate & compute expected result
  unsigned int *pExpectOutputs = static_cast<unsigned int *>(
      malloc(size * TUPLE_SZ * sizeof(unsigned int)));
  compute_local_prefixsum(pInputs, pExpectOutputs, size);

  // Start Timer
  esimd_test::Timer timer;
  double start;

  double kernel_times = 0;
  unsigned num_iters = 10;

  try {
    for (int iter = 0; iter <= num_iters; ++iter) {
      memcpy(pDeviceOutputs, pInputs, size * TUPLE_SZ * sizeof(unsigned int));
      auto e1 = q.submit([&](handler &cgh) {
        cgh.parallel_for<class Accum_iterative>(
            range<2>{size / PREFIX_ENTRIES, 1} * LocalRange,
            [=](item<2> it) SYCL_ESIMD_KERNEL {
              cmk_acum_iterative(pDeviceOutputs, it.get_id(0), 1,
                                 PREFIX_ENTRIES);
            });
      });
      e1.wait();
      double etime = esimd_test::report_time("kernel time", e1, e1);
      if (iter > 0)
        kernel_times += etime;
      else
        start = timer.Elapsed();
    }
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(pDeviceOutputs, q);
    free(pExpectOutputs);
    free(pInputs);
    return 1;
  }

  // End timer.
  double end = timer.Elapsed();

  esimd_test::display_timing_stats(kernel_times, num_iters,
                                   (end - start) * 1000);

  bool pass = memcmp(pDeviceOutputs, pExpectOutputs,
                     size * TUPLE_SZ * sizeof(unsigned int)) == 0;
  std::cout << "Prefix " << (pass ? "=> PASSED" : "=> FAILED") << std::endl
            << std::endl;

  free(pDeviceOutputs, q);
  free(pExpectOutputs);
  free(pInputs);
  return 0;
}
