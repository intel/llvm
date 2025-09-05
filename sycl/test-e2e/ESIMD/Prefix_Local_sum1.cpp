//==------- Prefix_Local_sum1.cpp  - DPC++ ESIMD on-device test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out 20

#include "esimd_test_utils.hpp"

#define MAX_TS_WIDTH 1024
// kernel can handle TUPLE_SZ 1, 2, or 4
#define TUPLE_SZ 1

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

using namespace sycl;
using namespace sycl::ext::intel::esimd;

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

// Local count : the local count stage partitions the table into chunks.
// Each chunk is 256*TUPLE_SZ.Each HW thread sums up values for each column
// within one chunk and stores the results in the last entry of the chunk
// All data chunks can be executed in parallel in this stage.
void cmk_sum_tuple_count(unsigned int *buf, unsigned int h_pos) {
  // h_pos indicates which 256-element chunk the kernel is processing

  // each thread handles PREFIX_ENTRIES entries. Each entry has 4 bins
  unsigned int offset = h_pos * PREFIX_ENTRIES * TUPLE_SZ;

  simd<unsigned, 32 * TUPLE_SZ> S, T;
#pragma unroll
  for (int i = 0; i < TUPLE_SZ; i++) {
    simd<unsigned, 32> data;
    data.copy_from(buf + offset + i * 32);
    S.select<32, 1>(i * 32) = data;
  }

#pragma unroll
  for (int i = 1; i < PREFIX_ENTRIES / 32; i++) {
#pragma unroll
    for (int j = 0; j < TUPLE_SZ; j++) {
      simd<unsigned, 32> data;
      data.copy_from(buf + offset + i * 32 * TUPLE_SZ + j * 32);
      T.select<32, 1>(j * 32) = data;
    }
    S += T;
  }

  // format S to be a 32xTUPLE_SZ matrix
  auto cnt_table = S.bit_cast_view<unsigned int, 32, TUPLE_SZ>();
  // sum reduction for each bin
  cnt_table.select<16, 1, TUPLE_SZ, 1>(0, 0) +=
      cnt_table.select<16, 1, TUPLE_SZ, 1>(16, 0);
  cnt_table.select<8, 1, TUPLE_SZ, 1>(0, 0) +=
      cnt_table.select<8, 1, TUPLE_SZ, 1>(8, 0);
  cnt_table.select<4, 1, TUPLE_SZ, 1>(0, 0) +=
      cnt_table.select<4, 1, TUPLE_SZ, 1>(4, 0);
  cnt_table.select<2, 1, TUPLE_SZ, 1>(0, 0) +=
      cnt_table.select<2, 1, TUPLE_SZ, 1>(2, 0);
  cnt_table.select<1, 1, TUPLE_SZ, 1>(0, 0) +=
      cnt_table.select<1, 1, TUPLE_SZ, 1>(1, 0);

  simd<unsigned, 8> voff(0, 1);        // 0, 1, 2, 3
  simd_mask<8> p = voff < TUPLE_SZ;    // predicate
  voff = (voff + ((h_pos + 1) * PREFIX_ENTRIES * TUPLE_SZ - TUPLE_SZ)) *
         sizeof(unsigned);
  scatter<unsigned, 8>(buf, voff, S.select<8, 1>(0), p);
}

//************************************
// Demonstrate summation of arrays both in scalar on CPU and parallel on device
// This is a ULT test variant of PrefixSum kernel with different implementation
// to increase test coverage of different usage cases and help isolate bugs.
// Difference from PrefixSum kernel:
// - Use copy_from<>() to read in data
// - Use scatter<>() to write output
//************************************
int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cout << "Usage: prefix [N]. N is 2^N entries x TUPLE_SZ" << std::endl;
    exit(1);
  }
  unsigned log2_element = atoi(argv[1]);
  unsigned int size = 1 << log2_element;

  sycl::range<2> LocalRange{1, 1};

  queue q = esimd_test::createQueue();

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<sycl::info::device::name>()
            << "\n";

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

  // compute local sum for every chunk of PREFIX_ENTRIES
  sycl::range<2> GlobalRange{size / PREFIX_ENTRIES, 1};

  // Start Timer
  esimd_test::Timer timer;
  double start;

  double kernel_times = 0;
  unsigned num_iters = 10;
  const bool profiling =
      q.has_property<sycl::property::queue::enable_profiling>();

  try {
    for (int iter = 0; iter <= num_iters; ++iter) {
      memcpy(pDeviceOutputs, pInputs, size * TUPLE_SZ * sizeof(unsigned int));
      auto e0 = q.submit([&](handler &cgh) {
        cgh.parallel_for<class Sum_tuple>(
            GlobalRange * LocalRange, [=](item<2> it) SYCL_ESIMD_KERNEL {
              cmk_sum_tuple_count(pDeviceOutputs, it.get_id(0));
            });
      });
      e0.wait();
      if (profiling) {
        double etime = esimd_test::report_time("kernel time", e0, e0);
        if (iter > 0)
          kernel_times += etime;
      }
      if (iter == 0)
        start = timer.Elapsed();
    }
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(pDeviceOutputs, q);
    free(pExpectOutputs);
    free(pInputs);
    return 1;
  }

  // End timer.
  double end = timer.Elapsed();

  esimd_test::display_timing_stats(profiling ? &kernel_times : nullptr,
                                   num_iters, (end - start) * 1000);

  bool pass = memcmp(pDeviceOutputs, pExpectOutputs,
                     size * TUPLE_SZ * sizeof(unsigned int)) == 0;
  std::cout << "Prefix " << (pass ? "=> PASSED" : "=> FAILED") << std::endl
            << std::endl;

  free(pDeviceOutputs, q);
  free(pExpectOutputs);
  free(pInputs);
  return 0;
}
