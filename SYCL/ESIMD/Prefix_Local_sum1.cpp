//==------- Prefix_Local_sum1.cpp  - DPC++ ESIMD on-device test -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu
// UNSUPPORTED: cuda || rocm
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out 20
// RUN: %GPU_RUN_PLACEHOLDER %t.out 20

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

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

using namespace cl::sycl;
using namespace sycl::ext::intel::experimental::esimd;

void compute_local_prefixsum(unsigned int input[], unsigned int prefixSum[],
                             unsigned int size) {

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
  simd<ushort, 8> p = voff < TUPLE_SZ; // predicate
  voff = (voff + ((h_pos + 1) * PREFIX_ENTRIES * TUPLE_SZ - TUPLE_SZ)) *
         sizeof(unsigned);
  scatter<unsigned, 8>(buf, S.select<8, 1>(0), voff, p);
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

  unsigned int *pInputs;
  if (argc < 2) {
    std::cout << "Usage: prefix [N]. N is 2^N entries x TUPLE_SZ" << std::endl;
    exit(1);
  }
  unsigned log2_element = atoi(argv[1]);
  unsigned int size = 1 << log2_element;

  cl::sycl::range<2> LocalRange{1, 1};

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();

  // allocate and initialized input
  pInputs = static_cast<unsigned int *>(
      malloc_shared(size * TUPLE_SZ * sizeof(unsigned int), dev, ctxt));
  for (unsigned int i = 0; i < size * TUPLE_SZ; ++i) {
    pInputs[i] = rand() % 128;
  }

  // allocate & compute expected result
  unsigned int *pExpectOutputs = static_cast<unsigned int *>(
      malloc(size * TUPLE_SZ * sizeof(unsigned int)));
  compute_local_prefixsum(pInputs, pExpectOutputs, size);

  // compute local sum for every chunk of PREFIX_ENTRIES
  cl::sycl::range<2> GlobalRange{size / PREFIX_ENTRIES, 1};
  try {
    auto e0 = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Sum_tuple>(
          GlobalRange * LocalRange, [=](item<2> it) SYCL_ESIMD_KERNEL {
            cmk_sum_tuple_count(pInputs, it.get_id(0));
          });
    });
    e0.wait();
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(pInputs, ctxt);
    free(pExpectOutputs);
    return e.get_cl_code();
  }

  bool pass = memcmp(pInputs, pExpectOutputs,
                     size * TUPLE_SZ * sizeof(unsigned int)) == 0;
  std::cout << "Prefix " << (pass ? "=> PASSED" : "=> FAILED") << std::endl
            << std::endl;

  free(pInputs, ctxt);
  free(pExpectOutputs);
  return 0;
}
