//==------- Prefix_Local_sum3.cpp  - DPC++ ESIMD on-device test -----------==//
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

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#define MAX_TS_WIDTH 1024
// kernel can handle TUPLE_SZ 1, 2, or 4
#define TUPLE_SZ 2

#if TUPLE_SZ == 1
#define GATHER_SCATTER_MASK rgba_channel_mask::R
#elif TUPLE_SZ == 2
#define GATHER_SCATTER_MASK rgba_channel_mask::GR
#elif TUPLE_SZ == 4
#define GATHER_SCATTER_MASK rgba_channel_mask::ABGR
#endif

#define LOG_ENTRIES 8
#define PREFIX_ENTRIES (1 << LOG_ENTRIES)
#define PREFIX_ENTRIES_LOW 32
#define ENTRIES_THRESHOLD 2048
// minimum number of threads to launch a kernel (power of 2)
#define MIN_NUM_THREADS 1
#define REMAINING_ENTRIES 64

using namespace cl::sycl;
using namespace sycl::ext::intel::esimd;

void compute_local_prefixsum(unsigned int prefixSum[], unsigned int size,
                             unsigned elem_stride, unsigned thread_stride,
                             unsigned entry_per_thread) {

  unsigned local_sum[TUPLE_SZ];
  for (unsigned k = 0; k < size; k += thread_stride) {
    memset(local_sum, 0, TUPLE_SZ * sizeof(unsigned)); // init 0

    for (int i = 0; i < entry_per_thread; i++) {
      for (int j = 0; j < TUPLE_SZ; j++) {
        local_sum[j] +=
            prefixSum[(k + (i + 1) * elem_stride - 1) * TUPLE_SZ + j];
      }
    }
    // store local_sum in the last entry
    memcpy(&prefixSum[(k + entry_per_thread * elem_stride - 1) * TUPLE_SZ],
           local_sum, TUPLE_SZ * sizeof(unsigned));
  }
}

void compute_local_prefixsum_remaining(unsigned int prefixSum[],
                                       unsigned int size,
                                       unsigned elem_stride) {

  unsigned local_sum[TUPLE_SZ];
  memset(local_sum, 0, TUPLE_SZ * sizeof(unsigned)); // init 0

  for (int i = 0; i < size / elem_stride; i++) {
    for (int j = 0; j < TUPLE_SZ; j++) {
      local_sum[j] += prefixSum[((i + 1) * elem_stride - 1) * TUPLE_SZ + j];
    }
    // update every elem_stride entry
    memcpy(&prefixSum[((i + 1) * elem_stride - 1) * TUPLE_SZ], local_sum,
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

#pragma unroll
  for (unsigned int i = 0; i < TUPLE_SZ; i++) {
    S.select<32, TUPLE_SZ>(i) =
        gather<unsigned, 32>(buf, element_offset + i * sizeof(unsigned));
  }

#pragma unroll
  for (int j = 1; j < PREFIX_ENTRIES / 32; j++) {
    element_offset += (stride_elems * 32 * TUPLE_SZ) * sizeof(unsigned);
#pragma unroll
    for (unsigned int i = 0; i < TUPLE_SZ; i++) {
      T.select<32, TUPLE_SZ>(i) =
          gather<unsigned, 32>(buf, element_offset + i * sizeof(unsigned));
    }
    S += T;
  }

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

  simd<unsigned, 8> voff(0, 1);     // 0, 1, 2, 3
  simd_mask<8> p = voff < TUPLE_SZ; // predicate
  voff = (voff + (global_offset + stride_threads * TUPLE_SZ - TUPLE_SZ)) *
         sizeof(unsigned);
  scatter<unsigned, 8>(buf, voff, S.select<8, 1>(0), p);
}

void cmk_acum_iterative_low(unsigned *buf, unsigned h_pos,
                            unsigned int stride_elems,
                            unsigned int stride_threads) {
  simd<unsigned, 32> element_offset(0, 1); // 0, 1, 2, ..., 31

  // global offset for a thread
  unsigned int global_offset = (h_pos * stride_threads * TUPLE_SZ);
  // element offsets for scattered read: [e0,e1,e2,...,e31] where e_i =
  // global_offset + # prefix_entries + prefix_entries - 1;
  element_offset =
      (((element_offset + 1) * stride_elems - 1) * TUPLE_SZ + global_offset) *
      sizeof(unsigned);

  simd<unsigned int, 32 * TUPLE_SZ> S, T;

#pragma unroll
  for (unsigned int i = 0; i < TUPLE_SZ; i++) {
    S.select<32, TUPLE_SZ>(i) =
        gather<unsigned, 32>(buf, element_offset + i * sizeof(unsigned));
  }
#pragma unroll
  for (int j = 1; j < PREFIX_ENTRIES_LOW / 32; j++) {
    element_offset += (stride_elems * 32 * TUPLE_SZ) * sizeof(unsigned);
#pragma unroll
    for (unsigned int i = 0; i < TUPLE_SZ; i++) {
      T.select<32, TUPLE_SZ>(i) =
          gather<unsigned, 32>(buf, element_offset + i * sizeof(unsigned));
    }
    S += T;
  }

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

  simd<unsigned, 8> voff(0, 1);     // 0, 1, 2, 3
  simd_mask<8> p = voff < TUPLE_SZ; // predicate
  voff = (voff + (global_offset + stride_threads * TUPLE_SZ - TUPLE_SZ)) *
         sizeof(unsigned);
  scatter<unsigned, 8>(buf, voff, S.select<8, 1>(0), p);
}

// final reduction. One thread to compute prefix all remaining entries
void cmk_acum_final(unsigned *buf, unsigned h_pos, unsigned int stride_elems,
                    unsigned remaining) {
  simd<unsigned, 32> elm32(0, 1);

  // element offsets for scattered read: [e0,e1,e2,...,e31] where e_i =
  // global_offset + # prefix_entries + prefix_entries - 1;
  simd<unsigned, 32> element_offset =
      (((elm32 + 1) * stride_elems - 1) * TUPLE_SZ) * sizeof(unsigned);

  simd<unsigned, 32 * TUPLE_SZ> S;
  simd<unsigned, TUPLE_SZ> prev = 0;
  for (unsigned i = 0; i < remaining; i += 32) {

    simd_mask<32> p = elm32 < remaining;

    S = gather_rgba<GATHER_SCATTER_MASK>(buf, element_offset, p);

    auto cnt_table = S.bit_cast_view<unsigned int, TUPLE_SZ, 32>();
    cnt_table.column(0) += prev;
    for (unsigned j = 0; j < TUPLE_SZ; j++) {
      // step 1
      cnt_table.select<1, 1, 16, 2>(j, 1) +=
          cnt_table.select<1, 1, 16, 2>(j, 0);
      // step 2
      cnt_table.select<1, 1, 8, 4>(j, 2) += cnt_table.select<1, 1, 8, 4>(j, 1);
      cnt_table.select<1, 1, 8, 4>(j, 3) += cnt_table.select<1, 1, 8, 4>(j, 1);
      // step 3
      cnt_table.select<1, 1, 4, 1>(j, 4) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 3);
      cnt_table.select<1, 1, 4, 1>(j, 12) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 11);
      cnt_table.select<1, 1, 4, 1>(j, 20) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 19);
      cnt_table.select<1, 1, 4, 1>(j, 28) +=
          cnt_table.replicate_vs_w_hs<1, 0, 4, 0>(j, 27);
      // step 4
      cnt_table.select<1, 1, 8, 1>(j, 8) +=
          cnt_table.replicate_vs_w_hs<1, 0, 8, 0>(j, 7);
      cnt_table.select<1, 1, 8, 1>(j, 24) +=
          cnt_table.replicate_vs_w_hs<1, 0, 8, 0>(j, 23);
      // step 5
      cnt_table.select<1, 1, 16, 1>(j, 16) +=
          cnt_table.replicate_vs_w_hs<1, 0, 16, 0>(j, 15);
    }
    scatter_rgba<GATHER_SCATTER_MASK>(buf, element_offset, S, p);
    elm32 += 32;
    element_offset += stride_elems * TUPLE_SZ * sizeof(unsigned) * 32;
    prev = cnt_table.column(31);
  }
}

double hierarchical_prefix(queue &q, unsigned *buf, unsigned elem_stride,
                           unsigned thrd_stride, unsigned n_entries,
                           unsigned entry_per_th) {
  double kernel_times = 0;
  try {
    if (n_entries <= REMAINING_ENTRIES) {
#ifdef DEBUG_DUMPS
      std::cout << "... n_entries: " << n_entries
                << " elem_stide: " << elem_stride
                << " thread_stride: " << thrd_stride
                << " entry per thread: " << entry_per_th << std::endl;
#endif // DEBUG_DUMPS
      // one single thread
      auto e = q.submit([&](handler &cgh) {
        cgh.parallel_for<class Accum_final>(
            range<2>{1, 1} * range<2>{1, 1}, [=](item<2> it) SYCL_ESIMD_KERNEL {
              cmk_acum_final(buf, it.get_id(0), elem_stride, n_entries);
            });
      });
      e.wait();
      kernel_times += esimd_test::report_time("kernel1 time", e, e);
      return kernel_times;
    }

#ifdef DEBUG_DUMPS
    std::cout << "*** n_entries: " << n_entries
              << " elem_stide: " << elem_stride
              << " thread_stride: " << thrd_stride
              << " entry per thread: " << entry_per_th << std::endl;
#endif // DEBUG_DUMPS

    if (entry_per_th == PREFIX_ENTRIES) {
      auto e = q.submit([&](handler &cgh) {
        cgh.parallel_for<class Accum_iterative1>(
            range<2>{n_entries / entry_per_th, 1} * range<2>{1, 1},
            [=](item<2> it) SYCL_ESIMD_KERNEL {
              cmk_acum_iterative(buf, it.get_id(0), elem_stride, thrd_stride);
            });
      });
      e.wait();
      kernel_times += esimd_test::report_time("kernel2 time", e, e);
    } else {
      auto e = q.submit([&](handler &cgh) {
        cgh.parallel_for<class Accum_iterative2>(
            range<2>{n_entries / entry_per_th, 1} * range<2>{1, 1},
            [=](item<2> it) SYCL_ESIMD_KERNEL {
              cmk_acum_iterative_low(buf, it.get_id(0), elem_stride,
                                     thrd_stride);
            });
      });
      e.wait();
      kernel_times += esimd_test::report_time("kernel3 time", e, e);
    }
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
  }

  // if number of remaining entries <= 4K , each thread  accumulates smaller
  // number of entries to keep EUs saturated
  if (n_entries / entry_per_th > 4096)
    kernel_times +=
        hierarchical_prefix(q, buf, thrd_stride, thrd_stride * PREFIX_ENTRIES,
                            n_entries / entry_per_th, PREFIX_ENTRIES);
  else
    kernel_times += hierarchical_prefix(
        q, buf, thrd_stride, thrd_stride * PREFIX_ENTRIES_LOW,
        n_entries / entry_per_th, PREFIX_ENTRIES_LOW);

#ifdef DEBUG_DUMPS
  std::cout << "=== n_entries: " << n_entries << " elem_stide: " << elem_stride
            << " thread_stride: " << thrd_stride
            << " entry per thread: " << entry_per_th << std::endl;
#endif // DEBUG_DUMPS

  return kernel_times;
}

//************************************
// Demonstrate summation of arrays both in scalar on CPU and parallel on device
// This is a ULT test variant of PrefixSum kernel with different implementation
// to increase test coverage of different usage cases and help isolate bugs.
// Difference from PrefixSum kernel:
// - Use gather<>() and gather4<>() to read in data
// - Use scatter<>() and scatter4<>() to write output
//************************************
int main(int argc, char *argv[]) {

  unsigned log2_element = 26;
  unsigned int size = 1 << log2_element;

  cl::sycl::range<2> LocalRange{1, 1};

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(),
          property::queue::enable_profiling{});

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  // allocate and initialized input data
  unsigned int *pInputs = static_cast<unsigned int *>(
      malloc(size * TUPLE_SZ * sizeof(unsigned int)));
  for (unsigned int i = 0; i < size * TUPLE_SZ; ++i) {
    pInputs[i] = rand() % 128;
  }

  // allocate kernel buffer
  unsigned int *pDeviceOutputs =
      malloc_shared<unsigned int>(size * TUPLE_SZ, q);

  // allocate & intialize expected result
  unsigned int *pExpectOutputs = static_cast<unsigned int *>(
      malloc(size * TUPLE_SZ * sizeof(unsigned int)));
  memcpy(pExpectOutputs, pInputs, size * TUPLE_SZ * sizeof(unsigned));

  // Start Timer
  esimd_test::Timer timer;
  double start;

  double kernel_times = 0;
  unsigned num_iters = 10;

  for (int iter = 0; iter <= num_iters; ++iter) {
    memcpy(pDeviceOutputs, pInputs, size * TUPLE_SZ * sizeof(unsigned));
    double etime = hierarchical_prefix(q, pDeviceOutputs, 1, PREFIX_ENTRIES,
                                       size, PREFIX_ENTRIES);
    if (iter > 0)
      kernel_times += etime;
    else
      start = timer.Elapsed();
  }

  // End timer.
  double end = timer.Elapsed();

  esimd_test::display_timing_stats(kernel_times, num_iters,
                                   (end - start) * 1000);

  compute_local_prefixsum(pExpectOutputs, size, 1, PREFIX_ENTRIES,
                          PREFIX_ENTRIES);
  compute_local_prefixsum(pExpectOutputs, size, PREFIX_ENTRIES,
                          PREFIX_ENTRIES * PREFIX_ENTRIES, PREFIX_ENTRIES);
  compute_local_prefixsum(pExpectOutputs, size, PREFIX_ENTRIES * PREFIX_ENTRIES,
                          PREFIX_ENTRIES * PREFIX_ENTRIES * PREFIX_ENTRIES_LOW,
                          PREFIX_ENTRIES_LOW);
  compute_local_prefixsum_remaining(pExpectOutputs, size,
                                    PREFIX_ENTRIES * PREFIX_ENTRIES *
                                        PREFIX_ENTRIES_LOW);

  bool pass = memcmp(pDeviceOutputs, pExpectOutputs,
                     size * TUPLE_SZ * sizeof(unsigned int)) == 0;
  std::cout << "Prefix " << (pass ? "=> PASSED" : "=> FAILED") << std::endl
            << std::endl;

  free(pDeviceOutputs, q);
  free(pExpectOutputs);
  free(pInputs);
  return 0;
}
