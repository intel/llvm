//==--------------- histogram_256_slm.cpp  - DPC++ ESIMD on-device test ----==//
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

static constexpr int NUM_BINS = 256;
static constexpr int SLM_SIZE = (NUM_BINS * 4);
static constexpr int BLOCK_WIDTH = 32;
static constexpr int NUM_BLOCKS = 32;

using namespace sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

// Histogram kernel: computes the distribution of pixel intensities
ESIMD_INLINE void histogram_atomic(const uint32_t *input_ptr, uint32_t *output,
                                   uint32_t gid, uint32_t lid,
                                   uint32_t local_size) {
  // Declare and initialize SLM
  slm_init<SLM_SIZE>();
  uint linear_id = gid * local_size + lid;

  simd<uint, 16> slm_offset(0, 1);
  slm_offset += 16 * lid;
  slm_offset *= sizeof(int);
  simd<uint, 16> slm_data = 0;
  slm_scatter<uint, 16>(slm_offset, slm_data);
  esimd::barrier();

  // Each thread handles NUM_BLOCKSxBLOCK_WIDTH pixel blocks
  auto start_off = (linear_id * BLOCK_WIDTH * NUM_BLOCKS);
  for (int y = 0; y < NUM_BLOCKS; y++) {
    auto start_addr = ((unsigned int *)input_ptr) + start_off;
    simd<uint, 32> data;
    data.copy_from(start_addr);
    auto in = data.bit_cast_view<uchar>();

#pragma unroll
    for (int j = 0; j < BLOCK_WIDTH * sizeof(int); j += 16) {
      // Accumulate local histogram for each pixel value
      simd<uint, 16> dataOffset = in.select<16, 1>(j).read();
      dataOffset *= sizeof(int);
      slm_atomic_update<atomic_op::inc, uint, 16>(dataOffset, 1);
    }
    start_off += BLOCK_WIDTH;
  }
  esimd::barrier();

  // Update global sum by atomically adding each local histogram
  simd<uint, 16> local_histogram;
  local_histogram = slm_gather<uint32_t, 16>(slm_offset);
  atomic_update<atomic_op::add, uint32_t, 8>(
      output, slm_offset.select<8, 1>(0), local_histogram.select<8, 1>(0), 1);
  atomic_update<atomic_op::add, uint32_t, 8>(
      output, slm_offset.select<8, 1>(8), local_histogram.select<8, 1>(8), 1);
}

// This function calculates histogram of the image with the CPU.
// @param size: the size of the input array.
// @param src: pointer to the input array.
// @param cpu_histogram: pointer to the histogram of the input image.
void HistogramCPU(unsigned int size, unsigned int *src,
                  unsigned int *cpu_histogram) {
  for (int i = 0; i < size; i++) {
    unsigned int x = src[i];
    cpu_histogram[(x)&0xFFU] += 1;
    cpu_histogram[(x >> 8) & 0xFFU] += 1;
    cpu_histogram[(x >> 16) & 0xFFU] += 1;
    cpu_histogram[(x >> 24) & 0xFFU] += 1;
  }
}

// This function compares the output data calculated by the CPU and the
// GPU separately.
// If they are identical, return 1, else return 0.
int CheckHistogram(unsigned int *cpu_histogram, unsigned int *gpu_histogram) {
  unsigned int bad = 0;
  for (int i = 0; i < NUM_BINS; i++) {
    if (cpu_histogram[i] != gpu_histogram[i]) {
      std::cout << "At " << i << ": CPU = " << cpu_histogram[i]
                << ", GPU = " << gpu_histogram[i] << std::endl;
      if (bad >= 256)
        return 0;
      bad++;
    }
  }
  if (bad > 0)
    return 0;

  return 1;
}

int main() {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(),
          sycl::property::queue::enable_profiling{});

  const char *input_file = nullptr;
  unsigned int width = 1024;
  unsigned int height = 1024;

  // Initializes input.
  unsigned int input_size = width * height;
  unsigned int *input_ptr = malloc_shared<unsigned int>(input_size, q);
  printf("Processing %dx%d inputs\n", width, height);

  srand(2009);
  for (int i = 0; i < input_size; ++i) {
    input_ptr[i] = rand() % 256;
    input_ptr[i] |= (rand() % 256) << 8;
    input_ptr[i] |= (rand() % 256) << 16;
    input_ptr[i] |= (rand() % 256) << 24;
  }

  // Allocates system memory for output buffer.
  int buffer_size = sizeof(unsigned int) * NUM_BINS;
  unsigned int *hist = new unsigned int[buffer_size];
  if (hist == nullptr) {
    free(input_ptr, q);
    std::cerr << "Out of memory\n";
    exit(1);
  }
  memset(hist, 0, buffer_size);

  // Uses the CPU to calculate the histogram output data.
  unsigned int cpu_histogram[NUM_BINS];
  memset(cpu_histogram, 0, sizeof(cpu_histogram));

  HistogramCPU(input_size, input_ptr, cpu_histogram);

  std::cout << "finish cpu_histogram\n";

  // Uses the GPU to calculate the histogram output data.
  unsigned int *output_surface = malloc_shared<unsigned int>(NUM_BINS, q);

  unsigned int num_threads;
  num_threads = width * height / (NUM_BLOCKS * BLOCK_WIDTH);

  auto GlobalRange = range<1>(num_threads);
  auto LocalRange = range<1>(NUM_BINS / 16);
  nd_range<1> Range(GlobalRange, LocalRange);

  // Start Timer
  esimd_test::Timer timer;
  double start;

  // Launches the task on the GPU.
  double kernel_times = 0;
  unsigned num_iters = 10;
  try {
    for (int iter = 0; iter <= num_iters; ++iter) {
      double etime = 0;
      memset(output_surface, 0, sizeof(unsigned int) * NUM_BINS);
      auto e = q.submit([&](handler &cgh) {
        cgh.parallel_for<class histogram_slm>(
            Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
              histogram_atomic(input_ptr, output_surface, ndi.get_group(0),
                               ndi.get_local_id(0), 16);
            });
      });
      e.wait();
      etime = esimd_test::report_time("kernel time", e, e);
      if (iter > 0)
        kernel_times += etime;
      else
        start = timer.Elapsed();
    }
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }

  // End timer.
  double end = timer.Elapsed();

  esimd_test::display_timing_stats(kernel_times, num_iters,
                                   (end - start) * 1000);

  std::cout << "finish GPU histogram\n";

  memcpy(hist, output_surface, 4 * NUM_BINS);

  free(output_surface, q);
  free(input_ptr, q);

  // Compares the CPU histogram output data with the
  // GPU histogram output data.
  // If there is no difference, the result is correct.
  // Otherwise there is something wrong.
  int res = CheckHistogram(cpu_histogram, hist);
  if (res)
    std::cout << "PASSED\n";
  else
    std::cout << "FAILED\n";

  return res ? 0 : -1;
}
