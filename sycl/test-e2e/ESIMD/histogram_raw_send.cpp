//==-histogram_raw_send.cpp  - DPC++ ESIMD on-device test-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------===//
// REQUIRES: gpu-intel-gen9
// UNSUPPORTED: gpu-intel-dg1,gpu-intel-dg2,arch-intel_gpu_pvc
// UNSUPPORTED: ze_debug
// RUN: %{build} -o %t1.out
// RUN: %{run} %t1.out
// RUN: %{build} -DUSE_CONSTEXPR_API -o %t2.out
// RUN: %{run} %t2.out
// RUN: %{build} -DUSE_SUPPORTED_API -o %t3.out
// RUN: %{run} %t3.out

// The test checks raw send functionality with atomic write implementation
// on SKL. It does not work on DG1 due to send instruction incompatibility.

#include "esimd_test_utils.hpp"

#include <sycl/accessor_image.hpp>

#include <array>

using namespace sycl;

#define NUM_BINS 256
#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024
//
// each parallel_for handles 64x32 bytes
//
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 64

void histogram_CPU(unsigned int width, unsigned int height, uint8_t *srcY,
                   unsigned int *cpuHistogram) {
  int i;
  for (i = 0; i < width * height; i++) {
    cpuHistogram[srcY[i]] += 1;
  }
}

void writeHist(unsigned int *hist) {
  int total = 0;

  std::cerr << "\nHistogram: \n";
  for (int i = 0; i < NUM_BINS; i += 8) {
    std::cerr << "\n  [" << i << " - " << i + 7 << "]:";
    for (int j = 0; j < 8; j++) {
      std::cerr << "\t" << hist[i + j];
      total += hist[i + j];
    }
  }
  std::cerr << "\nTotal = " << total << " \n";
}

int checkHistogram(unsigned int *refHistogram, unsigned int *hist) {

  for (int i = 0; i < NUM_BINS; i++) {
    if (refHistogram[i] != hist[i]) {
      return 0;
    }
  }
  return 1;
}

using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

template <atomic_op Op, typename T, int n>
ESIMD_INLINE void atomic_write(T *bins, simd<unsigned, n> offset,
                               simd<T, n> src0) {
  simd<T, n> oldDst;
  simd<uintptr_t, n> vAddr(reinterpret_cast<uintptr_t>(bins));
  simd<uintptr_t, n> vOffset = offset;
  vAddr += vOffset;

  uint32_t exDesc = 0x4C;
  uint32_t desc = 0x414A7FF;
  constexpr uint8_t execSize = 0x83;
  constexpr uint8_t sfid = 0x1;
  constexpr uint8_t numDst = 0x1;
  constexpr uint8_t numSrc0 = 0x2;
  constexpr uint8_t numSrc1 = 0x1;
  constexpr uint8_t isEOT = 0;
  constexpr uint8_t isSendc = 0;

#ifdef USE_CONSTEXPR_API
  experimental::esimd::raw_sends<execSize, sfid, numSrc0, numSrc1, numDst,
                                 isEOT, isSendc>(oldDst, vAddr, src0, exDesc,
                                                 desc);
#elif defined(USE_SUPPORTED_API)
  esimd::raw_sends<execSize, sfid, numSrc0, numSrc1, numDst,
                   raw_send_eot::not_eot, raw_send_sendc::not_sendc>(
      oldDst, vAddr, src0, exDesc, desc);

#else
  experimental::esimd::raw_sends(oldDst, vAddr, src0, exDesc, desc, execSize,
                                 sfid, numSrc0, numSrc1, numDst, isEOT,
                                 isSendc);
#endif
}

int main(int argc, char *argv[]) {

  const char *input_file = nullptr;
  unsigned int width = IMG_WIDTH * sizeof(unsigned int);
  unsigned int height = IMG_HEIGHT;

  if (argc == 2) {
    input_file = argv[1];
  } else {
    std::cerr << "Usage: Histogram.exe input_file" << std::endl;
    std::cerr << "No input file specificed. Use default random value ...."
              << std::endl;
  }

  // ------------------------------------------------------------------------
  // Read in image luma plane

  // Allocate Input Buffer
  queue q = esimd_test::createQueue();
  esimd_test::printTestLabel(q);

  esimd_test::shared_vector<uint8_t> srcY_vec(
      width * height, esimd_test::shared_allocator<uint8_t>{q});
  esimd_test::shared_vector<unsigned int> bins_vec(
      NUM_BINS, esimd_test::shared_allocator<unsigned int>{q});
  uint8_t *srcY = srcY_vec.data();
  ;
  unsigned int *bins = bins_vec.data();

  uint range_width = width / BLOCK_WIDTH;
  uint range_height = height / BLOCK_HEIGHT;

  // Initializes input.
  unsigned int input_size = width * height;
  std::cerr << "Processing inputs\n";

  if (input_file != nullptr) {
    FILE *f = fopen(input_file, "rb");
    if (f == NULL) {
      std::cerr << "Error opening file " << input_file;
      std::exit(1);
    }

    unsigned int cnt = fread(srcY, sizeof(unsigned char), input_size, f);
    if (cnt != input_size) {
      std::cerr << "Error reading input from " << input_file;
      std::exit(1);
    }
  } else {
    srand(2009);
    for (int i = 0; i < input_size; ++i) {
      srcY[i] = rand() % 256;
    }
  }

  for (int i = 0; i < NUM_BINS; i++) {
    bins[i] = 0;
  }

  // ------------------------------------------------------------------------
  // CPU Execution:

  unsigned int cpuHistogram[NUM_BINS];
  memset(cpuHistogram, 0, sizeof(cpuHistogram));
  histogram_CPU(width, height, srcY, cpuHistogram);

  sycl::image<2> Img(srcY, image_channel_order::rgba,
                     image_channel_type::unsigned_int32,
                     range<2>{width / sizeof(uint4), height});

  // Start Timer
  esimd_test::Timer timer;
  double start;

  double kernel_times = 0;
  unsigned num_iters = 10;
  const bool profiling =
      q.has_property<sycl::property::queue::enable_profiling>();
  try {
    // num_iters + 1, iteration#0 is for warmup
    for (int iter = 0; iter <= num_iters; ++iter) {
      double etime = 0;
      for (int b = 0; b < NUM_BINS; b++)
        bins[b] = 0;
      // create ranges
      // We need that many task groups
      auto GlobalRange = range<1>(range_width * range_height);
      // We need that many tasks in each group
      auto LocalRange = range<1>(1);
      nd_range<1> Range(GlobalRange, LocalRange);

      auto e = q.submit([&](handler &cgh) {
        auto readAcc = Img.get_access<uint4, sycl::access::mode::read>(cgh);

        cgh.parallel_for<class Hist>(
            Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
              // Get thread origin offsets
              uint tid = ndi.get_group(0);
              uint h_pos = (tid % range_width) * BLOCK_WIDTH;
              uint v_pos = (tid / range_width) * BLOCK_HEIGHT;

              // Declare a 8x32 uchar matrix to store the input block pixel
              // value
              simd<unsigned char, 8 * 32> in;

              // Declare a vector to store the local histogram
              simd<unsigned int, NUM_BINS> histogram(0);

              // Each thread handles BLOCK_HEIGHTxBLOCK_WIDTH pixel block
              for (int y = 0; y < BLOCK_HEIGHT / 8; y++) {
                // Perform 2D media block read to load 8x32 pixel block
                in = media_block_load<unsigned char, 8, 32>(readAcc, h_pos,
                                                            v_pos);

            // Accumulate local histogram for each pixel value
#pragma unroll
                for (int i = 0; i < 8; i++) {
#pragma unroll
                  for (int j = 0; j < 32; j++) {
                    histogram.select<1, 1>(in[i * 32 + j]) += 1;
                  }
                }

                // Update starting offset for the next work block
                v_pos += 8;
              }

              // Declare a vector to store the offset for atomic write operation
              simd<unsigned int, 8> offset(0, 1); // init to 0, 1, 2, ..., 7
              offset *= sizeof(unsigned int);

          // Update global sum by atomically adding each local histogram
#pragma unroll
              for (int i = 0; i < NUM_BINS; i += 8) {
                // Declare a vector to store the source for atomic write
                // operation
                simd<unsigned int, 8> src;
                src = histogram.select<8, 1>(i);

#ifdef __SYCL_DEVICE_ONLY__
                // flat_atomic<atomic_op::add, unsigned int,
                // 8>(bins, offset, src, 1);
                atomic_write<atomic_op::add, unsigned int, 8>(bins, offset,
                                                              src);
                offset += 8 * sizeof(unsigned int);
#else
                simd<unsigned int, 8> vals;
                vals.copy_from(bins + i);
                vals = vals + src;
                vals.copy_to(bins + i);
#endif
              }
            });
      });
      e.wait();
      if (profiling) {
        etime = esimd_test::report_time("kernel time", e, e);
        if (iter > 0)
          kernel_times += etime;
      }
      if (iter == 0)
        start = timer.Elapsed();
    }

    // SYCL will enqueue and run the kernel. Recall that the buffer's data is
    // given back to the host at the end of scope.
    // make sure data is given back to the host at the end of this scope
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }

  // End timer.
  double end = timer.Elapsed();

  esimd_test::display_timing_stats(profiling ? &kernel_times : nullptr,
                                   num_iters, (end - start) * 1000);

  writeHist(bins);
  writeHist(cpuHistogram);
  // Checking Histogram
  if (checkHistogram(cpuHistogram, bins)) {
    std::cerr << "PASSED\n";
    return 0;
  } else {
    std::cerr << "FAILED\n";
    return 1;
  }

  return 0;
}
