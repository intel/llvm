//==---------------- histogram.cpp  - DPC++ ESIMD on-device test -----------==//
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
#include <array>
#include <iostream>

using namespace cl::sycl;

#define NUM_BINS 256
#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024
//
// each parallel_for handles 64x32 bytes
//
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 64

void histogram_CPU(unsigned int width, unsigned int height, unsigned char *srcY,
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
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  auto ctxt = q.get_context();
  unsigned char *srcY =
      static_cast<unsigned char *>(malloc_shared(width * height, dev, ctxt));
  unsigned int *bins = static_cast<unsigned int *>(
      malloc_shared(NUM_BINS * sizeof(unsigned int), dev, ctxt));
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  uint range_width = width / BLOCK_WIDTH;
  uint range_height = height / BLOCK_HEIGHT;

  if (srcY == NULL) {
    std::cerr << "Out of memory\n";
    exit(1);
  }

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

  cl::sycl::image<2> Img(srcY, image_channel_order::rgba,
                         image_channel_type::unsigned_int32,
                         range<2>{width / sizeof(uint4), height});

  {
    // create ranges
    // We need that many workitems
    auto GlobalRange = range<1>(range_width * range_height);
    // Number of workitems in a workgroup
    auto LocalRange = range<1>(1);
    nd_range<1> Range(GlobalRange, LocalRange);

    auto e = q.submit([&](handler &cgh) {
      auto readAcc = Img.get_access<uint4, cl::sycl::access::mode::read>(cgh);

      cgh.parallel_for<class Hist>(
          Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            using namespace sycl::INTEL::gpu;

            // Get thread origin offsets
            uint tid = ndi.get_group(0);
            uint h_pos = (tid % range_width) * BLOCK_WIDTH;
            uint v_pos = (tid / range_width) * BLOCK_HEIGHT;

            // Declare a 8x32 uchar matrix to store the input block pixel value
            simd<unsigned char, 8 * 32> in;

            // Declare a vector to store the local histogram
            simd<unsigned int, NUM_BINS> histogram(0);

            // Each thread handles BLOCK_HEIGHTxBLOCK_WIDTH pixel block
            for (int y = 0; y < BLOCK_HEIGHT / 8; y++) {
              // Perform 2D media block read to load 8x32 pixel block
              in =
                  media_block_load<unsigned char, 8, 32>(readAcc, h_pos, v_pos);

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
              // Declare a vector to store the source for atomic write operation
              simd<unsigned int, 8> src;
              src = histogram.select<8, 1>(i);

#ifdef __SYCL_DEVICE_ONLY__
              flat_atomic<EsimdAtomicOpType::ATOMIC_ADD, unsigned int, 8>(
                  bins, offset, src, 1);
              offset += 8 * sizeof(unsigned int);
#else
              auto vals = block_load<unsigned int, 8>(bins + i);
              vals = vals + src;
              block_store<unsigned int, 8>(bins + i, vals);
#endif
            }
          });
    });
    e.wait();

    // SYCL will enqueue and run the kernel. Recall that the buffer's data is
    // given back to the host at the end of scope.
  } // make sure data is given back to the host at the end of this scope

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
