##Calculate a histogram of a 2d buffer.

    Compile and run :
```bash > clang++ - fsycl histogram_2d.cpp
> ONEAPI_DEVICE_SELECTOR = level_zero:gpu./ a.out 
Running on Intel(R) UHD Graphics 630 
Processing inputs
GPU Histogram :
.......................
CPU Histogram : 
........................ 
Passed
```

Source code :
```C++
#include <array>
#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

#define NUM_BINS 256
#define IMG_WIDTH 1024
#define IMG_HEIGHT 1024
//
// each parallel_for handles 1x32 bytes
//
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 1

void histogram_CPU(unsigned int width, unsigned int height, unsigned char *srcY,
                   unsigned int *cpuHistogram) {
  int i;
  for (i = 0; i < width * height; i++) {
    cpuHistogram[srcY[i]] += 1;
  }
}

void writeHist(unsigned int *hist) {
  int total = 0;

  std::cerr << " Histogram: \n";
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
  unsigned int width = IMG_WIDTH * sizeof(unsigned int);
  unsigned int height = IMG_HEIGHT;

  // ------------------------------------------------------------------------

  // Allocate Input Buffer
  queue q;

  auto dev = q.get_device();
  unsigned char *srcY = malloc_shared<unsigned char>(width * height, q);
  if (srcY == NULL) {
    std::cerr << "Out of memory\n";
    exit(1);
  }

  unsigned int *bins = malloc_shared<unsigned int>(NUM_BINS, q);
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  uint range_width = width / BLOCK_WIDTH;
  uint range_height = height / BLOCK_HEIGHT;

  // Initializes input.
  unsigned int input_size = width * height;
  std::cerr << "Processing inputs\n";

  srand(2009);
  for (int i = 0; i < input_size; ++i) {
    srcY[i] = rand() % 256;
  }

  for (int i = 0; i < NUM_BINS; i++) {
    bins[i] = 0;
  }

  // ------------------------------------------------------------------------
  // CPU Execution:

  unsigned int cpuHistogram[NUM_BINS];
  memset(cpuHistogram, 0, sizeof(cpuHistogram));
  histogram_CPU(width, height, srcY, cpuHistogram);

  try {
    // create ranges
    // We need that many workitems
    auto GlobalRange = range<2>(range_width, range_height);
    // Number of workitems in a workgroup
    auto LocalRange = range<2>(1, 1);
    nd_range<2> Range(GlobalRange, LocalRange);

    auto e = q.submit([&](handler &cgh) {
      cgh.parallel_for<class Hist>(
          Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            using namespace sycl::ext::intel::esimd;

            // Get thread origin offsets
            uint h_pos = ndi.get_group(0) * BLOCK_WIDTH;
            uint v_pos = ndi.get_group(1) * BLOCK_HEIGHT;

            // Declare a 1xBLOCK_WIDTH uchar matrix to store the input block
            // pixel value
            simd<unsigned char, BLOCK_WIDTH> in;

            // Declare a vector to store the local histogram
            simd<unsigned int, NUM_BINS> histogram(0);

            in.copy_from(srcY + v_pos * width + h_pos);

        // Accumulate local histogram for each pixel value
#pragma unroll
            for (int j = 0; j < 32; j++) {
              histogram.select<1, 1>(in[j]) += 1;
            }

            // Declare a vector to store the offset for atomic write operation
            simd<uint32_t, 8> offset(0, 1); // init to 0, 1, 2, ..., 7
            offset *= sizeof(unsigned int);

        // Update global sum by atomically adding each local histogram
#pragma unroll
            for (int i = 0; i < NUM_BINS; i += 8) {
              // Declare a vector to store the source for atomic write
              // operation
              simd<unsigned int, 8> src;
              src = histogram.select<8, 1>(i);

              atomic_update<atomic_op::add, unsigned int, 8>(bins, offset, src,
                                                             1);
              offset += 8 * sizeof(unsigned int);
            }
          });
    });
    e.wait();

    // SYCL will enqueue and run the kernel. Recall that the buffer's data is
    // given back to the host at the end of scope.
    // make sure data is given back to the host at the end of this scope
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    free(srcY, q);
    free(bins, q);
    return 1;
  }
  std::cerr << "\nGPU ";
  writeHist(bins);
  std::cerr << "\nCPU ";
  writeHist(cpuHistogram);
  // Checking Histogram
  int result = checkHistogram(cpuHistogram, bins);
  free(srcY, q);
  free(bins, q);
  if (result) {
    std::cerr << "PASSED\n";
    return 0;
  } else {
    std::cerr << "FAILED\n";
    return 1;
  }

  return 0;
}
```
