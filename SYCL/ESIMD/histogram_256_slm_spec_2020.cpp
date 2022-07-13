// TODO enable on Windows
// REQUIRES: linux && gpu
// UNSUPPORTED: cuda || hip
// TODO online_compiler check fails for esimd_emulator
// XFAIL: esimd_emulator
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out 16

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

static constexpr int NUM_BINS = 256;
static constexpr int SLM_SIZE = (NUM_BINS * 4);
static constexpr int BLOCK_WIDTH = 32;
static constexpr int NUM_BLOCKS = 32;

using namespace cl::sycl;
using namespace sycl::ext::intel;
using namespace sycl::ext::intel::esimd;

constexpr specialization_id<unsigned int> NumBlocksSpecId(NUM_BLOCKS);

// Histogram kernel: computes the distribution of pixel intensities
ESIMD_INLINE void histogram_atomic(const uint32_t *input_ptr, uint32_t *output,
                                   uint32_t gid, uint32_t lid,
                                   uint32_t local_size, uint32_t num_blocks) {
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
  auto start_off = (linear_id * BLOCK_WIDTH * num_blocks);
  for (int y = 0; y < num_blocks; y++) {
    auto start_addr = ((unsigned int *)input_ptr) + start_off;
    auto data = block_load<uint, 32>(start_addr);
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

class NumBlocksConst;
class histogram_slm;

int main(int argc, char **argv) {
  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(),
          cl::sycl::property::queue::enable_profiling{});
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  const char *input_file = nullptr;
  unsigned int width = 1024 * sizeof(unsigned int);
  unsigned int height = 1024;

  // Initializes input.
  unsigned int input_size = width * height;
  unsigned int *input_ptr =
      (unsigned int *)malloc_shared(input_size, dev, ctxt);
  printf("Processing %dx%d inputs\n", (int)(width / sizeof(unsigned int)),
         height);

  srand(2009);
  input_size = input_size / sizeof(int);
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
  unsigned int *output_surface =
      (uint32_t *)malloc_shared(4 * NUM_BINS, dev, ctxt);
  memset(output_surface, 0, 4 * NUM_BINS);

  unsigned int num_blocks{NUM_BLOCKS};
  if (argc == 2) {
    num_blocks = atoi(argv[1]);
    std::cout << "new num_blocks = " << num_blocks << "\n";
  }

  unsigned int num_threads;
  num_threads = width * height / (num_blocks * BLOCK_WIDTH * sizeof(int));

  auto GlobalRange = cl::sycl::range<1>(num_threads);
  auto LocalRange = cl::sycl::range<1>(NUM_BINS / 16);
  cl::sycl::nd_range<1> Range(GlobalRange, LocalRange);

  // Start Timer
  esimd_test::Timer timer;
  double start;

  double kernel_times = 0;
  unsigned num_iters = 10;
  try {
    // num_iters + 1, iteration#0 is for warmup
    for (int iter = 0; iter <= num_iters; ++iter) {
      double etime = 0;
      memset(output_surface, 0, 4 * NUM_BINS);
      auto e = q.submit([&](cl::sycl::handler &cgh) {
        cgh.set_specialization_constant<NumBlocksSpecId>(num_blocks);
        cgh.parallel_for<histogram_slm>(
            Range,
            [=](cl::sycl::nd_item<1> ndi, kernel_handler kh) SYCL_ESIMD_KERNEL {
              histogram_atomic(
                  input_ptr, output_surface, ndi.get_group(0),
                  ndi.get_local_id(0), 16,
                  kh.get_specialization_constant<NumBlocksSpecId>());
            });
      });
      e.wait();
      etime = esimd_test::report_time("kernel time", e, e);
      if (iter > 0)
        kernel_times += etime;
      else
        start = timer.Elapsed();
    }
  } catch (cl::sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }

  // End timer.
  double end = timer.Elapsed();

  esimd_test::display_timing_stats(kernel_times, num_iters,
                                   (end - start) * 1000);

  std::cout << "finish GPU histogram\n";

  memcpy(hist, output_surface, 4 * NUM_BINS);

  free(output_surface, ctxt);

  free(input_ptr, ctxt);

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
