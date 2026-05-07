// 1D version of sampled image read test.

// REQUIRES: aspect-ext_oneapi_bindless_images

// UNSUPPORTED: hip
// UNSUPPORTED-INTENDED: Returning non-FP values from sampling fails on HIP.

// UNSUPPORTED: linux && arch-intel_gpu_bmg_g21 && level_zero_v2_adapter
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/20223

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include "./read_sampled_common.hpp"

int main() {
  const unsigned int seed = 0;
  const float offset = 20.0;

  std::cout << "Running 1D Sampled Image Tests!\n";
  bool result1D = runAll1D(offset, seed);

  if (result1D) {
    std::cout << "All tests passed!\n";
  } else {
    std::cerr << "An error has occurred!\n";
    return 1;
  }

  return 0;
}
