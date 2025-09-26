// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: level_zero_v2_adapter
// UNSUPPORTED-TRACKER: CMPLRLLVM-67039

// Test that usm device pointer can be used in a kernel compiled for a context
// with multiple devices.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/platform.hpp>
#include <sycl/usm.hpp>
#include <vector>

using namespace sycl;

class AddIdxKernel;

int main() {
  sycl::platform plt;
  std::vector<sycl::device> devices = plt.get_devices();
  if (devices.size() < 2) {
    std::cout << "Need at least 2 GPU devices for this test.\n";
    return 0;
  }

  std::vector<sycl::device> ctx_devices{devices[0], devices[1]};
  sycl::context ctx(ctx_devices);

  constexpr size_t N = 16;
  std::vector<std::vector<int>> results(ctx_devices.size(),
                                        std::vector<int>(N, 0));

  // Create a kernel bundle compiled for both devices in the context
  auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(ctx);

  // For each device, create a queue and run a kernel using device USM
  for (size_t i = 0; i < ctx_devices.size(); ++i) {
    sycl::queue q(ctx, ctx_devices[i]);
    int *data = sycl::malloc_device<int>(N, q);
    q.fill(data, 1, N).wait();
    q.submit([&](sycl::handler &h) {
       h.use_kernel_bundle(kb);
       h.parallel_for<AddIdxKernel>(
           sycl::range<1>(N), [=](sycl::id<1> idx) { data[idx] += idx[0]; });
     }).wait();
    q.memcpy(results[i].data(), data, N * sizeof(int)).wait();
    sycl::free(data, q);
  }

  for (size_t i = 0; i < ctx_devices.size(); ++i) {
    std::cout << "Device " << i << " results: ";
    for (size_t j = 0; j < N; ++j) {
      if (results[i][j] != 1 + static_cast<int>(j)) {
        return -1;
      }
      std::cout << results[i][j] << " ";
    }
  }
  return 0;
}
