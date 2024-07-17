// REQUIRES: cuda || hip || level_zero
// RUN:  %{build} -o %t.out
// RUN:  %{run} %t.out

#include <cassert>
#include <numeric>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>
#include <vector>

using namespace sycl;

// Array size to copy
constexpr int N = 100;

int main() {

  auto Devs = platform(gpu_selector_v).get_devices(info::device_type::gpu);

  if (Devs.size() < 2) {
    std::cout << "Cannot test P2P capabilities, at least two devices are "
                 "required, exiting."
              << std::endl;
    return 0;
  }

  std::vector<sycl::queue> Queues;
  std::transform(Devs.begin(), Devs.end(), std::back_inserter(Queues),
                 [](const sycl::device &D) { return sycl::queue{D}; });
  ////////////////////////////////////////////////////////////////////////

  if (!Devs[0].ext_oneapi_can_access_peer(
          Devs[1], sycl::ext::oneapi::peer_access::access_supported)) {
    std::cout << "P2P access is not supported by devices, exiting."
              << std::endl;
    return 0;
  }

  // Enables Devs[0] to access Devs[1] memory.
  Devs[0].ext_oneapi_enable_peer_access(Devs[1]);

  std::vector<int> input(N);
  std::iota(input.begin(), input.end(), 0);

  int *arr0 = malloc<int>(N, Queues[0], usm::alloc::device);
  Queues[0].memcpy(arr0, &input[0], N * sizeof(int));

  int *arr1 = malloc<int>(N, Queues[1], usm::alloc::device);
  // P2P copy performed here:
  Queues[1].copy(arr0, arr1, N).wait();

  int out[N];
  Queues[1].copy(arr1, out, N).wait();

  sycl::free(arr0, Queues[0]);
  sycl::free(arr1, Queues[1]);

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (out[i] != input[i]) {
      printf("%d %d\n", out[i], input[i]);
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  return 0;
}
