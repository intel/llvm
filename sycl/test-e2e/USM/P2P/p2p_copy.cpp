// REQUIRES: cuda
// RUN: %{build} -o %t.out
// RUN: %if cuda %{ %{run} %t.out %}

#include <cassert>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>

using namespace sycl;

// Array size to copy
constexpr int N = 100;

int main() {

  // Note that this code will largely be removed: it is temporary due to the
  // temporary lack of multiple devices per sycl context in the Nvidia backend.
  // A portable implementation, using a single gpu platform, should be possible
  // once the Nvidia context issues are resolved.
  ////////////////////////////////////////////////////////////////////////
  std::vector<sycl::device> Devs;
  for (const auto &plt : sycl::platform::get_platforms()) {

    if (plt.get_backend() == sycl::backend::ext_oneapi_cuda)
      Devs.push_back(plt.get_devices()[0]);
  }
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
