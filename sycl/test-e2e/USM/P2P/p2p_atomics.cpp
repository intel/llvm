// REQUIRES: cuda || hip || level_zero
// RUN:  %{build} %if any-device-is-cuda %{ -Xsycl-target-backend --cuda-gpu-arch=sm_61 %} -o %t.out
// RUN:  %{run} %t.out

#include <cassert>
#include <numeric>
#include <vector>

#include <sycl/detail/core.hpp>

#include <sycl/atomic_ref.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

// number of atomic operations
constexpr size_t N = 512;

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

  if (!Devs[1].ext_oneapi_can_access_peer(
          Devs[0], sycl::ext::oneapi::peer_access::atomics_supported)) {
    std::cout << "P2P atomics are not supported by devices, exiting."
              << std::endl;
    return 0;
  }

  // Enables Devs[1] to access Devs[0] memory.
  Devs[1].ext_oneapi_enable_peer_access(Devs[0]);

  std::vector<int> input(N);
  std::iota(input.begin(), input.end(), 0);

  int h_sum = 0.;
  for (const auto &value : input) {
    h_sum += value;
  }

  int *d_sum = malloc_shared<int>(1, Queues[0]);
  int *d_in = malloc_device<int>(N, Queues[0]);

  Queues[0].memcpy(d_in, &input[0], N * sizeof(int));
  Queues[0].wait();

  range global_range{N};

  *d_sum = 0.;
  Queues[1].submit([&](handler &h) {
    h.parallel_for<class peer_atomic>(global_range, [=](id<1> i) {
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                       sycl::memory_scope::system,
                       access::address_space::global_space>(*d_sum) += d_in[i];
    });
  });
  Queues[1].wait();

  assert(*d_sum == h_sum);

  free(d_sum, Queues[0]);
  free(d_in, Queues[0]);
  std::cout << "PASS" << std::endl;
  return 0;
}
