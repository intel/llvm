// REQUIRES: cuda || hip || level_zero
// RUN:  %{build} -o %t.out
// RUN:  %{run} %t.out

#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

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

  auto *arr1 = malloc<int>(2, Queues[1], usm::alloc::device);

  // Calling fill on Devs[1] data with Devs[0] queue requires P2P enabled.
  Queues[0].fill(arr1, 2, 2).wait();

  // Access/write Devs[1] data with Devs[0] queue.
  Queues[0]
      .submit([&](handler &cgh) {
        auto myRange = range<1>(1);
        auto myKernel = ([=](id<1> idx) { arr1[0] *= 2; });

        cgh.parallel_for<class p2p_access>(myRange, myKernel);
      })
      .wait();

  int2 out;

  Queues[0].memcpy(&out, arr1, 2 * sizeof(int)).wait();
  assert(out[0] == 4);
  assert(out[1] == 2);

  sycl::free(arr1, Queues[1]);

  Devs[0].ext_oneapi_disable_peer_access(Devs[1]);
  std::cout << "PASS" << std::endl;
  return 0;
}
