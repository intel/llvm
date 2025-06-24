// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/platform.hpp>
#include <sycl/usm.hpp>
#include <vector>

// Test that a kernel can be run on a sub-device using a USM allocation from the
// parent device.

using namespace sycl;

class AddIdxKernel;

int main() {
  sycl::device parent_dev;
  auto partition_props =
      parent_dev.get_info<sycl::info::device::partition_properties>();
  if (partition_props.empty()) {
    std::cout << "Device does not support partitioning into sub-devices.\n";
    return 0;
  }

  // Partition the device into two sub-devices (if possible)
  std::vector<sycl::device> sub_devices;
  if (std::find(partition_props.begin(), partition_props.end(),
                sycl::info::partition_property::partition_by_affinity_domain) !=
      partition_props.end()) {
    sub_devices = parent_dev.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::numa);
  } else {
    std::cout << "partition_by_affinity_domain not supported.\n";
    return 0;
  }
  std::vector<sycl::device> all_devs;
  all_devs.push_back(parent_dev);
  all_devs.insert(all_devs.end(), sub_devices.begin(), sub_devices.end());
  sycl::context ctx(all_devs);

  constexpr size_t N = 16;
  std::vector<int> result(N, 0);

  int *data = sycl::malloc_device<int>(N, parent_dev, ctx);
  sycl::queue q(ctx, sub_devices[0]);
  q.fill(data, 1, N).wait();
  auto kb = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      ctx, {sub_devices[0]});
  q.submit([&](sycl::handler &h) {
     h.use_kernel_bundle(kb);
     h.parallel_for<AddIdxKernel>(
         sycl::range<1>(N), [=](sycl::id<1> idx) { data[idx] += idx[0]; });
   }).wait();
  q.memcpy(result.data(), data, N * sizeof(int)).wait();
  sycl::free(data, ctx);

  for (size_t j = 0; j < N; ++j) {
    if (result[j] != 1 + static_cast<int>(j)) {
      return -1;
    }
    std::cout << result[j] << " ";
  }

  return 0;
}
