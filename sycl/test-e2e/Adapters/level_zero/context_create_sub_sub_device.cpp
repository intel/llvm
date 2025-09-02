// REQUIRES:  arch-intel_gpu_pvc, level_zero
// UNSUPPORTED: gpu-intel-pvc-1T
// UNSUPPORTED-TRACKER: GSD-9121

// DEFINE: %{setup_env} = env ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE ZE_AFFINITY_MASK=0 ZEX_NUMBER_OF_CCS=0:4
// RUN: %{build} -o %t.out
// RUN: %{setup_env} %{run} %t.out

// Check that context can be created successfully when sub-sub-devices are
// exposed.
#include <iostream>
#include <sycl/detail/core.hpp>
#include <vector>

using namespace sycl;

int main() {
  std::cout << "[info] start context_create_sub_sub_device test" << std::endl;
  device d;
  std::vector<device> subsubdevices;

  auto subdevices = d.create_sub_devices<
      info::partition_property::partition_by_affinity_domain>(
      info::partition_affinity_domain::next_partitionable);
  std::cout << "[info] sub device size = " << subdevices.size() << std::endl;

  for (auto &subdev : subdevices) {
    subsubdevices = subdev.create_sub_devices<
        info::partition_property::ext_intel_partition_by_cslice>();

    std::cout << "[info] sub-sub device size = " << subsubdevices.size()
              << std::endl;
  }

  // Create contexts
  context ctx1(d);
  context ctx2(subdevices);
  context ctx3(subsubdevices);

  std::cout << "[info] contexts created successfully" << std::endl;
  return 0;
}
