// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env CreateMultipleSubDevices=2 env EnableTimestampPacket=1 \
// RUN: env NEOReadDebugKeys=1 env SYCL_DEVICE_FILTER="gpu" %t.out
//
// XFAIL: cuda

#include <CL/sycl.hpp>

#include <iostream>
using namespace sycl;

int main() {
  std::vector<device> devices = device::get_devices(info::device_type::gpu);
  for (int i = 0; i < devices.size(); ++i) {
    std::cout << std::endl;
    std::cout << "Device " << i << ": "
              << devices[i].get_info<info::device::name>() << std::endl;
    std::cout << "  Platform: "
              << devices[i].get_platform().get_info<info::platform::name>()
              << std::endl;
    std::cout << "  EUs: "
              << devices[i].get_info<info::device::max_compute_units>()
              << std::endl;
    std::vector<device> subdevices;
    try {
      std::cout << "  Tiles: ";
      subdevices =
          devices[i]
              .create_sub_devices<
                  info::partition_property::partition_by_affinity_domain>(
                  info::partition_affinity_domain::numa);
      std::cout << subdevices.size() << std::endl;
    } catch (std::exception) {
      std::cout << "Error -- cannot create subdevices" << std::endl;
    }
    for (int j = 0; j < subdevices.size(); ++j) {
      std::cout << "  Tile " << j << std::endl;
      std::cout << "    EUs: "
                << subdevices[j].get_info<info::device::max_compute_units>()
                << std::endl;
    }
  }
}
