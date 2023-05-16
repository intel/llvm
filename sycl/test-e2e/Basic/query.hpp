#include <sycl/sycl.hpp>

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
    std::cout << "  Max sub-devices: "
              << devices[i].get_info<info::device::partition_max_sub_devices>()
              << std::endl;

    std::vector<device> subdevices;
    if (devices[i].get_info<info::device::partition_max_sub_devices>() > 0 &&
        devices[i].get_info<info::device::max_compute_units>() > 0) {
      try {
        std::cout << "  Subdevices: ";
        subdevices =
            devices[i]
                .create_sub_devices<
                    info::partition_property::partition_by_affinity_domain>(
                    info::partition_affinity_domain::numa);
        std::cout << subdevices.size() << std::endl;
      } catch (exception e) {
        std::cout << "Error -- cannot create subdevices: " << e.what()
                  << std::endl;
        return 1;
      }
      for (int j = 0; j < subdevices.size(); ++j) {
        std::cout << "  Subdevice " << j << std::endl;
        std::cout << "    EUs: "
                  << subdevices[j].get_info<info::device::max_compute_units>()
                  << std::endl;
      }
    }
  }
  return 0;
}
