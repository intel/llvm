#include <iostream>
#include <sycl/sycl.hpp>

int main(int, char **) {
  for (const auto &device : sycl::device::get_devices()) {
    sycl::queue queue(device);
    std::cout << "Running on " << device.get_info<sycl::info::device::name>()
              << "\n";
    std::cout << "usm_device_allocations: "
              << device.has(sycl::aspect::usm_device_allocations) << std::endl;
    float *d_buf = sycl::malloc_device<float>(1, queue);
    std::cout << "malloc_device returned: " << d_buf << std::endl;
  }

  return 0;
}
