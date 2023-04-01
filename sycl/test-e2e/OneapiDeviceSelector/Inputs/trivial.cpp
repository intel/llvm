#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  for (auto &d : device::get_devices()) {
    std::cout << "Device: " << d.get_info<info::device::name>() << std::endl;
  }
  return 0;
}
