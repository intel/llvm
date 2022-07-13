// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_FILTER=%sycl_be %t.out
#include <iostream>
#include <sycl/sycl.hpp>

using namespace cl::sycl;

int main(int argc, char const *argv[]) {
  std::vector<platform> platforms = platform::get_platforms();
  if (platforms.size() < 2) {
    std::cout << " This test requires at least 2 SYCL platforms to be present "
                 "on the system. Skipping it"
              << std::endl;
    return 0;
  }
  try {
    std::vector<device> all_platforms_devices;
    for (unsigned i = 0; i < platforms.size(); i++) {
      std::vector<device> one_platform_devices = platforms[i].get_devices();
      all_platforms_devices.push_back(one_platform_devices[0]);
    }
    context all_context = context(all_platforms_devices);
    std::cerr << "Test failed: exception wasn't thrown" << std::endl;
    return 1;
  } catch (runtime_error &E) {
    if (std::string(E.what()).find(
            "Can't add devices across platforms to a single context.") ==
        std::string::npos) {
      std::cerr << "Received error is incorrect" << std::endl;
      return 1;
    }
  }
}
