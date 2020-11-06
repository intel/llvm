// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main(int argc, char const *argv[]) {
  vector_class<platform> platforms = platform::get_platforms();
  if (platforms.size() < 2) {
    std::cout << " This test requires at least 2 SYCL platforms to be present "
                 "on the system. Skipping it"
              << std::endl;
    return 0;
  }
  try {
    vector_class<device> all_platforms_devices;
    for (unsigned i = 0; i < platforms.size(); i++) {
      vector_class<device> one_platform_devices = platforms[i].get_devices();
      all_platforms_devices.push_back(one_platform_devices[0]);
    }
    context all_context = context(all_platforms_devices);
    std::cerr << "Test failed: exception wasn't thrown" << std::endl;
    return 1;
  } catch (runtime_error &E) {
    if (string_class(E.what()).find(
            "Can't add devices across platforms to a single context.") ==
        string_class::npos) {
      std::cerr << "Received error is incorrect" << std::endl;
      return 1;
    }
  }
}
