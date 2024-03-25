// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out
//
// Checks that no device is selected when no device of desired type is
// available.

#include <sycl/detail/core.hpp>

#include <iostream>

class RejectEverything : public sycl::device_selector {
public:
  int operator()(const sycl::device &Device) const final {
    // Negative value means that a device must not be selected
    return -1;
  }
};

int main() {
  RejectEverything Selector;
  try {
    sycl::device Device(Selector);
  } catch (sycl::runtime_error &E) {
    return 0;
  }
  std::cerr << "Error. A device is found." << std::endl;
  return 1;
}
