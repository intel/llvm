// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} %t.out
//
// Checks that no device is selected when no device of desired type is
// available.

#include <sycl/detail/core.hpp>

#include <iostream>

int main() {
  auto RejectEverything = [](const sycl::device &) { return -1; };
  try {
    sycl::device Device(RejectEverything);
  } catch (sycl::exception &e) {
    if (e.code() == sycl::errc::runtime &&
        std::string(e.what()).find("No device of requested type available.") !=
            std::string::npos) {
      return 0;
    }
    std::cerr << "Error. Incorrect exception was thrown." << std::endl;
    return 1;
  }
  std::cerr << "Error. A device is found." << std::endl;
  return 1;
}
