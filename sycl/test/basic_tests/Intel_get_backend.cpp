// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
//
// CHECK: opencl
//
//==--------------- Intel_get_backend.cpp --------------------==//
// This is a test of get_backend().
// Also prints handy info about the system.
// Do not set SYCL_BE.  We do not want the preferred backend.
//==----------------------------------------------------------==//

#include <CL/sycl.hpp>
#include <CL/sycl/backend_types.hpp>
#include <iostream>

using namespace cl::sycl;

std::string get_string(backend be) {
  switch (be) {
  case backend::opencl:
    return std::string("opencl");
    break;
  case backend::level0:
    return std::string("level0");
    break;
  case backend::cuda:
    return std::string("cuda");
    break;
  default:
    return std::string("unknown");
  }
}

int main() {
  int pindex = 1;
  for (const auto &plt : platform::get_platforms()) {
    std::cout << "Platform " << pindex++ << ": "
              << plt.get_info<info::platform::name>() << ": "
              << plt.get_info<info::platform::version>();
    if (!plt.is_host()) {
      std::cout << "  BE: " << get_string(plt.get_backend());
    }
    std::cout << std::endl;
    int dindex = 1;
    for (const auto &dev : plt.get_devices()) {
      std::cout << "  "
                << "Device " << dindex++ << ": "
                << dev.get_info<info::device::name>()
                << " (" << dev.get_info<info::device::driver_version>() << "): "
                << dev.get_info<info::device::version>()
                << std::endl;
    }
  }
  return 0;
}
