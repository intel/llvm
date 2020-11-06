// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
//
//==----------------- get_backend.cpp ------------------------==//
// This is a test of get_backend().
// Do not set SYCL_BE.  We do not want the preferred backend.
//==----------------------------------------------------------==//

#include <CL/sycl.hpp>
#include <CL/sycl/backend_types.hpp>
#include <iostream>

using namespace cl::sycl;

bool check(backend be) {
  switch (be) {
  case backend::opencl:
  case backend::level_zero:
  case backend::cuda:
  case backend::host:
    return true;
  default:
    return false;
  }
  return false;
}

int main() {
  for (const auto &plt : platform::get_platforms()) {
    if (!plt.is_host()) {
      if (check(plt.get_backend()) == false) {
        std::cout << "Failed" << std::endl;
        return 1;
      }
    }
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
