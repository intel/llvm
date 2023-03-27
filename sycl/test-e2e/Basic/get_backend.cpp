// REQUIRES: TEMPORARY_DISABLED
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
//
//==----------------- get_backend.cpp ------------------------==//
// This is a test of get_backend().
// Do not set ONEAPI_DEVICE_SELECTOR. We do not want the preferred
// backend.
//==----------------------------------------------------------==//

#include <iostream>
#include <sycl/backend_types.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

bool check(backend be) {
  switch (be) {
  case backend::opencl:
  case backend::ext_oneapi_level_zero:
  case backend::ext_oneapi_cuda:
  case backend::ext_oneapi_hip:
    return true;
  default:
    return false;
  }
  return false;
}

inline void return_fail() {
  std::cout << "Failed" << std::endl;
  exit(1);
}

int main() {
  for (const auto &plt : platform::get_platforms()) {
    if (check(plt.get_backend()) == false) {
      return_fail();
    }

    context c(plt);
    if (c.get_backend() != plt.get_backend()) {
      return_fail();
    }

    queue q(c, default_selector_v);
    if (q.get_backend() != plt.get_backend()) {
      return_fail();
    }

    auto device = q.get_device();
    if (device.get_backend() != plt.get_backend()) {
      return_fail();
    }

    unsigned char *HostAlloc = (unsigned char *)malloc_host(1, c);
    auto e = q.memset(HostAlloc, 42, 1);
    if (e.get_backend() != plt.get_backend()) {
      return_fail();
    }
    free(HostAlloc, c);
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
