// REQUIRES: TEMPORARY_DISABLED
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
//
//==----------------- get_backend.cpp ------------------------==//
// This is a test of get_backend().
// Do not set SYCL_DEVICE_FILTER. We do not want the preferred
// backend.
//==----------------------------------------------------------==//

#include <CL/sycl.hpp>
#include <CL/sycl/backend_types.hpp>
#include <iostream>

using namespace cl::sycl;

bool check(backend be) {
  switch (be) {
  case backend::opencl:
  case backend::ext_oneapi_level_zero:
  case backend::ext_oneapi_cuda:
  case backend::ext_oneapi_hip:
  case backend::host:
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
    if (!plt.is_host()) {
      if (check(plt.get_backend()) == false) {
        return_fail();
      }

      context c(plt);
      if (c.get_backend() != plt.get_backend()) {
        return_fail();
      }

      default_selector sel;
      queue q(c, sel);
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
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
