// RUN: %{build} -o %t.out
// FPGA RT returns random CL_INVALID_CONTEXT in some configurations, tracked
// internally. Avoid FPGA devices until that is fixed.
// RUN: env ONEAPI_DEVICE_SELECTOR="*:gpu;*:cpu" %{run-unfiltered-devices} %t.out
//
//==----------------- get_backend.cpp ------------------------==//
// This is a test of get_backend().
// Do not set ONEAPI_DEVICE_SELECTOR. We do not want the preferred
// backend.
//==----------------------------------------------------------==//

#include <iostream>
#include <sycl/backend_types.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl;

bool check(backend be) {
  switch (be) {
  case backend::opencl:
  case backend::ext_oneapi_level_zero:
  case backend::ext_oneapi_cuda:
  case backend::ext_oneapi_hip:
  case backend::ext_oneapi_native_cpu:
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

    auto device = c.get_devices()[0];
    if (device.get_backend() != plt.get_backend()) {
      return_fail();
    }

    queue q(c, device);
    if (q.get_backend() != plt.get_backend()) {
      return_fail();
    }

    buffer<int, 1> buf{range<1>(1)};
    event e = q.submit([&](handler &cgh) {
      auto acc = buf.get_access<access::mode::read_write>(cgh);
      cgh.fill(acc, 0);
    });
    if (e.get_backend() != plt.get_backend()) {
      return_fail();
    }
  }
  std::cout << "Passed" << std::endl;
  return 0;
}
