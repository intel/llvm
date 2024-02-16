// RUN: %{build} -o %t.out
// RUN: env SYCL_PI_TRACE=-1 UR_L0_DEBUG=-1 ONEAPI_DEVICE_SELECTOR="*:fpga" %{run-unfiltered-devices} %t.out
// RUN: env SYCL_PI_TRACE=-1 UR_L0_DEBUG=-1 ONEAPI_DEVICE_SELECTOR="opencl:*" %{run-unfiltered-devices} %t.out
// RUN: env SYCL_PI_TRACE=-1 UR_L0_DEBUG=-1 %{run-unfiltered-devices} %t.out
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
  case backend::ext_oneapi_native_cpu:
    return true;
  default:
    return false;
  }
  return false;
}

inline void return_fail(int line = __builtin_LINE()) {
  std::cout << "Failed at line " << line << std::endl;
  exit(1);
}

int main() {
  for (const auto &plt : platform::get_platforms()) {
    auto plt_name = plt.get_info<info::platform::name>();
    std::cout << "MY: Platform: " << plt_name << std::endl;
    if (check(plt.get_backend()) == false) {
      return_fail();
    }

    context c(plt);
    if (c.get_backend() != plt.get_backend()) {
      return_fail();
    }

    auto devices = c.get_devices();
    std::cout << "Devices in context:" << std::endl;
    for (auto d : devices) {
      auto d_name = d.get_info<info::device::name>();
      std::cout << "MY: Device: " << d_name << std::endl;
      auto d_backend_version = d.get_info<info::device::backend_version>();
      std::cout << "MY: Device: " << d_backend_version << std::endl;
    }
    std::cout << "END Devices in context:" << std::endl;
    auto device = devices[0];

    auto d_name = device.get_info<info::device::name>();
    std::cout << "MY: Device: " << d_name << std::endl;
    auto d_backend_version = device.get_info<info::device::backend_version>();
    std::cout << "MY: Device: " << d_backend_version << std::endl;
    if (device.get_backend() != plt.get_backend()) {
      return_fail();
    }

    context ctx{device};
    queue q(ctx, device);
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
