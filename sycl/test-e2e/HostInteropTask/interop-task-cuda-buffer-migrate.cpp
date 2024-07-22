// REQUIRES: cuda, cuda_dev_kit
//
// RUN: %{build} -o %t.out %cuda_options
// RUN: %{run} %t.out
//
// Test for buffer use in a context with multiple devices (all found
// root-devices)
//
// Make sure that memory migration works for buffers across devices in a context
// when using host tasks.
//

#include <cuda.h>
#include <iostream>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/interop_handle.hpp>

using namespace sycl;

int main() {

  int Data = 0;
  int Result = 0;
  buffer<int, 1> buf(&Data, range<1>(1));

  const auto &Devices =
      platform(gpu_selector_v).get_devices(info::device_type::gpu);
  std::cout << Devices.size() << " devices found" << std::endl;

  context C(Devices);

  int Index = 0;
  for (auto D : Devices) {
    std::cout << "Using on device " << Index << ": "
              << D.get_info<info::device::name>() << std::endl;

    queue Q(C, D);
    Q.submit([&](handler &cgh) {
      accessor acc{buf, cgh, read_write};
      cgh.host_task([=](interop_handle ih) {
        auto ptr = ih.get_native_mem<backend::ext_oneapi_cuda>(acc);
        int tmp = 0;
        cuMemcpyDtoH(&tmp, ptr, sizeof(int));
        tmp++;
        cuMemcpyHtoD(ptr, &tmp, sizeof(int));
      });
    });
    Q.wait();
    ++Index;
  }

  auto host_acc = buf.get_host_access();
  auto passed = (host_acc[0] == Index);
  std::cout << "Checking result on host: " << (passed ? "passed" : "FAILED")
            << std::endl;
  std::cout << host_acc[0] << " ?= " << Index << std::endl;
  return !passed;
}
