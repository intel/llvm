// REQUIRES: cuda
// RUN: %{build} -o %t.out -lcuda
// RUN: %{run} %t.out
#include <cuda.h>

#include <iostream>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/detail/host_task_impl.hpp>

using namespace sycl;

int main() {

  int Data = 0;
  int Result = 0;
  buffer<int, 1> buf(&Data, range<1>(1));

  const auto &Devices =
      platform(gpu_selector_v).get_devices(info::device_type::gpu);
  std::cout << Devices.size() << " devices found" << std::endl;
  context C(Devices);
  std::vector<queue> Queues;
  for (auto &Dev : Devices)
    Queues.emplace_back(C, Dev);

  int Index = 0;
  for (auto Q : Queues) {
    std::cout << "Using on device " << Index << ": "
              << Q.get_device().get_info<info::device::name>() << std::endl;

    Q.submit([&](handler &cgh) {
      accessor acc{buf, cgh, read_write};
      cgh.sycl_ext_oneapi_enqueue_custom_operation([=](interop_handle ih) {
        auto ptr = ih.get_native_mem<backend::ext_oneapi_cuda>(acc);
        auto stream = ih.get_native_queue<backend::ext_oneapi_cuda>();
        int tmp = 0;
        cuMemcpyDtoHAsync(&tmp, ptr, sizeof(int), stream);
        cuStreamSynchronize(stream);
        tmp++;
        cuMemcpyHtoDAsync(ptr, &tmp, sizeof(int), stream);
      });
    });
    ++Index;
  }

  auto host_acc = buf.get_host_access();
  auto passed = (host_acc[0] == Index);
  std::cout << "Checking result on host: " << (passed ? "passed" : "FAILED")
            << std::endl;
  std::cout << host_acc[0] << " ?= " << Index << std::endl;
  return !passed;
}
