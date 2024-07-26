// REQUIRES: cuda, cuda_dev_kit
// RUN: %{build} -o %t.out %cuda_options
// RUN: %{run} %t.out
#include <cuda.h>

#include <iostream>
#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/interop_handle.hpp>

using namespace sycl;

int main() {

  int Data = 0;
  int Result = 0;
  buffer<int, 1> Buf(&Data, range<1>(1));

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

    Q.submit([&](handler &CGH) {
      accessor Acc{Buf, CGH, read_write};
      CGH.ext_codeplay_enqueue_native_command([=](interop_handle IH) {
        auto Ptr = IH.get_native_mem<backend::ext_oneapi_cuda>(Acc);
        auto Stream = IH.get_native_queue<backend::ext_oneapi_cuda>();
        int Tmp = 0;
        cuMemcpyDtoHAsync(&Tmp, Ptr, sizeof(int), Stream);
        cuStreamSynchronize(Stream);
        Tmp++;
        cuMemcpyHtoDAsync(Ptr, &Tmp, sizeof(int), Stream);
      });
    });
    ++Index;
  }

  auto HostAcc = Buf.get_host_access();
  auto Passed = (HostAcc[0] == Index);
  std::cout << "Checking result on host: " << (Passed ? "Passed" : "FAILED")
            << std::endl;
  std::cout << HostAcc[0] << " ?= " << Index << std::endl;
  return !Passed;
}
