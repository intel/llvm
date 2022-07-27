// TODO: device_global without the device_image_scope property is not currently
//       initialized on device. Enable the following test cases when it is
//       supported.
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_source -DUSE_DEVICE_IMAGE_SCOPE %s -o %t_dev_img_scope.out
// RUN: %CPU_RUN_PLACEHOLDER %t_dev_img_scope.out
// RUN: %GPU_RUN_PLACEHOLDER %t_dev_img_scope.out
// RUN: %ACC_RUN_PLACEHOLDER %t_dev_img_scope.out
//
// Currently fails for CPUs due to missing support for the SPIR-V extension.
// Currently crashes on accelerators.
// XFAIL: cpu, accelerator
//
// Tests operator[] on device_global.
// NOTE: USE_DEVICE_IMAGE_SCOPE needs both kernels to be in the same image so
//       we set -fsycl-device-code-split=per_source.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

struct StructWithSubscript {
  int x[4];
  int &operator[](std::ptrdiff_t index) { return x[index]; }
};

#ifdef USE_DEVICE_IMAGE_SCOPE
device_global<int[4], decltype(properties{device_image_scope})>
    DeviceGlobalVar1;
device_global<StructWithSubscript, decltype(properties{device_image_scope})>
    DeviceGlobalVar2;
#else
device_global<int[4]> DeviceGlobalVar1;
device_global<StructWithSubscript> DeviceGlobalVar2;
#endif

int main() {
  queue Q;
  if (Q.is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }

  Q.single_task([]() {
     DeviceGlobalVar1[2] = 1234;
     DeviceGlobalVar2[1] = 4321;
   }).wait();

  int Out[2] = {0, 0};
  {
    buffer<int, 1> OutBuf{Out, 2};
    Q.submit([&](handler &CGH) {
      auto OutAcc = OutBuf.get_access<access::mode::write>(CGH);
      CGH.single_task([=]() {
        OutAcc[0] = DeviceGlobalVar1[2];
        OutAcc[1] = DeviceGlobalVar2[1];
      });
    });
  }
  assert(Out[0] == 1234 && "First value does not match.");
  assert(Out[1] == 4321 && "Second value does not match.");
  return 0;
}
