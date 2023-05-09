// TODO: device_global without the device_image_scope property is not currently
//       initialized on device. Enable the following test cases when it is
//       supported.
// RUNx: %{build} -o %t.out
// RUNx: %{run} %t.out
//
// RUN: %{build} -fsycl-device-code-split=per_source -DUSE_DEVICE_IMAGE_SCOPE -o %t_dev_img_scope.out
// RUN: %{run} %t_dev_img_scope.out
//
// CPU and accelerators are not currently guaranteed to support the required
// extensions they are disabled until they are.
// UNSUPPORTED: cpu, accelerator
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
