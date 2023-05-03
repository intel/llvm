// TODO: device_global without the device_image_scope property is not currently
//       initialized on device. Enable the following test cases when it is
//       supported.
// RUNx: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUNx: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -fsycl-device-code-split=per_source -DUSE_DEVICE_IMAGE_SCOPE %s -o %t_dev_img_scope.out
// RUN: %CPU_RUN_PLACEHOLDER %t_dev_img_scope.out
// RUN: %GPU_RUN_PLACEHOLDER %t_dev_img_scope.out
// RUN: %ACC_RUN_PLACEHOLDER %t_dev_img_scope.out
//
// CPU and accelerators are not currently guaranteed to support the required
// extensions they are disabled until they are.
// UNSUPPORTED: cpu, accelerator
//
// Tests the passthrough of operators on device_global.
// NOTE: USE_DEVICE_IMAGE_SCOPE needs both kernels to be in the same image so
//       we set -fsycl-device-code-split=per_source.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

#ifdef USE_DEVICE_IMAGE_SCOPE
device_global<int, decltype(properties{device_image_scope})> DeviceGlobalVar;
#else
device_global<int> DeviceGlobalVar;
#endif

int main() {
  queue Q;

  Q.single_task([]() {
     DeviceGlobalVar = 2;
     DeviceGlobalVar += 3;
     DeviceGlobalVar = DeviceGlobalVar * DeviceGlobalVar;
     DeviceGlobalVar = DeviceGlobalVar - 3;
     DeviceGlobalVar = 25 - DeviceGlobalVar;
   }).wait();

  int Out = 0;
  {
    buffer<int, 1> OutBuf{&Out, 1};
    Q.submit([&](handler &CGH) {
      auto OutAcc = OutBuf.get_access<access::mode::write>(CGH);
      CGH.single_task([=]() { OutAcc[0] = DeviceGlobalVar; });
    });
  }
  assert(Out == 3 && "Read value does not match.");
  return 0;
}
