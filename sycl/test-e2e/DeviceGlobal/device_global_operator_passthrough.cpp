// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// RUN: %{build} -fsycl-device-code-split=per_source -DUSE_DEVICE_IMAGE_SCOPE -o %t_dev_img_scope.out
// RUN: %{run} %t_dev_img_scope.out
//
// The HIP backend does not currently support device_global backend calls.
// UNSUPPORTED: hip
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
