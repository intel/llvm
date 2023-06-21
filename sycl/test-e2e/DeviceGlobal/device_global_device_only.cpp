// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// RUN: %{build} -fsycl-device-code-split=per_source -DUSE_DEVICE_IMAGE_SCOPE -o %t_dev_img_scope.out
// RUN: %{run} %t_dev_img_scope.out
//
// The HIP backend does not currently support device_global backend calls.
// UNSUPPORTED: hip
//
// Tests basic device_global access through device kernels.
// NOTE: USE_DEVICE_IMAGE_SCOPE needs both kernels to be in the same image so
//       we set -fsycl-device-code-split=per_source.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

#ifdef USE_DEVICE_IMAGE_SCOPE
device_global<int[4], decltype(properties{device_image_scope})> DeviceGlobalVar;
#else
device_global<int[4]> DeviceGlobalVar;
#endif

int main() {
  queue Q;

  Q.single_task([=]() { DeviceGlobalVar.get()[0] = 42; });
  // Make sure that the write happens before subsequent read
  Q.wait();

  int OutVal = 0;
  {
    buffer<int, 1> OutBuf(&OutVal, 1);
    Q.submit([&](handler &CGH) {
      auto OutAcc = OutBuf.get_access<access::mode::write>(CGH);
      CGH.single_task([=]() { OutAcc[0] = DeviceGlobalVar.get()[0]; });
    });
  }
  assert(OutVal == 42 && "Read value does not match.");
  return 0;
}
