// RUNx: %{build} -o %t.out
// RUNx: %{run} %t.out
//
// RUN: %{build} -fsycl-device-code-split=per_source -DUSE_DEVICE_IMAGE_SCOPE -o %t_dev_img_scope.out
// RUN: %{run} %t_dev_img_scope.out

// Tests that device_global with no kernel uses can be copied to and from.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

#ifdef USE_DEVICE_IMAGE_SCOPE
device_global<int, decltype(properties(device_image_scope))> MemcpyDeviceGlobal;
device_global<int, decltype(properties(device_image_scope))> CopyDeviceGlobal;
#else
device_global<int> MemcpyDeviceGlobal;
device_global<int> CopyDeviceGlobal;
#endif

int main() {
  queue Q;
  int MemcpyWrite = 42, CopyWrite = 24, MemcpyRead = 1, CopyRead = 2;

  // Copy from device globals before having written anything. This should act as
  // having zero-initialized values.
  Q.memcpy(&MemcpyRead, MemcpyDeviceGlobal);
  Q.copy(CopyDeviceGlobal, &CopyRead);
  Q.wait();
  assert(MemcpyRead == 0);
  assert(CopyRead == 0);

  // Write to device globals and then read their values.
  Q.memcpy(MemcpyDeviceGlobal, &MemcpyWrite);
  Q.copy(&CopyWrite, CopyDeviceGlobal);
  Q.wait();
  Q.memcpy(&MemcpyRead, MemcpyDeviceGlobal);
  Q.copy(CopyDeviceGlobal, &CopyRead);
  Q.wait();
  assert(MemcpyRead == MemcpyWrite);
  assert(CopyRead == CopyWrite);

  return 0;
}
