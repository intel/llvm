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
// Tests operator-> on device_global.
// NOTE: USE_DEVICE_IMAGE_SCOPE needs both kernels to be in the same image so
//       we set -fsycl-device-code-split=per_source.

#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::oneapi::experimental;

struct StructWithMember {
  int x;
  int getX() { return x; }
};

struct StructWithDeref {
  StructWithMember y[1];
  StructWithMember *operator->() { return y; }
};

#ifdef USE_DEVICE_IMAGE_SCOPE
device_global<StructWithMember *, decltype(properties{device_image_scope})>
    DeviceGlobalVar1;
device_global<StructWithDeref, decltype(properties{device_image_scope})>
    DeviceGlobalVar2;
#else
device_global<StructWithMember *> DeviceGlobalVar1;
device_global<StructWithDeref> DeviceGlobalVar2;
#endif

int main() {
  queue Q;

  StructWithMember *DGMem = malloc_device<StructWithMember>(1, Q);

  Q.single_task([=]() {
     DeviceGlobalVar1 = DGMem;
     DeviceGlobalVar1->x = 1234;
     DeviceGlobalVar2->x = 4321;
   }).wait();

  int Out[2] = {0, 0};
  {
    buffer<int, 1> OutBuf{Out, 2};
    Q.submit([&](handler &CGH) {
      auto OutAcc = OutBuf.get_access<access::mode::write>(CGH);
      CGH.single_task([=]() {
        OutAcc[0] = DeviceGlobalVar1->getX();
        OutAcc[1] = DeviceGlobalVar2->getX();
      });
    });
  }
  free(DGMem, Q);

  assert(Out[0] == 1234 && "First value does not match.");
  assert(Out[1] == 4321 && "Second value does not match.");
  return 0;
}
