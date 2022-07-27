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
  if (Q.is_host()) {
    std::cout << "Skipping test\n";
    return 0;
  }

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
