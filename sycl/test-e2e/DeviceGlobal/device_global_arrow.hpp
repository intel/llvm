#pragma once

#include "common.hpp"

#include <sycl/usm.hpp>

struct StructWithMember {
  int x;
  int getX() { return x; }
};

struct StructWithDeref {
  StructWithMember y[1];
  StructWithMember *operator->() { return y; }
};

device_global<StructWithMember *, TestProperties> DeviceGlobalVar1;
device_global<StructWithDeref, TestProperties> DeviceGlobalVar2;

int test() {
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
