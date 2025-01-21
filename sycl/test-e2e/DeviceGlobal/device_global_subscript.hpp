#pragma once

#include "common.hpp"

struct StructWithSubscript {
  int x[4];
  int &operator[](std::ptrdiff_t index) { return x[index]; }
};

device_global<int[4], TestProperties> DeviceGlobalVar1;
device_global<StructWithSubscript, TestProperties> DeviceGlobalVar2;

int test() {
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
