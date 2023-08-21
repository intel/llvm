#pragma once

#include "common.hpp"

device_global<int, TestProperties> DeviceGlobalVar;

int test() {
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
