#pragma once

#include "common.hpp"

device_global<int[4], TestProperties> DeviceGlobalVar;

int test() {
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
