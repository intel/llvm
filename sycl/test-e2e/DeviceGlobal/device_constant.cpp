// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The OpenCL GPU backends do not currently support device_global backend
// calls.
//
// UNSUPPORTED: opencl && gpu

#include "common.hpp"

device_global<int, decltype(properties{device_constant})> DeviceGlobalVar;

int main() {
  queue Q;

  int HostVal = 42;
  Q.memcpy(DeviceGlobalVar, &HostVal);
  Q.wait();
  int OutVal = 0;

  {
    buffer<int, 1> OutBuf(&OutVal, 1);
    Q.submit([&](handler &CGH) {
      auto OutAcc = OutBuf.get_access<access::mode::write>(CGH);
      CGH.single_task([=]() { OutAcc[0] = DeviceGlobalVar.get(); });
    });
  }
  assert(OutVal == 42 && "Read value does not match.");
  return 0;
}
