// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// The OpenCL GPU backends do not currently support device_global backend
// calls.
// UNSUPPORTED: opencl && gpu
//
// Tests static device_global access through device kernels.

#include "common.hpp"

static device_global<int[4], TestProperties> DeviceGlobalVar;

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
