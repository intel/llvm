// RUN: %{build} -std=c++23 -o %t.out
// RUN: %{run} %t.out
//
// The OpenCL GPU backends do not currently support device_global backend
// calls.
// UNSUPPORTED: opencl && gpu
//
// Tests the copy ctor on device_global without device_image_scope.

#include <sycl/detail/core.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;

oneapiext::device_global<const int> DGInit{3};
oneapiext::device_global<const int> DGCopy{DGInit};

int main() {
  sycl::queue Q;

  int ReadVals[2] = {0, 0};
  {
    sycl::buffer<int, 1> ReadValsBuff{ReadVals, 2};

    Q.submit([&](sycl::handler &CGH) {
       sycl::accessor ReadValsAcc{ReadValsBuff, CGH, sycl::write_only};
       CGH.single_task([=]() {
         ReadValsAcc[0] = DGInit.get();
         ReadValsAcc[1] = DGCopy.get();
       });
     }).wait_and_throw();
  }

  assert(ReadVals[0] == 3);
  assert(ReadVals[1] == 3);

  return 0;
}
