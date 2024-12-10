// DEFINE: %{cpp23} = %if cl_options %{/std:c++23%} %else %{-std=c++23%}

// RUN: %{build} %{cpp23} -o %t.out
// RUN: %{run} %t.out
//
// UNSUPPORTED: opencl && gpu
// UNSUPPORTED-TRACKER: GSD-4287
//
// Tests the copy ctor on device_global without device_image_scope.

#include <sycl/detail/core.hpp>

namespace oneapiext = sycl::ext::oneapi::experimental;

oneapiext::device_global<const int> DGInit1{3};
oneapiext::device_global<const int> DGCopy1{DGInit1};

oneapiext::device_global<int> DGInit2{4};
oneapiext::device_global<int> DGCopy2{DGInit2};

oneapiext::device_global<float> DGInit3{5.0f};
oneapiext::device_global<int> DGCopy3{DGInit3};

oneapiext::device_global<const int, decltype(oneapiext::properties{
                                        oneapiext::device_image_scope})>
    DGInit4{6};
oneapiext::device_global<const int> DGCopy4{DGInit4};

oneapiext::device_global<const int> DGInit5{7};
oneapiext::device_global<const int, decltype(oneapiext::properties{
                                        oneapiext::host_access_read})>
    DGCopy5{DGInit5};
oneapiext::device_global<const int, decltype(oneapiext::properties{
                                        oneapiext::device_constant})>
    DGInit6{8};
oneapiext::device_global<const int> DGCopy6{DGInit6};

int main() {
  sycl::queue Q;

  int ReadVals[12] = {0, 0};
  {
    sycl::buffer<int, 1> ReadValsBuff{ReadVals, 10};

    Q.submit([&](sycl::handler &CGH) {
       sycl::accessor ReadValsAcc{ReadValsBuff, CGH, sycl::write_only};
       CGH.single_task([=]() {
         ReadValsAcc[0] = DGInit1.get();
         ReadValsAcc[1] = DGCopy1.get();
         ReadValsAcc[2] = DGInit2.get();
         ReadValsAcc[3] = DGCopy2.get();
         ReadValsAcc[4] = DGInit3.get();
         ReadValsAcc[5] = DGCopy3.get();
         ReadValsAcc[6] = DGInit4.get();
         ReadValsAcc[7] = DGCopy4.get();
         ReadValsAcc[8] = DGInit5.get();
         ReadValsAcc[9] = DGCopy5.get();
         ReadValsAcc[10] = DGInit6.get();
         ReadValsAcc[11] = DGCopy6.get();
       });
     }).wait_and_throw();
  }

  assert(ReadVals[0] == 3);
  assert(ReadVals[1] == 3);
  assert(ReadVals[2] == 4);
  assert(ReadVals[3] == 4);
  assert(ReadVals[4] == 5);
  assert(ReadVals[5] == 5);
  assert(ReadVals[6] == 6);
  assert(ReadVals[7] == 6);
  assert(ReadVals[8] == 7);
  assert(ReadVals[9] == 7);
  assert(ReadVals[10] == 8);
  assert(ReadVals[11] == 8);

  return 0;
}
