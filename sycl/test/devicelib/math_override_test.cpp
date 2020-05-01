// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath.o -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>

#include "math_utils.hpp"
namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

// Dummy function provided by user to override device library
// version.
SYCL_EXTERNAL
extern "C" float sinf(float x) { return x + 100.f; }

class DeviceTest;

void device_test() {
  s::queue deviceQueue;
  s::range<1> numOfItems{1};
  float result_sin = 0;
  float result_cos = 0;
  {
    s::buffer<float, 1> buffer1(&result_sin, numOfItems);
    s::buffer<float, 1> buffer2(&result_cos, numOfItems);
    deviceQueue.submit([&](s::handler &cgh) {
      auto res_access_sin = buffer1.get_access<sycl_write>(cgh);
      auto res_access_cos = buffer2.get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceTest>([=]() {
        // Should use the sin function defined by user, device
        // library version should be ignored here
        res_access_sin[0] = sinf(0.f);
        res_access_cos[0] = cosf(0.f);
      });
    });
  }

  assert(approx_equal_fp(result_sin, 100.f) && approx_equal_fp(result_cos, 1.f));
}

int main() {
  device_test();
  std::cout << "Pass" << std::endl;
  return 0;
}
