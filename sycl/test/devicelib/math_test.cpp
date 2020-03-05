// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath.o -o %t.out
#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

class DeviceSin;

void device_sin_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float  result_f = -1;
  {
    s::buffer<float, 1> buffer1(&result_f, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access1 = buffer1.get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSin>([=]() {
        res_access1[0] = sinf(0);
      });
    });
  }

  assert(result_f == 0);
}

class DeviceCos;

void device_cos_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float  result_f = -1;
  {
    s::buffer<float, 1> buffer1(&result_f, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access1 = buffer1.get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCos>([=]() {
        res_access1[0] = cosf(0);
      });
    });
  }

  assert(result_f == 1);
}

class DeviceLog;

void device_log_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float  result_f = -1;
  {
    s::buffer<float, 1> buffer1(&result_f, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access1 = buffer1.get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog>([=]() {
        res_access1[0] = logf(1);
      });
    });
  }

  assert(result_f == 0);
}

void device_math_test(s::queue &deviceQueue) {
  device_cos_test(deviceQueue);
  device_sin_test(deviceQueue);
  device_log_test(deviceQueue);
}

int main() {
  s::queue deviceQueue;
  device_math_test(deviceQueue);
  std::cout << "Pass" << std::endl;
  return 0;
}
