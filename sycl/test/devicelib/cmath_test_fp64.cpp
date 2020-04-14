// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath-fp64.o -o %t.out
#include <CL/sycl.hpp>
#include <iostream>
#include <cmath>

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

template <class T>
class DeviceCos;

template <class T>
void device_cos_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCos<T> >([=]() {
        res_access[0] = std::cos(0);
      });
    });
  }

  assert(result == 1);
}

template <class T>
class DeviceSin;

template <class T>
void device_sin_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSin<T> >([=]() {
        res_access[0] = std::sin(0);
      });
    });
  }

  assert(result == 0);
}

template <class T>
class DeviceLog;

template <class T>
void device_log_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog<T> >([=]() {
        res_access[0] = std::log(1);
      });
    });
  }

  assert(result == 0);
}

template <class T>
void device_cmath_test(s::queue &deviceQueue) {
  device_cos_test<T>(deviceQueue);
  device_sin_test<T>(deviceQueue);
  device_log_test<T>(deviceQueue);
}

int main() {
  s::queue deviceQueue;
  if (deviceQueue.get_device().has_extension("cl_khr_fp64")) {
    device_cmath_test<double>(deviceQueue);
    std::cout << "Pass" << std::endl;
  }
  return 0;
}
