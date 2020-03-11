// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath-fp64.o -o %t.out
#include <CL/sycl.hpp>
#include <cmath>
#include <iostream>
#include "math_utils.hpp"

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

#define TEST_NUM 38

double ref[TEST_NUM] = {
1, 0, 0, 0, 0, 0, 0, 1, 1, 0.5,
0, 2, 0, 0, 1, 0, 2, 0, 0, 0,
0, 0, 1, 0, 1, 2, 0, 1, 2, 5,
0, 0, 0, 0, 0.5, 0.5, NAN, NAN,};

double refIptr = 1;

template <class T>
void device_cmath_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{TEST_NUM};
  T result[TEST_NUM] = {-1};
  // Variable exponent is an integer value to store the exponent in frexp function
  int exponent = -1;
  // Variable iptr stores the integral part of float point in modf function
  T iptr = -1;
  // Variable quo stores the sign and some bits of x/y in remquo function
  int quo = -1;
  {
    s::buffer<T, 1> buffer1(result, numOfItems);
    s::buffer<int, 1> buffer2(&exponent, s::range<1>{1});
    s::buffer<T, 1> buffer3(&iptr, s::range<1>{1});
    s::buffer<int, 1> buffer4(&quo, s::range<1>{1});
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto exp_access = buffer2.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer3.template get_access<sycl_write>(cgh);
      auto quo_access = buffer4.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceMathTest>([=]() {
        int i = 0;
        {
          T a = 0;
          res_access[i++] = std::cos(a);
        }
        {
          T a = 0;
          res_access[i++] = std::sin(a);
        }
        {
          T a = 1;
          res_access[i++] = std::log(a);
        }
        {
          T a = 1;
          res_access[i++] = std::acos(a);
        }
        {
          T a = 0;
          res_access[i++] = std::asin(a);
        }
        {
          T a = 0;
          res_access[i++] = std::atan(a);
        }
        {
          T a = 0;
          T b = 1;
          res_access[i++] = std::atan2(a, b);
        }
        {
          T a = 0;
          res_access[i++] = std::cosh(a);
        }
        {
          T a = 0;
          res_access[i++] = std::exp(a);
        }
        {
          T a = 1.5;
          T b = 1;
          res_access[i++] = std::fmod(a, b);
        }
        {
          T a = 0;
          res_access[i++] = std::frexp(a, &exp_access[0]);
        }
        {
          T a = 1;
          res_access[i++] = std::ldexp(a, 1);
        }
        {
          T a = 1;
          res_access[i++] = std::log10(a);
        }
        {
          T a = 1;
          res_access[i++] = std::modf(a, &iptr_access[0]);
        }
        {
          T a = 1;
          T b = 1;
          res_access[i++] = std::pow(a, b);
        }
        {
          T a = 0;
          res_access[i++] = std::sinh(a);
        }
        {
          T a = 4;
          res_access[i++] = std::sqrt(a);
        }
        {
          T a = 0;
          res_access[i++] = std::tan(a);
        }
        {
          T a = 0;
          res_access[i++] = std::tanh(a);
        }
        {
          T a = 1;
          res_access[i++] = std::acosh(a);
        }
        {
          T a = 0;
          res_access[i++] = std::asinh(a);
        }
        {
          T a = 0;
          res_access[i++] = std::atanh(a);
        }
        {
          T a = 1;
          res_access[i++] = std::cbrt(a);
        }
        {
          T a = 0;
          res_access[i++] = std::erf(a);
        }
        {
          T a = 0;
          res_access[i++] = std::erfc(a);
        }
        {
          T a = 1;
          res_access[i++] = std::exp2(a);
        }
        {
          T a = 0;
          res_access[i++] = std::expm1(a);
        }
        {
          T a = 1;
          T b = 0;
          res_access[i++] = std::fdim(a, b);
        }
        {
          T a = 1;
          T b = 1;
          T c = 1;
          res_access[i++] = std::fma(a, b, c);
        }
        {
          T a = 3;
          T b = 4;
          res_access[i++] = std::hypot(a, b);
        }
        {
          T a = 1;
          res_access[i++] = std::ilogb(a);
        }
        {
          T a = 0;
          res_access[i++] = std::log1p(a);
        }
        {
          T a = 1;
          res_access[i++] = std::log2(a);
        }
        {
          T a = 1;
          res_access[i++] = std::logb(a);
        }
        {
          T a = 0.5;
          T b = 1;
          res_access[i++] = std::remainder(a, b);
        }
        {
          T a = 0.5;
          T b = 1;
          res_access[i++] = std::remquo(a, b, &quo_access[0]);
        }
        {
          T a = NAN;
          res_access[i++] = std::tgamma(a);
        }
        {
          T a = NAN;
          res_access[i++] = std::lgamma(a);
        }
      });
    });
  }
  // Compare result with reference
  for (int i = 0; i < TEST_NUM; ++i) {
    assert(is_about_FP(result[i], ref[i]));
  }
  // Test modf integral part
  assert(is_about_FP(iptr, refIptr));
  // Test frexp exponent
  assert(exponent == 0);
  // Test remquo sign
  assert(quo == 0);
}

int main() {
  s::queue deviceQueue;
  if (deviceQueue.get_device().has_extension("cl_khr_fp64")) {
    device_cmath_test<double>(deviceQueue);
    std::cout << "Pass" << std::endl;
  }
  return 0;
}
