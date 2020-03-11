// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath-fp64.o -o %t.out
#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>
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

void device_math_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{TEST_NUM};
  double result[TEST_NUM] = {-1};
  // Variable exponent is an integer value to store the exponent in frexp function
  int exponent = -1;
  // Variable iptr stores the integral part of float point in modf function
  double iptr = -1;
  // Variable quo stores the sign and some bits of x/y in remquo function
  int quo = -1;
  {
    s::buffer<double, 1> buffer1(result, numOfItems);
    s::buffer<int, 1> buffer2(&exponent, s::range<1>{1});
    s::buffer<double, 1> buffer3(&iptr, s::range<1>{1});
    s::buffer<int, 1> buffer4(&quo, s::range<1>{1});
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto exp_access = buffer2.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer3.template get_access<sycl_write>(cgh);
      auto quo_access = buffer4.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceMathTest>([=]() {
        int i = 0;
        {
          double a = 0;
          res_access[i++] = cos(a);
        }
        {
          double a = 0;
          res_access[i++] = sin(a);
        }
        {
          double a = 1;
          res_access[i++] = log(a);
        }
        {
          double a = 1;
          res_access[i++] = acos(a);
        }
        {
          double a = 0;
          res_access[i++] = asin(a);
        }
        {
          double a = 0;
          res_access[i++] = atan(a);
        }
        {
          double a = 0;
          double b = 1;
          res_access[i++] = atan2(a, b);
        }
        {
          double a = 0;
          res_access[i++] = cosh(a);
        }
        {
          double a = 0;
          res_access[i++] = exp(a);
        }
        {
          double a = 1.5;
          double b = 1;
          res_access[i++] = fmod(a, b);
        }
        {
          double a = 0;
          res_access[i++] = frexp(a, &exp_access[0]);
        }
        {
          double a = 1;
          res_access[i++] = ldexp(a, 1);
        }
        {
          double a = 1;
          res_access[i++] = log10(a);
        }
        {
          double a = 1;
          res_access[i++] = modf(a, &iptr_access[0]);
        }
        {
          double a = 1;
          double b = 1;
          res_access[i++] = pow(a, b);
        }
        {
          double a = 0;
          res_access[i++] = sinh(a);
        }
        {
          double a = 4;
          res_access[i++] = sqrt(a);
        }
        {
          double a = 0;
          res_access[i++] = tan(a);
        }
        {
          double a = 0;
          res_access[i++] = tanh(a);
        }
        {
          double a = 1;
          res_access[i++] = acosh(a);
        }
        {
          double a = 0;
          res_access[i++] = asinh(a);
        }
        {
          double a = 0;
          res_access[i++] = atanh(a);
        }
        {
          double a = 1;
          res_access[i++] = cbrt(a);
        }
        {
          double a = 0;
          res_access[i++] = erf(a);
        }
        {
          double a = 0;
          res_access[i++] = erfc(a);
        }
        {
          double a = 1;
          res_access[i++] = exp2(a);
        }
        {
          double a = 0;
          res_access[i++] = expm1(a);
        }
        {
          double a = 1;
          double b = 0;
          res_access[i++] = fdim(a, b);
        }
        {
          double a = 1;
          double b = 1;
          double c = 1;
          res_access[i++] = fma(a, b, c);
        }
        {
          double a = 3;
          double b = 4;
          res_access[i++] = hypot(a, b);
        }
        {
          double a = 1;
          res_access[i++] = ilogb(a);
        }
        {
          double a = 0;
          res_access[i++] = log1p(a);
        }
        {
          double a = 1;
          res_access[i++] = log2(a);
        }
        {
          double a = 1;
          res_access[i++] = logb(a);
        }
        {
          double a = 0.5;
          double b = 1;
          res_access[i++] = remainder(a, b);
        }
        {
          double a = 0.5;
          double b = 1;
          res_access[i++] = remquo(a, b, &quo_access[0]);
        }
        {
          double a = NAN;
          res_access[i++] = tgamma(a);
        }
        {
          double a = NAN;
          res_access[i++] = lgamma(a);
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
    device_math_test(deviceQueue);
    std::cout << "Pass" << std::endl;
  }
  return 0;
}
