// REQUIRES: cpu, linux
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath-fp64.o -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
#include "math_utils.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

#define TEST_NUM 38

double ref_val[TEST_NUM] = {
    1, 0, 0, 0, 0, 0, 0, 1, 1, 0.5,
    0, 2, 0, 0, 1, 0, 2, 0, 0, 0,
    0, 0, 1, 0, 1, 2, 0, 1, 2, 5,
    0, 0, 0, 0, 0.5, 0.5, NAN, NAN};

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
        res_access[i++] = cos(0.0);
        res_access[i++] = sin(0.0);
        res_access[i++] = log(1.0);
        res_access[i++] = acos(1.0);
        res_access[i++] = asin(0.0);
        res_access[i++] = atan(0.0);
        res_access[i++] = atan2(0.0, 1.0);
        res_access[i++] = cosh(0.0);
        res_access[i++] = exp(0.0);
        res_access[i++] = fmod(1.5, 1.0);
        res_access[i++] = frexp(0.0, &exp_access[0]);
        res_access[i++] = ldexp(1.0, 1);
        res_access[i++] = log10(1.0);
        res_access[i++] = modf(1.0, &iptr_access[0]);
        res_access[i++] = pow(1.0, 1.0);
        res_access[i++] = sinh(0.0);
        res_access[i++] = sqrt(4.0);
        res_access[i++] = tan(0.0);
        res_access[i++] = tanh(0.0);
        res_access[i++] = acosh(1.0);
        res_access[i++] = asinh(0.0);
        res_access[i++] = atanh(0.0);
        res_access[i++] = cbrt(1.0);
        res_access[i++] = erf(0.0);
        res_access[i++] = erfc(0.0);
        res_access[i++] = exp2(1.0);
        res_access[i++] = expm1(0.0);
        res_access[i++] = fdim(1.0, 0.0);
        res_access[i++] = fma(1.0, 1.0, 1.0);
        res_access[i++] = hypot(3.0, 4.0);
        res_access[i++] = ilogb(1.0);
        res_access[i++] = log1p(0.0);
        res_access[i++] = log2(1.0);
        res_access[i++] = logb(1.0);
        res_access[i++] = remainder(0.5, 1.0);
        res_access[i++] = remquo(0.5, 1.0, &quo_access[0]);
        double a = NAN;
        res_access[i++] = tgamma(a);
        res_access[i++] = lgamma(a);
      });
    });
  }

  // Compare result with reference
  for (int i = 0; i < TEST_NUM; ++i) {
    assert(approx_equal_fp(result[i], ref_val[i]));
  }

  // Test modf integral part
  assert(approx_equal_fp(iptr, refIptr));

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
