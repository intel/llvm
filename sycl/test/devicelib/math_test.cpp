// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath.o -o %t.out
#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>
#include "math_utils.hpp"

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

#define TEST_NUM 38

float ref[TEST_NUM] = {
1, 0, 0, 0, 0, 0, 0, 1, 1, 0.5,
0, 2, 0, 0, 1, 0, 2, 0, 0, 0,
0, 0, 1, 0, 1, 2, 0, 1, 2, 5,
0, 0, 0, 0, 0.5, 0.5, NAN, NAN,};

float refIptr = 1;

void device_math_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{TEST_NUM};
  float result[TEST_NUM] = {-1};
  // Variable exponent is an integer value to store the exponent in frexp function
  int exponent = -1;
  // Variable iptr stores the integral part of float point in modf function
  float iptr = -1;
  // Variable quo stores the sign and some bits of x/y in remquo function
  int quo = -1;
  {
    s::buffer<float, 1> buffer1(result, numOfItems);
    s::buffer<int, 1> buffer2(&exponent, s::range<1>{1});
    s::buffer<float, 1> buffer3(&iptr, s::range<1>{1});
    s::buffer<int, 1> buffer4(&quo, s::range<1>{1});
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto exp_access = buffer2.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer3.template get_access<sycl_write>(cgh);
      auto quo_access = buffer4.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceMathTest>([=]() {
        int i = 0;
        {
          float a = 0;
          res_access[i++] = cosf(a);
        }
        {
          float a = 0;
          res_access[i++] = sinf(a);
        }
        {
          float a = 1;
          res_access[i++] = logf(a);
        }
        {
          float a = 1;
          res_access[i++] = acosf(a);
        }
        {
          float a = 0;
          res_access[i++] = asinf(a);
        }
        {
          float a = 0;
          res_access[i++] = atanf(a);
        }
        {
          float a = 0;
          float b = 1;
          res_access[i++] = atan2f(a, b);
        }
        {
          float a = 0;
          res_access[i++] = coshf(a);
        }
        {
          float a = 0;
          res_access[i++] = expf(a);
        }
        {
          float a = 1.5;
          float b = 1;
          res_access[i++] = fmodf(a, b);
        }
        {
          float a = 0;
          res_access[i++] = frexpf(a, &exp_access[0]);
        }
        {
          float a = 1;
          res_access[i++] = ldexpf(a, 1);
        }
        {
          float a = 1;
          res_access[i++] = log10f(a);
        }
        {
          float a = 1;
          res_access[i++] = modff(a, &iptr_access[0]);
        }
        {
          float a = 1;
          float b = 1;
          res_access[i++] = powf(a, b);
        }
        {
          float a = 0;
          res_access[i++] = sinhf(a);
        }
        {
          float a = 4;
          res_access[i++] = sqrtf(a);
        }
        {
          float a = 0;
          res_access[i++] = tanf(a);
        }
        {
          float a = 0;
          res_access[i++] = tanhf(a);
        }
        {
          float a = 1;
          res_access[i++] = acoshf(a);
        }
        {
          float a = 0;
          res_access[i++] = asinhf(a);
        }
        {
          float a = 0;
          res_access[i++] = atanhf(a);
        }
        {
          float a = 1;
          res_access[i++] = cbrtf(a);
        }
        {
          float a = 0;
          res_access[i++] = erff(a);
        }
        {
          float a = 0;
          res_access[i++] = erfcf(a);
        }
        {
          float a = 1;
          res_access[i++] = exp2f(a);
        }
        {
          float a = 0;
          res_access[i++] = expm1f(a);
        }
        {
          float a = 1;
          float b = 0;
          res_access[i++] = fdimf(a, b);
        }
        {
          float a = 1;
          float b = 1;
          float c = 1;
          res_access[i++] = fmaf(a, b, c);
        }
        {
          float a = 3;
          float b = 4;
          res_access[i++] = hypotf(a, b);
        }
        {
          float a = 1;
          res_access[i++] = ilogbf(a);
        }
        {
          float a = 0;
          res_access[i++] = log1pf(a);
        }
        {
          float a = 1;
          res_access[i++] = log2f(a);
        }
        {
          float a = 1;
          res_access[i++] = logbf(a);
        }
        {
          float a = 0.5;
          float b = 1;
          res_access[i++] = remainderf(a, b);
        }
        {
          float a = 0.5;
          float b = 1;
          res_access[i++] = remquof(a, b, &quo_access[0]);
        }
        {
          float a = NAN;
          res_access[i++] = tgammaf(a);
        }
        {
          float a = NAN;
          res_access[i++] = lgammaf(a);
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
  device_math_test(deviceQueue);
  std::cout << "Pass" << std::endl;
  return 0;
}
