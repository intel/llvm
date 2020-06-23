// REQUIRES: cpu, linux
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath.o -o %t.out
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

float ref_val[TEST_NUM] = {
    1, 0, 0, 0, 0, 0, 0, 1, 1, 0.5,
    0, 2, 0, 0, 1, 0, 2, 0, 0, 0,
    0, 0, 1, 0, 1, 2, 0, 1, 2, 5,
    0, 0, 0, 0, 0.5, 0.5, NAN, NAN};

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
        res_access[i++] = cosf(0.0f);
        res_access[i++] = sinf(0.0f);
        res_access[i++] = logf(1.0f);
        res_access[i++] = acosf(1.0f);
        res_access[i++] = asinf(0.0f);
        res_access[i++] = atanf(0.0f);
        res_access[i++] = atan2f(0.0f, 1.0f);
        res_access[i++] = coshf(0.0f);
        res_access[i++] = expf(0.0f);
        res_access[i++] = fmodf(1.5f, 1.0f);
        res_access[i++] = frexpf(0.0f, &exp_access[0]);
        res_access[i++] = ldexpf(1.0f, 1);
        res_access[i++] = log10f(1.0f);
        res_access[i++] = modff(1.0f, &iptr_access[0]);
        res_access[i++] = powf(1.0f, 1.0f);
        res_access[i++] = sinhf(0.0f);
        res_access[i++] = sqrtf(4.0f);
        res_access[i++] = tanf(0.0f);
        res_access[i++] = tanhf(0.0f);
        res_access[i++] = acoshf(1.0f);
        res_access[i++] = asinhf(0.0f);
        res_access[i++] = atanhf(0.0f);
        res_access[i++] = cbrtf(1.0f);
        res_access[i++] = erff(0.0f);
        res_access[i++] = erfcf(0.0f);
        res_access[i++] = exp2f(1.0f);
        res_access[i++] = expm1f(0.0f);
        res_access[i++] = fdimf(1.0f, 0.0f);
        res_access[i++] = fmaf(1.0f, 1.0f, 1.0f);
        res_access[i++] = hypotf(3.0f, 4.0f);
        res_access[i++] = ilogbf(1.0f);
        res_access[i++] = log1pf(0.0f);
        res_access[i++] = log2f(1.0f);
        res_access[i++] = logbf(1.0f);
        res_access[i++] = remainderf(0.5f, 1.0f);
        res_access[i++] = remquof(0.5f, 1.0f, &quo_access[0]);
        float a = NAN;
        res_access[i++] = tgammaf(a);
        res_access[i++] = lgammaf(a);
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
  device_math_test(deviceQueue);
  std::cout << "Pass" << std::endl;
  return 0;
}
