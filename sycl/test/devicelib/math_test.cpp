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

float a = 0;
float b = 1;
float c = 0.5;
float d = 2;
float e = 5;

void device_math_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{38};
  float result[38] = {-1};
  float ref[38] = {
      b,
      a,
      a,
      a,
      a,
      a,
      a,
      b,
      b,
      c,
      a,
      d,
      a,
      a,
      b,
      a,
      d,
      a,
      a,
      a,
      a,
      a,
      b,
      a,
      b,
      d,
      a,
      b,
      d,
      e,
      a,
      a,
      a,
      a,
      c,
      c,
      a,
      a,
  };
  int expv = -1;
  float iptr = -1;
  int quo = -1;
  {
    s::buffer<float, 1> buffer1(result, numOfItems);
    s::buffer<int, 1> buffer2(&expv, s::range<1>{1});
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
          float a = 0;
          res_access[i++] = fdimf(1, a);
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
  for (int i = 0; i < 36; ++i) {
    assert(is_about_FP(result[i], ref[i]));
  }
  assert(std::isnan(result[36]));
  assert(std::isnan(result[37]));
  assert(is_about_FP(iptr, b));
  assert(expv == 0);
  assert(quo == 0);
}

int main() {
  s::queue deviceQueue;
  device_math_test(deviceQueue);
  std::cout << "Pass" << std::endl;
  return 0;
}
