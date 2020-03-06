// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath.o -o %t.out
#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

class DeviceCos;

void device_cos_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 1;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCos>([=]() {
        float a = 0;
        res_access[0] = cosf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceSin;

void device_sin_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSin>([=]() {
        float a = 0;
        res_access[0] = sinf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceLog;

void device_log_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog>([=]() {
        float a = 1;
        res_access[0] = logf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAcos;

void device_acos_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAcos>([=]() {
        float a = 1;
        res_access[0] = acosf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAsin;

void device_asin_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAsin>([=]() {
        float a = 0;
        res_access[0] = asinf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAtan;

void device_atan_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAtan>([=]() {
        float a = 0;
        res_access[0] = atanf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAtan2;

void device_atan2_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAtan2>([=]() {
        float a = 0;
        float b = 1;
        res_access[0] = atan2f(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceCosh;

void device_cosh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 1;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCosh>([=]() {
        float a = 0;
        res_access[0] = coshf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceExp;

void device_exp_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 1;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceExp>([=]() {
        float a = 0;
        res_access[0] = expf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceFmod;

void device_fmod_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0.5;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFmod>([=]() {
        float a = 1.5;
        float b = 1;
        res_access[0] = fmodf(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceFrexp;

void device_frexp_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  int exp = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    s::buffer<int, 1> buffer2(&exp, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto exp_access = buffer2.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFrexp>([=]() {
        float a = 0;
        res_access[0] = frexpf(a, &exp_access[0]);
      });
    });
  }

  assert(result == ref && exp == 0);
}

class DeviceLdexp;

void device_ldexp_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 2;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLdexp>([=]() {
        float a = 1;
        res_access[0] = ldexpf(a, 1);
      });
    });
  }

  assert(result == ref);
}

class DeviceLog10;

void device_log10_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog10>([=]() {
        float a = 1;
        res_access[0] = log10f(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceModf;

void device_modf_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float iptr = -1;
  float ref1 = 0;
  float ref2 = 1;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    s::buffer<float, 1> buffer2(&iptr, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer2.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceModf>([=]() {
        float a = 1;
        res_access[0] = modff(a, &iptr_access[0]);
      });
    });
  }

  assert(result == ref1 && iptr == ref2);
}

class DevicePow;

void device_pow_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 1;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DevicePow>([=]() {
        float a = 1;
        float b = 1;
        res_access[0] = powf(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceSinh;

void device_sinh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSinh>([=]() {
        float a = 0;
        res_access[0] = sinhf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceSqrt;

void device_sqrt_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 2;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSqrt>([=]() {
        float a = 4;
        res_access[0] = sqrtf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceTan;

void device_tan_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceTan>([=]() {
        float a = 0;
        res_access[0] = tanf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceTanh;

void device_tanh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceTanh>([=]() {
        float a = 0;
        res_access[0] = tanhf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAcosh;

void device_acosh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAcosh>([=]() {
        float a = 1;
        res_access[0] = acoshf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAsinh;

void device_asinh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAsinh>([=]() {
        float a = 0;
        res_access[0] = asinhf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAtanh;

void device_atanh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAtanh>([=]() {
        float a = 0;
        res_access[0] = atanhf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceCbrt;

void device_cbrt_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 1;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCbrt>([=]() {
        float a = 1;
        res_access[0] = cbrtf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceErf;

void device_erf_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceErf>([=]() {
        float a = 0;
        res_access[0] = erff(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceErfc;

void device_erfc_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 1;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceErfc>([=]() {
        float a = 0;
        res_access[0] = erfcf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceExp2;

void device_exp2_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 2;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceExp2>([=]() {
        float a = 1;
        res_access[0] = exp2f(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceExpm1;

void device_expm1_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceExpm1>([=]() {
        float a = 0;
        res_access[0] = expm1f(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceFdim;

void device_fdim_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 1;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFdim>([=]() {
        float a = 0;
        res_access[0] = fdimf(1, a);
      });
    });
  }

  assert(result == ref);
}

class DeviceFma;

void device_fma_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 2;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFma>([=]() {
        float a = 1;
        float b = 1;
        float c = 1;
        res_access[0] = fmaf(a, b, c);
      });
    });
  }

  assert(result == ref);
}

class DeviceHypot;

void device_hypot_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 5;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceHypot>([=]() {
        float a = 3;
        float b = 4;
        res_access[0] = hypotf(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceIlogb;

void device_ilogb_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceIlogb>([=]() {
        float a = 1;
        res_access[0] = ilogbf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceLgamma;

void device_lgamma_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLgamma>([=]() {
        float a = NAN;
        res_access[0] = lgammaf(a);
      });
    });
  }

  assert(std::isnan(result));
}

class DeviceLog1p;

void device_log1p_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog1p>([=]() {
        float a = 0;
        res_access[0] = log1pf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceLog2;

void device_log2_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog2>([=]() {
        float a = 1;
        res_access[0] = log2f(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceLogb;

void device_logb_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLogb>([=]() {
        float a = 1;
        res_access[0] = logbf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceRemainder;

void device_remainder_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  float ref = 0.5;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceRemainder>([=]() {
        float a = 0.5;
        float b = 1;
        res_access[0] = remainderf(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceRemquo;

void device_remquo_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  int quo = -1;
  float ref = 0.5;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    s::buffer<int, 1> buffer2(&quo, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto quo_access = buffer2.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceRemquo>([=]() {
        float a = 0.5;
        float b = 1;
        res_access[0] = remquof(a, b, &quo_access[0]);
      });
    });
  }

  assert(result == ref);
}

class DeviceTgamma;

void device_tgamma_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  float result = -1;
  {
    s::buffer<float, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceTgamma>([=]() {
        float a = NAN;
        res_access[0] = tgammaf(a);
      });
    });
  }

  assert(std::isnan(result));
}

void device_math_test(s::queue &deviceQueue) {
  device_cos_test(deviceQueue);
  device_sin_test(deviceQueue);
  device_log_test(deviceQueue);
  device_acos_test(deviceQueue);
  device_asin_test(deviceQueue);
  device_atan_test(deviceQueue);
  device_atan2_test(deviceQueue);
  device_cosh_test(deviceQueue);
  device_exp_test(deviceQueue);
  device_fmod_test(deviceQueue);
  device_frexp_test(deviceQueue);
  device_ldexp_test(deviceQueue);
  device_log10_test(deviceQueue);
  device_modf_test(deviceQueue);
  device_pow_test(deviceQueue);
  device_sinh_test(deviceQueue);
  device_sqrt_test(deviceQueue);
  device_tan_test(deviceQueue);
  device_tanh_test(deviceQueue);
  device_acosh_test(deviceQueue);
  device_asinh_test(deviceQueue);
  device_atanh_test(deviceQueue);
  device_cbrt_test(deviceQueue);
  device_erf_test(deviceQueue);
  device_erfc_test(deviceQueue);
  device_exp2_test(deviceQueue);
  device_expm1_test(deviceQueue);
  device_fdim_test(deviceQueue);
  device_fma_test(deviceQueue);
  device_hypot_test(deviceQueue);
  device_ilogb_test(deviceQueue);
  device_lgamma_test(deviceQueue);
  device_log1p_test(deviceQueue);
  device_log2_test(deviceQueue);
  device_logb_test(deviceQueue);
  device_remainder_test(deviceQueue);
  device_remquo_test(deviceQueue);
  device_tgamma_test(deviceQueue);
}

int main() {
  s::queue deviceQueue;
  device_math_test(deviceQueue);
  std::cout << "Pass" << std::endl;
  return 0;
}
