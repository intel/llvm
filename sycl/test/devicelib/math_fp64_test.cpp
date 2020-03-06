// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath-fp64.o -o %t.out
#include <CL/sycl.hpp>
#include <iostream>
#include <math.h>

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

class DeviceCos;

void device_cos_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 1;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCos>([=]() {
        double a = 0;
        res_access[0] = cos(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceSin;

void device_sin_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSin>([=]() {
        double a = 0;
        res_access[0] = sin(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceLog;

void device_log_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog>([=]() {
        double a = 1;
        res_access[0] = log(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAcos;

void device_acos_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAcos>([=]() {
        double a = 1;
        res_access[0] = acos(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAsin;

void device_asin_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAsin>([=]() {
        double a = 0;
        res_access[0] = asin(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAtan;

void device_atan_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAtan>([=]() {
        double a = 0;
        res_access[0] = atan(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAtan2;

void device_atan2_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAtan2>([=]() {
        double a = 0;
        double b = 1;
        res_access[0] = atan2(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceCosh;

void device_cosh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 1;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCosh>([=]() {
        double a = 0;
        res_access[0] = cosh(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceExp;

void device_exp_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 1;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceExp>([=]() {
        double a = 0;
        res_access[0] = exp(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceFmod;

void device_fmod_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0.5;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFmod>([=]() {
        double a = 1.5;
        double b = 1;
        res_access[0] = fmod(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceFrexp;

void device_frexp_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  int exp = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    s::buffer<int, 1> buffer2(&exp, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto exp_access = buffer2.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFrexp>([=]() {
        double a = 0;
        res_access[0] = frexp(a, &exp_access[0]);
      });
    });
  }

  assert(result == ref && exp == 0);
}

class DeviceLdexp;

void device_ldexp_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 2;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLdexp>([=]() {
        double a = 1;
        res_access[0] = ldexp(a, 1);
      });
    });
  }

  assert(result == ref);
}

class DeviceLog10;

void device_log10_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog10>([=]() {
        double a = 1;
        res_access[0] = log10(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceModf;

void device_modf_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double iptr = -1;
  double ref1 = 0;
  double ref2 = 1;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    s::buffer<double, 1> buffer2(&iptr, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer2.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceModf>([=]() {
        double a = 1;
        res_access[0] = modf(a, &iptr_access[0]);
      });
    });
  }

  assert(result == ref1 && iptr == ref2);
}

class DevicePow;

void device_pow_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 1;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DevicePow>([=]() {
        double a = 1;
        double b = 1;
        res_access[0] = pow(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceSinh;

void device_sinh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSinh>([=]() {
        double a = 0;
        res_access[0] = sinh(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceSqrt;

void device_sqrt_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 2;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSqrt>([=]() {
        double a = 4;
        res_access[0] = sqrt(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceTan;

void device_tan_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceTan>([=]() {
        double a = 0;
        res_access[0] = tan(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceTanh;

void device_tanh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceTanh>([=]() {
        double a = 0;
        res_access[0] = tanh(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAcosh;

void device_acosh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAcosh>([=]() {
        double a = 1;
        res_access[0] = acosh(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAsinh;

void device_asinh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAsinh>([=]() {
        double a = 0;
        res_access[0] = asinh(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceAtanh;

void device_atanh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAtanh>([=]() {
        double a = 0;
        res_access[0] = atanh(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceCbrt;

void device_cbrt_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 1;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCbrt>([=]() {
        double a = 1;
        res_access[0] = cbrt(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceErf;

void device_erf_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceErf>([=]() {
        double a = 0;
        res_access[0] = erf(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceErfc;

void device_erfc_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 1;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceErfc>([=]() {
        double a = 0;
        res_access[0] = erfc(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceExp2;

void device_exp2_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 2;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceExp2>([=]() {
        double a = 1;
        res_access[0] = exp2(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceExpm1;

void device_expm1_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceExpm1>([=]() {
        double a = 0;
        res_access[0] = expm1(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceFdim;

void device_fdim_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 1;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFdim>([=]() {
        double a = 1;
        double b = 0;
        res_access[0] = fdim(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceFma;

void device_fma_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 2;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFma>([=]() {
        double a = 1;
        double b = 1;
        double c = 1;
        res_access[0] = fma(a, b, c);
      });
    });
  }

  assert(result == ref);
}

class DeviceHypot;

void device_hypot_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 5;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceHypot>([=]() {
        double a = 3;
        double b = 4;
        res_access[0] = hypot(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceIlogb;

void device_ilogb_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceIlogb>([=]() {
        double a = 1;
        res_access[0] = ilogb(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceLgamma;

void device_lgamma_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLgamma>([=]() {
        double a = NAN;
        res_access[0] = lgamma(a);
      });
    });
  }

  assert(std::isnan(result));
}

class DeviceLog1p;

void device_log1p_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog1p>([=]() {
        double a = 0;
        res_access[0] = log1p(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceLog2;

void device_log2_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog2>([=]() {
        double a = 1;
        res_access[0] = log2(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceLogb;

void device_logb_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLogb>([=]() {
        double a = 1;
        res_access[0] = logb(a);
      });
    });
  }

  assert(result == ref);
}

class DeviceRemainder;

void device_remainder_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  double ref = 0.5;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceRemainder>([=]() {
        double a = 0.5;
        double b = 1;
        res_access[0] = remainder(a, b);
      });
    });
  }

  assert(result == ref);
}

class DeviceRemquo;

void device_remquo_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  int quo = -1;
  double ref = 0.5;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    s::buffer<int, 1> buffer2(&quo, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto quo_access = buffer2.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceRemquo>([=]() {
        double a = 0.5;
        double b = 1;
        res_access[0] = remquo(a, b, &quo_access[0]);
      });
    });
  }

  assert(result == ref);
}

class DeviceTgamma;
void device_tgamma_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  double result = -1;
  {
    s::buffer<double, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceTgamma>([=]() {
        double a = NAN;
        res_access[0] = tgamma(a);
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
  if (deviceQueue.get_device().has_extension("cl_khr_fp64")) {
    device_math_test(deviceQueue);
    std::cout << "Pass" << std::endl;
  }
  return 0;
}
