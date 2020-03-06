// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-cmath-fp64.o -o %t.out
#include <CL/sycl.hpp>
#include <cmath>
#include <iostream>

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

template <class T>
class DeviceCos;

template <class T>
void device_cos_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCos<T> >([=]() {
        T a = 0;
        res_access[0] = std::cos(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceSin;

template <class T>
void device_sin_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSin<T> >([=]() {
        T a = 0;
        res_access[0] = std::sin(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceLog;

template <class T>
void device_log_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog<T> >([=]() {
        T a = 1;
        res_access[0] = std::log(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceAcos;

template <class T>
void device_acos_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAcos<T> >([=]() {
        T a = 1;
        res_access[0] = std::acos(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceAsin;

template <class T>
void device_asin_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAsin<T> >([=]() {
        T a = 0;
        res_access[0] = std::asin(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceAtan;

template <class T>
void device_atan_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAtan<T> >([=]() {
        T a = 0;
        res_access[0] = std::atan(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceAtan2;

template <class T>
void device_atan2_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAtan2<T> >([=]() {
        T a = 0;
        T b = 1;
        res_access[0] = std::atan2(a, b);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceCosh;

template <class T>
void device_cosh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCosh<T> >([=]() {
        T a = 0;
        res_access[0] = std::cosh(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceExp;

template <class T>
void device_exp_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceExp<T> >([=]() {
        T a = 0;
        res_access[0] = std::exp(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceFmod;

template <class T>
void device_fmod_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0.5;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFmod<T> >([=]() {
        T a = 1.5;
        T b = 1;
        res_access[0] = std::fmod(a, b);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceFrexp;

template <class T>
void device_frexp_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  int exp = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    s::buffer<int, 1> buffer2(&exp, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto exp_access = buffer2.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFrexp<T> >([=]() {
        T a = 0;
        res_access[0] = std::frexp(a, &exp_access[0]);
      });
    });
  }

  assert(result == ref && exp == 0);
}

template <class T>
class DeviceLdexp;

template <class T>
void device_ldexp_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 2;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLdexp<T> >([=]() {
        T a = 1;
        res_access[0] = std::ldexp(a, 1);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceLog10;

template <class T>
void device_log10_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog10<T> >([=]() {
        T a = 1;
        res_access[0] = std::log10(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceModf;

template <class T>
void device_modf_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T iptr = -1;
  T ref1 = 0;
  T ref2 = 1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    s::buffer<T, 1> buffer2(&iptr, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto iptr_access = buffer2.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceModf<T> >([=]() {
        T a = 1;
        res_access[0] = std::modf(a, &iptr_access[0]);
      });
    });
  }

  assert(result == ref1 && iptr == ref2);
}

template <class T>
class DevicePow;

template <class T>
void device_pow_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DevicePow<T> >([=]() {
        T a = 1;
        T b = 1;
        res_access[0] = std::pow(a, b);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceSinh;

template <class T>
void device_sinh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSinh<T> >([=]() {
        T a = 0;
        res_access[0] = std::sinh(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceSqrt;

template <class T>
void device_sqrt_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 2;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceSqrt<T> >([=]() {
        T a = 4;
        res_access[0] = std::sqrt(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceTan;

template <class T>
void device_tan_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceTan<T> >([=]() {
        T a = 0;
        res_access[0] = std::tan(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceTanh;

template <class T>
void device_tanh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceTanh<T> >([=]() {
        T a = 0;
        res_access[0] = std::tanh(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceAcosh;

template <class T>
void device_acosh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAcosh<T> >([=]() {
        T a = 1;
        res_access[0] = std::acosh(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceAsinh;

template <class T>
void device_asinh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAsinh<T> >([=]() {
        T a = 0;
        res_access[0] = std::asinh(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceAtanh;

template <class T>
void device_atanh_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceAtanh<T> >([=]() {
        T a = 0;
        res_access[0] = std::atanh(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceCbrt;

template <class T>
void device_cbrt_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceCbrt<T> >([=]() {
        T a = 1;
        res_access[0] = std::cbrt(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceErf;

template <class T>
void device_erf_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceErf<T> >([=]() {
        T a = 0;
        res_access[0] = std::erf(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceErfc;

template <class T>
void device_erfc_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceErfc<T> >([=]() {
        T a = 0;
        res_access[0] = std::erfc(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceExp2;

template <class T>
void device_exp2_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 2;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceExp2<T> >([=]() {
        T a = 1;
        res_access[0] = std::exp2(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceExpm1;

template <class T>
void device_expm1_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceExpm1<T> >([=]() {
        T a = 0;
        res_access[0] = std::expm1(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceFdim;

template <class T>
void device_fdim_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFdim<T> >([=]() {
        T a = 1;
        T b = 0;
        res_access[0] = std::fdim(a, b);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceFma;

template <class T>
void device_fma_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 2;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceFma<T> >([=]() {
        T a = 1;
        T b = 1;
        T c = 1;
        res_access[0] = std::fma(a, b, c);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceHypot;

template <class T>
void device_hypot_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 5;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceHypot<T> >([=]() {
        T a = 3;
        T b = 4;
        res_access[0] = std::hypot(a, b);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceIlogb;

template <class T>
void device_ilogb_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceIlogb<T> >([=]() {
        T a = 1;
        res_access[0] = std::ilogb(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceLgamma;

template <class T>
void device_lgamma_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLgamma<T> >([=]() {
        T a = NAN;
        res_access[0] = std::lgamma(a);
      });
    });
  }

  assert(std::isnan(result));
}

template <class T>
class DeviceLog1p;

template <class T>
void device_log1p_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog1p<T> >([=]() {
        T a = 0;
        res_access[0] = std::log1p(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceLog2;

template <class T>
void device_log2_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLog2<T> >([=]() {
        T a = 1;
        res_access[0] = std::log2(a);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceLogb;

template <class T>
void device_logb_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceLogb<T> >([=]() {
        T a = 1;
        res_access[0] = std::logb(1);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceRemainder;

template <class T>
void device_remainder_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  T ref = 0.5;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceRemainder<T> >([=]() {
        T a = 0.5;
        T b = 1;
        res_access[0] = std::remainder(a, b);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceRemquo;

template <class T>
void device_remquo_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  int quo = -1;
  T ref = 0.5;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    s::buffer<int, 1> buffer2(&quo, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      auto quo_access = buffer2.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceRemquo<T> >([=]() {
        T a = 0.5;
        T b = 1;
        res_access[0] = std::remquo(a, b, &quo_access[0]);
      });
    });
  }

  assert(result == ref);
}

template <class T>
class DeviceTgamma;

template <class T>
void device_tgamma_test(s::queue &deviceQueue) {
  s::range<1> numOfItems{1};
  T result = -1;
  {
    s::buffer<T, 1> buffer1(&result, numOfItems);
    deviceQueue.submit([&](cl::sycl::handler &cgh) {
      auto res_access = buffer1.template get_access<sycl_write>(cgh);
      cgh.single_task<class DeviceTgamma<T> >([=]() {
        T a = NAN;
        res_access[0] = std::tgamma(a);
      });
    });
  }

  assert(std::isnan(result));
}

template <class T>
void device_cmath_test(s::queue &deviceQueue) {
  device_cos_test<T>(deviceQueue);
  device_sin_test<T>(deviceQueue);
  device_log_test<T>(deviceQueue);
  device_acos_test<T>(deviceQueue);
  device_asin_test<T>(deviceQueue);
  device_atan_test<T>(deviceQueue);
  device_atan2_test<T>(deviceQueue);
  device_cosh_test<T>(deviceQueue);
  device_exp_test<T>(deviceQueue);
  device_fmod_test<T>(deviceQueue);
  device_frexp_test<T>(deviceQueue);
  device_ldexp_test<T>(deviceQueue);
  device_log10_test<T>(deviceQueue);
  device_modf_test<T>(deviceQueue);
  device_pow_test<T>(deviceQueue);
  device_sinh_test<T>(deviceQueue);
  device_sqrt_test<T>(deviceQueue);
  device_tan_test<T>(deviceQueue);
  device_tanh_test<T>(deviceQueue);
  device_acosh_test<T>(deviceQueue);
  device_asinh_test<T>(deviceQueue);
  device_atanh_test<T>(deviceQueue);
  device_cbrt_test<T>(deviceQueue);
  device_erf_test<T>(deviceQueue);
  device_erfc_test<T>(deviceQueue);
  device_exp2_test<T>(deviceQueue);
  device_expm1_test<T>(deviceQueue);
  device_fdim_test<T>(deviceQueue);
  device_fma_test<T>(deviceQueue);
  device_hypot_test<T>(deviceQueue);
  device_ilogb_test<T>(deviceQueue);
  device_lgamma_test<T>(deviceQueue);
  device_log1p_test<T>(deviceQueue);
  device_log2_test<T>(deviceQueue);
  device_logb_test<T>(deviceQueue);
  device_remainder_test<T>(deviceQueue);
  device_remquo_test<T>(deviceQueue);
  device_tgamma_test<T>(deviceQueue);
}

int main() {
  s::queue deviceQueue;
  if (deviceQueue.get_device().has_extension("cl_khr_fp64")) {
    device_cmath_test<double>(deviceQueue);
    std::cout << "Pass" << std::endl;
  }
  return 0;
}
