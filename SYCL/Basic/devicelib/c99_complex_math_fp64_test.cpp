// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-complex-fp64.o -o %t.out
#include <CL/sycl.hpp>
#include <cassert>
#include <complex.h>
#include "math_utils.hpp"
#ifndef CMPLX
#define CMPLX(r, i) ((double __complex__){ (double)r, (double)i })
#endif

bool approx_equal_c99_cmplx(double __complex__ x, double __complex__ y) {
  return approx_equal_fp(creal(x), creal(y)) && approx_equal_fp(cimag(x), cimag(y));
}

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

class DeviceComplexTimes;

void device_c99_complex_times(s::queue &deviceQueue) {
  double __complex__ buf_in3[4] = {CMPLX(0, 1), CMPLX(1, 1),
                                   CMPLX(2, 3), CMPLX(4, 5)};
  double __complex__ buf_in4[4] = {CMPLX(1, 1), CMPLX(2, 1),
                                   CMPLX(2, 2), CMPLX(3, 4)};
  double __complex__ buf_out2[4];

  double __complex__ ref_results2[4] = {CMPLX(-1, 1),  CMPLX(1, 3),
                                        CMPLX(-2, 10), CMPLX(-8, 31)};
  s::range<1> numOfItems{4};
  {
  s::buffer<double __complex__, 1> buffer4(buf_in3, numOfItems);
  s::buffer<double __complex__, 1> buffer5(buf_in4, numOfItems);
  s::buffer<double __complex__, 1> buffer6(buf_out2, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in3_access = buffer4.get_access<sycl_read>(cgh);
    auto buf_in4_access = buffer5.get_access<sycl_read>(cgh);
    auto buf_out2_access = buffer6.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexTimes>(numOfItems, [=](s::id<1>WIid) {
      buf_out2_access[WIid] = buf_in3_access[WIid] * buf_in4_access[WIid];
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_c99_cmplx(buf_out2[idx], ref_results2[idx]));
  }
}

class DeviceComplexDivides;

void device_c99_complex_divides(s::queue &deviceQueue) {
  double __complex__ buf_in3[8] = {CMPLX(-1, 1),  CMPLX(1, 3),
                                   CMPLX(-2, 10), CMPLX(-8, 31),
                                   CMPLX(4, 2), CMPLX(-1, 0),
                                   CMPLX(0, 10), CMPLX(0 , 0)};
  double __complex__ buf_in4[8] = {CMPLX(0, 1), CMPLX(1, 1),
                                   CMPLX(2, 3), CMPLX(4, 5),
                                   CMPLX(2, 0), CMPLX(0, 1),
                                   CMPLX(0, 5), CMPLX(1, 0)};
  double __complex__ ref_results2[8] = {CMPLX(1, 1), CMPLX(2, 1),
                                        CMPLX(2, 2), CMPLX(3, 4),
                                        CMPLX(2, 1), CMPLX(0, 1),
                                        CMPLX(2, 0), CMPLX(0, 0)};
  double __complex__ buf_out2[8];

  s::range<1> numOfItems{8};
  {
  s::buffer<double __complex__, 1> buffer4(buf_in3, numOfItems);
  s::buffer<double __complex__, 1> buffer5(buf_in4, numOfItems);
  s::buffer<double __complex__, 1> buffer6(buf_out2, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in3_access = buffer4.get_access<sycl_read>(cgh);
    auto buf_in4_access = buffer5.get_access<sycl_read>(cgh);
    auto buf_out2_access = buffer6.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexDivides>(numOfItems, [=](s::id<1>WIid) {
      buf_out2_access[WIid] = buf_in3_access[WIid] / buf_in4_access[WIid];
    });
  });
  }

  for (size_t idx = 0; idx < 8; ++idx) {
    assert(approx_equal_c99_cmplx(buf_out2[idx], ref_results2[idx]));
  }
}

class DeviceComplexSqrt;

void device_c99_complex_sqrt(s::queue &deviceQueue) {
  double __complex__ buf_in2[4] = {CMPLX(-1, 0), CMPLX(0, 2),
                                   CMPLX(4, 0),  CMPLX(-5, 12)};
  double __complex__ buf_out2[4];
  double __complex__ ref_results2[4] = {CMPLX(0, 1), CMPLX(1, 1),
                                        CMPLX(2, 0), CMPLX(2, 3)};
  s::range<1> numOfItems{4};
  {
  s::buffer<double __complex__, 1> buffer3(buf_in2, numOfItems);
  s::buffer<double __complex__, 1> buffer4(buf_out2, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in2_access = buffer3.get_access<sycl_read>(cgh);
    auto buf_out2_access = buffer4.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexSqrt>(numOfItems, [=](s::id<1>WIid) {
      buf_out2_access[WIid] = csqrt(buf_in2_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_c99_cmplx(buf_out2[idx], ref_results2[idx]));
  }
}

class DeviceComplexAbs;

void device_c99_complex_abs(s::queue &deviceQueue) {
  double __complex__ buf_in2[4] = {CMPLX(0, 0),  CMPLX(3, 4),
                                   CMPLX(12, 5), CMPLX(INFINITY, 1)};
  double buf_out2[4];
  double ref_results2[4] = {0, 5, 13, INFINITY};
  s::range<1> numOfItems{4};
  {
  s::buffer<double __complex__, 1> buffer3(buf_in2, numOfItems);
  s::buffer<double, 1> buffer4(buf_out2, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in2_access = buffer3.get_access<sycl_read>(cgh);
    auto buf_out2_access = buffer4.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexAbs>(numOfItems, [=](s::id<1>WIid) {
      buf_out2_access[WIid] = cabs(buf_in2_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_fp(buf_out2[idx], ref_results2[idx]));
  }
}

class DeviceComplexExp;

void device_c99_complex_exp(s::queue &deviceQueue) {
  double __complex__ buf_in2[4] = {CMPLX(0, 0), CMPLX(0, M_PI_2),
                                   CMPLX(0, M_PI), CMPLX(1, M_PI_2)};
  double __complex__ buf_out2[4];
  double __complex__ ref_results2[4] = {CMPLX(1, 0), CMPLX(0, 1),
                                        CMPLX(-1, 0),CMPLX(0, M_E)};
  s::range<1> numOfItems{4};
  {
  s::buffer<double __complex__, 1> buffer3(buf_in2, numOfItems);
  s::buffer<double __complex__, 1> buffer4(buf_out2, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in2_access = buffer3.get_access<sycl_read>(cgh);
    auto buf_out2_access = buffer4.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexExp>(numOfItems, [=](s::id<1>WIid) {
      buf_out2_access[WIid] = cexp(buf_in2_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_c99_cmplx(buf_out2[idx], ref_results2[idx]));
  }
}

class DeviceComplexLog;

void device_c99_complex_log(s::queue &deviceQueue) {
  double __complex__ buf_in2[4] = {CMPLX(1, 0),  CMPLX(0, 1),
                                   CMPLX(-1, 0), CMPLX(0, M_E)};
  double __complex__ buf_out2[4];
  double __complex__ ref_results2[4] = {CMPLX(0, 0), CMPLX(0, M_PI_2),
                                        CMPLX(0, M_PI), CMPLX(1, M_PI_2)};
  s::range<1> numOfItems{4};
  {
  s::buffer<double __complex__, 1> buffer3(buf_in2, numOfItems);
  s::buffer<double __complex__, 1> buffer4(buf_out2, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in2_access = buffer3.get_access<sycl_read>(cgh);
    auto buf_out2_access = buffer4.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexLog>(numOfItems, [=](s::id<1>WIid) {
      buf_out2_access[WIid] = ::clog(buf_in2_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_c99_cmplx(buf_out2[idx], ref_results2[idx]));
  }
}

class DeviceComplexSin;

void device_c99_complex_sin(s::queue &deviceQueue) {
  double __complex__ buf_in2[2] = {CMPLX(0, 0), CMPLX(M_PI_2, 0)};
  double __complex__ buf_out2[2];
  double __complex__ ref_results2[2] = {CMPLX(0, 0), CMPLX(1, 0)};
  s::range<1> numOfItems{2};
  {
  s::buffer<double __complex__, 1> buffer3(buf_in2, numOfItems);
  s::buffer<double __complex__, 1> buffer4(buf_out2, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in2_access = buffer3.get_access<sycl_read>(cgh);
    auto buf_out2_access = buffer4.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexSin>(numOfItems, [=](s::id<1>WIid) {
      buf_out2_access[WIid] = csin(buf_in2_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 2; ++idx) {
    assert(approx_equal_c99_cmplx(buf_out2[idx], ref_results2[idx]));
  }
}

class DeviceComplexCos;

void device_c99_complex_cos(s::queue &deviceQueue) {
  double __complex__ buf_in2[2] = {CMPLX(0, 0), CMPLX(M_PI, 0)};
  double __complex__ buf_out2[2];
  double __complex__ ref_results2[2] = {CMPLX(1, 0), CMPLX(-1, 0)};
  s::range<1> numOfItems{2};
  {
  s::buffer<double __complex__, 1> buffer3(buf_in2, numOfItems);
  s::buffer<double __complex__, 1> buffer4(buf_out2, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in2_access = buffer3.get_access<sycl_read>(cgh);
    auto buf_out2_access = buffer4.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexCos>(numOfItems, [=](s::id<1>WIid) {
      buf_out2_access[WIid] = ccos(buf_in2_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 2; ++idx) {
    assert(approx_equal_c99_cmplx(buf_out2[idx], ref_results2[idx]));
  }
}

void device_c99_complex_test(s::queue &deviceQueue) {
  device_c99_complex_times(deviceQueue);
  device_c99_complex_divides(deviceQueue);
  device_c99_complex_sqrt(deviceQueue);
  device_c99_complex_abs(deviceQueue);
  device_c99_complex_exp(deviceQueue);
  device_c99_complex_log(deviceQueue);
  device_c99_complex_sin(deviceQueue);
  device_c99_complex_cos(deviceQueue);
}

int main() {
  s::queue deviceQueue;
  if (deviceQueue.get_device().has_extension("cl_khr_fp64")) {
    device_c99_complex_test(deviceQueue);
    std::cout << "Pass" << std::endl;
  }
}
