// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-complex.o -o %t.out
#include <CL/sycl.hpp>
#include <cassert>
#include <complex.h>
#include "math_utils.hpp"

#ifndef CMPLXF
#define CMPLXF(r, i) ((float __complex__){ (float)r, (float)i })
#endif

bool approx_equal_c99_cmplxf(float __complex__ x, float __complex__ y) {
  return approx_equal_fp(crealf(x), crealf(y)) && approx_equal_fp(cimagf(x), cimagf(y));
}

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

class DeviceComplexTimes;

void device_c99_complex_times(s::queue &deviceQueue) {
  float __complex__ buf_in1[4] = {CMPLXF(0, 1), CMPLXF(1, 1),
                                  CMPLXF(2, 3), CMPLXF(4, 5)};
  float __complex__ buf_in2[4] = {CMPLXF(1, 1), CMPLXF(2, 1),
                                  CMPLXF(2, 2), CMPLXF(3, 4)};
  float __complex__ buf_out1[4];

  float __complex__ ref_results1[4] = {CMPLXF(-1, 1),  CMPLXF(1, 3),
                                       CMPLXF(-2, 10), CMPLXF(-8, 31)};

  s::range<1> numOfItems{4};
  {
  s::buffer<float __complex__, 1> buffer1(buf_in1, numOfItems);
  s::buffer<float __complex__, 1> buffer2(buf_in2, numOfItems);
  s::buffer<float __complex__, 1> buffer3(buf_out1, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in1_access = buffer1.get_access<sycl_read>(cgh);
    auto buf_in2_access = buffer2.get_access<sycl_read>(cgh);
    auto buf_out1_access = buffer3.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexTimes>(numOfItems, [=](s::id<1>WIid) {
      buf_out1_access[WIid] = buf_in1_access[WIid] * buf_in2_access[WIid];
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_c99_cmplxf(buf_out1[idx], ref_results1[idx]));
  }
}

class DeviceComplexDivides;

void device_c99_complex_divides(s::queue &deviceQueue) {
  float __complex__ buf_in1[8] = {CMPLXF(-1, 1),  CMPLXF(1, 3),
                                  CMPLXF(-2, 10), CMPLXF(-8, 31),
                                  CMPLXF(4, 2), CMPLXF(-1, 0),
                                  CMPLXF(0, 10), CMPLXF(0 , 0)};
  float __complex__ buf_in2[8] = {CMPLXF(0, 1), CMPLXF(1, 1),
                                  CMPLXF(2, 3), CMPLXF(4, 5),
                                  CMPLXF(2, 0), CMPLXF(0, 1),
                                  CMPLXF(0, 5), CMPLXF(1, 0)};
  float __complex__ ref_results1[8] = {CMPLXF(1, 1), CMPLXF(2, 1),
                                       CMPLXF(2, 2), CMPLXF(3, 4),
                                       CMPLXF(2, 1), CMPLXF(0, 1),
                                       CMPLXF(2, 0), CMPLXF(0, 0)};
  float __complex__ buf_out1[8];

  s::range<1> numOfItems{8};
  {
  s::buffer<float __complex__, 1> buffer1(buf_in1, numOfItems);
  s::buffer<float __complex__, 1> buffer2(buf_in2, numOfItems);
  s::buffer<float __complex__, 1> buffer3(buf_out1,numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in1_access = buffer1.get_access<sycl_read>(cgh);
    auto buf_in2_access = buffer2.get_access<sycl_read>(cgh);
    auto buf_out1_access = buffer3.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexDivides>(numOfItems, [=](s::id<1>WIid) {
      buf_out1_access[WIid] = buf_in1_access[WIid] / buf_in2_access[WIid];
    });
  });
  }

  for (size_t idx = 0; idx < 8; ++idx) {
    assert(approx_equal_c99_cmplxf(buf_out1[idx], ref_results1[idx]));
  }
}

class DeviceComplexSqrt;

void device_c99_complex_sqrt(s::queue &deviceQueue) {
  float __complex__ buf_in1[4] = {CMPLXF(-1, 0), CMPLXF(0, 2),
                                 CMPLXF(4, 0),  CMPLXF(-5, 12)};
  float __complex__ buf_out1[4];
  float __complex__ ref_results1[4] = {CMPLXF(0, 1), CMPLXF(1, 1),
                                       CMPLXF(2, 0), CMPLXF(2, 3)};

  s::range<1> numOfItems{4};
  {
  s::buffer<float __complex__, 1> buffer1(buf_in1, numOfItems);
  s::buffer<float __complex__, 1> buffer2(buf_out1, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in1_access = buffer1.get_access<sycl_read>(cgh);
    auto buf_out1_access = buffer2.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexSqrt>(numOfItems, [=](s::id<1>WIid) {
      buf_out1_access[WIid] = csqrtf(buf_in1_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_c99_cmplxf(buf_out1[idx], ref_results1[idx]));
  }
}

class DeviceComplexAbs;

void device_c99_complex_abs(s::queue &deviceQueue) {
  float __complex__ buf_in1[4] = {CMPLXF(0, 0),  CMPLXF(3, 4),
                                  CMPLXF(12, 5), CMPLXF(INFINITY, 1)};
  float buf_out1[4];
  float ref_results1[4] = {0, 5, 13, INFINITY};

  s::range<1> numOfItems{4};
  {
  s::buffer<float __complex__, 1> buffer1(buf_in1, numOfItems);
  s::buffer<float, 1> buffer2(buf_out1, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in1_access = buffer1.get_access<sycl_read>(cgh);
    auto buf_out1_access = buffer2.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexAbs>(numOfItems, [=](s::id<1>WIid) {
      buf_out1_access[WIid] = cabsf(buf_in1_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_fp(buf_out1[idx], ref_results1[idx]));
  }
}

class DeviceComplexExp;

void device_c99_complex_exp(s::queue &deviceQueue) {
  float __complex__ buf_in1[4] = {CMPLXF(0, 0), CMPLXF(0, M_PI_2),
                                 CMPLXF(0, M_PI), CMPLXF(1, M_PI_2)};
  float __complex__ buf_out1[4];
  float __complex__ ref_results1[4] = {CMPLXF(1, 0), CMPLXF(0, 1),
                                       CMPLXF(-1, 0),CMPLXF(0, M_E)};
  s::range<1> numOfItems{4};
  {
  s::buffer<float __complex__, 1> buffer1(buf_in1, numOfItems);
  s::buffer<float __complex__, 1> buffer2(buf_out1, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in1_access = buffer1.get_access<sycl_read>(cgh);
    auto buf_out1_access = buffer2.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexExp>(numOfItems, [=](s::id<1>WIid) {
      buf_out1_access[WIid] = cexpf(buf_in1_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_c99_cmplxf(buf_out1[idx], ref_results1[idx]));
  }
}

class DeviceComplexLog;

void device_c99_complex_log(s::queue &deviceQueue) {
  float __complex__ buf_in1[4] = {CMPLXF(1, 0),  CMPLXF(0, 1),
                                  CMPLXF(-1, 0), CMPLXF(0, M_E)};
  float __complex__ buf_out1[4];
  float __complex__ ref_results1[4] = {CMPLXF(0, 0), CMPLXF(0, M_PI_2),
                                       CMPLXF(0, M_PI), CMPLXF(1, M_PI_2)};
  s::range<1> numOfItems{4};
  {
  s::buffer<float __complex__, 1> buffer1(buf_in1, numOfItems);
  s::buffer<float __complex__, 1> buffer2(buf_out1, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in1_access = buffer1.get_access<sycl_read>(cgh);
    auto buf_out1_access = buffer2.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexLog>(numOfItems, [=](s::id<1>WIid) {
      buf_out1_access[WIid] = clogf(buf_in1_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_c99_cmplxf(buf_out1[idx], ref_results1[idx]));
  }
}

class DeviceComplexSin;

void device_c99_complex_sin(s::queue &deviceQueue) {
  float __complex__ buf_in1[2] = {CMPLXF(0, 0), CMPLXF(M_PI_2, 0)};
  float __complex__ buf_out1[2];
  float __complex__ ref_results1[2] = {CMPLXF(0, 0), CMPLXF(1, 0)};
  s::range<1> numOfItems{2};
  {
  s::buffer<float __complex__, 1> buffer1(buf_in1, numOfItems);
  s::buffer<float __complex__, 1> buffer2(buf_out1, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in1_access = buffer1.get_access<sycl_read>(cgh);
    auto buf_out1_access = buffer2.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexSin>(numOfItems, [=](s::id<1>WIid) {
      buf_out1_access[WIid] = csinf(buf_in1_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 2; ++idx) {
    assert(approx_equal_c99_cmplxf(buf_out1[idx], ref_results1[idx]));
  }
}

class DeviceComplexCos;

void device_c99_complex_cos(s::queue &deviceQueue) {
  float __complex__ buf_in1[2] = {CMPLXF(0, 0), CMPLXF(M_PI, 0)};
  float __complex__ buf_out1[2];
  float __complex__ ref_results1[2] = {CMPLXF(1, 0), CMPLXF(-1, 0)};
  s::range<1> numOfItems{2};
  {
  s::buffer<float __complex__, 1> buffer1(buf_in1, numOfItems);
  s::buffer<float __complex__, 1> buffer2(buf_out1, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in1_access = buffer1.get_access<sycl_read>(cgh);
    auto buf_out1_access = buffer2.get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexCos>(numOfItems, [=](s::id<1>WIid) {
      buf_out1_access[WIid] = ccosf(buf_in1_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 2; ++idx) {
    assert(approx_equal_c99_cmplxf(buf_out1[idx], ref_results1[idx]));
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
  device_c99_complex_test(deviceQueue);
  std::cout << "Pass" << std::endl;
}
