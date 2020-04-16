// UNSUPPORTED: windows
// RUN: %clangxx -fsycl -c %s -o %t.o
// RUN: %clangxx -fsycl %t.o %sycl_libs_dir/libsycl-complex-fp64.o %sycl_libs_dir/libsycl-cmath-fp64.o -o %t.out
#include <CL/sycl.hpp>
#include <cassert>
#include "math_utils.hpp"
using namespace std;

namespace s = cl::sycl;
constexpr s::access::mode sycl_read = s::access::mode::read;
constexpr s::access::mode sycl_write = s::access::mode::write;

template <typename T>
bool approx_equal_cmplx(complex<T> x, complex<T> y) {
  return approx_equal_fp(x.real(), y.real()) && approx_equal_fp(x.imag(), y.imag());
}

template <class T>
class DeviceComplexTimes;

template <class T>
void device_complex_times(s::queue &deviceQueue) {
  complex<T> buf_in1[4] = {complex<T>(0, 1), complex<T>(1, 1),
                           complex<T>(2, 3), complex<T>(4, 5)};
  complex<T> buf_in2[4] = {complex<T>(1, 1), complex<T>(2, 1),
                           complex<T>(2, 2), complex<T>(3, 4)};
  complex<T> buf_out[4];

  complex<T> ref_results[4] = {complex<T>(-1, 1), complex<T>(1, 3),
                               complex<T>(-2, 10), complex<T>(-8, 31)};

  s::range<1> numOfItems{4};
  {
  s::buffer<complex<T>, 1> buffer1(buf_in1, numOfItems);
  s::buffer<complex<T>, 1> buffer2(buf_in2, numOfItems);
  s::buffer<complex<T>, 1> buffer3(buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in1_access = buffer1.template get_access<sycl_read>(cgh);
    auto buf_in2_access = buffer2.template get_access<sycl_read>(cgh);
    auto buf_out_access = buffer3.template get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexTimes<T>>(numOfItems, [=](s::id<1>WIid) {
      buf_out_access[WIid] = buf_in1_access[WIid] * buf_in2_access[WIid];
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_cmplx(buf_out[idx], ref_results[idx]));
  }
}

template <class T>
class DeviceComplexDivides;

template <class T>
void device_complex_divides(s::queue &deviceQueue) {
  complex<T> buf_in1[8] = {complex<T>(-1, 1), complex<T>(1, 3),
                           complex<T>(-2, 10), complex<T>(-8, 31),
                           complex<T>(4, 2), complex<T>(-1, 0),
                           complex<T>(0, 10), complex<T>(0, 0)};
  complex<T> buf_in2[8] = {complex<T>(0, 1), complex<T>(1, 1),
                           complex<T>(2, 3), complex<T>(4, 5),
                           complex<T>(2, 0), complex<T>(0, 1),
                           complex<T>(0, 5), complex<T>(1, 0)};
  complex<T> ref_results[8] = {complex<T>(1, 1), complex<T>(2, 1),
                               complex<T>(2, 2), complex<T>(3, 4),
                               complex<T>(2, 1), complex<T>(0, 1),
                               complex<T>(2, 0), complex<T>(0, 0)};
  complex<T> buf_out[8];

  s::range<1> numOfItems{8};
  {
  s::buffer<complex<T>, 1> buffer1(buf_in1, numOfItems);
  s::buffer<complex<T>, 1> buffer2(buf_in2, numOfItems);
  s::buffer<complex<T>, 1> buffer3(buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in1_access = buffer1.template get_access<sycl_read>(cgh);
    auto buf_in2_access = buffer2.template get_access<sycl_read>(cgh);
    auto buf_out_access = buffer3.template get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexDivides<T>>(numOfItems, [=](s::id<1>WIid) {
      buf_out_access[WIid] = buf_in1_access[WIid] / buf_in2_access[WIid];
    });
  });
  }

  for (size_t idx = 0; idx < 8; ++idx) {
    assert(approx_equal_cmplx(buf_out[idx], ref_results[idx]));
  }
}

template <class T>
class DeviceComplexSqrt;

template <class T>
void device_complex_sqrt(s::queue &deviceQueue) {
  complex<T> buf_in[4] = { complex<T>(-1, 0), complex<T>(0, 2),
                           complex<T>(4, 0), complex<T>(-5, 12)};
  complex<T> buf_out[4];
  complex<T> ref_results[4] = {complex<T>(0, 1), complex<T>(1, 1),
                               complex<T>(2, 0), complex<T>(2, 3)};
  s::range<1> numOfItems{4};
  {
  s::buffer<complex<T>, 1> buffer1(buf_in, numOfItems);
  s::buffer<complex<T>, 1> buffer2(buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in_access = buffer1.template get_access<sycl_read>(cgh);
    auto buf_out_access = buffer2.template get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexSqrt<T>>(numOfItems, [=](s::id<1>WIid) {
      buf_out_access[WIid] = sqrt(buf_in_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_cmplx(buf_out[idx], ref_results[idx]));
  }
}

template <class T>
class DeviceComplexNorm;

template <class T>
void device_complex_norm(s::queue &deviceQueue) {
  complex<T> buf_in[4] = {complex<T>(0, 0), complex<T>(3, 4),
                          complex<T>(12, 5), complex<T>(INFINITY, 1)};
  T buf_out[4];
  T ref_results[4] = {0, 25, 169, INFINITY};
  s::range<1> numOfItems{4};
  {
  s::buffer<complex<T>, 1> buffer1(buf_in, numOfItems);
  s::buffer<T, 1> buffer2(buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in_access = buffer1.template get_access<sycl_read>(cgh);
    auto buf_out_access = buffer2.template get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexNorm<T>>(numOfItems, [=](s::id<1>WIid) {
      buf_out_access[WIid] = norm(buf_in_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_fp(buf_out[idx], ref_results[idx]));
  }
}


template <class T>
class DeviceComplexAbs;

template <class T>
void device_complex_abs(s::queue &deviceQueue) {
  complex<T> buf_in[4] = {complex<T>(0, 0), complex<T>(3, 4),
                          complex<T>(12, 5), complex<T>(INFINITY, 1)};
  T buf_out[4];
  T ref_results[4] = {0, 5, 13, INFINITY};
  s::range<1> numOfItems{4};
  {
  s::buffer<complex<T>, 1> buffer1(buf_in, numOfItems);
  s::buffer<T, 1> buffer2(buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in_access = buffer1.template get_access<sycl_read>(cgh);
    auto buf_out_access = buffer2.template get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexAbs<T>>(numOfItems, [=](s::id<1>WIid) {
      buf_out_access[WIid] = abs(buf_in_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_fp(buf_out[idx], ref_results[idx]));
  }
}

template <class T>
class DeviceComplexExp;

template <class T>
void device_complex_exp(s::queue &deviceQueue) {
  complex<T> buf_in[4] = {complex<T>(0, 0), complex<T>(0, M_PI_2),
                          complex<T>(0, M_PI), complex<T>(1, M_PI_2)};
  complex<T> buf_out[4];
  complex<T> ref_results[4] = {complex<T>(1, 0), complex<T>(0, 1),
                               complex<T>(-1, 0), complex<T>(0, M_E)};
  s::range<1> numOfItems{4};
  {
  s::buffer<complex<T>, 1> buffer1(buf_in, numOfItems);
  s::buffer<complex<T>, 1> buffer2(buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in_access = buffer1.template get_access<sycl_read>(cgh);
    auto buf_out_access = buffer2.template get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexExp<T>>(numOfItems, [=](s::id<1>WIid) {
      buf_out_access[WIid] = exp(buf_in_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_cmplx(buf_out[idx], ref_results[idx]));
  }
}

template <class T>
class DeviceComplexLog;

template <class T>
void device_complex_log(s::queue &deviceQueue) {
  complex<T> buf_in[4] = {complex<T>(1, 0), complex<T>(0, 1),
                          complex<T>(-1, 0), complex<T>(0, M_E)};
  complex<T> buf_out[4];
  complex<T> ref_results[4] = {complex<T>(0, 0), complex<T>(0, M_PI_2),
                               complex<T>(0, M_PI), complex<T>(1, M_PI_2)};
  s::range<1> numOfItems{4};
  {
  s::buffer<complex<T>, 1> buffer1(buf_in, numOfItems);
  s::buffer<complex<T>, 1> buffer2(buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in_access = buffer1.template get_access<sycl_read>(cgh);
    auto buf_out_access = buffer2.template get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexLog<T>>(numOfItems, [=](s::id<1>WIid) {
      buf_out_access[WIid] = log(buf_in_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_cmplx(buf_out[idx], ref_results[idx]));
  }
}

template <class T>
class DeviceComplexLog10;

template <class T>
void device_complex_log10(s::queue &deviceQueue) {
  complex<T> buf_in = complex<T>(0, 0);
  complex<T> buf_out;
  complex<T> ref_result = complex<T>(-INFINITY, 0);
  s::range<1> numOfItems{1};
  {
  s::buffer<complex<T>, 1> buffer1(&buf_in, numOfItems);
  s::buffer<complex<T>, 1> buffer2(&buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in_access = buffer1.template get_access<sycl_read>(cgh);
    auto buf_out_access = buffer2.template get_access<sycl_write>(cgh);
    cgh.single_task<class DeviceComplexLog10<T>>([=]() {
      buf_out_access[0] = log10(buf_in_access[0]);
    });
  });
  }

  assert(approx_equal_cmplx(buf_out, ref_result));
}

template <class T>
class DeviceComplexSin;

template <class T>
void device_complex_sin(s::queue &deviceQueue) {
  complex<T> buf_in[2] = {complex<T>(0, 0), complex<T>(M_PI_2, 0)};
  complex<T> buf_out[2];
  complex<T> ref_results[2] = {complex<T>(0, 0), complex<T>(1, 0)};
  s::range<1> numOfItems{2};
  {
  s::buffer<complex<T>, 1> buffer1(buf_in, numOfItems);
  s::buffer<complex<T>, 1> buffer2(buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in_access = buffer1.template get_access<sycl_read>(cgh);
    auto buf_out_access = buffer2.template get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexSin<T>>(numOfItems, [=](s::id<1>WIid) {
      buf_out_access[WIid] = sin(buf_in_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 2; ++idx) {
    assert(approx_equal_cmplx(buf_out[idx], ref_results[idx]));
  }
}

template <class T>
class DeviceComplexCos;

template <class T>
void device_complex_cos(s::queue &deviceQueue) {
  complex<T> buf_in[2] = {complex<T>(0, 0), complex<T>(M_PI, 0)};
  complex<T> buf_out[2];
  complex<T> ref_results[2] = {complex<T>(1, 0), complex<T>(-1, 0)};
  s::range<1> numOfItems{2};
  {
  s::buffer<complex<T>, 1> buffer1(buf_in, numOfItems);
  s::buffer<complex<T>, 1> buffer2(buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_in_access = buffer1.template get_access<sycl_read>(cgh);
    auto buf_out_access = buffer2.template get_access<sycl_write>(cgh);
    cgh.parallel_for<class DeviceComplexCos<T>>(numOfItems, [=](s::id<1>WIid) {
      buf_out_access[WIid] = std::cos(buf_in_access[WIid]);
    });
  });
  }

  for (size_t idx = 0; idx < 2; ++idx) {
    assert(approx_equal_cmplx(buf_out[idx], ref_results[idx]));
  }
}

template <class T>
class DeviceComplexPolar;

template <class T>
void device_complex_polar(s::queue &deviceQueue) {
  complex<T> buf_out[4];
  complex<T> ref_results[4] = {complex<T>(1, 0), complex<T>(10, 0),
                               complex<T>(100, 0), complex<T>(200, 0)};
  s::range<1> numOfItems{4};
  {
  s::buffer<complex<T>, 1> buffer1(buf_out, numOfItems);
  deviceQueue.submit([&](s::handler &cgh) {
    auto buf_out_access = buffer1.template get_access<sycl_write>(cgh);
    cgh.single_task<class DeviceComplexPolar<T>>([=]() {
      buf_out_access[0] = std::polar(T(1));
      buf_out_access[1] = std::polar(T(10), T(0));
      buf_out_access[2] = std::polar(T(100));
      buf_out_access[3] = std::polar(T(200), T(0));
    });
  });
  }

  for (size_t idx = 0; idx < 4; ++idx) {
    assert(approx_equal_cmplx(buf_out[idx], ref_results[idx]));
  }
}

template <class T>
void device_complex_test(s::queue &deviceQueue) {
  device_complex_times<T>(deviceQueue);
  device_complex_divides<T>(deviceQueue);
  device_complex_sqrt<T>(deviceQueue);
  device_complex_norm<T>(deviceQueue);
  device_complex_abs<T>(deviceQueue);
  device_complex_exp<T>(deviceQueue);
  device_complex_log<T>(deviceQueue);
  device_complex_log10<T>(deviceQueue);
  device_complex_sin<T>(deviceQueue);
  device_complex_cos<T>(deviceQueue);
  device_complex_polar<T>(deviceQueue);
}

int main() {
  s::queue deviceQueue;
  if (deviceQueue.get_device().has_extension("cl_khr_fp64")) {
    device_complex_test<double>(deviceQueue);
    cout << "Pass" << endl;
  }
}
