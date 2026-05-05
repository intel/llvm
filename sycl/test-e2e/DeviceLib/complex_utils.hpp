#pragma once
#include <array>
#include <cmath>
#include <complex>
#include <sycl/detail/core.hpp>
enum { zero, non_zero, inf, NaN, non_zero_nan };
template <typename T> int complex_classify(std::complex<T> x) {
  if (x == std::complex<T>(0, 0))
    return zero;
  if (std::isinf(x.real()) || std::isinf(x.imag()))
    return inf;
  if (std::isnan(x.real()) && std::isnan(x.imag()))
    return NaN;
  if (std::isnan(x.real())) {
    if (x.imag() == 0)
      return NaN;
    return non_zero_nan;
  }
  if (std::isnan(x.imag())) {
    if (x.real() == 0)
      return NaN;
    return non_zero_nan;
  }
  return non_zero;
}

template <typename T>
int complex_compare_mul(std::complex<T> x, std::complex<T> y,
                        std::complex<T> z) {
  switch (complex_classify<T>(x)) {
  case zero:
    switch (complex_classify<T>(y)) {
    case zero:
      if (complex_classify<T>(z) != zero)
        return 1;
      break;
    case non_zero:
      if (complex_classify<T>(z) != zero)
        return 1;
      break;
    case inf:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case NaN:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    }
    break;
  case non_zero:
    switch (complex_classify<T>(y)) {
    case zero:
      if (complex_classify<T>(z) != zero)
        return 1;
      break;
    case non_zero:
      if (complex_classify<T>(z) != non_zero)
        return 1;
      {
        T a = x.real(), b = x.imag(), c = y.real(), d = y.imag();
        std::complex<T> t(a * c - b * d, a * d + b * c);
        // relaxed tolerance to arbitrary (1.e-6) amount.
        if (std::abs((z - t) / z) > 1.e-6)
          return 1;
      }
      break;
    case inf:
      if (complex_classify<T>(z) != inf)
        return 1;
      break;
    case NaN:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    }
    break;
  case inf:
    switch (complex_classify<T>(y)) {
    case zero:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero:
      if (complex_classify<T>(z) != inf)
        return 1;
      break;
    case inf:
      if (complex_classify<T>(z) != inf)
        return 1;
      break;
    case NaN:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (complex_classify<T>(z) != inf)
        return 1;
      break;
    }
    break;
  case NaN:
    switch (complex_classify<T>(y)) {
    case zero:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case inf:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case NaN:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    }
    break;
  case non_zero_nan:
    switch (complex_classify<T>(y)) {
    case zero:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case inf:
      if (complex_classify<T>(z) != inf)
        return 1;
      break;
    case NaN:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    }
    break;
  }

  return 0;
}

template <typename T>
int complex_compare_div(std::complex<T> x, std::complex<T> y,
                        std::complex<T> z) {
  switch (complex_classify<T>(x)) {
  case zero:
    switch (complex_classify<T>(y)) {
    case zero:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero:
      if (complex_classify<T>(z) != zero)
        return 1;
      break;
    case inf:
      if (complex_classify<T>(z) != zero)
        return 1;
      break;
    case NaN:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    }
    break;
  case non_zero:
    switch (complex_classify<T>(y)) {
    case zero:
      if (complex_classify<T>(z) != inf)
        return 1;
      break;
    case non_zero:
      if (complex_classify<T>(z) != non_zero)
        return 1;
      {
        T a = x.real(), b = x.imag(), c = y.real(), d = y.imag();
        std::complex<T> t((a * c + b * d) / (c * c + d * d),
                          (b * c - a * d) / (c * c + d * d));
        if (std::abs((z - t) / z) > 1.e-6)
          return 1;
      }
      break;
    case inf:
      if (complex_classify<T>(z) != zero)
        return 1;
      break;
    case NaN:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    }
    break;
  case inf:
    switch (complex_classify<T>(y)) {
    case zero:
      if (complex_classify<T>(z) != inf)
        return 1;
      break;
    case non_zero:
      if (complex_classify<T>(z) != inf)
        return 1;
      break;
    case inf:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case NaN:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    }
    break;
  case NaN:
    switch (complex_classify<T>(y)) {
    case zero:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case inf:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case NaN:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    }
    break;
  case non_zero_nan:
    switch (complex_classify<T>(y)) {
    case zero:
      if (complex_classify<T>(z) != inf)
        return 1;
      break;
    case non_zero:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case inf:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case NaN:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    case non_zero_nan:
      if (complex_classify<T>(z) != NaN)
        return 1;
      break;
    }
    break;
  }

  return 0;
}
template <typename T, size_t InputSize>
int device_complex_test_mul(sycl::queue &deviceQueue,
                            std::complex<T> *complex_input) {
  constexpr size_t OutputSize = InputSize * InputSize;
  sycl::range<1> numOfMulInput{InputSize};
  sycl::range<1> numOfMulOutput{OutputSize};
  std::array<std::complex<T>, OutputSize> complex_mul_result;
  {
    sycl::buffer<std::complex<T>, 1> buffer_complex_mul(complex_input,
                                                        numOfMulInput);
    sycl::buffer<std::complex<T>, 1> buffer_complex_mul_res(
        complex_mul_result.data(), numOfMulOutput);
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto complex_mul_access =
          buffer_complex_mul.template get_access<sycl::access::mode::read>(cgh);
      auto complex_mul_res_access =
          buffer_complex_mul_res.template get_access<sycl::access::mode::write>(
              cgh);
      cgh.single_task<class DeviceComplexMulTest>([=]() {
        size_t i, j;
        for (i = 0; i < InputSize; ++i) {
          for (j = 0; j < InputSize; ++j)
            complex_mul_res_access[i * InputSize + j] =
                complex_mul_access[i] * complex_mul_access[j];
        }
      });
    });
  }

  size_t i, j;
  for (i = 0; i < InputSize; ++i)
    for (j = 0; j < InputSize; ++j) {
      if (complex_compare_mul(complex_input[i], complex_input[j],
                              complex_mul_result[i * InputSize + j])) {
        return 1;
      }
    }
  return 0;
}

template <typename T, size_t InputSize>
int device_complex_test_div(sycl::queue &deviceQueue,
                            std::complex<T> *complex_input) {
  constexpr size_t OutputSize = InputSize * InputSize;
  sycl::range<1> numOfDivInput{InputSize};
  sycl::range<1> numOfDivOutput{OutputSize};
  std::array<std::complex<T>, OutputSize> complex_div_result;
  {
    sycl::buffer<std::complex<T>, 1> buffer_complex_div(complex_input,
                                                        numOfDivInput);
    sycl::buffer<std::complex<T>, 1> buffer_complex_div_res(
        complex_div_result.data(), numOfDivOutput);
    deviceQueue.submit([&](sycl::handler &cgh) {
      auto complex_div_access =
          buffer_complex_div.template get_access<sycl::access::mode::read>(cgh);
      auto complex_div_res_access =
          buffer_complex_div_res.template get_access<sycl::access::mode::write>(
              cgh);
      cgh.single_task<class DeviceComplexDivTest>([=]() {
        size_t i, j;
        for (i = 0; i < InputSize; ++i) {
          for (j = 0; j < InputSize; ++j)
            complex_div_res_access[i * InputSize + j] =
                complex_div_access[i] / complex_div_access[j];
        }
      });
    });
  }

  size_t i, j;
  for (i = 0; i < InputSize; ++i)
    for (j = 0; j < InputSize; ++j) {
      if (complex_compare_div(complex_input[i], complex_input[j],
                              complex_div_result[i * InputSize + j])) {
        return 1;
      }
    }
  return 0;
}
