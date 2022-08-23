#include <cmath>
#include <iomanip>

#define SYCL_EXT_ONEAPI_COMPLEX
#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi;

#define SYCL_CPLX_TOL_ULP 5

#define PI 3.14159265358979323846

constexpr double INFINITYd(std::numeric_limits<double>::infinity());
constexpr double NANd(std::numeric_limits<double>::quiet_NaN());

template <typename T = double> struct cmplx {
  cmplx(T real, T imag) : re(real), im(imag) {}

  template <typename X> cmplx(cmplx<X> c) {
    re = c.re;
    im = c.im;
  }

  T re;
  T im;
};

// Helpers for displaying results

template <typename T> const char *get_typename() { return "Unknown type"; }
template <> const char *get_typename<double>() { return "double"; }
template <> const char *get_typename<float>() { return "float"; }
template <> const char *get_typename<sycl::half>() { return "sycl::half"; }

// Helper to test each complex specilization
template <template <typename> typename action, typename... argsT>
bool test_valid_types(sycl::queue &Q, argsT... args) {
  bool test_passes = true;

  if (Q.get_device().has(sycl::aspect::fp64)) {
    action<double> test;
    test_passes &= test(Q, args...);
  }

  {
    action<float> test;
    test_passes &= test(Q, args...);
  }

  if (Q.get_device().has(sycl::aspect::fp16)) {
    action<sycl::half> test;
    test_passes &= test(Q, args...);
  }

  return test_passes;
}

// Overload for host only tests
template <template <typename> typename action, typename... argsT>
bool test_valid_types(argsT... args) {
  bool test_passes = true;

  {
    action<double> test;
    test_passes &= test(args...);
  }

  {
    action<float> test;
    test_passes &= test(args...);
  }

  {
    action<sycl::half> test;
    test_passes &= test(args...);
  }

  return test_passes;
}

// Helpers for comparison

template <typename T> bool almost_equal_scalar(T x, T y, int ulp) {
  if (std::isnan(x) && std::isnan(y))
    return true;

  if (std::isinf(x) && std::isinf(y))
    return true;

  return std::abs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp ||
         std::abs(x - y) < std::numeric_limits<T>::min();
}

template <typename T> T complex_magnitude(std::complex<T> a) {
  return std::sqrt(a.real() * a.real() + a.imag() * a.imag());
}

template <typename T> T complex_magnitude(experimental::complex<T> a) {
  return std::sqrt(a.real() * a.real() + a.imag() * a.imag());
}

template <typename T, typename U> inline bool is_nan_or_inf(T x, U y) {
  return (std::isnan(x.real()) && std::isnan(y.real())) ||
         (std::isnan(x.imag()) && std::isnan(y.imag())) ||
         (std::isinf(x.real()) && std::isinf(y.real())) ||
         (std::isinf(x.imag()) && std::isinf(y.imag()));
}

template <typename T>
bool almost_equal_cplx(experimental::complex<T> x, experimental::complex<T> y,
                       int ulp) {
  auto diff = complex_magnitude(x - y);
  return diff <= std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp ||
         diff < std::numeric_limits<T>::min() || is_nan_or_inf(x, y);
}

template <typename T>
bool almost_equal_cplx(experimental::complex<T> x, std::complex<T> y, int ulp) {
  auto stdx = std::complex{x.real(), x.imag()};
  auto diff = complex_magnitude(stdx - y);
  return diff <= std::numeric_limits<T>::epsilon() * std::abs(stdx + y) * ulp ||
         diff < std::numeric_limits<T>::min() || is_nan_or_inf(x, y);
}

template <typename T>
bool almost_equal_cplx(std::complex<T> x, experimental::complex<T> y, int ulp) {
  auto stdy = std::complex{y.real(), y.imag()};
  auto diff = complex_magnitude(x - stdy);
  return diff <= std::numeric_limits<T>::epsilon() * std::abs(x + stdy) * ulp ||
         diff < std::numeric_limits<T>::min() || is_nan_or_inf(x, y);
}

template <typename T>
bool almost_equal_cplx(std::complex<T> x, std::complex<T> y, int ulp) {
  auto diff = complex_magnitude(x - y);
  return diff <= std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp ||
         diff < std::numeric_limits<T>::min() || is_nan_or_inf(x, y);
}

// Helpers for testing half

std::complex<float> sycl_half_to_float(experimental::complex<sycl::half> c) {
  auto c_sycl_float = static_cast<experimental::complex<float>>(c);
  return static_cast<std::complex<float>>(c_sycl_float);
}

std::complex<sycl::half> sycl_float_to_half(std::complex<float> c) {
  auto c_sycl_half = static_cast<experimental::complex<sycl::half>>(c);
  return static_cast<std::complex<sycl::half>>(c_sycl_half);
}

std::complex<float> trunc_float(std::complex<float> c) {
  auto c_sycl_half = static_cast<experimental::complex<sycl::half>>(c);
  return sycl_half_to_float(c_sycl_half);
}

// Helper for initializing std::complex values for tests only needed because
// sycl::half cases are emulated with float for std::complex class

template <typename T_in> auto constexpr init_std_complex(T_in re, T_in im) {
  return std::complex<T_in>(re, im);
}

template <> auto constexpr init_std_complex(sycl::half re, sycl::half im) {
  return trunc_float(std::complex<float>(re, im));
}

template <typename T_in> auto constexpr init_deci(T_in re) { return re; }

template <> auto constexpr init_deci(sycl::half re) {
  return static_cast<float>(re);
}

// Helper for comparing SyclCPLX and standard c++ results

template <typename T>
bool check_results(experimental::complex<T> output, std::complex<T> reference,
                   bool is_device) {
  if (!almost_equal_cplx(output, reference, SYCL_CPLX_TOL_ULP)) {
    std::cerr << std::setprecision(std::numeric_limits<T>::max_digits10)
              << "Test failed with complex_type: " << get_typename<T>()
              << " Computed on " << (is_device ? "device" : "host")
              << " Output: " << output << " Reference: " << reference
              << std::endl;
    return false;
  }
  return true;
}

template <typename T>
bool check_results(T output, T reference, bool is_device) {
  if (!almost_equal_scalar(output, reference, SYCL_CPLX_TOL_ULP)) {
    std::cerr << std::setprecision(std::numeric_limits<T>::max_digits10)
              << "Test failed with complex_type: " << get_typename<T>()
              << " Computed on " << (is_device ? "device" : "host")
              << " Output: " << output << " Reference: " << reference
              << std::endl;
    return false;
  }
  return true;
}
