#include <cmath>
#include <iomanip>

#define SYCL_EXT_ONEAPI_COMPLEX
#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi;

#define SYCL_CPLX_TOL_ULP 5

#define PI 3.14159265358979323846

// Helper for passing infinity and nan values
template <typename T>
inline constexpr T inf_val = std::numeric_limits<T>::infinity();
template <typename T>
inline constexpr T nan_val = std::numeric_limits<T>::quiet_NaN();

/// Helpers for checking if type is supported.
template <typename T> inline bool is_type_supported(sycl::queue &Q) {
  return false;
}
template <> inline bool is_type_supported<double>(sycl::queue &Q) {
  return Q.get_device().has(sycl::aspect::fp64);
}
template <> inline bool is_type_supported<float>(sycl::queue &Q) {
  return true;
}
template <> inline bool is_type_supported<sycl::half>(sycl::queue &Q) {
  return Q.get_device().has(sycl::aspect::fp16);
}

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

/// Helper to test each complex specilization

// Overload for cplx_test_cases
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

// Overload for deci_test_cases
template <template <typename, typename> typename action, typename... argsT>
bool test_valid_types(sycl::queue &Q, argsT... args) {
  bool test_passes = true;

  if (Q.get_device().has(sycl::aspect::fp64)) {
    test_passes &= action<double, bool>{}(Q, args...);
    test_passes &= action<double, char>{}(Q, args...);
    test_passes &= action<double, int>{}(Q, args...);
    test_passes &= action<double, double>{}(Q, args...);
  }

  { test_passes &= action<float, float>{}(Q, args...); }

  if (Q.get_device().has(sycl::aspect::fp16)) {
    test_passes &= action<sycl::half, sycl::half>{}(Q, args...);
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

namespace helper::detail {

template <typename T, typename U> inline bool is_nan_or_inf(T x, U y) {
  return (std::isnan(x.real()) && std::isnan(y.real())) ||
         (std::isnan(x.imag()) && std::isnan(y.imag())) ||
         (std::isinf(x.real()) && std::isinf(y.real())) ||
         (std::isinf(x.imag()) && std::isinf(y.imag()));
}

template <typename T> T complex_magnitude(std::complex<T> a) {
  return std::sqrt(a.real() * a.real() + a.imag() * a.imag());
}

template <typename T>
bool almost_equal(std::complex<T> x, std::complex<T> y, int ulp) {
  auto diff = complex_magnitude(x - y);
  return diff <= std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp ||
         diff < std::numeric_limits<T>::min() || is_nan_or_inf(x, y);
}

template <typename T> bool almost_equal(T x, T y, int ulp) {
  if (std::isnan(x) && std::isnan(y)) {
    return true;
  } else if (std::isinf(x) && std::isinf(y)) {
    return true;
  }

  return std::abs(x - y) <=
             std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp ||
         std::abs(x - y) < std::numeric_limits<T>::min();
}

} // namespace helper::detail

/// Helpers for testing half

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

// Helpers for initializing std::complex values for tests only needed because
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

template <typename Tout, typename Tin, std::size_t NumElements>
auto constexpr convert_marray(sycl::marray<std::complex<Tin>, NumElements> c) {
  sycl::marray<std::complex<Tout>, NumElements> rtn;
  for (std::size_t i = 0; i < NumElements; ++i)
    rtn[i] = c[i];
  return rtn;
}

/// Helpers for comparing SyclCPLX and standard c++ results

namespace helper::detail {

template <typename T>
struct is_a_complex
    : std::integral_constant<bool, experimental::is_gencomplex_v<T> ||
                                       sycl::detail::is_complex<T>::value> {};

template <typename T>
inline constexpr bool is_a_complex_v = is_a_complex<T>::value;

} // namespace helper::detail

/// Specialization for scalar
template <typename T>
typename std::enable_if_t<experimental::is_genfloat_v<T>, bool>
check_results(T output, T reference, bool is_device) {
  if (!helper::detail::almost_equal(output, reference, SYCL_CPLX_TOL_ULP)) {
    std::cerr << std::setprecision(std::numeric_limits<T>::max_digits10)
              << "Test failed with complex_type: " << get_typename<T>()
              << " Computed on " << (is_device ? "device" : "host")
              << " Output: " << output << " Reference: " << reference
              << std::endl;
    return false;
  }
  return true;
}

/// Specialization for sycl::complex and std::complex
template <typename LHS, typename RHS>
typename std::enable_if_t<helper::detail::is_a_complex_v<LHS> &&
                              helper::detail::is_a_complex_v<RHS>,
                          bool>
check_results(LHS output, RHS reference, bool is_device) {
  using T1 = typename LHS::value_type;
  using T2 = typename RHS::value_type;

  if (!std::is_same_v<T1, T2>) {
    std::cerr << "check_results can be called with sycl::complex and/or "
                 "std::complex but with the same value_type"
              << std::endl;
  }

  if (!helper::detail::almost_equal(static_cast<std::complex<T1>>(output),
                                    static_cast<std::complex<T2>>(reference),
                                    SYCL_CPLX_TOL_ULP)) {
    std::cerr << std::setprecision(std::numeric_limits<T1>::max_digits10)
              << "Test failed with complex_type: " << get_typename<T1>()
              << " Computed on " << (is_device ? "device" : "host")
              << " Output: " << output << " Reference: " << reference
              << std::endl;

    return false;
  }

  return true;
}

/// Specialization for sycl::marray
template <typename LHS, std::size_t NumElementsLHS, typename RHS,
          std::size_t NumElementsRHS>
bool check_results(sycl::marray<LHS, NumElementsLHS> output,
                   sycl::marray<RHS, NumElementsRHS> reference,
                   bool is_device) {
  if (NumElementsLHS != NumElementsRHS) {
    std::cerr << "check_results can be called with sycl::marray but with the "
                 "same NumElement"
              << std::endl;
  }

  for (std::size_t i = 0; i < NumElementsLHS; ++i) {
    if (!check_results(output[i], reference[i], is_device)) {
      return false;
    }
  }

  return true;
}

/// Specialization for std::array
template <typename T, std::size_t NumElements>
bool check_results(std::array<T, NumElements> output,
                   std::array<T, NumElements> reference, bool is_device) {
  for (std::size_t i = 0; i < NumElements; ++i) {
    if (!check_results(output[i], reference[i], is_device)) {
      return false;
    }
  }

  return true;
}
