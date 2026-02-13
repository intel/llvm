// RUN: %clangxx -fsycl -fsyntax-only %s

#define SYCL_EXT_ONEAPI_COMPLEX

#include <sycl/ext/oneapi/experimental/complex/complex.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

// Helper to test each complex specilization
template <template <typename> typename action> void test_valid_types() {
  action<double> testd;
  testd();

  action<float> testf;
  testf();

  action<sycl::half> testh;
  testh();
}

// Define math operations tests
#define TEST_MATH_OP_TYPE(op_name, op)                                         \
  template <typename T> struct test##_##op_name##_##types {                    \
    bool operator()() {                                                        \
      static_assert(                                                           \
          std::is_same_v<complex<T>,                                           \
                         decltype(std::declval<complex<T>>()                   \
                                      op std::declval<complex<T>>())>);        \
      return true;                                                             \
    }                                                                          \
  };

TEST_MATH_OP_TYPE(add, +)
TEST_MATH_OP_TYPE(sub, -)
TEST_MATH_OP_TYPE(mul, *)
TEST_MATH_OP_TYPE(div, /)
#undef TEST_MATH_FUNC_TYPE

// Check operations return correct types
void check_math_operator_types() {
  test_valid_types<test_add_types>();
  test_valid_types<test_sub_types>();
  test_valid_types<test_mul_types>();
  test_valid_types<test_div_types>();
}

// Define math function tests
#define TEST_MATH_FUNC_TYPE(func)                                              \
  template <typename T> struct test##_##func##_##types {                       \
    bool operator()() {                                                        \
      static_assert(std::is_same_v<complex<T>, decltype(func(complex<T>()))>); \
      return true;                                                             \
    }                                                                          \
  };

TEST_MATH_FUNC_TYPE(acos)
TEST_MATH_FUNC_TYPE(asin)
TEST_MATH_FUNC_TYPE(atan)
TEST_MATH_FUNC_TYPE(acosh)
TEST_MATH_FUNC_TYPE(asinh)
TEST_MATH_FUNC_TYPE(atanh)
TEST_MATH_FUNC_TYPE(conj)
TEST_MATH_FUNC_TYPE(cos)
TEST_MATH_FUNC_TYPE(cosh)
TEST_MATH_FUNC_TYPE(exp)
TEST_MATH_FUNC_TYPE(log)
TEST_MATH_FUNC_TYPE(log10)
TEST_MATH_FUNC_TYPE(proj)
TEST_MATH_FUNC_TYPE(sin)
TEST_MATH_FUNC_TYPE(sinh)
TEST_MATH_FUNC_TYPE(sqrt)
TEST_MATH_FUNC_TYPE(tan)
TEST_MATH_FUNC_TYPE(tanh)
#undef TEST_MATH_FUNC_TYPE

template <typename T> struct test_abs_types {
  bool operator()() {
    static_assert(std::is_same_v<T, decltype(abs(complex<T>()))>);
    return true;
  }
};

template <typename T> struct test_polar_types {
  bool operator()() {
    static_assert(std::is_same_v<complex<T>, decltype(polar(T()))>);
    static_assert(std::is_same_v<complex<T>, decltype(polar(T(), T()))>);
    return true;
  }
};

template <typename T> struct test_pow_types {
  bool operator()() {
    static_assert(std::is_same_v<complex<T>, decltype(pow(complex<T>(), T()))>);
    static_assert(
        std::is_same_v<complex<T>, decltype(pow(complex<T>(), complex<T>()))>);
    static_assert(std::is_same_v<complex<T>, decltype(pow(T(), complex<T>()))>);
    return true;
  }
};

// Check functions return correct types
void check_math_function_types() {

  test_valid_types<test_abs_types>();
  test_valid_types<test_acos_types>();
  test_valid_types<test_asin_types>();
  test_valid_types<test_atan_types>();
  test_valid_types<test_acosh_types>();
  test_valid_types<test_asinh_types>();
  test_valid_types<test_atanh_types>();
  test_valid_types<test_conj_types>();
  test_valid_types<test_cos_types>();
  test_valid_types<test_cosh_types>();
  test_valid_types<test_exp_types>();
  test_valid_types<test_log_types>();
  test_valid_types<test_log10_types>();
  test_valid_types<test_polar_types>();
  test_valid_types<test_pow_types>();
  test_valid_types<test_proj_types>();
  test_valid_types<test_sin_types>();
  test_valid_types<test_sinh_types>();
  test_valid_types<test_sqrt_types>();
  test_valid_types<test_tan_types>();
  test_valid_types<test_tanh_types>();
}

// Check is_gencomplex
void check_is_gencomplex() {
  static_assert(is_gencomplex<complex<double>>::value == true);
  static_assert(is_gencomplex<complex<float>>::value == true);
  static_assert(is_gencomplex<complex<sycl::half>>::value == true);

  static_assert(is_gencomplex<complex<long long>>::value == false);
  static_assert(is_gencomplex<complex<long>>::value == false);
  static_assert(is_gencomplex<complex<int>>::value == false);
  static_assert(is_gencomplex<complex<unsigned long long>>::value == false);
  static_assert(is_gencomplex<complex<unsigned long>>::value == false);
  static_assert(is_gencomplex<complex<unsigned int>>::value == false);
}

// Check that a std::complex can be cast to a sycl::complex using all genfloat
// types
void check_std_to_sycl_conversion() {
  auto complex_f = sycl::ext::oneapi::experimental::complex<float>{42.f, 42.f};
  auto complex_d = sycl::ext::oneapi::experimental::complex<double>{42.f, 42.f};
  auto complex_h =
      sycl::ext::oneapi::experimental::complex<sycl::half>{42.f, 42.f};

  {
    auto f = static_cast<std::complex<float>>(complex_f);
    auto d = static_cast<std::complex<double>>(complex_f);
    auto h = static_cast<std::complex<sycl::half>>(complex_f);
  }
  {
    auto f = static_cast<std::complex<float>>(complex_d);
    auto d = static_cast<std::complex<double>>(complex_d);
    auto h = static_cast<std::complex<sycl::half>>(complex_d);
  }
  {
    auto f = static_cast<std::complex<float>>(complex_h);
    auto d = static_cast<std::complex<double>>(complex_h);
    auto h = static_cast<std::complex<sycl::half>>(complex_h);
  }
}

// Check that a sycl::complex can be constructed from a std::complex
void check_sycl_constructor_from_std() {
  auto complex_f = std::complex<float>{42.f, 42.f};
  auto complex_d = std::complex<double>{42.f, 42.f};
  auto complex_h = std::complex<sycl::half>{42.f, 42.f};

  {
    sycl::ext::oneapi::experimental::complex<float> f{complex_f};
    sycl::ext::oneapi::experimental::complex<double> d{complex_f};
    sycl::ext::oneapi::experimental::complex<sycl::half> h{complex_f};
  }
  {
    sycl::ext::oneapi::experimental::complex<float> f{complex_d};
    sycl::ext::oneapi::experimental::complex<double> d{complex_d};
    sycl::ext::oneapi::experimental::complex<sycl::half> h{complex_d};
  }
  {
    sycl::ext::oneapi::experimental::complex<float> f{complex_h};
    sycl::ext::oneapi::experimental::complex<double> d{complex_h};
    sycl::ext::oneapi::experimental::complex<sycl::half> h{complex_h};
  }
}

int main() {
  check_math_function_types();
  check_math_operator_types();

  check_is_gencomplex();

  check_std_to_sycl_conversion();
  check_sycl_constructor_from_std();

  return 0;
}
