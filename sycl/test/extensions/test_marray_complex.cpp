// RUN: %clangxx -fsycl -fsyntax-only %s

#define SYCL_EXT_ONEAPI_COMPLEX

#include <sycl/ext/oneapi/experimental/sycl_complex.hpp>
#include <sycl/sycl.hpp>

using namespace sycl::ext::oneapi::experimental;

// Helper to test each complex specilization
template <template <typename, std::size_t> typename action>
void test_valid_types() {
  action<double, 42> testd;
  testd();

  action<float, 42> testf;
  testf();

  action<sycl::half, 42> testh;
  testh();
}

// Define math function tests - marray<complex<T>> -> marray<T>
#define TEST_MATH_FUNC_TYPE(name)                                              \
  template <typename T, std::size_t NumElements>                               \
  struct test##_##name##_##types {                                             \
    bool operator()() {                                                        \
      sycl::queue Q;                                                           \
                                                                               \
      static_assert(std::is_same_v<                                            \
                    sycl::marray<T, NumElements>,                              \
                    decltype(name(sycl::marray<complex<T>, NumElements>()))>); \
                                                                               \
      return true;                                                             \
    }                                                                          \
  };

TEST_MATH_FUNC_TYPE(abs)
#undef TEST_MATH_FUNC_TYPE

// Define math function tests - marray<complex<T>> -> marray<complex<T>>
#define TEST_MATH_FUNC_TYPE(name)                                              \
  template <typename T, std::size_t NumElements>                               \
  struct test##_##name##_##types {                                             \
    bool operator()() {                                                        \
      sycl::queue Q;                                                           \
                                                                               \
      static_assert(std::is_same_v<                                            \
                    sycl::marray<complex<T>, NumElements>,                     \
                    decltype(name(sycl::marray<complex<T>, NumElements>()))>); \
                                                                               \
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

// Define math operator tests
#define TEST_MATH_OP_TYPE(name, op)                                            \
  template <typename T, std::size_t NumElements>                               \
  struct test##_##name##_##types {                                             \
    bool operator()() {                                                        \
      sycl::queue Q;                                                           \
                                                                               \
      static_assert(                                                           \
          std::is_same_v<                                                      \
              sycl::marray<complex<T>, NumElements>,                           \
              decltype(std::declval<sycl::marray<complex<T>, NumElements>>()   \
                           op std::declval<                                    \
                               sycl::marray<complex<T>, NumElements>>())>);    \
                                                                               \
      static_assert(                                                           \
          std::is_same_v<                                                      \
              sycl::marray<complex<T>, NumElements>,                           \
              decltype(std::declval<sycl::marray<complex<T>, NumElements>>()   \
                           op std::declval<sycl::marray<T, NumElements>>())>); \
                                                                               \
      static_assert(                                                           \
          std::is_same_v<sycl::marray<complex<T>, NumElements>,                \
                         decltype(std::declval<sycl::marray<T, NumElements>>() \
                                      op std::declval<sycl::marray<            \
                                          complex<T>, NumElements>>())>);      \
                                                                               \
      return true;                                                             \
    }                                                                          \
  };

TEST_MATH_OP_TYPE(add, +)
TEST_MATH_OP_TYPE(sub, -)
TEST_MATH_OP_TYPE(mul, *)
TEST_MATH_OP_TYPE(div, /)
#undef TEST_MATH_OP_TYPE

// Define logic operator tests
#define TEST_LOGIC_OP_TYPE(name, op)                                           \
  template <typename T, std::size_t NumElements>                               \
  struct test##_##name##_##types {                                             \
    bool operator()() {                                                        \
      sycl::queue Q;                                                           \
                                                                               \
      static_assert(                                                           \
          std::is_same_v<                                                      \
              sycl::marray<bool, NumElements>,                                 \
              decltype(std::declval<sycl::marray<complex<T>, NumElements>>()   \
                           op std::declval<                                    \
                               sycl::marray<complex<T>, NumElements>>())>);    \
                                                                               \
      static_assert(                                                           \
          std::is_same_v<                                                      \
              sycl::marray<bool, NumElements>,                                 \
              decltype(std::declval<sycl::marray<complex<T>, NumElements>>()   \
                           op std::declval<sycl::marray<T, NumElements>>())>); \
                                                                               \
      static_assert(                                                           \
          std::is_same_v<sycl::marray<bool, NumElements>,                      \
                         decltype(std::declval<sycl::marray<T, NumElements>>() \
                                      op std::declval<sycl::marray<            \
                                          complex<T>, NumElements>>())>);      \
                                                                               \
      return true;                                                             \
    }                                                                          \
  };

TEST_LOGIC_OP_TYPE(equal, ==)
TEST_LOGIC_OP_TYPE(not_equal, !=)
#undef TEST_LOGIC_OP_TYPE

// Define polar function tests
template <typename T, std::size_t NumElements> struct test_polar_types {
  bool operator()() {
    sycl::queue Q;

    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(polar(sycl::marray<T, NumElements>()))>);
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(polar(sycl::marray<T, NumElements>(),
                                      sycl::marray<T, NumElements>()))>);
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(polar(sycl::marray<T, NumElements>(), T()))>);
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(polar(T(), sycl::marray<T, NumElements>()))>);

    return true;
  }
};

// Define pow function tests
template <typename T, std::size_t NumElements> struct test_pow_types {
  bool operator()() {
    sycl::queue Q;

    // complex-deci
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                    sycl::marray<T, NumElements>()))>);
    static_assert(std::is_same_v<
                  sycl::marray<complex<T>, NumElements>,
                  decltype(pow(sycl::marray<complex<T>, NumElements>(), T()))>);
    static_assert(std::is_same_v<
                  sycl::marray<complex<T>, NumElements>,
                  decltype(pow(complex<T>(), sycl::marray<T, NumElements>()))>);

    // complex-complex
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                    sycl::marray<complex<T>, NumElements>()))>);
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(pow(sycl::marray<complex<T>, NumElements>(),
                                    complex<T>()))>);
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(pow(complex<T>(),
                                    sycl::marray<complex<T>, NumElements>()))>);

    // deci-complx
    static_assert(
        std::is_same_v<sycl::marray<complex<T>, NumElements>,
                       decltype(pow(sycl::marray<T, NumElements>(),
                                    sycl::marray<complex<T>, NumElements>()))>);
    static_assert(std::is_same_v<sycl::marray<complex<T>, NumElements>,
                                 decltype(pow(sycl::marray<T, NumElements>(),
                                              complex<T>()))>);
    static_assert(std::is_same_v<
                  sycl::marray<complex<T>, NumElements>,
                  decltype(pow(T(), sycl::marray<complex<T>, NumElements>()))>);

    return true;
  }
};

int main() {
  // Check math function types
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
  test_valid_types<test_proj_types>();
  test_valid_types<test_sin_types>();
  test_valid_types<test_sinh_types>();
  test_valid_types<test_sqrt_types>();
  test_valid_types<test_tan_types>();
  test_valid_types<test_tanh_types>();

  // Check math function types
  test_valid_types<test_add_types>();
  test_valid_types<test_sub_types>();
  test_valid_types<test_mul_types>();
  test_valid_types<test_div_types>();

  // Check logic operator types
  test_valid_types<test_equal_types>();
  test_valid_types<test_not_equal_types>();

  // Check special function types
  test_valid_types<test_polar_types>();
  test_valid_types<test_pow_types>();

  return 0;
}
