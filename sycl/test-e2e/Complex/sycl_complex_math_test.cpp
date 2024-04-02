// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} -fsycl-device-code-split=per_kernel %{mathflags} -o %t.out
// RUN: %{run} %t.out

#include "sycl_complex_helper.hpp"
#include "sycl_complex_math_test_cases.hpp"

// Macro for testing complex in, complex out functions

#define TEST_MATH_OP_TYPE(math_func)                                           \
  template <typename T> struct test_##math_func {                              \
    bool operator()(sycl::queue &Q, cmplx<T> init,                             \
                    cmplx<T> ref = cmplx<T>(0, 0), bool use_ref = false) {     \
      bool pass = true;                                                        \
      auto std_in = init_std_complex(init.re, init.im);                        \
      experimental::complex<T> cplx_input{init.re, init.im};                   \
      sycl::buffer<experimental::complex<T>> cplx_out_buf{sycl::range{1}};     \
      /*Get std::complex output*/                                              \
      std::complex<T> std_out{ref.re, ref.im};                                 \
      if (!use_ref)                                                            \
        std_out = std::math_func(std_in);                                      \
      /*Check cplx::complex output from device*/                               \
      Q.submit([&](sycl::handler &h) {                                         \
        sycl::accessor cplx_out{cplx_out_buf, h};                              \
        h.single_task(                                                         \
            [=]() { cplx_out[0] = experimental::math_func<T>(cplx_input); });  \
      });                                                                      \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ true);     \
                                                                               \
      /*Check cplx::complex output from host*/                                 \
      cplx_out_acc[0] = experimental::math_func<T>(cplx_input);                \
                                                                               \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ false);    \
      return pass;                                                             \
    }                                                                          \
  };

TEST_MATH_OP_TYPE(acos)
TEST_MATH_OP_TYPE(asin)
TEST_MATH_OP_TYPE(atan)
TEST_MATH_OP_TYPE(acosh)
TEST_MATH_OP_TYPE(asinh)
TEST_MATH_OP_TYPE(atanh)
TEST_MATH_OP_TYPE(conj)
TEST_MATH_OP_TYPE(cos)
TEST_MATH_OP_TYPE(cosh)
TEST_MATH_OP_TYPE(exp)
TEST_MATH_OP_TYPE(log)
TEST_MATH_OP_TYPE(log10)
TEST_MATH_OP_TYPE(proj)
TEST_MATH_OP_TYPE(sin)
TEST_MATH_OP_TYPE(sinh)
TEST_MATH_OP_TYPE(sqrt)
TEST_MATH_OP_TYPE(tan)
TEST_MATH_OP_TYPE(tanh)

#undef TEST_MATH_OP_TYPE

// Macro for testing complex in, decimal out functions

#define TEST_MATH_OP_TYPE(math_func)                                           \
  template <typename T> struct test_##math_func {                              \
    bool operator()(sycl::queue &Q, cmplx<T> init,                             \
                    cmplx<T> ref = cmplx<T>(0, 0), bool use_ref = false) {     \
      bool pass = true;                                                        \
                                                                               \
      auto std_in = init_std_complex(init.re, init.im);                        \
      experimental::complex<T> cplx_input{init.re, init.im};                   \
      sycl::buffer<T> cplx_out_buf{sycl::range{1}};                            \
                                                                               \
      /*Get std::complex output*/                                              \
      T std_out = ref.re;                                                      \
      if (!use_ref)                                                            \
        std_out = std::math_func(std_in);                                      \
                                                                               \
      /*Check cplx::complex output from device*/                               \
      Q.submit([&](sycl::handler &h) {                                         \
        sycl::accessor cplx_out{cplx_out_buf, h};                              \
        h.single_task(                                                         \
            [=]() { cplx_out[0] = experimental::math_func<T>(cplx_input); });  \
      });                                                                      \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ true);     \
                                                                               \
      /*Check cplx::complex output from host*/                                 \
      cplx_out_acc[0] = experimental::math_func<T>(cplx_input);                \
                                                                               \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ false);    \
      return pass;                                                             \
    }                                                                          \
  };

TEST_MATH_OP_TYPE(abs)
TEST_MATH_OP_TYPE(arg)
TEST_MATH_OP_TYPE(norm)
TEST_MATH_OP_TYPE(real)
TEST_MATH_OP_TYPE(imag)

#undef TEST_MATH_OP_TYPE

// Macro for testing decimal in, complex out functions

#define TEST_MATH_OP_TYPE(math_func)                                           \
  template <typename T, typename X> struct test_deci_cplx_##math_func {        \
    bool operator()(sycl::queue &Q, X init, T ref = T{},                       \
                    bool use_ref = false) {                                    \
      bool pass = true;                                                        \
                                                                               \
      auto std_in = init_deci(init);                                           \
                                                                               \
      /*Get std::complex output*/                                              \
      std::complex<T> std_out = ref;                                           \
      if (!use_ref)                                                            \
        std_out = std::math_func(std_in);                                      \
      sycl::buffer<experimental::complex<T>> cplx_out_buf{sycl::range{1}};     \
      /*Check cplx::complex output from device*/                               \
      Q.submit([&](sycl::handler &h) {                                         \
        sycl::accessor cplx_out{cplx_out_buf, h};                              \
        h.single_task(                                                         \
            [=]() { cplx_out[0] = experimental::math_func<X>(std_in); });      \
      });                                                                      \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
                                                                               \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ true);     \
                                                                               \
      /*Check cplx::complex output from host*/                                 \
      cplx_out_acc[0] = experimental::math_func<X>(std_in);                    \
                                                                               \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ false);    \
      return pass;                                                             \
    }                                                                          \
  };

TEST_MATH_OP_TYPE(conj)
TEST_MATH_OP_TYPE(proj)

#undef TEST_MATH_OP_TYPE

// Macro for testing decimal in, decimal out functions

#define TEST_MATH_OP_TYPE(math_func)                                           \
  template <typename T, typename X> struct test_deci_deci_##math_func {        \
    bool operator()(sycl::queue &Q, X init, T ref = T{},                       \
                    bool use_ref = false) {                                    \
      bool pass = true;                                                        \
                                                                               \
      auto std_in = init_deci(init);                                           \
                                                                               \
      /*Get std::complex output*/                                              \
      T std_out = ref;                                                         \
      if (!use_ref)                                                            \
        std_out = std::math_func(std_in);                                      \
      sycl::buffer<T> cplx_out_buf{sycl::range{1}};                            \
      /*Check cplx::complex output from device*/                               \
      Q.submit([&](sycl::handler &h) {                                         \
        sycl::accessor cplx_out{cplx_out_buf, h};                              \
        h.single_task(                                                         \
            [=]() { cplx_out[0] = experimental::math_func<X>(std_in); });      \
      });                                                                      \
      sycl::host_accessor cplx_out_acc{cplx_out_buf};                          \
                                                                               \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ true);     \
                                                                               \
      /*Check cplx::complex output from host*/                                 \
      cplx_out_acc[0] = experimental::math_func<X>(init);                      \
                                                                               \
      pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ false);    \
      return pass;                                                             \
    }                                                                          \
  };

TEST_MATH_OP_TYPE(arg)
TEST_MATH_OP_TYPE(norm)
TEST_MATH_OP_TYPE(real)
TEST_MATH_OP_TYPE(imag)

#undef TEST_MATH_OP_TYPE

// Test for polar function
// The real component is treated as radius rho, and the imaginary component as
// angular value theta
template <typename T> struct test_polar {
  bool operator()(sycl::queue &Q, cmplx<T> init, cmplx<T> ref = cmplx<T>(0, 0),
                  bool use_ref = false) {
    bool pass = true;

    sycl::buffer<experimental::complex<T>> cplx_out_buf{sycl::range(1)};
    /*Get std::complex output*/
    std::complex<T> std_out{ref.re, ref.im};
    if (!use_ref)
      std_out = std::polar(init.re, init.im);

    /*Check cplx::complex output from device*/
    Q.submit([&](sycl::handler &h) {
      sycl::accessor cplx_out{cplx_out_buf, h};
      h.single_task(
          [=]() { cplx_out[0] = experimental::polar<T>(init.re, init.im); });
    });
    sycl::host_accessor cplx_out_acc{cplx_out_buf};
    pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ true);

    /*Check cplx::complex output from host*/
    cplx_out_acc[0] = experimental::polar<T>(init.re, init.im);

    pass &= check_results(cplx_out_acc[0], std_out, /*is_device*/ false);

    return pass;
  }
};

int main() {
  sycl::queue Q;

  bool test_passes = true;

  /* Test complex in, complex out functions */

  {
    cplx_test_cases<test_acos> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_asin> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_atan> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_acosh> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_asinh> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_atanh> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_conj> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_cos> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_cosh> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_log> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_log10> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_proj> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_sin> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_sinh> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_sqrt> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_tan> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_tanh> test;
    test_passes &= test(Q);
  }

  /* Test complex in, decimal out functions */

  {
    cplx_test_cases<test_abs> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_arg> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_norm> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_real> test;
    test_passes &= test(Q);
  }

  {
    cplx_test_cases<test_imag> test;
    test_passes &= test(Q);
  }

  /* Test decimal in, complex out functions */

  {
    deci_test_cases<test_deci_cplx_conj> test;
    test_passes &= test(Q);
  }

  {
    deci_test_cases<test_deci_cplx_proj> test;
    test_passes &= test(Q);
  }

  /* Test decimal in, decimal out functions */

  {
    deci_test_cases<test_deci_deci_arg> test;
    test_passes &= test(Q);
  }

  {
    deci_test_cases<test_deci_deci_norm> test;
    test_passes &= test(Q);
  }

  {
    deci_test_cases<test_deci_deci_real> test;
    test_passes &= test(Q);
  }

  {
    deci_test_cases<test_deci_deci_imag> test;
    test_passes &= test(Q);
  }

  /* Test polar function */

  {
    cplx_test_cases<test_polar> test;
    test_passes &= test(Q);
  }

  return !test_passes;
}
