// REQUIRES: usm_shared_allocations
// DEFINE: %{mathflags} = %if cl_options %{/clang:-fno-fast-math%} %else %{-fno-fast-math%}

// RUN: %{build} -fsycl-device-code-split=per_kernel %{mathflags} -o %t.out
// RUN: %{run} %t.out

#include "sycl_complex_helper.hpp"

using std::get;
using std::pair;
using std::tuple;
using std::vector;

template <template <typename> typename test_struct> struct test_cases {
  static vector<pair<cmplx<double>, cmplx<double>>> std_test_values;

  static vector<tuple<cmplx<double>, cmplx<double>, cmplx<double>>>
      comp_test_values;

  static const char *test_name;

  bool operator()(sycl::queue &Q) {
    bool test_passes = true;

    for (auto &test_pair : std_test_values) {
      test_passes &=
          test_valid_types<test_struct>(Q, test_pair.first, test_pair.second);
    }

    for (auto &test_tuple : comp_test_values) {
      test_passes &= test_valid_types<test_struct>(
          Q, get<0>(test_tuple), get<1>(test_tuple), get<2>(test_tuple),
          /*use_ref*/ true);
    }

    if (!test_passes)
      std::cerr << test_name << " failed\n";

    return test_passes;
  }
};

// Tests for different pow function overloads

template <typename T> struct test_pow_cplx_cplx {
  bool operator()(sycl::queue &Q, cmplx<T> init1, cmplx<T> init2,
                  cmplx<T> ref = {0, 0}, bool use_ref = false) {
    bool pass = true;

    auto std_in1 = init_std_complex(init1.re, init1.im);
    auto std_in2 = init_std_complex(init2.re, init2.im);
    experimental::complex<T> cplx_input1{init1.re, init1.im};
    experimental::complex<T> cplx_input2{init2.re, init2.im};

    auto *cplx_out = sycl::malloc_shared<experimental::complex<T>>(1, Q);

    // Get std::complex output
    std::complex<T> std_out{ref.re, ref.im};
    if (!use_ref)
      std_out = std::pow(std_in1, std_in2);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = experimental::pow<T>(cplx_input1, cplx_input2);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = experimental::pow<T>(cplx_input1, cplx_input2);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

// Only real component of init2 is used
template <typename T> struct test_pow_cplx_deci {
  bool operator()(sycl::queue &Q, cmplx<T> init1, cmplx<T> init2,
                  cmplx<T> ref = {0, 0}, bool use_ref = false) {
    bool pass = true;

    auto std_in = init_std_complex(init1.re, init1.im);
    auto std_deci_in = init_deci(init2.re);
    experimental::complex<T> cplx_input{init1.re, init1.im};
    T deci_input = init2.re;

    auto *cplx_out = sycl::malloc_shared<experimental::complex<T>>(1, Q);

    // Get std::complex output
    std::complex<T> std_out{ref.re, ref.im};
    if (!use_ref)
      std_out = std::pow(std_in, std_deci_in);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = experimental::pow(cplx_input, deci_input);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = experimental::pow(cplx_input, deci_input);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

// Only real component of init1 is used
template <typename T> struct test_pow_deci_cplx {
  bool operator()(sycl::queue &Q, cmplx<T> init1, cmplx<T> init2,
                  cmplx<T> ref = {0, 0}, bool use_ref = false) {
    bool pass = true;

    auto std_in = init_std_complex(init2.re, init2.im);
    auto std_deci_in = init_deci(init1.re);
    experimental::complex<T> cplx_input{init2.re, init2.im};
    T deci_input = init1.re;

    auto *cplx_out = sycl::malloc_shared<experimental::complex<T>>(1, Q);

    // Get std::complex output
    std::complex<T> std_out{ref.re, ref.im};
    if (!use_ref)
      std_out = std::pow(std_deci_in, std_in);

    // Check cplx::complex output from device
    Q.single_task([=]() {
       cplx_out[0] = experimental::pow(deci_input, cplx_input);
     }).wait();

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ true);

    // Check cplx::complex output from host
    cplx_out[0] = experimental::pow(deci_input, cplx_input);

    pass &= check_results(cplx_out[0], std_out, /*is_device*/ false);

    sycl::free(cplx_out, Q);

    return pass;
  }
};

// test_pow_cplx_cplx
template <>
vector<pair<cmplx<double>, cmplx<double>>>
    test_cases<test_pow_cplx_cplx>::std_test_values = {
        pair(cmplx(4.5, 2.0), cmplx(3.0, 1.5)),
        pair(cmplx(9.0, 0.0), cmplx(0.5, 0.0)),
        pair(cmplx(1.0, 1.0), cmplx(-1.0, -1.0)),
        pair(cmplx(1.0, 1.0), cmplx(0.5, 0.5)),
        pair(cmplx(-1.0, -1.0), cmplx(-1.0, -1.0)),
};

template <>
vector<tuple<cmplx<double>, cmplx<double>, cmplx<double>>> test_cases<
    test_pow_cplx_cplx>::comp_test_values = {
    tuple(cmplx(INFINITYd, 2.02), cmplx(4.42, 2.02), cmplx(INFINITYd, NANd)),
    tuple(cmplx(4.42, INFINITYd), cmplx(4.42, 2.02), cmplx(INFINITYd, NANd)),
    tuple(cmplx(INFINITYd, INFINITYd), cmplx(4.42, 2.02),
          cmplx(INFINITYd, NANd)),
    tuple(cmplx(NANd, 2.02), cmplx(4.42, 2.02), cmplx(NANd, NANd)),
    tuple(cmplx(4.42, NANd), cmplx(4.42, 2.02), cmplx(NANd, NANd)),
    tuple(cmplx(NANd, NANd), cmplx(4.42, 2.02), cmplx(NANd, NANd)),
    tuple(cmplx(NANd, INFINITYd), cmplx(4.42, 2.02), cmplx(INFINITYd, NANd)),
    tuple(cmplx(INFINITYd, NANd), cmplx(4.42, 2.02), cmplx(INFINITYd, NANd)),
    tuple(cmplx(NANd, INFINITYd), cmplx(4.42, 2.02), cmplx(INFINITYd, NANd)),
    tuple(cmplx(INFINITYd, NANd), cmplx(4.42, 2.02), cmplx(INFINITYd, NANd)),
};

template <>
const char *test_cases<test_pow_cplx_cplx>::test_name =
    "pow complex complex test";

// test_pow_cplx_deci
template <>
vector<pair<cmplx<double>, cmplx<double>>>
    test_cases<test_pow_cplx_deci>::std_test_values = {
        pair(cmplx(4.5, 2.0), cmplx(3.0, 0.0)),
        pair(cmplx(9.0, 0.0), cmplx(0.5, 0.0)),
        pair(cmplx(1.0, 1.0), cmplx(-1.0, 0.0)),
        pair(cmplx(1.0, 1.0), cmplx(0.5, 0.0)),
        pair(cmplx(-1.0, -1.0), cmplx(-1.0, -0.0)),
};

template <>
vector<tuple<cmplx<double>, cmplx<double>, cmplx<double>>>
    test_cases<test_pow_cplx_deci>::comp_test_values = {
        tuple(cmplx(INFINITYd, 2.02), cmplx(4.42, 0.0), cmplx(INFINITYd, NANd)),
        tuple(cmplx(4.42, INFINITYd), cmplx(4.42, 0.0), cmplx(INFINITYd, NANd)),
        tuple(cmplx(INFINITYd, INFINITYd), cmplx(4.42, 0.0),
              cmplx(INFINITYd, NANd)),
        tuple(cmplx(NANd, 2.02), cmplx(4.42, 0.0), cmplx(NANd, NANd)),
        tuple(cmplx(4.42, NANd), cmplx(4.42, 0.0), cmplx(NANd, NANd)),
        tuple(cmplx(NANd, NANd), cmplx(4.42, 0.0), cmplx(NANd, NANd)),
        tuple(cmplx(NANd, INFINITYd), cmplx(4.42, 0.0), cmplx(INFINITYd, NANd)),
        tuple(cmplx(INFINITYd, NANd), cmplx(4.42, 0.0), cmplx(INFINITYd, NANd)),
        tuple(cmplx(NANd, INFINITYd), cmplx(4.42, 0.0), cmplx(INFINITYd, NANd)),
        tuple(cmplx(INFINITYd, NANd), cmplx(4.42, 0.0), cmplx(INFINITYd, NANd)),
};

template <>
const char *test_cases<test_pow_cplx_deci>::test_name =
    "pow complex decimal test";

// test_pow_deci_cplx
template <>
vector<pair<cmplx<double>, cmplx<double>>>
    test_cases<test_pow_deci_cplx>::std_test_values = {
        pair(cmplx(4.5, 0.0), cmplx(3.0, 1.5)),
        pair(cmplx(9.0, 0.0), cmplx(0.5, 0.0)),
        pair(cmplx(1.0, 0.0), cmplx(-1.0, -1.0)),
        pair(cmplx(1.0, 0.0), cmplx(0.5, 0.5)),
        pair(cmplx(-1.0, 0.0), cmplx(-1.0, -1.0)),
};

template <>
vector<tuple<cmplx<double>, cmplx<double>, cmplx<double>>> test_cases<
    test_pow_deci_cplx>::comp_test_values = {
    tuple(cmplx(4.42, 0.0), cmplx(INFINITYd, 2.02), cmplx(INFINITYd, -NANd)),
    tuple(cmplx(4.42, 0.0), cmplx(4.42, INFINITYd), cmplx(NANd, -NANd)),
    tuple(cmplx(4.42, 0.0), cmplx(INFINITYd, INFINITYd),
          cmplx(INFINITYd, NANd)),
    tuple(cmplx(4.42, 0.0), cmplx(NANd, 2.02), cmplx(NANd, NANd)),
    tuple(cmplx(4.42, 0.0), cmplx(4.42, NANd), cmplx(NANd, NANd)),
    tuple(cmplx(4.42, 0.0), cmplx(NANd, NANd), cmplx(NANd, NANd)),
    tuple(cmplx(4.42, 0.0), cmplx(NANd, INFINITYd), cmplx(-NANd, -NANd)),
    tuple(cmplx(4.42, 0.0), cmplx(INFINITYd, NANd), cmplx(INFINITYd, -NANd)),
    tuple(cmplx(4.42, 0.0), cmplx(NANd, INFINITYd), cmplx(-NANd, -NANd)),
    tuple(cmplx(4.42, 0.0), cmplx(INFINITYd, NANd), cmplx(INFINITYd, -NANd)),
};

template <>
const char *test_cases<test_pow_deci_cplx>::test_name =
    "pow decimal complex test";

int main() {
  sycl::queue Q;

  bool test_passes = true;

  {
    test_cases<test_pow_cplx_cplx> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_pow_cplx_deci> test;
    test_passes &= test(Q);
  }

  {
    test_cases<test_pow_deci_cplx> test;
    test_passes &= test(Q);
  }

  if (!test_passes)
    std::cerr << "pow test failed\n";

  return !test_passes;
}
