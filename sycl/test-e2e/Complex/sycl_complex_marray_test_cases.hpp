#include <iostream>
#include <tuple>
#include <vector>

#include "sycl_complex_helper.hpp"

#define DEFAULT_TEST_CASE_SIZE 47
#define POLAR_TEST_CASE_SIZE 9
#define GETTERS_TEST_CASE_SIZE 8

////////////////////////////////////////////////////////////////////////////////
/// FORWARD DECLARATION OF TESTS
////////////////////////////////////////////////////////////////////////////////

// Maths
template <typename T> struct test_abs;
template <typename T> struct test_arg;
template <typename T> struct test_norm;

template <typename T> struct test_acos;
template <typename T> struct test_asin;
template <typename T> struct test_atan;
template <typename T> struct test_acosh;
template <typename T> struct test_asinh;
template <typename T> struct test_atanh;
template <typename T> struct test_conj;
template <typename T> struct test_cos;
template <typename T> struct test_cosh;
template <typename T> struct test_exp;
template <typename T> struct test_log;
template <typename T> struct test_log10;
template <typename T> struct test_proj;
template <typename T> struct test_sin;
template <typename T> struct test_sinh;
template <typename T> struct test_sqrt;
template <typename T> struct test_tan;
template <typename T> struct test_tanh;

template <typename T> struct test_polar;

template <typename T> struct test_pow_cplx_cplx;
template <typename T> struct test_pow_cplx_deci;
template <typename T> struct test_pow_deci_cplx;

// Operators
template <typename T> struct test_add;
template <typename T> struct test_sub;
template <typename T> struct test_mul;
template <typename T> struct test_div;

template <typename T> struct test_assign_add;
template <typename T> struct test_assign_sub;
template <typename T> struct test_assign_mul;
template <typename T> struct test_assign_div;

template <typename T> struct test_unary_add;
template <typename T> struct test_unary_sub;

// Getters
template <typename T> struct test_real;
template <typename T> struct test_imag;

////////////////////////////////////////////////////////////////////////////////
/// TEST CASES COMPLEX
////////////////////////////////////////////////////////////////////////////////

// Driver
template <template <typename> typename test_struct>
struct marray_cplx_test_cases {
  static std::vector<cmplx<double>> std_test_values;
  static std::tuple<std::vector<cmplx<double>>, std::vector<cmplx<double>>>
      comp_test_values;

  static const char *test_name;

  bool operator()(sycl::queue &Q) {
    bool test_passes = true;

    if (!std_test_values.empty()) {
      test_passes &= test_valid_types<test_struct>(Q, std_test_values);
    }

    const std::vector<cmplx<double>> init = std::get<0>(comp_test_values);
    const std::vector<cmplx<double>> ref = std::get<1>(comp_test_values);

    if (!init.empty() && !ref.empty()) {
      test_passes &=
          test_valid_types<test_struct>(Q, init, ref, /*use_ref*/ true);
    }

    if (!test_passes)
      std::cerr << test_name << " failed\n";

    return test_passes;
  }
};

// Default values
template <template <typename> typename test_struct>
std::vector<cmplx<double>>
    marray_cplx_test_cases<test_struct>::std_test_values = {
        cmplx<double>(0, 1),           cmplx<double>(0, -1),
        cmplx<double>(0, 0.5),         cmplx<double>(0, -0.5),
        cmplx<double>(0, 0),           cmplx<double>(0, INFINITYd),
        cmplx<double>(0, NANd),

        cmplx<double>(1, 1),           cmplx<double>(1, -1),
        cmplx<double>(1, 0.5),         cmplx<double>(1, -0.5),
        cmplx<double>(1, 0),           cmplx<double>(1, INFINITYd),
        cmplx<double>(1, NANd),

        cmplx<double>(-1, 1),          cmplx<double>(-1, -1),
        cmplx<double>(-1, 0.5),        cmplx<double>(-1, -0.5),
        cmplx<double>(-1, 0),          cmplx<double>(-1, INFINITYd),
        cmplx<double>(-1, NANd),

        cmplx<double>(0.5, 1),         cmplx<double>(0.5, -1),
        cmplx<double>(0.5, 0.5),       cmplx<double>(0.5, -0.5),
        cmplx<double>(0.5, 0),         cmplx<double>(0.5, INFINITYd),
        cmplx<double>(0.5, NANd),

        cmplx<double>(-0.5, 1),        cmplx<double>(-0.5, -1),
        cmplx<double>(-0.5, 0.5),      cmplx<double>(-0.5, -0.5),
        cmplx<double>(-0.5, 0),        cmplx<double>(-0.5, INFINITYd),
        cmplx<double>(-0.5, NANd),

        cmplx<double>(INFINITYd, 1),   cmplx<double>(INFINITYd, -1),
        cmplx<double>(INFINITYd, 0.5), cmplx<double>(INFINITYd, -0.5),
        cmplx<double>(INFINITYd, 0),   cmplx<double>(INFINITYd, INFINITYd),

        cmplx<double>(NANd, 1),        cmplx<double>(NANd, -1),
        cmplx<double>(NANd, 0.5),      cmplx<double>(NANd, -0.5),
        cmplx<double>(NANd, 0),        cmplx<double>(NANd, NANd),
};
template <template <typename> typename test_struct>
std::tuple<std::vector<cmplx<double>>, std::vector<cmplx<double>>>
    marray_cplx_test_cases<test_struct>::comp_test_values({}, {});

/// Maths

// test_abs
template <>
const char *marray_cplx_test_cases<test_abs>::test_name = "abs test";
// test_arg
template <>
const char *marray_cplx_test_cases<test_arg>::test_name = "arg test";
// test_norm
template <>
const char *marray_cplx_test_cases<test_norm>::test_name = "norm test";

// test_acos
template <>
const char *marray_cplx_test_cases<test_acos>::test_name = "acos test";
// test_asin
template <>
const char *marray_cplx_test_cases<test_asin>::test_name = "asin test";
// test_atan
template <>
const char *marray_cplx_test_cases<test_atan>::test_name = "atan test";
// test_acosh
template <>
const char *marray_cplx_test_cases<test_acosh>::test_name = "acos test";
// test_asinh
template <>
const char *marray_cplx_test_cases<test_asinh>::test_name = "asinh test";
// test_atanh
template <>
const char *marray_cplx_test_cases<test_atanh>::test_name = "atanh test";
// test_conj
template <>
const char *marray_cplx_test_cases<test_conj>::test_name = "conj test";
// test_cos
template <>
const char *marray_cplx_test_cases<test_cos>::test_name = "cos test";
// test_cosh
template <>
const char *marray_cplx_test_cases<test_cosh>::test_name = "cosh test";
// test_exp
template <>
const char *marray_cplx_test_cases<test_exp>::test_name = "exp test";
// test_log
template <>
const char *marray_cplx_test_cases<test_log>::test_name = "log test";
// test_log10
template <>
const char *marray_cplx_test_cases<test_log10>::test_name = "log10 test";
// test_proj
template <>
const char *marray_cplx_test_cases<test_proj>::test_name = "proj test";
// test_sin
template <>
const char *marray_cplx_test_cases<test_sin>::test_name = "sin test";
// test_sinh
template <>
const char *marray_cplx_test_cases<test_sinh>::test_name = "sinh test";
// test_sqrt
template <>
const char *marray_cplx_test_cases<test_sqrt>::test_name = "sqrt test";
// test_tan
template <>
const char *marray_cplx_test_cases<test_tan>::test_name = "tan test";
// test_tanh
template <>
const char *marray_cplx_test_cases<test_tanh>::test_name = "tanh test";

// test_polar
template <>
std::vector<cmplx<double>> marray_cplx_test_cases<test_polar>::std_test_values =
    {
        cmplx<double>(0, 1),   cmplx<double>(0, 0.5),   cmplx<double>(0, 0),

        cmplx<double>(1, 1),   cmplx<double>(1, 0.5),   cmplx<double>(1, 0),

        cmplx<double>(0.5, 1), cmplx<double>(0.5, 0.5), cmplx<double>(0.5, 0),
};
template <>
const char *marray_cplx_test_cases<test_polar>::test_name = "polar test";

// test_pow_cplx_cplx
template <>
const char *marray_cplx_test_cases<test_pow_cplx_cplx>::test_name =
    "pow cplx cplx test";
// test_pow_cplx_deci
template <>
const char *marray_cplx_test_cases<test_pow_cplx_deci>::test_name =
    "pow cplx deci test";
// test_pow_deci_cplx
template <>
const char *marray_cplx_test_cases<test_pow_deci_cplx>::test_name =
    "pow deci cplx test";

/// Operators

// test_add
template <>
const char *marray_cplx_test_cases<test_add>::test_name = "add test";
// test_sub
template <>
const char *marray_cplx_test_cases<test_sub>::test_name = "sub test";
// test_mul
template <>
const char *marray_cplx_test_cases<test_mul>::test_name = "mul test";
// test_div
template <>
const char *marray_cplx_test_cases<test_div>::test_name = "div test";

// test_assign_add
template <>
const char *marray_cplx_test_cases<test_assign_add>::test_name =
    "assign add test";
// test_assign_sub
template <>
const char *marray_cplx_test_cases<test_assign_sub>::test_name =
    "assign sub test";
// test_assign_mul
template <>
const char *marray_cplx_test_cases<test_assign_mul>::test_name =
    "assign mul test";
// test_assign_div
template <>
const char *marray_cplx_test_cases<test_assign_div>::test_name =
    "assign div test";

// test_unary_add
template <>
const char *marray_cplx_test_cases<test_unary_add>::test_name =
    "unary add test";
// test_unary_sub
template <>
const char *marray_cplx_test_cases<test_unary_sub>::test_name =
    "unary sub test";

////////////////////////////////////////////////////////////////////////////////
/// TEST CASES SCALAR
////////////////////////////////////////////////////////////////////////////////

// Driver
template <template <typename> typename test_struct>
struct marray_scalar_test_cases {
  static std::vector<double> std_test_values;
  static std::tuple<std::vector<double>, std::vector<double>> comp_test_values;

  static const char *test_name;

  bool operator()(sycl::queue &Q) {
    bool test_passes = true;

    if (!std_test_values.empty()) {
      test_passes &= test_valid_types<test_struct>(Q, std_test_values);
    }

    const std::vector<double> init = std::get<0>(comp_test_values);
    const std::vector<double> ref = std::get<1>(comp_test_values);

    if (!init.empty() && !ref.empty()) {
      test_passes &=
          test_valid_types<test_struct>(Q, init, ref, /*use_ref*/ true);
    }

    if (!test_passes)
      std::cerr << test_name << " failed\n";

    return test_passes;
  }
};

// Default values
template <template <typename> typename test_struct>
std::vector<double> marray_scalar_test_cases<test_struct>::std_test_values = {
    0, 0.5, 1, -0, -0.5, -1, INFINITYd, NANd,
};
template <template <typename> typename test_struct>
std::tuple<std::vector<double>, std::vector<double>>
    marray_scalar_test_cases<test_struct>::comp_test_values({}, {});

// test_real
template <>
const char *marray_scalar_test_cases<test_real>::test_name = "real test";
// test_imag
template <>
const char *marray_scalar_test_cases<test_imag>::test_name = "imag test";
