using std::pair;
using std::tuple;
using std::vector;

// Forward decleration of tests

template <typename T> class test_acos;
template <typename T> class test_asin;
template <typename T> class test_atan;
template <typename T> class test_acosh;
template <typename T> class test_asinh;
template <typename T> class test_atanh;
template <typename T> class test_conj;
template <typename T> class test_cos;
template <typename T> class test_cosh;
template <typename T> class test_exp;
template <typename T> class test_log;
template <typename T> class test_log10;
template <typename T> class test_proj;
template <typename T> class test_sin;
template <typename T> class test_sinh;
template <typename T> class test_sqrt;
template <typename T> class test_tan;
template <typename T> class test_tanh;

template <typename T> class test_abs;
template <typename T> class test_arg;
template <typename T> class test_norm;

template <typename T> class test_polar;

// Stores test cases for each math function used in sycl_complex_math_test.cpp
// Values are stored in the highest precision type, in this case that is double

template <template <typename> typename test_struct> struct test_cases {
  // Default test cases
  static vector<cmplx<double>> std_test_values;
  static vector<tuple<cmplx<double>, cmplx<double>>> comp_test_values;

  static const char *test_name;

  bool operator()(sycl::queue &Q) {
    bool test_passes = true;

    for (auto &test_value : std_test_values) {
      test_passes &= test_valid_types<test_struct>(Q, test_value);
    }

    for (auto &test_tuple : comp_test_values) {
      test_passes &= test_valid_types<test_struct>(Q, get<0>(test_tuple),
                                                   get<1>(test_tuple),
                                                   /*use_ref*/ true);
    }

    if (!test_passes)
      std::cerr << test_name << " failed\n";

    return test_passes;
  }
};

template <template <typename> typename test_struct>
vector<cmplx<double>> test_cases<test_struct>::std_test_values = {
    cmplx(1, 1),
    cmplx(-1, 1),
    cmplx(1, -1),
    cmplx(-1, -1),
    cmplx(INFINITYd, 2.02),
    cmplx(4.42, INFINITYd),
    cmplx(INFINITYd, INFINITYd),
    cmplx(NANd, 2.02),
    cmplx(4.42, NANd),
    cmplx(NANd, NANd),
    cmplx(NANd, INFINITYd),
    cmplx(INFINITYd, NANd),
};

template <template <typename> typename test_struct>
vector<tuple<cmplx<double>, cmplx<double>>>
    test_cases<test_struct>::comp_test_values = {};

// test_acos
template <> const char *test_cases<test_acos>::test_name = "acos test";

// test_asin
template <> const char *test_cases<test_asin>::test_name = "asin test";

// test_atan
template <> const char *test_cases<test_atan>::test_name = "atan test";

template <>
vector<cmplx<double>> test_cases<test_atan>::std_test_values = {
    cmplx(1, 1),
    cmplx(-1, 1),
    cmplx(1, -1),
    cmplx(-1, -1),
    cmplx(INFINITYd, 2.02),
    cmplx(4.42, INFINITYd),
    cmplx(INFINITYd, INFINITYd),
    cmplx(NANd, 2.02),
    cmplx(4.42, NANd),
    cmplx(NANd, NANd),
    cmplx(NANd, INFINITYd),
};

// MSVC complex handles some error codes differently
// So check against reference value
template <>
vector<tuple<cmplx<double>, cmplx<double>>>
    test_cases<test_atan>::comp_test_values = {
        tuple(cmplx(INFINITYd, NANd), cmplx(PI / 2.0, 0.0))};

// test_acosh
template <> const char *test_cases<test_acosh>::test_name = "acosh test";

// test_asinh
template <> const char *test_cases<test_asinh>::test_name = "asinh test";

// test_atanh
template <> const char *test_cases<test_atanh>::test_name = "atanh test";

template <>
vector<cmplx<double>> test_cases<test_atanh>::std_test_values = {
    cmplx(1, 1),
    cmplx(-1, 1),
    cmplx(1, -1),
    cmplx(-1, -1),
    cmplx(INFINITYd, 2.02),
    cmplx(4.42, INFINITYd),
    cmplx(INFINITYd, INFINITYd),
    cmplx(NANd, 2.02),
    cmplx(4.42, NANd),
    cmplx(NANd, NANd),
    cmplx(INFINITYd, NANd),
};

// MSVC complex handles some error codes differently
// So check against reference value
template <>
vector<tuple<cmplx<double>, cmplx<double>>>
    test_cases<test_atanh>::comp_test_values = {
        tuple(cmplx(NANd, INFINITYd), cmplx(0.0, PI / 2.0))};

// test_conj
template <> const char *test_cases<test_conj>::test_name = "conj test";

// test_cos
template <> const char *test_cases<test_cos>::test_name = "cos test";

// test_cosh
template <> const char *test_cases<test_cosh>::test_name = "cosh test";

// test_exp
template <> const char *test_cases<test_exp>::test_name = "exp test";

// test_log
template <> const char *test_cases<test_log>::test_name = "log test";

// test_log10
template <> const char *test_cases<test_log10>::test_name = "log10 test";

// test_proj
template <> const char *test_cases<test_proj>::test_name = "proj test";

// test_sin
template <> const char *test_cases<test_sin>::test_name = "sin test";

// test_sinh
template <> const char *test_cases<test_sinh>::test_name = "sinh test";

// test_sqrt
template <> const char *test_cases<test_sqrt>::test_name = "sqrt test";

// test_tan
template <> const char *test_cases<test_tan>::test_name = "tan test";

template <>
vector<cmplx<double>> test_cases<test_tan>::std_test_values = {
    cmplx(1, 1),
    cmplx(-1, 1),
    cmplx(1, -1),
    cmplx(-1, -1),
    cmplx(INFINITYd, 2.02),
    cmplx(4.42, INFINITYd),
    cmplx(NANd, 2.02),
    cmplx(4.42, NANd),
    cmplx(NANd, NANd),
    cmplx(INFINITYd, NANd),
};

// MSVC complex handles some error codes differently
// So check against reference value
template <>
vector<tuple<cmplx<double>, cmplx<double>>>
    test_cases<test_tan>::comp_test_values = {
        tuple(cmplx(INFINITYd, INFINITYd), cmplx(0.0, 1.0)),
        tuple(cmplx(NANd, INFINITYd), cmplx(0.0, 1.0))};

// test_tanh
template <> const char *test_cases<test_tanh>::test_name = "tanh test";

template <>
vector<cmplx<double>> test_cases<test_tanh>::std_test_values = {
    cmplx(1, 1),
    cmplx(-1, 1),
    cmplx(1, -1),
    cmplx(-1, -1),
    cmplx(INFINITYd, 2.02),
    cmplx(4.42, INFINITYd),
    cmplx(NANd, 2.02),
    cmplx(4.42, NANd),
    cmplx(NANd, NANd),
    cmplx(NANd, INFINITYd),
};

// MSVC complex handles some error codes differently
// So check against reference value
template <>
vector<tuple<cmplx<double>, cmplx<double>>>
    test_cases<test_tanh>::comp_test_values = {
        tuple(cmplx(INFINITYd, INFINITYd), cmplx(1.0, 0.0)),
        tuple(cmplx(INFINITYd, NANd), cmplx(1.0, 0.0))};

// test_abs
template <> const char *test_cases<test_abs>::test_name = "abs test";

// test_arg
template <> const char *test_cases<test_arg>::test_name = "arg test";

// test_norm
// Difference between libstdc++ and libc++ when NaN's and Inf values are
// combined.
template <>
vector<cmplx<double>> test_cases<test_norm>::std_test_values = {
    cmplx(1, 1),
    cmplx(-1, 1),
    cmplx(1, -1),
    cmplx(-1, -1),
    cmplx(INFINITYd, 2.02),
    cmplx(4.42, INFINITYd),
    cmplx(INFINITYd, INFINITYd),
    cmplx(NANd, 2.02),
    cmplx(4.42, NANd),
    cmplx(NANd, NANd),
};

template <> const char *test_cases<test_norm>::test_name = "norm test";

// test_polar
// Note: values represent rho and theta, not real and imaginary values
// Output is undefined if rho is negative or Nan, or theta is Inf
template <>
vector<cmplx<double>> test_cases<test_polar>::std_test_values = {
    cmplx(1, 1),
    cmplx(1, -1),
    cmplx(2, 0),
    cmplx(0.5, 3.14),
};

template <> const char *test_cases<test_polar>::test_name = "polar test";
