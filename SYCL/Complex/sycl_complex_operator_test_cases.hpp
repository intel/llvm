using std::pair;
using std::vector;

// Forward decleration of tests

template <typename T> class test_add;
template <typename T> class test_sub;
template <typename T> class test_mul;
template <typename T> class test_div;

template <typename T> class test_add_assign;
template <typename T> class test_sub_assign;
template <typename T> class test_mul_assign;
template <typename T> class test_div_assign;

// Stores test cases for each math function used in sycl_complex_math_test.cpp
// Values are stored in the highest precision type, in this case that is double

template <template <typename> typename test_struct> struct test_cases {
  static vector<pair<cmplx<double>, cmplx<double>>> test_values;
  static const char *test_name;

  bool operator()(sycl::queue &Q) {
    bool test_passes = true;

    for (auto &test_value : test_values) {
      test_passes &= test_valid_types<test_struct>(
          Q, test_value.first.re, test_value.first.im, test_value.second.re,
          test_value.second.im);
    }

    if (!test_passes)
      std::cerr << test_name << " failed\n";

    return test_passes;
  }
};

template <template <typename> typename test_struct>
vector<pair<cmplx<double>, cmplx<double>>>
    test_cases<test_struct>::test_values = {
        pair(cmplx(-1, 1), cmplx(1, 1)),  pair(cmplx(-1, 1), cmplx(-1, 1)),
        pair(cmplx(-1, 1), cmplx(1, -1)), pair(cmplx(-1, 1), cmplx(-1, -1)),

        pair(cmplx(1, 1), cmplx(-1, 1)),  pair(cmplx(-1, 1), cmplx(-1, 1)),
        pair(cmplx(1, -1), cmplx(-1, 1)), pair(cmplx(-1, -1), cmplx(-1, 1)),
};

// test_add
template <> const char *test_cases<test_add>::test_name = "addition test";

// test_sub
template <> const char *test_cases<test_sub>::test_name = "subtraction test";

// test_mul
template <> const char *test_cases<test_mul>::test_name = "multiplication test";

// test_div
template <> const char *test_cases<test_div>::test_name = "division test";

// test_add_assign
template <>
const char *test_cases<test_add_assign>::test_name = "addition assign test";

// test_sub_assign
template <>
const char *test_cases<test_sub_assign>::test_name = "subtraction assign test";

// test_mul_assign
template <>
const char *test_cases<test_mul_assign>::test_name =
    "muliplication assign test";

// test_div_assign
template <>
const char *test_cases<test_div_assign>::test_name = "division assign test";
