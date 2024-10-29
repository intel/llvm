// This test checks edge cases handling for std::exp(std::complex<T>) used
// in SYCL kernels.

#include <sycl/detail/core.hpp>

#include <cmath>
#include <complex>

bool check(bool cond, const std::string &cond_str, int line,
           unsigned testcase) {
  if (!cond) {
    std::cout << "Assertion " << cond_str << " (line " << line
              << ") failed for testcase #" << testcase << std::endl;
    return false;
  }

  return true;
}

template <typename T> bool test() {
  // To simplify maintanence of those comments specifying indexes of test cases
  // in the array below, please add new test cases at the end of the list.
  constexpr std::complex<T> testcases[] = {
      /* 0 */ std::complex<T>(1.e-6, 1.e-6),
      /* 1 */ std::complex<T>(-1.e-6, 1.e-6),
      /* 2 */ std::complex<T>(-1.e-6, -1.e-6),
      /* 3 */ std::complex<T>(1.e-6, -1.e-6),

      /* 4 */ std::complex<T>(1.e+6, 1.e-6),
      /* 5 */ std::complex<T>(-1.e+6, 1.e-6),
      /* 6 */ std::complex<T>(-1.e+6, -1.e-6),
      /* 7 */ std::complex<T>(1.e+6, -1.e-6),

      /* 8 */ std::complex<T>(1.e-6, 1.e+6),
      /* 9 */ std::complex<T>(-1.e-6, 1.e+6),
      /* 10 */ std::complex<T>(-1.e-6, -1.e+6),
      /* 11 */ std::complex<T>(1.e-6, -1.e+6),

      /* 12 */ std::complex<T>(1.e+6, 1.e+6),
      /* 13 */ std::complex<T>(-1.e+6, 1.e+6),
      /* 14 */ std::complex<T>(-1.e+6, -1.e+6),
      /* 15 */ std::complex<T>(1.e+6, -1.e+6),

      /* 16 */ std::complex<T>(-0, -1.e-6),
      /* 17 */ std::complex<T>(-0, 1.e-6),
      /* 18 */ std::complex<T>(-0, 1.e+6),
      /* 19 */ std::complex<T>(-0, -1.e+6),
      /* 20 */ std::complex<T>(0, -1.e-6),
      /* 21 */ std::complex<T>(0, 1.e-6),
      /* 22 */ std::complex<T>(0, 1.e+6),
      /* 23 */ std::complex<T>(0, -1.e+6),

      /* 24 */ std::complex<T>(-1.e-6, -0),
      /* 25 */ std::complex<T>(1.e-6, -0),
      /* 26 */ std::complex<T>(1.e+6, -0),
      /* 27 */ std::complex<T>(-1.e+6, -0),
      /* 28 */ std::complex<T>(-1.e-6, 0),
      /* 29 */ std::complex<T>(1.e-6, 0),
      /* 30 */ std::complex<T>(1.e+6, 0),
      /* 31 */ std::complex<T>(-1.e+6, 0),

      /* 32 */ std::complex<T>(NAN, NAN),
      /* 33 */ std::complex<T>(-INFINITY, NAN),
      /* 34 */ std::complex<T>(-2, NAN),
      /* 35 */ std::complex<T>(-1, NAN),
      /* 36 */ std::complex<T>(-0.5, NAN),
      /* 37 */ std::complex<T>(-0., NAN),
      /* 38 */ std::complex<T>(+0., NAN),
      /* 39 */ std::complex<T>(0.5, NAN),
      /* 40 */ std::complex<T>(1, NAN),
      /* 41 */ std::complex<T>(2, NAN),
      /* 42 */ std::complex<T>(INFINITY, NAN),

      /* 43 */ std::complex<T>(NAN, -INFINITY),
      /* 44 */ std::complex<T>(-INFINITY, -INFINITY),
      /* 45 */ std::complex<T>(-2, -INFINITY),
      /* 46 */ std::complex<T>(-1, -INFINITY),
      /* 47 */ std::complex<T>(-0.5, -INFINITY),
      /* 48 */ std::complex<T>(-0., -INFINITY),
      /* 49 */ std::complex<T>(+0., -INFINITY),
      /* 50 */ std::complex<T>(0.5, -INFINITY),
      /* 51 */ std::complex<T>(1, -INFINITY),
      /* 52 */ std::complex<T>(2, -INFINITY),
      /* 53 */ std::complex<T>(INFINITY, -INFINITY),

      /* 54 */ std::complex<T>(NAN, -2),
      /* 55 */ std::complex<T>(-INFINITY, -2),
      /* 56 */ std::complex<T>(-2, -2),
      /* 57 */ std::complex<T>(-1, -2),
      /* 58 */ std::complex<T>(-0.5, -2),
      /* 59 */ std::complex<T>(-0., -2),
      /* 60 */ std::complex<T>(+0., -2),
      /* 61 */ std::complex<T>(0.5, -2),
      /* 62 */ std::complex<T>(1, -2),
      /* 63 */ std::complex<T>(2, -2),
      /* 64 */ std::complex<T>(INFINITY, -2),

      /* 65 */ std::complex<T>(NAN, -1),
      /* 66 */ std::complex<T>(-INFINITY, -1),
      /* 67 */ std::complex<T>(-2, -1),
      /* 68 */ std::complex<T>(-1, -1),
      /* 69 */ std::complex<T>(-0.5, -1),
      /* 70 */ std::complex<T>(-0., -1),
      /* 71 */ std::complex<T>(+0., -1),
      /* 72 */ std::complex<T>(0.5, -1),
      /* 73 */ std::complex<T>(1, -1),
      /* 74 */ std::complex<T>(2, -1),
      /* 75 */ std::complex<T>(INFINITY, -1),

      /* 76 */ std::complex<T>(NAN, -0.5),
      /* 77 */ std::complex<T>(-INFINITY, -0.5),
      /* 78 */ std::complex<T>(-2, -0.5),
      /* 79 */ std::complex<T>(-1, -0.5),
      /* 80 */ std::complex<T>(-0.5, -0.5),
      /* 81 */ std::complex<T>(-0., -0.5),
      /* 82 */ std::complex<T>(+0., -0.5),
      /* 83 */ std::complex<T>(0.5, -0.5),
      /* 84 */ std::complex<T>(1, -0.5),
      /* 85 */ std::complex<T>(2, -0.5),
      /* 86 */ std::complex<T>(INFINITY, -0.5),

      /* 87 */ std::complex<T>(NAN, -0.),
      /* 88 */ std::complex<T>(-INFINITY, -0.),
      /* 89 */ std::complex<T>(-2, -0.),
      /* 90 */ std::complex<T>(-1, -0.),
      /* 91 */ std::complex<T>(-0.5, -0.),
      /* 92 */ std::complex<T>(-0., -0.),
      /* 93 */ std::complex<T>(+0., -0.),
      /* 94 */ std::complex<T>(0.5, -0.),
      /* 95 */ std::complex<T>(1, -0.),
      /* 96 */ std::complex<T>(2, -0.),
      /* 97 */ std::complex<T>(INFINITY, -0.),

      /* 98 */ std::complex<T>(NAN, +0.),
      /* 99 */ std::complex<T>(-INFINITY, +0.),
      /* 100 */ std::complex<T>(-2, +0.),
      /* 101 */ std::complex<T>(-1, +0.),
      /* 102 */ std::complex<T>(-0.5, +0.),
      /* 103 */ std::complex<T>(-0., +0.),
      /* 104 */ std::complex<T>(+0., +0.),
      /* 105 */ std::complex<T>(0.5, +0.),
      /* 106 */ std::complex<T>(1, +0.),
      /* 107 */ std::complex<T>(2, +0.),
      /* 108 */ std::complex<T>(INFINITY, +0.),

      /* 109 */ std::complex<T>(NAN, 0.5),
      /* 110 */ std::complex<T>(-INFINITY, 0.5),
      /* 111 */ std::complex<T>(-2, 0.5),
      /* 112 */ std::complex<T>(-1, 0.5),
      /* 113 */ std::complex<T>(-0.5, 0.5),
      /* 114 */ std::complex<T>(-0., 0.5),
      /* 115 */ std::complex<T>(+0., 0.5),
      /* 116 */ std::complex<T>(0.5, 0.5),
      /* 117 */ std::complex<T>(1, 0.5),
      /* 118 */ std::complex<T>(2, 0.5),
      /* 119 */ std::complex<T>(INFINITY, 0.5),

      /* 120 */ std::complex<T>(NAN, 1),
      /* 121 */ std::complex<T>(-INFINITY, 1),
      /* 122 */ std::complex<T>(-2, 1),
      /* 123 */ std::complex<T>(-1, 1),
      /* 124 */ std::complex<T>(-0.5, 1),
      /* 125 */ std::complex<T>(-0., 1),
      /* 126 */ std::complex<T>(+0., 1),
      /* 127 */ std::complex<T>(0.5, 1),
      /* 128 */ std::complex<T>(1, 1),
      /* 129 */ std::complex<T>(2, 1),
      /* 130 */ std::complex<T>(INFINITY, 1),

      /* 131 */ std::complex<T>(NAN, 2),
      /* 132 */ std::complex<T>(-INFINITY, 2),
      /* 133 */ std::complex<T>(-2, 2),
      /* 134 */ std::complex<T>(-1, 2),
      /* 135 */ std::complex<T>(-0.5, 2),
      /* 136 */ std::complex<T>(-0., 2),
      /* 137 */ std::complex<T>(+0., 2),
      /* 138 */ std::complex<T>(0.5, 2),
      /* 139 */ std::complex<T>(1, 2),
      /* 140 */ std::complex<T>(2, 2),
      /* 141 */ std::complex<T>(INFINITY, 2),

      /* 142 */ std::complex<T>(NAN, INFINITY),
      /* 143 */ std::complex<T>(-INFINITY, INFINITY),
      /* 144 */ std::complex<T>(-2, INFINITY),
      /* 145 */ std::complex<T>(-1, INFINITY),
      /* 146 */ std::complex<T>(-0.5, INFINITY),
      /* 147 */ std::complex<T>(-0., INFINITY),
      /* 148 */ std::complex<T>(+0., INFINITY),
      /* 149 */ std::complex<T>(0.5, INFINITY),
      /* 150 */ std::complex<T>(1, INFINITY),
      /* 151 */ std::complex<T>(2, INFINITY),
      /* 152 */ std::complex<T>(INFINITY, INFINITY)};

  try {
    sycl::queue q;

    constexpr unsigned N = sizeof(testcases) / sizeof(testcases[0]);

    sycl::buffer<std::complex<T>> data(testcases, sycl::range{N});
    sycl::buffer<std::complex<T>> results(sycl::range{N});

    q.submit([&](sycl::handler &cgh) {
       sycl::accessor acc_data(data, cgh, sycl::read_only);
       sycl::accessor acc(results, cgh, sycl::write_only);
       cgh.parallel_for(sycl::range{N}, [=](sycl::item<1> it) {
         acc[it] = std::exp(acc_data[it]);
       });
     }).wait_and_throw();

    bool passed = true;

    // Note: this macro is expected to be used within a loop
#define CHECK(cond, pass_marker, ...)                                          \
  if (!check((cond), #cond, __LINE__, __VA_ARGS__)) {                          \
    pass_marker = false;                                                       \
    continue;                                                                  \
  }

    // Based on https://en.cppreference.com/w/cpp/numeric/complex/exp
    // z below refers to the argument passed to std::exp(complex<T>)
    sycl::host_accessor acc(results);
    for (unsigned i = 0; i < N; ++i) {
      std::complex<T> r = acc[i];
      // If z is (+/-0, +0), the result is (1, +0)
      if (testcases[i].real() == 0 && testcases[i].imag() == 0 &&
          !std::signbit(testcases[i].imag())) {
        CHECK(r.real() == 1.0, passed, i);
        CHECK(r.imag() == 0, passed, i);
        CHECK(std::signbit(testcases[i].imag()) == std::signbit(r.imag()),
              passed, i);
        // If z is (x, +inf) (for any finite x), the result is (NaN, NaN)
      } else if (std::isfinite(testcases[i].real()) &&
                 std::isinf(testcases[i].imag())) {
        CHECK(std::isnan(r.real()), passed, i);
        CHECK(std::isnan(r.imag()), passed, i);
        // If z is (x, NaN) (for any finite x), the result is (NaN, NaN)
      } else if (std::isfinite(testcases[i].real()) &&
                 std::isnan(testcases[i].imag())) {
        CHECK(std::isnan(r.real()), passed, i);
        CHECK(std::isnan(r.imag()), passed, i);
        // If z is (+inf, +0), the result is (+inf, +0)
      } else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 &&
                 testcases[i].imag() == 0) {
        CHECK(std::isinf(r.real()), passed, i);
        CHECK(r.real() > 0, passed, i);
        CHECK(r.imag() == 0, passed, i);
        CHECK(std::signbit(testcases[i].imag()) == std::signbit(r.imag()),
              passed, i);
        // If z is (-inf, +inf), the result is (+/-0, +/-0) (signs are
        // unspecified)
      } else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 &&
                 std::isinf(testcases[i].imag())) {
        CHECK(r.real() == 0, passed, i);
        CHECK(r.imag() == 0, passed, i);
        // If z is (+inf, +inf), the result is (+/-inf, NaN), (the sign of the
        // real part is unspecified)
      } else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 &&
                 std::isinf(testcases[i].imag())) {
        CHECK(std::isinf(r.real()), passed, i);
        CHECK(std::isnan(r.imag()), passed, i);
        // If z is (-inf, NaN), the result is (+/-0, +/-0) (signs are
        // unspecified)
      } else if (std::isinf(testcases[i].real()) && testcases[i].real() < 0 &&
                 std::isnan(testcases[i].imag())) {
        CHECK(r.real() == 0, passed, i);
        CHECK(r.imag() == 0, passed, i);
        // If z is (+inf, NaN), the result is (+/-inf, NaN) (the sign of the
        // real part is unspecified)
      } else if (std::isinf(testcases[i].real()) && testcases[i].real() > 0 &&
                 std::isnan(testcases[i].imag())) {
        CHECK(std::isinf(r.real()), passed, i);
        CHECK(std::isnan(r.imag()), passed, i);
        // If z is (NaN, +0), the result is (NaN, +0)
      } else if (std::isnan(testcases[i].real()) && testcases[i].imag() == 0) {
        CHECK(std::isnan(r.real()), passed, i);
        CHECK(r.imag() == 0, passed, i);
        CHECK(std::signbit(testcases[i].imag()) == std::signbit(r.imag()),
              passed, i);
        // If z is (NaN, y) (for any nonzero y), the result is (NaN,NaN)
      } else if (std::isnan(testcases[i].real()) && testcases[i].imag() != 0) {
        CHECK(std::isnan(r.real()), passed, i);
        CHECK(std::isnan(r.imag()), passed, i);
        // If z is (NaN, NaN), the result is (NaN, NaN)
      } else if (std::isnan(testcases[i].real()) &&
                 std::isnan(testcases[i].imag())) {
        CHECK(std::isnan(r.real()), passed, i);
        CHECK(std::isnan(r.imag()), passed, i);
        // Those tests were taken from oneDPL, not sure what is the corner case
        // they are covering here
      } else if (std::isfinite(testcases[i].imag()) &&
                 std::abs(testcases[i].imag()) <= 1) {
        CHECK(!std::signbit(r.real()), passed, i);
        CHECK(std::signbit(r.imag()) == std::signbit(testcases[i].imag()),
              passed, i);
        // Those tests were taken from oneDPL, not sure what is the corner case
        // they are covering here
      } else if (std::isinf(r.real()) && testcases[i].imag() == 0) {
        CHECK(r.imag() == 0, passed, i);
        CHECK(std::signbit(testcases[i].imag()) == std::signbit(r.imag()),
              passed, i);
      }
      // FIXME: do we have the following cases covered?
      // If z is (-inf, y) (for any finite y), the result is +0 cis(y)
      // If z is (+inf, y) (for any finite nonzero y), the result is +inf cis(y)
    }

    return passed ? 0 : 1;
  } catch (sycl::exception &e) {
    std::cout << "Caught sync sycl exception: " << e.what() << std::endl;
    return 2;
  }
}
