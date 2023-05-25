// RUN: %clangxx -fsycl -fsyntax-only %s

#include <CL/sycl.hpp>

// Some helper macros to verify return type of the builtins. To be used like
// this
//
//   CHECK(Expected return type,
//         builtin name,
//         parameters' types...)
//
// C++17 doesn't allow lambdas in unevaluated context. Could be simplified
// further in C++20 including more std::declval usage.
template <class... Args> struct CheckHelper {
  template <class F> static auto call(F f) { return f(Args()...); }
};

#define CHECK(EXPECTED, FUNC, ...)                            \
  {                                                                            \
    auto ret = CheckHelper<__VA_ARGS__>::call(                                 \
        [](auto... args) { return cl::sycl::FUNC(args...); });                 \
    static_assert(std::is_same_v<decltype(ret), EXPECTED>);                \
  }
// To be used for marray tests. Not yet implemented
// #define CHECK_MARRAY(...) CHECK(__VA_ARGS__)
#define CHECK_MARRAY(...)

void foo() {
  using namespace cl::sycl;
  using boolm = marray<bool, 2>;

  using int16v = vec<int16_t, 2>;
  using int16m = marray<int16_t, 2>;

  using uint16v = vec<uint16_t, 2>;
  using uint16m = marray<uint16_t, 2>;

  using halfv = vec<half, 2>;
  using halfm = marray<half, 2>;

  using int32v = vec<int32_t, 2>;
  using int32m = marray<int32_t, 2>;

  using uint32v = vec<uint32_t, 2>;
  using uint32m = marray<uint32_t, 2>;

  using floatv = vec<float, 2>;
  using floatm = marray<float, 2>;

  using int64v = vec<int64_t, 2>;
  using int64m = marray<int64_t, 2>;

  using uint64v = vec<uint64_t, 2>;
  using uint64m = marray<uint64_t, 2>;

  using doublev = vec<double, 2>;
  using doublem = marray<double, 2>;

  // isequal
  CHECK(bool, isequal, half, half);
  CHECK(int16v, isequal, halfv, halfv);
  CHECK_MARRAY(boolm, isequal, halfm, halfm);

  CHECK(bool, isequal, float, float);
  CHECK(int32v, isequal, floatv, floatv);
  CHECK_MARRAY(boolm, isequal, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isequal, double, double);
  CHECK(int64v, isequal, doublev, doublev);
  CHECK_MARRAY(boolm, isequal, doublem, doublem);

  // isnotequal
  CHECK(bool, isnotequal, half, half);
  CHECK(int16v, isnotequal, halfv, halfv);
  CHECK_MARRAY(boolm, isnotequal, halfm, halfm);

  CHECK(bool, isnotequal, float, float);
  CHECK(int32v, isnotequal, floatv, floatv);
  CHECK_MARRAY(boolm, isnotequal, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isnotequal, double, double);
  CHECK(int64v, isnotequal, doublev, doublev);
  CHECK_MARRAY(boolm, isnotequal, doublem, doublem);

  // isgreater
  CHECK(bool, isgreater, half, half);
  CHECK(int16v, isgreater, halfv, halfv);
  CHECK_MARRAY(boolm, isgreater, halfm, halfm);

  CHECK(bool, isgreater, float, float);
  CHECK(int32v, isgreater, floatv, floatv);
  CHECK_MARRAY(boolm, isgreater, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isgreater, double, double);
  CHECK(int64v, isgreater, doublev, doublev);
  CHECK_MARRAY(boolm, isgreater, doublem, doublem);

  // isgreaterequal
  CHECK(bool, isgreaterequal, half, half);
  CHECK(int16v, isgreaterequal, halfv, halfv);
  CHECK_MARRAY(boolm, isgreaterequal, halfm, halfm);

  CHECK(bool, isgreaterequal, float, float);
  CHECK(int32v, isgreaterequal, floatv, floatv);
  CHECK_MARRAY(boolm, isgreaterequal, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isgreaterequal, double, double);
  CHECK(int64v, isgreaterequal, doublev, doublev);
  CHECK_MARRAY(boolm, isgreaterequal, doublem, doublem);

  // isless
  CHECK(bool, isless, half, half);
  CHECK(int16v, isless, halfv, halfv);
  CHECK_MARRAY(boolm, isless, halfm, halfm);

  CHECK(bool, isless, float, float);
  CHECK(int32v, isless, floatv, floatv);
  CHECK_MARRAY(boolm, isless, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isless, double, double);
  CHECK(int64v, isless, doublev, doublev);
  CHECK_MARRAY(boolm, isless, doublem, doublem);

  // islessequal
  CHECK(bool, islessequal, half, half);
  CHECK(int16v, islessequal, halfv, halfv);
  CHECK_MARRAY(boolm, islessequal, halfm, halfm);

  CHECK(bool, islessequal, float, float);
  CHECK(int32v, islessequal, floatv, floatv);
  CHECK_MARRAY(boolm, islessequal, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, islessequal, double, double);
  CHECK(int64v, islessequal, doublev, doublev);
  CHECK_MARRAY(boolm, islessequal, doublem, doublem);

  // islessgreater
  CHECK(bool, islessgreater, half, half);
  CHECK(int16v, islessgreater, halfv, halfv);
  CHECK_MARRAY(boolm, islessgreater, halfm, halfm);

  CHECK(bool, islessgreater, float, float);
  CHECK(int32v, islessgreater, floatv, floatv);
  CHECK_MARRAY(boolm, islessgreater, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, islessgreater, double, double);
  CHECK(int64v, islessgreater, doublev, doublev);
  CHECK_MARRAY(boolm, islessgreater, doublem, doublem);

  // isfinite
  CHECK(bool, isfinite, half);
  CHECK(int16v, isfinite, halfv);
  CHECK_MARRAY(boolm, isfinite, halfm);

  CHECK(bool, isfinite, float);
  CHECK(int32v, isfinite, floatv);
  CHECK_MARRAY(boolm, isfinite, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isfinite, double);
  CHECK(int64v, isfinite, doublev);
  CHECK_MARRAY(boolm, isfinite, doublem);

  // isinf
  CHECK(bool, isinf, half);
  CHECK(int16v, isinf, halfv);
  CHECK_MARRAY(boolm, isinf, halfm);

  CHECK(bool, isinf, float);
  CHECK(int32v, isinf, floatv);
  CHECK_MARRAY(boolm, isinf, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isinf, double);
  CHECK(int64v, isinf, doublev);
  CHECK_MARRAY(boolm, isinf, doublem);

  // isnan
  CHECK(bool, isnan, half);
  CHECK(int16v, isnan, halfv);
  CHECK_MARRAY(boolm, isnan, halfm);

  CHECK(bool, isnan, float);
  CHECK(int32v, isnan, floatv);
  CHECK_MARRAY(boolm, isnan, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isnan, double);
  CHECK(int64v, isnan, doublev);
  CHECK_MARRAY(boolm, isnan, doublem);

  // isnormal
  CHECK(bool, isnormal, half);
  CHECK(int16v, isnormal, halfv);
  CHECK_MARRAY(boolm, isnormal, halfm);

  CHECK(bool, isnormal, float);
  CHECK(int32v, isnormal, floatv);
  CHECK_MARRAY(boolm, isnormal, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isnormal, double);
  CHECK(int64v, isnormal, doublev);
  CHECK_MARRAY(boolm, isnormal, doublem);

  // isordered
  CHECK(bool, isordered, half, half);
  CHECK(int16v, isordered, halfv, halfv);
  CHECK_MARRAY(boolm, isordered, halfm, halfm);

  CHECK(bool, isordered, float, float);
  CHECK(int32v, isordered, floatv, floatv);
  CHECK_MARRAY(boolm, isordered, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isordered, double, double);
  CHECK(int64v, isordered, doublev, doublev);
  CHECK_MARRAY(boolm, isordered, doublem, doublem);

  // isunordered
  CHECK(bool, isunordered, half, half);
  CHECK(int16v, isunordered, halfv, halfv);
  CHECK_MARRAY(boolm, isunordered, halfm, halfm);

  CHECK(bool, isunordered, float, float);
  CHECK(int32v, isunordered, floatv, floatv);
  CHECK_MARRAY(boolm, isunordered, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, isunordered, double, double);
  CHECK(int64v, isunordered, doublev, doublev);
  CHECK_MARRAY(boolm, isunordered, doublem, doublem);

  // signbit
  CHECK(bool, signbit, half);
  CHECK(int16v, signbit, halfv);
  CHECK_MARRAY(boolm, signbit, halfm);

  CHECK(bool, signbit, float);
  CHECK(int32v, signbit, floatv);
  CHECK_MARRAY(boolm, signbit, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(bool, signbit, double);
  CHECK(int64v, signbit, doublev);
  CHECK_MARRAY(boolm, signbit, doublem);

  // any
  CHECK(bool, any, int16_t)
  CHECK(int, any, int16v)
  CHECK_MARRAY(bool, any, int16m)

  CHECK(bool, any, int32_t)
  CHECK(int, any, int32v)
  CHECK_MARRAY(bool, any, int32m)

  CHECK(bool, any, int64_t)
  CHECK(int, any, int64v)
  CHECK_MARRAY(bool, any, int64m)

  // all
  CHECK(bool, all, int16_t)
  CHECK(int, all, int16v)
  CHECK_MARRAY(bool, all, int16m)

  CHECK(bool, all, int32_t)
  CHECK(int, all, int32v)
  CHECK_MARRAY(bool, all, int32m)

  CHECK(bool, all, int64_t)
  CHECK(int, all, int64v)
  CHECK_MARRAY(bool, all, int64m)

  // bitselect
  CHECK(int16_t, bitselect, int16_t, int16_t, int16_t)
  CHECK(int16v, bitselect, int16v, int16v, int16v)
  CHECK_MARRAY(int16m, bitselect, int16m, int16m, int16m)

  CHECK(uint16_t, bitselect, uint16_t, uint16_t, uint16_t)
  CHECK(uint16v, bitselect, uint16v, uint16v, uint16v)
  CHECK_MARRAY(uint16m, bitselect, uint16m, uint16m, uint16m)

  CHECK(half, bitselect, half, half, half)
  CHECK(halfv, bitselect, halfv, halfv, halfv)

  CHECK(int32_t, bitselect, int32_t, int32_t, int32_t)
  CHECK(int32v, bitselect, int32v, int32v, int32v)
  CHECK_MARRAY(int32m, bitselect, int32m, int32m, int32m)

  CHECK(uint32_t, bitselect, uint32_t, uint32_t, uint32_t)
  CHECK(uint32v, bitselect, uint32v, uint32v, uint32v)
  CHECK_MARRAY(uint32m, bitselect, uint32m, uint32m, uint32m)

  CHECK(float, bitselect, float, float, float)
  CHECK(floatv, bitselect, floatv, floatv, floatv)
  CHECK_MARRAY(floatm, bitselect, floatm, floatm, floatm)
  CHECK_MARRAY(floatm, bitselect, floatm, floatm, floatm)

  CHECK(int64_t, bitselect, int64_t, int64_t, int64_t)
  CHECK(int64v, bitselect, int64v, int64v, int64v)
  CHECK_MARRAY(int64m, bitselect, int64m, int64m, int64m)

  CHECK(uint64_t, bitselect, uint64_t, uint64_t, uint64_t)
  CHECK(uint64v, bitselect, uint64v, uint64v, uint64v)
  CHECK_MARRAY(uint64m, bitselect, uint64m, uint64m, uint64m)

  CHECK(double, bitselect, double, double, double)
  CHECK(doublev, bitselect, doublev, doublev, doublev)
  CHECK_MARRAY(doublem, bitselect, doublem, doublem, doublem)
}

int main() {
  cl::sycl::queue q;
  foo(); // Verify host.
  q.submit([&](cl::sycl::handler &cgh) {
    cgh.single_task<class test>([]() {
      foo(); // verify device
    });
  });
}
