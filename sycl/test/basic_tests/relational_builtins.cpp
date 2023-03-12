// RUN: %clangxx -DSYCL2020_CONFORMANT_APIS -fsycl -fsyntax-only %s
// RUN: %clangxx -sycl-std=121 -fsycl -fsyntax-only %s

#include <CL/sycl.hpp>

// Some helper macros to verify return type of the builtins. To be used like
// this
//
//   CHECK(Expected return type in SYCL 1.2.1,
//         Expected return type in SYCL 2020,
//         builtin name,
//         parameters' types...)
//
// C++17 doesn't allow lambdas in unevaluated context. Could be simplified
// further in C++20 including more std::declval usage.
template <class... Args> struct CheckHelper {
  template <class F> static auto call(F f) { return f(Args()...); }
};

#if defined(SYCL2020_CONFORMANT_APIS) && SYCL_LANGUAGE_VERSION >= 202001
#define CHECK(EXPECTED121, EXPECTED2020, FUNC, ...)                            \
  {                                                                            \
    auto ret = CheckHelper<__VA_ARGS__>::call(                                 \
        [](auto... args) { return cl::sycl::FUNC(args...); });                 \
    static_assert(std::is_same_v<decltype(ret), EXPECTED2020>);                \
  }
// To be used for marray tests. Not yet implemented
// #define CHECK2020(...) CHECK(__VA_ARGS__)
#define CHECK2020(...)
#else
#define CHECK(EXPECTED121, EXPECTED2020, FUNC, ...)                            \
  {                                                                            \
    auto ret = CheckHelper<__VA_ARGS__>::call(                                 \
        [](auto... args) { return cl::sycl::FUNC(args...); });                 \
    static_assert(std::is_same_v<decltype(ret), EXPECTED121>);                 \
  }
#define CHECK2020(...)
#endif

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
  CHECK(int32_t, bool, isequal, half, half);
  CHECK(int16v, int16v, isequal, halfv, halfv);
  CHECK2020(_, boolm, isequal, halfm, halfm);

  CHECK(int32_t, bool, isequal, float, float);
  CHECK(int32v, int32v, isequal, floatv, floatv);
  CHECK2020(_, boolm, isequal, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isequal, double, double);
  CHECK(int64v, int64v, isequal, doublev, doublev);
  CHECK2020(_, boolm, isequal, doublem, doublem);

  // isnotequal
  CHECK(int32_t, bool, isnotequal, half, half);
  CHECK(int16v, int16v, isnotequal, halfv, halfv);
  CHECK2020(_, boolm, isnotequal, halfm, halfm);

  CHECK(int32_t, bool, isnotequal, float, float);
  CHECK(int32v, int32v, isnotequal, floatv, floatv);
  CHECK2020(_, boolm, isnotequal, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isnotequal, double, double);
  CHECK(int64v, int64v, isnotequal, doublev, doublev);
  CHECK2020(_, boolm, isnotequal, doublem, doublem);

  // isgreater
  CHECK(int32_t, bool, isgreater, half, half);
  CHECK(int16v, int16v, isgreater, halfv, halfv);
  CHECK2020(_, boolm, isgreater, halfm, halfm);

  CHECK(int32_t, bool, isgreater, float, float);
  CHECK(int32v, int32v, isgreater, floatv, floatv);
  CHECK2020(_, boolm, isgreater, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isgreater, double, double);
  CHECK(int64v, int64v, isgreater, doublev, doublev);
  CHECK2020(_, boolm, isgreater, doublem, doublem);

  // isgreaterequal
  CHECK(int32_t, bool, isgreaterequal, half, half);
  CHECK(int16v, int16v, isgreaterequal, halfv, halfv);
  CHECK2020(_, boolm, isgreaterequal, halfm, halfm);

  CHECK(int32_t, bool, isgreaterequal, float, float);
  CHECK(int32v, int32v, isgreaterequal, floatv, floatv);
  CHECK2020(_, boolm, isgreaterequal, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isgreaterequal, double, double);
  CHECK(int64v, int64v, isgreaterequal, doublev, doublev);
  CHECK2020(_, boolm, isgreaterequal, doublem, doublem);

  // isless
  CHECK(int32_t, bool, isless, half, half);
  CHECK(int16v, int16v, isless, halfv, halfv);
  CHECK2020(_, boolm, isless, halfm, halfm);

  CHECK(int32_t, bool, isless, float, float);
  CHECK(int32v, int32v, isless, floatv, floatv);
  CHECK2020(_, boolm, isless, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isless, double, double);
  CHECK(int64v, int64v, isless, doublev, doublev);
  CHECK2020(_, boolm, isless, doublem, doublem);

  // islessequal
  CHECK(int32_t, bool, islessequal, half, half);
  CHECK(int16v, int16v, islessequal, halfv, halfv);
  CHECK2020(_, boolm, islessequal, halfm, halfm);

  CHECK(int32_t, bool, islessequal, float, float);
  CHECK(int32v, int32v, islessequal, floatv, floatv);
  CHECK2020(_, boolm, islessequal, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, islessequal, double, double);
  CHECK(int64v, int64v, islessequal, doublev, doublev);
  CHECK2020(_, boolm, islessequal, doublem, doublem);

  // islessgreater
  CHECK(int32_t, bool, islessgreater, half, half);
  CHECK(int16v, int16v, islessgreater, halfv, halfv);
  CHECK2020(_, boolm, islessgreater, halfm, halfm);

  CHECK(int32_t, bool, islessgreater, float, float);
  CHECK(int32v, int32v, islessgreater, floatv, floatv);
  CHECK2020(_, boolm, islessgreater, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, islessgreater, double, double);
  CHECK(int64v, int64v, islessgreater, doublev, doublev);
  CHECK2020(_, boolm, islessgreater, doublem, doublem);

  // isfinite
  CHECK(int32_t, bool, isfinite, half);
  CHECK(int16v, int16v, isfinite, halfv);
  CHECK2020(_, boolm, isfinite, halfm);

  CHECK(int32_t, bool, isfinite, float);
  CHECK(int32v, int32v, isfinite, floatv);
  CHECK2020(_, boolm, isfinite, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isfinite, double);
  CHECK(int64v, int64v, isfinite, doublev);
  CHECK2020(_, boolm, isfinite, doublem);

  // isinf
  CHECK(int32_t, bool, isinf, half);
  CHECK(int16v, int16v, isinf, halfv);
  CHECK2020(_, boolm, isinf, halfm);

  CHECK(int32_t, bool, isinf, float);
  CHECK(int32v, int32v, isinf, floatv);
  CHECK2020(_, boolm, isinf, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isinf, double);
  CHECK(int64v, int64v, isinf, doublev);
  CHECK2020(_, boolm, isinf, doublem);

  // isnan
  CHECK(int32_t, bool, isnan, half);
  CHECK(int16v, int16v, isnan, halfv);
  CHECK2020(_, boolm, isnan, halfm);

  CHECK(int32_t, bool, isnan, float);
  CHECK(int32v, int32v, isnan, floatv);
  CHECK2020(_, boolm, isnan, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isnan, double);
  CHECK(int64v, int64v, isnan, doublev);
  CHECK2020(_, boolm, isnan, doublem);

  // isnormal
  CHECK(int32_t, bool, isnormal, half);
  CHECK(int16v, int16v, isnormal, halfv);
  CHECK2020(_, boolm, isnormal, halfm);

  CHECK(int32_t, bool, isnormal, float);
  CHECK(int32v, int32v, isnormal, floatv);
  CHECK2020(_, boolm, isnormal, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isnormal, double);
  CHECK(int64v, int64v, isnormal, doublev);
  CHECK2020(_, boolm, isnormal, doublem);

  // isordered
  CHECK(int32_t, bool, isordered, half, half);
  CHECK(int16v, int16v, isordered, halfv, halfv);
  CHECK2020(_, boolm, isordered, halfm, halfm);

  CHECK(int32_t, bool, isordered, float, float);
  CHECK(int32v, int32v, isordered, floatv, floatv);
  CHECK2020(_, boolm, isordered, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isordered, double, double);
  CHECK(int64v, int64v, isordered, doublev, doublev);
  CHECK2020(_, boolm, isordered, doublem, doublem);

  // isunordered
  CHECK(int32_t, bool, isunordered, half, half);
  CHECK(int16v, int16v, isunordered, halfv, halfv);
  CHECK2020(_, boolm, isunordered, halfm, halfm);

  CHECK(int32_t, bool, isunordered, float, float);
  CHECK(int32v, int32v, isunordered, floatv, floatv);
  CHECK2020(_, boolm, isunordered, floatm, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, isunordered, double, double);
  CHECK(int64v, int64v, isunordered, doublev, doublev);
  CHECK2020(_, boolm, isunordered, doublem, doublem);

  // signbit
  CHECK(int32_t, bool, signbit, half);
  CHECK(int16v, int16v, signbit, halfv);
  CHECK2020(_, boolm, signbit, halfm);

  CHECK(int32_t, bool, signbit, float);
  CHECK(int32v, int32v, signbit, floatv);
  CHECK2020(_, boolm, signbit, floatm);

  // SYCL 1.2.1 has an ABI-affecting bug here (int32_t instead of int64_t for
  // scalar case).
  CHECK(int32_t, bool, signbit, double);
  CHECK(int64v, int64v, signbit, doublev);
  CHECK2020(_, boolm, signbit, doublem);

  // any
  CHECK(int, bool, any, int16_t)
  CHECK(int, bool, any, int16v)
  CHECK2020(_, bool, any, int16m)

  CHECK(int, bool, any, int32_t)
  CHECK(int, bool, any, int32v)
  CHECK2020(_, bool, any, int32m)

  CHECK(int, bool, any, int64_t)
  CHECK(int, bool, any, int64v)
  CHECK2020(_, bool, any, int64m)

  // all
  CHECK(int, bool, all, int16_t)
  CHECK(int, bool, all, int16v)
  CHECK2020(_, bool, all, int16m)

  CHECK(int, bool, all, int32_t)
  CHECK(int, bool, all, int32v)
  CHECK2020(_, bool, all, int32m)

  CHECK(int, bool, all, int64_t)
  CHECK(int, bool, all, int64v)
  CHECK2020(_, bool, all, int64m)

  // bitselect
  CHECK(int16_t, int16_t, bitselect, int16_t, int16_t, int16_t)
  CHECK(int16v, int16v, bitselect, int16v, int16v, int16v)
  CHECK2020(int16m, int16m, bitselect, int16m, int16m, int16m)

  CHECK(uint16_t, uint16_t, bitselect, uint16_t, uint16_t, uint16_t)
  CHECK(uint16v, uint16v, bitselect, uint16v, uint16v, uint16v)
  CHECK2020(uint16m, uint16m, bitselect, uint16m, uint16m, uint16m)

  CHECK(half, half, bitselect, half, half, half)
  CHECK(halfv, halfv, bitselect, halfv, halfv, halfv)

  CHECK(int32_t, int32_t, bitselect, int32_t, int32_t, int32_t)
  CHECK(int32v, int32v, bitselect, int32v, int32v, int32v)
  CHECK2020(int32m, int32m, bitselect, int32m, int32m, int32m)

  CHECK(uint32_t, uint32_t, bitselect, uint32_t, uint32_t, uint32_t)
  CHECK(uint32v, uint32v, bitselect, uint32v, uint32v, uint32v)
  CHECK2020(uint32m, uint32m, bitselect, uint32m, uint32m, uint32m)

  CHECK(float, float, bitselect, float, float, float)
  CHECK(floatv, floatv, bitselect, floatv, floatv, floatv)
  CHECK2020(floatm, floatm, bitselect, floatm, floatm, floatm)
  CHECK2020(floatm, floatm, bitselect, floatm, floatm, floatm)

  CHECK(int64_t, int64_t, bitselect, int64_t, int64_t, int64_t)
  CHECK(int64v, int64v, bitselect, int64v, int64v, int64v)
  CHECK2020(int64m, int64m, bitselect, int64m, int64m, int64m)

  CHECK(uint64_t, uint64_t, bitselect, uint64_t, uint64_t, uint64_t)
  CHECK(uint64v, uint64v, bitselect, uint64v, uint64v, uint64v)
  CHECK2020(uint64m, uint64m, bitselect, uint64m, uint64m, uint64m)

  CHECK(double, double, bitselect, double, double, double)
  CHECK(doublev, doublev, bitselect, doublev, doublev, doublev)
  CHECK2020(doublem, doublem, bitselect, doublem, doublem, doublem)
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
