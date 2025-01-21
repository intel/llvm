// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fpreview-breaking-changes %s -o %t.out %}

// NOTE: Compile the test fully to ensure the library exports the right host
// symbols.

#include <sycl/sycl.hpp>

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

#define CHECK(EXPECTED, FUNC, ...)                                             \
  {                                                                            \
    auto ret = CheckHelper<__VA_ARGS__>::call(                                 \
        [](auto... args) { return sycl::FUNC(args...); });                     \
    static_assert(std::is_same_v<decltype(ret), EXPECTED>);                    \
  }

void foo() {
  using namespace sycl;
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

  CHECK(bool, isequal, float, float);
  CHECK(int32v, isequal, floatv, floatv);

  CHECK(bool, isequal, double, double);
  CHECK(int64v, isequal, doublev, doublev);

  // isnotequal
  CHECK(bool, isnotequal, half, half);
  CHECK(int16v, isnotequal, halfv, halfv);

  CHECK(bool, isnotequal, float, float);
  CHECK(int32v, isnotequal, floatv, floatv);

  CHECK(bool, isnotequal, double, double);
  CHECK(int64v, isnotequal, doublev, doublev);

  // isgreater
  CHECK(bool, isgreater, half, half);
  CHECK(int16v, isgreater, halfv, halfv);

  CHECK(bool, isgreater, float, float);
  CHECK(int32v, isgreater, floatv, floatv);

  CHECK(bool, isgreater, double, double);
  CHECK(int64v, isgreater, doublev, doublev);

  // isgreaterequal
  CHECK(bool, isgreaterequal, half, half);
  CHECK(int16v, isgreaterequal, halfv, halfv);

  CHECK(bool, isgreaterequal, float, float);
  CHECK(int32v, isgreaterequal, floatv, floatv);

  CHECK(bool, isgreaterequal, double, double);
  CHECK(int64v, isgreaterequal, doublev, doublev);

  // isless
  CHECK(bool, isless, half, half);
  CHECK(int16v, isless, halfv, halfv);

  CHECK(bool, isless, float, float);
  CHECK(int32v, isless, floatv, floatv);

  CHECK(bool, isless, double, double);
  CHECK(int64v, isless, doublev, doublev);

  // islessequal
  CHECK(bool, islessequal, half, half);
  CHECK(int16v, islessequal, halfv, halfv);

  CHECK(bool, islessequal, float, float);
  CHECK(int32v, islessequal, floatv, floatv);

  CHECK(bool, islessequal, double, double);
  CHECK(int64v, islessequal, doublev, doublev);

  // islessgreater
  CHECK(bool, islessgreater, half, half);
  CHECK(int16v, islessgreater, halfv, halfv);

  CHECK(bool, islessgreater, float, float);
  CHECK(int32v, islessgreater, floatv, floatv);

  CHECK(bool, islessgreater, double, double);
  CHECK(int64v, islessgreater, doublev, doublev);

  // isfinite
  CHECK(bool, isfinite, half);
  CHECK(int16v, isfinite, halfv);

  CHECK(bool, isfinite, float);
  CHECK(int32v, isfinite, floatv);

  CHECK(bool, isfinite, double);
  CHECK(int64v, isfinite, doublev);

  // isinf
  CHECK(bool, isinf, half);
  CHECK(int16v, isinf, halfv);

  CHECK(bool, isinf, float);
  CHECK(int32v, isinf, floatv);

  CHECK(bool, isinf, double);
  CHECK(int64v, isinf, doublev);

  // isnan
  CHECK(bool, isnan, half);
  CHECK(int16v, isnan, halfv);

  CHECK(bool, isnan, float);
  CHECK(int32v, isnan, floatv);

  CHECK(bool, isnan, double);
  CHECK(int64v, isnan, doublev);

  // isnormal
  CHECK(bool, isnormal, half);
  CHECK(int16v, isnormal, halfv);

  CHECK(bool, isnormal, float);
  CHECK(int32v, isnormal, floatv);

  CHECK(bool, isnormal, double);
  CHECK(int64v, isnormal, doublev);

  // isordered
  CHECK(bool, isordered, half, half);
  CHECK(int16v, isordered, halfv, halfv);

  CHECK(bool, isordered, float, float);
  CHECK(int32v, isordered, floatv, floatv);

  CHECK(bool, isordered, double, double);
  CHECK(int64v, isordered, doublev, doublev);

  // isunordered
  CHECK(bool, isunordered, half, half);
  CHECK(int16v, isunordered, halfv, halfv);

  CHECK(bool, isunordered, float, float);
  CHECK(int32v, isunordered, floatv, floatv);

  CHECK(bool, isunordered, double, double);
  CHECK(int64v, isunordered, doublev, doublev);

  // signbit
  CHECK(bool, signbit, half);
  CHECK(int16v, signbit, halfv);

  CHECK(bool, signbit, float);
  CHECK(int32v, signbit, floatv);

  CHECK(bool, signbit, double);
  CHECK(int64v, signbit, doublev);

  // any
  CHECK(bool, any, int16_t)
  CHECK(int, any, int16v)

  CHECK(bool, any, int32_t)
  CHECK(int, any, int32v)

  CHECK(bool, any, int64_t)
  CHECK(int, any, int64v)

  // all
  CHECK(bool, all, int16_t)
  CHECK(int, all, int16v)

  CHECK(bool, all, int32_t)
  CHECK(int, all, int32v)

  CHECK(bool, all, int64_t)
  CHECK(int, all, int64v)

  // bitselect
  CHECK(int16_t, bitselect, int16_t, int16_t, int16_t)
  CHECK(int16v, bitselect, int16v, int16v, int16v)

  CHECK(uint16_t, bitselect, uint16_t, uint16_t, uint16_t)
  CHECK(uint16v, bitselect, uint16v, uint16v, uint16v)

  CHECK(half, bitselect, half, half, half)
  CHECK(halfv, bitselect, halfv, halfv, halfv)

  CHECK(int32_t, bitselect, int32_t, int32_t, int32_t)
  CHECK(int32v, bitselect, int32v, int32v, int32v)

  CHECK(uint32_t, bitselect, uint32_t, uint32_t, uint32_t)
  CHECK(uint32v, bitselect, uint32v, uint32v, uint32v)

  CHECK(float, bitselect, float, float, float)
  CHECK(floatv, bitselect, floatv, floatv, floatv)

  CHECK(int64_t, bitselect, int64_t, int64_t, int64_t)
  CHECK(int64v, bitselect, int64v, int64v, int64v)

  CHECK(uint64_t, bitselect, uint64_t, uint64_t, uint64_t)
  CHECK(uint64v, bitselect, uint64v, uint64v, uint64v)

  CHECK(double, bitselect, double, double, double)
  CHECK(doublev, bitselect, doublev, doublev, doublev)
}

int main() {
  sycl::queue q;
  foo(); // Verify host.
  q.submit([&](sycl::handler &cgh) {
    cgh.single_task<class test>([]() {
      foo(); // verify device
    });
  });
}
