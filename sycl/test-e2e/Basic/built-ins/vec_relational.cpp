// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#define TEST(FUNC, TYPE, EXPECTED, N, ...)                                     \
  {                                                                            \
    {                                                                          \
      TYPE result[N];                                                          \
      {                                                                        \
        sycl::buffer<TYPE> b(result, sycl::range{N});                          \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::vec<TYPE, N> res = FUNC(__VA_ARGS__);                        \
            for (int i = 0; i < N; i++)                                        \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < N; i++)                                              \
        assert(EXPECTED[i] ? bool(result[i]) : !bool(result[i]));              \
    }                                                                          \
  }

#define TEST2(FUNC, TYPE, EXPECTED, N, ...)                                    \
  {                                                                            \
    {                                                                          \
      TYPE result[1];                                                          \
      {                                                                        \
        sycl::buffer<TYPE> b(result, sycl::range{1});                          \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            int res = FUNC(__VA_ARGS__);                                       \
            for (int i = 0; i < N; i++)                                        \
              res_access[0] = res;                                             \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      assert(EXPECTED[0] ? bool(result[0]) : !bool(result[0]));                \
    }                                                                          \
  }

#define EXPECTED(TYPE, ...) ((TYPE[]){__VA_ARGS__})

int main() {
  sycl::queue deviceQueue;

  sycl::vec<float, 2> va1{1.0, 2.0};
  sycl::vec<float, 2> va2{1.0, 2.0};
  sycl::vec<float, 2> va3{2.0, 1.0};
  sycl::vec<float, 2> va4{2.0, 2.0};
  sycl::vec<float, 3> va5{2.0, 2.0, 1.0};
  sycl::vec<float, 3> va6{1.0, 5.0, 8.0};
  sycl::vec<int32_t, 3> va7{50, 2, 31};
  sycl::vec<float, 2> va8{1.0, 1.0};
  sycl::vec<float, 2> va9{0.5, 0.5};
  sycl::vec<float, 2> va10{2.0, 2.0};
  sycl::vec<float, 3> va11{1.0, 2.0, 3.0};
  sycl::vec<float, 3> va12{2.0, 1.0, 3.0};
  sycl::vec<float, 3> va13{2.0, 2.0, 2.0};
  sycl::vec<int32_t, 4> va14{50, 2, 31, 18};
  sycl::vec<float, 2> va15{1.0, 5.0};
  sycl::vec<float, 3> va16{1.0, 1.0, 1.0f};
  sycl::vec<float, 3> va17{0.5, 0.5, 1.0f};
  sycl::vec<float, 3> va18{2.0, 2.0, 1.0f};
  sycl::vec<int32_t, 3> c1(1, 0, 1);
  sycl::vec<int32_t, 2> c2(1, 0);

  TEST(sycl::isequal, int32_t, EXPECTED(int32_t, 1, 1), 2, va1, va2);
  TEST(sycl::isnotequal, int32_t, EXPECTED(int32_t, 0, 0), 2, va1, va2);
  TEST(sycl::isgreater, int32_t, EXPECTED(int32_t, 0, 1), 2, va1, va3);
  TEST(sycl::isgreaterequal, int32_t, EXPECTED(int32_t, 0, 1), 2, va1, va4);
  TEST(sycl::isless, int32_t, EXPECTED(int32_t, 0, 1), 2, va3, va1);
  TEST(sycl::islessequal, int32_t, EXPECTED(int32_t, 0, 1), 2, va4, va1);
  TEST(sycl::islessgreater, int32_t, EXPECTED(int32_t, 0, 0), 2, va1, va2);
  TEST(sycl::isfinite, int32_t, EXPECTED(int32_t, 1, 1), 2, va1);
  TEST(sycl::isinf, int32_t, EXPECTED(int32_t, 0, 0), 2, va1);
  TEST(sycl::isnan, int32_t, EXPECTED(int32_t, 0, 0), 2, va1);
  TEST(sycl::isnormal, int32_t, EXPECTED(int32_t, 1, 1), 2, va1);
  TEST(sycl::isordered, int32_t, EXPECTED(int32_t, 1, 1), 2, va1, va2);
  TEST(sycl::isunordered, int32_t, EXPECTED(int32_t, 0, 0), 2, va1, va2);
  TEST(sycl::signbit, int32_t, EXPECTED(int32_t, 0, 0), 2, va1);
  TEST2(sycl::all, int, EXPECTED(int32_t, 0), 3, va7);
  TEST2(sycl::any, int, EXPECTED(int32_t, 0), 3, va7);
  TEST(sycl::bitselect, float, EXPECTED(float, 1.0, 1.0), 2, va8, va9, va10);
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0, 8.0), 3, va5, va6, c1);
  {
    // Extra tests for select/bitselect due to special handling required for
    // integer return types.

    auto Test = [&](auto F, auto Expected, auto... Args) {
      std::tuple ArgsTuple{Args...};
      auto Result = std::apply(F, ArgsTuple);
      static_assert(std::is_same_v<decltype(Expected), decltype(Result)>);

      // Note: operator==(vec, vec) return vec.
      auto Equal = [](auto x, auto y) {
        for (size_t i = 0; i < x.size(); ++i)
          if (x[i] != y[i])
            return false;

        return true;
      };

      assert(Equal(Result, Expected));

      sycl::buffer<bool, 1> ResultBuf{1};
      deviceQueue.submit([&](sycl::handler &cgh) {
        sycl::accessor Result{ResultBuf, cgh};
        cgh.single_task([=]() {
          auto R = std::apply(F, ArgsTuple);
          static_assert(std::is_same_v<decltype(Expected), decltype(R)>);
          Result[0] = Equal(R, Expected);
        });
      });
      assert(sycl::host_accessor{ResultBuf}[0]);
    };

    // Note that only int8_t/uint8_t are supported by the bitselect/select
    // builtins and not all three char data types. Also, use positive numbers
    // for the values below so that we could use the same for both
    // signed/unsigned tests.
    sycl::vec<uint8_t, 2> a{0b1100, 0b0011};
    sycl::vec<uint8_t, 2> b{0b0011, 0b1100};
    sycl::vec<uint8_t, 2> c{0b1010, 0b1010};
    sycl::vec<uint8_t, 2> r{0b0110, 0b1001};

    auto BitSelect = [](auto... xs) { return sycl::bitselect(xs...); };
    Test(BitSelect, r, a, b, c);
    [&](auto... xs) {
      Test(BitSelect, xs.template as<sycl::vec<int8_t, 2>>()...);
    }(r, a, b, c);

    auto Select = [](auto... xs) { return sycl::select(xs...); };
    sycl::vec<uint8_t, 2> c2{0x7F, 0xFF};
    sycl::vec<uint8_t, 2> r2{a[0], b[1]};

    Test(Select, r2, a, b, c2);
    [&](auto... xs) {
      Test(Select, xs.template as<sycl::vec<int8_t, 2>>()..., c2);
    }(r2, a, b);

    // Assume that MSB of a signed data type is the leftmost bit (signbit).
    auto c3 = c2.template as<sycl::vec<int8_t, 2>>();

    Test(Select, r2, a, b, c3);
    [&](auto... xs) {
      Test(Select, xs.template as<sycl::vec<int8_t, 2>>()..., c3);
    }(r2, a, b);
  }

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  TEST(sycl::isequal, int32_t, EXPECTED(int32_t, 1, 1), 2, va11.swizzle<0, 1>(),
       va2);
  TEST(sycl::isequal, int32_t, EXPECTED(int32_t, 1, 1), 2, va1,
       va11.swizzle<0, 1>());
  TEST(sycl::isequal, int32_t, EXPECTED(int32_t, 1, 1), 2, va11.swizzle<0, 1>(),
       va11.swizzle<0, 1>());
  TEST(sycl::isnotequal, int32_t, EXPECTED(int32_t, 0, 0), 2,
       va11.swizzle<0, 1>(), va2);
  TEST(sycl::isnotequal, int32_t, EXPECTED(int32_t, 0, 0), 2, va1,
       va11.swizzle<0, 1>());
  TEST(sycl::isnotequal, int32_t, EXPECTED(int32_t, 0, 0), 2,
       va11.swizzle<0, 1>(), va11.swizzle<0, 1>());
  TEST(sycl::isgreater, int32_t, EXPECTED(int32_t, 0, 1), 2,
       va11.swizzle<0, 1>(), va3);
  TEST(sycl::isgreater, int32_t, EXPECTED(int32_t, 0, 1), 2, va1,
       va12.swizzle<0, 1>());
  TEST(sycl::isgreater, int32_t, EXPECTED(int32_t, 0, 1), 2,
       va11.swizzle<0, 1>(), va12.swizzle<0, 1>());
  TEST(sycl::isgreaterequal, int32_t, EXPECTED(int32_t, 0, 1), 2,
       va11.swizzle<0, 1>(), va4);
  TEST(sycl::isgreaterequal, int32_t, EXPECTED(int32_t, 0, 1), 2, va1,
       va13.swizzle<0, 1>());
  TEST(sycl::isgreaterequal, int32_t, EXPECTED(int32_t, 0, 1), 2,
       va11.swizzle<0, 1>(), va13.swizzle<0, 1>());
  TEST(sycl::isless, int32_t, EXPECTED(int32_t, 0, 1), 2, va3,
       va11.swizzle<0, 1>());
  TEST(sycl::isless, int32_t, EXPECTED(int32_t, 0, 1), 2, va12.swizzle<0, 1>(),
       va1);
  TEST(sycl::isless, int32_t, EXPECTED(int32_t, 0, 1), 2, va12.swizzle<0, 1>(),
       va11.swizzle<0, 1>());
  TEST(sycl::islessequal, int32_t, EXPECTED(int32_t, 0, 1), 2,
       va13.swizzle<0, 1>(), va1);
  TEST(sycl::islessequal, int32_t, EXPECTED(int32_t, 0, 1), 2, va4,
       va11.swizzle<0, 1>());
  TEST(sycl::islessequal, int32_t, EXPECTED(int32_t, 0, 1), 2,
       va13.swizzle<0, 1>(), va11.swizzle<0, 1>());
  TEST(sycl::islessgreater, int32_t, EXPECTED(int32_t, 0, 0), 2,
       va11.swizzle<0, 1>(), va2);
  TEST(sycl::islessgreater, int32_t, EXPECTED(int32_t, 0, 0), 2, va1,
       va11.swizzle<0, 1>());
  TEST(sycl::islessgreater, int32_t, EXPECTED(int32_t, 0, 0), 2,
       va11.swizzle<0, 1>(), va11.swizzle<0, 1>());
  TEST(sycl::isfinite, int32_t, EXPECTED(int32_t, 1, 1), 2,
       va11.swizzle<0, 1>());
  TEST(sycl::isinf, int32_t, EXPECTED(int32_t, 0, 0), 2, va11.swizzle<0, 1>());
  TEST(sycl::isnan, int32_t, EXPECTED(int32_t, 0, 0), 2, va11.swizzle<0, 1>());
  TEST(sycl::isnormal, int32_t, EXPECTED(int32_t, 1, 1), 2,
       va11.swizzle<0, 1>());
  TEST(sycl::isordered, int32_t, EXPECTED(int32_t, 1, 1), 2,
       va11.swizzle<0, 1>(), va2);
  TEST(sycl::isordered, int32_t, EXPECTED(int32_t, 1, 1), 2, va1,
       va11.swizzle<0, 1>());
  TEST(sycl::isordered, int32_t, EXPECTED(int32_t, 1, 1), 2,
       va11.swizzle<0, 1>(), va11.swizzle<0, 1>());
  TEST(sycl::isunordered, int32_t, EXPECTED(int32_t, 0, 0), 2,
       va11.swizzle<0, 1>(), va2);
  TEST(sycl::isunordered, int32_t, EXPECTED(int32_t, 0, 0), 2, va1,
       va11.swizzle<0, 1>());
  TEST(sycl::isunordered, int32_t, EXPECTED(int32_t, 0, 0), 2,
       va11.swizzle<0, 1>(), va11.swizzle<0, 1>());
  TEST(sycl::signbit, int32_t, EXPECTED(int32_t, 0, 0), 2,
       va11.swizzle<0, 1>());
  TEST2(sycl::all, int, EXPECTED(int32_t, 0), 3, va14.swizzle<0, 1, 2>());
  TEST2(sycl::any, int, EXPECTED(int32_t, 0), 3, va14.swizzle<0, 1, 2>());
  TEST(sycl::bitselect, float, EXPECTED(float, 1.0, 1.0), 2,
       va16.swizzle<0, 1>(), va9, va10);
  TEST(sycl::bitselect, float, EXPECTED(float, 1.0, 1.0), 2, va8,
       va17.swizzle<0, 1>(), va10);
  TEST(sycl::bitselect, float, EXPECTED(float, 1.0, 1.0), 2, va8, va9,
       va18.swizzle<0, 1>());
  TEST(sycl::bitselect, float, EXPECTED(float, 1.0, 1.0), 2,
       va16.swizzle<0, 1>(), va17.swizzle<0, 1>(), va10);
  TEST(sycl::bitselect, float, EXPECTED(float, 1.0, 1.0), 2,
       va16.swizzle<0, 1>(), va9, va18.swizzle<0, 1>());
  TEST(sycl::bitselect, float, EXPECTED(float, 1.0, 1.0), 2, va8,
       va17.swizzle<0, 1>(), va18.swizzle<0, 1>());
  TEST(sycl::bitselect, float, EXPECTED(float, 1.0, 1.0), 2,
       va16.swizzle<0, 1>(), va17.swizzle<0, 1>(), va18.swizzle<0, 1>());
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0), 2, va5.swizzle<0, 1>(),
       va15, c2);
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0), 2, va4,
       va6.swizzle<0, 1>(), c2);
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0), 2, va4, va15,
       c1.swizzle<0, 1>());
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0), 2, va5.swizzle<0, 1>(),
       va6.swizzle<0, 1>(), c2);
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0), 2, va5.swizzle<0, 1>(),
       va15, c1.swizzle<0, 1>());
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0), 2, va4,
       va6.swizzle<0, 1>(), c1.swizzle<0, 1>());
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0), 2, va5.swizzle<0, 1>(),
       va6.swizzle<0, 1>(), c1.swizzle<0, 1>());
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0), 2, va5.swizzle<0, 1>(),
       va6.swizzle<0, 1>(), c1.swizzle<0, 1>());
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

  return 0;
}
