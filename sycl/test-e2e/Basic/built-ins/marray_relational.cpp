// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out%}

#include <CL/sycl.hpp>

#define TEST(FUNC, TYPE, EXPECTED, N, ...)                                     \
  {                                                                            \
    {                                                                          \
      TYPE result[N];                                                          \
      {                                                                        \
        sycl::buffer<TYPE> b(result, sycl::range{N});                          \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::marray<TYPE, N> res = FUNC(__VA_ARGS__);                     \
            for (int i = 0; i < N; i++)                                        \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < N; i++)                                              \
        assert(result[i] == EXPECTED[i]);                                      \
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
      assert(result[0] == EXPECTED[0]);                                        \
    }                                                                          \
  }

#define EXPECTED(TYPE, ...) ((TYPE[]){__VA_ARGS__})

int main() {
  sycl::device Dev;
  sycl::queue deviceQueue(Dev);

  sycl::marray<sycl::half, 2> ma1_f16{1.f, 2.f};
  sycl::marray<sycl::half, 2> ma2_f16{1.f, 2.f};
  sycl::marray<sycl::half, 2> ma3_f16{2.f, 1.f};
  sycl::marray<sycl::half, 2> ma4_f16{2.f, 2.f};
  sycl::marray<sycl::half, 3> ma5_f16{2.f, 2.f, 1.f};
  sycl::marray<sycl::half, 3> ma6_f16{1.f, 5.f, 8.f};
  sycl::marray<sycl::half, 2> ma8_f16{1.f, 1.f};
  sycl::marray<sycl::half, 2> ma9_f16{0.5f, 0.5f};
  sycl::marray<sycl::half, 2> ma10_f16{2.f, 2.f};

  sycl::marray<float, 2> ma1{1.f, 2.f};
  sycl::marray<float, 2> ma2{1.f, 2.f};
  sycl::marray<float, 2> ma3{2.f, 1.f};
  sycl::marray<float, 2> ma4{2.f, 2.f};
  sycl::marray<float, 3> ma5{2.f, 2.f, 1.f};
  sycl::marray<float, 3> ma6{1.f, 5.f, 8.f};
  sycl::marray<int, 3> ma7{50, 2, 31};
  sycl::marray<float, 2> ma8{1.f, 1.f};
  sycl::marray<float, 2> ma9{0.5f, 0.5f};
  sycl::marray<float, 2> ma10{2.f, 2.f};
  sycl::marray<bool, 3> c(1, 0, 1);

  if (Dev.has(sycl::aspect::fp16)) {
    TEST(sycl::isequal, bool, EXPECTED(bool, 1, 1), 2, ma1_f16, ma2_f16);
    TEST(sycl::isnotequal, bool, EXPECTED(bool, 0, 0), 2, ma1_f16, ma2_f16);
    TEST(sycl::isgreater, bool, EXPECTED(bool, 0, 1), 2, ma1_f16, ma3_f16);
    TEST(sycl::isgreaterequal, bool, EXPECTED(bool, 0, 1), 2, ma1_f16, ma4_f16);
    TEST(sycl::isless, bool, EXPECTED(bool, 0, 1), 2, ma3_f16, ma1_f16);
    TEST(sycl::islessequal, bool, EXPECTED(bool, 0, 1), 2, ma4_f16, ma1_f16);
    TEST(sycl::islessgreater, bool, EXPECTED(bool, 0, 0), 2, ma1_f16, ma2_f16);
    TEST(sycl::isfinite, bool, EXPECTED(bool, 1, 1), 2, ma1_f16);
    TEST(sycl::isinf, bool, EXPECTED(bool, 0, 0), 2, ma1_f16);
    TEST(sycl::isnan, bool, EXPECTED(bool, 0, 0), 2, ma1_f16);
    TEST(sycl::isnormal, bool, EXPECTED(bool, 1, 1), 2, ma1_f16);
    TEST(sycl::isordered, bool, EXPECTED(bool, 1, 1), 2, ma1_f16, ma2_f16);
    TEST(sycl::isunordered, bool, EXPECTED(bool, 0, 0), 2, ma1_f16, ma2_f16);
    TEST(sycl::signbit, bool, EXPECTED(bool, 0, 0), 2, ma1_f16);
    TEST(sycl::bitselect, sycl::half, EXPECTED(float, 1.0, 1.0), 2, ma8_f16,
         ma9_f16, ma10_f16);
    TEST(sycl::select, sycl::half, EXPECTED(float, 1.0, 2.0, 8.0), 3, ma5_f16,
         ma6_f16, c);
  }

  TEST(sycl::isequal, bool, EXPECTED(bool, 1, 1), 2, ma1, ma2);
  TEST(sycl::isnotequal, bool, EXPECTED(bool, 0, 0), 2, ma1, ma2);
  TEST(sycl::isgreater, bool, EXPECTED(bool, 0, 1), 2, ma1, ma3);
  TEST(sycl::isgreaterequal, bool, EXPECTED(bool, 0, 1), 2, ma1, ma4);
  TEST(sycl::isless, bool, EXPECTED(bool, 0, 1), 2, ma3, ma1);
  TEST(sycl::islessequal, bool, EXPECTED(bool, 0, 1), 2, ma4, ma1);
  TEST(sycl::islessgreater, bool, EXPECTED(bool, 0, 0), 2, ma1, ma2);
  TEST(sycl::isfinite, bool, EXPECTED(bool, 1, 1), 2, ma1);
  TEST(sycl::isinf, bool, EXPECTED(bool, 0, 0), 2, ma1);
  TEST(sycl::isnan, bool, EXPECTED(bool, 0, 0), 2, ma1);
  TEST(sycl::isnormal, bool, EXPECTED(bool, 1, 1), 2, ma1);
  TEST(sycl::isordered, bool, EXPECTED(bool, 1, 1), 2, ma1, ma2);
  TEST(sycl::isunordered, bool, EXPECTED(bool, 0, 0), 2, ma1, ma2);
  TEST(sycl::signbit, bool, EXPECTED(bool, 0, 0), 2, ma1);
  TEST2(sycl::all, int, EXPECTED(bool, false), 3, ma7);
  TEST2(sycl::any, int, EXPECTED(bool, false), 3, ma7);
  TEST(sycl::bitselect, float, EXPECTED(float, 1.0, 1.0), 2, ma8, ma9, ma10);
  TEST(sycl::select, float, EXPECTED(float, 1.0, 2.0, 8.0), 3, ma5, ma6, c);

  return 0;
}
