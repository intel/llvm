// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

#include <sycl/sycl.hpp>

#define TEST(FUNC, VEC_ELEM_TYPE, DIM, EXPECTED, DELTA, ...)                   \
  {                                                                            \
    {                                                                          \
      VEC_ELEM_TYPE result[DIM];                                               \
      {                                                                        \
        sycl::buffer<VEC_ELEM_TYPE> b(result, sycl::range{DIM});               \
        Queue.submit([&](sycl::handler &cgh) {                                 \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::vec<VEC_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__);             \
            for (int i = 0; i < DIM; i++)                                      \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < DIM; i++) {                                          \
        assert(abs(result[i] - EXPECTED[i]) <= DELTA);                         \
      }                                                                        \
    }                                                                          \
  }

#define TEST2(FUNC, TYPE, EXPECTED, DELTA, ...)                                \
  {                                                                            \
    {                                                                          \
      TYPE result;                                                             \
      {                                                                        \
        sycl::buffer<TYPE> b(&result, 1);                                      \
        Queue.submit([&](sycl::handler &cgh) {                                 \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() { res_access[0] = FUNC(__VA_ARGS__); });       \
        });                                                                    \
      }                                                                        \
      assert(abs(result - EXPECTED) <= DELTA);                                 \
    }                                                                          \
  }

#define EXPECTED(TYPE, ...) ((TYPE[]){__VA_ARGS__})

int main() {
  sycl::device Dev;
  sycl::queue Queue(Dev);
  // clang-format off
  sycl::vec<float, 2> VFloatD2   = {1.f, 2.f};
  sycl::vec<float, 2> VFloatD2_2 = {3.f, 5.f};
  sycl::vec<float, 3> VFloatD3   = {1.f, 2.f, 3.f};
  sycl::vec<float, 3> VFloatD3_2 = {1.f, 5.f, 7.f};
  sycl::vec<float, 4> VFloatD4   = {1.f, 2.f, 3.f, 4.f};
  sycl::vec<float, 4> VFloatD4_2 = {1.f, 5.f, 7.f, 4.f};

  sycl::vec<double, 2> VDoubleD2   = {1.0, 2.0};
  sycl::vec<double, 2> VDoubleD2_2 = {3.0, 5.0};
  sycl::vec<double, 3> VDoubleD3   = {1.0, 2.0, 3.0};
  sycl::vec<double, 3> VDoubleD3_2 = {1.0, 5.0, 7.0};
  sycl::vec<double, 4> VDoubleD4   = {1.0, 2.0, 3.0, 4.0};
  sycl::vec<double, 4> VDoubleD4_2 = {1.0, 5.0, 7.0, 4.0};
  // clang-format on

  TEST(sycl::cross, float, 3, EXPECTED(float, -1.f, -4.f, 3.f), 0, VFloatD3,
       VFloatD3_2);
  TEST(sycl::cross, float, 4, EXPECTED(float, -1.f, -4.f, 3.f, 0.f), 0,
       VFloatD4, VFloatD4_2);
  if (Dev.has(sycl::aspect::fp64)) {
    TEST(sycl::cross, double, 3, EXPECTED(double, -1.f, -4.f, 3.f), 0,
         VDoubleD3, VDoubleD3_2);
    TEST(sycl::cross, double, 4, EXPECTED(double, -1.f, -4.f, 3.f, 0.f), 0,
         VDoubleD4, VDoubleD4_2);
  }

  TEST2(sycl::dot, float, 13.f, 0, VFloatD2, VFloatD2_2);
  TEST2(sycl::dot, float, 32.f, 0, VFloatD3, VFloatD3_2);
  TEST2(sycl::dot, float, 48.f, 0, VFloatD4, VFloatD4_2);
  if (Dev.has(sycl::aspect::fp64)) {
    TEST2(sycl::dot, double, 13, 0, VDoubleD2, VDoubleD2_2);
    TEST2(sycl::dot, double, 32, 0, VDoubleD3, VDoubleD3_2);
    TEST2(sycl::dot, double, 48, 0, VDoubleD4, VDoubleD4_2);
  }

  TEST2(sycl::length, float, 2.236068f, 1e-6, VFloatD2);
  TEST2(sycl::length, float, 3.741657f, 1e-6, VFloatD3);
  TEST2(sycl::length, float, 5.477225f, 1e-6, VFloatD4);
  if (Dev.has(sycl::aspect::fp64)) {
    TEST2(sycl::length, double, 2.236068, 1e-6, VDoubleD2);
    TEST2(sycl::length, double, 3.741657, 1e-6, VDoubleD3);
    TEST2(sycl::length, double, 5.477225, 1e-6, VDoubleD4);
  }

  TEST2(sycl::distance, float, 3.605551f, 1e-6, VFloatD2, VFloatD2_2);
  TEST2(sycl::distance, float, 5.f, 0, VFloatD3, VFloatD3_2);
  TEST2(sycl::distance, float, 5.f, 0, VFloatD4, VFloatD4_2);
  if (Dev.has(sycl::aspect::fp64)) {
    TEST2(sycl::distance, double, 3.605551, 1e-6, VDoubleD2, VDoubleD2_2);
    TEST2(sycl::distance, double, 5.0, 0, VDoubleD3, VDoubleD3_2);
    TEST2(sycl::distance, double, 5.0, 0, VDoubleD4, VDoubleD4_2);
  }

  TEST(sycl::normalize, float, 2, EXPECTED(float, 0.447213f, 0.894427f), 1e-6,
       VFloatD2);
  TEST(sycl::normalize, float, 3,
       EXPECTED(float, 0.267261f, 0.534522f, 0.801784f), 1e-6, VFloatD3);
  TEST(sycl::normalize, float, 4,
       EXPECTED(float, 0.182574f, 0.365148f, 0.547723f, 0.730297f), 1e-6,
       VFloatD4);
  if (Dev.has(sycl::aspect::fp64)) {
    TEST(sycl::normalize, double, 2, EXPECTED(double, 0.447213, 0.894427), 1e-6,
         VDoubleD2);
    TEST(sycl::normalize, double, 3,
         EXPECTED(double, 0.267261, 0.534522, 0.801784), 1e-6, VDoubleD3);
    TEST(sycl::normalize, double, 4,
         EXPECTED(double, 0.182574, 0.365148, 0.547723, 0.730297), 1e-6,
         VDoubleD4);
  }

  TEST2(sycl::fast_distance, float, 3.605551f, 1e-6, VFloatD2, VFloatD2_2);
  TEST2(sycl::fast_distance, float, 5.f, 0, VFloatD3, VFloatD3_2);
  TEST2(sycl::fast_distance, float, 5.f, 0, VFloatD4, VFloatD4_2);

  TEST2(sycl::fast_length, float, 2.236068f, 1e-6, VFloatD2);
  TEST2(sycl::fast_length, float, 3.741657f, 1e-6, VFloatD3);
  TEST2(sycl::fast_length, float, 5.477225f, 1e-6, VFloatD4);

  TEST(sycl::fast_normalize, float, 2, EXPECTED(float, 0.447213f, 0.894427f),
       1e-3, VFloatD2);
  TEST(sycl::fast_normalize, float, 3,
       EXPECTED(float, 0.267261f, 0.534522f, 0.801784f), 1e-3, VFloatD3);
  TEST(sycl::fast_normalize, float, 4,
       EXPECTED(float, 0.182574f, 0.365148f, 0.547723f, 0.730297f), 1e-3,
       VFloatD4);

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  TEST(sycl::cross, float, 3, EXPECTED(float, -1.f, -4.f, 3.f), 0,
       VFloatD4.swizzle<0, 1, 2>(), VFloatD3_2);
  TEST(sycl::cross, float, 3, EXPECTED(float, -1.f, -4.f, 3.f), 0, VFloatD3,
       VFloatD4_2.swizzle<0, 1, 2>());
  TEST(sycl::cross, float, 3, EXPECTED(float, -1.f, -4.f, 3.f), 0,
       VFloatD4.swizzle<0, 1, 2>(), VFloatD4_2.swizzle<0, 1, 2>());
  if (Dev.has(sycl::aspect::fp64)) {
    TEST(sycl::cross, double, 3, EXPECTED(double, -1.f, -4.f, 3.f), 0,
         VDoubleD3, VDoubleD4_2.swizzle<0, 1, 2>());
    TEST(sycl::cross, double, 3, EXPECTED(double, -1.f, -4.f, 3.f), 0,
         VDoubleD4.swizzle<0, 1, 2>(), VDoubleD3_2);
    TEST(sycl::cross, double, 3, EXPECTED(double, -1.f, -4.f, 3.f), 0,
         VDoubleD4.swizzle<0, 1, 2>(), VDoubleD4_2.swizzle<0, 1, 2>());
  }

  TEST2(sycl::dot, float, 32.f, 0, VFloatD4.swizzle<0, 1, 2>(), VFloatD3_2);
  TEST2(sycl::dot, float, 32.f, 0, VFloatD3, VFloatD4_2.swizzle<0, 1, 2>());
  TEST2(sycl::dot, float, 32.f, 0, VFloatD4.swizzle<0, 1, 2>(),
        VFloatD4_2.swizzle<0, 1, 2>());
  if (Dev.has(sycl::aspect::fp64)) {
    TEST2(sycl::dot, double, 32, 0, VDoubleD4.swizzle<0, 1, 2>(), VDoubleD3_2);
    TEST2(sycl::dot, double, 32, 0, VDoubleD3, VDoubleD4_2.swizzle<0, 1, 2>());
    TEST2(sycl::dot, double, 32, 0, VDoubleD4.swizzle<0, 1, 2>(),
          VDoubleD4_2.swizzle<0, 1, 2>());
  }

  TEST2(sycl::length, float, 3.741657f, 1e-6, VFloatD4.swizzle<0, 1, 2>());
  if (Dev.has(sycl::aspect::fp64)) {
    TEST2(sycl::length, double, 3.741657, 1e-6, VDoubleD4.swizzle<0, 1, 2>());
  }

  TEST2(sycl::distance, float, 5.f, 0, VFloatD4.swizzle<0, 1, 2>(), VFloatD3_2);
  TEST2(sycl::distance, float, 5.f, 0, VFloatD3, VFloatD4_2.swizzle<0, 1, 2>());
  TEST2(sycl::distance, float, 5.f, 0, VFloatD4.swizzle<0, 1, 2>(),
        VFloatD4_2.swizzle<0, 1, 2>());
  if (Dev.has(sycl::aspect::fp64)) {
    TEST2(sycl::distance, double, 5.0, 0, VDoubleD4.swizzle<0, 1, 2>(),
          VDoubleD3_2);
    TEST2(sycl::distance, double, 5.0, 0, VDoubleD3,
          VDoubleD4_2.swizzle<0, 1, 2>());
    TEST2(sycl::distance, double, 5.0, 0, VDoubleD4.swizzle<0, 1, 2>(),
          VDoubleD4_2.swizzle<0, 1, 2>());
  }

  TEST(sycl::normalize, float, 3, EXPECTED(float, 0.267261, 0.534522, 0.801784),
       1e-6, VFloatD4.swizzle<0, 1, 2>());
  if (Dev.has(sycl::aspect::fp64)) {
    TEST(sycl::normalize, double, 3,
         EXPECTED(double, 0.267261, 0.534522, 0.801784), 1e-6,
         VDoubleD4.swizzle<0, 1, 2>());
  }

  TEST2(sycl::fast_distance, float, 5.f, 0, VFloatD4.swizzle<0, 1, 2>(),
        VFloatD3_2);
  TEST2(sycl::fast_distance, float, 5.f, 0, VFloatD3,
        VFloatD4_2.swizzle<0, 1, 2>());
  TEST2(sycl::fast_distance, float, 5.f, 0, VFloatD4.swizzle<0, 1, 2>(),
        VFloatD4_2.swizzle<0, 1, 2>());

  TEST2(sycl::fast_length, float, 3.741657f, 1e-6, VFloatD4.swizzle<0, 1, 2>());

  TEST(sycl::fast_normalize, float, 3,
       EXPECTED(float, 0.267261f, 0.534522f, 0.801784f), 1e-3,
       VFloatD4.swizzle<0, 1, 2>());
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

  return 0;
}
