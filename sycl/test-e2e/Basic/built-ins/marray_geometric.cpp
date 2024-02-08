// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out %}

#include <CL/sycl.hpp>

#define TEST(FUNC, MARRAY_ELEM_TYPE, DIM, EXPECTED, DELTA, ...)                \
  {                                                                            \
    {                                                                          \
      MARRAY_ELEM_TYPE result[DIM];                                            \
      {                                                                        \
        sycl::buffer<MARRAY_ELEM_TYPE> b(result, sycl::range{DIM});            \
        Queue.submit([&](sycl::handler &cgh) {                                 \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::marray<MARRAY_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__);       \
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
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  sycl::marray<sycl::half, 2> MHalfD2   = {1.f, 2.f};
  sycl::marray<sycl::half, 2> MHalfD2_2 = {3.f, 5.f};
  sycl::marray<sycl::half, 3> MHalfD3   = {1.f, 2.f, 3.f};
  sycl::marray<sycl::half, 3> MHalfD3_2 = {1.f, 5.f, 7.f};
  sycl::marray<sycl::half, 4> MHalfD4   = {1.f, 2.f, 3.f, 4.f};
  sycl::marray<sycl::half, 4> MHalfD4_2 = {1.f, 5.f, 7.f, 4.f};
#endif

  sycl::marray<float, 2> MFloatD2   = {1.f, 2.f};
  sycl::marray<float, 2> MFloatD2_2 = {3.f, 5.f};
  sycl::marray<float, 3> MFloatD3   = {1.f, 2.f, 3.f};
  sycl::marray<float, 3> MFloatD3_2 = {1.f, 5.f, 7.f};
  sycl::marray<float, 4> MFloatD4   = {1.f, 2.f, 3.f, 4.f};
  sycl::marray<float, 4> MFloatD4_2 = {1.f, 5.f, 7.f, 4.f};

  sycl::marray<double, 2> MDoubleD2   = {1.0, 2.0};
  sycl::marray<double, 2> MDoubleD2_2 = {3.0, 5.0};
  sycl::marray<double, 3> MDoubleD3   = {1.0, 2.0, 3.0};
  sycl::marray<double, 3> MDoubleD3_2 = {1.0, 5.0, 7.0};
  sycl::marray<double, 4> MDoubleD4   = {1.0, 2.0, 3.0, 4.0};
  sycl::marray<double, 4> MDoubleD4_2 = {1.0, 5.0, 7.0, 4.0};
  // clang-format on

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  if (Dev.has(sycl::aspect::fp16)) {
    TEST(sycl::cross, sycl::half, 3, EXPECTED(sycl::half, -1.f, -4.f, 3.f), 0,
         MHalfD3, MHalfD3_2);
    TEST(sycl::cross, sycl::half, 4, EXPECTED(sycl::half, -1.f, -4.f, 3.f, 0.f),
         0, MHalfD4, MHalfD4_2);
  }
#endif

  TEST(sycl::cross, float, 3, EXPECTED(float, -1.f, -4.f, 3.f), 0, MFloatD3,
       MFloatD3_2);
  TEST(sycl::cross, float, 4, EXPECTED(float, -1.f, -4.f, 3.f, 0.f), 0,
       MFloatD4, MFloatD4_2);
  if (Dev.has(sycl::aspect::fp64)) {
    TEST(sycl::cross, double, 3, EXPECTED(double, -1.f, -4.f, 3.f), 0,
         MDoubleD3, MDoubleD3_2);
    TEST(sycl::cross, double, 4, EXPECTED(double, -1.f, -4.f, 3.f, 0.f), 0,
         MDoubleD4, MDoubleD4_2);
  }

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  if (Dev.has(sycl::aspect::fp16)) {
    TEST2(sycl::dot, sycl::half, 13.f, 0, MHalfD2, MHalfD2_2);
    TEST2(sycl::dot, sycl::half, 32.f, 0, MHalfD3, MHalfD3_2);
    TEST2(sycl::dot, sycl::half, 48.f, 0, MHalfD4, MHalfD4_2);
  }
#endif

  TEST2(sycl::dot, float, 13.f, 0, MFloatD2, MFloatD2_2);
  TEST2(sycl::dot, float, 32.f, 0, MFloatD3, MFloatD3_2);
  TEST2(sycl::dot, float, 48.f, 0, MFloatD4, MFloatD4_2);
  if (Dev.has(sycl::aspect::fp64)) {
    TEST2(sycl::dot, double, 13.0, 0, MDoubleD2, MDoubleD2_2);
    TEST2(sycl::dot, double, 32.0, 0, MDoubleD3, MDoubleD3_2);
    TEST2(sycl::dot, double, 48.0, 0, MDoubleD4, MDoubleD4_2);
  }

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  if (Dev.has(sycl::aspect::fp16)) {
    TEST2(sycl::length, sycl::half, 2.236f, 1e-3, MHalfD2);
    TEST2(sycl::length, sycl::half, 3.742f, 1e-3, MHalfD3);
    TEST2(sycl::length, sycl::half, 5.477f, 1e-3, MHalfD4);
  }
#endif

  TEST2(sycl::length, float, 2.236068f, 1e-6, MFloatD2);
  TEST2(sycl::length, float, 3.741657f, 1e-6, MFloatD3);
  TEST2(sycl::length, float, 5.477225f, 1e-6, MFloatD4);
  if (Dev.has(sycl::aspect::fp64)) {
    TEST2(sycl::length, double, 2.236068, 1e-6, MDoubleD2);
    TEST2(sycl::length, double, 3.741657, 1e-6, MDoubleD3);
    TEST2(sycl::length, double, 5.477225, 1e-6, MDoubleD4);
  }

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  if (Dev.has(sycl::aspect::fp16)) {
    TEST2(sycl::distance, sycl::half, 3.605f, 1e-3, MHalfD2, MHalfD2_2);
    TEST2(sycl::distance, sycl::half, 5.f, 0, MHalfD3, MHalfD3_2);
    TEST2(sycl::distance, sycl::half, 5.f, 0, MHalfD4, MHalfD4_2);
  }
#endif

  TEST2(sycl::distance, float, 3.605551f, 1e-6, MFloatD2, MFloatD2_2);
  TEST2(sycl::distance, float, 5.f, 0, MFloatD3, MFloatD3_2);
  TEST2(sycl::distance, float, 5.f, 0, MFloatD4, MFloatD4_2);
  if (Dev.has(sycl::aspect::fp64)) {
    TEST2(sycl::distance, double, 3.605551, 1e-6, MDoubleD2, MDoubleD2_2);
    TEST2(sycl::distance, double, 5.0, 0, MDoubleD3, MDoubleD3_2);
    TEST2(sycl::distance, double, 5.0, 0, MDoubleD4, MDoubleD4_2);
  }

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  if (Dev.has(sycl::aspect::fp16)) {
    TEST(sycl::normalize, sycl::half, 2,
         EXPECTED(sycl::half, 0.447213f, 0.894427f), 1e-6, MHalfD2);
    TEST(sycl::normalize, sycl::half, 3,
         EXPECTED(sycl::half, 0.267261f, 0.534522f, 0.801784f), 1e-6, MHalfD3);
    TEST(sycl::normalize, sycl::half, 4,
         EXPECTED(sycl::half, 0.182574f, 0.365148f, 0.547723f, 0.730297f), 1e-6,
         MHalfD4);
  }
#endif

  TEST(sycl::normalize, float, 2, EXPECTED(float, 0.447213f, 0.894427f), 1e-6,
       MFloatD2);
  TEST(sycl::normalize, float, 3,
       EXPECTED(float, 0.267261f, 0.534522f, 0.801784f), 1e-6, MFloatD3);
  TEST(sycl::normalize, float, 4,
       EXPECTED(float, 0.182574f, 0.365148f, 0.547723f, 0.730297f), 1e-6,
       MFloatD4);
  if (Dev.has(sycl::aspect::fp64)) {
    TEST(sycl::normalize, double, 2, EXPECTED(double, 0.447213, 0.894427), 1e-6,
         MDoubleD2);
    TEST(sycl::normalize, double, 3,
         EXPECTED(double, 0.267261, 0.534522, 0.801784), 1e-6, MDoubleD3);
    TEST(sycl::normalize, double, 4,
         EXPECTED(double, 0.182574, 0.365148, 0.547723, 0.730297), 1e-6,
         MDoubleD4);
  }

  TEST2(sycl::fast_distance, float, 3.605551f, 1e-6, MFloatD2, MFloatD2_2);
  TEST2(sycl::fast_distance, float, 5.f, 0, MFloatD3, MFloatD3_2);
  TEST2(sycl::fast_distance, float, 5.f, 0, MFloatD4, MFloatD4_2);

  TEST2(sycl::fast_length, float, 2.236068f, 1e-6, MFloatD2);
  TEST2(sycl::fast_length, float, 3.741657f, 1e-6, MFloatD3);
  TEST2(sycl::fast_length, float, 5.477225f, 1e-6, MFloatD4);

  TEST(sycl::fast_normalize, float, 2, EXPECTED(float, 0.447213f, 0.894427f),
       1e-3, MFloatD2);
  TEST(sycl::fast_normalize, float, 3,
       EXPECTED(float, 0.267261f, 0.534522f, 0.801784f), 1e-3, MFloatD3);
  TEST(sycl::fast_normalize, float, 4,
       EXPECTED(float, 0.182574f, 0.365148f, 0.547723f, 0.730297f), 1e-3,
       MFloatD4);

  return 0;
}
