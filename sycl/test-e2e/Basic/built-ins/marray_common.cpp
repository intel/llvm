// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out%}

#ifdef _WIN32
#define _USE_MATH_DEFINES // To use math constants
#endif
#include <cmath>

#include <sycl/sycl.hpp>

#define TEST(FUNC, MARRAY_ELEM_TYPE, DIM, EXPECTED, DELTA, ...)                \
  {                                                                            \
    {                                                                          \
      MARRAY_ELEM_TYPE result[DIM];                                            \
      {                                                                        \
        sycl::buffer<MARRAY_ELEM_TYPE> b(result, sycl::range{DIM});            \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::marray<MARRAY_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__);       \
            for (int i = 0; i < DIM; i++)                                      \
              res_access[i] = res[i];                                          \
          });                                                                  \
        });                                                                    \
      }                                                                        \
      for (int i = 0; i < DIM; i++)                                            \
        assert(abs(result[i] - EXPECTED[i]) <= DELTA);                         \
    }                                                                          \
  }

#define EXPECTED(TYPE, ...) ((TYPE[]){__VA_ARGS__})

int main() {
  sycl::queue deviceQueue;
  sycl::device dev = deviceQueue.get_device();

  sycl::marray<float, 2> ma1{1.0f, 2.0f};
  sycl::marray<float, 2> ma2{1.0f, 2.0f};
  sycl::marray<float, 2> ma3{3.0f, 2.0f};
  sycl::marray<double, 2> ma4{1.0, 2.0};
  sycl::marray<float, 3> ma5{M_PI, M_PI, M_PI};
  sycl::marray<double, 3> ma6{M_PI, M_PI, M_PI};
  sycl::marray<sycl::half, 3> ma7{M_PI, M_PI, M_PI};
  sycl::marray<float, 2> ma8{0.3f, 0.6f};
  sycl::marray<double, 2> ma9{5.0, 8.0};
  sycl::marray<float, 3> ma10{180, 180, 180};
  sycl::marray<double, 3> ma11{180, 180, 180};
  sycl::marray<sycl::half, 3> ma12{180, 180, 180};
  sycl::marray<sycl::half, 3> ma13{181, 179, 181};
  sycl::marray<float, 2> ma14{+0.0f, -0.6f};
  sycl::marray<double, 2> ma15{-0.0, 0.6f};

  // sycl::clamp
  TEST(sycl::clamp, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, ma1, ma2, ma3);
  TEST(sycl::clamp, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, ma1, 1.0f, 3.0f);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::clamp, double, 2, EXPECTED(double, 1.0, 2.0), 0, ma4, 1.0, 3.0);
  // sycl::degrees
  TEST(sycl::degrees, float, 3, EXPECTED(float, 180, 180, 180), 0, ma5);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::degrees, double, 3, EXPECTED(double, 180, 180, 180), 0, ma6);
  if (dev.has(sycl::aspect::fp16))
    TEST(sycl::degrees, sycl::half, 3, EXPECTED(sycl::half, 180, 180, 180), 0.2,
         ma7);
  // sycl::max
  TEST(sycl::max, float, 2, EXPECTED(float, 3.0f, 2.0f), 0, ma1, ma3);
  TEST(sycl::max, float, 2, EXPECTED(float, 1.5f, 2.0f), 0, ma1, 1.5f);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::max, double, 2, EXPECTED(double, 1.5, 2.0), 0, ma4, 1.5);
  // sycl::min
  TEST(sycl::min, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, ma1, ma3);
  TEST(sycl::min, float, 2, EXPECTED(float, 1.0f, 1.5f), 0, ma1, 1.5f);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::min, double, 2, EXPECTED(double, 1.0, 1.5), 0, ma4, 1.5);
  // sycl::mix
  TEST(sycl::mix, float, 2, EXPECTED(float, 1.6f, 2.0f), 0, ma1, ma3, ma8);
  TEST(sycl::mix, float, 2, EXPECTED(float, 1.4f, 2.0f), 0, ma1, ma3, 0.2);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::mix, double, 2, EXPECTED(double, 3.0, 5.0), 0, ma4, ma9, 0.5);
  // sycl::radians
  TEST(sycl::radians, float, 3, EXPECTED(float, M_PI, M_PI, M_PI), 0, ma10);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::radians, double, 3, EXPECTED(double, M_PI, M_PI, M_PI), 0, ma11);
  if (dev.has(sycl::aspect::fp16))
    TEST(sycl::radians, sycl::half, 3, EXPECTED(sycl::half, M_PI, M_PI, M_PI),
         0.002, ma12);
  // sycl::step
  TEST(sycl::step, float, 2, EXPECTED(float, 1.0f, 1.0f), 0, ma1, ma3);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::step, double, 2, EXPECTED(double, 1.0, 1.0), 0, ma4, ma9);
  if (dev.has(sycl::aspect::fp16))
    TEST(sycl::step, sycl::half, 3, EXPECTED(sycl::half, 1.0, 0.0, 1.0), 0,
         ma12, ma13);
  TEST(sycl::step, float, 2, EXPECTED(float, 1.0f, 0.0f), 0, 2.5f, ma3);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::step, double, 2, EXPECTED(double, 0.0f, 1.0f), 0, 6.0f, ma9);
  // sycl::smoothstep
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 1.0f, 1.0f), 0, ma8, ma1,
       ma2);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 1.0, 1.0f), 0.00000001,
         ma4, ma9, ma9);
  if (dev.has(sycl::aspect::fp16))
    TEST(sycl::smoothstep, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0),
         0, ma7, ma12, ma13);
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 0.0553936f, 0.0f), 0.0000001,
       2.5f, 6.0f, ma3);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 0.0f, 1.0f), 0, 6.0f,
         8.0f, ma9);
  // sign
  TEST(sycl::sign, float, 2, EXPECTED(float, +0.0f, -1.0f), 0, ma14);
  if (dev.has(sycl::aspect::fp64))
    TEST(sycl::sign, double, 2, EXPECTED(double, -0.0, 1.0), 0, ma15);
  if (dev.has(sycl::aspect::fp16))
    TEST(sycl::sign, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0), 0,
         ma12);

  return 0;
}
