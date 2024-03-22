// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t2.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t2.out %}

#ifdef _WIN32
#define _USE_MATH_DEFINES // To use math constants
#endif
#include <cmath>

#include <sycl/detail/core.hpp>

#include <sycl/builtins.hpp>

#define TEST(FUNC, VEC_ELEM_TYPE, DIM, EXPECTED, DELTA, ...)                   \
  {                                                                            \
    {                                                                          \
      VEC_ELEM_TYPE result[DIM];                                               \
      {                                                                        \
        sycl::buffer<VEC_ELEM_TYPE> b(result, sycl::range{DIM});               \
        deviceQueue.submit([&](sycl::handler &cgh) {                           \
          sycl::accessor res_access{b, cgh};                                   \
          cgh.single_task([=]() {                                              \
            sycl::vec<VEC_ELEM_TYPE, DIM> res = FUNC(__VA_ARGS__);             \
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

  sycl::vec<float, 2> va1{1.0f, 2.0f};
  sycl::vec<float, 2> va2{1.0f, 2.0f};
  sycl::vec<float, 2> va3{3.0f, 2.0f};
  sycl::vec<double, 2> va4{1.0, 2.0};
  sycl::vec<float, 3> va5{M_PI, M_PI, M_PI};
  sycl::vec<double, 3> va6{M_PI, M_PI, M_PI};
  sycl::vec<sycl::half, 3> va7{M_PI, M_PI, M_PI};
  sycl::vec<float, 2> va8{0.3f, 0.6f};
  sycl::vec<double, 2> va9{5.0, 8.0};
  sycl::vec<float, 3> va10{180, 180, 180};
  sycl::vec<double, 3> va11{180, 180, 180};
  sycl::vec<sycl::half, 3> va12{180, 180, 180};
  sycl::vec<sycl::half, 3> va13{181, 179, 181};
  sycl::vec<float, 2> va14{+0.0f, -0.6f};
  sycl::vec<double, 2> va15{-0.0, 0.6f};
  sycl::vec<float, 3> va16{180, 360, 540};
  sycl::vec<double, 3> va17{180, 360, 540};
  sycl::vec<float, 3> va18{0.3f, 0.6f, 0.9f};

  // sycl::clamp
  TEST(sycl::clamp, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, va1, va2, va3);
  TEST(sycl::clamp, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, va1, 1.0f, 3.0f);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::clamp, double, 2, EXPECTED(double, 1.0, 2.0), 0, va4, 1.0, 3.0);
  }
  // sycl::degrees
  TEST(sycl::degrees, float, 3, EXPECTED(float, 180, 180, 180), 0, va5);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::degrees, double, 3, EXPECTED(double, 180, 180, 180), 0, va6);
  }
  if (dev.has(sycl::aspect::fp16)) {
    TEST(sycl::degrees, sycl::half, 3, EXPECTED(sycl::half, 180, 180, 180), 0.2,
         va7);
  }
  // sycl::max
  TEST(sycl::max, float, 2, EXPECTED(float, 3.0f, 2.0f), 0, va1, va3);
  TEST(sycl::max, float, 2, EXPECTED(float, 1.5f, 2.0f), 0, va1, 1.5f);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::max, double, 2, EXPECTED(double, 1.5, 2.0), 0, va4, 1.5);
  }
  // sycl::min
  TEST(sycl::min, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, va1, va3);
  TEST(sycl::min, float, 2, EXPECTED(float, 1.0f, 1.5f), 0, va1, 1.5f);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::min, double, 2, EXPECTED(double, 1.0, 1.5), 0, va4, 1.5);
  }
  // sycl::mix
  TEST(sycl::mix, float, 2, EXPECTED(float, 1.6f, 2.0f), 0, va1, va3, va8);
  TEST(sycl::mix, float, 2, EXPECTED(float, 1.4f, 2.0f), 0, va1, va3, 0.2);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::mix, double, 2, EXPECTED(double, 3.0, 5.0), 0, va4, va9, 0.5);
  }
  // sycl::radians
  TEST(sycl::radians, float, 3, EXPECTED(float, M_PI, M_PI, M_PI), 0, va10);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::radians, double, 3, EXPECTED(double, M_PI, M_PI, M_PI), 0, va11);
  }
  if (dev.has(sycl::aspect::fp16)) {
    TEST(sycl::radians, sycl::half, 3, EXPECTED(sycl::half, M_PI, M_PI, M_PI),
         0.002, va12);
  }
  // sycl::step
  TEST(sycl::step, float, 2, EXPECTED(float, 1.0f, 1.0f), 0, va1, va3);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::step, double, 2, EXPECTED(double, 1.0, 1.0), 0, va4, va9);
  }
  if (dev.has(sycl::aspect::fp16)) {
    TEST(sycl::step, sycl::half, 3, EXPECTED(sycl::half, 1.0, 0.0, 1.0), 0,
         va12, va13);
  }
  TEST(sycl::step, float, 2, EXPECTED(float, 1.0f, 0.0f), 0, 2.5f, va3);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::step, double, 2, EXPECTED(double, 0.0f, 1.0f), 0, 6.0f, va9);
  }
  // sycl::smoothstep
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 1.0f, 1.0f), 0, va8, va1,
       va2);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 1.0, 1.0f), 0.00000001,
         va4, va9, va9);
  }
  if (dev.has(sycl::aspect::fp16)) {
    TEST(sycl::smoothstep, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0),
         0, va7, va12, va13);
  }
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 0.0553936f, 0.0f), 0.0000001,
       2.5f, 6.0f, va3);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 0.0f, 1.0f), 0, 6.0f,
         8.0f, va9);
  }
  // sign
  TEST(sycl::sign, float, 2, EXPECTED(float, +0.0f, -1.0f), 0, va14);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::sign, double, 2, EXPECTED(double, -0.0, 1.0), 0, va15);
  }
  if (dev.has(sycl::aspect::fp16)) {
    TEST(sycl::sign, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0), 0,
         va12);
  }

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  // sycl::clamp swizzled
  TEST(sycl::clamp, float, 2, EXPECTED(float, 3.0f, 2.0f), 0,
       va16.swizzle<1, 0>(), va2, va3);
  TEST(sycl::clamp, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, va1, va2,
       va16.swizzle<1, 0>());
  TEST(sycl::clamp, float, 2, EXPECTED(float, 360.0f, 180.0f), 0,
       va16.swizzle<1, 0>(), va2, va16.swizzle<1, 0>());
  TEST(sycl::clamp, float, 2, EXPECTED(float, 360.0f, 180.0f), 0, va1,
       va16.swizzle<1, 0>(), va16.swizzle<1, 0>());
  TEST(sycl::clamp, float, 2, EXPECTED(float, 360.0f, 180.0f), 0,
       va16.swizzle<1, 0>(), va16.swizzle<1, 0>(), va16.swizzle<1, 0>());
  TEST(sycl::clamp, float, 2, EXPECTED(float, 3.0f, 3.0f), 0,
       va16.swizzle<1, 0>(), 1.0f, 3.0f);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::clamp, double, 2, EXPECTED(double, 3.0, 3.0), 0,
         va11.swizzle<1, 0>(), 1.0, 3.0);
  }
  // sycl::degrees swizzled
  TEST(sycl::degrees, float, 2, EXPECTED(float, 180, 180), 0,
       va5.swizzle<1, 0>());
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::degrees, double, 2, EXPECTED(double, 180, 180), 0,
         va6.swizzle<1, 0>());
  }
  if (dev.has(sycl::aspect::fp16)) {
    TEST(sycl::degrees, sycl::half, 2, EXPECTED(sycl::half, 180, 180), 0.2,
         va7.swizzle<1, 0>());
  }
  // sycl::max swizzled
  TEST(sycl::max, float, 2, EXPECTED(float, 360.0f, 180.0f), 0,
       va16.swizzle<1, 0>(), va3);
  TEST(sycl::max, float, 2, EXPECTED(float, 360.0f, 180.0f), 0, va1,
       va16.swizzle<1, 0>());
  TEST(sycl::max, float, 2, EXPECTED(float, 360.0f, 190.0f), 0,
       va16.swizzle<1, 0>(), 190.0f);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::max, double, 2, EXPECTED(double, 360.0, 190.0), 0,
         va17.swizzle<1, 0>(), 190.0);
  }
  // sycl::min swizzled
  TEST(sycl::min, float, 2, EXPECTED(float, 3.0f, 2.0f), 0,
       va16.swizzle<1, 0>(), va3);
  TEST(sycl::min, float, 2, EXPECTED(float, 1.0f, 2.0f), 0, va1,
       va16.swizzle<1, 0>());
  TEST(sycl::min, float, 2, EXPECTED(float, 190.0f, 180.0f), 0,
       va16.swizzle<1, 0>(), 190.0f);
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::min, double, 2, EXPECTED(double, 190.0f, 180.0f), 0,
         va17.swizzle<1, 0>(), 190.0);
  }
  // sycl::mix swizzled
  TEST(sycl::mix, float, 2, EXPECTED(float, 252.9f, 73.2f), 0,
       va16.swizzle<1, 0>(), va3, va8);
  TEST(sycl::mix, float, 2, EXPECTED(float, 252.9f, 73.2f), 0,
       va16.swizzle<1, 0>(), va3, va18.swizzle<0, 1>());
  TEST(sycl::mix, float, 2, EXPECTED(float, 108.7f, 108.8f), 0.00001, va1,
       va16.swizzle<1, 0>(), va8);
  TEST(sycl::mix, float, 2, EXPECTED(float, 108.7f, 108.8f), 0.00001, va1,
       va16.swizzle<1, 0>(), va18.swizzle<0, 1>());
  TEST(sycl::mix, float, 2, EXPECTED(float, 360.0f, 180.0f), 0,
       va16.swizzle<1, 0>(), va16.swizzle<1, 0>(), va8);
  TEST(sycl::mix, float, 2, EXPECTED(float, 360.0f, 180.0f), 0,
       va16.swizzle<1, 0>(), va16.swizzle<1, 0>(), va18.swizzle<0, 1>());
  TEST(sycl::mix, float, 2, EXPECTED(float, 1.6f, 2.0f), 0, va1, va3,
       va18.swizzle<0, 1>());
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::mix, double, 2, EXPECTED(double, 182.5, 94.0), 0,
         va17.swizzle<1, 0>(), va9, 0.5);
    TEST(sycl::mix, double, 2, EXPECTED(double, 180.5, 91.0), 0, va4,
         va17.swizzle<1, 0>(), 0.5);
    TEST(sycl::mix, double, 2, EXPECTED(double, 360.0, 180.0), 0,
         va17.swizzle<1, 0>(), va17.swizzle<1, 0>(), 0.5);
  }
  // sycl::radians swizzled
  TEST(sycl::radians, float, 2, EXPECTED(float, M_PI, M_PI), 0,
       va10.swizzle<1, 0>());
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::radians, double, 2, EXPECTED(double, M_PI, M_PI), 0,
         va11.swizzle<1, 0>());
  }
  if (dev.has(sycl::aspect::fp16)) {
    TEST(sycl::radians, sycl::half, 2, EXPECTED(sycl::half, M_PI, M_PI), 0.002,
         va12.swizzle<1, 0>());
  }
  // sycl::step swizzled
  TEST(sycl::step, float, 2, EXPECTED(float, 0.0f, 0.0f), 0,
       va16.swizzle<1, 0>(), va3);
  TEST(sycl::step, float, 2, EXPECTED(float, 1.0f, 1.0f), 0, va1,
       va16.swizzle<1, 0>());
  TEST(sycl::step, float, 2, EXPECTED(float, 1.0f, 1.0f), 0,
       va16.swizzle<1, 0>(), va16.swizzle<1, 0>());
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::step, double, 2, EXPECTED(double, 0.0, 0.0), 0,
         va17.swizzle<1, 0>(), va9);
    TEST(sycl::step, double, 2, EXPECTED(double, 1.0, 1.0), 0, va4,
         va17.swizzle<1, 0>());
    TEST(sycl::step, double, 2, EXPECTED(double, 1.0, 1.0), 0,
         va17.swizzle<1, 0>(), va17.swizzle<1, 0>());
  }
  if (dev.has(sycl::aspect::fp16)) {
    TEST(sycl::step, sycl::half, 3, EXPECTED(sycl::half, 1.0, 0.0, 1.0), 0,
         va12.swizzle<0, 1, 2>(), va13);
    TEST(sycl::step, sycl::half, 3, EXPECTED(sycl::half, 1.0, 0.0, 1.0), 0,
         va12, va13.swizzle<0, 1, 2>());
    TEST(sycl::step, sycl::half, 3, EXPECTED(sycl::half, 1.0, 0.0, 1.0), 0,
         va12.swizzle<0, 1, 2>(), va13.swizzle<0, 1, 2>());
  }
  TEST(sycl::step, float, 2, EXPECTED(float, 0.0f, 1.0f), 0, 2.5f,
       va3.swizzle<1, 0>());
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::step, double, 2, EXPECTED(double, 1.0f, 0.0f), 0, 6.0f,
         va9.swizzle<1, 0>());
  }
  // sycl::smoothstep swizzled
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 1.0f, 1.0f), 0,
       va8.swizzle<0, 1>(), va1, va2);
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 1.0f, 1.0f), 0, va8,
       va1.swizzle<0, 1>(), va2);
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 1.0f, 1.0f), 0, va8, va1,
       va2.swizzle<0, 1>());
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 1.0f, 1.0f), 0,
       va8.swizzle<0, 1>(), va1.swizzle<0, 1>(), va2);
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 1.0f, 1.0f), 0,
       va8.swizzle<0, 1>(), va1, va2.swizzle<0, 1>());
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 1.0f, 1.0f), 0, va8,
       va1.swizzle<0, 1>(), va2.swizzle<0, 1>());
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 1.0f, 1.0f), 0,
       va8.swizzle<0, 1>(), va1.swizzle<0, 1>(), va2.swizzle<0, 1>());
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 1.0, 1.0f), 0.00000001,
         va4.swizzle<0, 1>(), va9, va9);
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 1.0, 1.0f), 0.00000001,
         va4, va9.swizzle<0, 1>(), va9);
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 1.0, 1.0f), 0.00000001,
         va4, va9, va9.swizzle<0, 1>());
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 1.0, 1.0f), 0.00000001,
         va4.swizzle<0, 1>(), va9.swizzle<0, 1>(), va9);
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 1.0, 1.0f), 0.00000001,
         va4.swizzle<0, 1>(), va9, va9.swizzle<0, 1>());
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 1.0, 1.0f), 0.00000001,
         va4, va9.swizzle<0, 1>(), va9.swizzle<0, 1>());
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 1.0, 1.0f), 0.00000001,
         va4.swizzle<0, 1>(), va9.swizzle<0, 1>(), va9.swizzle<0, 1>());
  }
  if (dev.has(sycl::aspect::fp16)) {
    TEST(sycl::smoothstep, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0),
         0, va7.swizzle<0, 1, 2>(), va12, va13);
    TEST(sycl::smoothstep, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0),
         0, va7, va12.swizzle<0, 1, 2>(), va13);
    TEST(sycl::smoothstep, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0),
         0, va7, va12, va13.swizzle<0, 1, 2>());
    TEST(sycl::smoothstep, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0),
         0, va7.swizzle<0, 1, 2>(), va12.swizzle<0, 1, 2>(), va13);
    TEST(sycl::smoothstep, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0),
         0, va7.swizzle<0, 1, 2>(), va12, va13.swizzle<0, 1, 2>());
    TEST(sycl::smoothstep, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0),
         0, va7, va12.swizzle<0, 1, 2>(), va13.swizzle<0, 1, 2>());
    TEST(sycl::smoothstep, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0),
         0, va7.swizzle<0, 1, 2>(), va12.swizzle<0, 1, 2>(),
         va13.swizzle<0, 1, 2>());
  }
  TEST(sycl::smoothstep, float, 2, EXPECTED(float, 0.0f, 0.0553936f), 0.0000001,
       2.5f, 6.0f, va3.swizzle<1, 0>());
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 0.0f, 1.0f), 0, 6.0f,
         8.0f, va9);
    TEST(sycl::smoothstep, double, 2, EXPECTED(double, 1.0f, 0.0f), 0, 6.0f,
         8.0f, va9.swizzle<1, 0>());
  }
  // sign swizzled
  TEST(sycl::sign, float, 2, EXPECTED(float, -1.0f, +0.0f), 0,
       va14.swizzle<1, 0>());
  if (dev.has(sycl::aspect::fp64)) {
    TEST(sycl::sign, double, 2, EXPECTED(double, 1.0, -0.0), 0,
         va15.swizzle<1, 0>());
  }
  if (dev.has(sycl::aspect::fp16)) {
    TEST(sycl::sign, sycl::half, 3, EXPECTED(sycl::half, 1.0, 1.0, 1.0), 0,
         va12.swizzle<2, 1, 0>());
  }
#endif // __INTEL_PREVIEW_BREAKING_CHANGES

  return 0;
}
