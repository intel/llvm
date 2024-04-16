// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out%}

#ifdef _WIN32
#define _USE_MATH_DEFINES // To use math constants
#endif
#include <cmath>

#include "helpers.hpp"

int main() {
  using namespace sycl;

  marray<float, 2> ma1{1.0f, 2.0f};
  marray<float, 2> ma2{1.0f, 2.0f};
  marray<float, 2> ma3{3.0f, 2.0f};
  marray<double, 2> ma4{1.0, 2.0};
  marray<float, 3> ma5{M_PI, M_PI, M_PI};
  marray<double, 3> ma6{M_PI, M_PI, M_PI};
  marray<half, 3> ma7{M_PI, M_PI, M_PI};
  marray<float, 2> ma8{0.3f, 0.6f};
  marray<double, 2> ma9{5.0, 8.0};
  marray<float, 3> ma10{180, 180, 180};
  marray<double, 3> ma11{180, 180, 180};
  marray<half, 3> ma12{180, 180, 180};
  marray<half, 3> ma13{181, 179, 181};
  marray<float, 2> ma14{+0.0f, -0.6f};
  marray<double, 2> ma15{-0.0, 0.6f};

  bool has_fp16 = queue{}.get_device().has(sycl::aspect::fp16);
  bool has_fp64 = queue{}.get_device().has(sycl::aspect::fp64);

  // clamp
  test(F(clamp), marray<float, 2>{1.0f, 2.0f}, ma1, ma2, ma3);
  test(F(clamp), marray<float, 2>{1.0f, 2.0f}, ma1, 1.0f, 3.0f);
  test(has_fp64, F(clamp), marray<double, 2>{1.0, 2.0}, ma4, 1.0, 3.0);
  // degrees
  test(F(degrees), marray<float, 3>{180, 180, 180}, ma5);
  test(has_fp64, F(degrees), marray<double, 3>{180, 180, 180}, ma6);
  test(has_fp16, 0.2, F(degrees), marray<half, 3>{180, 180, 180}, ma7);
  // max
  test(F(max), marray<float, 2>{3.0f, 2.0f}, ma1, ma3);
  test(F(max), marray<float, 2>{1.5f, 2.0f}, ma1, 1.5f);
  test(has_fp64, F(max), marray<double, 2>{1.5, 2.0}, ma4, 1.5);
  // min
  test(F(min), marray<float, 2>{1.0f, 2.0f}, ma1, ma3);
  test(F(min), marray<float, 2>{1.0f, 1.5f}, ma1, 1.5f);
  test(has_fp64, F(min), marray<double, 2>{1.0, 1.5}, ma4, 1.5);
  // mix
  test(F(mix), marray<float, 2>{1.6f, 2.0f}, ma1, ma3, ma8);
  test(F(mix), marray<float, 2>{1.4f, 2.0f}, ma1, ma3, 0.2f);
  test(has_fp64, F(mix), marray<double, 2>{3.0, 5.0}, ma4, ma9, 0.5);
  // radians
  test(F(radians), marray<float, 3>{M_PI, M_PI, M_PI}, ma10);
  test(has_fp64, F(radians), marray<double, 3>{M_PI, M_PI, M_PI}, ma11);
  test(has_fp16, 0.002, F(radians), marray<half, 3>{M_PI, M_PI, M_PI}, ma12);
  // step
  test(F(step), marray<float, 2>{1.0f, 1.0f}, ma1, ma3);
  test(has_fp64, F(step), marray<double, 2>{1.0, 1.0}, ma4, ma9);
  test(has_fp16, F(step), marray<half, 3>{1.0, 0.0, 1.0}, ma12, ma13);
  test(F(step), marray<float, 2>{1.0f, 0.0f}, 2.5f, ma3);
  test(has_fp64, F(step), marray<double, 2>{0.0f, 1.0f}, 6.0f, ma9);
  // smoothstep
  test(F(smoothstep), marray<float, 2>{1.0f, 1.0f}, ma8, ma1, ma2);
  test(has_fp64, 0.00000001, F(smoothstep), marray<double, 2>{1.0, 1.0f}, ma4,
       ma9, ma9);
  test(has_fp16, F(smoothstep), marray<half, 3>{1.0, 1.0, 1.0}, ma7, ma12,
       ma13);
  test(0.0000001, F(smoothstep), marray<float, 2>{0.0553936f, 0.0f}, 2.5f, 6.0f,
       ma3);
  test(has_fp64, F(smoothstep), marray<double, 2>{0.0f, 1.0f}, 6.0f, 8.0f, ma9);
  // sign
  test(F(sign), marray<float, 2>{+0.0f, -1.0f}, ma14);
  test(has_fp64, F(sign), marray<double, 2>{-0.0, 1.0}, ma15);
  test(has_fp16, F(sign), marray<half, 3>{1.0, 1.0, 1.0}, ma12);

  return 0;
}
