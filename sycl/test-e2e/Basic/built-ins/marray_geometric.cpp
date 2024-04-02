// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: %if preview-breaking-changes-supported %{ %{build} -fpreview-breaking-changes -o %t_preview.out %}
// RUN: %if preview-breaking-changes-supported %{ %{run} %t_preview.out %}

#include "helpers.hpp"

int main() {
  using namespace sycl;

  // clang-format off
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  marray<half, 2> MHalfD2   = {1.f, 2.f};
  marray<half, 2> MHalfD2_2 = {3.f, 5.f};
  marray<half, 3> MHalfD3   = {1.f, 2.f, 3.f};
  marray<half, 3> MHalfD3_2 = {1.f, 5.f, 7.f};
  marray<half, 4> MHalfD4   = {1.f, 2.f, 3.f, 4.f};
  marray<half, 4> MHalfD4_2 = {1.f, 5.f, 7.f, 4.f};
#endif

  marray<float, 2> MFloatD2   = {1.f, 2.f};
  marray<float, 2> MFloatD2_2 = {3.f, 5.f};
  marray<float, 3> MFloatD3   = {1.f, 2.f, 3.f};
  marray<float, 3> MFloatD3_2 = {1.f, 5.f, 7.f};
  marray<float, 4> MFloatD4   = {1.f, 2.f, 3.f, 4.f};
  marray<float, 4> MFloatD4_2 = {1.f, 5.f, 7.f, 4.f};

  marray<double, 2> MDoubleD2   = {1.0, 2.0};
  marray<double, 2> MDoubleD2_2 = {3.0, 5.0};
  marray<double, 3> MDoubleD3   = {1.0, 2.0, 3.0};
  marray<double, 3> MDoubleD3_2 = {1.0, 5.0, 7.0};
  marray<double, 4> MDoubleD4   = {1.0, 2.0, 3.0, 4.0};
  marray<double, 4> MDoubleD4_2 = {1.0, 5.0, 7.0, 4.0};
  // clang-format on

  bool has_fp16 = queue{}.get_device().has(sycl::aspect::fp16);
  bool has_fp64 = queue{}.get_device().has(sycl::aspect::fp64);

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  test(has_fp16, F(cross), marray<half, 3>{-1.f, -4.f, 3.f}, MHalfD3,
       MHalfD3_2);
  test(has_fp16, F(cross), marray<half, 4>{-1.f, -4.f, 3.f, 0.f}, MHalfD4,
       MHalfD4_2);
#endif

  test(F(cross), marray<float, 3>{-1.f, -4.f, 3.f}, MFloatD3, MFloatD3_2);
  test(F(cross), marray<float, 4>{-1.f, -4.f, 3.f, 0.f}, MFloatD4, MFloatD4_2);
  test(has_fp64, F(cross), marray<double, 3>{-1.f, -4.f, 3.f}, MDoubleD3,
       MDoubleD3_2);
  test(has_fp64, F(cross), marray<double, 4>{-1.f, -4.f, 3.f, 0.f}, MDoubleD4,
       MDoubleD4_2);

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  test(has_fp16, F(dot), half{13.f}, MHalfD2, MHalfD2_2);
  test(has_fp16, F(dot), half{32.f}, MHalfD3, MHalfD3_2);
  test(has_fp16, F(dot), half{48.f}, MHalfD4, MHalfD4_2);
#endif

  test(F(dot), float{13.f}, MFloatD2, MFloatD2_2);
  test(F(dot), float{32.f}, MFloatD3, MFloatD3_2);
  test(F(dot), float{48.f}, MFloatD4, MFloatD4_2);
  test(has_fp64, F(dot), double{13.0}, MDoubleD2, MDoubleD2_2);
  test(has_fp64, F(dot), double{32.0}, MDoubleD3, MDoubleD3_2);
  test(has_fp64, F(dot), double{48.0}, MDoubleD4, MDoubleD4_2);

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  test(has_fp16, 1e-3, F(length), half{2.236f}, MHalfD2);
  test(has_fp16, 1e-3, F(length), half{3.742f}, MHalfD3);
  test(has_fp16, 1e-3, F(length), half{5.477f}, MHalfD4);
#endif

  test(1e-6, F(length), float{2.236068f}, MFloatD2);
  test(1e-6, F(length), float{3.741657f}, MFloatD3);
  test(1e-6, F(length), float{5.477225f}, MFloatD4);
  test(has_fp64, 1e-6, F(length), double{2.236068}, MDoubleD2);
  test(has_fp64, 1e-6, F(length), double{3.741657}, MDoubleD3);
  test(has_fp64, 1e-6, F(length), double{5.477225}, MDoubleD4);

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  test(has_fp16, 1e-3, F(distance), half{3.605f}, MHalfD2, MHalfD2_2);
  test(has_fp16, F(distance), half{5.f}, MHalfD3, MHalfD3_2);
  test(has_fp16, F(distance), half{5.f}, MHalfD4, MHalfD4_2);
#endif

  test(1e-6, F(distance), float{3.605551f}, MFloatD2, MFloatD2_2);
  test(F(distance), float{5.f}, MFloatD3, MFloatD3_2);
  test(F(distance), float{5.f}, MFloatD4, MFloatD4_2);
  test(has_fp64, 1e-6, F(distance), double{3.605551}, MDoubleD2, MDoubleD2_2);
  test(has_fp64, F(distance), double{5.0}, MDoubleD3, MDoubleD3_2);
  test(has_fp64, F(distance), double{5.0}, MDoubleD4, MDoubleD4_2);

#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  test(has_fp16, 1e-6, F(normalize), marray<half, 2>{0.447213f, 0.894427f},
       MHalfD2);
  test(has_fp16, 1e-6, F(normalize),
       marray<half, 3>{0.267261f, 0.534522f, 0.801784f}, MHalfD3);
  test(has_fp16, 1e-6, F(normalize),
       marray<half, 4>{0.182574f, 0.365148f, 0.547723f, 0.730297f}, MHalfD4);
#endif

  test(1e-6, F(normalize), marray<float, 2>{0.447213f, 0.894427f}, MFloatD2);
  test(1e-6, F(normalize), marray<float, 3>{0.267261f, 0.534522f, 0.801784f},
       MFloatD3);
  test(1e-6, F(normalize),
       marray<float, 4>{0.182574f, 0.365148f, 0.547723f, 0.730297f}, MFloatD4);
  test(has_fp64, 1e-6, F(normalize), marray<double, 2>{0.447213, 0.894427},
       MDoubleD2);
  test(has_fp64, 1e-6, F(normalize),
       marray<double, 3>{0.267261, 0.534522, 0.801784}, MDoubleD3);
  test(has_fp64, 1e-6, F(normalize),
       marray<double, 4>{0.182574, 0.365148, 0.547723, 0.730297}, MDoubleD4);

  test(1e-6, F(fast_distance), float{3.605551f}, MFloatD2, MFloatD2_2);
  test(F(fast_distance), float{5.f}, MFloatD3, MFloatD3_2);
  test(F(fast_distance), float{5.f}, MFloatD4, MFloatD4_2);

  test(1e-6, F(fast_length), float{2.236068f}, MFloatD2);
  test(1e-6, F(fast_length), float{3.741657f}, MFloatD3);
  test(1e-6, F(fast_length), float{5.477225f}, MFloatD4);

  test(1e-3, F(fast_normalize), marray<float, 2>{0.447213f, 0.894427f},
       MFloatD2);
  test(1e-3, F(fast_normalize),
       marray<float, 3>{0.267261f, 0.534522f, 0.801784f}, MFloatD3);
  test(1e-3, F(fast_normalize),
       marray<float, 4>{0.182574f, 0.365148f, 0.547723f, 0.730297f}, MFloatD4);

  return 0;
}
