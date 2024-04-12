#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

TEST(VectorCross, Double) {
  sycl::double4 Res = sycl::cross(
      sycl::double4(2.5, 3.0, 4.0, 0.0),
      sycl::double4(5.2, 6.0, 7.0, 0.0));

  ASSERT_NEAR(Res.x(), -3.0, 0.001);
  ASSERT_NEAR(Res.y(), 3.3, 0.001);
  ASSERT_NEAR(Res.z(), -0.6, 0.001);
  ASSERT_NEAR(Res.w(), 0.0, 0.001);
}

TEST(VectorOfSize1, Mad) {
  sycl::vec<float, 1> A{1.0f}, B{2.0f}, C{3.0f};
  sycl::vec<float, 1> Res = sycl::mad(A, B, C);
  ASSERT_NEAR(Res.x(), 5.0f, 0.001f);
}

TEST(VectorOfSize1, Clamp) {
  sycl::vec<float, 1> A{1.0f}, B{2.0f}, C{3.0f};
  sycl::vec<float, 1> Res = sycl::clamp(A, B, C);
  ASSERT_FLOAT_EQ(Res.x(), 2.0f);
}
