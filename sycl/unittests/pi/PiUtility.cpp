//==--------------------- PiUtility.cpp -- check for internal PI utilities -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/opencl.h>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

namespace {
using namespace sycl;

TEST(PiUtilityTest, CheckPiCastScalar) {
  std::int32_t I = 42;
  std::int64_t L = 1234;
  float F = 31.2f;
  double D = 4321.1234;
  float ItoF = detail::pi::cast<float>(I);
  double LtoD = detail::pi::cast<double>(L);
  std::int32_t FtoI = detail::pi::cast<std::int32_t>(F);
  std::int32_t DtoL = detail::pi::cast<std::int64_t>(D);
  EXPECT_EQ((std::int32_t)F, FtoI);
  EXPECT_EQ((float)I, ItoF);
  EXPECT_EQ((std::int64_t)D, DtoL);
  EXPECT_EQ((double)L, LtoD);
}

TEST(PiUtilityTest, CheckPiCastVector) {
  std::vector<std::int32_t> IVec{6, 1, 5, 2, 3, 4};
  std::vector<float> IVecToFVec = detail::pi::cast<std::vector<float>>(IVec);
  ASSERT_EQ(IVecToFVec.size(), IVec.size());
  for (size_t I = 0; I < IVecToFVec.size(); ++I)
    EXPECT_EQ(IVecToFVec[I], (float)IVec[I]);
}

TEST(PiUtilityTest, CheckPiCastOCLEventVector) {
  // Current special case for vectors of OpenCL vectors. This may change in the
  // future.
  std::vector<cl_event> EVec{(cl_event)0};
  pi_native_handle ENativeHandle = detail::pi::cast<pi_native_handle>(EVec);
  EXPECT_EQ(ENativeHandle, (pi_native_handle)EVec[0]);
}

} // namespace
