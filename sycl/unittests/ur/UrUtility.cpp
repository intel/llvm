//==--------------------- UrUtility.cpp -- check for internal ur utilities -==//
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

TEST(UrUtilityTest, CheckUrCastOCLEventVector) {
  // Current special case for vectors of OpenCL vectors. This may change in the
  // future.
  std::vector<cl_event> EVec{(cl_event)0};
  ur_native_handle_t ENativeHandle = detail::ur::cast<ur_native_handle_t>(EVec);
  EXPECT_EQ(ENativeHandle, (ur_native_handle_t)EVec[0]);
}

} // namespace
