//==---- CircularBuffer.cpp ------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <helpers/CommonRedefinitions.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

TEST(DefaultContextTest, DefaultContextTest) {

  sycl::platform Plt1{sycl::default_selector()};
  sycl::unittest::PiMock Mock1{Plt1};
  setupDefaultMockAPIs(Mock1);

  sycl::platform Plt2{sycl::default_selector()};
  sycl::unittest::PiMock Mock2{Plt2};
  setupDefaultMockAPIs(Mock2);

  const sycl::device Dev1 = Plt1.get_devices()[0];
  const sycl::device Dev2 = Plt2.get_devices()[0];

  sycl::queue Queue1{Dev1};
  sycl::queue Queue2{Dev2};

  ASSERT_EQ(Queue1.get_context(), Queue2.get_context());

  ASSERT_EQ(Dev1.get_platform().ext_oneapi_get_default_context(),
            Dev2.get_platform().ext_oneapi_get_default_context());
}
