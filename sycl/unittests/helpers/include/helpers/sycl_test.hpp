//==------------- sycl_test.hpp --- SYCL unit test wrapper -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "helpers/PiMock.hpp"

#include <CL/sycl/detail/common.hpp>
#include <detail/context_impl.hpp>
#include <detail/global_handler.hpp>

#include <gtest/gtest.h>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace unittest {

template <typename T> class SYCLUnitTest : public ::testing::Test {
protected:
  void SetUp() override {
    for (auto &Ctx :
         detail::GlobalHandler::instance().getPlatformToDefaultContextCache()) {
      Ctx.second->getKernelProgramCache().reset();
    }
    hijackPlugins();
    setupDefaultMockAPIs();
  }
  void TearDown() override { resetMockAPIs(); }
};

} // namespace unittest
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

#define SYCL_TEST(suite, name)                                                 \
  class suite;                                                                 \
  using SYCL_##suite = sycl::unittest::SYCLUnitTest<suite>;                    \
  TEST_F(SYCL_##suite, name)
