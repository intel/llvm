//==------------------------- Properties.cpp -------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Common.hpp"

using namespace sycl;
using namespace sycl::ext::oneapi;

class UnknownProperty : public ::sycl::detail::DataLessProperty<
                            ::sycl::detail::LastKnownDataLessPropKind + 1> {
public:
  UnknownProperty() = default;
};

// Negative tests for properties of graph. Positive checks are included to other
// graph tests verifying exact properties usage.
TEST_F(CommandGraphTest, PropertiesCheckInvalidNode) {
  try {
    auto Node1 = Graph.add(
        [&](sycl::handler &cgh) { cgh.single_task<TestKernel<>>([]() {}); },
        UnknownProperty{});
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}

TEST_F(CommandGraphTest, PropertiesCheckFinalize) {
  try {
    auto ExecGraphUpdatable = Graph.finalize(UnknownProperty{});
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}
