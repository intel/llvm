//==------- Context.cpp --- Check context constructors and methods ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/UrMock.hpp>

#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

using namespace sycl;

class ContextTest : public ::testing::Test {
public:
  // NOTE: Devices must be initialized as part of the constructor to prevent
  //       default initialization, in case no devices are available before mock
  //       has been initialized.
  ContextTest()
      : mock{}, deviceA{sycl::platform().get_devices().front()},
        deviceB{sycl::platform().get_devices().back()} {}

protected:
  unittest::UrMock<> mock;
  device deviceA, deviceB;
};

TEST_F(ContextTest, MoveConstructor) {
  context Context(deviceA);
  size_t hash = std::hash<context>()(Context);
  context MovedContext(std::move(Context));
  ASSERT_EQ(hash, std::hash<context>()(MovedContext));
}

TEST_F(ContextTest, MoveAssignmentConstructor) {
  context Context(deviceA);
  size_t hash = std::hash<context>()(Context);
  context WillMovedContext(deviceB);
  WillMovedContext = std::move(Context);
  ASSERT_EQ(hash, std::hash<context>()(WillMovedContext));
}

TEST_F(ContextTest, CopyConstructor) {
  context Context(deviceA);
  size_t hash = std::hash<context>()(Context);
  context ContextCopy(Context);
  ASSERT_EQ(hash, std::hash<context>()(Context));
  ASSERT_EQ(hash, std::hash<context>()(ContextCopy));
  ASSERT_EQ(Context, ContextCopy);
}

TEST_F(ContextTest, CopyAssignmentOperator) {
  context Context(deviceA);
  size_t hash = std::hash<context>()(Context);
  context WillContextCopy(deviceB);
  WillContextCopy = Context;
  ASSERT_EQ(hash, std::hash<context>()(Context));
  ASSERT_EQ(hash, std::hash<context>()(WillContextCopy));
  ASSERT_EQ(Context, WillContextCopy);
}

TEST_F(ContextTest, Properties) {
  try {
    sycl::context Context(sycl::property::queue::in_order{});
  } catch (sycl::exception &e) {
    EXPECT_EQ(e.code(), sycl::errc::invalid);
    EXPECT_STREQ(e.what(), "The property list contains property unsupported "
                           "for the current object");
    return;
  }

  FAIL() << "Test must exit in exception handler. Exception is not thrown.";
}
