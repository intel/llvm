//==------- Context.cpp --- Check context constructors and methods ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <sycl/sycl.hpp>

using namespace sycl;

class ContextTest : public ::testing::Test {
public:
  ContextTest() {}

protected:
  void SetUp() override {
    unittest::PiMock::EnsureMockPluginInitialized();

    auto devices = device::get_devices();
    deviceA = devices[0];
    deviceB = devices[(devices.size() > 1 ? 1 : 0)];
  }

protected:
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
