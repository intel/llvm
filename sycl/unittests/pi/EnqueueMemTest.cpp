//==---- EnqueueMemTest.cpp --- PI unit tests ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestGetPlugin.hpp"
#include <CL/sycl/detail/pi.hpp>
#include <detail/plugin.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

namespace {
class EnqueueMemTest : public testing::TestWithParam<detail::plugin> {
protected:
  constexpr static size_t _numElementsX = 8;
  constexpr static size_t _numElementsY = 4;

  pi_device _device = nullptr;
  pi_context _context = nullptr;
  pi_queue _queue = nullptr;
  pi_mem _mem = nullptr;

  EnqueueMemTest() = default;

  ~EnqueueMemTest() = default;

  void SetUp() override {

    detail::plugin plugin = GetParam();

    pi_platform platform = nullptr;
    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piPlatformsGet>(
                  1, &platform, nullptr)),
              PI_SUCCESS);

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piDevicesGet>(
                  platform, PI_DEVICE_TYPE_DEFAULT, 1, &_device, nullptr)),
              PI_SUCCESS);

    pi_result result = PI_INVALID_VALUE;
    result = plugin.call_nocheck<detail::PiApiKind::piContextCreate>(
        nullptr, 1u, &_device, nullptr, nullptr, &_context);
    ASSERT_EQ(result, PI_SUCCESS);

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piQueueCreate>(
                  _context, _device, 0, &_queue)),
              PI_SUCCESS);

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemBufferCreate>(
                  _context, PI_MEM_FLAGS_ACCESS_RW,
                  _numElementsX * _numElementsY * sizeof(pi_int32), nullptr,
                  &_mem, nullptr)),
              PI_SUCCESS);
  }

  void TearDown() override {

    detail::plugin plugin = GetParam();

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piMemRelease>(_mem)),
              PI_SUCCESS);
    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piQueueRelease>(_queue)),
              PI_SUCCESS);
    ASSERT_EQ(
        (plugin.call_nocheck<detail::PiApiKind::piContextRelease>(_context)),
        PI_SUCCESS);
  }

  template <typename T> void TestBufferFill(const T &pattern) {

    detail::plugin plugin = GetParam();

    T inValues[_numElementsX] = {};

    for (size_t i = 0; i < _numElementsX; ++i) {
      ASSERT_NE(pattern, inValues[i]);
    }

    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueMemBufferWrite>(
                  _queue, _mem, PI_TRUE, 0, _numElementsX * sizeof(T), inValues,
                  0, nullptr, nullptr)),
              PI_SUCCESS);

    pi_event event;
    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueMemBufferFill>(
                  _queue, _mem, &pattern, sizeof(T), 0, sizeof(inValues), 0,
                  nullptr, &event)),
              PI_SUCCESS);
    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEventsWait>(1, &event)),
              PI_SUCCESS);

    T outValues[_numElementsX] = {};
    ASSERT_EQ((plugin.call_nocheck<detail::PiApiKind::piEnqueueMemBufferRead>(
                  _queue, _mem, PI_TRUE, 0, _numElementsX * sizeof(T),
                  outValues, 0, nullptr, nullptr)),
              PI_SUCCESS);

    for (size_t i = 0; i < _numElementsX; ++i) {
      ASSERT_EQ(pattern, outValues[i]);
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    EnqueueMemTestImpl, EnqueueMemTest,
    testing::ValuesIn(pi::initializeAndRemoveInvalid()),
    [](const testing::TestParamInfo<EnqueueMemTest::ParamType> &info) {
      return pi::GetBackendString(info.param.getBackend());
    });

template <typename T> struct vec4 {
  T x, y, z, w;

  bool operator==(const vec4 &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w;
  }

  bool operator!=(const vec4 &rhs) const { return !(*this == rhs); }
};

template <typename T> struct vec2 {
  T x, y;

  bool operator==(const vec2 &rhs) const { return x == rhs.x && y == rhs.y; }

  bool operator!=(const vec2 &rhs) const { return !(*this == rhs); }
};

TEST_P(EnqueueMemTest, piEnqueueMemBufferFill) {

  TestBufferFill(float{1});
  TestBufferFill(vec2<float>{1, 2});
  TestBufferFill(vec4<float>{1, 2, 3, 4});

  TestBufferFill(uint8_t{1});
  TestBufferFill(vec2<uint8_t>{1, 2});
  TestBufferFill(vec4<uint8_t>{1, 2, 3, 4});

  TestBufferFill(uint16_t{1});
  TestBufferFill(vec2<uint16_t>{1, 2});
  TestBufferFill(vec4<uint16_t>{1, 2, 3, 4});

  TestBufferFill(uint32_t{1});
  TestBufferFill(vec2<uint32_t>{1, 2});
  TestBufferFill(vec4<uint32_t>{1, 2, 3, 4});
}
} // namespace
