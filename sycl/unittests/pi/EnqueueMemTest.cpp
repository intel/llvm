//==---- EnqueueMemTest.cpp --- PI unit tests ------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/pi.hpp>
#include <gtest/gtest.h>

using namespace cl::sycl;

namespace {
class DISABLED_EnqueueMemTest : public ::testing::Test {
protected:
  constexpr static size_t _numElementsX = 8;
  constexpr static size_t _numElementsY = 4;

  pi_device _device = nullptr;
  pi_context _context = nullptr;
  pi_queue _queue = nullptr;
  pi_mem _mem = nullptr;

  DISABLED_EnqueueMemTest() = default;

  ~DISABLED_EnqueueMemTest() = default;

  void SetUp() override {
    detail::pi::initialize();

    pi_platform platform = nullptr;
    ASSERT_EQ((PI_CALL_NOCHECK(piPlatformsGet)(1, &platform, nullptr)), PI_SUCCESS);

    ASSERT_EQ(
        (PI_CALL_NOCHECK(piDevicesGet)(platform, PI_DEVICE_TYPE_GPU, 1, &_device, nullptr)),
        PI_SUCCESS);

    pi_result result = PI_INVALID_VALUE;
    result = PI_CALL_NOCHECK(piContextCreate)(nullptr, 1u, &_device, nullptr, nullptr, &_context);
    ASSERT_EQ(result, PI_SUCCESS);

    ASSERT_EQ((PI_CALL_NOCHECK(piQueueCreate)(_context, _device, 0, &_queue)), PI_SUCCESS);

    ASSERT_EQ((PI_CALL_NOCHECK(piMemBufferCreate)(
        _context, 0, _numElementsX * _numElementsY * sizeof(pi_int32),
        nullptr, &_mem)), PI_SUCCESS);
  }

  void TearDown() override {
    ASSERT_EQ((PI_CALL_NOCHECK(piMemRelease)(_mem)), PI_SUCCESS);
    ASSERT_EQ((PI_CALL_NOCHECK(piQueueRelease)(_queue)), PI_SUCCESS);
    ASSERT_EQ((PI_CALL_NOCHECK(piContextRelease)(_context)), PI_SUCCESS);
  }

  template <typename T> void TestBufferFill(const T &pattern) {

    T inValues[_numElementsX] = {};

    for (size_t i = 0; i < _numElementsX; ++i) {
      ASSERT_NE(pattern, inValues[i]);
    }

    ASSERT_EQ((PI_CALL_NOCHECK(piEnqueueMemBufferWrite)(_queue, _mem, PI_TRUE, 0,
                                          _numElementsX * sizeof(T),
                                          inValues, 0, nullptr, nullptr)),
              PI_SUCCESS);

    ASSERT_EQ((PI_CALL_NOCHECK(piEnqueueMemBufferFill)(_queue, _mem, &pattern, sizeof(T), 0,
                                         sizeof(inValues), 0, nullptr,
                                         nullptr)),
              PI_SUCCESS);

    T outValues[_numElementsX] = {};
    ASSERT_EQ((PI_CALL_NOCHECK(piEnqueueMemBufferRead)(_queue, _mem, PI_TRUE, 0,
                                         _numElementsX * sizeof(T),
                                         outValues, 0, nullptr, nullptr)),
              PI_SUCCESS);

    for (size_t i = 0; i < _numElementsX; ++i) {
      ASSERT_EQ(pattern, outValues[i]);
    }
  }
};

template<typename T>
struct vec4 {
  T x, y, z, w;

  bool operator==(const vec4 &rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z && w == rhs.w;
  }

  bool operator!=(const vec4 &rhs) const {
    return !(*this == rhs);
  }
};

template<typename T>
struct vec2 {
  T x, y;

  bool operator==(const vec2 &rhs) const {
    return x == rhs.x && y == rhs.y;
  }

  bool operator!=(const vec2 &rhs) const {
    return !(*this == rhs);
  }
};

TEST_F(DISABLED_EnqueueMemTest, piEnqueueMemBufferFill) {

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
