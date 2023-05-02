//==------------ MemChannel.cpp --- check mem_channel property -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

#include <sycl/accessor.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <detail/buffer_impl.hpp>

constexpr uint32_t DEFAULT_VALUE = 7777;
static uint32_t PassedChannel = DEFAULT_VALUE;

static pi_result
redefinedMemBufferCreateBefore(pi_context, pi_mem_flags, size_t size, void *,
                               pi_mem *, const pi_mem_properties *properties) {
  PassedChannel = DEFAULT_VALUE;
  if (!properties)
    return PI_SUCCESS;

  // properties must ended by 0
  size_t I = 0;
  while (properties[I] != 0) {
    if (properties[I] == PI_MEM_PROPERTIES_CHANNEL) {
      PassedChannel = properties[I + 1];
      break;
    }
    I += 2;
  }

  return PI_SUCCESS;
}

template <bool RetVal>
static pi_result
redefinedDeviceGetInfoAfter(pi_device device, pi_device_info param_name,
                            size_t param_value_size, void *param_value,
                            size_t *param_value_size_ret) {
  if (param_name == PI_EXT_INTEL_DEVICE_INFO_MEM_CHANNEL_SUPPORT) {
    if (param_value)
      *reinterpret_cast<pi_bool *>(param_value) = RetVal;
    if (param_value_size_ret)
      *param_value_size_ret = sizeof(pi_bool);
  }
  return PI_SUCCESS;
}

class BufferMemChannelTest : public ::testing::Test {
public:
  BufferMemChannelTest() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {}

protected:
  sycl::unittest::PiMock Mock;
  sycl::platform Plt;
};

// Test that the mem channel aspect and info query correctly reports true when
// device supports it.
TEST_F(BufferMemChannelTest, MemChannelAspectTrue) {
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter<true>);

  sycl::device Dev = Plt.get_devices()[0];
  EXPECT_TRUE(Dev.get_info<sycl::info::device::ext_intel_mem_channel>());
  EXPECT_TRUE(Dev.has(sycl::aspect::ext_intel_mem_channel));
}

// Test that the mem channel aspect and info query correctly reports false when
// device supports it.
TEST_F(BufferMemChannelTest, MemChannelAspectFalse) {
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter<false>);

  sycl::device Dev = Plt.get_devices()[0];
  EXPECT_FALSE(Dev.get_info<sycl::info::device::ext_intel_mem_channel>());
  EXPECT_FALSE(Dev.has(sycl::aspect::ext_intel_mem_channel));
}

// Tests that the right buffer property identifier and values are passed to
// buffer creation.
TEST_F(BufferMemChannelTest, MemChannelProp) {
  Mock.redefineAfter<sycl::detail::PiApiKind::piDeviceGetInfo>(
      redefinedDeviceGetInfoAfter<true>);
  Mock.redefineBefore<sycl::detail::PiApiKind::piMemBufferCreate>(
      redefinedMemBufferCreateBefore);

  sycl::queue Q{Plt.get_devices()[0]};
  sycl::buffer<int, 1> Buf(3, sycl::property::buffer::mem_channel{42});
  Q.submit([&](sycl::handler &CGH) {
     sycl::accessor Acc{Buf, CGH, sycl::read_write};
     constexpr size_t KS = sizeof(decltype(Acc));
     CGH.single_task<TestKernel<KS>>([=]() { Acc[0] = 4; });
   }).wait();
  EXPECT_EQ(PassedChannel, (uint32_t)42);
}
