//==------------ MemChannel.cpp --- check mem_channel property -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/TestKernel.hpp>
#include <helpers/UrMock.hpp>

#include <sycl/accessor.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <detail/buffer_impl.hpp>

constexpr uint32_t DEFAULT_VALUE = 7777;
static uint32_t PassedChannel = DEFAULT_VALUE;

static ur_result_t redefinedMemBufferCreateBefore(void *pParams) {
  auto &Params = *reinterpret_cast<ur_mem_buffer_create_params_t *>(pParams);
  PassedChannel = DEFAULT_VALUE;
  if (!*Params.ppProperties)
    return UR_RESULT_SUCCESS;

  auto Next =
      reinterpret_cast<ur_base_properties_t *>((*Params.ppProperties)->pNext);
  while (Next) {
    if (Next->stype == UR_STRUCTURE_TYPE_BUFFER_CHANNEL_PROPERTIES) {
      auto ChannelProperties =
          reinterpret_cast<ur_buffer_channel_properties_t *>(Next);
      PassedChannel = ChannelProperties->channel;
    }
    Next = reinterpret_cast<ur_base_properties_t *>(Next->pNext);
  }

  return UR_RESULT_SUCCESS;
}

template <bool RetVal>
static ur_result_t redefinedDeviceGetInfoAfter(void *pParams) {
  auto &Params = *reinterpret_cast<ur_device_get_info_params_t *>(pParams);
  if (*Params.ppropName == UR_DEVICE_INFO_MEM_CHANNEL_SUPPORT) {
    if (*Params.ppPropValue)
      *reinterpret_cast<ur_bool_t *>(*Params.ppPropValue) = RetVal;
    if (*Params.ppPropSizeRet)
      **Params.ppPropSizeRet = sizeof(ur_bool_t);
  }
  return UR_RESULT_SUCCESS;
}

class BufferMemChannelTest : public ::testing::Test {
public:
  BufferMemChannelTest() : Mock{}, Plt{sycl::platform()} {}

protected:
  sycl::unittest::UrMock<> Mock;
  sycl::platform Plt;
};

// Test that the mem channel aspect and info query correctly reports true when
// device supports it.
TEST_F(BufferMemChannelTest, MemChannelAspectTrue) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter<true>);

  sycl::device Dev = Plt.get_devices()[0];
  EXPECT_TRUE(Dev.get_info<sycl::info::device::ext_intel_mem_channel>());
  EXPECT_TRUE(Dev.has(sycl::aspect::ext_intel_mem_channel));
}

// Test that the mem channel aspect and info query correctly reports false when
// device supports it.
TEST_F(BufferMemChannelTest, MemChannelAspectFalse) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter<false>);

  sycl::device Dev = Plt.get_devices()[0];
  EXPECT_FALSE(Dev.get_info<sycl::info::device::ext_intel_mem_channel>());
  EXPECT_FALSE(Dev.has(sycl::aspect::ext_intel_mem_channel));
}

// Tests that the right buffer property identifier and values are passed to
// buffer creation.
TEST_F(BufferMemChannelTest, MemChannelProp) {
  mock::getCallbacks().set_after_callback("urDeviceGetInfo",
                                          &redefinedDeviceGetInfoAfter<true>);
  mock::getCallbacks().set_before_callback("urMemBufferCreate",
                                           &redefinedMemBufferCreateBefore);

  sycl::queue Q{Plt.get_devices()[0]};
  sycl::buffer<int, 1> Buf(3, sycl::property::buffer::mem_channel{42});

  ASSERT_TRUE(Buf.has_property<sycl::property::buffer::mem_channel>());
  ASSERT_EQ(
      Buf.get_property<sycl::property::buffer::mem_channel>().get_channel(),
      (uint32_t)42);

  Q.submit([&](sycl::handler &CGH) {
     sycl::accessor Acc{Buf, CGH, sycl::read_write};
     constexpr size_t KS = sizeof(decltype(Acc));
     CGH.single_task<TestKernel<KS>>([=]() { Acc[0] = 4; });
   }).wait();
  EXPECT_EQ(PassedChannel, (uint32_t)42);
}
