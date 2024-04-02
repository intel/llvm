//==-------- SubbufferLargeSize.cpp --- checks buffer sizes more than 2^32 -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>
#include <vector>

#include <helpers/PiMock.hpp>
#include <helpers/TestKernel.hpp>

std::vector<pi_buffer_region_struct> PiMethodData;

inline pi_result redefinedMemBufferPartition(pi_mem, pi_mem_flags,
                                             pi_buffer_create_type,
                                             void *buffer_create_info,
                                             pi_mem *) {
  PiMethodData.push_back(
      *reinterpret_cast<pi_buffer_region>(buffer_create_info));

  return PI_SUCCESS;
}

class LargeBufferSizeTest : public ::testing::Test {
public:
  LargeBufferSizeTest() : Mock{}, Plt{Mock.getPlatform()} {}

protected:
  void SetUp() override {
    Mock.redefineAfter<sycl::detail::PiApiKind::piMemBufferPartition>(
        redefinedMemBufferPartition);
  }

protected:
  sycl::unittest::PiMock Mock;
  sycl::platform Plt;
};

TEST_F(LargeBufferSizeTest, MoreThan32bit) {
  sycl::context Context{Plt};
  sycl::queue Queue{Context, sycl::accelerator_selector{}};

  using DataType = double;
  const size_t IndexStart = 16;
  const size_t SubbufferElemCount1 =
      IndexStart + (static_cast<size_t>(1U) << 32) / sizeof(DataType);
  const size_t SubbufferElemCount2 = 7;

  const size_t SubbufferSize1 = SubbufferElemCount1 * sizeof(DataType);
  const size_t SubbufferSize2 = SubbufferElemCount2 * sizeof(DataType);
  const size_t BufferSize = SubbufferSize1 + SubbufferSize2;

  sycl::buffer<std::uint8_t, 1> Buf(BufferSize);

  size_t OffsetInBytes = 0;
  auto ReinterpretBuf =
      Buf.reinterpret<DataType, 1>(BufferSize / sizeof(DataType));
  sycl::buffer<DataType, 1> Subbuffer1(
      ReinterpretBuf, OffsetInBytes / sizeof(DataType), SubbufferElemCount1);

  OffsetInBytes += SubbufferSize1;
  sycl::buffer<DataType, 1> Subbuffer2(
      ReinterpretBuf, OffsetInBytes / sizeof(DataType), SubbufferElemCount2);

  Queue
      .submit([&](sycl::handler &cgh) {
        auto SubbufferAcc1 =
            Subbuffer1.get_access<sycl::access::mode::read>(cgh);
        auto SubbufferAcc2 =
            Subbuffer2.get_access<sycl::access::mode::read>(cgh);
        cgh.single_task<TestKernel<1>>([=]() {});
      })
      .wait();

  ASSERT_EQ(PiMethodData.size(), 2ul);
  EXPECT_EQ(PiMethodData[0].origin, 0ul);
  EXPECT_EQ(PiMethodData[0].size, SubbufferSize1);
  EXPECT_EQ(PiMethodData[1].origin, OffsetInBytes);
  EXPECT_EQ(PiMethodData[1].size, SubbufferSize2);
}
