//==----------- BufferUrApi.cpp - check buffer-related UR calls -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include <helpers/UrMock.hpp>
#include <sycl/detail/core.hpp>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

using namespace sycl;

namespace {

static size_t NumMemBufferFillCalls = 0;
static size_t NumKernelLaunchWithArgsExpCalls = 0;
static std::vector<ur_buffer_region_t> BufferPartitionRegions;
static std::vector<ur_mem_flags_t> BufferCreateFlags;

inline ur_result_t after_urEnqueueMemBufferFill(void *pParams) {
  (void)pParams;
  ++NumMemBufferFillCalls;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t after_urEnqueueKernelLaunchWithArgsExp(void *pParams) {
  (void)pParams;
  ++NumKernelLaunchWithArgsExpCalls;
  return UR_RESULT_SUCCESS;
}

inline ur_result_t after_urMemBufferPartition(void *pParams) {
  auto Params = *static_cast<ur_mem_buffer_partition_params_t *>(pParams);
  BufferPartitionRegions.push_back(**Params.ppRegion);
  return UR_RESULT_SUCCESS;
}

inline ur_result_t before_urMemBufferCreate(void *pParams) {
  auto Params = *static_cast<ur_mem_buffer_create_params_t *>(pParams);
  BufferCreateFlags.push_back(*Params.pflags);
  return UR_RESULT_SUCCESS;
}

class BufferUrApiTests : public ::testing::Test {
public:
  BufferUrApiTests()
      : Mock{}, Q{context(sycl::platform()), default_selector_v} {}

protected:
  void SetUp() override {
    NumMemBufferFillCalls = 0;
    NumKernelLaunchWithArgsExpCalls = 0;
    BufferPartitionRegions.clear();
    BufferCreateFlags.clear();
  }

  unittest::UrMock<> Mock;
  queue Q;
};

TEST_F(BufferUrApiTests, FillAccessorUsesExpectedUrCommandTypes) {
  mock::getCallbacks().set_after_callback("urEnqueueMemBufferFill",
                                          &after_urEnqueueMemBufferFill);
  mock::getCallbacks().set_after_callback(
      "urEnqueueKernelLaunchWithArgsExp",
      &after_urEnqueueKernelLaunchWithArgsExp);

  constexpr int Width = 32;
  constexpr int Height = 16;
  constexpr int Depth = 8;

  std::vector<float> Data1D(Width, 0.0f);
  std::vector<float> Data2D(Width * Height, 0.0f);
  std::vector<float> Data3D(Width * Height * Depth, 0.0f);
  std::vector<float> Data0D(1, 0.0f);

  buffer<float, 1> Buffer1D(Data1D.data(), range<1>(Width));
  buffer<float, 2> Buffer2D(Data2D.data(), range<2>(Height, Width));
  buffer<float, 3> Buffer3D(Data3D.data(), range<3>(Depth, Height, Width));
  buffer<float, 1> Buffer0D(Data0D.data(), range<1>(1));

  auto ExpectDelta = [&](size_t ExpectedFillDelta, size_t ExpectedKernelDelta,
                         auto &&SubmitWork) {
    const size_t FillBefore = NumMemBufferFillCalls;
    const size_t KernelBefore = NumKernelLaunchWithArgsExpCalls;
    SubmitWork();
    EXPECT_EQ(NumMemBufferFillCalls, FillBefore + ExpectedFillDelta);
    EXPECT_EQ(NumKernelLaunchWithArgsExpCalls,
              KernelBefore + ExpectedKernelDelta);
  };

  // 1D full accessor -> urEnqueueMemBufferFill
  ExpectDelta(1, 0, [&]() {
    Q.submit([&](handler &CGH) {
       auto Acc = Buffer1D.get_access<access::mode::write>(CGH);
       CGH.fill(Acc, float{1});
     }).wait();
  });

  // 1D ranged accessor -> urEnqueueMemBufferFill
  ExpectDelta(1, 0, [&]() {
    Q.submit([&](handler &CGH) {
       auto Acc =
           Buffer1D.get_access<access::mode::write>(CGH, id<1>{4}, range<1>{2});
       CGH.fill(Acc, float{2});
     }).wait();
  });

  // 2D full accessor -> urEnqueueMemBufferFill
  ExpectDelta(1, 0, [&]() {
    Q.submit([&](handler &CGH) {
       auto Acc = Buffer2D.get_access<access::mode::write>(CGH);
       CGH.fill(Acc, float{3});
     }).wait();
  });

  // 2D ranged accessor -> urEnqueueKernelLaunchWithArgsExp
  ExpectDelta(0, 1, [&]() {
    Q.submit([&](handler &CGH) {
       auto Acc = Buffer2D.get_access<access::mode::write>(
           CGH, id<2>{8, 12}, range<2>{2, 2});
       CGH.fill(Acc, float{4});
     }).wait();
  });

  // 3D full accessor -> urEnqueueMemBufferFill
  ExpectDelta(1, 0, [&]() {
    Q.submit([&](handler &CGH) {
       auto Acc = Buffer3D.get_access<access::mode::write>(CGH);
       CGH.fill(Acc, float{5});
     }).wait();
  });

  // 3D ranged accessor -> urEnqueueKernelLaunchWithArgsExp
  ExpectDelta(0, 1, [&]() {
    Q.submit([&](handler &CGH) {
       auto Acc = Buffer3D.get_access<access::mode::write>(
           CGH, id<3>{4, 8, 12}, range<3>{3, 3, 3});
       CGH.fill(Acc, float{6});
     }).wait();
  });

  // 0D accessor -> urEnqueueMemBufferFill
  ExpectDelta(1, 0, [&]() {
    Q.submit([&](handler &CGH) {
       accessor<float, 0, access::mode::write> Acc0(Buffer0D, CGH);
       CGH.fill(Acc0, float{7});
     }).wait();
  });
}

TEST_F(BufferUrApiTests, SubbufferOverlapUsesExpectedUrPartitionRegions) {
  mock::getCallbacks().set_after_callback("urMemBufferPartition",
                                          &after_urMemBufferPartition);

  buffer<int, 1> BaseBuf{1024};
  id<1> StartOffset{64};
  constexpr size_t Size = 16;

  buffer<int, 1> Sub1{BaseBuf, StartOffset, range<1>{Size}};
  buffer<int, 1> Sub2{BaseBuf, StartOffset, range<1>{Size * 2}};

  std::array<int, Size> Out1{};
  std::array<int, Size * 2> Out2{};

  Q.submit([&](handler &CGH) {
     auto Acc = Sub1.get_access<access::mode::read>(CGH);
     CGH.copy(Acc, Out1.data());
   }).wait();

  Q.submit([&](handler &CGH) {
     auto Acc = Sub2.get_access<access::mode::read>(CGH);
     CGH.copy(Acc, Out2.data());
   }).wait();

  ASSERT_EQ(BufferPartitionRegions.size(), size_t{2});
  EXPECT_EQ(BufferPartitionRegions[0].origin, size_t{256});
  EXPECT_EQ(BufferPartitionRegions[0].size, size_t{64});
  EXPECT_EQ(BufferPartitionRegions[1].origin, size_t{256});
  EXPECT_EQ(BufferPartitionRegions[1].size, size_t{128});
}

TEST_F(BufferUrApiTests, NativeBufferCreationUsesHostPointerFlag) {
  mock::getCallbacks().set_before_callback("urMemBufferCreate",
                                           &before_urMemBufferCreate);

  const int BufVal = 42;
  buffer<int, 1> Buf{&BufVal, range<1>{1}};

  {
    // This write access to a const user pointer forces creation of a
    // read-write host allocation.
    host_accessor BufAcc(Buf, write_only);
    (void)BufAcc;
  }

  BufferCreateFlags.clear();

  int Out = 0;
  Q.submit([&](handler &CGH) {
     auto BufAcc = Buf.get_access<access::mode::read>(CGH);
     CGH.copy(BufAcc, &Out);
   }).wait();

  ASSERT_EQ(BufferCreateFlags.size(), size_t{1});
  EXPECT_EQ(BufferCreateFlags[0] & UR_MEM_FLAG_USE_HOST_POINTER,
            UR_MEM_FLAG_USE_HOST_POINTER);
}

TEST_F(BufferUrApiTests, AllocPinnedHostMemoryUsesAllocHostPointerFlag) {
  mock::getCallbacks().set_before_callback("urMemBufferCreate",
                                           &before_urMemBufferCreate);

  int Data[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  buffer<int, 1> A(Data, range<1>(10), {property::buffer::use_host_ptr()});
  buffer<int, 1> B(
      range<1>(10),
      {ext::oneapi::property::buffer::use_pinned_host_memory()});

  BufferCreateFlags.clear();

  Q.submit([&](handler &CGH) {
     auto Src = B.get_access<access::mode::read>(CGH);
     auto Dst = A.get_access<access::mode::write>(CGH);
     CGH.copy(Src, Dst);
   }).wait();

  ASSERT_EQ(BufferCreateFlags.size(), size_t{2});
  EXPECT_TRUE(std::any_of(BufferCreateFlags.begin(), BufferCreateFlags.end(),
                          [](ur_mem_flags_t Flags) {
                            return (Flags & UR_MEM_FLAG_ALLOC_HOST_POINTER) !=
                                   0;
                          }));
}

TEST_F(BufferUrApiTests, UsePinnedHostMemoryUsesAllocHostPointerFlag) {
  mock::getCallbacks().set_before_callback("urMemBufferCreate",
                                           &before_urMemBufferCreate);

  range<1> N{1};
  buffer<int, 1> Buf(
      N, {ext::oneapi::property::buffer::use_pinned_host_memory()});

  ASSERT_TRUE(
      Buf.has_property<ext::oneapi::property::buffer::use_pinned_host_memory>());

  BufferCreateFlags.clear();

  Q.submit([&](handler &CGH) {
     auto Acc = Buf.get_access<access::mode::write>(CGH);
     CGH.fill(Acc, 7);
   }).wait();

  ASSERT_EQ(BufferCreateFlags.size(), size_t{1});
  EXPECT_EQ(BufferCreateFlags[0] & UR_MEM_FLAG_ALLOC_HOST_POINTER,
            UR_MEM_FLAG_ALLOC_HOST_POINTER);
}

TEST_F(BufferUrApiTests, UsePinnedHostMemoryWithHostPointerThrows) {
  int Data = 0;
  try {
    buffer<int, 1> Buf(
        &Data, range<1>{1},
        {ext::oneapi::property::buffer::use_pinned_host_memory()});
    (void)Buf;
    FAIL() << "Expected exception was not thrown";
  } catch (sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::errc::invalid);
    EXPECT_NE(std::string(E.what()).find(
                  "The use_pinned_host_memory cannot be used with host "
                  "pointer"),
              std::string::npos);
  }
}

} // namespace
