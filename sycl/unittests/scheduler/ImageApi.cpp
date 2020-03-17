//==----------------- ImageApi.cpp --- Scheduler unit tests ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MockHandler.hpp"
#include "SchedulerTest.hpp"

#include <CL/sycl.hpp>

#include <gtest/gtest.h>

TEST_F(SchedulerTest, ImageApi) {

  constexpr size_t Size = 256;
  constexpr int Dimensions = 2;

  std::array<sycl::float4, Size> Src;
  std::array<sycl::float4, Size> Dest;
  std::fill(Src.begin(), Src.end(), sycl::float4{1.0f, 2.0f, 3.0f, 4.0f});
  std::fill(Dest.begin(), Dest.end(), sycl::float4{0.0f, 0.0f, 0.0f, 0.0});

  constexpr sycl::image_channel_order ChannelOrder =
      sycl::image_channel_order::rgba;
  constexpr sycl::image_channel_type ChannelType =
      sycl::image_channel_type::fp32;
  const sycl::range<Dimensions> Range{16, 16};

  sycl::image<Dimensions> SrcImg(Src.data(), ChannelOrder, ChannelType, Range);
  sycl::image<Dimensions> DstImg(Dest.data(), ChannelOrder, ChannelType, Range);

  auto CGHLambda = [&](MockHandler &CGH) {
    auto SrcAcc =
        SrcImg.template get_access<sycl::float4, sycl::access::mode::read>(CGH);
    auto DstAcc =
        DstImg.template get_access<sycl::float4, sycl::access::mode::write>(
            CGH);

    EXPECT_EQ(CGH.getKernelAccessors().size(), 2UL);

    sycl::sampler Sampler(sycl::coordinate_normalization_mode::unnormalized,
                          sycl::addressing_mode::clamp,
                          sycl::filtering_mode::nearest);

    CGH.parallel_for<class ImgTest>(Range, [=](sycl::id<Dimensions> ID) {
      sycl::int2 Coords{ID[1], ID[0]};
      sycl::float4 Color = SrcAcc.read(Coords, Sampler);
      Color *= 10.0f;
      DstAcc.write(Coords, Color);
    });
  };

  MockHandler MockCGH(sycl::detail::getSyclObjImpl(MQueue));

  CGHLambda(MockCGH);

  EXPECT_EQ(MockCGH.getKernelArgs().size(), 2UL);

  auto Event = MockCGH.mockFinalize();
  Event.wait();

  sycl::float4 Expected{10.f, 20.f, 30.f, 40.f};

  bool Result = std::all_of(Dest.cbegin(), Dest.cend(),
                            [Expected](const sycl::float4 &Value) -> bool {
                              return sycl::all(sycl::isequal(Value, Expected));
                            });

  EXPECT_TRUE(Result);
}
