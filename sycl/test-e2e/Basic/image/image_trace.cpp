// REQUIRES: aspect-ext_intel_legacy_image
//
// l0 may use createUrMemFromZeImage instead of the usual urMemImageCreate
// depending on the arch
// UNSUPPORTED: level_zero
//
// RUN: %{build} -o %t.out
// RUN: env SYCL_UR_TRACE=-1 %{run} %t.out | FileCheck %s

//==------------------- image_trace.cpp - SYCL image trace test ------------==//
//
// Ensures the correct params are being passed to urMemImageCreate
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/accessor_image.hpp>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>

#include <iostream>
#include <vector>

#include "../../helpers.hpp"

#include <sycl/accessor_image.hpp>
#include <sycl/image.hpp>

int main() {
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<2> Img1Size(4, 4);
  const sycl::range<2> Img2Size(4, 4);

  std::vector<sycl::float4> Img1HostData(Img1Size.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> Img2HostData(Img2Size.size(), {0, 0, 0, 0});

  {
    sycl::image<2> Img1(Img1HostData.data(), ChanOrder, ChanType, Img1Size);
    sycl::image<2> Img2(Img2HostData.data(), ChanOrder, ChanType, Img2Size);
    TestQueue Q{sycl::default_selector_v};
    Q.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img1.get_access<sycl::float4, SYCLRead>(CGH);
      auto Img2Acc = Img2.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopy>(Img1Size, [=](sycl::item<2> Item) {
        sycl::float4 Data = Img1Acc.read(sycl::int2{Item[0], Item[1]});
        Img2Acc.write(sycl::int2{Item[0], Item[1]}, Data);
      });
      // CHECK: urMemImageCreate({{.*}} .flags = UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_USE_HOST_POINTER, .pImageFormat = {{.*}} ((struct ur_image_format_t){.channelOrder = UR_IMAGE_CHANNEL_ORDER_RGBA, .channelType = UR_IMAGE_CHANNEL_TYPE_FLOAT}), .pImageDesc = {{.*}} ((struct ur_image_desc_t){.stype = UR_STRUCTURE_TYPE_IMAGE_DESC, .pNext = nullptr, .type = UR_MEM_TYPE_IMAGE2D, .width = 4, .height = 4, .depth = 1, .arraySize = 0, .rowPitch = 64, .slicePitch = 256, .numMipLevel = 0, .numSamples = 0}), .pHost = {{.*}}, .phMem = {{.*}}) -> UR_RESULT_SUCCESS;
      // CHECK: urMemImageCreate({{.*}} .flags = UR_MEM_FLAG_READ_WRITE | UR_MEM_FLAG_USE_HOST_POINTER, .pImageFormat = {{.*}} ((struct ur_image_format_t){.channelOrder = UR_IMAGE_CHANNEL_ORDER_RGBA, .channelType = UR_IMAGE_CHANNEL_TYPE_FLOAT}), .pImageDesc = {{.*}} ((struct ur_image_desc_t){.stype = UR_STRUCTURE_TYPE_IMAGE_DESC, .pNext = nullptr, .type = UR_MEM_TYPE_IMAGE2D, .width = 4, .height = 4, .depth = 1, .arraySize = 0, .rowPitch = 64, .slicePitch = 256, .numMipLevel = 0, .numSamples = 0}), {{.*}}) -> UR_RESULT_SUCCESS;
    });
  }

  for (int X = 0; X < Img2Size[0]; ++X)
    for (int Y = 0; Y < Img2Size[1]; ++Y) {
      sycl::float4 Vec1 = Img1HostData[X * Img1Size[1] + Y];
      sycl::float4 Vec2 = Img2HostData[X * Img2Size[1] + Y];

      if (sycl::any(sycl::isnotequal(Vec1, Vec2))) {
        std::cerr << "Failed" << std::endl;
        std::cerr << "Element [ " << X << ", " << Y << " ]" << std::endl;
        std::cerr << "Expected: " << printableVec(Vec1) << std::endl;
        std::cerr << " Got    : " << printableVec(Vec2) << std::endl;
        return 1;
      }
    }
}
