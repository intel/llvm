// UNSUPPORTED: hip || gpu-intel-pvc
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==------------------- image.cpp - SYCL image basic test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <iostream>
#include <vector>

#include "../../helpers.hpp"

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

  {
    const sycl::range<1> ImgPitch(4 * 4 * 4 * 2);
    sycl::image<2> Img(ChanOrder, ChanType, Img1Size, ImgPitch);
    TestQueue Q{sycl::default_selector_v};
    Q.submit([&](sycl::handler &CGH) {
      auto ImgAcc = Img.get_access<sycl::float4, SYCLRead>(CGH);
      CGH.single_task<class EmptyKernel>([=]() { ImgAcc.get_range(); });
    });
  }

  {
    const sycl::range<1> ImgPitch(4 * 4 * 4 * 2);
    sycl::image<2> Img(Img1HostData.data(), ChanOrder, ChanType, Img1Size,
                       ImgPitch);
    TestQueue Q{sycl::default_selector_v};
    Q.submit([&](sycl::handler &CGH) {
      auto ImgAcc = Img.get_access<sycl::float4, SYCLRead>(CGH);
      CGH.single_task<class ConstTestPitch>([=] { ImgAcc.get_range(); });
    });
  }

  // check image accessor
  {
    sycl::queue queue;

    constexpr int dims = 1;

    using data_img = sycl::cl_float4;
    constexpr auto mode_img = sycl::access::mode::read;
    constexpr auto target_img = sycl::target::image;
    const auto range_img = sycl::range<dims>(3);
    auto image = sycl::image<dims>(sycl::image_channel_order::rgba,
                                   sycl::image_channel_type::fp32, range_img);

    {
      queue.submit([&](sycl::handler &cgh) {
        auto properties = sycl::property_list{};

        auto acc_img_p = sycl::accessor<data_img, dims, mode_img, target_img>(
            image, cgh, properties);
        auto acc_img =
            sycl::accessor<data_img, dims, mode_img, target_img>(image, cgh);

        cgh.single_task<class loc_img_acc>([=]() {});
      });
    }
  }

  // image with write accessor to it in kernel
  {
    int NX = 32;
    int NY = 32;

    sycl::image<2> Img(sycl::image_channel_order::rgba,
                       sycl::image_channel_type::fp32, sycl::range<2>(NX, NY));

    sycl::queue Q;
    Q.submit([&](sycl::handler &CGH) {
       auto ImgAcc =
           Img.get_access<sycl::float4, sycl::access::mode::write>(CGH);

       sycl::nd_range<2> Rng(sycl::range<2>(NX, NY), sycl::range<2>(16, 16));

       CGH.parallel_for<class sample>(Rng, [=](sycl::nd_item<2> Item) {
         sycl::id<2> Idx = Item.get_global_id();
         sycl::float4 C(0.5f, 0.5f, 0.2f, 1.0f);
         ImgAcc.write(sycl::int2(Idx[0], Idx[1]), C);
       });
     }).wait_and_throw();
  }

  std::cout << "Success" << std::endl;
  return 0;
}
