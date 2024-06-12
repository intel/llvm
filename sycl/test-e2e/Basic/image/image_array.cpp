// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: hip
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: gpu

//==------------------- image.cpp - SYCL image basic test -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/accessor_image.hpp>
#include <sycl/detail/core.hpp>

#include <iostream>
#include <vector>

#include "../../helpers.hpp"

#include <sycl/accessor_image.hpp>
#include <sycl/image.hpp>

int main() {
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

  const sycl::range<2> ImgSize(4, 4);

  std::vector<sycl::float4> Img1HostData(ImgSize.size(), {1, 2, 3, 4});

  enum ResType {
    READ_I = 0,
    READ_SAMPLER_F = 0,
    READ_SAMPLER_I = 0,
    GET_RANGE,
    GET_COUNT,
    WRITE1,
    WRITE2,
    ENUM_SIZE
  };

  constexpr int ResBufSize = ENUM_SIZE;
  std::vector<int> ResBufData(ResBufSize, 0);

  {
    sycl::image<2> Img(Img1HostData.data(), ChanOrder, ChanType, ImgSize);

    sycl::buffer<int, 1> ResBuf(ResBufData.data(), {ResBufSize});

    TestQueue Q{sycl::default_selector_v};

    constexpr auto SYCLRead = sycl::access::mode::read;
    constexpr auto SYCLWrite = sycl::access::mode::write;
    constexpr auto SYCLReadWrite = sycl::access::mode::read_write;

    Q.submit([&](sycl::handler &CGH) {
      auto ImgAcc = Img.get_access<sycl::float4, SYCLRead>(CGH);
      sycl::accessor<sycl::float4, /*Dims=*/1, SYCLRead,
                     sycl::access::target::image_array>
          ImgArrayAcc(Img, CGH);

      sycl::sampler Sampler(sycl::coordinate_normalization_mode::unnormalized,
                            sycl::addressing_mode::clamp,
                            sycl::filtering_mode::nearest);

      auto ResAcc = ResBuf.get_access<SYCLReadWrite>(CGH);

      CGH.parallel_for<class Check1>(ImgSize, [=](sycl::item<2> Item) {
        sycl::int2 CoordI{Item[0], Item[1]};
        sycl::float2 CoordF{(Item[0] / (float)ImgSize[0]),
                            (Item[1] / (float)ImgSize[1])};

        // Check that pixels read using image accessor and image_array
        // accessor are the same.
        {
          auto ValRef = ImgAcc.read(sycl::int2{Item[0], Item[1]});
          auto Val = ImgArrayAcc[Item[1]].read((int)Item[0]);

          ResAcc[READ_I] |= sycl::any(sycl::isnotequal(Val, ValRef));
        }

        {
          auto ValRef = ImgAcc.read(CoordI, Sampler);
          auto Val = ImgArrayAcc[CoordI.y()].read((int)CoordI.x(), Sampler);

          ResAcc[READ_SAMPLER_I] |= sycl::any(sycl::isnotequal(Val, ValRef));
        }

        {
          auto ValRef = ImgAcc.read(CoordF, Sampler);
          auto Val = ImgArrayAcc[CoordI.y()].read((float)CoordF.x(), Sampler);

          ResAcc[READ_SAMPLER_F] |= sycl::any(sycl::isnotequal(Val, ValRef));
        }

        // Check that the range and count of 1D image in 1D image array == width
        // of 2d image.

        ResAcc[GET_RANGE] |= sycl::range<1>(ImgAcc.get_range()[0]) !=
                             ImgArrayAcc[CoordI.y()].get_range();

        ResAcc[GET_COUNT] |=
            (ImgAcc.size() / ImgSize[1]) != ImgArrayAcc[CoordI.y()].size();
      });
    });

    Q.submit([&](sycl::handler &CGH) {
      auto ImgAcc = Img.get_access<sycl::float4, SYCLRead>(CGH);
      sycl::accessor<sycl::float4, /*Dims=*/1, SYCLWrite,
                     sycl::access::target::image_array>
          ImgArrayAcc(Img, CGH);

      auto ResAcc = ResBuf.get_access<SYCLReadWrite>(CGH);

      CGH.parallel_for<class Check2>(ImgSize, [=](sycl::item<2> Item) {
        sycl::int2 CoordI{Item[0], Item[1]};

        // CHeck that data written using image array
        const sycl::float4 ValRef{CoordI.x(), 42, 42, CoordI.y()};
        ImgArrayAcc[CoordI.y()].write((int)CoordI.x(), ValRef);
        auto Val = ImgAcc.read(CoordI);

        ResAcc[WRITE1] |= sycl::any(sycl::isnotequal(Val, ValRef));
      });
    });

    Q.submit([&](sycl::handler &CGH) {
      auto ImgAcc = Img.get_access<sycl::float4, SYCLWrite>(CGH);
      sycl::accessor<sycl::float4, /*Dims=*/1, SYCLRead,
                     sycl::access::target::image_array>
          ImgArrayAcc(Img, CGH);

      auto ResAcc = ResBuf.get_access<SYCLReadWrite>(CGH);

      CGH.parallel_for<class Check3>(ImgSize, [=](sycl::item<2> Item) {
        sycl::int2 CoordI{Item[0], Item[1]};

        // CHeck that data read using image array
        const sycl::float4 ValRef{CoordI.x(), 42, 42, CoordI.y()};
        ImgAcc.write(CoordI, ValRef);
        auto Val = ImgArrayAcc[CoordI.y()].read((int)CoordI.x());

        ResAcc[WRITE2] |= sycl::any(sycl::isnotequal(Val, ValRef));
      });
    });
  }

  for (const auto &Elem : ResBufData)
    if (Elem) {
      std::cout << "Failed" << std::endl;
      return 1;
    }

  std::cout << "Success" << std::endl;
  return 0;
}
