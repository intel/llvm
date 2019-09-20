// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//==------------------- image_accessor_readsampler.cpp ---------------------==//
//==-----------------image_accessor read API test with sampler--------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

#include <cassert>
#include <iomanip>
#include <iostream>

namespace s = cl::sycl;

template <int unique_number> class kernel_class;

void validateReadData(s::cl_float4 ReadData, s::cl_float4 ExpectedColor) {
  // Maximum difference of 1.5 ULP is allowed.
  s::cl_int4 PixelDataInt = ReadData.template as<s::cl_int4>();
  s::cl_int4 ExpectedDataInt = ExpectedColor.template as<s::cl_int4>();
  s::cl_int4 Diff = ExpectedDataInt - PixelDataInt;
#if DEBUG_OUTPUT
  {
    if (((s::cl_int)Diff.x() <= 1 && (s::cl_int)Diff.x() >= -1) &&
        ((s::cl_int)Diff.y() <= 1 && (s::cl_int)Diff.y() >= -1) &&
        ((s::cl_int)Diff.z() <= 1 && (s::cl_int)Diff.z() >= -1) &&
        ((s::cl_int)Diff.w() <= 1 && (s::cl_int)Diff.w() >= -1)) {
      std::cout << "Read Data is correct within precision: " << std::endl;
    } else {
      std::cout << "Read Data is WRONG/ outside precision: " << std::endl;
    }
    std::cout << "ReadData: \t"
              << std::setprecision(std::numeric_limits<long double>::digits10 +
                                   1)
              << (float)ReadData.x() * 127 << "  " << (float)ReadData.y() * 127
              << "  " << (float)ReadData.z() * 127 << "  "
              << (float)ReadData.w() * 127 << std::endl;

    std::cout << "ExpectedColor: \t" << (float)ExpectedColor.x() * 127 << "  "
              << (float)ExpectedColor.y() * 127 << "  "
              << (float)ExpectedColor.z() * 127 << "  "
              << (float)ExpectedColor.w() * 127 << std::endl;
  }
#else
  {
    assert((s::cl_int)Diff.x() <= 1 && (s::cl_int)Diff.x() >= -1);
    assert((s::cl_int)Diff.y() <= 1 && (s::cl_int)Diff.y() >= -1);
    assert((s::cl_int)Diff.z() <= 1 && (s::cl_int)Diff.z() >= -1);
    assert((s::cl_int)Diff.w() <= 1 && (s::cl_int)Diff.w() >= -1);
  }
#endif
}

template <int i>
void checkReadSampler(char *host_ptr, s::sampler Sampler, s::cl_float4 Coord,
                      s::cl_float4 ExpectedColor) {

  s::cl_float4 ReadData;
  {
    // image with dim = 3
    s::image<3> Img(host_ptr, s::image_channel_order::rgba,
                    s::image_channel_type::snorm_int8, s::range<3>{2, 3, 4});
    s::queue myQueue;
    s::buffer<s::cl_float4, 1> ReadDataBuf(&ReadData, s::range<1>(1));
    myQueue.submit([&](s::handler &cgh) {
      auto ReadAcc = Img.get_access<s::cl_float4, s::access::mode::read>(cgh);
      s::accessor<s::cl_float4, 1, s::access::mode::write> ReadDataBufAcc(
          ReadDataBuf, cgh);

    cgh.single_task<class kernel_class<i>>([=](){
      s::cl_float4 RetColor = ReadAcc.read(Coord, Sampler);
      ReadDataBufAcc[0] = RetColor;
    });
    });
  }
  validateReadData(ReadData, ExpectedColor);
}

void checkSamplerNearest() {

  // create image:
  char host_ptr[100];
  for (int i = 0; i < 100; i++)
    host_ptr[i] = i;

  // Calling only valid configurations.
  // A. coordinate normalization mode::normalized
  // addressing_mode::mirrored_repeat
  {
    s::cl_float4 Coord(0.0f, 1.5f, 2.5f,
                       0.0f); // Out-of-range mirrored_repeat mode
    auto Sampler = s::sampler(s::coordinate_normalization_mode::normalized,
                              s::addressing_mode::mirrored_repeat,
                              s::filtering_mode::nearest);
    checkReadSampler<1>(host_ptr, Sampler, Coord,
                        s::cl_float4((56.0f / 127.0f), (57.0f / 127.0f),
                                     (58.0f / 127.0f),
                                     (59.0f / 127.0f)) /*Expected Value*/);
  }

  // addressing_mode::repeat
  {
    s::cl_float4 Coord(0.0f, 1.5f, 2.5f, 0.0f); // Out-of-range repeat mode
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::repeat, s::filtering_mode::nearest);
    checkReadSampler<2>(host_ptr, Sampler, Coord,
                        s::cl_float4((56.0f / 127.0f), (57.0f / 127.0f),
                                     (58.0f / 127.0f),
                                     (59.0f / 127.0f)) /*Expected Value*/);
  }

  // addressing_mode::clamp_to_edge
  {
    s::cl_float4 Coord(0.0f, 1.5f, 2.5f, 0.0f); // Out-of-range Edge Color
    auto Sampler = s::sampler(s::coordinate_normalization_mode::normalized,
                              s::addressing_mode::clamp_to_edge,
                              s::filtering_mode::nearest);
    checkReadSampler<3>(host_ptr, Sampler, Coord,
                        s::cl_float4((88.0f / 127.0f), (89.0f / 127.0f),
                                     (90.0f / 127.0f),
                                     (91.0f / 127.0f)) /*Expected Value*/);
  }

  // addressing_mode::clamp
  {
    s::cl_float4 Coord(0.0f, 1.5f, 2.5f, 0.0f); // Out-of-range Border Color
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::clamp, s::filtering_mode::nearest);
    checkReadSampler<4>(
        host_ptr, Sampler, Coord,
        s::cl_float4(0.0f, 0.0f, 0.0f, 0.0f) /*Expected Value*/);
  }

  // addressing_mode::none
  {
    s::cl_float4 Coord(0.0f, 0.5f, 0.75f,
                       0.0f); // In-range for consistent return value.
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::none, s::filtering_mode::nearest);
    checkReadSampler<5>(host_ptr, Sampler, Coord,
                        s::cl_float4((80.0f / 127.0f), (81.0f / 127.0f),
                                     (82.0f / 127.0f),
                                     (83.0f / 127.0f)) /*Expected Value*/);
  }

  // B. coordinate_normalization_mode::unnormalized
  // addressing_mode::clamp_to_edge
  {
    s::cl_float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    auto Sampler = s::sampler(s::coordinate_normalization_mode::unnormalized,
                              s::addressing_mode::clamp_to_edge,
                              s::filtering_mode::nearest);
    checkReadSampler<6>(host_ptr, Sampler, Coord,
                        s::cl_float4((56.0f / 127.0f), (57.0f / 127.0f),
                                     (58.0f / 127.0f),
                                     (59.0f / 127.0f)) /*Expected Value*/);
  }

  // addressing_mode::clamp
  {
    s::cl_float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::unnormalized,
                   s::addressing_mode::clamp, s::filtering_mode::nearest);
    checkReadSampler<7>(host_ptr, Sampler, Coord,
                        s::cl_float4((56.0f / 127.0f), (57.0f / 127.0f),
                                     (58.0f / 127.0f),
                                     (59.0f / 127.0f)) /*Expected Value*/);
  }

  // addressing_mode::none
  {
    s::cl_float4 Coord(0.0f, 1.0f, 2.0f,
                       0.0f); // In-range for consistent return value.
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::unnormalized,
                   s::addressing_mode::none, s::filtering_mode::nearest);
    checkReadSampler<8>(host_ptr, Sampler, Coord,
                        s::cl_float4((56.0f / 127.0f), (57.0f / 127.0f),
                                     (58.0f / 127.0f),
                                     (59.0f / 127.0f)) /*Expected Value*/);
  }
}

void checkSamplerLinear(){
    // TODO. Implement this code.
};

int main() {

  checkSamplerNearest();
  // checkSamplerLinear();
}
