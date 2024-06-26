// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: cuda || hip || (windows && level_zero)
// unsupported on windows (level-zero) due to fail of Jenkins/pre-ci-windows
// CUDA cannot support SYCL 1.2.1 images.
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
//==-----------------image_accessor read API test with sampler--------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/accessor_image.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/image.hpp>

#include <cassert>
#include <iomanip>
#include <iostream>

namespace s = sycl;

template <int unique_number> class kernel_class;

void validateReadData(s::float4 ReadData, s::float4 ExpectedColor,
                      int precision = 1) {
  // Maximum difference of 1.5 ULP is allowed when precision = 1.
  s::int4 PixelDataInt = ReadData.template as<s::int4>();
  s::int4 ExpectedDataInt = ExpectedColor.template as<s::int4>();
  s::int4 Diff = ExpectedDataInt - PixelDataInt;
  int DataIsCorrect = s::all((Diff <= precision) && (Diff >= (-precision)));
#if DEBUG_OUTPUT
  {
    if (DataIsCorrect) {
      std::cout << "Read Data is correct within precision: " << std::endl;
    } else {
      std::cout << "Read Data is WRONG/ outside precision: " << std::endl;
    }
    std::cout << "ReadData: " << std::endl;
    (ReadData * 127).dump();
    std::cout << "ExpectedColor: " << std::endl;
    (ExpectedColor * 127).dump();
    std::cout << "Diff: " << std::endl;
    Diff.dump();
  }
#else
  {
    assert(DataIsCorrect);
  }
#endif
}

template <int i>
void checkReadSampler(char *host_ptr, s::sampler Sampler, s::float4 Coord,
                      s::float4 ExpectedColor, int precision = 1) {

  s::float4 ReadData;
  {
    // image with dim = 3
    s::image<3> Img(host_ptr, s::image_channel_order::rgba,
                    s::image_channel_type::snorm_int8, s::range<3>{2, 3, 4});
    s::queue myQueue;
    s::buffer<s::float4, 1> ReadDataBuf(&ReadData, s::range<1>(1));
    myQueue.submit([&](s::handler &cgh) {
      auto ReadAcc = Img.get_access<s::float4, s::access::mode::read>(cgh);
      s::accessor<s::float4, 1, s::access::mode::write> ReadDataBufAcc(
          ReadDataBuf, cgh);

      cgh.single_task<class kernel_class<i>>([=]() {
        s::float4 RetColor = ReadAcc.read(Coord, Sampler);
        ReadDataBufAcc[0] = RetColor;
      });
    });
  }
  validateReadData(ReadData, ExpectedColor, precision);
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
    // Out-of-range mirrored_repeat mode
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(56.0f, 57.0f, 58.0f, 59.0f) / 127.0f;
    auto Sampler = s::sampler(s::coordinate_normalization_mode::normalized,
                              s::addressing_mode::mirrored_repeat,
                              s::filtering_mode::nearest);
    checkReadSampler<1>(host_ptr, Sampler, Coord, ExpectedValue);
  }

  // addressing_mode::repeat
  {
    // Out-of-range repeat mode
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(56.0f, 57.0f, 58.0f, 59.0f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::repeat, s::filtering_mode::nearest);
    checkReadSampler<2>(host_ptr, Sampler, Coord, ExpectedValue);
  }

  // addressing_mode::clamp_to_edge
  {
    // Out-of-range Edge Color
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(88.0f, 89.0f, 90.0f, 91.0f) / 127.0f;
    auto Sampler = s::sampler(s::coordinate_normalization_mode::normalized,
                              s::addressing_mode::clamp_to_edge,
                              s::filtering_mode::nearest);
    checkReadSampler<3>(host_ptr, Sampler, Coord, ExpectedValue);
  }

  // addressing_mode::clamp
  {
    // Out-of-range Border Color
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(0.0f, 0.0f, 0.0f, 0.0f);
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::clamp, s::filtering_mode::nearest);
    checkReadSampler<4>(host_ptr, Sampler, Coord, ExpectedValue);
  }

  // addressing_mode::none
  {
    // In-range for consistent return value.
    s::float4 Coord(0.0f, 0.5f, 0.75f, 0.0f);
    s::float4 ExpectedValue = s::float4(80.0f, 81.0f, 82.0f, 83.0f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::none, s::filtering_mode::nearest);
    checkReadSampler<5>(host_ptr, Sampler, Coord, ExpectedValue);
  }

  // B. coordinate_normalization_mode::unnormalized
  // addressing_mode::clamp_to_edge
  {
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(56.0f, 57.0f, 58.0f, 59.0f) / 127.0f;
    auto Sampler = s::sampler(s::coordinate_normalization_mode::unnormalized,
                              s::addressing_mode::clamp_to_edge,
                              s::filtering_mode::nearest);
    checkReadSampler<6>(host_ptr, Sampler, Coord, ExpectedValue);
  }

  // addressing_mode::clamp
  {
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(56.0f, 57.0f, 58.0f, 59.0f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::unnormalized,
                   s::addressing_mode::clamp, s::filtering_mode::nearest);
    checkReadSampler<7>(host_ptr, Sampler, Coord, ExpectedValue);
  }

  // addressing_mode::none
  {
    // In-range for consistent return value.
    s::float4 Coord(0.0f, 1.0f, 2.0f, 0.0f);
    s::float4 ExpectedValue = s::float4(56.0f, 57.0f, 58.0f, 59.0f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::unnormalized,
                   s::addressing_mode::none, s::filtering_mode::nearest);
    checkReadSampler<8>(host_ptr, Sampler, Coord, ExpectedValue);
  }
}

// Precision requirement for linear filtering mode is not defined by OpenCL
// specification. Based on observation, Host and CPU device produce values
// within +-1 ULP. GPU return values have been found to be varying. For this
// reason and to allow compatibility with all OpenCL device testing, a higher
// value of precision (in ULP) is used for Linear Filter Mode. In this case a
// value of 15000 ULP is used.
void checkSamplerLinear() {

  const int PrecisionInULP = 15000;
  // create image:
  char host_ptr[100];
  for (int i = 0; i < 100; i++)
    host_ptr[i] = i;

  // Calling only valid configurations.
  // A. coordinate normalization mode::normalized
  // addressing_mode::mirrored_repeat
  {
    // Out-of-range mirrored_repeat mode
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(44.0f, 45.0f, 46.0f, 47.0f) / 127.0f;
    auto Sampler = s::sampler(s::coordinate_normalization_mode::normalized,
                              s::addressing_mode::mirrored_repeat,
                              s::filtering_mode::linear);
    checkReadSampler<1>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }
  {
    // In-range mirrored_repeat mode
    s::float4 Coord(0.0f, 0.25f, 0.55f, 0.0f);
    s::float4 ExpectedValue = s::float4(42.8f, 43.8f, 44.8f, 45.8f) / 127.0f;
    auto Sampler = s::sampler(s::coordinate_normalization_mode::normalized,
                              s::addressing_mode::mirrored_repeat,
                              s::filtering_mode::linear);
    checkReadSampler<1>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }

  // addressing_mode::repeat
  {
    // Out-of-range repeat mode
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(46.0f, 47.0f, 48.0f, 49.0f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::repeat, s::filtering_mode::linear);
    checkReadSampler<2>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }
  {
    // In-range repeat mode
    s::float4 Coord(0.0f, 0.25f, 0.55f, 0.0f);
    s::float4 ExpectedValue = s::float4(44.8f, 45.8f, 46.8f, 47.8f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::repeat, s::filtering_mode::linear);
    checkReadSampler<2>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }

  // addressing_mode::clamp_to_edge
  {
    // Out-of-range Edge Color
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(88.0f, 89.0f, 90.0f, 91.0f) / 127.0f;
    auto Sampler = s::sampler(s::coordinate_normalization_mode::normalized,
                              s::addressing_mode::clamp_to_edge,
                              s::filtering_mode::linear);
    checkReadSampler<3>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }
  {
    s::float4 Coord(0.0f, 0.2f, 0.5f, 0.0f); // In-range
    s::float4 ExpectedValue = s::float4(36.8f, 37.8f, 38.8f, 39.8f) / 127.0f;
    auto Sampler = s::sampler(s::coordinate_normalization_mode::normalized,
                              s::addressing_mode::clamp_to_edge,
                              s::filtering_mode::linear);
    checkReadSampler<3>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }

  // addressing_mode::clamp
  {
    // Out-of-range
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(0.0f, 0.0f, 0.0f, 0.0f);
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::clamp, s::filtering_mode::linear);
    checkReadSampler<4>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }
  {
    // In-range
    s::float4 Coord(0.0f, 0.2f, 0.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(18.4f, 18.9f, 19.4f, 19.9f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::clamp, s::filtering_mode::linear);
    checkReadSampler<4>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }

  // addressing_mode::none
  {
    // In-range for consistent return value.
    s::float4 Coord(0.5f, 0.5f, 0.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(46.0f, 47.0f, 48.0f, 49.0f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::normalized,
                   s::addressing_mode::none, s::filtering_mode::linear);
    checkReadSampler<5>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }

  // B. coordinate_normalization_mode::unnormalized
  // addressing_mode::clamp_to_edge
  {
    // Out-of-range
    s::float4 Coord(0.0f, 1.5f, 2.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(56.0f, 57.0f, 58.0f, 59.0f) / 127.0f;
    auto Sampler = s::sampler(s::coordinate_normalization_mode::unnormalized,
                              s::addressing_mode::clamp_to_edge,
                              s::filtering_mode::linear);
    checkReadSampler<6>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }
  {
    // In-range
    s::float4 Coord(0.0f, 0.2f, 0.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(0.0f, 1.0f, 2.0f, 3.0f) / 127.0f;
    auto Sampler = s::sampler(s::coordinate_normalization_mode::unnormalized,
                              s::addressing_mode::clamp_to_edge,
                              s::filtering_mode::linear);
    checkReadSampler<6>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }

  // addressing_mode::clamp
  {
    // Out-of-range
    s::float4 Coord(0.0f, 1.5f, 1.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(16.0f, 16.5f, 17.0f, 17.5f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::unnormalized,
                   s::addressing_mode::clamp, s::filtering_mode::linear);
    checkReadSampler<7>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }
  {
    // In-range
    s::float4 Coord(0.0f, 0.2f, 0.5f, 0.0f);
    s::float4 ExpectedValue = s::float4(0.0f, 0.35f, 0.7f, 1.05f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::unnormalized,
                   s::addressing_mode::clamp, s::filtering_mode::linear);
    checkReadSampler<7>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }

  // addressing_mode::none
  {
    // In-range for consistent return value.
    s::float4 Coord(1.0f, 2.0f, 3.0f, 0.0f);
    s::float4 ExpectedValue = s::float4(74.0f, 75.0f, 76.0f, 77.0f) / 127.0f;
    auto Sampler =
        s::sampler(s::coordinate_normalization_mode::unnormalized,
                   s::addressing_mode::none, s::filtering_mode::linear);
    checkReadSampler<8>(host_ptr, Sampler, Coord, ExpectedValue,
                        PrecisionInULP);
  }
}

int main() {

  // Note: Currently these functions only check for vec<float, 4> return
  // datatype, the test case can be extended to test all return datatypes.
  checkSamplerNearest();
  checkSamplerLinear();
}
