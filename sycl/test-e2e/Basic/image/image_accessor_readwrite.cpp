// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: cuda
// UNSUPPORTED-INTENDED: CUDA doesn't fully support SYCL 1.2.1 images. Bindless
// images should be used instead.
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

//==--------------------image_accessor_readwrite.cpp ----------------------==//
//==----------image_accessor read without sampler & write API test---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iomanip>
#include <sycl/accessor_image.hpp>
#include <sycl/detail/core.hpp>
#if DEBUG_OUTPUT
#include <iostream>
#endif

namespace s = sycl;

template <typename WriteDataT, int ImgType, int read_write> class kernel_class;

template <typename ReadDataT,
          typename = typename std::enable_if<
              (!(std::is_same_v<ReadDataT, s::float4>) &&
               !(std::is_same_v<ReadDataT, s::half4>))>::type>
void check_read_data(ReadDataT ReadData, ReadDataT ExpectedColor) {
  using ReadDataType = typename ReadDataT::element_type;
  bool CorrectData = false;
  if ((ReadData.x() == ExpectedColor.x()) &&
      (ReadData.y() == ExpectedColor.y()) &&
      (ReadData.z() == ExpectedColor.z()) &&
      (ReadData.w() == ExpectedColor.w()))
    CorrectData = true;

#if DEBUG_OUTPUT
  if (CorrectData)
    std::cout << "Read Data is correct: " << std::endl;
  else
    std::cout << "Read Data is WRONG: " << std::endl;

  std::cout << "ReadData: \t"
            << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
            << ReadData.x() << "  " << ReadData.y() << "  " << ReadData.z()
            << "  " << ReadData.w() << std::endl;

  std::cout << "ExpectedColor: \t"
            << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
            << ExpectedColor.x() << "  " << ExpectedColor.y() << "  "
            << ExpectedColor.z() << "  " << ExpectedColor.w() << std::endl;
#else
  assert(CorrectData);
#endif
}

void check_read_data(s::float4 ReadData, s::float4 ExpectedColor) {
  // Maximum difference of 1.5 ULP is allowed.
  s::int4 PixelDataInt = ReadData.template as<s::int4>();
  s::int4 ExpectedDataInt = ExpectedColor.template as<s::int4>();
  s::int4 Diff = ExpectedDataInt - PixelDataInt;
  bool CorrectData = false;
  if ((Diff.x() <= 1 && Diff.x() >= -1) && (Diff.y() <= 1 && Diff.y() >= -1) &&
      (Diff.z() <= 1 && Diff.z() >= -1) && (Diff.w() <= 1 && Diff.w() >= -1))
    CorrectData = true;

#if DEBUG_OUTPUT
  if (CorrectData)
    std::cout << "Read Data is correct within precision: " << std::endl;
  else
    std::cout << "Read Data is WRONG/ outside precision: " << std::endl;

  std::cout << "ReadData: \t"
            << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
            << ReadData.x() << "  " << ReadData.y() << "  " << ReadData.z()
            << "  " << ReadData.w() << std::endl;

  std::cout << "ExpectedColor: \t"
            << std::setprecision(std::numeric_limits<long double>::digits10 + 1)
            << ExpectedColor.x() << "  " << ExpectedColor.y() << "  "
            << ExpectedColor.z() << "  " << ExpectedColor.w() << std::endl;
#else
  assert(CorrectData);
#endif
}

void check_read_data(s::half4 ReadData, s::half4 ExpectedColor) {
  // Maximum difference of 1.5 ULP is allowed.
  s::float4 ReadDatafloat = ReadData.template convert<float>();
  s::float4 ExpectedColorfloat = ExpectedColor.template convert<float>();
  check_read_data(ReadDatafloat, ExpectedColorfloat);
}

template <typename WriteDataT, s::image_channel_type ImgType>
void write_type_order(char *HostPtr, const s::image_channel_order ImgOrder,
                      WriteDataT Color) {

  int Coord(2);
  {
    // image with dim = 1;
    s::image<1> Img(HostPtr, ImgOrder, ImgType, s::range<1>{10});
    s::queue Queue;
    Queue.submit([&](s::handler &cgh) {
      auto WriteAcc = Img.get_access<WriteDataT, s::access::mode::write>(cgh);
      cgh.single_task<
          class kernel_class<WriteDataT, static_cast<int>(ImgType), 0>>(
          [=]() { WriteAcc.write(Coord, Color); });
    });
  }
}

template <typename ReadDataT, s::image_channel_type ImgType>
void check_read_type_order(char *HostPtr, const s::image_channel_order ImgOrder,
                           ReadDataT ExpectedColor) {

  int Coord(2);
  ReadDataT ReadData;
  {
    // image with dim = 1
    s::image<1> Img(HostPtr, ImgOrder, ImgType, s::range<1>{10});
    s::queue Queue;
    s::buffer<ReadDataT, 1> ReadDataBuf(&ReadData, s::range<1>(1));
    Queue.submit([&](s::handler &cgh) {
      auto ReadAcc = Img.get_access<ReadDataT, s::access::mode::read>(cgh);
      s::accessor<ReadDataT, 1, s::access::mode::write> ReadDataBufAcc(
          ReadDataBuf, cgh);

      cgh.single_task<
          class kernel_class<ReadDataT, static_cast<int>(ImgType), 1>>([=]() {
        ReadDataT RetColor = ReadAcc.read(Coord);
        ReadDataBufAcc[0] = RetColor;
      });
    });
  }
  check_read_data(ReadData, ExpectedColor);
}

template <typename T> void check(char *);

template <> void check<s::int4>(char *HostPtr) {
  // valid channel types:
  // s::image_channel_type::signed_int8,
  write_type_order<s::int4, s::image_channel_type::signed_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::int4(std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
              123, 0));
  check_read_type_order<s::int4, s::image_channel_type::signed_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::int4(std::numeric_limits<char>::max(),
              std::numeric_limits<char>::min(), 123, 0));

  // s::image_channel_type::signed_int16,
  write_type_order<s::int4, s::image_channel_type::signed_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::int4(std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
              123, 0));
  check_read_type_order<s::int4, s::image_channel_type::signed_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::int4(std::numeric_limits<short>::max(),
              std::numeric_limits<short>::min(), 123, 0));

  // s::image_channel_type::signed_int32.
  write_type_order<s::int4, s::image_channel_type::signed_int32>(
      HostPtr, s::image_channel_order::rgba,
      s::int4(std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
              123, 0));
  check_read_type_order<s::int4, s::image_channel_type::signed_int32>(
      HostPtr, s::image_channel_order::rgba,
      s::int4(std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
              123, 0));
};

template <> void check<s::uint4>(char *HostPtr) {
  // Calling only valid channel types with s::uint4.
  // s::image_channel_type::signed_int8
  write_type_order<s::uint4, s::image_channel_type::unsigned_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::uint4(std::numeric_limits<unsigned int>::max(),
               std::numeric_limits<unsigned int>::min(), 123, 0));
  check_read_type_order<s::uint4, s::image_channel_type::unsigned_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::uint4(std::numeric_limits<unsigned char>::max(),
               std::numeric_limits<unsigned char>::min(), 123, 0));

  // s::image_channel_type::signed_int16
  write_type_order<s::uint4, s::image_channel_type::unsigned_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::uint4(std::numeric_limits<unsigned int>::max(),
               std::numeric_limits<unsigned int>::min(), 123, 0));
  check_read_type_order<s::uint4, s::image_channel_type::unsigned_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::uint4(std::numeric_limits<unsigned short>::max(),
               std::numeric_limits<unsigned short>::min(), 123, 0));

  // s::image_channel_type::signed_int32
  write_type_order<s::uint4, s::image_channel_type::unsigned_int32>(
      HostPtr, s::image_channel_order::rgba,
      s::uint4(std::numeric_limits<unsigned int>::max(),
               std::numeric_limits<unsigned int>::min(), 123, 0));
  check_read_type_order<s::uint4, s::image_channel_type::unsigned_int32>(
      HostPtr, s::image_channel_order::rgba,
      s::uint4(std::numeric_limits<unsigned int>::max(),
               std::numeric_limits<unsigned int>::min(), 123, 0));
};

template <> void check<s::float4>(char *HostPtr) {
  // Calling only valid channel types with s::float4.
  // TODO: Correct the values below.
  // s::image_channel_type::snorm_int8,
  write_type_order<s::float4, s::image_channel_type::snorm_int8>(
      HostPtr, s::image_channel_order::rgba, s::float4(2, -2, 0.375f, 0));
  check_read_type_order<s::float4, s::image_channel_type::snorm_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::float4(1, -1, ((float)48 / 127) /*0.3779527544975280762f*/, 0));

  // s::image_channel_type::snorm_int16,
  write_type_order<s::float4, s::image_channel_type::snorm_int16>(
      HostPtr, s::image_channel_order::rgba, s::float4(2, -2, 0.375f, 0));
  check_read_type_order<s::float4, s::image_channel_type::snorm_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::float4(1, -1, ((float)12288 / 32767) /*0.375011444091796875f*/, 0));

  // s::image_channel_type::unorm_int8,
  write_type_order<s::float4, s::image_channel_type::unorm_int8>(
      HostPtr, s::image_channel_order::rgba, s::float4(2, -2, 0.375f, 0));
  check_read_type_order<s::float4, s::image_channel_type::unorm_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::float4(1, 0, ((float)96 / 255) /*0.3764705955982208252f*/, 0));

  // s::image_channel_type::unorm_int16
  write_type_order<s::float4, s::image_channel_type::unorm_int16>(
      HostPtr, s::image_channel_order::rgba, s::float4(2, -2, 0.375f, 0));
  check_read_type_order<s::float4, s::image_channel_type::unorm_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::float4(1, 0, ((float)24576 / 65535) /*0.3750057220458984375f*/, 0));

  // s::image_channel_type::unorm_short_565, order::rgbx
  // Currently unsupported since OpenCL has no information on this.

  // TODO: Enable the below call, causing a runtime error in OpenCL CPU/GPU:
  // OpenCL API returns: -10 (CL_IMAGE_FORMAT_NOT_SUPPORTED) -10
  // (CL_IMAGE_FORMAT_NOT_SUPPORTED) s::image_channel_type::unorm_short_555,
  // order::rgbx
  /*
  write_type_order<s::float4, s::image_channel_type::unorm_short_555>(
      HostPtr, s::image_channel_order::rgbx, s::float4(2, -2, 0.375f,
  0));

  // s::image_channel_type::unorm_int_101010, order::rgbx
  write_type_order<s::float4, s::image_channel_type::unorm_int_101010>(
      HostPtr, s::image_channel_order::rgbx, s::float4(2, -2, 0.375f,
  0));
  */

  // s::image_channel_type::fp16
  write_type_order<s::float4, s::image_channel_type::fp16>(
      HostPtr, s::image_channel_order::rgba, s::float4(2, -2, 0.375f, 0));
  check_read_type_order<s::float4, s::image_channel_type::fp16>(
      HostPtr, s::image_channel_order::rgba, s::float4(2, -2, 0.375f, 0));

  // s::image_channel_type::fp32
  write_type_order<s::float4, s::image_channel_type::fp32>(
      HostPtr, s::image_channel_order::rgba, s::float4(2, -2, 0.375f, 0));
  check_read_type_order<s::float4, s::image_channel_type::fp32>(
      HostPtr, s::image_channel_order::rgba, s::float4(2, -2, 0.375f, 0));
};

int main() {
  // Checking only for dimension=1.
  // 4 datatypes possible: s::uint4, s::int4, s::float4,
  // s::half4. s::half4 datatype is checked in a different test case. create
  // image:
  char HostPtr[100];
  for (int i = 0; i < 100; i++)
    HostPtr[i] = i;

  check<s::int4>(HostPtr);
  check<s::uint4>(HostPtr);
  check<s::float4>(HostPtr);
}
