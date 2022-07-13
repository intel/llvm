// UNSUPPORTED: cuda || hip
// CUDA cannot support SYCL 1.2.1 images.
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==------------------ image_accessor_readwrite_half.cpp -------------------==//
//=-image_accessor read (without sampler)& write API test for half datatype-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iomanip>
#include <sycl/sycl.hpp>
#if DEBUG_OUTPUT
#include <iostream>
#endif

namespace s = cl::sycl;

template <typename WriteDataT, int ImgType, int read_write> class kernel_class;

void check_read_data(s::cl_float4 ReadData, s::cl_float4 ExpectedColor) {
  // Maximum difference of 1.5 ULP is allowed.
  s::cl_int4 PixelDataInt = ReadData.template as<s::cl_int4>();
  s::cl_int4 ExpectedDataInt = ExpectedColor.template as<s::cl_int4>();
  s::cl_int4 Diff = ExpectedDataInt - PixelDataInt;
  bool CorrectData = false;
  if (((s::cl_int)Diff.x() <= 1 && (s::cl_int)Diff.x() >= -1) &&
      ((s::cl_int)Diff.y() <= 1 && (s::cl_int)Diff.y() >= -1) &&
      ((s::cl_int)Diff.z() <= 1 && (s::cl_int)Diff.z() >= -1) &&
      ((s::cl_int)Diff.w() <= 1 && (s::cl_int)Diff.w() >= -1))
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

void check_read_data(s::cl_half4 ReadData, s::cl_half4 ExpectedColor) {
  s::cl_float4 ReadDatafloat = ReadData.convert<float>();
  s::cl_float4 ExpectedColorfloat = ExpectedColor.convert<float>();
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
    cgh.single_task<class kernel_class<WriteDataT, static_cast<int>(ImgType), 0>>([=](){
      WriteAcc.write(Coord, Color);
    });
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

    cgh.single_task<class kernel_class<ReadDataT, static_cast<int>(ImgType), 1>>([=](){
      ReadDataT RetColor = ReadAcc.read(Coord);
      ReadDataBufAcc[0] = RetColor;
    });
    });
  }
  check_read_data(ReadData, ExpectedColor);
}

void check_half4(char *HostPtr) {

  // Calling only valid channel types with s::cl_half4.
  // s::image_channel_type::snorm_int8,
  write_type_order<s::cl_half4, s::image_channel_type::snorm_int8>(
      HostPtr, s::image_channel_order::rgba, s::cl_half4(2, -2, 0.375f, 0));
  check_read_type_order<s::cl_half4, s::image_channel_type::snorm_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_half4(1, -1, ((float)48 / 127) /*0.3779527544975280762f*/, 0));

  // s::image_channel_type::snorm_int16,
  write_type_order<s::cl_half4, s::image_channel_type::snorm_int16>(
      HostPtr, s::image_channel_order::rgba, s::cl_half4(2, -2, 0.375f, 0));
  check_read_type_order<s::cl_half4, s::image_channel_type::snorm_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_half4(1, -1, ((float)12288 / 32767) /*0.375011444091796875f*/, 0));

  // s::image_channel_type::unorm_int8,
  write_type_order<s::cl_half4, s::image_channel_type::unorm_int8>(
      HostPtr, s::image_channel_order::rgba, s::cl_half4(2, -2, 0.375f, 0));
  check_read_type_order<s::cl_half4, s::image_channel_type::unorm_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_half4(1, 0, ((float)96 / 255) /*0.3764705955982208252f*/, 0));

  // s::image_channel_type::unorm_int16
  write_type_order<s::cl_half4, s::image_channel_type::unorm_int16>(
      HostPtr, s::image_channel_order::rgba, s::cl_half4(1, -1, 0.375f, 0));
  check_read_type_order<s::cl_half4, s::image_channel_type::unorm_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_half4(1, 0, ((float)24576 / 65535) /*0.3750057220458984375f*/, 0));

  // s::image_channel_type::fp16
  write_type_order<s::cl_half4, s::image_channel_type::fp16>(
      HostPtr, s::image_channel_order::rgba, s::cl_half4(2, -2, 0.375f, 0));
  check_read_type_order<s::cl_half4, s::image_channel_type::fp16>(
      HostPtr, s::image_channel_order::rgba, s::cl_half4(2, -2, 0.375f, 0));
};

int main() {
  // Checking if default selected device supports half datatype.
  // Same device will be selected in the write/read functions.
  s::device Dev{s::default_selector()};
  if (!Dev.is_host() && !Dev.has(sycl::aspect::fp16)) {
    std::cout << "This device doesn't support the extension cl_khr_fp16"
              << std::endl;
    return 0;
  }
  // Checking only for dimension=1.
  // create image:
  char HostPtr[100];
  for (int i = 0; i < 100; i++)
    HostPtr[i] = i;

  check_half4(HostPtr);
}
