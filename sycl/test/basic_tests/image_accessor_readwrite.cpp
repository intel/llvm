// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUNx: %GPU_RUN_PLACEHOLDER %t.out
// RUNx: %ACC_RUN_PLACEHOLDER %t.out
//==--------------------image_accessor_readwrite.cpp ----------------------==//
//==----------image_accessor read without sampler & write API test---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>
#include <iomanip>
#if DEBUG_OUTPUT
#include <iostream>
#endif

namespace s = cl::sycl;

template <typename WriteDataT, int ImgType, int read_write> class kernel_class;

template <typename PixelDataType, typename PixelDataT,
          typename = typename std::enable_if<
              (!(std::is_same<PixelDataT, s::cl_half4>::value))>::type>
void check_write_data(PixelDataType *HostDataPtr, PixelDataT ExpectedData) {
#if DEBUG_OUTPUT
  {
    if ((HostDataPtr[0] == (PixelDataType)ExpectedData.x()) &&
        (HostDataPtr[1] == (PixelDataType)ExpectedData.y()) &&
        (HostDataPtr[2] == (PixelDataType)ExpectedData.z()) &&
        (HostDataPtr[3] == (PixelDataType)ExpectedData.w())) {
      std::cout << "Data written is correct: " << std::endl;
    } else {
      std::cout << "Data written is WRONG: " << std::endl;
    }
    std::cout << "HostDataPtr: \t" << (float)HostDataPtr[0] << "  "
              << (float)HostDataPtr[1] << "  " << (float)HostDataPtr[2] << "  "
              << (float)HostDataPtr[3] << std::endl;

    std::cout << "ExpectedData: \t" << (float)ExpectedData.x() << "  "
              << (float)ExpectedData.y() << "  " << (float)ExpectedData.z()
              << "  " << (float)ExpectedData.w() << std::endl;
  }
#else
  assert(HostDataPtr[0] == (PixelDataType)ExpectedData.x());
  assert(HostDataPtr[1] == (PixelDataType)ExpectedData.y());
  assert(HostDataPtr[2] == (PixelDataType)ExpectedData.z());
  assert(HostDataPtr[3] == (PixelDataType)ExpectedData.w());
#endif
}

void check_write_data(s::cl_half *HostDataPtr, s::cl_half4 ExpectedData) {
#if DEBUG_OUTPUT
  {
    if ((HostDataPtr[0] == (float)ExpectedData.x()) &&
        (HostDataPtr[1] == (float)ExpectedData.y()) &&
        (HostDataPtr[2] == (float)ExpectedData.z()) &&
        (HostDataPtr[3] == (float)ExpectedData.w())) {
      std::cout << "Data written is correct: " << std::endl;
    } else {
      std::cout << "Data written is WRONG: " << std::endl;
    }
    std::cout << "HostDataPtr: \t" << (float)HostDataPtr[0] << "  "
              << (float)HostDataPtr[1] << "  " << (float)HostDataPtr[2] << "  "
              << (float)HostDataPtr[3] << std::endl;

    std::cout << "ExpectedData: \t" << (float)ExpectedData.x() << "  "
              << (float)ExpectedData.y() << "  " << (float)ExpectedData.z()
              << "  " << (float)ExpectedData.w() << std::endl;
  }
#else
  assert(HostDataPtr[0] == (float)ExpectedData.x());
  assert(HostDataPtr[1] == (float)ExpectedData.y());
  assert(HostDataPtr[2] == (float)ExpectedData.z());
  assert(HostDataPtr[3] == (float)ExpectedData.w());
#endif
}

template <typename ReadDataT,
          typename = typename std::enable_if<
              (!(std::is_same<ReadDataT, s::cl_float4>::value) &&
               !(std::is_same<ReadDataT, s::cl_half4>::value))>::type>
void check_read_data(ReadDataT ReadData, ReadDataT ExpectedColor) {
  using ReadDataType = typename s::detail::TryToGetElementType<ReadDataT>::type;
#if DEBUG_OUTPUT
  {
    if (((ReadDataType)ReadData.x() == (ReadDataType)ExpectedColor.x()) &&
        ((ReadDataType)ReadData.y() == (ReadDataType)ExpectedColor.y()) &&
        ((ReadDataType)ReadData.z() == (ReadDataType)ExpectedColor.z()) &&
        ((ReadDataType)ReadData.w() == (ReadDataType)ExpectedColor.w())) {
      std::cout << "Read Data is correct: " << std::endl;
    } else {
      std::cout << "Read Data is WRONG: " << std::endl;
    }
    std::cout << "ReadData: \t"
              << std::setprecision(std::numeric_limits<long double>::digits10 +
                                   1)
              << (ReadDataType)ReadData.x() << "  "
              << (ReadDataType)ReadData.y() << "  "
              << (ReadDataType)ReadData.z() << "  "
              << (ReadDataType)ReadData.w() << std::endl;

    std::cout << "ExpectedColor: \t"
              << std::setprecision(std::numeric_limits<long double>::digits10 +
                                   1)
              << (ReadDataType)ExpectedColor.x() << "  "
              << (ReadDataType)ExpectedColor.y() << "  "
              << (ReadDataType)ExpectedColor.z() << "  "
              << (ReadDataType)ExpectedColor.w() << std::endl;
  }
#else
  {
    assert((ReadDataType)ReadData.x() == (ReadDataType)ExpectedColor.x());
    assert((ReadDataType)ReadData.y() == (ReadDataType)ExpectedColor.y());
    assert((ReadDataType)ReadData.z() == (ReadDataType)ExpectedColor.z());
    assert((ReadDataType)ReadData.w() == (ReadDataType)ExpectedColor.w());
  }
#endif
}

void check_read_data(s::cl_float4 ReadData, s::cl_float4 ExpectedColor) {
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
              << (float)ReadData.x() << "  " << (float)ReadData.y() << "  "
              << (float)ReadData.z() << "  " << (float)ReadData.w()
              << std::endl;

    std::cout << "ExpectedColor: \t"
              << std::setprecision(std::numeric_limits<long double>::digits10 +
                                   1)
              << (float)ExpectedColor.x() << "  " << (float)ExpectedColor.y()
              << "  " << (float)ExpectedColor.z() << "  "
              << (float)ExpectedColor.w() << std::endl;
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

void check_read_data(s::cl_half4 ReadData, s::cl_half4 ExpectedColor) {
  // Maximum difference of 1.5 ULP is allowed.
  s::cl_float4 ReadDatafloat = ReadData.template convert<float>();
  s::cl_float4 ExpectedColorfloat = ExpectedColor.template convert<float>();
  check_read_data(ReadDatafloat, ExpectedColorfloat);
}

template <typename WriteDataT, s::image_channel_type ImgType,
          typename PixelDataType>
void check_write_type_order(char *HostPtr,
                            const s::image_channel_order ImgOrder,
                            WriteDataT Color, PixelDataType ExpectedData) {

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

  // Check Written Data.
  using PixelElementType =
      typename s::detail::TryToGetElementType<PixelDataType>::type;
  int NumChannels = 4;
  HostPtr =
      HostPtr + (2 * s::detail::getImageElementSize(NumChannels, ImgType));
  // auto HostDataPtr = reinterpret_cast<PixelElementType *>(HostPtr);
  auto HostDataPtr = (PixelElementType *)(HostPtr);

  check_write_data((PixelElementType *)(HostPtr), ExpectedData);
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

template <typename T> void check(char *);

template <> void check<s::cl_int4>(char *HostPtr) {
  // valid channel types:
  // s::image_channel_type::signed_int8,
  check_write_type_order<s::cl_int4, s::image_channel_type::signed_int8,
                         s::cl_char4>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_int4(std::numeric_limits<s::cl_int>::max(),
                 std::numeric_limits<s::cl_int>::min(), 123, 0),
      s::cl_char4(std::numeric_limits<s::cl_char>::max(),
                  std::numeric_limits<s::cl_char>::min(), 123, 0));
  check_read_type_order<s::cl_int4, s::image_channel_type::signed_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_int4(std::numeric_limits<s::cl_char>::max(),
                 std::numeric_limits<s::cl_char>::min(), 123, 0));

  // s::image_channel_type::signed_int16,
  check_write_type_order<s::cl_int4, s::image_channel_type::signed_int16,
                         s::cl_short4>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_int4(std::numeric_limits<s::cl_int>::max(),
                 std::numeric_limits<s::cl_int>::min(), 123, 0),
      s::cl_short4(std::numeric_limits<s::cl_short>::max(),
                   std::numeric_limits<s::cl_short>::min(), 123, 0));
  check_read_type_order<s::cl_int4, s::image_channel_type::signed_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_int4(std::numeric_limits<s::cl_short>::max(),
                 std::numeric_limits<s::cl_short>::min(), 123, 0));

  // s::image_channel_type::signed_int32.
  check_write_type_order<s::cl_int4, s::image_channel_type::signed_int32,
                         s::cl_int4>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_int4(std::numeric_limits<s::cl_int>::max(),
                 std::numeric_limits<s::cl_int>::min(), 123, 0),
      s::cl_int4(std::numeric_limits<s::cl_int>::max(),
                 std::numeric_limits<s::cl_int>::min(), 123, 0));
  check_read_type_order<s::cl_int4, s::image_channel_type::signed_int32>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_int4(std::numeric_limits<s::cl_int>::max(),
                 std::numeric_limits<s::cl_int>::min(), 123, 0));
};

template <> void check<s::cl_uint4>(char *HostPtr) {
  // Calling only valid channel types with s::cl_uint4.
  // s::image_channel_type::signed_int8
  check_write_type_order<s::cl_uint4, s::image_channel_type::unsigned_int8,
                         s::cl_uchar4>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_uint4(std::numeric_limits<s::cl_uint>::max(),
                  std::numeric_limits<s::cl_uint>::min(), 123, 0),
      s::cl_uchar4(std::numeric_limits<s::cl_uchar>::max(),
                   std::numeric_limits<s::cl_uchar>::min(), 123, 0));
  check_read_type_order<s::cl_uint4, s::image_channel_type::unsigned_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_uint4(std::numeric_limits<s::cl_uchar>::max(),
                  std::numeric_limits<s::cl_uchar>::min(), 123, 0));

  // s::image_channel_type::signed_int16
  check_write_type_order<s::cl_uint4, s::image_channel_type::unsigned_int16,
                         s::cl_ushort4>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_uint4(std::numeric_limits<s::cl_uint>::max(),
                  std::numeric_limits<s::cl_uint>::min(), 123, 0),
      s::cl_ushort4(std::numeric_limits<s::cl_ushort>::max(),
                    std::numeric_limits<s::cl_ushort>::min(), 123, 0));
  check_read_type_order<s::cl_uint4, s::image_channel_type::unsigned_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_uint4(std::numeric_limits<s::cl_ushort>::max(),
                  std::numeric_limits<s::cl_ushort>::min(), 123, 0));

  // s::image_channel_type::signed_int32
  check_write_type_order<s::cl_uint4, s::image_channel_type::unsigned_int32,
                         s::cl_uint4>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_uint4(std::numeric_limits<s::cl_uint>::max(),
                  std::numeric_limits<s::cl_uint>::min(), 123, 0),
      s::cl_uint4(std::numeric_limits<s::cl_uint>::max(),
                  std::numeric_limits<s::cl_uint>::min(), 123, 0));
  check_read_type_order<s::cl_uint4, s::image_channel_type::unsigned_int32>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_uint4(std::numeric_limits<s::cl_uint>::max(),
                  std::numeric_limits<s::cl_uint>::min(), 123, 0));
};

template <> void check<s::cl_float4>(char *HostPtr) {
  // Calling only valid channel types with s::cl_float4.
  // TODO: Correct the values below.
  // s::image_channel_type::snorm_int8,
  check_write_type_order<s::cl_float4, s::image_channel_type::snorm_int8,
                         s::cl_char4>(
      HostPtr, s::image_channel_order::rgba, s::cl_float4(2, -2, 0.375f, 0),
      s::cl_char4(std::numeric_limits<s::cl_char>::max(),
                  std::numeric_limits<s::cl_char>::min(), 48, 0));
  check_read_type_order<s::cl_float4, s::image_channel_type::snorm_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_float4(1, -1, ((float)48 / 127) /*0.3779527544975280762f*/, 0));

  // s::image_channel_type::snorm_int16,
  check_write_type_order<s::cl_float4, s::image_channel_type::snorm_int16,
                         s::cl_short4>(
      HostPtr, s::image_channel_order::rgba, s::cl_float4(2, -2, 0.375f, 0),
      s::cl_short4(std::numeric_limits<s::cl_short>::max(),
                   std::numeric_limits<s::cl_short>::min(), 12288, 0));
  check_read_type_order<s::cl_float4, s::image_channel_type::snorm_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_float4(1, -1, ((float)12288 / 32767) /*0.375011444091796875f*/, 0));

  // s::image_channel_type::unorm_int8,
  check_write_type_order<s::cl_float4, s::image_channel_type::unorm_int8,
                         s::cl_uchar4>(
      HostPtr, s::image_channel_order::rgba, s::cl_float4(2, -2, 0.375f, 0),
      s::cl_uchar4(std::numeric_limits<s::cl_uchar>::max(),
                   std::numeric_limits<s::cl_uchar>::min(), 96, 0));
  check_read_type_order<s::cl_float4, s::image_channel_type::unorm_int8>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_float4(1, 0, ((float)96 / 255) /*0.3764705955982208252f*/, 0));

  // s::image_channel_type::unorm_int16
  check_write_type_order<s::cl_float4, s::image_channel_type::unorm_int16,
                         s::cl_ushort4>(
      HostPtr, s::image_channel_order::rgba, s::cl_float4(2, -2, 0.375f, 0),
      s::cl_ushort4(std::numeric_limits<s::cl_ushort>::max(),
                    std::numeric_limits<s::cl_ushort>::min(), 24576, 0));
  check_read_type_order<s::cl_float4, s::image_channel_type::unorm_int16>(
      HostPtr, s::image_channel_order::rgba,
      s::cl_float4(1, 0, ((float)24576 / 65535) /*0.3750057220458984375f*/, 0));

  // s::image_channel_type::unorm_short_565, order::rgbx
  // Currently unsupported since OpenCL has no information on this.

  // TODO: Enable the below call, causing an error in scheduler
  // s::image_channel_type::unorm_short_555, order::rgbx
  /*
  check_write_type_order<s::cl_float4, s::image_channel_type::unorm_short_555,
      s::cl_short4>(
      HostPtr, s::image_channel_order::rgbx, s::cl_float4(2, -2, 0.375f, 0),
      s::cl_short4(std::numeric_limits<s::cl_short>::max(),
                std::numeric_limits<s::cl_short>::min(), 3, 0));

  // s::image_channel_type::unorm_int_101010, order::rgbx
  check_write_type_order<s::cl_float4, s::image_channel_type::unorm_int_101010,
                         s::cl_uint4>(
      HostPtr, s::image_channel_order::rgbx, s::cl_float4(2, -2, 0.375f, 0),
      s::cl_uint4(std::numeric_limits<s::cl_uint>::max(),
               std::numeric_limits<s::cl_uint>::min(), 3, 0));
  */

  // s::image_channel_type::fp16
  check_write_type_order<s::cl_float4, s::image_channel_type::fp16,
                         s::cl_half4>(HostPtr, s::image_channel_order::rgba,
                                      s::cl_float4(2, -2, 0.375f, 0),
                                      s::cl_half4(2, -2, 0.375, 0));
  check_read_type_order<s::cl_float4, s::image_channel_type::fp16>(
      HostPtr, s::image_channel_order::rgba, s::cl_float4(2, -2, 0.375f, 0));

  // s::image_channel_type::fp32
  check_write_type_order<s::cl_float4, s::image_channel_type::fp32,
                         s::cl_float4>(HostPtr, s::image_channel_order::rgba,
                                       s::cl_float4(2, -2, 0.375f, 0),
                                       s::cl_float4(2, -2, 0.375f, 0));
  check_read_type_order<s::cl_float4, s::image_channel_type::fp32>(
      HostPtr, s::image_channel_order::rgba, s::cl_float4(2, -2, 0.375f, 0));
};
/*
template <> void check<s::cl_half4>(char *HostPtr) {

  // Calling only valid channel types with s::cl_half4.
  // s::image_channel_type::fp16
  // TODO: Enable the below call. Currently it doesn't work because of
s::cl_half
  // Datatype explicit conversion issues on stmt 71-74
  check_write_type_order<s::cl_half4, s::image_channel_type::fp16, s::cl_half4>(
      HostPtr, s::image_channel_order::rgba, s::cl_half4(2, -2, 0.375f, 0),
      s::cl_half4(2, -2, 0.375, 0));
  check_read_type_order<s::cl_half4, s::image_channel_type::fp16>(
      HostPtr, s::image_channel_order::rgba, s::cl_half4(2, -2, 0.375f, 0));
};*/

int main() {
  // Checking only for dimension=1.
  // 4 datatypes possible: s::cl_uint4, s::cl_int4, s::cl_float4, s::cl_half4.
  // create image:
  char HostPtr[100];
  for (int i = 0; i < 100; i++)
    HostPtr[i] = i;

  check<s::cl_int4>(HostPtr);
  check<s::cl_uint4>(HostPtr);
  check<s::cl_float4>(HostPtr);
  // check<s::cl_half4>(HostPtr);
}
