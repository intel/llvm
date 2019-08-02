// RUN: clang++ -fsycl %s -o %t.out -lOpenCL
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//==----- image_accessor_write_api.cpp - image_accessor write API test-----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

namespace s = cl::sycl;

using namespace s;

template <typename WriteDataT, int ImgType> class kernel_class;

template <typename WriteDataT, image_channel_type ImgType,
          typename PixelDataType>
void check_write_type_order(char *host_ptr, const image_channel_order ImgOrder,
                            WriteDataT Color, PixelDataType ExpectedData) {
  // const image_channel_order ImgOrder = image_channel_order::rgba;
  cl_int Coord(2);
  {
    s::image<1> Img(host_ptr, ImgOrder, ImgType,
                    s::range<1>{10}); // 1 dim for now.
                                      // kernel.
    s::queue myQueue;
    myQueue.submit([&](s::handler &cgh) {
      auto WriteAcc = Img.get_access<WriteDataT, s::access::mode::write>(cgh);
    cgh.single_task<class kernel_class<WriteDataT, static_cast<int>(ImgType)>>([=](){
      WriteAcc.write(Coord, Color);
    });
    });
  }

  using PixelElementType =
      typename detail::TryToGetElementType<PixelDataType>::type;
  int NumChannels = 4;
  host_ptr = host_ptr + (2 * detail::getImageElementSize(NumChannels, ImgType));
  auto HostDataPtr = reinterpret_cast<PixelElementType *>(host_ptr);

  using WriteDataType = typename detail::TryToGetElementType<WriteDataT>::type;
#if DEBUG_OUTPUT
  {
    if ((HostDataPtr[0] == (WriteDataType)ExpectedData.x()) &&
        (HostDataPtr[1] == (WriteDataType)ExpectedData.y()) &&
        (HostDataPtr[2] == (WriteDataType)ExpectedData.z()) &&
        (HostDataPtr[3] == (WriteDataType)ExpectedData.w())) {
      std::cout << "Data written is correct: " << std::endl;
    } else {
      std::cout << "Data written is WRONG: " << std::endl;
    }
    std::cout << "HostDataPtr: " << (float)HostDataPtr[0] << "  "
              << (float)HostDataPtr[1] << "  " << (float)HostDataPtr[2] << "  "
              << (float)HostDataPtr[3] << std::endl;

    std::cout << "ExpectedData: " << (WriteDataType)ExpectedData.x() << "  "
              << (WriteDataType)ExpectedData.y() << "  "
              << (WriteDataType)ExpectedData.z() << "  "
              << (WriteDataType)ExpectedData.w() << std::endl;
  }
#else
  {
    assert(HostDataPtr[0] == (WriteDataType)ExpectedData.x());
    assert(HostDataPtr[1] == (WriteDataType)ExpectedData.y());
    assert(HostDataPtr[2] == (WriteDataType)ExpectedData.z());
    assert(HostDataPtr[3] == (WriteDataType)ExpectedData.w());
  }
#endif
}

template <typename T> void check(char *);

template <> void check<s::cl_int4>(char *host_ptr) {
  // valid channel types: image_channel_type::signed_int8,
  // image_channel_type::signed_int16, image_channel_type::signed_int32.
  check_write_type_order<s::int4, image_channel_type::signed_int8, s::schar4>(
      host_ptr, image_channel_order::rgba,
      s::int4(std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
              123, 0),
      s::schar4(std::numeric_limits<schar>::max(),
                std::numeric_limits<schar>::min(), 123, 0));
  check_write_type_order<s::cl_int4, image_channel_type::signed_int16,
                         s::short4>(
      host_ptr, image_channel_order::rgba,
      s::int4(std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
              123, 0),
      s::short4(std::numeric_limits<short>::max(),
                std::numeric_limits<short>::min(), 123, 0));
  check_write_type_order<s::cl_int4, image_channel_type::signed_int32, s::int4>(
      host_ptr, image_channel_order::rgba,
      s::int4(std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
              123, 0),
      s::int4(std::numeric_limits<int>::max(), std::numeric_limits<int>::min(),
              123, 0));
};

template <> void check<s::cl_uint4>(char *host_ptr) {
  // Calling only valid channel types with cl_uint4.
  // image_channel_type::signed_int8
  check_write_type_order<s::cl_uint4, image_channel_type::unsigned_int8,
                         s::uchar4>(
      host_ptr, image_channel_order::rgba,
      s::uint4(std::numeric_limits<uint>::max(),
               std::numeric_limits<uint>::min(), 123, 0),
      s::uchar4(std::numeric_limits<uchar>::max(),
                std::numeric_limits<uchar>::min(), 123, 0));
  // image_channel_type::signed_int16
  check_write_type_order<s::cl_uint4, image_channel_type::unsigned_int16,
                         s::ushort4>(
      host_ptr, image_channel_order::rgba,
      s::uint4(std::numeric_limits<uint>::max(),
               std::numeric_limits<uint>::min(), 123, 0),
      s::ushort4(std::numeric_limits<ushort>::max(),
                 std::numeric_limits<ushort>::min(), 123, 0));
  // image_channel_type::signed_int32
  check_write_type_order<s::cl_uint4, image_channel_type::unsigned_int32,
                         s::uint4>(
      host_ptr, image_channel_order::rgba,
      s::uint4(std::numeric_limits<uint>::max(),
               std::numeric_limits<uint>::min(), 123, 0),
      s::uint4(std::numeric_limits<uint>::max(),
               std::numeric_limits<uint>::min(), 123, 0));
};

template <> void check<s::cl_float4>(char *host_ptr) {
  // Calling only valid channel types with cl_float4.
  // TODO: Correct the values below.
  // image_channel_type::snorm_int8,
  check_write_type_order<s::cl_float4, image_channel_type::snorm_int8,
                         s::char4>(
      host_ptr, image_channel_order::rgba, s::float4(2, -2, 0.375f, 0),
      s::char4(std::numeric_limits<char>::max(),
               std::numeric_limits<char>::min(), 48, 0));

  // image_channel_type::snorm_int16,
  check_write_type_order<s::cl_float4, image_channel_type::snorm_int16,
                         s::short4>(
      host_ptr, image_channel_order::rgba, s::float4(2, -2, 0.375f, 0),
      s::short4(std::numeric_limits<short>::max(),
                std::numeric_limits<short>::min(), 12288, 0));

  // image_channel_type::unorm_int8,
  check_write_type_order<s::cl_float4, image_channel_type::unorm_int8,
                         s::uchar4>(
      host_ptr, image_channel_order::rgba, s::float4(2, -2, 0.375f, 0),
      s::uchar4(std::numeric_limits<uchar>::max(),
                std::numeric_limits<uchar>::min(), 96, 0));

  // image_channel_type::unorm_int16
  check_write_type_order<s::cl_float4, image_channel_type::unorm_int16,
                         s::ushort4>(
      host_ptr, image_channel_order::rgba, s::float4(2, -2, 0.375f, 0),
      s::ushort4(std::numeric_limits<ushort>::max(),
                 std::numeric_limits<ushort>::min(), 24576, 0));

  /*
  // image_channel_type::unorm_short_565, order::rgbx
  // Currently unsupported since OpenCL has no information on this.
  check_write_type_order<s::cl_float4, image_channel_type::unorm_short_565,
                         s::short4>(
      host_ptr, image_channel_order::rgbx, s::float4(2, -2, 0.375f, 0),
      s::short4(std::numeric_limits<short>::max(),
                std::numeric_limits<short>::min(), 3, 0));

  // TODO: Causing error in scheduler
  // image_channel_type::unorm_short_555, order::rgbx
  check_write_type_order<s::cl_float4, image_channel_type::unorm_short_555,
                         s::short4>(
      host_ptr, image_channel_order::rgbx, s::float4(2, -2, 0.375f, 0),
      s::short4(std::numeric_limits<short>::max(),
                std::numeric_limits<short>::min(), 3, 0));

  // image_channel_type::unorm_int_101010, order::rgbx
  check_write_type_order<s::cl_float4, image_channel_type::unorm_int_101010,
                         s::uint4>(
      host_ptr, image_channel_order::rgbx, s::float4(2, -2, 0.375f, 0),
      s::uint4(std::numeric_limits<uint>::max(),
               std::numeric_limits<uint>::min(), 3, 0));
  */

  // image_channel_type::fp16
  check_write_type_order<s::cl_float4, image_channel_type::fp16, s::half4>(
      host_ptr, image_channel_order::rgba, s::float4(2, -2, 0.375f, 0),
      s::half4(2, -2, 0.375, 0));

  // image_channel_type::fp32
  check_write_type_order<s::cl_float4, image_channel_type::fp32, s::float4>(
      host_ptr, image_channel_order::rgba, s::float4(2, -2, 0.375f, 0),
      s::float4(2, -2, 0.375f, 0));
};

template <> void check<s::cl_half4>(char *host_ptr) {

  // Calling only valid channel types with cl_half4.
  // image_channel_type::fp16
  // TODO: Enable the below call. Currently it doesn't work because of half
  // Datatype explicit conversion issues on stmt 71-74
  // check_write_type_order<s::cl_half4, image_channel_type::fp16, s::half4>(
  //    host_ptr, image_channel_order::rgba, s::half4(2, -2, 0.375f, 0),
  //    s::half4(2, -2, 0.375, 0));
};

int main() {
  // Checking only for dimension=1.
  // 4 datatypes possible: uint4, int4, float4, half4.
  // create image:
  char host_ptr[100];
  for (int i = 0; i < 100; i++)
    host_ptr[i] = i;

  check<s::cl_int4>(host_ptr);
  check<s::cl_uint4>(host_ptr);
  check<s::cl_float4>(host_ptr);
  check<s::cl_half4>(host_ptr);
}
