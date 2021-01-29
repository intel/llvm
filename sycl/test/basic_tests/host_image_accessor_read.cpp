// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

//==---- host_image_accessor_read.cpp - SYCL host image accessor check ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

int foo(float *image_data) {

  int result[2];
  const auto channelOrder = cl::sycl::image_channel_order::rgba;
  const auto channelType = cl::sycl::image_channel_type::fp32;

  cl::sycl::range<3> r(3, 3, 3);
  {
    cl::sycl::buffer<int, 1> ResultBuf(result, cl::sycl::range<1>(2));
    cl::sycl::queue Q;
    cl::sycl::image<3> Image(image_data, channelOrder, channelType, r);

    cl::sycl::range<2> pitch = Image.get_pitch();

    cl::sycl::cl_int4 Coords{0, 1, 2, 0};
    {
      auto host_image_acc =
          Image.template get_access<cl::sycl::float4,
                                    cl::sycl::access::mode::read>();

      auto Sampler = cl::sycl::sampler(
          cl::sycl::coordinate_normalization_mode::unnormalized,
          cl::sycl::addressing_mode::none, cl::sycl::filtering_mode::nearest);
      // Test image read function.
      cl::sycl::cl_float4 Ret_data = host_image_acc.read(Coords);
      assert((float)Ret_data.x() == 85);
      assert((float)Ret_data.y() == 86);
      assert((float)Ret_data.z() == 87);
      assert((float)Ret_data.w() == 88);

      // Test image read with sampler.
      cl::sycl::cl_float4 Ret_data2 = host_image_acc.read(Coords, Sampler);
      assert((float)Ret_data2.x() == 85);
      assert((float)Ret_data2.y() == 86);
      assert((float)Ret_data2.z() == 87);
      assert((float)Ret_data2.w() == 88);
    }

    {
      auto host_image_acc =
          Image.template get_access<cl::sycl::float4,
                                    cl::sycl::access::mode::write>();

      // Test image write function.
      host_image_acc.write(Coords, cl::sycl::cl_float4{120, 121, 122, 123});
    }

    {
      auto host_image_acc =
          Image.template get_access<cl::sycl::float4,
                                    cl::sycl::access::mode::read>();
      cl::sycl::cl_float4 Ret_data = host_image_acc.read(Coords);
      assert((float)Ret_data.x() == 120);
      assert((float)Ret_data.y() == 121);
      assert((float)Ret_data.z() == 122);
      assert((float)Ret_data.w() == 123);

      // Test Out-of-bounds access for clamp_to_edge Addressing Mode.
      auto Sampler = cl::sycl::sampler(
          cl::sycl::coordinate_normalization_mode::unnormalized,
          cl::sycl::addressing_mode::clamp_to_edge,
          cl::sycl::filtering_mode::nearest);
      cl::sycl::cl_int4 OutBnds_Coords{2, 2, 3, 0};
      cl::sycl::cl_float4 OutBnds_RetData =
          host_image_acc.read(OutBnds_Coords, Sampler);
      assert((float)OutBnds_RetData.x() == 105);
      assert((float)OutBnds_RetData.y() == 106);
      assert((float)OutBnds_RetData.z() == 107);
      assert((float)OutBnds_RetData.w() == 108);
    }

  }
  return 0;
}

int main() {
  float image_data[108]; // rgba*27 = 108.
  for (int i = 1; i < 109; i++)
    image_data[i - 1] = (float(i));
  const int Res1 = foo(image_data);
  return 0;
}
