// RUN: not %clangxx -fsyntax-only -std=c++17 %s -I %sycl_include/sycl 2>&1 | FileCheck %s
#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;
class mod_image;
int main() {
  half4 src[16];
  image<2> srcImage(src, image_channel_order::rgba, image_channel_type::fp16,
                    range<2>(4, 4));
  queue myQueue;
  myQueue.submit([&](handler &cgh) {
    accessor<float, 2, access::mode::read, access::target::image> NotValidType1(
        srcImage, cgh);
    // CHECK: The data type of an image accessor must be only cl_int4, cl_uint4,
    // CHECK-SAME:  cl_float4 or cl_half4
    accessor<int2, 2, access::mode::read, access::target::image> NotValidType2(
        srcImage, cgh);
    // CHECK: The data type of an image accessor must be only cl_int4, cl_uint4,
    // CHECK-SAME:  cl_float4 or cl_half4
    accessor<float4, 2, access::mode::read, access::target::image>
        ValidSYCLFloat(srcImage, cgh);
    accessor<int4, 2, access::mode::read, access::target::image> ValidSYCLInt(
        srcImage, cgh);
    accessor<uint4, 2, access::mode::read, access::target::image>
        ValidSYCLUnsigned(srcImage, cgh);
    accessor<half4, 2, access::mode::read, access::target::image> ValidSYCLHalf(
        srcImage, cgh);
  });
  return 0;
}
