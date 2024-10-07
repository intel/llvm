// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: hip

// spir-v gen for legacy images at O0 not working
// UNSUPPORTED: O0

// RUN: %{build} -o %t.out
// Native images are created with host pointers only with host unified memory
// support, enforce it for this test.
// RUN: env SYCL_HOST_UNIFIED_MEMORY=1 SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s

#include <iostream>

#include <sycl/accessor_image.hpp>
#include <sycl/detail/core.hpp>

using namespace sycl;

constexpr long width = 16;
constexpr long height = 5;
constexpr long total = width * height;

constexpr long depth = 3;
constexpr long total3D = total * depth;

void remind() {
  /*
    https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadImage.html

    row_pitch in clEnqueueReadImage and input_row_pitch in clEnqueueWriteImage
    is the length of each row in bytes. This value must be greater than or equal
    to the element size in bytes × width. If row_pitch (or input_row_pitch) is
    set to 0, the appropriate row pitch is calculated based on the size of each
    element in bytes multiplied by width.

    slice_pitch in clEnqueueReadImage and input_slice_pitch in
    clEnqueueWriteImage is the size in bytes of the 2D slice of the 3D region of
    a 3D image or each image of a 1D or 2D image array being read or written
    respectively.
  */

  std::cout << "For IMAGES" << std::endl;
  std::cout << "           Region SHOULD be : " << width << "/" << height << "/"
            << depth << std::endl; // 16/5/3
  std::cout << "   row_pitch SHOULD be 0 or : " << width * sizeof(sycl::float4)
            << std::endl; // 0 or 256
  std::cout << " slice_pitch SHOULD be 0 or : "
            << width * sizeof(sycl::float4) * height << std::endl
            << std::endl; // 0 or 1280
}

// ----------- IMAGES

void testcopyD2HImage() {
  // copyD2H
  std::cout << "start copyD2H-Image" << std::endl;
  // image with write accessor to it in kernel
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;
  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<1> ImgSize_1D(width);
  // for a buffer, a range<2> would be  (height, width).
  // but for an image, the interpretation is reversed. (width, height).
  const sycl::range<2> ImgSize_2D(width, height);
  // for a buffer, a range<3> would be (depth, height, width)
  // but for an image, the interpretation is reversed (width, height, depth)
  const sycl::range<3> ImgSize_3D(width, height, depth);

  std::vector<sycl::float4> data_from_1D(ImgSize_1D.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> data_to_1D(ImgSize_1D.size(), {0, 0, 0, 0});
  std::vector<sycl::float4> data_from_2D(ImgSize_2D.size(), {7, 7, 7, 7});
  std::vector<sycl::float4> data_to_2D(ImgSize_2D.size(), {0, 0, 0, 0});
  std::vector<sycl::float4> data_from_3D(ImgSize_3D.size(), {11, 11, 11, 11});
  std::vector<sycl::float4> data_to_3D(ImgSize_3D.size(), {0, 0, 0, 0});

  {
    std::cout << "-- 1D" << std::endl;
    sycl::image<1> image_from_1D(data_from_1D.data(), ChanOrder, ChanType,
                                 ImgSize_1D);
    sycl::image<1> image_to_1D(data_to_1D.data(), ChanOrder, ChanType,
                               ImgSize_1D);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_1D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_1D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyD2H_1D>(
          ImgSize_1D, [=](sycl::item<1> Item) {
            sycl::float4 Data = readAcc.read(int(Item[0]));
            writeAcc.write(int(Item[0]), Data);
          });
    });
    std::cout << "about to destruct 1D" << std::endl;
  } // ~image 1D

  {
    std::cout << "-- 2D" << std::endl;
    sycl::image<2> image_from_2D(data_from_2D.data(), ChanOrder, ChanType,
                                 ImgSize_2D);
    sycl::image<2> image_to_2D(data_to_2D.data(), ChanOrder, ChanType,
                               ImgSize_2D);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_2D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_2D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyD2H_2D>(
          ImgSize_2D, [=](sycl::item<2> Item) {
            sycl::float4 Data = readAcc.read(sycl::int2{Item[0], Item[1]});
            writeAcc.write(sycl::int2{Item[0], Item[1]}, Data);
          });
    });
    std::cout << "about to destruct 2D" << std::endl;
  } // ~image 2D

  {
    std::cout << "-- 3D" << std::endl;
    sycl::image<3> image_from_3D(data_from_3D.data(), ChanOrder, ChanType,
                                 ImgSize_3D);
    sycl::image<3> image_to_3D(data_to_3D.data(), ChanOrder, ChanType,
                               ImgSize_3D);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_3D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_3D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyD2H_3D>(
          ImgSize_3D, [=](sycl::item<3> Item) {
            sycl::float4 Data =
                readAcc.read(sycl::int4{Item[0], Item[1], Item[2], 0});
            writeAcc.write(sycl::int4{Item[0], Item[1], Item[2], 0}, Data);
          });
    });
    std::cout << "about to destruct 3D" << std::endl;
  } // ~image 3D

  std::cout << "end copyD2H-Image" << std::endl;
}

void testcopyH2DImage() {
  // copy between two queues triggers a copyH2D, followed by a copyD2H
  // Here we only care about checking copyH2D
  std::cout << "start copyH2D-image" << std::endl;

  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;
  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<1> ImgSize_1D(width);
  // for a buffer, a range<2> would be  (height, width).
  // but for an image, the interpretation is reversed. (width, height).
  const sycl::range<2> ImgSize_2D(width, height);
  // for a buffer, a range<3> would be (depth, height, width)
  // but for an image, the interpretation is reversed (width, height, depth)
  const sycl::range<3> ImgSize_3D(width, height, depth);

  std::vector<sycl::float4> data_from_1D(ImgSize_1D.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> data_to_1D(ImgSize_1D.size(), {0, 0, 0, 0});
  std::vector<sycl::float4> data_from_2D(ImgSize_2D.size(), {7, 7, 7, 7});
  std::vector<sycl::float4> data_to_2D(ImgSize_2D.size(), {0, 0, 0, 0});
  std::vector<sycl::float4> data_from_3D(ImgSize_3D.size(), {11, 11, 11, 11});
  std::vector<sycl::float4> data_to_3D(ImgSize_3D.size(), {0, 0, 0, 0});

  // 1D
  {
    std::cout << "-- 1D" << std::endl;
    sycl::image<1> image_from_1D(data_from_1D.data(), ChanOrder, ChanType,
                                 ImgSize_1D);
    sycl::image<1> image_to_1D(data_to_1D.data(), ChanOrder, ChanType,
                               ImgSize_1D);
    device Dev;
    context Ctx{Dev};
    context otherCtx{Dev};

    queue Q{Ctx, Dev};
    queue otherQueue{otherCtx, Dev};
    // first op
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_1D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_1D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_1D>(
          ImgSize_1D, [=](sycl::item<1> Item) {
            sycl::float4 Data = readAcc.read(int(Item[0]));
            writeAcc.write(int(Item[0]), Data);
          });
    });
    Q.wait();
    // second op
    otherQueue.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_1D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_1D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_1D_2nd>(
          ImgSize_1D, [=](sycl::item<1> Item) {
            sycl::float4 Data = readAcc.read(int(Item[0]));
            writeAcc.write(int(Item[0]), Data);
          });
    });
    otherQueue.wait();
    std::cout << "about to destruct 1D" << std::endl;
  } // ~image 1D

  // 2D
  {
    std::cout << "-- 2D" << std::endl;
    sycl::image<2> image_from_2D(data_from_2D.data(), ChanOrder, ChanType,
                                 ImgSize_2D);
    sycl::image<2> image_to_2D(data_to_2D.data(), ChanOrder, ChanType,
                               ImgSize_2D);
    device Dev{default_selector_v};
    context Ctx{Dev};
    context otherCtx{Dev};

    queue Q{Ctx, Dev};
    queue otherQueue{otherCtx, Dev};
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_2D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_2D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_2D>(
          ImgSize_2D, [=](sycl::item<2> Item) {
            sycl::float4 Data = readAcc.read(sycl::int2{Item[0], Item[1]});
            writeAcc.write(sycl::int2{Item[0], Item[1]}, Data);
          });
    });
    Q.wait();
    // second op
    otherQueue.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_2D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_2D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_2D_2nd>(
          ImgSize_2D, [=](sycl::item<2> Item) {
            sycl::float4 Data = readAcc.read(sycl::int2{Item[0], Item[1]});
            writeAcc.write(sycl::int2{Item[0], Item[1]}, Data);
          });
    });
    otherQueue.wait();
    std::cout << "about to destruct 2D" << std::endl;
  } // ~image 2D

  // 3D
  {
    std::cout << "-- 3D" << std::endl;
    sycl::image<3> image_from_3D(data_from_3D.data(), ChanOrder, ChanType,
                                 ImgSize_3D);
    sycl::image<3> image_to_3D(data_to_3D.data(), ChanOrder, ChanType,
                               ImgSize_3D);
    device Dev{default_selector_v};
    context Ctx{Dev};
    context otherCtx{Dev};

    queue Q{Ctx, Dev};
    queue otherQueue{otherCtx, Dev};
    Q.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_3D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_3D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_3D>(
          ImgSize_3D, [=](sycl::item<3> Item) {
            sycl::float4 Data =
                readAcc.read(sycl::int4{Item[0], Item[1], Item[2], 0});
            writeAcc.write(sycl::int4{Item[0], Item[1], Item[2], 0}, Data);
          });
    });
    Q.wait();
    // second op
    otherQueue.submit([&](sycl::handler &CGH) {
      auto readAcc = image_from_3D.get_access<sycl::float4, SYCLRead>(CGH);
      auto writeAcc = image_to_3D.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyH2D_3D_2nd>(
          ImgSize_3D, [=](sycl::item<3> Item) {
            sycl::float4 Data =
                readAcc.read(sycl::int4{Item[0], Item[1], Item[2], 0});
            writeAcc.write(sycl::int4{Item[0], Item[1], Item[2], 0}, Data);
          });
    });
    otherQueue.wait();
    std::cout << "about to destruct 3D" << std::endl;
  } // ~image 3D

  std::cout << "end copyH2D-image" << std::endl;
}

// --------------

int main() {

  remind();
  testcopyD2HImage();
  testcopyH2DImage();
  // TODO  .copy() and .fill() not yet supported for images
  // add tests once they are.
}

// ----------- IMAGES

// clang-format off
//CHECK: start copyD2H-Image
//CHECK: -- 1D
//CHECK: <--- urMemImageCreate({{.*}} .type = UR_MEM_TYPE_IMAGE1D, .width = 16, .height = 1, .depth = 1, .arraySize = 0, .rowPitch = 256, .slicePitch = 256, .numMipLevel = 0, .numSamples = 0
//CHECK: <--- urMemImageCreate({{.*}} .type = UR_MEM_TYPE_IMAGE1D, .width = 16, .height = 1, .depth = 1, .arraySize = 0, .rowPitch = 256, .slicePitch = 256, .numMipLevel = 0, .numSamples = 0
//CHECK: about to destruct 1D
//CHECK: <--- urEnqueueMemImageRead({{.*}} .region = (struct ur_rect_region_t){.width = 16, .height = 1, .depth = 1}
//CHECK: -- 2D
//CHECK: <--- urMemImageCreate({{.*}} .type = UR_MEM_TYPE_IMAGE2D, .width = 16, .height = 5, .depth = 1, .arraySize = 0, .rowPitch = 256, .slicePitch = 1280, .numMipLevel = 0, .numSamples = 0
//CHECK: <--- urMemImageCreate({{.*}} .type = UR_MEM_TYPE_IMAGE2D, .width = 16, .height = 5, .depth = 1, .arraySize = 0, .rowPitch = 256, .slicePitch = 1280, .numMipLevel = 0, .numSamples = 0
//CHECK: about to destruct 2D
//CHECK: <--- urEnqueueMemImageRead({{.*}} .region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 1}, .rowPitch = 256
//CHECK: -- 3D
//CHECK: <--- urMemImageCreate({{.*}} .type = UR_MEM_TYPE_IMAGE3D, .width = 16, .height = 5, .depth = 3, .arraySize = 0, .rowPitch = 256, .slicePitch = 1280, .numMipLevel = 0, .numSamples = 
//CHECK: <--- urMemImageCreate({{.*}} .type = UR_MEM_TYPE_IMAGE3D, .width = 16, .height = 5, .depth = 3, .arraySize = 0, .rowPitch = 256, .slicePitch = 1280, .numMipLevel = 0, .numSamples = 
//CHECK: about to destruct 3D
//CHECK: <--- urEnqueueMemImageRead({{.*}}.region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 3}, .rowPitch = 256, .slicePitch = 1280
//CHECK: end copyD2H-Image

// CHECK: start copyH2D-image
// CHECK: -- 1D
// CHECK: <--- urMemImageCreate({{.*}} .type = UR_MEM_TYPE_IMAGE1D, .width = 16, .height = 1, .depth = 1, .arraySize = 0, .rowPitch = 256, .slicePitch = 256, .numMipLevel = 0, .numSamples = 0
// CHECK: <--- urMemImageCreate({{.*}} .type = UR_MEM_TYPE_IMAGE1D, .width = 16, .height = 1, .depth = 1, .arraySize = 0, .rowPitch = 256, .slicePitch = 256, .numMipLevel = 0, .numSamples = 0
// CHECK: <--- urMemImageCreate({{.*}} .type = UR_MEM_TYPE_IMAGE1D, .width = 16, .height = 1, .depth = 1, .arraySize = 0, .rowPitch = 0, .slicePitch = 0, .numMipLevel = 0, .numSamples = 0
// CHECK: <--- urEnqueueMemImageRead({{.*}} .region = (struct ur_rect_region_t){.width = 16, .height = 1, .depth = 1}
// The order of the following calls may vary since some of them are made by a
// host task (in a separate thread). Don't check for the actual function name
// as it may be interleaved with other tracing output.
// CHECK-DAG: .type = UR_MEM_TYPE_IMAGE1D, .width = 16, .height = 1, .depth = 1, .arraySize = 0, .rowPitch = 0, .slicePitch = 0, .numMipLevel = 0, .numSamples = 0
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 1, .depth = 1}
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 1, .depth = 1}
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 1, .depth = 1}
// CHECK: about to destruct 1D
// CHECK: <--- urEnqueueMemImageRead({{.*}}.region = (struct ur_rect_region_t){.width = 16, .height = 1, .depth = 1}
// CHECK: -- 2D
// The order of the following calls may vary since some of them are made by a
// host task (in a separate thread). Don't check for the actual function name
// as it may be interleaved with other tracing output.
// CHECK-DAG: .type = UR_MEM_TYPE_IMAGE2D, .width = 16, .height = 5, .depth = 1, .arraySize = 0, .rowPitch = 256, .slicePitch = 1280, .numMipLevel = 0, .numSamples = 0
// CHECK-DAG: .type = UR_MEM_TYPE_IMAGE2D, .width = 16, .height = 5, .depth = 1, .arraySize = 0, .rowPitch = 256, .slicePitch = 1280, .numMipLevel = 0, .numSamples = 0
// CHECK-DAG: .type = UR_MEM_TYPE_IMAGE2D, .width = 16, .height = 5, .depth = 1, .arraySize = 0, .rowPitch = 0, .slicePitch = 0, .numMipLevel = 0, .numSamples = 0
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 1}
// CHECK-DAG: .type = UR_MEM_TYPE_IMAGE2D, .width = 16, .height = 5, .depth = 1, .arraySize = 0, .rowPitch = 0, .slicePitch = 0, .numMipLevel = 0, .numSamples = 0
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 1}
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 1}
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 1}
// CHECK: about to destruct 2D
// CHECK: <--- urEnqueueMemImageRead({{.*}}.region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 1}
// CHECK: -- 3D
// CHECK-DAG: .type = UR_MEM_TYPE_IMAGE3D, .width = 16, .height = 5, .depth = 3, .arraySize = 0, .rowPitch = 256, .slicePitch = 1280, .numMipLevel = 0, .numSamples = 0
// CHECK-DAG: .type = UR_MEM_TYPE_IMAGE3D, .width = 16, .height = 5, .depth = 3, .arraySize = 0, .rowPitch = 256, .slicePitch = 1280, .numMipLevel = 0, .numSamples = 0
// CHECK-DAG: .type = UR_MEM_TYPE_IMAGE3D, .width = 16, .height = 5, .depth = 3, .arraySize = 0, .rowPitch = 0, .slicePitch = 0, .numMipLevel = 0, .numSamples = 0
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 3}
// The order of the following calls may vary since some of them are made by a
// host task (in a separate thread). Don't check for the actual function name
// as it may be interleaved with other tracing output.
// CHECK-DAG: .type = UR_MEM_TYPE_IMAGE3D, .width = 16, .height = 5, .depth = 3, .arraySize = 0, .rowPitch = 0, .slicePitch = 0, .numMipLevel = 0, .numSamples = 0
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 3}
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 3}, .rowPitch = 256, .slicePitch = 1280
// CHECK-DAG: .region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 3}, .rowPitch = 256, .slicePitch = 1280
// CHECK: about to destruct 3D
// CHECK: <--- urEnqueueMemImageRead({{.*}} .region = (struct ur_rect_region_t){.width = 16, .height = 5, .depth = 3}, .rowPitch = 256, .slicePitch = 1280

// CHECK: end copyH2D-image
// clang-format on
