// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: env SYCL_PI_TRACE=2 %GPU_RUN_PLACEHOLDER %t.out | FileCheck %s
// RUN: env SYCL_PI_TRACE=2 %CPU_RUN_PLACEHOLDER %t.out | FileCheck %s
// XFAIL: *

/*
  Manual
    clang++ -fsycl -o eao.bin enqueue-arg-order.cpp
    SYCL_PI_TRACE=2 ./eao.bin

    clang++ --driver-mode=g++ -fsycl
  -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -o eao.bin enqueue-arg-order.cpp
    SYCL_PI_TRACE=2 SYCL_BE=PI_CUDA ./eao.bin

    llvm-lit --param SYCL_BE=PI_CUDA -v enqueue-arg-order.cpp
*/

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

constexpr long width = 16;
constexpr long height = 5;
constexpr long total = width * height;

void remind() {
  /*
    https://www.khronos.org/registry/OpenCL/sdk/2.2/docs/man/html/clEnqueueReadBufferRect.html

    buffer_origin defines the (x, y, z) offset in the memory region associated
    with buffer. For a 2D rectangle region, the z value given by
    buffer_origin[2] should be 0. The offset in bytes is computed as
    buffer_origin[2] × buffer_slice_pitch + buffer_origin[1] × buffer_row_pitch
    + buffer_origin[0].

    region defines the (width in bytes, height in rows, depth in slices) of the
    2D or 3D rectangle being read or written. For a 2D rectangle copy, the depth
    value given by region[2] should be 1. The values in region cannot be 0.


    buffer_row_pitch is the length of each row in bytes to be used for the
    memory region associated with buffer. If buffer_row_pitch is 0,
    buffer_row_pitch is computed as region[0].

    buffer_slice_pitch is the length of each 2D slice in bytes to be used for
    the memory region associated with buffer. If buffer_slice_pitch is 0,
    buffer_slice_pitch is computed as region[1] × buffer_row_pitch.
  */
  std::cout << "For BUFFERS" << std::endl;
  std::cout << "         Region SHOULD be : " << width * sizeof(float) << "/"
            << height << "/" << 1 << std::endl; // 64/5/1
  std::cout << "  RowPitch SHOULD be 0 or : " << width * sizeof(float)
            << std::endl; // 0 or 64
  std::cout << "SlicePitch SHOULD be 0 or : " << width * sizeof(float) * height
            << std::endl
            << std::endl; // 0 or 320

  // NOTE: presently we see 20/16/1 for Region and 20 for row pitch.  both
  // incorrect.

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
            << 1 << std::endl; // 16/5/1
  std::cout << "   row_pitch SHOULD be 0 or : " << width * sizeof(sycl::float4)
            << std::endl; // 0 or 256
  std::cout << " slice_pitch SHOULD be 0 or : "
            << width * sizeof(sycl::float4) * height << std::endl
            << std::endl; // 0 or 1280

  // NOTE: presently we see 5/16/1 for image Region and 80 for row pitch.  both
  // incorrect
}

void testCopyD2HBuffer() {
  // copyD2H
  std::cout << "start copyD2H-Buffer" << std::endl;
  std::vector<float> data(total, 0);
  {
    buffer<float, 2> base(data.data(), range<2>(height, width));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto acc = base.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class copyD2H>(base.get_range(), [=](id<2> index) {
        float y_term = (float)(index[0]);
        float x_term = (float)(index[1]);
        acc[index] = x_term + (y_term / 10);
      });
    });
  } // ~buffer
  std::cout << "end copyD2H-Buffer" << std::endl;
}

void testcopyTwiceBuffer() {
  // copy between two queues triggers a piEnqueueMemBufferMap followed by
  // copyH2D, followed by a copyD2H, followed by a piEnqueueMemUnmap this may
  // change in the future. Here we only care that the 2D offset and region args
  // are passed in the right order to copyH2D and copyD2H

  std::cout << "start copyTwice-buffer" << std::endl;
  std::vector<float> data(total, 0);
  {
    // initialize buffer with data
    buffer<float, 2> base(data.data(), range<2>(height, width));

    // first op
    queue myQueue;
    queue otherQueue;
    myQueue.submit([&](handler &cgh) {
      auto acc = base.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class copyH2D0>(
          base.get_range(), [=](id<2> index) { acc[index] = acc[index] * -1; });
    });
    myQueue.wait();

    otherQueue.submit([&](handler &cgh) {
      auto acc = base.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class copyH2D1>(
          base.get_range(), [=](id<2> index) { acc[index] = acc[index] * -1; });
    });

  } // ~buffer
  std::cout << "end copyTwice-buffer" << std::endl;
}

void testCopyD2HImage() {
  // copyD2H
  std::cout << "start copyD2H-Image" << std::endl;
  // image with write accessor to it in kernel
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<2> Img1Size(height, width);
  const sycl::range<2> Img2Size(height, width);

  std::vector<sycl::float4> Img1HostData(Img1Size.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> Img2HostData(Img2Size.size(), {0, 0, 0, 0});

  {
    sycl::image<2> Img1(Img1HostData.data(), ChanOrder, ChanType, Img1Size);
    sycl::image<2> Img2(Img2HostData.data(), ChanOrder, ChanType, Img2Size);
    queue Q;
    Q.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img1.get_access<sycl::float4, SYCLRead>(CGH);
      auto Img2Acc = Img2.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopy>(Img1Size, [=](sycl::item<2> Item) {
        sycl::float4 Data = Img1Acc.read(sycl::int2{Item[0], Item[1]});
        Img2Acc.write(sycl::int2{Item[0], Item[1]}, Data);
      });
    });
  } // ~image
  std::cout << "end copyD2H-Image" << std::endl;
}

void testCopyTwiceImage() {
  // copyD2H and copyH2D
  std::cout << "start copyTwiceImage" << std::endl;
  // image with write accessor to it in kernel
  const sycl::image_channel_order ChanOrder = sycl::image_channel_order::rgba;
  const sycl::image_channel_type ChanType = sycl::image_channel_type::fp32;

  constexpr auto SYCLRead = sycl::access::mode::read;
  constexpr auto SYCLWrite = sycl::access::mode::write;

  const sycl::range<2> Img1Size(height, width);
  const sycl::range<2> Img2Size(height, width);

  std::vector<sycl::float4> Img1HostData(Img1Size.size(), {1, 2, 3, 4});
  std::vector<sycl::float4> Img2HostData(Img2Size.size(), {0, 0, 0, 0});

  {
    sycl::image<2> Img1(Img1HostData.data(), ChanOrder, ChanType, Img1Size);
    sycl::image<2> Img2(Img2HostData.data(), ChanOrder, ChanType, Img2Size);
    queue Q;
    queue otherQueue;

    // first op
    Q.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img1.get_access<sycl::float4, SYCLRead>(CGH);
      auto Img2Acc = Img2.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyTwice0>(Img1Size, [=](sycl::item<2> Item) {
        sycl::float4 Data = Img1Acc.read(sycl::int2{Item[0], Item[1]});
        Img2Acc.write(sycl::int2{Item[0], Item[1]}, Data);
      });
    });

    // second op
    otherQueue.submit([&](sycl::handler &CGH) {
      auto Img1Acc = Img2.get_access<sycl::float4, SYCLRead>(CGH);
      auto Img2Acc = Img1.get_access<sycl::float4, SYCLWrite>(CGH);

      CGH.parallel_for<class ImgCopyTwice1>(Img1Size, [=](sycl::item<2> Item) {
        sycl::float4 Data = Img1Acc.read(sycl::int2{Item[0], Item[1]});
        Img2Acc.write(sycl::int2{Item[0], Item[1]}, Data);
      });
    });
  } // ~image
  std::cout << "end copyTwiceImage" << std::endl;
}

int main() {
  remind();
  testCopyD2HBuffer();
  testcopyTwiceBuffer();

  testCopyD2HImage();
  testCopyTwiceImage();
}

//CHECK: start copyD2H-Buffer
//CHECK: ---> piEnqueueMemBufferReadRect(
//CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
//CHECK: <unknown> : 64
//CHECK: end copyD2H-Buffer

//CHECK: start copyTwice-buffer
//CHECK:  ---> piEnqueueMemBufferWriteRect(
//CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
//CHECK: <unknown> : 64
//CHECK: <unknown> : 0
//CHECK: <unknown> : 64

//CHECK: ---> piEnqueueMemBufferReadRect(
//CHECK: pi_buff_rect_region width_bytes/height/depth : 64/5/1
//CHECK: <unknown> : 64
//CHECK: end copyTwice-buffer

//CHECK: start copyD2H-Image
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK: <unknown> : 256
//CHECK: end copyD2H-Image

//CHECK: start copyTwiceImage
//CHECK: ---> piEnqueueMemImageRead(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK: <unknown> : 256
//CHECK: ---> piEnqueueMemImageWrite(
//CHECK: pi_image_region width/height/depth : 16/5/1
//CHECK: <unknown> : 256
//CHECK: end copyTwiceImage