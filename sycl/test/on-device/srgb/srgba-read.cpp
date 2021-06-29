// RUN: %clangxx -fsycl -std=c++17  -DCL_TARGET_OPENCL_VERSION=220 -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PROGRAM_COMPILE_OPTIONS="-cl-std=CL2.0" SYCL_PROGRAM_LINK_OPTIONS="-cl-std=CL2.0" %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PROGRAM_COMPILE_OPTIONS="-cl-std=CL2.0" SYCL_PROGRAM_LINK_OPTIONS="-cl-std=CL2.0" %t.out %GPU_CHECK_PLACEHOLDER

// UNSUPPORTED: CUDA

/// to build
// clang++ -fsycl -DCL_TARGET_OPENCL_VERSION=220 -o srgba.bin srgba-read.cpp

/// to run
// SYCL_PROGRAM_COMPILE_OPTIONS="-cl-std=CL2.0" SYCL_PROGRAM_LINK_OPTIONS="-cl-std=CL2.0" SYCL_DEVICE_FILTER=opencl:gpu ./srgba.bin
// SYCL_PROGRAM_COMPILE_OPTIONS="-cl-std=CL2.0" SYCL_PROGRAM_LINK_OPTIONS="-cl-std=CL2.0" SYCL_DEVICE_FILTER=opencl:cpu ./srgba.bin
// SYCL_PROGRAM_COMPILE_OPTIONS="-cl-std=CL2.0" SYCL_PROGRAM_LINK_OPTIONS="-cl-std=CL2.0" SYCL_DEVICE_FILTER=level_zero:gpu ./srgba.bin

#include <CL/sycl.hpp>

using namespace cl::sycl;

using accessorPixelT = sycl::float4;
using dataPixelT = uint32_t;

// will output a pixel as {r,g,b,a}.  provide override if a different pixelT is
// defined.
void outputPixel(sycl::float4 somePixel) {
  std::cout << "{" << somePixel[0] << "," << somePixel[1] << "," << somePixel[2]
            << "," << somePixel[3] << "} ";
}

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;
constexpr long height = 3;

void test_rd(image_channel_order ChanOrder, image_channel_type ChanType) {

  int numTests = 4; // drives the size of the testResults buffer, and the number
                    // of report iterations. Kludge.

  // this should yield a read of approximate 0.5 for each channel
  // when read directly.  For sRGB, this should be the point
  // with the maximum conversion. So we should read values of
  // 0.2 or 0.7 (but I'm not sure yet which way that conversion goes)
  dataPixelT basicPixel{127 << 24 | 127 << 16 | 127 << 8 | 127};

  queue Q;
  const sycl::range<2> ImgRange_2D(width, height);
  std::vector<dataPixelT> ImgData(ImgRange_2D.size(), basicPixel);
  try { // closure

    image<2> image_2D(ImgData.data(), ChanOrder, ChanType, ImgRange_2D);
    // use a buffer to report back test results.
    buffer<accessorPixelT, 1> testResults((range<1>(numTests)));

    Q.submit([&](handler &cgh) {
      auto image_acc =
          image_2D.get_access<accessorPixelT, access::mode::read>(cgh);
      auto test_acc = testResults.get_access<access::mode::write>(cgh);

      cgh.single_task<class im2D_rw>([=]() {
        int i = 0; // the index for writing into the testResult buffer.

        // verify our four pixels were set up correctly.
        // 0-3 read four pixels. no sampler
        test_acc[i++] = image_acc.read(sycl::int2{0, 0});
        test_acc[i++] = image_acc.read(sycl::int2{1, 0});
        test_acc[i++] = image_acc.read(sycl::int2{0, 1});
        test_acc[i++] = image_acc.read(sycl::int2{2, 2});
      });
    });
    Q.wait_and_throw();

    // REPORT RESULTS
    auto test_acc = testResults.get_access<access::mode::read>();
    for (int i = 0, idx = 0; i < numTests; i++, idx++) {
      if (i == 0) {
        idx = 0;
        std::cout << "read four pixels, no sampler" << std::endl;
      }

      accessorPixelT testPixel = test_acc[i];
      std::cout << i << /* " -- " << idx << */ ": ";
      outputPixel(testPixel);
      std::cout << std::endl;
    }
  } catch (sycl::exception e) {
    std::cout << "exception caught: " << e.what() << std::endl;
  } // ~image / ~buffer
}

int main() {

  queue Q;
  device D = Q.get_device();

  if (D.has(aspect::image)) {
    // the _int8 channels are one byte per channel, or four bytes per pixel (for
    // RGBA) the _int16/fp16 channels are two bytes per channel, or eight bytes
    // per pixel (for RGBA) the _int32/fp32  channels are four bytes per
    // channel, or sixteen bytes per pixel (for RGBA).

    // RGBx -- CL_IMAGE_FORMAT_NOT_SUPPORTED on CPU
    //         CL_INVALID_IMAGE_FORMAT_DESCRIPTOR on GPU.
    //          LevelZero dies in piMemImageCreate (shouldn't it throw?)
    //      I believe both CPU and GPU are in error.  Should be fine.
    //      (irrelevant to sRGB enablement though)
    // std::cout << "rgbx -------" << std::endl;
    // test_rw(image_channel_order::rgbx, image_channel_type::unorm_int8);

    // RGBA -- WORKS
    std::cout << "rgba -------" << std::endl;
    test_rd(image_channel_order::rgba, image_channel_type::unorm_int8);

    // srgb (24 bit) throws exception, as size is supposed to be power of 2.
    // srgbx and srgba tests follow.

    // sRGBx  -- CL_INVALID_IMAGE_DESCRIPTOR on CPU
    //           CL_IMAGE_FORMAT_NOT_SUPPORTED on GPU
    //     This is the reverse of how CPU/GPU handle 'rgbx'
    //     LevelZero dies in piMemImageCreate
    // std::cout << "srgbx -------" << std::endl;
    // test_rw(image_channel_order::srgbx, image_channel_type::unorm_int8);

    // sRGBA -- CL_IMAGE_FORMAT_NOT_SUPPORTED on both OpenCL CPU and GPU
    //          LevelZero accepts this, but I suspect it does not apply any
    //          linear scaling.
    std::cout << "srgba -------" << std::endl;
    test_rd(image_channel_order::srgba, image_channel_type::unorm_int8);
  } else {
    std::cout << "device does not support image operations" << std::endl;
  }

  return 0;
}

// clang-format off
// CHECK: rgba -------
// CHECK-NEXT: read four pixels, no sampler
//   these next four reads should all be close to 0.5
// CHECK-NEXT: 0: {0.498039,0.498039,0.498039,0.498039} 
// CHECK-NEXT: 1: {0.498039,0.498039,0.498039,0.498039} 
// CHECK-NEXT: 2: {0.498039,0.498039,0.498039,0.498039} 
// CHECK-NEXT: 3: {0.498039,0.498039,0.498039,0.498039} 
// CHECK: srgba -------
// CHECK-NEXT: read four pixels, no sampler
//   these next four reads should all be close to 0.7 
//   or maybe 0.2.  I don't know yet
// CHECK-NEXT: 0: {0.7,0.7,0.7,0.7} 
// CHECK-NEXT: 1: {0.2,0.2,0.2,0.2} 
// CHECK-NEXT: 2: {0.7,0.7,0.7,0.7} 
// CHECK-NEXT: 3: {0.2,0.2,0.2,0.2}
// clang-format on