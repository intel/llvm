// UNSUPPORTED: rocm
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
// XFAIL: cuda

// CUDA is not handling repeat or mirror correctly with normalized coordinates.
// Waiting on a fix.

/*
    This file sets up an image, initializes it with data,
    and verifies that the data is sampled correctly with a
    sampler configured NORMALIZED coordinate_normalization_mode
    REPEAT address_mode and NEAREST filter_mode

*/

#include <CL/sycl.hpp>

using namespace cl::sycl;

// pixel data-type for RGBA operations (which is the minimum image type)
using pixelT = sycl::uint4;

// will output a pixel as {r,g,b,a}.  provide override if a different pixelT is
// defined.
void outputPixel(sycl::uint4 somePixel) {
  std::cout << "{" << somePixel[0] << "," << somePixel[1] << "," << somePixel[2]
            << "," << somePixel[3] << "} ";
}

// some constants.

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

constexpr auto normalized = coordinate_normalization_mode::normalized;
constexpr auto repeat = addressing_mode::repeat;
constexpr auto nearest = filtering_mode::nearest;

void test_normalized_repeat_nearest_sampler(image_channel_order ChanOrder,
                                            image_channel_type ChanType) {
  int numTests = 12; // drives the size of the testResults buffer, and the
                     // number of report iterations. Kludge.

  // we'll use these four pixels for our image. Makes it easy to measure
  // interpolation and spot "off-by-one" probs.
  pixelT leftEdge{1, 2, 3, 4};
  pixelT body{49, 48, 47, 46};
  pixelT bony{59, 58, 57, 56};
  pixelT rightEdge{11, 12, 13, 14};

  queue Q;
  const sycl::range<1> ImgRange_1D(width);
  { // closure
    // - create an image
    image<1> image_1D(ChanOrder, ChanType, ImgRange_1D);
    event E_Setup = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::write>(cgh);
      cgh.single_task<class setupUnormLinear>([=]() {
        image_acc.write(0, leftEdge);
        image_acc.write(1, body);
        image_acc.write(2, bony);
        image_acc.write(3, rightEdge);
      });
    });
    E_Setup.wait();

    // use a buffer to report back test results.
    buffer<pixelT, 1> testResults((range<1>(numTests)));

    // sampler
    auto Norm_Repeat_Nearest_sampler = sampler(normalized, repeat, nearest);

    event E_Test = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::read>(cgh);
      auto test_acc = testResults.get_access<access::mode::write>(cgh);

      cgh.single_task<class im1D_norm_nearest>([=]() {
        int i = 0; // the index for writing into the testResult buffer.

        // clang-format off
        // Normalized Pixel Locations.  
        //      .125        .375        .625        .875            <-- exact center
        //  |-----^-----|-----^-----|-----^-----|-----^-----
        //[0.0         .25         .50         .75          (1)     <-- low boundary (included in pixel)
        //                                                              upper boundary inexact. (e.g. .2499999)
        // clang-format on

        // 0-3 read four pixels at low-boundary locations,  sample:   Normalized
        // +  Repeat  + Nearest
        test_acc[i++] =
            image_acc.read(0.00f, Norm_Repeat_Nearest_sampler); // {1,2,3,4}
        test_acc[i++] =
            image_acc.read(0.25f, Norm_Repeat_Nearest_sampler); // {49,48,47,46}
        test_acc[i++] =
            image_acc.read(0.50f, Norm_Repeat_Nearest_sampler); // {59,58,57,56}
        test_acc[i++] =
            image_acc.read(0.75f, Norm_Repeat_Nearest_sampler); // {11,12,13,14}

        // 4-7 read four pixels above right bound,   sample: Normalized + Repeat
        // + Nearest
        test_acc[i++] =
            image_acc.read(1.125f, Norm_Repeat_Nearest_sampler); // {1,2,3,4}
        test_acc[i++] = image_acc.read(
            1.375f, Norm_Repeat_Nearest_sampler); // {49,48,47,46}
        test_acc[i++] = image_acc.read(
            1.625f, Norm_Repeat_Nearest_sampler); // {59,58,57,56}
        test_acc[i++] = image_acc.read(
            1.875f, Norm_Repeat_Nearest_sampler); // {11,12,13,14}
        // 8-11 read four pixels below left bound. sample: Normalized + Repeat +
        // Nearest
        test_acc[i++] =
            image_acc.read(-0.875f, Norm_Repeat_Nearest_sampler); // {1,2,3,4}
        test_acc[i++] = image_acc.read(
            -0.625f, Norm_Repeat_Nearest_sampler); // {49,48,47,46}
        test_acc[i++] = image_acc.read(
            -0.375f, Norm_Repeat_Nearest_sampler); // {59,58,57,56}
        test_acc[i++] = image_acc.read(
            -0.125f, Norm_Repeat_Nearest_sampler); // {11,12,13,14}
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    auto test_acc = testResults.get_access<access::mode::read>();
    for (int i = 0, idx = 0; i < numTests; i++, idx++) {
      if (i == 0) {
        idx = 0;
        std::cout << "read four pixels at low-boundary locations,  sample:   "
                     "Normalized +  Repeat  + Nearest"
                  << std::endl;
      }
      if (i == 4) {
        idx = 0;
        std::cout << "read four pixels above right bound,   sample: Normalized "
                     "+ Repeat + Nearest"
                  << std::endl;
      }
      if (i == 8) {
        idx = 0;
        std::cout << "read four pixels below left bound. sample: Normalized + "
                     "Repeat + Nearest"
                  << std::endl;
      }
      pixelT testPixel = test_acc[i];
      std::cout << i << " -- " << idx << ": ";
      outputPixel(testPixel);
      std::cout << std::endl;
    }
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
    // CUDA has limited support for image_channel_type, so the tests use
    // unsigned_int32
    test_normalized_repeat_nearest_sampler(image_channel_order::rgba,
                                           image_channel_type::unsigned_int32);
  } else {
    std::cout << "device does not support image operations" << std::endl;
  }

  return 0;
}

// clang-format off
// CHECK: read four pixels at low-boundary locations,  sample:   Normalized +  Repeat  + Nearest
// CHECK-NEXT: 0 -- 0: {1,2,3,4}
// CHECK-NEXT: 1 -- 1: {49,48,47,46}
// CHECK-NEXT: 2 -- 2: {59,58,57,56}
// CHECK-NEXT: 3 -- 3: {11,12,13,14}
// CHECK-NEXT: read four pixels above right bound,   sample: Normalized + Repeat + Nearest
// CHECK-NEXT: 4 -- 0: {1,2,3,4}
// CHECK-NEXT: 5 -- 1: {49,48,47,46}
// CHECK-NEXT: 6 -- 2: {59,58,57,56}
// CHECK-NEXT: 7 -- 3: {11,12,13,14}
// CHECK-NEXT: read four pixels below left bound. sample: Normalized + Repeat + Nearest
// CHECK-NEXT: 8 -- 0: {1,2,3,4}
// CHECK-NEXT: 9 -- 1: {49,48,47,46}
// CHECK-NEXT: 10 -- 2: {59,58,57,56}
// CHECK-NEXT: 11 -- 3: {11,12,13,14}
// clang-format on
