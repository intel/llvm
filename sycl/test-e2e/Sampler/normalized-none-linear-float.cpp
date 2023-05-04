// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: hip, cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// CUDA works with image_channel_type::fp32, but not with any 8-bit per channel
// type (such as unorm_int8)

/*
    This file sets up an image, initializes it with data,
    and verifies that the data is sampled correctly with a
    sampler configured NORMALIZED coordinate_normalization_mode
    NONE address_mode and LINEAR filter_mode

*/

#include "common.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

// pixel data-type for RGBA operations (which is the minimum image type)
using pixelT = sycl::float4;

// Three pixels at inner boundary locations, sample: Normalized + None + Linear
std::vector<pixelT> ref_in = {
    {0.4, 0.4, 0.4, 0.4}, {0.4, 0.4, 0.4, 0.4}, {0.4, 0.4, 0.4, 0.4}};
// Four pixels at exact center locations, sample: Normalized + None + Linear
std::vector<pixelT> ref_center = {{0.2, 0.4, 0.6, 0.8},
                                  {0.6, 0.4, 0.2, 0},
                                  {0.2, 0.4, 0.6, 0.8},
                                  {0.6, 0.4, 0.2, 0}};

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

constexpr auto normalized = coordinate_normalization_mode::normalized;
constexpr auto none = addressing_mode::none;
constexpr auto linear = filtering_mode::linear;

void test_normalized_none_linear_sampler(image_channel_order ChanOrder,
                                         image_channel_type ChanType) {
  int numTests = 7; // drives the size of the testResults buffer, and the number
                    // of report iterations. Kludge.

  // we'll use these four pixels for our image. Makes it easy to measure
  // interpolation and spot "off-by-one" probs.
  // These values will work consistently with different levels of float
  // precision (like unorm_int8 vs. fp32)
  pixelT leftEdge{0.2f, 0.4f, 0.6f, 0.8f};
  pixelT body{0.6f, 0.4f, 0.2f, 0.0f};
  pixelT bony{0.2f, 0.4f, 0.6f, 0.8f};
  pixelT rightEdge{0.6f, 0.4f, 0.2f, 0.0f};

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
    auto Norm_None_Linear_sampler = sampler(normalized, none, linear);

    event E_Test = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::read>(cgh);
      auto test_acc = testResults.get_access<access::mode::write>(cgh);

      cgh.single_task<class im1D_norm_linear>([=]() {
        int i = 0; // the index for writing into the testResult buffer.

        // clang-format off
        // Normalized Pixel Locations.  
        //      .125        .375        .625        .875            <-- exact center
        //  |-----^-----|-----^-----|-----^-----|-----^-----
        //[0.0         .25         .50         .75          (1)     <-- low boundary (included in pixel)
        //                                                              upper boundary inexact. (e.g. .2499999)
        // clang-format on

        // 0-2 read three pixels at inner boundary locations,  sample:
        // Normalized +  None  + Linear
        test_acc[i++] = image_acc.read(
            0.25f, Norm_None_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            0.50f, Norm_None_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            0.75f, Norm_None_Linear_sampler); // {0.4,0.4,0.4,0.4}

        // 3-6 read four pixels at exact center locations,  sample:   Normalized
        // +  None  + Linear
        test_acc[i++] = image_acc.read(
            0.125f, Norm_None_Linear_sampler); // {0.2,0.4,0.6,0.8}
        test_acc[i++] =
            image_acc.read(0.375f, Norm_None_Linear_sampler); // {0.6,0.4,0.2,0}
        test_acc[i++] = image_acc.read(
            0.625f, Norm_None_Linear_sampler); // {0.2,0.4,0.6,0.8}
        test_acc[i++] =
            image_acc.read(0.875f, Norm_None_Linear_sampler); // {0.6,0.4,0.2,0}
        // OCL:CPU returning inconsistent result, same thing happening for
        // unnormalized-none-linear.  Temporarily skippin check (below) until
        // fixed.
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    size_t offset = 0;
    auto test_acc = testResults.get_access<access::mode::read>();
    std::cout << "read three pixels at inner boundary locations,  sample:  "
                 " Normalized +  None  + Linear"
              << std::endl;
    check_pixels(test_acc, ref_in, offset);

    std::cout << "read four pixels at exact center locations,  sample:   "
                 "Normalized +  None  + Linear"
              << std::endl;
    check_pixels(test_acc, ref_center, offset);
  } // ~image / ~buffer
}

int main() {

  queue Q;
  // the _int8 channels are one byte per channel, or four bytes per pixel (for
  // RGBA) the _int16/fp16 channels are two bytes per channel, or eight bytes
  // per pixel (for RGBA) the _int32/fp32  channels are four bytes per
  // channel, or sixteen bytes per pixel (for RGBA).

  std::cout << "fp32 -------------" << std::endl;
  test_normalized_none_linear_sampler(image_channel_order::rgba,
                                      image_channel_type::fp32);

  std::cout << "unorm_int8 -------" << std::endl;
  test_normalized_none_linear_sampler(image_channel_order::rgba,
                                      image_channel_type::unorm_int8);

  return 0;
}
