// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: hip, cuda
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// CUDA works with image_channel_type::fp32, but not with any 8-bit per channel
// type (such as unorm_int8)

/*
    This file sets up an image, initializes it with data,
    and verifies that the data is sampled correctly with a
    sampler configured NORMALIZED coordinate_normalization_mode
    CLAMP address_mode and LINEAR filter_mode

*/

#include "common.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

// pixel data-type for RGBA operations (which is the minimum image type)
using pixelT = sycl::float4;

// Seven pixels at 'boundary' locations, starting out of bounds
// sample: Normalized + Clamp + Linear
std::vector<pixelT> ref_boundary = {{0, 0, 0, 0},         {0.1, 0.2, 0.3, 0.4},
                                    {0.4, 0.4, 0.4, 0.4}, {0.4, 0.4, 0.4, 0.4},
                                    {0.4, 0.4, 0.4, 0.4}, {0.3, 0.2, 0.1, 0},
                                    {0, 0, 0, 0}};
// Two pixels on either side of first pixel. float coordinates.
// sample: Normalized +Clamp + Linear
std::vector<pixelT> ref_side = {{0.4, 0.4, 0.4, 0.4}, {0.4, 0.4, 0.4, 0.4}};

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

constexpr auto normalized = coordinate_normalization_mode::normalized;
constexpr auto linear = filtering_mode::linear;

void test_normalized_clamp_linear_sampler(image_channel_order ChanOrder,
                                          image_channel_type ChanType) {
  int numTests = 9; // drives the size of the testResults buffer, and the number
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
    auto Norm_Clamp_Linear_sampler =
        sampler(normalized, addressing_mode::clamp, linear);

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

        // 0-6 read seven pixels at 'boundary' locations, starting out of
        // bounds,  sample:   Normalized +  Clamp  + Linear
        test_acc[i++] =
            image_acc.read(-0.25f, Norm_Clamp_Linear_sampler); // {0,0,0,0}
        test_acc[i++] = image_acc.read(
            0.00f,
            Norm_Clamp_Linear_sampler); // {0.1,0.2,0.3,0.4} // interpolating
                                        // with bg color. consistent with
                                        // unnormalized. Doesn't seem 100%
                                        // correct to me, but don't ahve
                                        // anything to compare against presnetly
        test_acc[i++] = image_acc.read(
            0.25f, Norm_Clamp_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            0.50f, Norm_Clamp_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            0.75f, Norm_Clamp_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] =
            image_acc.read(1.00f,
                           Norm_Clamp_Linear_sampler); // {0.3,0.2,0.1,0}  //
                                                       // interpolating with bg
        test_acc[i++] =
            image_acc.read(1.25f, Norm_Clamp_Linear_sampler); // {0,0,0,0}

        // 7-8 read two pixels on either side of first pixel. float coordinates.
        // CLAMP
        //  on GPU CLAMP is apparently stopping the interpolation. ( values on
        //  right are expected value)
        test_acc[i++] = image_acc.read(
            0.2499999f, Norm_Clamp_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            0.2500001f, Norm_Clamp_Linear_sampler); // {0.4,0.4,0.4,0.4}
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    size_t offset = 0;
    auto test_acc = testResults.get_host_access();
    std::cout << "read seven pixels at 'boundary' locations, starting out of "
                 "bounds,  sample:   Normalized +  Clamp  + Linear"
              << std::endl;
    check_pixels(test_acc, ref_boundary, offset);

    std::cout << "read two pixels on either side of first pixel. float "
                 "coordinates. Normalized +  Clamp  + Linear"
              << std::endl;
    check_pixels(test_acc, ref_side, offset);
  } // ~image / ~buffer
}

int main() {

  queue Q;

  // the _int8 channels are one byte per channel, or four bytes per pixel (for
  // RGBA) the _int16/fp16 channels are two bytes per channel, or eight bytes
  // per pixel (for RGBA) the _int32/fp32  channels are four bytes per
  // channel, or sixteen bytes per pixel (for RGBA).

  std::cout << "fp32 -------------" << std::endl;
  test_normalized_clamp_linear_sampler(image_channel_order::rgba,
                                       image_channel_type::fp32);

  std::cout << "unorm_int8 -------" << std::endl;
  test_normalized_clamp_linear_sampler(image_channel_order::rgba,
                                       image_channel_type::unorm_int8);

  return 0;
}
