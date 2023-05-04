// REQUIRES: aspect-ext_intel_legacy_image
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Missing __spirv_ImageWrite, __spirv_SampledImage,
// __spirv_ImageSampleExplicitLod on AMD
// XFAIL: hip_amd

/*
    This file sets up an image, initializes it with data,
    and verifies that the data is sampled correctly with a
    sampler configured NORMALIZED coordinate_normalization_mode
    CLAMPEDGE address_mode and NEAREST filter_mode

*/

#include "common.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

// pixel data-type for RGBA operations (which is the minimum image type)
using pixelT = sycl::uint4;

// Four pixels at low-boundary locations
// sample: Normalized + Clamp To Edge + Nearest
std::vector<pixelT> ref_in = {
    {1, 2, 3, 4}, {49, 48, 47, 46}, {59, 58, 57, 56}, {11, 12, 13, 14}};
// Three pixels out of bounds, sample: Normalized + Clamp To Edge + Nearest
std::vector<pixelT> ref_out = {
    {1, 2, 3, 4}, {11, 12, 13, 14}, {11, 12, 13, 14}};

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

constexpr auto normalized = coordinate_normalization_mode::normalized;
constexpr auto clamp_edge = addressing_mode::clamp_to_edge;
constexpr auto nearest = filtering_mode::nearest;

void test_normalized_clampedge_nearest_sampler(image_channel_order ChanOrder,
                                               image_channel_type ChanType) {
  int numTests = 7; // drives the size of the testResults buffer, and the number
                    // of report iterations. Kludge.

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
    auto Norm_ClampEdge_Nearest_sampler =
        sampler(normalized, clamp_edge, nearest);

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
        // +  Clamp To Edge  + Nearest
        test_acc[i++] =
            image_acc.read(0.00f, Norm_ClampEdge_Nearest_sampler); // {1,2,3,4}
        test_acc[i++] = image_acc.read(
            0.25f, Norm_ClampEdge_Nearest_sampler); // {49,48,47,46}
        test_acc[i++] = image_acc.read(
            0.50f, Norm_ClampEdge_Nearest_sampler); // {59,58,57,56}
        test_acc[i++] = image_acc.read(
            0.75f, Norm_ClampEdge_Nearest_sampler); // {11,12,13,14}

        // 4-6 read three pixels out of buonds,   sample:   Normalized + Clamp
        // To Edge + Nearest
        test_acc[i++] = image_acc.read(
            -0.125f, Norm_ClampEdge_Nearest_sampler); // {1,2,3,4}
        test_acc[i++] = image_acc.read(
            1.0f, Norm_ClampEdge_Nearest_sampler); // {11,12,13,14}
        test_acc[i++] = image_acc.read(
            1.125f, Norm_ClampEdge_Nearest_sampler); // {11,12,13,14}
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    size_t offset = 0;
    auto test_acc = testResults.get_access<access::mode::read>();
    std::cout << "read four pixels at low-boundary locations,  sample:   "
                 "Normalized +  Clamp To Edge  + Nearest"
              << std::endl;
    check_pixels(test_acc, ref_in, offset);

    std::cout << "read three pixels out of buonds,   sample:   Normalized "
                 "+ Clamp To Edge + Nearest"
              << std::endl;
    check_pixels(test_acc, ref_out, offset);
  } // ~image / ~buffer
}

int main() {

  queue Q;

  // the _int8 channels are one byte per channel, or four bytes per pixel (for
  // RGBA) the _int16/fp16 channels are two bytes per channel, or eight bytes
  // per pixel (for RGBA) the _int32/fp32  channels are four bytes per
  // channel, or sixteen bytes per pixel (for RGBA).
  // CUDA has limited support for image_channel_type, so the tests use
  // unsigned_int32
  test_normalized_clampedge_nearest_sampler(image_channel_order::rgba,
                                            image_channel_type::unsigned_int32);

  return 0;
}
