// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: hip, cuda
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// CUDA is not handling repeat or mirror correctly with normalized coordinates.
// Waiting on a fix.

/*
    This file sets up an image, initializes it with data,
    and verifies that the data is sampled correctly with a
    sampler configured NORMALIZED coordinate_normalization_mode
    MIRROR_REPEAT address_mode and NEAREST filter_mode

*/

#include "common.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

// pixel data-type for RGBA operations (which is the minimum image type)
using pixelT = sycl::uint4;

// Four pixels at low-boundary locations
// sample: Normalized + Mirrored Repeat + Nearest
std::vector<pixelT> ref_boundary = {
    {1, 2, 3, 4}, {49, 48, 47, 46}, {59, 58, 57, 56}, {11, 12, 13, 14}};
// Four pixels outside rightmost boundary
// sample: Normalized + Mirrored Repeat + Nearest
std::vector<pixelT> ref_right = {
    {1, 2, 3, 4}, {49, 48, 47, 46}, {59, 58, 57, 56}, {11, 12, 13, 14}};
// Four pixels outside leftmost boundary
// sample: Normalized + Mirrored Repeat + Nearest
std::vector<pixelT> ref_left = {
    {1, 2, 3, 4}, {49, 48, 47, 46}, {59, 58, 57, 56}, {11, 12, 13, 14}};

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

constexpr auto normalized = coordinate_normalization_mode::normalized;
constexpr auto mirrored = addressing_mode::mirrored_repeat;
constexpr auto nearest = filtering_mode::nearest;

void test_normalized_mirrored_nearest_sampler(image_channel_order ChanOrder,
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
    auto Norm_Mirror_Nearest_sampler = sampler(normalized, mirrored, nearest);

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
        // + Mirrored Repeat  + Nearest
        test_acc[i++] =
            image_acc.read(0.00f, Norm_Mirror_Nearest_sampler); // {1,2,3,4}
        test_acc[i++] =
            image_acc.read(0.25f, Norm_Mirror_Nearest_sampler); // {49,48,47,46}
        test_acc[i++] =
            image_acc.read(0.50f, Norm_Mirror_Nearest_sampler); // {59,58,57,56}
        test_acc[i++] =
            image_acc.read(0.75f, Norm_Mirror_Nearest_sampler); // {11,12,13,14}

        // 4-7 read four pixels outside rightmost boundary,   sample: Normalized
        // + Mirrored Repeat + Nearest
        test_acc[i++] =
            image_acc.read(1.875f, Norm_Mirror_Nearest_sampler); // {1,2,3,4}
        test_acc[i++] = image_acc.read(
            1.625f, Norm_Mirror_Nearest_sampler); // {49,48,47,46}
        test_acc[i++] = image_acc.read(
            1.375f, Norm_Mirror_Nearest_sampler); // {59,58,57,56}
        test_acc[i++] = image_acc.read(
            1.125f, Norm_Mirror_Nearest_sampler); // {11,12,13,14}
        // 8-11 read four pixels outside leftmost boundary,   sample: Normalized
        // + Mirrored Repeat + Nearest
        test_acc[i++] =
            image_acc.read(-0.125f, Norm_Mirror_Nearest_sampler); // {1,2,3,4}
        test_acc[i++] = image_acc.read(
            -0.375f, Norm_Mirror_Nearest_sampler); // {49,48,47,46}
        test_acc[i++] = image_acc.read(
            -0.625f, Norm_Mirror_Nearest_sampler); // {59,58,57,56}
        test_acc[i++] = image_acc.read(
            -0.875f, Norm_Mirror_Nearest_sampler); // {11,12,13,14}
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    size_t offset = 0;
    auto test_acc = testResults.get_host_access();
    std::cout << "read four pixels at low-boundary locations,  sample:   "
                 "Normalized + Mirrored Repeat  + Nearest"
              << std::endl;
    check_pixels(test_acc, ref_boundary, offset);

    std::cout << "read four pixels outside rightmost boundary,   sample: "
                 "Normalized + Mirrored Repeat + Nearest"
              << std::endl;
    check_pixels(test_acc, ref_right, offset);

    std::cout << "read four pixels outside leftmost boundary,   sample: "
                 "Normalized + Mirrored Repeat + Nearest"
              << std::endl;
    check_pixels(test_acc, ref_left, offset);
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
  test_normalized_mirrored_nearest_sampler(image_channel_order::rgba,
                                           image_channel_type::unsigned_int32);

  return 0;
}
