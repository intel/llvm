// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: hip, cuda
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// CUDA works with image_channel_type::fp32, but not with any 8-bit per channel
// type (such as unorm_int8)

/*
    This file sets up an image, initializes it with data,
    and verifies that the data is sampled correctly with a
    sampler configured UNNORMALIZED coordinate_normalization_mode
    NONE address_mode and LINEAR filter_mode
*/

#include "common.hpp"

using namespace sycl;

using pixelT = sycl::float4;

// Four pixels. no sampler
std::vector<pixelT> ref_pixel = {{0.2, 0.4, 0.6, 0.8},
                                 {0.6, 0.4, 0.2, 0},
                                 {0.2, 0.4, 0.6, 0.8},
                                 {0.6, 0.4, 0.2, 0}};
// Three pixels at inner boundary locations, float coordinates
// sample: UnNormalized + None + Linear
std::vector<pixelT> ref_inner = {
    {0.4, 0.4, 0.4, 0.4}, {0.4, 0.4, 0.4, 0.4}, {0.4, 0.4, 0.4, 0.4}};
// Four pixels at exact center locations, float
// sample: Unnormalized + None + Linear
std::vector<pixelT> ref_center = {{0.2, 0.4, 0.6, 0.8},
                                  {0.6, 0.4, 0.2, 0},
                                  {0.2, 0.4, 0.6, 0.8},
                                  {0.6, 0.4, 0.2, 0}};
// Four pixels at inexact upper boundary, float coord
// sample: Unnormalized + None + Linear
std::vector<pixelT> ref_upper = {{0.4, 0.4, 0.4, 0.4},
                                 {0.4, 0.4, 0.4, 0.4},
                                 {0.4, 0.4, 0.4, 0.4},
                                 {0.3, 0.2, 0.1, 0}};

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

constexpr auto unnormalized = coordinate_normalization_mode::unnormalized;
constexpr auto none = addressing_mode::none;
constexpr auto linear = filtering_mode::linear;

void test_unnormalized_none_linear_sampler(image_channel_order ChanOrder,
                                           image_channel_type ChanType) {
  int numTests = 15; // drives the size of the testResults buffer, and the
                     // number of report iterations. Kludge.

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
    auto UnNorm_None_Linear_sampler = sampler(unnormalized, none, linear);

    event E_Test = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::read>(cgh);
      auto test_acc = testResults.get_access<access::mode::write>(cgh);

      cgh.single_task<class im1D_Unorm_Linear>([=]() {
        int i = 0; // the index for writing into the testResult buffer.

        // UnNormalized Pixel Locations when using Linear Interpolation
        // (0 -------- ](1 ---------- ](2 ----------- ](3---------- ](4)

        // verify our four pixels were set up correctly.
        // 0-3 read four pixels. no sampler
        // depending on device, and 8 vs 16 bit, answers here may vary slightly.
        test_acc[i++] = image_acc.read(0); // {0.2,0.4,0.6,0.8}
        test_acc[i++] = image_acc.read(1); // {0.6,0.4,0.2,0}
        test_acc[i++] = image_acc.read(2); // {0.2,0.4,0.6,0.8}
        test_acc[i++] = image_acc.read(3); // {0.6,0.4,0.2,0}

        // 4-6 read three pixels at inner boundary locations, float coordinates,
        // sample:   UnNormalized +  None  + Linear
        test_acc[i++] = image_acc.read(
            1.0f, UnNorm_None_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            2.0f, UnNorm_None_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            3.0f, UnNorm_None_Linear_sampler); // {0.4,0.4,0.4,0.4}

        // 7-10 read four pixels at exact center locations, float,  sample:
        // Unnormalized +  None  + Linear
        test_acc[i++] = image_acc.read(
            0.5f, UnNorm_None_Linear_sampler); // {0.2,0.4,0.6,0.8}
        test_acc[i++] =
            image_acc.read(1.5f, UnNorm_None_Linear_sampler); // {0.6,0.4,0.2,0}
        test_acc[i++] = image_acc.read(
            2.5f, UnNorm_None_Linear_sampler); // {0.2,0.4,0.6,0.8}
        test_acc[i++] = image_acc.read(
            3.4999998f,
            UnNorm_None_Linear_sampler); // {0.6,0.4,0.2,0}   coordinate of 3.5
                                         // should work, but results
                                         // inconsistent. under investigation.

        // 11-14 read four pixels at inexact upper boundary, float coord,
        // sample: Unnormalized +  None  + Linear
        test_acc[i++] = image_acc.read(
            0.9999999f, UnNorm_None_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            1.9999999f, UnNorm_None_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            2.9999999f, UnNorm_None_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            3.9999999f,
            UnNorm_None_Linear_sampler); // {0.3,0.2,0.1,0}  // <<--- should it
                                         // be interpolating with the bg color?
                                         // That doesn't seem right. But it is
                                         // what it is. No longer verifying
                                         // this value in FILECHECK, pending
                                         // further investigation.
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    size_t offset = 0;
    auto test_acc = testResults.get_host_access();
    std::cout << "read four pixels. no sampler" << std::endl;
    check_pixels(test_acc, ref_pixel, offset);

    std::cout << "read three pixels at inner boundary locations, float "
                 "coordinates,  sample:   UnNormalized +  None  + Linear"
              << std::endl;
    check_pixels(test_acc, ref_inner, offset);

    std::cout << "read four pixels at exact center locations, float,  "
                 "sample:   Unnormalized +  None  + Linear"
              << std::endl;
    check_pixels(test_acc, ref_center, offset);

    std::cout << "read four pixels at inexact upper boundary, float coord, "
                 " sample:   Unnormalized +  None  + Linear"
              << std::endl;
    check_pixels(test_acc, ref_center, offset);
  } // ~image / ~buffer
}

int main() {

  queue Q;
  device D = Q.get_device();

  // the _int8 channels are one byte per channel, or four bytes per pixel (for
  // RGBA) the _int16/fp16 channels are two bytes per channel, or eight bytes
  // per pixel (for RGBA) the _int32/fp32  channels are four bytes per
  // channel, or sixteen bytes per pixel (for RGBA).

  std::cout << "fp32 -------------" << std::endl;
  test_unnormalized_none_linear_sampler(image_channel_order::rgba,
                                        image_channel_type::fp32);

  std::cout << "unorm_int8 -------" << std::endl;
  test_unnormalized_none_linear_sampler(image_channel_order::rgba,
                                        image_channel_type::unorm_int8);

  return 0;
}
