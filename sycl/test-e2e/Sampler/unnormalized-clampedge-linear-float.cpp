// UNSUPPORTED: hip, cuda
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

// CUDA works with image_channel_type::fp32, but not with any 8-bit per channel
// type (such as unorm_int8)

/*
    This file sets up an image, initializes it with data,
    and verifies that the data is sampled correctly with a
    sampler configured UNNORMALIZED coordinate_normalization_mode
    CLAMPEDGE address_mode and LINEAR filter_mode
*/

#include "common.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

using pixelT = sycl::float4;

// Six pixels, float coordinates, sample: NonNormalized + ClampEDGE + Linear
std::vector<pixelT> ref_pixel = {{0.2, 0.4, 0.6, 0.8}, {0.2, 0.4, 0.6, 0.8},
                                 {0.4, 0.4, 0.4, 0.4}, {0.4, 0.4, 0.4, 0.4},
                                 {0.4, 0.4, 0.4, 0.4}, {0.6, 0.4, 0.2, 0}};
// Two pixels on either side of 1. float coordinates.ClampEDGE
std::vector<pixelT> ref_side = {{0.4, 0.4, 0.4, 0.4}, {0.4, 0.4, 0.4, 0.4}};

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

constexpr auto unnormalized = coordinate_normalization_mode::unnormalized;
constexpr auto clamp_edge = addressing_mode::clamp_to_edge;
constexpr auto linear = filtering_mode::linear;

void test_unnormalized_clampedge_linear_sampler(image_channel_order ChanOrder,
                                                image_channel_type ChanType) {
  int numTests = 8; // drives the size of the testResults buffer, and the number
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
    auto UnNorm_ClampEdge_Linear_sampler =
        sampler(unnormalized, clamp_edge, linear);

    event E_Test = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::read>(cgh);
      auto test_acc = testResults.get_access<access::mode::write>(cgh);

      cgh.single_task<class im1D_Unorm_Linear>([=]() {
        int i = 0; // the index for writing into the testResult buffer.

        // UnNormalized Pixel Locations when using Linear Interpolation
        // (0 -------- ](1 ---------- ](2 ----------- ](3---------- ](4)

        // 0-5 read six pixels, float coordinates,   sample:   NonNormalized +
        // ClampEDGE + Linear
        test_acc[i++] = image_acc.read(
            -1.0f, UnNorm_ClampEdge_Linear_sampler); // {0.2,0.4,0.6,0.8}
        test_acc[i++] = image_acc.read(
            0.0f, UnNorm_ClampEdge_Linear_sampler); // {0.2,0.4,0.6,0.8}
        test_acc[i++] = image_acc.read(
            1.0f,
            UnNorm_ClampEdge_Linear_sampler); // {0.4,0.4,0.4,0.4}
                                              // //interpolated
        test_acc[i++] = image_acc.read(
            2.0f, UnNorm_ClampEdge_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            3.0f, UnNorm_ClampEdge_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            4.0f, UnNorm_ClampEdge_Linear_sampler); // {0.6,0.4,0.2,0}

        // 6-7 read two pixels on either side of 1. float coordinates. ClampEDGE
        //  ClampEDGE correctly interpolates
        test_acc[i++] = image_acc.read(
            0.9999999f, UnNorm_ClampEdge_Linear_sampler); // {0.4,0.4,0.4,0.4}
        test_acc[i++] = image_acc.read(
            1.0000001f, UnNorm_ClampEdge_Linear_sampler); // {0.4,0.4,0.4,0.4}
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    size_t offset = 0;
    auto test_acc = testResults.get_access<access::mode::read>();

    std::cout << "read six pixels, float coordinates,   sample:   "
                 "NonNormalized + ClampEDGE + Linear"
              << std::endl;
    check_pixels(test_acc, ref_pixel, offset);

    std::cout << "read two pixels on either side of 1. float coordinates. "
                 "ClampEDGE"
              << std::endl;
    check_pixels(test_acc, ref_pixel, offset);
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

    std::cout << "fp32 -------------" << std::endl;
    test_unnormalized_clampedge_linear_sampler(image_channel_order::rgba,
                                               image_channel_type::fp32);

    std::cout << "unorm_int8 -------" << std::endl;
    test_unnormalized_clampedge_linear_sampler(image_channel_order::rgba,
                                               image_channel_type::unorm_int8);

  } else {
    std::cout << "device does not support image operations" << std::endl;
  }

  return 0;
}
