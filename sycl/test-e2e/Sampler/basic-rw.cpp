// REQUIRES: aspect-ext_intel_legacy_image
// UNSUPPORTED: hip
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

/*
    This file sets up an image, initializes it with data, and verifies that the
   data can be read directly.

    Use it as a base file for testing any condition.

    clang++ -fsycl -sycl-std=121 -o binx.bin basic-rw.cpp

    ONEAPI_DEVICE_SELECTOR=opencl:gpu ./binx.bin
    ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./binx.bin
    ONEAPI_DEVICE_SELECTOR=opencl:cpu ./binx.bin

    ONEAPI_DEVICE_SELECTOR=opencl:fpga ../binx.bin    <--  does not support
   image operations at this time.

*/

#include "common.hpp"
#include <sycl/sycl.hpp>

using namespace sycl;

// pixel data-type for RGBA operations (which is the minimum image type)
using pixelT = sycl::uint4;

// Reference for four pixels, no sampler
std::vector<pixelT> ref = {
    {1, 2, 3, 4}, {49, 48, 47, 46}, {59, 58, 57, 56}, {11, 12, 13, 14}};

// 4 pixels on a side. 1D at the moment
constexpr long width = 4;

void test_rw(image_channel_order ChanOrder, image_channel_type ChanType) {
  int numTests = 4; // drives the size of the testResults buffer, and the number
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

    event E_Test = Q.submit([&](handler &cgh) {
      auto image_acc = image_1D.get_access<pixelT, access::mode::read>(cgh);
      auto test_acc = testResults.get_access<access::mode::write>(cgh);

      cgh.single_task<class im1D_Unorm_Linear>([=]() {
        int i = 0; // the index for writing into the testResult buffer.

        // verify our four pixels were set up correctly.
        // 0-3 read four pixels. no sampler
        test_acc[i++] = image_acc.read(0); // {1,2,3,4}
        test_acc[i++] = image_acc.read(1); // {49,48,47,46}
        test_acc[i++] = image_acc.read(2); // {59,58,57,56}
        test_acc[i++] = image_acc.read(3); // {11,12,13,14}

        // Add more tests below. Just be sure to increase the numTests counter
        // at the beginning of this function
      });
    });
    E_Test.wait();

    // REPORT RESULTS
    size_t offset = 0;
    auto test_acc = testResults.get_host_access();
    std::cout << "read four pixels, no sampler" << std::endl;
    check_pixels(test_acc, ref, offset);
  } // ~image / ~buffer
}

int main() {

  queue Q;
  device D = Q.get_device();

  // the _int8 channels are one byte per channel, or four bytes per pixel (for
  // RGBA) the _int16/fp16 channels are two bytes per channel, or eight bytes
  // per pixel (for RGBA) the _int32/fp32  channels are four bytes per
  // channel, or sixteen bytes per pixel (for RGBA).
  // CUDA has limited support for image_channel_type, so the tests use
  // unsigned_int32
  test_rw(image_channel_order::rgba, image_channel_type::unsigned_int32);

  return 0;
}
