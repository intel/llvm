// RUN: %clangxx -fsycl  -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER

// XFAIL: level_zero
// UNSUPPORTED: cuda
// UNSUPPORTED: hip

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

using accessorPixelT = sycl::float4;
using dataPixelT = uint32_t;

// will output a pixel as {r,g,b,a}.  provide override if a different pixelT is
// defined.
void outputPixel(sycl::float4 somePixel) {
  std::cout << "{" << somePixel[0] << "," << somePixel[1] << "," << somePixel[2]
            << "," << somePixel[3] << "} ";
}

constexpr long width = 4;
constexpr long height = 3;

void test_rd(image_channel_order ChanOrder, image_channel_type ChanType) {

  int numTests = 4; // drives the size of the testResults buffer, and the number
                    // of report iterations. Kludge.

  // this should yield a read of approximate 0.5 for each channel
  // when read directly with a normal non-linearized image (e.g.
  // image_channel_order::rgba). For sRGB
  // (image_channel_order::ext_oneapi_srgba), this is the value with maximum
  // conversion. So we should read values of approximately 0.2
  dataPixelT basicPixel{127 << 24 | 127 << 16 | 127 << 8 | 127};

  queue Q;
  const sycl::range<2> ImgRange_2D(width, height);

  // IMPORTANT: const data is *required* for sRGBA images.
  // OpenCL support is limited for 2D/3D images that are read only.
  const std::vector<dataPixelT> ImgData(ImgRange_2D.size(), basicPixel);
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

#ifdef SYCL_EXT_ONEAPI_SRGB
  std::cout << "SYCL_EXT_ONEAPI_SRGB defined" << std::endl;
#endif

  queue Q;
  device D = Q.get_device();

  // test aspect
  if (D.has(aspect::ext_oneapi_srgb))
    std::cout << "aspect::ext_oneapi_srgb detected" << std::endl;

  if (D.has(aspect::image)) {
    // RGBA -- (normal, non-linearized)
    std::cout << "rgba -------" << std::endl;
    test_rd(image_channel_order::rgba, image_channel_type::unorm_int8);

    // sRGBA -- (linearized reads)
    std::cout << "srgba -------" << std::endl;
    test_rd(image_channel_order::ext_oneapi_srgba,
            image_channel_type::unorm_int8);
  } else {
    std::cout << "device does not support image operations" << std::endl;
  }

  return 0;
}

// clang-format off
// CHECK: SYCL_EXT_ONEAPI_SRGB defined
// CHECK: aspect::ext_oneapi_srgb detected

// CHECK: rgba -------
// CHECK-NEXT: read four pixels, no sampler
//   these next four reads should all be close to 0.5
// CHECK-NEXT: 0: {0.498039,0.498039,0.498039,0.498039} 
// CHECK-NEXT: 1: {0.498039,0.498039,0.498039,0.498039} 
// CHECK-NEXT: 2: {0.498039,0.498039,0.498039,0.498039} 
// CHECK-NEXT: 3: {0.498039,0.498039,0.498039,0.498039} 
// CHECK: srgba -------
// CHECK-NEXT: read four pixels, no sampler
//   these next four reads should have R, G, B values close to 0.2 
//   presently the values differ slightly between OpenCL GPU and CPU
// (e.g. GPU: 0.21231, CPU: 0.211795 )
// CHECK-NEXT: 0: {0.21 
// CHECK-NEXT: 1: {0.21 
// CHECK-NEXT: 2: {0.21 
// CHECK-NEXT: 3: {0.21
// clang-format on
