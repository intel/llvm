// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_bindless_images_srgb

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Unsampled fetch of an sRGB image_descriptor: hardware must apply the
// IEC 61966-2-1 decode on fetch, returning linear values. A linear
// image_descriptor with the same raw bytes must return the raw unorm value
// unchanged.

#include "../helpers/common.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  sycl::device dev;
  sycl::queue q(dev);

  const size_t width = 4;
  const size_t height = 4;
  const size_t numElems = width * height;

  const uint8_t rawByte = 186;
  const float rawNorm = static_cast<float>(rawByte) / 255.0f;
  const float expectedLinear = bindless_helpers::srgb_to_linear(rawNorm);

  std::vector<uint8_t> input(numElems * 4);
  for (size_t i = 0; i < numElems; i++) {
    input[i * 4 + 0] = rawByte;
    input[i * 4 + 1] = rawByte;
    input[i * 4 + 2] = rawByte;
    // Non-trivial alpha: decoding it would visibly change the value.
    input[i * 4 + 3] = rawByte;
  }

  std::vector<sycl::float4> outputSrgb(numElems);
  std::vector<sycl::float4> outputLinear(numElems);

  try {
    syclexp::image_descriptor srgbDesc(
        sycl::range<2>{width, height}, 4,
        sycl::image_channel_type::ext_oneapi_unorm_int8_srgb);
    syclexp::image_descriptor linearDesc(sycl::range<2>{width, height}, 4,
                                         sycl::image_channel_type::unorm_int8);

    syclexp::image_mem srgbMem(srgbDesc, q);
    syclexp::image_mem linearMem(linearDesc, q);

    q.ext_oneapi_copy(input.data(), srgbMem.get_handle(), srgbDesc);
    q.ext_oneapi_copy(input.data(), linearMem.get_handle(), linearDesc);
    q.wait_and_throw();

    // Copies are raw byte moves and must never apply the sRGB decode, in
    // either direction or between images.
    {
      syclexp::image_mem srgbCopyMem(srgbDesc, q);
      std::vector<uint8_t> readback(input.size());
      q.ext_oneapi_copy(srgbMem.get_handle(), srgbDesc,
                        srgbCopyMem.get_handle(), srgbDesc);
      // No implicit dependency between bindless image copies.
      q.wait_and_throw();
      q.ext_oneapi_copy(srgbCopyMem.get_handle(), readback.data(), srgbDesc);
      q.wait_and_throw();
      if (readback != input) {
        std::cerr << "sRGB image copy altered raw bytes" << std::endl;
        return 1;
      }
    }

    auto srgbImg = syclexp::create_image(srgbMem, srgbDesc, q);
    auto linearImg = syclexp::create_image(linearMem, linearDesc, q);

    {
      sycl::buffer<sycl::float4, 2> srgbBuf(outputSrgb.data(),
                                            sycl::range<2>{height, width});
      sycl::buffer<sycl::float4, 2> linearBuf(outputLinear.data(),
                                              sycl::range<2>{height, width});

      sycl::range<2> globalSize{height, width};
      sycl::range<2> localSize{1, 1};

      q.submit([&](sycl::handler &cgh) {
        auto acc = srgbBuf.get_access<sycl::access_mode::write>(cgh);
        cgh.parallel_for(
            sycl::nd_range<2>{globalSize, localSize}, [=](sycl::nd_item<2> it) {
              size_t y = it.get_global_id(0);
              size_t x = it.get_global_id(1);
              acc[sycl::id<2>(y, x)] =
                  syclexp::fetch_image<sycl::float4>(srgbImg, sycl::int2(x, y));
            });
      });

      q.submit([&](sycl::handler &cgh) {
        auto acc = linearBuf.get_access<sycl::access_mode::write>(cgh);
        cgh.parallel_for(
            sycl::nd_range<2>{globalSize, localSize}, [=](sycl::nd_item<2> it) {
              size_t y = it.get_global_id(0);
              size_t x = it.get_global_id(1);
              acc[sycl::id<2>(y, x)] = syclexp::fetch_image<sycl::float4>(
                  linearImg, sycl::int2(x, y));
            });
      });
    }

    syclexp::destroy_image_handle(srgbImg, q);
    syclexp::destroy_image_handle(linearImg, q);

    bool passed = true;
    const float epsilon = 0.01f;
    for (size_t i = 0; i < numElems; i++) {
      if (std::abs(outputSrgb[i].x() - expectedLinear) > epsilon ||
          std::abs(outputSrgb[i].y() - expectedLinear) > epsilon ||
          std::abs(outputSrgb[i].z() - expectedLinear) > epsilon) {
        std::cerr << "sRGB decode mismatch at " << i << ": expected "
                  << expectedLinear << ", got (" << outputSrgb[i].x() << ", "
                  << outputSrgb[i].y() << ", " << outputSrgb[i].z() << ")"
                  << std::endl;
        passed = false;
      }
      if (std::abs(outputSrgb[i].w() - rawNorm) > epsilon) {
        std::cerr << "sRGB alpha should not be decoded at " << i
                  << ": expected " << rawNorm << ", got " << outputSrgb[i].w()
                  << std::endl;
        passed = false;
      }
      if (std::abs(outputLinear[i].x() - rawNorm) > epsilon ||
          std::abs(outputLinear[i].y() - rawNorm) > epsilon ||
          std::abs(outputLinear[i].z() - rawNorm) > epsilon) {
        std::cerr << "Linear passthrough mismatch at " << i << ": expected "
                  << rawNorm << ", got (" << outputLinear[i].x() << ", "
                  << outputLinear[i].y() << ", " << outputLinear[i].z() << ")"
                  << std::endl;
        passed = false;
      }
    }

    if (!passed) {
      std::cerr << "Test failed!" << std::endl;
      return 1;
    }
  } catch (sycl::exception &e) {
    std::cerr << "SYCL exception caught! : " << e.what() << std::endl;
    return 2;
  }

  std::cout << "Test passed!" << std::endl;
  return 0;
}
