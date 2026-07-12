// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: preview-breaking-changes-supported

// UNSUPPORTED: cuda, hip
// UNSUPPORTED-INTENDED: sRGB hardware decode (UR_IMAGE_CHANNEL_ORDER_SRGBA) is
// only implemented in the Level Zero adapter in this change.

// RUN: %clangxx -fsycl -fpreview-breaking-changes %s -Wno-error=unused-command-line-argument -o %t.out
// RUN: %{run} %t.out

// Unsampled fetch of an sRGB image_descriptor: hardware must apply the
// IEC 61966-2-1 decode on fetch, returning linear values. A linear
// image_descriptor with the same raw bytes must return the raw unorm value
// unchanged.

#include <cmath>
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

static float srgb_to_linear(float c) {
  if (c <= 0.04045f)
    return c / 12.92f;
  return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

int main() {
  sycl::device dev;
  sycl::queue q(dev);

  const size_t width = 4;
  const size_t height = 4;
  const size_t numElems = width * height;

  const uint8_t rawByte = 186;
  const float rawNorm = static_cast<float>(rawByte) / 255.0f;
  const float expectedLinear = srgb_to_linear(rawNorm);

  std::vector<uint8_t> input(numElems * 4);
  for (size_t i = 0; i < numElems; i++) {
    input[i * 4 + 0] = rawByte;
    input[i * 4 + 1] = rawByte;
    input[i * 4 + 2] = rawByte;
    input[i * 4 + 3] = 255;
  }

  std::vector<sycl::float4> outputSrgb(numElems);
  std::vector<sycl::float4> outputLinear(numElems);

  try {
    syclexp::image_descriptor srgbDesc(
        sycl::range<2>{width, height}, 4,
        sycl::image_channel_type::unorm_int8,
        syclexp::image_color_space::srgb);
    syclexp::image_descriptor linearDesc(
        sycl::range<2>{width, height}, 4,
        sycl::image_channel_type::unorm_int8);

    syclexp::image_mem srgbMem(srgbDesc, q);
    syclexp::image_mem linearMem(linearDesc, q);

    q.ext_oneapi_copy(input.data(), srgbMem.get_handle(), srgbDesc);
    q.ext_oneapi_copy(input.data(), linearMem.get_handle(), linearDesc);
    q.wait_and_throw();

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
        cgh.parallel_for(sycl::nd_range<2>{globalSize, localSize},
                         [=](sycl::nd_item<2> it) {
                           size_t dim0 = it.get_global_id(0);
                           size_t dim1 = it.get_global_id(1);
                           acc[sycl::id<2>(dim0, dim1)] =
                               syclexp::fetch_image<sycl::float4>(
                                   srgbImg, sycl::int2(dim0, dim1));
                         });
      });

      q.submit([&](sycl::handler &cgh) {
        auto acc = linearBuf.get_access<sycl::access_mode::write>(cgh);
        cgh.parallel_for(sycl::nd_range<2>{globalSize, localSize},
                         [=](sycl::nd_item<2> it) {
                           size_t dim0 = it.get_global_id(0);
                           size_t dim1 = it.get_global_id(1);
                           acc[sycl::id<2>(dim0, dim1)] =
                               syclexp::fetch_image<sycl::float4>(
                                   linearImg, sycl::int2(dim0, dim1));
                         });
      });

      q.wait_and_throw();
    }

    syclexp::destroy_image_handle(srgbImg, q);
    syclexp::destroy_image_handle(linearImg, q);

    bool passed = true;
    const float epsilon = 0.01f;
    for (size_t i = 0; i < numElems; i++) {
      if (std::abs(outputSrgb[i].x() - expectedLinear) > epsilon) {
        std::cerr << "sRGB decode mismatch at " << i << ": expected "
                  << expectedLinear << ", got " << outputSrgb[i].x()
                  << std::endl;
        passed = false;
      }
      if (std::abs(outputLinear[i].x() - rawNorm) > epsilon) {
        std::cerr << "Linear passthrough mismatch at " << i << ": expected "
                  << rawNorm << ", got " << outputLinear[i].x() << std::endl;
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