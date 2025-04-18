// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_bindless_images_2d_usm
// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cmath>
#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/memcpy2d.hpp>
#include <sycl/usm.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();
  const size_t width = 4;
  const size_t height = 5;
  size_t N = width * height;
  size_t widthInBytes = width * sizeof(float);

  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float> dataIn(N);

  for (size_t i = 0; i < width; i++) {
    for (size_t j = 0; j < height; j++) {
      expected[i + (width * j)] = static_cast<float>(i + (width * j));
      dataIn[i + (width * j)] = static_cast<float>(N);
      out[i + (width * j)] = static_cast<float>(N);
    }
  }

  try {
    auto devicePitchAlign = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::image_row_pitch_align>();
    auto deviceMaxPitch =
        dev.get_info<sycl::ext::oneapi::experimental::info::device::
                         max_image_linear_row_pitch>();

    // Pitch requirements:
    //  - pitch % devicePitchAlign == 0
    //  - pitch >= widthInBytes
    //  - pitch <= deviceMaxPitch
    size_t pitch =
        devicePitchAlign *
        static_cast<size_t>(std::ceil(static_cast<float>(widthInBytes) /
                                      static_cast<float>(devicePitchAlign)));
    assert(pitch <= deviceMaxPitch);
    assert((devicePitchAlign & devicePitchAlign >> 1) == 0);

    void *srcMem =
        sycl::aligned_alloc_host(devicePitchAlign, (pitch * height), ctxt);
    void *dstMem =
        sycl::aligned_alloc_host(devicePitchAlign, (pitch * height), ctxt);
    if (srcMem == nullptr || dstMem == nullptr) {
      std::cerr << "Error allocating memory!" << std::endl;
      return 1;
    }

    // Extension: image descriptor
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, 1, sycl::image_channel_type::fp32);

    // Copy pitched data to host USM
    std::vector<sycl::event> depEvents;
    depEvents.push_back(q.ext_oneapi_memcpy2d(srcMem, devicePitchAlign,
                                              expected.data(), widthInBytes,
                                              widthInBytes, height));
    // Set incorrect data to dstMem to ensure later it is correctly
    // overwritten
    depEvents.push_back(q.ext_oneapi_memcpy2d(dstMem, devicePitchAlign,
                                              dataIn.data(), widthInBytes,
                                              widthInBytes, height));

    // Extension: copy pitched data from device USM to device USM
    depEvents = {
        q.ext_oneapi_copy(srcMem, desc, pitch, dstMem, desc, pitch, depEvents)};

    // Copy pitched data from host USM
    q.ext_oneapi_memcpy2d(out.data(), widthInBytes, dstMem, devicePitchAlign,
                          widthInBytes, height, depEvents);

    q.wait_and_throw();
    sycl::free(srcMem, ctxt);
    sycl::free(dstMem, ctxt);
  } catch (sycl::exception &e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  // collect and validate output
  bool validated = true;
  for (size_t i = 0; i < N; i++) {
    bool mismatch = false;
    if (out[i] != expected[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i] << std::endl;
#else
      break;
#endif
    }
  }
  if (validated) {
    std::cout << "Test passed!" << std::endl;
    return 0;
  }

  std::cout << "Test failed!" << std::endl;
  return 3;
}
