// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_bindless_images_2d_usm
// REQUIRES: cuda
//
// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/17231

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <cmath>
#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();
  const size_t width = 3;
  const size_t height = 5;
  size_t N = width * height;

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
    // Extension: image descriptor
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, 1, sycl::image_channel_type::fp32);

    size_t src_pitch = 0;
    size_t dst_pitch = 0;

    // Extension: returns the device pointer to USM allocated pitched memory
    void *srcMem = sycl::ext::oneapi::experimental::pitched_alloc_device(
        &src_pitch, desc, q);
    auto dstMem = sycl::ext::oneapi::experimental::pitched_alloc_device(
        &dst_pitch, desc, q);

    if (srcMem == nullptr || dstMem == nullptr) {
      std::cerr << "Error allocating memory!" << std::endl;
      return 1;
    }

    // Extension: copy pitched data to device USM
    std::vector<sycl::event> depEvents;
    depEvents.push_back(
        q.ext_oneapi_copy(expected.data(), srcMem, desc, src_pitch));
    // Extension: set incorrect data to dstMem to ensure later it is correctly
    // overwritten
    depEvents.push_back(
        q.ext_oneapi_copy(dataIn.data(), dstMem, desc, dst_pitch));

    // Extension: copy pitched data from device USM to device USM
    depEvents = {q.ext_oneapi_copy(srcMem, desc, src_pitch, dstMem, desc,
                                   dst_pitch, depEvents)};

    // Extension: copy pitched data from device USM
    q.ext_oneapi_copy(dstMem, out.data(), desc, dst_pitch, depEvents);

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
