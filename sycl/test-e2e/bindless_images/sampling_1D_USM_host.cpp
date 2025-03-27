// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_bindless_images_sample_1d_usm

// UNSUPPORTED: hip
// UNSUPPORTED-INTENDED: Host USM backed image support is not yet enabled in UR
// adapter. Also, when provionally enabled, the test crashes upon image
// creation, whereas Device USM backed images do not crash. This issue is
// undetermined.

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include <cmath>
#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class sample_host_usm_image_kernel;

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  size_t width = 32;
  size_t widthInBytes = width * sizeof(float);
  std::vector<float> out(width);
  std::vector<float> expected(width);
  for (int i = 0; i < width; ++i) {
    expected[i] = static_cast<float>(i);
  }

  try {
    sycl::ext::oneapi::experimental::bindless_image_sampler samp(
        sycl::addressing_mode::clamp,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    // Extension: image descriptor
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, 1, sycl::image_channel_type::fp32);

    // Host USM allocation
    float *imgMem = sycl::malloc_host<float>(width, ctxt);

    if (imgMem == nullptr) {
      std::cerr << "Error allocating host USM!" << std::endl;
      return 1;
    }

    // Initialize input data
    for (int i = 0; i < width; ++i) {
      imgMem[i] = static_cast<float>(i);
    }

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::sampled_image_handle imgHandle =
        sycl::ext::oneapi::experimental::create_image(imgMem, 0 /* pitch */,
                                                      samp, desc, dev, ctxt);

    sycl::buffer<float, 1> buf((float *)out.data(), sycl::range<1>{width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc =
          buf.get_access<sycl::access_mode::write>(cgh, sycl::range<1>{width});

      cgh.parallel_for<sample_host_usm_image_kernel>(
          sycl::nd_range<1>{{width}, {width}}, [=](sycl::nd_item<1> it) {
            size_t dim0 = it.get_local_id(0);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5f) / (float)width;

            // Extension: sample image data from handle
            float px = sycl::ext::oneapi::experimental::sample_image<float>(
                imgHandle, (float)fdim0);

            outAcc[sycl::id<1>{dim0}] = px;
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle, dev, ctxt);
    sycl::free(imgMem, ctxt);
  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  // collect and validate output
  bool validated = true;
  for (int i = 0; i < width; i++) {
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
