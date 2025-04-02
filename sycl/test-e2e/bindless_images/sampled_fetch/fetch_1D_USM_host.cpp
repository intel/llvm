// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_bindless_sampled_image_fetch_1d_usm
// UNSUPPORTED: target-amd
// UNSUPPORTED-INTENDED: Sampled fetch not currently supported on AMD

// RUN: %{build} -o %t.out
// RUN: %{run-unfiltered-devices} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

class kernel_sampled_fetch;

// Uncomment to print additional test information
// #define VERBOSE_PRINT

int main() {
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // Declare image size, and expected output and actual output vectors
  constexpr size_t width = 32;
  constexpr size_t widthInBytes = width * sizeof(float);
  std::vector<float> out(width);
  std::vector<float> expected(width);
  for (int i = 0; i < width; ++i) {
    expected[i] = static_cast<float>(i);
  }

  namespace syclexp = sycl::ext::oneapi::experimental;

  try {
    // Extension: image descriptor
    syclexp::image_descriptor desc({width}, 1, sycl::image_channel_type::fp32);

    // Extension: Image creation requires a sampler, but it will have no effect
    //            on the result, as we will use `fetch_image` in the kernel.
    syclexp::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    // Allocate Host USM and initialize with expected data
    float *imgMem = sycl::malloc_host<float>(width, q);
    memcpy(imgMem, expected.data(), widthInBytes);

    // Extension: create the image backed by Host USM and return the handle
    auto imgHandle = syclexp::create_image(imgMem, 0, samp, desc, q);

    // Create a buffer to output the result from `fetch_image`
    sycl::buffer outBuf(out.data(), sycl::range{width});
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor outAcc{outBuf, cgh, sycl::write_only};

      cgh.parallel_for<kernel_sampled_fetch>(width, [=](sycl::id<1> id) {
        // Extension: fetch data from sampled image handle
        outAcc[id] = syclexp::fetch_image<float>(imgHandle, int(id[0]));
      });
    });

    q.wait_and_throw();

    // Extension: cleanup
    syclexp::destroy_image_handle(imgHandle, dev, ctxt);
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
