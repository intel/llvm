// REQUIRES: linux
// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/sycl.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class kernel_sampled_fetch;

int main() {
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // Check if device supports 1D USM sampled image fetches
  if (!dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_1d_usm)) {
#ifdef VERBOSE_PRINT
    std::cout << "Test skipped due to lack of device support for fetching 1D "
                 "USM backed sampled images\n";
#endif
    return 0;
  }

  // declare image data
  constexpr size_t width = 16;
  std::vector<float> out(width);
  std::vector<float> expected(width);
  std::vector<float> dataIn(width);
  auto imgMem = sycl::malloc_shared<float>(width, q);
  for (int i = 0; i < width; i++) {
    expected[i] = i;
    imgMem[i] = i;
  }

  namespace syclexp = sycl::ext::oneapi::experimental;

  try {
    // Extension: image descriptor
    syclexp::image_descriptor desc({width}, sycl::image_channel_order::r,
                                   sycl::image_channel_type::fp32);

    syclexp::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    auto imgHandle = syclexp::create_image(imgMem, 0, samp, desc, q);

    sycl::buffer buf(out.data(), sycl::range{width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(cgh, width);

      cgh.parallel_for<kernel_sampled_fetch>(width, [=](sycl::id<1> id) {
        // Extension: fetch data from sampled image handle
        float px1 = syclexp::read_image<float>(imgHandle, int(id[0]));

        outAcc[id] = px1;
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
