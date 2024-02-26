// REQUIRES: linux
// REQUIRES: cuda

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %s -o %t.out
// RUN: %t.out

#include <iostream>
#include <sycl/sycl.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class image_addition;

int main() {

#if defined(SYCL_EXT_ONEAPI_BINDLESS_IMAGES)
  assert(SYCL_EXT_ONEAPI_BINDLESS_IMAGES == 1);
#if defined(VERBOSE_PRINT)
  std::cout << "SYCL_EXT_ONEAPI_BINDLESS_IMAGES is defined!" << std::endl;
#endif
#else
  std::cerr << "Bindless images feature test macro is not defined!"
            << std::endl;
  return 1;
#endif // defined(SYCL_EXT_ONEAPI_BINDLESS_IMAGES)

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  constexpr size_t N = 32;
  size_t width = N;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float> dataIn(N);
  for (int i = 0; i < N; i++) {
    expected[i] = i;
    dataIn[i] = float(i);
  }

  try {
    // Extension: image descriptor
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, sycl::image_channel_order::r, sycl::image_channel_type::fp32);

    sycl::ext::oneapi::experimental::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    // Extension: allocate memory on device
    sycl::ext::oneapi::experimental::image_mem imgMem(desc, dev, ctxt);

    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn.data(), imgMem.get_handle(), desc);
    q.wait_and_throw();

    // Extension: create the image and return the handle
    auto imgHandle = sycl::ext::oneapi::experimental::create_image(
        imgMem, samp, desc, dev, ctxt);

    sycl::buffer<float, 1> buf((float *)out.data(), N);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(cgh, N);

      cgh.parallel_for<image_addition>(N, [=](sycl::id<1> id) {
        // Normalize coordinate -- +0.5 to look towards centre of pixel
        float x = float(id[0] + 0.5f) / (float)N;
        // Extension: sample image data from handle
        float px1 =
            sycl::ext::oneapi::experimental::sample_image<float>(imgHandle, x);

        outAcc[id] = px1;
      });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle, dev, ctxt);

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    return 2;
  }

  // collect and validate output
  bool validated = true;
  for (int i = 0; i < N; i++) {
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

  std::cout << "Test passed!" << std::endl;
  return 3;
}
