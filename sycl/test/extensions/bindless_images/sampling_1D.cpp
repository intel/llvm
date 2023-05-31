// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out

#include <CL/sycl.hpp>
#include <iostream>

class image_addition;

int main() {

#if defined(SYCL_EXT_ONEAPI_FREE_FUNCTION_QUERIES)
  assert(SYCL_EXT_ONEAPI_FREE_FUNCTION_QUERIES == 1);
  std::cout << "SYCL_EXT_ONEAPI_BINDLESS_IMAGES is defined!" << std::endl;
#else
  std::cerr << "Bindless images feature test macro is not defined!"
            << std::endl;
  assert(false);
#endif // defined(SYCL_EXT_ONEAPI_BINDLESS_IMAGES)

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  constexpr size_t N = 32;
  size_t width = N;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float> dataIn1(N);
  for (int i = 0; i < N; i++) {
    expected[i] = i;
    dataIn1[i] = float(i);
  }

  try {
    // Extension: image descriptor
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width}, sycl::image_channel_order::r, sycl::image_channel_type::fp32);

    sycl::sampler samp1(sycl::coordinate_normalization_mode::normalized,
                        sycl::addressing_mode::repeat,
                        sycl::filtering_mode::linear);

    // Extension: allocate memory on device
    sycl::ext::oneapi::experimental::image_mem img_mem_0(ctxt, desc);

    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn1.data(), img_mem_0, desc);
    q.wait_and_throw();

    // Extension: create the image and return the handle
    auto imgHandle1 = sycl::ext::oneapi::experimental::create_image(
        ctxt, img_mem_0, samp1, desc);

    sycl::buffer<float, 1> buf((float *)out.data(), N);
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(cgh, N);

      cgh.parallel_for<image_addition>(N, [=](sycl::id<1> id) {
        // Normalize coordinate -- +0.5 to look towards centre of pixel
        float x = float(id[0] + 0.5) / (float)N;
        // Extension: read image data from handle
        float px1 =
            sycl::ext::oneapi::experimental::read_image<float>(imgHandle1, x);

        outAcc[id] = px1;
      });
    });

    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(ctxt, imgHandle1);

  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    exit(-1);
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    exit(-1);
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
      std::cout << "Result mismatch! Expected: " << expected[i]
                << ", Actual: " << out[i] << std::endl;
    }
  }
  if (validated) {
    std::cout << "Correct output!" << std::endl;
  }

  return 0;
}
