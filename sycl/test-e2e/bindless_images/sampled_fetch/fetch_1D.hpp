#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

class kernel_sampled_fetch;
namespace syclexp = sycl::ext::oneapi::experimental;

int test() {

  sycl::queue q{};

  // declare image data
  constexpr size_t N = 30;
  std::vector<float> out(N);
  std::vector<float> dataIn(N);
  for (int i = 0; i < N; i++) {
    dataIn[i] = i;
  }

  try {
    syclexp::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::unnormalized,
        sycl::filtering_mode::nearest);

    // Extension: image descriptor
    syclexp::image_descriptor desc(N, 1, sycl::image_channel_type::fp32);

    // Extension: allocate memory on device
    syclexp::image_mem imgMem(desc, q);

    // Extension: copy over data to device for non-USM image
    q.ext_oneapi_copy(dataIn.data(), imgMem.get_handle(), desc);
    q.wait_and_throw();

    // Extension: create the images and return the handles
    syclexp::sampled_image_handle imgHandle =
        syclexp::create_image(imgMem, samp, desc, q);

    sycl::buffer buf(out.data(), sycl::range{N});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc =
          buf.get_access<sycl::access_mode::write>(cgh, sycl::range<1>{N});

      cgh.parallel_for<kernel_sampled_fetch>(
          sycl::nd_range<1>{N, N}, [=](sycl::nd_item<1> it) {
            size_t dim0 = it.get_local_id(0);
            // Extension: fetch data from sampled image handle
            outAcc[dim0] = syclexp::fetch_image<float>(imgHandle, int(dim0));
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    syclexp::destroy_image_handle(imgHandle, q);
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
    if (out[i] != dataIn[i]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << dataIn[i]
                << ", Actual: " << out[i] << "\n";
#else
      break;
#endif
    }
  }
  if (validated) {
    std::cout << "Test passed!"
              << "\n";
    return 0;
  }

  std::cout << "Test failed!"
            << "\n";
  return 3;
}
