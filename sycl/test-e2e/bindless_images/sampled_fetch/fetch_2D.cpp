// REQUIRES: cuda
// REQUIRES: aspect-ext_oneapi_bindless_sampled_image_fetch_2d

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

class kernel_sampled_fetch;

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  constexpr size_t width = 5;
  constexpr size_t height = 6;
  constexpr size_t N = width * height;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float> dataIn(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      auto index = i + (width * j);
      expected[index] = index;
      dataIn[index] = index;
    }
  }

  namespace syclexp = sycl::ext::oneapi::experimental;

  try {
    syclexp::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::unnormalized,
        sycl::filtering_mode::nearest);

    // Extension: image descriptor
    syclexp::image_descriptor desc({width, height}, 1,
                                   sycl::image_channel_type::fp32);

    // Extension: allocate memory on device
    syclexp::image_mem imgMem(desc, dev, ctxt);

    // Extension: copy over data to device for non-USM image
    q.ext_oneapi_copy(dataIn.data(), imgMem.get_handle(), desc);
    q.wait_and_throw();

    // Extension: create the images and return the handles
    syclexp::sampled_image_handle imgHandle =
        syclexp::create_image(imgMem, samp, desc, q);

    sycl::buffer buf(out.data(), sycl::range{height, width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(
          cgh, sycl::range<2>{height, width});

      cgh.parallel_for<kernel_sampled_fetch>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Extension: fetch data from sampled image handle
            float px1 =
                syclexp::fetch_image<float>(imgHandle, sycl::int2(dim0, dim1));

            outAcc[sycl::id<2>{dim1, dim0}] = px1;
          });
    });

    q.wait_and_throw();

    // Extension: cleanup
    syclexp::destroy_image_handle(imgHandle, dev, ctxt);
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

  std::cout << "Test failed!" << std::endl;
  return 3;
}
