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

  // Check if device supports 2D USM sampled image fetches
  if (!dev.has(sycl::aspect::ext_oneapi_bindless_sampled_image_fetch_2d_usm)) {
#ifdef VERBOSE_PRINT
    std::cout << "Test skipped due to lack of device support for fetching 2D "
                 "USM backed sampled images\n";
#endif
    return 0;
  }

  // declare image data
  constexpr size_t width = 5;
  constexpr size_t height = 6;
  constexpr size_t N = width * height;
  std::vector<sycl::vec<uint16_t, 4>> out(N);
  std::vector<sycl::vec<uint16_t, 4>> expected(N);
  std::vector<sycl::vec<uint16_t, 4>> dataIn(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      auto index = i + (width * j);
      expected[index] = {index, index, index, index};
      dataIn[index] = {index, index, index, index};
    }
  }

  namespace syclexp = sycl::ext::oneapi::experimental;

  try {
    syclexp::bindless_image_sampler samp(
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear);

    // Extension: image descriptor
    syclexp::image_descriptor desc({width, height},
                                   sycl::image_channel_order::rgba,
                                   sycl::image_channel_type::unsigned_int16);
    size_t pitch = 0;

    // Extension: returns the device pointer to USM allocated pitched memory
    auto imgMem = syclexp::pitched_alloc_device(&pitch, desc, q);

    if (imgMem == nullptr) {
      std::cout << "Error allocating pitched image memory!" << std::endl;
      return 1;
    }

    // Extension: copy over data to device for USM image
    q.ext_oneapi_copy(dataIn.data(), imgMem, desc, pitch);
    q.wait_and_throw();

    // Extension: create the images and return the handles
    syclexp::sampled_image_handle imgHandle =
        syclexp::create_image(imgMem, pitch, samp, desc, q);

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
            auto px1 = syclexp::read_image<sycl::vec<uint16_t, 4>>(
                imgHandle, sycl::int2(dim0, dim1));

            outAcc[sycl::id<2>{dim1, dim0}] = px1;
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
  for (int i = 0; i < N; i++) {
    bool mismatch = false;
    if (out[i][0] != expected[i][0]) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected[i][0]
                << ", Actual: " << out[i][0] << std::endl;
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
