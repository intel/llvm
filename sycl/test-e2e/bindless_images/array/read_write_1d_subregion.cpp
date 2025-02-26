// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

// Uncomment to print additional test information
// #define VERBOSE_PRINT

class image_addition;

int main() {

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  size_t width = 4;
  size_t layers = 2;
  size_t N = width * layers;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float> dataIn1(N);
  std::vector<float> dataIn2(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < layers; j++) {
      expected[j + ((layers)*i)] = (j + (layers)*i) * 3;
      dataIn1[j + ((layers)*i)] = (j + (layers)*i);
      dataIn2[j + ((layers)*i)] = (j + (layers)*i) * 2;
    }
  }

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::experimental::image_descriptor desc(
      {width}, 1, sycl::image_channel_type::fp32,
      sycl::ext::oneapi::experimental::image_type::array, 1, layers);

  try {
    // Extension: allocate memory on device and create the handle
    sycl::ext::oneapi::experimental::image_mem imgMem0(desc, q);
    sycl::ext::oneapi::experimental::image_mem imgMem1(desc, q);
    sycl::ext::oneapi::experimental::image_mem imgMem2(desc, q);

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle1 =
        sycl::ext::oneapi::experimental::create_image(imgMem0, desc, q);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle2 =
        sycl::ext::oneapi::experimental::create_image(imgMem1, desc, q);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle3 =
        sycl::ext::oneapi::experimental::create_image(imgMem2, desc, q);

    // The subregion size for the copies.
    sycl::range copyExtent = {width / 2, 1, layers / 2};
    // The extent of data provided on the host (vector).
    sycl::range srcExtent = {width, 1, layers};

    // the 4 subregion offsets used for the copies.
    std::vector<sycl::range<3>> offsets{{0, 0, 0},
                                        {width / 2, 0, 0},
                                        {0, 0, layers / 2},
                                        {width / 2, 0, layers / 2}};

    for (auto offset : offsets) {
      // Extension: Copy to image array subregion.
      q.ext_oneapi_copy(dataIn1.data(), offset, srcExtent, imgMem0.get_handle(),
                        offset, desc, copyExtent);
      // Extension: Copy to image array subregion.
      q.ext_oneapi_copy(dataIn2.data(), offset, srcExtent, imgMem1.get_handle(),
                        offset, desc, copyExtent);
    }
    q.wait_and_throw();

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<image_addition>(
          sycl::nd_range<2>{{width, layers}, {width, layers}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            float sum = 0;
            // Extension: fetch image data from handle
            float px1 =
                sycl::ext::oneapi::experimental::fetch_image_array<float>(
                    imgHandle1, int(dim0), dim1);
            float px2 =
                sycl::ext::oneapi::experimental::fetch_image_array<float>(
                    imgHandle2, int(dim0), dim1);

            sum = px1 + px2;

            // Extension: write to image with handle
            sycl::ext::oneapi::experimental::write_image_array<float>(
                imgHandle3, int(dim0), dim1, sum);
          });
    });
    q.wait_and_throw();

    // Extension: copy data from device to host (four subregions/quadrants)
    for (auto offset : offsets) {
      q.ext_oneapi_copy(imgMem2.get_handle(), offset, desc, out.data(), offset,
                        srcExtent, copyExtent);
    }
    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle1, q);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle2, q);
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
