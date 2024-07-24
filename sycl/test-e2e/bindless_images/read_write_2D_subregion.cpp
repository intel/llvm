// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %t.out

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
  size_t width = 32;
  size_t height = 32;
  size_t N = width * height;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float> dataIn1(N);
  std::vector<float> dataIn2(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[j + ((height)*i)] = j * 3;
      dataIn1[j + ((height)*i)] = j;
      dataIn2[j + ((height)*i)] = j * 2;
    }
  }

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::experimental::image_descriptor desc(
      {width, height}, 1, sycl::image_channel_type::fp32);

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

    // Extension: copy over data to device (four subregions/quadrants)
    sycl::range copyExtent1 = {width / 2, height / 2, 1};
    sycl::range srcExtent = {width, height, 0};

    q.ext_oneapi_copy(dataIn1.data(), {0, 0, 0}, srcExtent,
                      imgMem0.get_handle(), {0, 0, 0}, desc, copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {width / 2, 0, 0}, srcExtent,
                      imgMem0.get_handle(), {width / 2, 0, 0}, desc,
                      copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {0, height / 2, 0}, srcExtent,
                      imgMem0.get_handle(), {0, height / 2, 0}, desc,
                      copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {width / 2, height / 2, 0}, srcExtent,
                      imgMem0.get_handle(), {width / 2, height / 2, 0}, desc,
                      copyExtent1);

    q.ext_oneapi_copy(dataIn2.data(), {0, 0, 0}, srcExtent,
                      imgMem1.get_handle(), {0, 0, 0}, desc, copyExtent1);
    q.ext_oneapi_copy(dataIn2.data(), {width / 2, 0, 0}, srcExtent,
                      imgMem1.get_handle(), {width / 2, 0, 0}, desc,
                      copyExtent1);
    q.ext_oneapi_copy(dataIn2.data(), {0, height / 2, 0}, srcExtent,
                      imgMem1.get_handle(), {0, height / 2, 0}, desc,
                      copyExtent1);
    q.ext_oneapi_copy(dataIn2.data(), {width / 2, height / 2, 0}, srcExtent,
                      imgMem1.get_handle(), {width / 2, height / 2, 0}, desc,
                      copyExtent1);

    q.wait_and_throw();

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<image_addition>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            float sum = 0;
            // Extension: fetch image data from handle
            float px1 = sycl::ext::oneapi::experimental::fetch_image<float>(
                imgHandle1, sycl::int2(dim0, dim1));
            float px2 = sycl::ext::oneapi::experimental::fetch_image<float>(
                imgHandle2, sycl::int2(dim0, dim1));

            sum = px1 + px2;
            // Extension: write to image with handle
            sycl::ext::oneapi::experimental::write_image<float>(
                imgHandle3, sycl::int2(dim0, dim1), sum);
          });
    });
    q.wait_and_throw();

    // Extension: copy data from device to host (four subregions/quadrants)
    auto destExtent = srcExtent;
    q.ext_oneapi_copy(imgMem2.get_handle(), {0, 0, 0}, desc, out.data(),
                      {0, 0, 0}, destExtent, copyExtent1);
    q.ext_oneapi_copy(imgMem2.get_handle(), {width / 2, 0, 0}, desc, out.data(),
                      {width / 2, 0, 0}, destExtent, copyExtent1);
    q.ext_oneapi_copy(imgMem2.get_handle(), {0, height / 2, 0}, desc,
                      out.data(), {0, height / 2, 0}, destExtent, copyExtent1);
    q.ext_oneapi_copy(imgMem2.get_handle(), {width / 2, height / 2, 0}, desc,
                      out.data(), {width / 2, height / 2, 0}, destExtent,
                      copyExtent1);

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
