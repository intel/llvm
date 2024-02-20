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

  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // declare image data
  size_t width = 16;
  size_t height = 16;
  size_t depth = 8;
  size_t N = width * height * depth;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<float> dataIn1(N);
  std::vector<float> dataIn2(N);
  // ROW-MAJOR
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      for (int k = 0; k < depth; k++) {
        expected[k + (depth) * (j + (height)*i)] =
            (k + (depth) * (j + (height)*i)) * 3;
        dataIn1[k + (depth) * (j + (height)*i)] =
            k + (depth) * (j + (height)*i);
        dataIn2[k + (depth) * (j + (height)*i)] =
            (k + (depth) * (j + (height)*i)) * 2;
      }
    }
  }

  try {

    // Extension: image descriptor - can use the same for both images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height, depth}, sycl::image_channel_order::r,
        sycl::image_channel_type::fp32);

    // Extension: allocate memory on device and create the handle
    sycl::ext::oneapi::experimental::image_mem imgMem0(desc, q);
    sycl::ext::oneapi::experimental::image_mem imgMem1(desc, q);
    sycl::ext::oneapi::experimental::image_mem imgMem2(desc, q);

    // Extension: copy over data to device (8 sub-regions)
    sycl::range copyExtent1 = {width / 2, height / 2, depth / 2};
    sycl::range srcExtent1 = {width, height, depth};

    // First image with 8 sub-regions
    q.ext_oneapi_copy(dataIn1.data(), {0, 0, 0}, srcExtent1,
                      imgMem0.get_handle(), {0, 0, 0}, desc, copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {width / 2, 0, 0}, srcExtent1,
                      imgMem0.get_handle(), {width / 2, 0, 0}, desc,
                      copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {0, height / 2, 0}, srcExtent1,
                      imgMem0.get_handle(), {0, height / 2, 0}, desc,
                      copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {0, 0, depth / 2}, srcExtent1,
                      imgMem0.get_handle(), {0, 0, depth / 2}, desc,
                      copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {width / 2, height / 2, 0}, srcExtent1,
                      imgMem0.get_handle(), {width / 2, height / 2, 0}, desc,
                      copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {0, height / 2, depth / 2}, srcExtent1,
                      imgMem0.get_handle(), {0, height / 2, depth / 2}, desc,
                      copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {width / 2, 0, depth / 2}, srcExtent1,
                      imgMem0.get_handle(), {width / 2, 0, depth / 2}, desc,
                      copyExtent1);
    q.ext_oneapi_copy(dataIn1.data(), {width / 2, height / 2, depth / 2},
                      srcExtent1, imgMem0.get_handle(),
                      {width / 2, height / 2, depth / 2}, desc, copyExtent1);

    // Second image with 2 sub-regions
    sycl::range copyExtent2 = {width, height, depth / 2};
    sycl::range srcExtent2 = {width, height, depth};
    q.ext_oneapi_copy(dataIn2.data(), {0, 0, 0}, srcExtent2,
                      imgMem1.get_handle(), {0, 0, 0}, desc, copyExtent2);
    q.ext_oneapi_copy(dataIn2.data(), {0, 0, depth / 2}, srcExtent2,
                      imgMem1.get_handle(), {0, 0, depth / 2}, desc,
                      copyExtent2);

    q.wait_and_throw();

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle1 =
        sycl::ext::oneapi::experimental::create_image(imgMem0, desc, q);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle2 =
        sycl::ext::oneapi::experimental::create_image(imgMem1, desc, q);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle3 =
        sycl::ext::oneapi::experimental::create_image(imgMem2, desc, q);

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<image_addition>(
          sycl::nd_range<3>{{width, height, depth}, {16, 16, 2}},
          [=](sycl::nd_item<3> it) {
            size_t dim0 = it.get_global_id(0);
            size_t dim1 = it.get_global_id(1);
            size_t dim2 = it.get_global_id(2);
            float sum = 0;
            // Extension: read image data from handle
            float px1 = sycl::ext::oneapi::experimental::read_image<float>(
                imgHandle1, sycl::int3(dim0, dim1, dim2));
            float px2 = sycl::ext::oneapi::experimental::read_image<float>(
                imgHandle2, sycl::int3(dim0, dim1, dim2));

            sum = px1 + px2;
            // Extension: write to image with handle
            sycl::ext::oneapi::experimental::write_image<float>(
                imgHandle3, sycl::int3(dim0, dim1, dim2), sum);
          });
    });

    q.wait_and_throw();

    // Extension: copy data from device to host (two sub-regions)
    sycl::range copyExtent3 = {width, height, depth / 2};
    sycl::range destExtent = {width, height, depth};
    q.ext_oneapi_copy(imgMem2.get_handle(), {0, 0, 0}, desc, out.data(),
                      {0, 0, 0}, destExtent, copyExtent3);
    q.ext_oneapi_copy(imgMem2.get_handle(), {0, 0, depth / 2}, desc, out.data(),
                      {0, 0, depth / 2}, destExtent, copyExtent3);
    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle1, q);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle2, q);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle3, q);
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
