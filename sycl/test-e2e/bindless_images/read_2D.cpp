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
  size_t width = 7;
  size_t height = 3;
  size_t N = width * height;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<sycl::float4> dataIn1(N);
  std::vector<sycl::float4> dataIn2(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[i + (width * j)] = j * 3;
      dataIn1[i + (width * j)] = {j, j, j, j};
      dataIn2[i + (width * j)] = {j * 2, j * 2, j * 2, j * 2};
    }
  }

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::experimental::image_descriptor desc(
      {width, height}, 4, sycl::image_channel_type::fp32);

  try {
    // Extension: allocate memory on device and create the handle
    sycl::ext::oneapi::experimental::image_mem imgMem0(desc, dev, ctxt);
    sycl::ext::oneapi::experimental::image_mem imgMem1(desc, dev, ctxt);

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle1 =
        sycl::ext::oneapi::experimental::create_image(imgMem0, desc, dev, ctxt);
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle2 =
        sycl::ext::oneapi::experimental::create_image(imgMem1, desc, dev, ctxt);

    sycl::buffer<float, 2> buf((float *)out.data(),
                               sycl::range<2>{height, width});

    // Extension: copy over data to device (handler variant)
    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_copy(dataIn1.data(), imgMem0.get_handle(), desc);
    });

    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_copy(dataIn2.data(), imgMem1.get_handle(), desc);
    });
    q.wait_and_throw();

    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(
          cgh, sycl::range<2>{height, width});

      cgh.parallel_for<image_addition>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);
            float sum = 0;
            // Extension: fetch image data from handle
            sycl::float4 px1 =
                sycl::ext::oneapi::experimental::fetch_image<sycl::float4>(
                    imgHandle1, sycl::int2(dim0, dim1));
            sycl::float4 px2 =
                sycl::ext::oneapi::experimental::fetch_image<sycl::float4>(
                    imgHandle2, sycl::int2(dim0, dim1));

            sum = px1[0] + px2[0];
            outAcc[sycl::id<2>{dim1, dim0}] = sum;
          });
    });
    q.wait_and_throw();

    // Extension: cleanup
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle1, dev,
                                                          ctxt);
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandle2, dev,
                                                          ctxt);
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
