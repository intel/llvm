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
  size_t numImages = 5;
  size_t width = 7;
  size_t height = 3;
  size_t N = width * height;
  std::vector<float> out(N);
  std::vector<float> expected(N);
  std::vector<sycl::float4> dataIn(N);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      expected[i + (width * j)] = (i + (width * j)) * numImages;
      dataIn[i + (width * j)] = {i + (width * j), 0, 0, 0};
    }
  }

  try {

    // Extension: image descriptor - can use the same for all images
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, 4, sycl::image_channel_type::fp32);

    // Allocate each image and save the device memory handles
    std::vector<std::shared_ptr<sycl::ext::oneapi::experimental::image_mem>>
        imgAllocations;
    for (int i = 0; i < numImages; i++) {
      // Extension: returns the handle to the device allocated memory
      imgAllocations.push_back(
          std::make_shared<sycl::ext::oneapi::experimental::image_mem>(desc,
                                                                       q));
    }

    // Copy over data to device for each image
    for (int i = 0; i < numImages; i++) {
      // Extension: copy over data to device
      q.ext_oneapi_copy(dataIn.data(), imgAllocations[i]->get_handle(), desc);
    }
    q.wait_and_throw();

    // Create the images and return the handles
    std::vector<sycl::ext::oneapi::experimental::unsampled_image_handle>
        imgHandles;
    for (int i = 0; i < numImages; i++) {
      // Extension: create the image and return the handle
      sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle =
          sycl::ext::oneapi::experimental::create_image(*imgAllocations[i],
                                                        desc, q);
      imgHandles.push_back(imgHandle);
    }

    sycl::buffer<float, 2> buf(out.data(), sycl::range<2>{height, width});
    sycl::buffer imgHandlesBuf{imgHandles};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor outAcc{buf, cgh, sycl::write_only};
      sycl::accessor imgHandleAcc{imgHandlesBuf, cgh, sycl::read_only};

      cgh.parallel_for<image_addition>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Sum each image by reading their handle
            float sum = 0;
            for (int i = 0; i < numImages; i++) {
              // Extension: fetch image data from handle
              sum +=
                  (sycl::ext::oneapi::experimental::fetch_image<sycl::float4>(
                      imgHandleAcc[i], sycl::int2(dim0, dim1)))[0];
            }
            outAcc[sycl::id<2>{dim1, dim0}] = sum;
          });
    });

    // Using image handles requires manual synchronization
    q.wait_and_throw();

    // Extension: cleanup
    for (int i = 0; i < numImages; i++) {
      sycl::ext::oneapi::experimental::destroy_image_handle(imgHandles[i], q);
    }
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
