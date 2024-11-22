// REQUIRES: cuda

// RUN: %{build} -o %t.out
// RUN: %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

int main() {
  // Set up device, queue, and context
  sycl::device dev;
  sycl::queue q(dev);
  sycl::context ctxt = q.get_context();

  // declare image data
  size_t numImages = 5;
  size_t width = 8;
  size_t height = 8;
  size_t numPixels = width * height;
  std::vector<float> dataIn(numPixels);
  std::vector<float> dataOut(numPixels);
  std::vector<float> dataExpected(numPixels);
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      int index = j + (height * i);
      dataIn[index] = index;
      dataExpected[index] = index * numImages;
    }
  }

  // Image descriptor - can use the same for all images
  sycl::ext::oneapi::experimental::image_descriptor desc(
      {width, height}, 1, sycl::image_channel_type::fp32);

  // Allocate each image and save the handles
  std::vector<sycl::ext::oneapi::experimental::image_mem> imgAllocations;
  for (int i = 0; i < numImages; i++) {
    // Extension: move-construct device allocated memory
    imgAllocations.emplace_back(
        sycl::ext::oneapi::experimental::image_mem{desc, q});
  }

  // Copy over data to device for each image
  for (int i = 0; i < numImages; i++) {
    // Extension: copy over data to device
    q.ext_oneapi_copy(dataIn.data(), imgAllocations[i].get_handle(), desc);
  }

  // Wait for copy operations to finish
  q.wait_and_throw();

  // Create the images and return the handles
  std::vector<sycl::ext::oneapi::experimental::unsampled_image_handle>
      imgHandles;
  for (int i = 0; i < numImages; i++) {
    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle imgHandle =
        sycl::ext::oneapi::experimental::create_image(imgAllocations[i], desc,
                                                      q);
    imgHandles.push_back(imgHandle);
  }

  {
    sycl::buffer outBuf{dataOut.data(), sycl::range{height, width}};
    sycl::buffer imgHandlesBuf{imgHandles.data(), sycl::range{numImages}};
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor outAcc{outBuf, cgh, sycl::write_only};
      sycl::accessor imgHandleAcc{imgHandlesBuf, cgh, sycl::read_only};

      cgh.parallel_for(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Sum each image by reading via its handle
            float sum = 0;
            for (int i = 0; i < numImages; i++) {
              // Extension: read image data from handle
              sum += (sycl::ext::oneapi::experimental::fetch_image<float>(
                  imgHandleAcc[i], sycl::vec<int, 2>(dim0, dim1)));
            }
            outAcc[sycl::id{dim1, dim0}] = sum;
          });
    });
  }

  // Using image handles requires manual synchronization
  q.wait_and_throw();

  // Cleanup
  for (int i = 0; i < numImages; i++) {
    sycl::ext::oneapi::experimental::destroy_image_handle(imgHandles[i], q);
  }

  // Validate that `dataOut` is correct
  for (size_t i = 0; i < numPixels; i++) {
    if (dataOut[i] != dataExpected[i]) {
      return 1;
    }
  }
  return 0;
}
