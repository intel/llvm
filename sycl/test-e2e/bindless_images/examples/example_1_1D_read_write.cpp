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

  // Initialize input data
  constexpr size_t width = 512;
  std::vector<float> dataIn(width);
  std::vector<float> dataOut(width);
  for (int i = 0; i < width; i++) {
    dataIn[i] = static_cast<float>(i);
  }

  // Image descriptor - can use the same for both images
  sycl::ext::oneapi::experimental::image_descriptor desc(
      sycl::range{width}, 1, sycl::image_channel_type::fp32);

  // Extension: returns the device pointer to the allocated memory
  sycl::ext::oneapi::experimental::image_mem imgMemoryIn(desc, q);
  sycl::ext::oneapi::experimental::image_mem imgMemoryOut(desc, q);

  // Extension: create the image and return the handle
  sycl::ext::oneapi::experimental::unsampled_image_handle imgIn =
      sycl::ext::oneapi::experimental::create_image(imgMemoryIn, desc, q);
  sycl::ext::oneapi::experimental::unsampled_image_handle imgOut =
      sycl::ext::oneapi::experimental::create_image(imgMemoryOut, desc, q);

  // Extension: copy over data to device
  q.ext_oneapi_copy(dataIn.data(), imgMemoryIn.get_handle(), desc);

  // Bindless images require manual synchronization
  // Wait for copy operation to finish
  q.wait_and_throw();

  q.submit([&](sycl::handler &cgh) {
    // No need to request access, handles captured by value

    cgh.parallel_for(width, [=](sycl::id<1> id) {
      // Extension: read image data from handle
      float pixel = sycl::ext::oneapi::experimental::fetch_image<float>(
          imgIn, int(id[0]));

      // Extension: write to image data using handle
      sycl::ext::oneapi::experimental::write_image(imgOut, int(id[0]), pixel);
    });
  });

  // Using image handles requires manual synchronization
  q.wait_and_throw();

  // Copy data written to imgOut to host
  q.ext_oneapi_copy(imgMemoryOut.get_handle(), dataOut.data(), desc);

  // Cleanup
  sycl::ext::oneapi::experimental::destroy_image_handle(imgIn, q);
  sycl::ext::oneapi::experimental::destroy_image_handle(imgOut, q);

  // Validate that `dataIn` correctly transferred to `dataOut`
  for (size_t i = 0; i < width; i++) {
    if (dataOut[i] != dataIn[i]) {
      return 1;
    }
  }
  return 0;
}
