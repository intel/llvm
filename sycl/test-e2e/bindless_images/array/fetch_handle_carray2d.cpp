// REQUIRES: aspect-ext_oneapi_bindless_images
// XFAIL: level_zero
// UNSUPPORTED: hip
// UNSUPPORTED-INTENDED: Undetermined issue in 'create_image' in this test.

// RUN: %{build} -o %t.out
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include <sycl/sycl.hpp>

int main() {

  sycl::queue q{};

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

  void *imageHandlePtrGen = static_cast<void *>(sycl::malloc_device(
      sizeof(sycl::ext::oneapi::experimental::unsampled_image_handle), q));
  q.memcpy(static_cast<void *>(imageHandlePtrGen),
           static_cast<const void *>(&imgIn),
           sizeof(sycl::ext::oneapi::experimental::unsampled_image_handle));
  q.wait_and_throw();
  q.ext_oneapi_copy(dataIn.data(), imgMemoryIn.get_handle(), desc);
  q.wait_and_throw();
  void *imageHandlePtrPtrGen = static_cast<void *>(sycl::malloc_device(
      sizeof(sycl::ext::oneapi::experimental::unsampled_image_handle *), q));
  q.memcpy(static_cast<void *>(imageHandlePtrPtrGen),
           static_cast<const void *>(&imageHandlePtrGen),
           sizeof(sycl::ext::oneapi::experimental::unsampled_image_handle *));
  q.wait_and_throw();

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>{{width}, {width}}, [=](sycl::nd_item<1>
                                                                  it) {
      sycl::ext::oneapi::experimental::unsampled_image_handle *
          *imageHandlePtrPtr = static_cast<
              sycl::ext::oneapi::experimental::unsampled_image_handle **>(
              imageHandlePtrPtrGen);
      sycl::ext::oneapi::experimental::unsampled_image_handle *imageHandlePtr =
          static_cast<
              sycl::ext::oneapi::experimental::unsampled_image_handle *>(
              imageHandlePtrPtr[0]);
      sycl::ext::oneapi::experimental::unsampled_image_handle imageHandle =
          imageHandlePtr[0];

      size_t dim0 = it.get_local_id(0);
      // Extension: read image data from handle
      float pixel = sycl::ext::oneapi::experimental::fetch_image<float>(
          imageHandle, int(dim0));

      // Extension: write to image data using handle
      sycl::ext::oneapi::experimental::write_image(imgOut, int(dim0), pixel);
    });
  });

  q.wait_and_throw();

  // Copy data written to imgOut to host
  q.ext_oneapi_copy(imgMemoryOut.get_handle(), dataOut.data(), desc);

  // Ensure copying data from the device to host is finished before validate
  q.wait_and_throw();

  // Cleanup
  sycl::ext::oneapi::experimental::destroy_image_handle(imgIn, q);
  sycl::ext::oneapi::experimental::destroy_image_handle(imgOut, q);
  sycl::free(imageHandlePtrGen, q);
  sycl::free(imageHandlePtrPtrGen, q);

  for (size_t i = 0; i < width; i++) {
    if (dataOut[i] != dataIn[i]) {
      std::cout << "Test failed"
                << "\n";
      return 1;
    }
  }
  return 0;
}
