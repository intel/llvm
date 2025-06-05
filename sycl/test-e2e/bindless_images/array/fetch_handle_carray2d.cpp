// REQUIRES: aspect-ext_oneapi_bindless_images
// XFAIL: level_zero && windows
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/18727
// UNSUPPORTED: hip
// UNSUPPORTED-INTENDED: Undetermined issue in 'create_image' in this test.

// RUN: %{build} -o %t.out
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

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
  syclexp::image_descriptor desc(sycl::range{width}, 1,
                                 sycl::image_channel_type::fp32);

  // Extension: returns the device pointer to the allocated memory
  syclexp::image_mem imgMemoryIn(desc, q);
  syclexp::image_mem imgMemoryOut(desc, q);

  q.ext_oneapi_copy(dataIn.data(), imgMemoryIn.get_handle(), desc);
  q.wait_and_throw();
  // Extension: create the image and return the handle
  syclexp::sampled_image_handle imgIn =
      syclexp::create_image(imgMemoryIn, desc, q);
  syclexp::sampled_image_handle imgOut =
      syclexp::create_image(imgMemoryOut, desc, q);

  // Copy the input data to the image_mem of the device sampled_image_handle
  q.ext_oneapi_copy(dataIn.data(), imgMemoryIn.get_handle(), desc);
  q.wait_and_throw();

  // Allocate an sampled_image_handle manually instead of using create_image so
  // we can allocate it on heap
  void *imageHandlePtrGen =
      sycl::malloc_device(sizeof(syclexp::sampled_image_handle), q);

  // Copy the create_image returned device sampled_image_handle to the
  // contents of the void* pointing to the heap allocated
  // sampled_image_handle
  q.memcpy(static_cast<void *>(imageHandlePtrGen),
           static_cast<const void *>(&imgIn),
           sizeof(syclexp::sampled_image_handle));

  q.wait_and_throw();

  // Allocate a device generic pointer pointing to an sampled_image_handle*
  void *imageHandlePtrPtrGen =
      sycl::malloc_device(sizeof(syclexp::sampled_image_handle *), q);

  // Copy the address of the manually allocated sampled_image_handle to the
  // contents of the generic device pointer allocated above
  q.memcpy(static_cast<void *>(imageHandlePtrPtrGen),
           static_cast<const void *>(&imageHandlePtrGen),
           sizeof(syclexp::sampled_image_handle *));

  q.wait_and_throw();

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>{{width}, {width}}, [=](sycl::nd_item<1> it) {
          syclexp::sampled_image_handle **imageHandlePtrPtr =
              static_cast<syclexp::sampled_image_handle **>(
                  imageHandlePtrPtrGen);
          // Dereference the generic pointer to the sampled_image_handle
          // pointer
          syclexp::sampled_image_handle *imageHandlePtr =
              static_cast<syclexp::sampled_image_handle *>(
                  imageHandlePtrPtr[0]);
          // Dereference the sampled_image_handle pointer
          syclexp::sampled_image_handle imageHandle = imageHandlePtr[0];

          size_t dim0 = it.get_local_id(0);
          // Extension: read image data from handle
          float pixel = syclexp::fetch_image<float>(imageHandle, int(dim0));

          // Extension: write to image data using handle
          syclexp::write_image(imgOut, int(dim0), pixel);
        });
  });

  q.wait_and_throw();

  // Copy data written to imgOut to host
  q.ext_oneapi_copy(imgMemoryOut.get_handle(), dataOut.data(), desc);

  // Ensure copying data from the device to host is finished before validate
  q.wait_and_throw();

  // Cleanup
  syclexp::destroy_image_handle(imgIn, q);
  syclexp::destroy_image_handle(imgOut, q);
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
