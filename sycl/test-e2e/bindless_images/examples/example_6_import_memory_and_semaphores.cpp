// REQUIRES: cuda

// RUN: %{build} -o %t.out

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

int main() {
  // Set up device, queue, and context
  sycl::device dev;
  sycl::queue q(dev);
  sycl::context ctxt = q.get_context();

  size_t width = 123 /* passed from external API */;
  size_t height = 123 /* passed from external API */;

  /* mapped from external API */
  unsigned int num_channels = 1;
  /* we assume there is one channel */;

  sycl::image_channel_type channel_type =
      /* mapped from external API */
      /* we assume */ sycl::image_channel_type::unsigned_int32;

  // Image descriptor - mapped to external API image layout
  sycl::ext::oneapi::experimental::image_descriptor desc(
      {width, height}, num_channels, channel_type);

  size_t img_size_in_bytes = width * height * sizeof(uint32_t);

  int external_input_image_file_descriptor = 123 /* passed from external API */;
  int external_output_image_file_descriptor =
      123 /* passed from external API */;

  // Extension: populate external memory descriptors
  sycl::ext::oneapi::experimental::external_mem_descriptor<
      sycl::ext::oneapi::experimental::resource_fd>
      input_ext_mem_desc{
          external_input_image_file_descriptor,
          sycl::ext::oneapi::experimental::external_mem_handle_type::opaque_fd,
          img_size_in_bytes};

  sycl::ext::oneapi::experimental::external_mem_descriptor<
      sycl::ext::oneapi::experimental::resource_fd>
      output_ext_mem_desc{
          external_output_image_file_descriptor,
          sycl::ext::oneapi::experimental::external_mem_handle_type::opaque_fd,
          img_size_in_bytes};

  // An external API semaphore will signal this semaphore before our SYCL
  // commands can begin execution
  int wait_semaphore_file_descriptor = 123 /* passed from external API */;

  // An external API will wait on this semaphore to be signalled by us before it
  // can execute some commands
  int done_semaphore_file_descriptor = 123 /* passed from external API */;

  // Extension: populate external semaphore descriptor.
  //            We assume POSIX file descriptor resource types
  sycl::ext::oneapi::experimental::external_semaphore_descriptor<
      sycl::ext::oneapi::experimental::resource_fd>
      wait_external_semaphore_desc{
          wait_semaphore_file_descriptor,
          sycl::ext::oneapi::experimental::external_semaphore_handle_type::
              opaque_fd};

  sycl::ext::oneapi::experimental::external_semaphore_descriptor<
      sycl::ext::oneapi::experimental::resource_fd>
      done_external_semaphore_desc{
          done_semaphore_file_descriptor,
          sycl::ext::oneapi::experimental::external_semaphore_handle_type::
              opaque_fd};

  // Extension: import external semaphores
  sycl::ext::oneapi::experimental::external_semaphore wait_external_semaphore =
      sycl::ext::oneapi::experimental::import_external_semaphore(
          wait_external_semaphore_desc, q);

  sycl::ext::oneapi::experimental::external_semaphore done_external_semaphore =
      sycl::ext::oneapi::experimental::import_external_semaphore(
          done_external_semaphore_desc, q);

  // Extension: import external memory from descriptors
  sycl::ext::oneapi::experimental::external_mem input_external_mem =
      sycl::ext::oneapi::experimental::import_external_memory(
          input_ext_mem_desc, q);

  sycl::ext::oneapi::experimental::external_mem output_external_mem =
      sycl::ext::oneapi::experimental::import_external_memory(
          output_ext_mem_desc, q);

  // Extension: map imported external memory to image memory
  sycl::ext::oneapi::experimental::image_mem_handle input_mapped_mem_handle =
      sycl::ext::oneapi::experimental::map_external_image_memory(
          input_external_mem, desc, q);
  sycl::ext::oneapi::experimental::image_mem_handle output_mapped_mem_handle =
      sycl::ext::oneapi::experimental::map_external_image_memory(
          output_external_mem, desc, q);

  // Extension: create images from mapped memory and return the handles
  sycl::ext::oneapi::experimental::unsampled_image_handle img_input =
      sycl::ext::oneapi::experimental::create_image(input_mapped_mem_handle,
                                                    desc, q);
  sycl::ext::oneapi::experimental::unsampled_image_handle img_output =
      sycl::ext::oneapi::experimental::create_image(output_mapped_mem_handle,
                                                    desc, q);

  // Extension: wait for imported semaphore
  q.ext_oneapi_wait_external_semaphore(wait_external_semaphore);

  // Submit our kernel that depends on imported
  // "wait_semaphore_file_descriptor"
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<>(
        sycl::nd_range<2>{{width, height}, {32, 32}}, [=](sycl::nd_item<2> it) {
          size_t dim0 = it.get_global_id(0);
          size_t dim1 = it.get_global_id(1);

          // Extension: read image data from handle to imported image
          uint32_t pixel =
              sycl::ext::oneapi::experimental::fetch_image<uint32_t>(
                  img_input, sycl::vec<int, 2>(dim0, dim1));

          // Modify the data before writing back
          pixel *= 10;

          // Extension: write image data using handle to imported image
          sycl::ext::oneapi::experimental::write_image(
              img_output, sycl::vec<int, 2>(dim0, dim1), pixel);
        });
  });

  // Extension: signal imported semaphore
  q.ext_oneapi_signal_external_semaphore(done_external_semaphore);

  // The external API can now use the semaphore it exported to
  // "done_semaphore_file_descriptor" to schedule its own command
  // submissions

  q.wait_and_throw();

  // Extension: destroy all external resources
  sycl::ext::oneapi::experimental::release_external_memory(input_external_mem,
                                                           q);
  sycl::ext::oneapi::experimental::release_external_memory(output_external_mem,
                                                           q);
  sycl::ext::oneapi::experimental::release_external_semaphore(
      wait_external_semaphore, q);
  sycl::ext::oneapi::experimental::release_external_semaphore(
      done_external_semaphore, q);
  sycl::ext::oneapi::experimental::destroy_image_handle(img_input, q);
  sycl::ext::oneapi::experimental::destroy_image_handle(img_output, q);

  return 0;
}
