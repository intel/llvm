
//
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
//
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Importing an external semaphore on a queue backed by a regular
// (non-immediate) command list must throw sycl::exception.
//
// sycl_ext_oneapi_bindless_images requires queues used with external
// semaphores to be constructed with BOTH
//   - sycl::property::queue::in_order
//   - sycl::ext::intel::property::queue::immediate_command_list
// This test violates the second requirement (explicitly requests
// no_immediate_command_list) and verifies the runtime rejects the import.
//
// This is a contract test, not an interop test: no real Vulkan context is
// created. The runtime rejects at import before inspecting the handle, so
// a bogus fd / handle is enough. That's also why this test does not gate
// on `vulkan` or link against the Vulkan loader -- it lives here because
// the Vulkan-flavored handle types are what most readers will look for.

#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/properties/all_properties.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q{
      {sycl::property::queue::in_order{},
       sycl::ext::intel::property::queue::no_immediate_command_list{}}};

#ifdef _WIN32
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle> desc{
      /*handle=*/nullptr,
      syclexp::external_semaphore_handle_type::win32_nt_handle};
#else
  syclexp::external_semaphore_descriptor<syclexp::resource_fd> desc{
      /*file_descriptor=*/-1,
      syclexp::external_semaphore_handle_type::opaque_fd};
#endif

  try {
    (void)syclexp::import_external_semaphore(desc, q);
  } catch (const sycl::exception &e) {
    std::cout << "Got expected sycl::exception: " << e.what() << std::endl;
    return 0;
  }

  std::cerr << "FAIL: import_external_semaphore on a non-immediate-CL queue "
               "did not throw."
            << std::endl;
  return 1;
}
