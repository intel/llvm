
//
// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: vulkan && level_zero
//
// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

// Waiting on an external semaphore from a queue backed by a regular
// (non-immediate) command list must throw sycl::exception.
//
// sycl_ext_oneapi_bindless_images requires queues used with external
// semaphore wait/signal to be constructed with BOTH
//   - sycl::property::queue::in_order
//   - sycl::ext::intel::property::queue::immediate_command_list
// The Level Zero adapter rejects external_semaphore wait/signal at the
// point of submission when the queue is not using immediate command
// lists. This test verifies that contract by:
//   1. Creating a real, exportable Vulkan binary semaphore.
//   2. Importing it into SYCL via a (lawful) immediate-CL queue.
//   3. Calling ext_oneapi_wait_external_semaphore on a separate queue
//      that explicitly opts into no_immediate_command_list, and
//      expecting a sycl::exception.

#include "vulkan_setup.hpp"
#include <iostream>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/properties/queue_properties.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

int main() {
  VulkanContext vkCtx = createVulkanContext();
  VkSemaphore vkSem = createExportableSemaphore(vkCtx);

  // Lawful queue: import the semaphore here.
  sycl::queue immQ{
      {sycl::property::queue::in_order{},
       sycl::ext::intel::property::queue::immediate_command_list{}}};
  auto device = immQ.get_device();
  auto context = immQ.get_context();

#ifdef _WIN32
  HANDLE semHandle = getSemaphoreHandle(vkCtx, vkSem);
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle> desc{
      semHandle, syclexp::external_semaphore_handle_type::win32_nt_handle};
#else
  int semFd = getSemaphoreFd(vkCtx, vkSem);
  syclexp::external_semaphore_descriptor<syclexp::resource_fd> desc{
      semFd, syclexp::external_semaphore_handle_type::opaque_fd};
#endif

  syclexp::external_semaphore syclSem =
      syclexp::import_external_semaphore(desc, device, context);

  // The non-immediate-CL queue is what should trigger rejection on use.
  sycl::queue regQ{
      context, device,
      sycl::property_list{
          sycl::property::queue::in_order{},
          sycl::ext::intel::property::queue::no_immediate_command_list{}}};

  int ret = 1;
  try {
    regQ.ext_oneapi_wait_external_semaphore(syclSem);
    regQ.wait_and_throw();
    std::cerr << "FAIL: ext_oneapi_wait_external_semaphore on a "
                 "non-immediate-CL queue did not throw."
              << std::endl;
  } catch (const sycl::exception &e) {
    std::cout << "Got expected sycl::exception: " << e.what() << std::endl;
    ret = 0;
  }

  syclexp::release_external_semaphore(syclSem, device, context);
  vkDestroySemaphore(vkCtx.device, vkSem, nullptr);
  cleanupVulkanContext(vkCtx);
  return ret;
}
