// REQUIRES: aspect-ext_oneapi_external_semaphore_import
// REQUIRES: aspect-usm_shared_allocations
// REQUIRES: vulkan && level_zero

// Regular command-list external semaphore support requires these driver
// versions.
// REQUIRES-INTEL-DRIVER: lin: 39758, win: 101.9030

// XFAIL: windows && run-mode && gpu-intel-gen12
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/23249

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: env UR_L0_BATCH_SIZE=8 %{run} %t.out

// Verify that external semaphore wait and signal operations work on an
// in-order queue backed by regular (non-immediate) command lists. A timeline
// semaphore provides equivalent coverage on both platforms while avoiding
// Linux binary semaphore sharing issue CMPLRLLVM-78008.

#include "sycl_vulkan_setup.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sycl/atomic_ref.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>
#include <thread>

namespace syclexp = sycl::ext::oneapi::experimental;

[[noreturn]] void failWithoutCleanup(const std::string &message) {
  std::cerr << message << std::endl;
  std::_Exit(EXIT_FAILURE);
}

using MarkerAtomicRef =
    sycl::atomic_ref<int, sycl::memory_order::relaxed,
                     sycl::memory_scope::system,
                     sycl::access::address_space::global_space>;

bool waitForMarker(int *marker) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (MarkerAtomicRef(*marker).load() == 0) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

bool remainsIncomplete(const sycl::event &event) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  do {
    if (event.get_info<sycl::info::event::command_execution_status>() ==
        sycl::info::event_command_status::complete)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  return true;
}

bool waitForCompletion(const sycl::event &event) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (event.get_info<sycl::info::event::command_execution_status>() !=
         sycl::info::event_command_status::complete) {
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

int main() {
  constexpr uint64_t VulkanSignalValue = 1;
  constexpr uint64_t SyclSignalValue = 2;

  sycl::queue q{
      {sycl::property::queue::in_order{},
       sycl::ext::intel::property::queue::no_immediate_command_list{}}};
  auto device = q.get_device();
  auto context = q.get_context();

  VulkanContext vkCtx = createSyclVulkanContext(device);
  VkSemaphore vkSem = createExportableTimelineSemaphore(vkCtx);
#ifdef _WIN32
  HANDLE semHandle = getSemaphoreHandle(vkCtx, vkSem);
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle> desc{
      semHandle,
      syclexp::external_semaphore_handle_type::timeline_win32_nt_handle};
#else
  int semFd = getSemaphoreFd(vkCtx, vkSem);
  syclexp::external_semaphore_descriptor<syclexp::resource_fd> desc{
      semFd, syclexp::external_semaphore_handle_type::timeline_fd};
#endif
  syclexp::external_semaphore syclSem =
      syclexp::import_external_semaphore(desc, device, context);
  int *marker = sycl::malloc_shared<int>(1, q);
  MarkerAtomicRef(*marker).store(0);

  try {
    q.single_task([=]() { MarkerAtomicRef(*marker).store(1); });
    sycl::event waitEvent =
        q.ext_oneapi_wait_external_semaphore(syclSem, VulkanSignalValue);

    if (!waitForMarker(marker))
      failWithoutCleanup(
          "The external semaphore wait did not submit the preceding batch.");
    if (!remainsIncomplete(waitEvent))
      failWithoutCleanup(
          "The external semaphore wait completed before Vulkan signaled it.");

    VkSemaphoreSignalInfo signalInfo = {
        VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
    signalInfo.semaphore = vkSem;
    signalInfo.value = VulkanSignalValue;
    VkResult signalResult = vkSignalSemaphore(vkCtx.device, &signalInfo);
    if (signalResult != VK_SUCCESS)
      failWithoutCleanup("Failed to signal the Vulkan timeline semaphore: " +
                         std::to_string(signalResult));
    if (!waitForCompletion(waitEvent))
      failWithoutCleanup(
          "The external semaphore wait did not complete after Vulkan signaled "
          "it.");

    q.ext_oneapi_signal_external_semaphore(syclSem, SyclSignalValue);

    VkSemaphoreWaitInfo waitInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &vkSem;
    waitInfo.pValues = &SyclSignalValue;

    constexpr uint64_t TimeoutNs = 5'000'000'000;
    VkResult waitResult = vkWaitSemaphores(vkCtx.device, &waitInfo, TimeoutNs);
    if (waitResult == VK_TIMEOUT)
      failWithoutCleanup(
          "Timed out waiting for the regular command-list semaphore signal.");
    if (waitResult != VK_SUCCESS)
      failWithoutCleanup("Waiting for the Vulkan timeline semaphore failed: " +
                         std::to_string(waitResult));

    q.wait_and_throw();
  } catch (const std::exception &e) {
    failWithoutCleanup(e.what());
  }

  sycl::free(marker, q);
  syclexp::release_external_semaphore(syclSem, device, context);
  vkDestroySemaphore(vkCtx.device, vkSem, nullptr);
  cleanupVulkanContext(vkCtx);
  return 0;
}
