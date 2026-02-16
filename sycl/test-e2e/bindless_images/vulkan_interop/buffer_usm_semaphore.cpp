// REQUIRES: aspect-ext_oneapi_external_memory_import, aspect-ext_oneapi_external_semaphore_import
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -DTEST_TIMELINE_SEMAPHORE -o %timeline_semaphore.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{build} %link-vulkan -o %binary_semaphore.out %if target-spir %{ -Wno-ignored-attributes %}

// RUN: %{run} %timeline_semaphore.out
// RUN: %{run} %binary_semaphore.out

/**
 * This test does not use any image specific APIs.
 *
 * It only tests exporting VkBuffer memory to SYCL and exporting a Vulkan
 * timeline semaphore and binary semaphore to SYCL. The data is manipulated in
 * Vulkan. SYCL waits to copy the data back to the host until the exported
 * semaphore signals that.
 */

#include "../../CommonUtils/vulkan_common.hpp"
#include "../helpers/common.hpp"

#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>

#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;
// Number of runs to test the case multiple times to detect race conditions when
// sharing semaphores between Vulkan and SYCL.
constexpr size_t NUMBER_RUNS = 100;

int runTest(sycl::queue &syclQueue) {

  int runStatus = 0;

  sycl::device syclDevice = syclQueue.get_device();
  constexpr size_t bufferSizeBytes = sizeof(float);
  float hostData = 111.0f;
  auto *hostPtr = sycl::malloc_host<float>(1, syclQueue);

  VkSemaphore vkCreatedSemaphore;

#ifdef TEST_TIMELINE_SEMAPHORE
  std::cout << "Running test with timeline sempahore. \n";
  uint64_t timelineValue = 0;
  VkSemaphoreTypeCreateInfo semaphoreTypeCreateInfo = {};
  semaphoreTypeCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  semaphoreTypeCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  semaphoreTypeCreateInfo.initialValue = timelineValue;
#else
  std::cout << "Running test with binary sempahore. \n";
#endif

  VkExportSemaphoreCreateInfo exportSemaphoreCreateInfo = {};
  exportSemaphoreCreateInfo.sType =
      VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;

#ifdef TEST_TIMELINE_SEMAPHORE
  exportSemaphoreCreateInfo.pNext = &semaphoreTypeCreateInfo;
#endif

#ifdef _WIN32
  exportSemaphoreCreateInfo.handleTypes =
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  exportSemaphoreCreateInfo.handleTypes =
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

  VkSemaphoreCreateInfo semaphoreCreateInfo = {};
  semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphoreCreateInfo.pNext = &exportSemaphoreCreateInfo;

  VK_CHECK_CALL(vkCreateSemaphore(vk_device, &semaphoreCreateInfo, nullptr,
                                  &vkCreatedSemaphore));

  VkBuffer vkDataBuffer;
  VkDeviceMemory vkDataBufferMemory;

  vkDataBuffer = vkutil::createBuffer(bufferSizeBytes,
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      true);

  auto dataBufferMemTypeIndex = vkutil::getBufferMemoryTypeIndex(
      vkDataBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkDataBufferMemory = vkutil::allocateDeviceMemory(
      bufferSizeBytes, dataBufferMemTypeIndex, VK_NULL_HANDLE, true);
  VK_CHECK_CALL(
      vkBindBufferMemory(vk_device, vkDataBuffer, vkDataBufferMemory, 0));

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingMemory;

  stagingBuffer = vkutil::createBuffer(bufferSizeBytes,
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  auto inputStagingMemTypeIndex = vkutil::getBufferMemoryTypeIndex(
      stagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  stagingMemory = vkutil::allocateDeviceMemory(
      bufferSizeBytes, inputStagingMemTypeIndex, VK_NULL_HANDLE, false);
  VK_CHECK_CALL(vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory, 0));

  float *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, stagingMemory, 0, bufferSizeBytes, 0,
                            (void **)&inputStagingData));
  memcpy(inputStagingData, &hostData, bufferSizeBytes);
  vkUnmapMemory(vk_device, stagingMemory);

  VK_CHECK_CALL(vkResetCommandBuffer(vk_transferCmdBuffers[0], 0));

  VkCommandBufferBeginInfo cmdBeginInfo = {};
  cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VkBufferCopy copyRegion = {};
  copyRegion.size = bufferSizeBytes;

  VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &cmdBeginInfo));
  vkCmdCopyBuffer(vk_transferCmdBuffers[0], stagingBuffer, vkDataBuffer, 1,
                  &copyRegion);
  VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[0]));

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &vk_transferCmdBuffers[0];

  VK_CHECK_CALL(
      vkQueueSubmit(vk_transfer_queue, 1, &submitInfo, VK_NULL_HANDLE));
  VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));

  vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
  vkFreeMemory(vk_device, stagingMemory, nullptr);

#ifdef _WIN32
  auto bufferVulkanExternalHandle =
      vkutil::getMemoryWin32Handle(vkDataBufferMemory);

  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      externalMemoryDescriptor{
          bufferVulkanExternalHandle,
          syclexp::external_mem_handle_type::win32_nt_handle, bufferSizeBytes};
#else
  auto bufferVulkanExternalHandle =
      vkutil::getMemoryOpaqueFD(vkDataBufferMemory);

  syclexp::external_mem_descriptor<syclexp::resource_fd>
      externalMemoryDescriptor{bufferVulkanExternalHandle,
                               syclexp::external_mem_handle_type::opaque_fd,
                               bufferSizeBytes};
#endif

  syclexp::external_mem externalMemory =
      syclexp::import_external_memory(externalMemoryDescriptor, syclQueue);

  float *externalMemPtr =
      static_cast<float *>(syclexp::map_external_linear_memory(
          externalMemory, 0, bufferSizeBytes, syclQueue));

#ifdef _WIN32
  auto syclWaitSemaphoreHandle =
      vkutil::getSemaphoreWin32Handle(vkCreatedSemaphore);

#ifdef TEST_TIMELINE_SEMAPHORE
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      syclWaitExternalSemaphoreDesc{
          syclWaitSemaphoreHandle,
          syclexp::external_semaphore_handle_type::timeline_win32_nt_handle};
#else
  syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
      syclWaitExternalSemaphoreDesc{
          syclWaitSemaphoreHandle,
          syclexp::external_semaphore_handle_type::win32_nt_handle};
#endif
#else
  auto syclWaitSemaphoreHandle =
      vkutil::getSemaphoreOpaqueFD(vkCreatedSemaphore);

#ifdef TEST_TIMELINE_SEMAPHORE
  syclexp::external_semaphore_descriptor<syclexp::resource_fd>
      syclWaitExternalSemaphoreDesc{
          syclWaitSemaphoreHandle,
          syclexp::external_semaphore_handle_type::timeline_fd};
#else
  syclexp::external_semaphore_descriptor<syclexp::resource_fd>
      syclWaitExternalSemaphoreDesc{
          syclWaitSemaphoreHandle,
          syclexp::external_semaphore_handle_type::opaque_fd};
#endif
#endif

  syclexp::external_semaphore syclWaitExternalSemaphore =
      syclexp::import_external_semaphore(syclWaitExternalSemaphoreDesc,
                                         syclDevice, syclQueue.get_context());

  VK_CHECK_CALL(vkResetCommandBuffer(vk_transferCmdBuffers[0], 0));

  cmdBeginInfo = VkCommandBufferBeginInfo{};
  cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  float newData = 69.0f;
  VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &cmdBeginInfo));
  vkCmdUpdateBuffer(vk_transferCmdBuffers[0], vkDataBuffer, 0, bufferSizeBytes,
                    &newData);
  VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[0]));

#ifdef TEST_TIMELINE_SEMAPHORE
  timelineValue++;
  VkTimelineSemaphoreSubmitInfo timelineInfo{};
  timelineInfo.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timelineInfo.signalSemaphoreValueCount = 1;
  timelineInfo.pSignalSemaphoreValues = &timelineValue;
#endif

  submitInfo = VkSubmitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &vk_transferCmdBuffers[0];
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &vkCreatedSemaphore;

#ifdef TEST_TIMELINE_SEMAPHORE
  submitInfo.pNext = &timelineInfo;
#endif

  VK_CHECK_CALL(
      vkQueueSubmit(vk_transfer_queue, 1, &submitInfo, VK_NULL_HANDLE));

#ifdef TEST_TIMELINE_SEMAPHORE
  syclQueue.ext_oneapi_wait_external_semaphore(syclWaitExternalSemaphore,
                                               timelineValue);
#else
  syclQueue.ext_oneapi_wait_external_semaphore(syclWaitExternalSemaphore);
#endif
  syclQueue.memcpy(hostPtr, externalMemPtr, bufferSizeBytes);
  syclQueue.wait();

  if (*hostPtr != newData) {
    std::cerr << "Race condition occurred.\n";
    ++runStatus;
  }

  syclexp::unmap_external_linear_memory(externalMemPtr, syclQueue);
  syclexp::release_external_memory(externalMemory, syclQueue);
  syclexp::release_external_semaphore(syclWaitExternalSemaphore, syclQueue);

  vkDeviceWaitIdle(vk_device);
  vkDestroyBuffer(vk_device, vkDataBuffer, nullptr);
  vkFreeMemory(vk_device, vkDataBufferMemory, nullptr);
  vkDestroySemaphore(vk_device, vkCreatedSemaphore, nullptr);

  sycl::free(hostPtr, syclQueue);

  return runStatus;
}

int main() {

  sycl::device syclDevice;
  sycl::queue syclQueue{syclDevice,
                        sycl::property_list{sycl::property::queue::in_order()}};

  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupDevice(syclDevice) != VK_SUCCESS) {
    std::cerr << "Device setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Command buffers setup failed!\n";
    return EXIT_FAILURE;
  }

  int numFails = 0;
  for (size_t i = 0; i < NUMBER_RUNS; ++i) {
    numFails += runTest(syclQueue);
  }

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  return numFails;
}
