// REQUIRES: aspect-ext_oneapi_memory_export_linear
// REQUIRES: target-spir
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

#include <iostream>
#include <sycl/ext/oneapi/memory_export.hpp>
#include <sycl/sycl.hpp>

#include "../CommonUtils/vulkan_common.hpp"

namespace syclexp = sycl::ext::oneapi::experimental;

using DataT = uint32_t;

namespace {
void *syclExportableLinearMemory;

#ifndef _WIN32
syclexp::resource_fd exportableMemoryHandle;
#else
syclexp::resource_win32_handle exportableMemoryHandle;
#endif // _WIN32

std::vector<DataT> syclInput;
std::vector<DataT> vulkanOutput;

} // namespace

void initSycl(const sycl::device &syclDevice, const size_t memorySizeBytes,
              size_t memoryAlignment) {
  sycl::context syclContext = sycl::context(syclDevice);
  sycl::queue syclQueue(syclContext, syclDevice);

  // Allocate SYCL exportable memory.
  syclExportableLinearMemory = syclexp::alloc_exportable_memory(
      memoryAlignment, memorySizeBytes,
      syclexp::external_mem_handle_type::opaque_fd, syclDevice, syclContext);

  // Fill the SYCL allocated memory with some data.
  syclInput.resize(memorySizeBytes / sizeof(DataT), 0);
  std::iota(syclInput.begin(), syclInput.end(), 0);

  syclQueue.copy<DataT>(syclInput.data(),
                        static_cast<DataT *>(syclExportableLinearMemory),
                        memorySizeBytes / sizeof(DataT));
  syclQueue.wait_and_throw();

  // Export the SYCL allocated memory handle.
#ifndef _WIN32
  exportableMemoryHandle = syclexp::export_memory_handle<syclexp::resource_fd>(
      syclExportableLinearMemory, syclDevice, syclContext);
#else
  exportableMemoryHandle =
      syclexp::export_memory_handle<syclexp::resource_win32_handle>(
          syclExportableLinearMemory, syclDevice, syclContext);
#endif // _WIN32

  return;
}

void cleanupSycl(const sycl::device &syclDevice) {
  sycl::context syclContext = sycl::context(syclDevice);
  syclexp::free_exportable_memory(syclExportableLinearMemory, syclDevice,
                                  syclContext);
}

int runTest(sycl::device &syclDevice, const size_t memorySizeBytes) {

  sycl::context syclContext = sycl::context(syclDevice);
  sycl::queue syclQueue(syclContext, syclDevice);

  VkBuffer vkImportedBuffer;
  VkDeviceMemory vkImportedBufferMemory;

  {
    vkImportedBuffer = vkutil::createBuffer(
        memorySizeBytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        true /*exportable*/);
    auto inputBufferMemTypeIndex = vkutil::getBufferMemoryTypeIndex(
        vkImportedBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

#ifndef _WIN32
    vkImportedBufferMemory = vkutil::importDeviceMemory<int>(
        memorySizeBytes, inputBufferMemTypeIndex,
        exportableMemoryHandle.file_descriptor);
#else
    vkImportedBufferMemory = vkutil::importDeviceMemoryWin32<void *>(
        memorySizeBytes, inputBufferMemTypeIndex,
        exportableMemoryHandle.win32_handle);
#endif

    VK_CHECK_CALL(vkBindBufferMemory(vk_device, vkImportedBuffer,
                                     vkImportedBufferMemory,
                                     0 /*memoryOffset*/));
  }

  // Allocate temporary staging buffer and copy imported data to host.
  vulkanOutput.resize(memorySizeBytes / sizeof(DataT), 0);
  {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    stagingBuffer = vkutil::createBuffer(memorySizeBytes,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto inputStagingMemTypeIndex = vkutil::getBufferMemoryTypeIndex(
        stagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingMemory = vkutil::allocateDeviceMemory(
        memorySizeBytes, inputStagingMemTypeIndex, VK_NULL_HANDLE /*image*/,
        false /*exportable*/);
    VK_CHECK_CALL(vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory,
                                     0 /*memoryOffset*/));

    // Copy imported buffer to host visible staging buffer.
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferCopy copyRegion = {};
    copyRegion.size = memorySizeBytes;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &cbbi));
    vkCmdCopyBuffer(vk_transferCmdBuffers[0], vkImportedBuffer, stagingBuffer,
                    1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[0]));

    std::vector<VkPipelineStageFlags> stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[0];
    submission.pWaitDstStageMask = stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));

    // Copy host visible staging buffer data to host.
    DataT *stagingData = nullptr;
    VK_CHECK_CALL(vkMapMemory(vk_device, stagingMemory, 0 /*offset*/,
                              memorySizeBytes, 0 /*flags*/,
                              (void **)&stagingData));
    for (int i = 0; i < memorySizeBytes / sizeof(DataT); ++i) {
      vulkanOutput[i] = stagingData[i];
    }
    vkUnmapMemory(vk_device, stagingMemory);

    // Destroy temporary staging buffer and free memory.
    vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
    vkFreeMemory(vk_device, stagingMemory, nullptr);
  }

  vkDestroyBuffer(vk_device, vkImportedBuffer, nullptr);
  vkFreeMemory(vk_device, vkImportedBufferMemory, nullptr);

  // Print the SYCL imported data.
  bool validated = true;
  for (size_t i = 0; i < vulkanOutput.size(); ++i) {
    if (vulkanOutput[i] != syclInput[i]) {
      std::cerr << "Data mismatch at index " << i << ": expected "
                << syclInput[i] << ", actual " << vulkanOutput[i] << "\n";
      validated = false;
      break;
    }
  }

  return validated;
}

int main(int argc, char *argv[]) {

  // Default values for memory buffer size and alignment.
  // These can be overridden by command line arguments.
  // Usage: ./export_memory_to_vulkan <buffer_elements> <buffer_alignment>
  size_t bufferElems = 1024;
  size_t memoryAlignment = 0;

  if (argc >= 2) {
    bufferElems = static_cast<size_t>(std::stoull(argv[1]));
  }
  if (argc >= 3) {
    memoryAlignment = static_cast<size_t>(std::stoull(argv[2]));
  }

  const size_t memorySizeBytes = bufferElems * sizeof(DataT);

  sycl::device syclDevice;

  // Check if the device supports memory export
  bool syclHasExportSupport =
      syclDevice.has(sycl::aspect::ext_oneapi_memory_export_linear);

  if (!syclHasExportSupport) {
    std::cerr << "Device does not support memory export.\n";
    return 1;
  } else {
    std::cout << "Device supports memory export.\n";
  }

  // Init SYCL. Allocate exportable memory and get interop handle.
  initSycl(syclDevice, memorySizeBytes, memoryAlignment);

  // Init Vulkan
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

  auto testPassed = runTest(syclDevice, memorySizeBytes);

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  // Cleanup SYCL
  cleanupSycl(syclDevice);

  if (testPassed) {
    std::cout << "Test passed!\n";
    return EXIT_SUCCESS;
  }

  std::cerr << "Test failed\n";
  return EXIT_FAILURE;
}
