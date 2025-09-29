// REQUIRES: aspect-ext_oneapi_exportable_device_mem
// REQUIRES: target-spir
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

#include <iostream>
#include <numeric>
#include <sycl/ext/oneapi/memory_export.hpp>

#include "../CommonUtils/vulkan_common.hpp"

namespace syclexp = sycl::ext::oneapi::experimental;

using DataT = uint32_t;

#ifdef _WIN32
using exported_handle_type = void *;
#else
using exported_handle_type = int;
#endif // _WIN32
namespace {
void *SyclExportableLinearMemory;

std::vector<DataT> SyclInput;
std::vector<DataT> VulkanOutput;

#ifdef _WIN32
constexpr auto ExportHandleType =
    syclexp::external_mem_handle_type::win32_nt_handle;
#else
constexpr auto ExportHandleType = syclexp::external_mem_handle_type::opaque_fd;
#endif // _WIN32

exported_handle_type ExportableMemoryHandle;

} // namespace

void initSycl(const sycl::device &SyclDevice, const size_t MemorySizeBytes,
              size_t MemoryAlignment) {
  sycl::context SyclContext = sycl::context(SyclDevice);
  sycl::queue SyclQueue(SyclContext, SyclDevice);

  // Allocate SYCL exportable memory.
  SyclExportableLinearMemory = syclexp::alloc_exportable_device_mem(
      MemoryAlignment, MemorySizeBytes, ExportHandleType, SyclDevice,
      SyclContext);

  // Fill the SYCL allocated memory with some data.
  SyclInput.resize(MemorySizeBytes / sizeof(DataT), 0);
  std::iota(SyclInput.begin(), SyclInput.end(), 0);

  SyclQueue.copy<DataT>(SyclInput.data(),
                        static_cast<DataT *>(SyclExportableLinearMemory),
                        MemorySizeBytes / sizeof(DataT));
  SyclQueue.wait_and_throw();

  // Export the SYCL allocated memory handle.
  ExportableMemoryHandle = syclexp::export_device_mem_handle<ExportHandleType>(
      SyclExportableLinearMemory, SyclDevice, SyclContext);

  return;
}

void cleanupSycl(const sycl::device &SyclDevice) {
  sycl::context SyclContext = sycl::context(SyclDevice);
  syclexp::free_exportable_memory(SyclExportableLinearMemory, SyclDevice,
                                  SyclContext);
}

int runTest(sycl::device &SyclDevice, const size_t MemorySizeBytes) {

  sycl::context SyclContext = sycl::context(SyclDevice);
  sycl::queue SyclQueue(SyclContext, SyclDevice);

  VkBuffer VkImportedBuffer;
  VkDeviceMemory VkImportedBufferMemory;

  {
    VkImportedBuffer = vkutil::createBuffer(
        MemorySizeBytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        true /*exportable*/);
    auto InputBufferMemTypeIndex = vkutil::getBufferMemoryTypeIndex(
        VkImportedBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkImportedBufferMemory = vkutil::importDeviceMemory<exported_handle_type>(
        MemorySizeBytes, InputBufferMemTypeIndex, ExportableMemoryHandle);

    VK_CHECK_CALL(vkBindBufferMemory(vk_device, VkImportedBuffer,
                                     VkImportedBufferMemory,
                                     0 /*memoryOffset*/));
  }

  // Allocate temporary staging buffer and copy imported data to host.
  VulkanOutput.resize(MemorySizeBytes / sizeof(DataT), 0);
  {
    VkBuffer StagingBuffer;
    VkDeviceMemory StagingMemory;

    StagingBuffer = vkutil::createBuffer(MemorySizeBytes,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto InputStagingMemTypeIndex = vkutil::getBufferMemoryTypeIndex(
        StagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    StagingMemory = vkutil::allocateDeviceMemory(
        MemorySizeBytes, InputStagingMemTypeIndex, VK_NULL_HANDLE /*image*/,
        false /*exportable*/);
    VK_CHECK_CALL(vkBindBufferMemory(vk_device, StagingBuffer, StagingMemory,
                                     0 /*memoryOffset*/));

    // Copy imported buffer to host visible staging buffer.
    VkCommandBufferBeginInfo Cbbi = {};
    Cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    Cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferCopy CopyRegion = {};
    CopyRegion.size = MemorySizeBytes;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &Cbbi));
    vkCmdCopyBuffer(vk_transferCmdBuffers[0], VkImportedBuffer, StagingBuffer,
                    1 /*regionCount*/, &CopyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[0]));

    std::vector<VkPipelineStageFlags> Stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    VkSubmitInfo Submission = {};
    Submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    Submission.commandBufferCount = 1;
    Submission.pCommandBuffers = &vk_transferCmdBuffers[0];
    Submission.pWaitDstStageMask = Stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &Submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));

    // Copy host visible staging buffer data to host.
    DataT *StagingData = nullptr;
    VK_CHECK_CALL(vkMapMemory(vk_device, StagingMemory, 0 /*offset*/,
                              MemorySizeBytes, 0 /*flags*/,
                              (void **)&StagingData));
    for (int i = 0; i < MemorySizeBytes / sizeof(DataT); ++i) {
      VulkanOutput[i] = StagingData[i];
    }
    vkUnmapMemory(vk_device, StagingMemory);

    // Destroy temporary staging buffer and free memory.
    vkDestroyBuffer(vk_device, StagingBuffer, nullptr);
    vkFreeMemory(vk_device, StagingMemory, nullptr);
  }

  vkDestroyBuffer(vk_device, VkImportedBuffer, nullptr);
  vkFreeMemory(vk_device, VkImportedBufferMemory, nullptr);

  // Print the SYCL imported data.
  bool Validated = true;
  for (size_t i = 0; i < VulkanOutput.size(); ++i) {
    if (VulkanOutput[i] != SyclInput[i]) {
      std::cerr << "Data mismatch at index " << i << ": expected "
                << SyclInput[i] << ", actual " << VulkanOutput[i] << "\n";
      Validated = false;
      break;
    }
  }

  return Validated;
}

int main(int argc, char *argv[]) {

  // Default values for memory buffer size and alignment.
  // These can be overridden by command line arguments.
  // Usage: ./export_memory_to_vulkan <buffer_elements> <buffer_alignment>
  size_t BufferElems = 1024;
  size_t MemoryAlignment = 0;

  if (argc >= 2) {
    BufferElems = static_cast<size_t>(std::stoull(argv[1]));
  }
  if (argc >= 3) {
    MemoryAlignment = static_cast<size_t>(std::stoull(argv[2]));
  }

  const size_t MemorySizeBytes = BufferElems * sizeof(DataT);

  sycl::device SyclDevice;

  // Check if the device supports memory export
  bool SyclHasExportSupport =
      SyclDevice.has(sycl::aspect::ext_oneapi_exportable_device_mem);

  if (!SyclHasExportSupport) {
    std::cerr << "Device does not support memory export.\n";
    return 1;
  } else {
    std::cout << "Device supports memory export.\n";
  }

  // Init SYCL. Allocate exportable memory and get interop handle.
  try {
    initSycl(SyclDevice, MemorySizeBytes, MemoryAlignment);
  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception caught: " << e.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "Unknown exception caught.\n";
    return 3;
  }

  // Init Vulkan.
  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return 4;
  }

  if (vkutil::setupDevice(SyclDevice) != VK_SUCCESS) {
    std::cerr << "Device setup failed!\n";
    return 5;
  }

  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Command buffers setup failed!\n";
    return 6;
  }

  auto TestPassed = runTest(SyclDevice, MemorySizeBytes);

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return 7;
  }

  // Cleanup SYCL.
  try {
    cleanupSycl(SyclDevice);
  } catch (const sycl::exception &e) {
    std::cerr << "SYCL exception caught: " << e.what() << "\n";
    return 8;
  } catch (...) {
    std::cerr << "Unknown exception caught.\n";
    return 9;
  }

  if (TestPassed) {
    std::cout << "Test passed!\n";
    return 0;
  }

  std::cerr << "Test failed\n";
  return 10;
}
