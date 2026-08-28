// REQUIRES: aspect-ext_oneapi_exportable_device_mem
// REQUIRES: target-spir
// REQUIRES: vulkan

// XFAIL: windows && run-mode
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/21125

// clang-format off

// UNSUPPORTED: arch-intel_gpu_pvc
// UNSUPPORTED-INTENDED: Our PVC runners don't have the userspace Vulkan driver installed

// clang-format on

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

#include <iostream>
#include <numeric>
#include <sycl/aspects.hpp>
#include <sycl/ext/oneapi/memory_export.hpp>

#include "../bindless_images/vulkan_interop/vulkan_setup.hpp"

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

int runTest(VulkanContext &VulkanCtx, sycl::device &SyclDevice,
            const size_t MemorySizeBytes) {

  sycl::context SyclContext = sycl::context(SyclDevice);
  sycl::queue SyclQueue(SyclContext, SyclDevice);

  VkBuffer VkImportedBuffer;
  VkDeviceMemory VkImportedBufferMemory;

  {
    VkExternalMemoryBufferCreateInfo ExternalInfo = {
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ExternalInfo.handleTypes = PLATFORM_MEM_HANDLE_TYPE;
    VkBufferCreateInfo BufferInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    BufferInfo.pNext = &ExternalInfo;
    BufferInfo.size = MemorySizeBytes;
    BufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    BufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(VulkanCtx.device, &BufferInfo, nullptr,
                            &VkImportedBuffer));

    VkMemoryRequirements Requirements;
    vkGetBufferMemoryRequirements(VulkanCtx.device, VkImportedBuffer,
                                  &Requirements);
    VkMemoryAllocateInfo AllocateInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    AllocateInfo.allocationSize = Requirements.size;
    AllocateInfo.memoryTypeIndex =
        findMemoryType(VulkanCtx.physicalDevice, Requirements.memoryTypeBits,
                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
#ifdef _WIN32
    VkImportMemoryWin32HandleInfoKHR ImportInfo = {
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    ImportInfo.handleType = PLATFORM_MEM_HANDLE_TYPE;
    ImportInfo.handle = ExportableMemoryHandle;
#else
    VkImportMemoryFdInfoKHR ImportInfo = {
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR};
    ImportInfo.handleType = PLATFORM_MEM_HANDLE_TYPE;
    ImportInfo.fd = ExportableMemoryHandle;
#endif
    AllocateInfo.pNext = &ImportInfo;
    VK_CHECK(vkAllocateMemory(VulkanCtx.device, &AllocateInfo, nullptr,
                              &VkImportedBufferMemory));
    VK_CHECK(vkBindBufferMemory(VulkanCtx.device, VkImportedBuffer,
                                VkImportedBufferMemory, 0));
  }

  // Allocate temporary staging buffer and copy imported data to host.
  VulkanOutput.resize(MemorySizeBytes / sizeof(DataT), 0);
  {
    VkBuffer StagingBuffer;
    VkDeviceMemory StagingMemory;

    auto Staging = createStagingBuffer(VulkanCtx, MemorySizeBytes,
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    StagingBuffer = Staging.buffer;
    StagingMemory = Staging.memory;

    // Copy imported buffer to host visible staging buffer.
    VkCommandBufferBeginInfo Cbbi = {};
    Cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    Cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferCopy CopyRegion = {};
    CopyRegion.size = MemorySizeBytes;

    VkCommandPoolCreateInfo PoolInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    PoolInfo.queueFamilyIndex = VulkanCtx.queueFamilyIndex;
    VkCommandPool Pool;
    VK_CHECK(vkCreateCommandPool(VulkanCtx.device, &PoolInfo, nullptr, &Pool));
    VkCommandBufferAllocateInfo CmdAllocInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    CmdAllocInfo.commandPool = Pool;
    CmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    CmdAllocInfo.commandBufferCount = 1;
    VkCommandBuffer CommandBuffer;
    VK_CHECK(vkAllocateCommandBuffers(VulkanCtx.device, &CmdAllocInfo,
                                      &CommandBuffer));
    VK_CHECK(vkBeginCommandBuffer(CommandBuffer, &Cbbi));
    vkCmdCopyBuffer(CommandBuffer, VkImportedBuffer, StagingBuffer,
                    1 /*regionCount*/, &CopyRegion);
    VK_CHECK(vkEndCommandBuffer(CommandBuffer));

    std::vector<VkPipelineStageFlags> Stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    VkSubmitInfo Submission = {};
    Submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    Submission.commandBufferCount = 1;
    Submission.pCommandBuffers = &CommandBuffer;
    Submission.pWaitDstStageMask = Stages.data();

    VK_CHECK(vkQueueSubmit(VulkanCtx.queue, 1, &Submission, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(VulkanCtx.queue));

    // Copy host visible staging buffer data to host.
    DataT *StagingData = nullptr;
    VK_CHECK(vkMapMemory(VulkanCtx.device, StagingMemory, 0 /*offset*/,
                         MemorySizeBytes, 0 /*flags*/, (void **)&StagingData));
    for (int i = 0; i < MemorySizeBytes / sizeof(DataT); ++i) {
      VulkanOutput[i] = StagingData[i];
    }
    vkUnmapMemory(VulkanCtx.device, StagingMemory);

    // Destroy temporary staging buffer and free memory.
    vkDestroyCommandPool(VulkanCtx.device, Pool, nullptr);
    cleanupBuffer(VulkanCtx, Staging);
  }

  vkDestroyBuffer(VulkanCtx.device, VkImportedBuffer, nullptr);
  vkFreeMemory(VulkanCtx.device, VkImportedBufferMemory, nullptr);

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
  VulkanContext VulkanCtx;
  try {
    VulkanCtx = createVulkanContext();
  } catch (const std::exception &e) {
    std::cerr << "Vulkan setup failed: " << e.what() << "\n";
    return 4;
  }

  auto TestPassed = runTest(VulkanCtx, SyclDevice, MemorySizeBytes);
  cleanupVulkanContext(VulkanCtx);

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
