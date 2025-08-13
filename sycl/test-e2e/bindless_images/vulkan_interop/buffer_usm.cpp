// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

/**
 * This test does not use any image specific APIs.
 *
 * It is only testing the exportation of VkBuffer memory, and importing and
 * mapping the VkBuffer memory into SYCL device memory. The imported SYCL device
 * memory is then manipulated purely through pointers inside the kernel.
 */

// Uncomment to print additional test information
// #define VERBOSE_PRINT

#include "../../CommonUtils/vulkan_common.hpp"
#include "../helpers/common.hpp"

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/usm.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

template <syclexp::external_mem_handle_type ExtMemHandleTypeV,
          typename InteropMemHandleT>
void runSycl(sycl::queue &syclQueue, sycl::range<1> globalSize,
             sycl::range<1> localSize, InteropMemHandleT extMemInHandle,
             InteropMemHandleT extMemOutHandle) {
  const size_t bufferSizeBytes = globalSize.size() * sizeof(uint32_t);

  using ResourceT =
      std::conditional_t<(ExtMemHandleTypeV ==
                          syclexp::external_mem_handle_type::win32_nt_handle),
                         syclexp::resource_win32_handle, syclexp::resource_fd>;

  syclexp::external_mem_descriptor<ResourceT> extMemInDesc{
      extMemInHandle, ExtMemHandleTypeV, bufferSizeBytes};
  syclexp::external_mem_descriptor<ResourceT> extMemOutDesc{
      extMemOutHandle, ExtMemHandleTypeV, bufferSizeBytes};

  // Extension: create interop memory handles.
  syclexp::external_mem externalMemIn =
      syclexp::import_external_memory(extMemInDesc, syclQueue);
  syclexp::external_mem externalMemOut =
      syclexp::import_external_memory(extMemOutDesc, syclQueue);

  // Extension: map linear memory handles.
  uint32_t *memIn = static_cast<uint32_t *>(syclexp::map_external_linear_memory(
      externalMemIn, 0 /* offset */, bufferSizeBytes, syclQueue));
  uint32_t *memOut =
      static_cast<uint32_t *>(syclexp::map_external_linear_memory(
          externalMemOut, 0 /* offset */, bufferSizeBytes, syclQueue));

  try {
    syclQueue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>{globalSize, localSize},
                       [=](sycl::nd_item<1> it) {
                         size_t index = it.get_global_id(0);

                         uint32_t bufferValue = memIn[index];
                         memOut[index] = bufferValue * 2;
                       });
    });

    // Wait for kernel completion before destroying external objects.
    syclQueue.wait_and_throw();

    // Cleanup.
    syclexp::unmap_external_linear_memory(memIn, syclQueue);
    syclexp::unmap_external_linear_memory(memOut, syclQueue);
    syclexp::release_external_memory(externalMemIn, syclQueue);
    syclexp::release_external_memory(externalMemOut, syclQueue);

  } catch (sycl::exception e) {
    std::cerr << "\tKernel submission failed! " << e.what() << std::endl;
    exit(-1);
  } catch (...) {
    std::cerr << "\tKernel submission failed!" << std::endl;
    exit(-1);
  }
}

template <syclexp::external_mem_handle_type ExtMemHandleTypeV>
bool runTest(const sycl::device &syclDevice, sycl::range<1> bufferSize,
             sycl::range<1> localSize) {
  sycl::queue syclQueue{syclDevice};
  if constexpr (ExtMemHandleTypeV ==
                syclexp::external_mem_handle_type::dma_buf) {
    if (!supportsDmaBuf) {
      std::cout
          << "dma_buf test skipped because Vulkan driver does not support it\n";
      return true;
    }
    if (!syclexp::supports_importing_handle_type(ExtMemHandleTypeV,
                                                 syclDevice)) {
      std::cout
          << "dma_buf test skipped because SYCL backend does not support it\n";
      return true;
    }
  }

  const size_t bufferSizeElems = bufferSize[0];
  const size_t bufferSizeBytes = bufferSizeElems * sizeof(uint32_t);

  VkBuffer vkInputBuffer;
  VkDeviceMemory vkInputBufferMemory;
  VkBuffer vkOutputBuffer;
  VkDeviceMemory vkOutputBufferMemory;

  // Initialize buffer input data.
  std::vector<uint32_t> inputVec(bufferSizeElems, 0.f);
  for (uint32_t i = 0; i < bufferSizeElems; ++i) {
    inputVec[i] = i;
  }

  // Create/allocate device buffers.
  {
    vkInputBuffer = vkutil::createBuffer(bufferSizeBytes,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                         true /*exportable*/);
    auto inputBufferMemTypeIndex = vkutil::getBufferMemoryTypeIndex(
        vkInputBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkInputBufferMemory = vkutil::allocateDeviceMemory(
        bufferSizeBytes, inputBufferMemTypeIndex, VK_NULL_HANDLE /*image*/);
    VK_CHECK_CALL(vkBindBufferMemory(vk_device, vkInputBuffer,
                                     vkInputBufferMemory, 0 /*memoryOffset*/));

    vkOutputBuffer = vkutil::createBuffer(
        bufferSizeBytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        true /* exportable */);
    auto outputBufferMemTypeIndex = vkutil::getBufferMemoryTypeIndex(
        vkOutputBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vkOutputBufferMemory = vkutil::allocateDeviceMemory(
        bufferSizeBytes, outputBufferMemTypeIndex, VK_NULL_HANDLE /*image*/);
    VK_CHECK_CALL(vkBindBufferMemory(vk_device, vkOutputBuffer,
                                     vkOutputBufferMemory, 0 /*memoryOffset*/));
  }

  // Allocate temporary staging buffer and copy input data to device.
  printString("Allocating staging memory and copying to device buffer\n");
  {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    stagingBuffer = vkutil::createBuffer(bufferSizeBytes,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto inputStagingMemTypeIndex = vkutil::getBufferMemoryTypeIndex(
        stagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingMemory = vkutil::allocateDeviceMemory(
        bufferSizeBytes, inputStagingMemTypeIndex, VK_NULL_HANDLE /*image*/,
        false /*exportable*/);
    VK_CHECK_CALL(vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory,
                                     0 /*memoryOffset*/));

    // Copy host data to temporary staging buffer.
    uint32_t *inputStagingData = nullptr;
    VK_CHECK_CALL(vkMapMemory(vk_device, stagingMemory, 0 /*offset*/,
                              bufferSizeBytes, 0 /*flags*/,
                              (void **)&inputStagingData));
    for (int i = 0; i < bufferSizeElems; ++i) {
      inputStagingData[i] = inputVec[i];
    }
    vkUnmapMemory(vk_device, stagingMemory);

    // Copy temporary staging buffer to device local buffer.
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferCopy copyRegion = {};
    copyRegion.size = bufferSizeBytes;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &cbbi));
    vkCmdCopyBuffer(vk_transferCmdBuffers[0], stagingBuffer, vkInputBuffer,
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

    // Destroy temporary staging buffer and free memory.
    vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
    vkFreeMemory(vk_device, stagingMemory, nullptr);
  }

  printString("Getting memory interop handles\n");
  // Get memory interop handles.
  const auto get_memory_handle = [](VkDeviceMemory vulkanDeviceMem) {
#ifdef _WIN32
    return vkutil::getMemoryWin32Handle(vulkanDeviceMem);
#else
    if constexpr (ExtMemHandleTypeV ==
                  syclexp::external_mem_handle_type::dma_buf) {
      return vkutil::getMemoryOpaqueFD<
          VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT>(vulkanDeviceMem);
    } else {
      return vkutil::getMemoryOpaqueFD(vulkanDeviceMem);
    }
#endif
  };
  auto bufferMemIn = get_memory_handle(vkInputBufferMemory);
  auto bufferMemOut = get_memory_handle(vkOutputBufferMemory);

  // Call into SYCL to read from input buffer, and populate the output buffer.
  printString("Calling into SYCL with interop memory handles\n");
  runSycl<ExtMemHandleTypeV>(syclQueue, bufferSize, localSize, bufferMemIn,
                             bufferMemOut);

  // Copy device buffer memory to temporary staging buffer, and back to host.
  printString("Copying buffer memory to host\n");
  std::vector<uint32_t> outputVec(bufferSizeElems, 0);
  {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    stagingBuffer = vkutil::createBuffer(bufferSizeBytes,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto outputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
        stagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingMemory = vkutil::allocateDeviceMemory(
        bufferSizeBytes, outputStagingMemoryTypeIndex, VK_NULL_HANDLE /*image*/,
        false /*exportable*/);
    VK_CHECK_CALL(vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory,
                                     0 /*memoryOffset*/));

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferCopy copyRegion = {};
    copyRegion.size = bufferSizeBytes;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[1], &cbbi));
    vkCmdCopyBuffer(vk_transferCmdBuffers[1], vkOutputBuffer, stagingBuffer,
                    1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[1]));

    std::vector<VkPipelineStageFlags> stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[1];
    submission.pWaitDstStageMask = stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));

    // Copy temporary staging buffer output data to host output vector.
    uint32_t *outputStagingData = (uint32_t *)outputVec.data();
    VK_CHECK_CALL(vkMapMemory(vk_device, stagingMemory, 0 /*offset*/,
                              bufferSizeBytes, 0 /*flags*/,
                              (void **)&outputStagingData));
    for (int i = 0; i < bufferSizeElems; ++i) {
      outputVec[i] = outputStagingData[i];
    }
    vkUnmapMemory(vk_device, stagingMemory);

    // Destroy temporary staging buffer and free memory.
    vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
    vkFreeMemory(vk_device, stagingMemory, nullptr);
  }

  // Destroy buffers and free their memory.
  vkDestroyBuffer(vk_device, vkInputBuffer, nullptr);
  vkDestroyBuffer(vk_device, vkOutputBuffer, nullptr);
  vkFreeMemory(vk_device, vkInputBufferMemory, nullptr);
  vkFreeMemory(vk_device, vkOutputBufferMemory, nullptr);

  // Validate that SYCL made changes to the memory.
  bool validated = true;
  for (int i = 0; i < bufferSizeElems; ++i) {
    uint32_t expected = inputVec[i] * 2;
    if (outputVec[i] != expected) {
      std::cerr << "Result mismatch! actual[" << i << "] == " << outputVec[i]
                << " : expected == " << expected << "\n";
      validated = false;
    }
    if (!validated)
      break;
  }

  if (validated) {
    printString("Results are correct!\n");
  }

  return validated;
}

int main() {

  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }

  sycl::device syclDevice;

  if (vkutil::setupDevice(syclDevice) != VK_SUCCESS) {
    std::cerr << "Device setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Command buffers setup failed!\n";
    return EXIT_FAILURE;
  }

  const auto globalSize = sycl::range<1>{1024};
  const auto localSize = sycl::range<1>{256};
#ifdef _WIN32
  const bool opaqueTestPassed =
      runTest<syclexp::external_mem_handle_type::win32_nt_handle>(
          syclDevice, globalSize, localSize);
  constexpr bool dmaBufTestPassed = true;
  // No check for opaqueTestPassed here, there is a common check later
#else
  const bool opaqueTestPassed =
      runTest<syclexp::external_mem_handle_type::opaque_fd>(
          syclDevice, globalSize, localSize);
  if (!opaqueTestPassed) {
    std::cout << "opaque_fd test failed!\n";
  }
  const bool dmaBufTestPassed =
      runTest<syclexp::external_mem_handle_type::dma_buf>(
          syclDevice, globalSize, localSize);
  if (!dmaBufTestPassed) {
    std::cout << "dma_buf test failed!\n";
  }
#endif

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  if (opaqueTestPassed && dmaBufTestPassed) {
    std::cout << "Test passed!\n";
    return EXIT_SUCCESS;
  }

  std::cerr << "Test failed\n";
  return EXIT_FAILURE;
}
