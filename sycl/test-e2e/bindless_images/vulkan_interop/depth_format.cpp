// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT

#include "../helpers/common.hpp"
#include "vulkan_common.hpp"

#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

template <typename InteropMemHandleT>
void runSycl(const sycl::device &syclDevice, sycl::range<2> globalSize,
             sycl::range<2> localSize, InteropMemHandleT extMemInHandle,
             InteropMemHandleT extMemOutHandle) {

  sycl::queue syclQueue{syclDevice};

  const size_t imgSizeBytes = globalSize.size() * sizeof(float);

#ifdef _WIN32
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle> extMemInDesc{
      extMemInHandle, syclexp::external_mem_handle_type::win32_nt_handle,
      imgSizeBytes};
  syclexp::external_mem_descriptor<syclexp::resource_win32_handle>
      extMemOutDesc{extMemOutHandle,
                    syclexp::external_mem_handle_type::win32_nt_handle,
                    imgSizeBytes};
#else
  syclexp::external_mem_descriptor<syclexp::resource_fd> extMemInDesc{
      extMemInHandle, syclexp::external_mem_handle_type::opaque_fd,
      imgSizeBytes};
  syclexp::external_mem_descriptor<syclexp::resource_fd> extMemOutDesc{
      extMemOutHandle, syclexp::external_mem_handle_type::opaque_fd,
      imgSizeBytes};
#endif

  // Extension: create interop memory handles.
  syclexp::external_mem externalMemIn =
      syclexp::import_external_memory(extMemInDesc, syclQueue);
  syclexp::external_mem externalMemOut =
      syclexp::import_external_memory(extMemOutDesc, syclQueue);

  // Image descriptor - Vulkan depth texture mapped to single channel fp32
  // image.
  syclexp::image_descriptor imgDesc(globalSize, 1,
                                    sycl::image_channel_type::fp32);

  // Extension: map image memory handles.
  syclexp::image_mem_handle imgMemIn =
      syclexp::map_external_image_memory(externalMemIn, imgDesc, syclQueue);
  syclexp::image_mem_handle imgMemOut =
      syclexp::map_external_image_memory(externalMemOut, imgDesc, syclQueue);

  // Extension: create the image and return the handle.
  syclexp::unsampled_image_handle imgIn =
      syclexp::create_image(imgMemIn, imgDesc, syclQueue);
  syclexp::unsampled_image_handle imgOut =
      syclexp::create_image(imgMemOut, imgDesc, syclQueue);

  try {
    syclQueue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class TestDepthTextureFetch>(
          sycl::nd_range<2>{globalSize, localSize}, [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_global_id(0);
            size_t dim1 = it.get_global_id(1);

            float depth =
                syclexp::fetch_image<float>(imgIn, sycl::int2(dim0, dim1));

            syclexp::write_image<float>(imgOut, sycl::int2(dim0, dim1), depth);
          });
    });

    // Wait for kernel completion before destroying external objects.
    syclQueue.wait_and_throw();

    // Cleanup.
    syclexp::destroy_image_handle(imgIn, syclQueue);
    syclexp::destroy_image_handle(imgOut, syclQueue);
    syclexp::unmap_external_image_memory(
        imgMemIn, syclexp::image_type::standard, syclQueue);
    syclexp::free_image_mem(imgMemOut, syclexp::image_type::standard,
                            syclQueue);
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

bool runTest(const sycl::device &syclDevice, sycl::range<2> dims,
             sycl::range<2> localSize) {
  const uint32_t imgWidth = static_cast<uint32_t>(dims[0]);
  const uint32_t imgHeight = static_cast<uint32_t>(dims[1]);

  const VkImageType imgType = VK_IMAGE_TYPE_2D;
  const VkFormat imgInFormat = VK_FORMAT_D32_SFLOAT;
  const VkFormat imgOutFormat = VK_FORMAT_D32_SFLOAT;

  const size_t imgSizeElems = imgWidth * imgHeight;
  const size_t imgSizeBytes = imgSizeElems * sizeof(float);

  const VkExtent3D imgExtent = {imgWidth, imgHeight, 1};

  VkImage vkInputImage;
  VkDeviceMemory vkInputImageMemory;
  VkImage vkOutputImage;
  VkDeviceMemory vkOutputImageMemory;

  // Initialize image input data.
  std::vector<float> inputVec(imgSizeElems, 0.f);
  for (int i = 0; i < imgSizeElems; ++i) {
    // Default Vulkan depth textures clmap values to between 0 and 1.
    inputVec[i] = float(i) / float(imgSizeElems);
  }

  // Create/allocate device images.
  {
    vkInputImage = vkutil::createImage(imgType, imgInFormat, imgExtent,
                                       VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                           VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                       1 /*mipLevels*/);
    VkMemoryRequirements memRequirements;
    auto inputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
        vkInputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements);
    vkInputImageMemory = vkutil::allocateDeviceMemory(
        imgSizeBytes, inputImageMemoryTypeIndex, vkInputImage);
    VK_CHECK_CALL(vkBindImageMemory(vk_device, vkInputImage, vkInputImageMemory,
                                    0 /*memoryOffset*/));

    vkOutputImage = vkutil::createImage(imgType, imgOutFormat, imgExtent,
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                        1 /*mipLevels*/);
    VkMemoryRequirements outputMemRequirements;
    auto outputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
        vkOutputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        outputMemRequirements);
    vkOutputImageMemory = vkutil::allocateDeviceMemory(
        imgSizeBytes, outputImageMemoryTypeIndex, vkOutputImage);
    VK_CHECK_CALL(vkBindImageMemory(vk_device, vkOutputImage,
                                    vkOutputImageMemory, 0 /*memoryOffset*/));
  }

  // Transition image layouts.
  printString("Submitting image layout transition\n");
  {
    VkImageMemoryBarrier imgInBarrier =
        vkutil::createImageMemoryBarrier(vkInputImage, 1 /*mipLevels*/);
    VkImageMemoryBarrier imgOutBarrier =
        vkutil::createImageMemoryBarrier(vkOutputImage, 1 /*mipLevels*/);

    // Update aspect mask for the images to VK_IMAGE_ASPECT_DEPTH_BIT.
    imgInBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    imgOutBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_computeCmdBuffer, &cbbi));
    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &imgInBarrier);

    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &imgOutBarrier);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_computeCmdBuffer));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_computeCmdBuffer;

    VK_CHECK_CALL(vkQueueSubmit(vk_compute_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_compute_queue));
  }

  // Allocate temporary staging buffer and copy input data to device.
  printString("Allocating staging memory and copying to device image\n");
  {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    stagingBuffer = vkutil::createBuffer(imgSizeBytes,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto inputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
        stagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingMemory =
        vkutil::allocateDeviceMemory(imgSizeBytes, inputStagingMemoryTypeIndex,
                                     nullptr /*image*/, false /*exportable*/);
    VK_CHECK_CALL(vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory,
                                     0 /*memoryOffset*/));

    // Copy host data to temporary staging buffer.
    float *inputStagingData = nullptr;
    VK_CHECK_CALL(vkMapMemory(vk_device, stagingMemory, 0 /*offset*/,
                              imgSizeBytes, 0 /*flags*/,
                              (void **)&inputStagingData));
    for (int i = 0; i < (imgSizeElems); ++i) {
      inputStagingData[i] = inputVec[i];
    }
    vkUnmapMemory(vk_device, stagingMemory);

    // Copy temporary staging buffer to device image memory.
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {imgWidth, imgHeight, 1};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &cbbi));
    vkCmdCopyBufferToImage(vk_transferCmdBuffers[0], stagingBuffer,
                           vkInputImage, VK_IMAGE_LAYOUT_GENERAL,
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
#ifdef _WIN32
  auto imgMemIn = vkutil::getMemoryWin32Handle(vkInputImageMemory);
  auto imgMemOut = vkutil::getMemoryWin32Handle(vkOutputImageMemory);
#else
  auto imgMemIn = vkutil::getMemoryOpaqueFD(vkInputImageMemory);
  auto imgMemOut = vkutil::getMemoryOpaqueFD(vkOutputImageMemory);
#endif

  // Call into SYCL to fetch from input image, and populate the output image.
  printString("Calling into SYCL with interop memory handles\n");
  runSycl(syclDevice, dims, localSize, imgMemIn, imgMemOut);

  // Copy image memory to temporary staging buffer, and back to host.
  printString("Copying image memory to host\n");
  std::vector<float> outputVec(imgSizeElems, 0.f);
  {
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;

    stagingBuffer = vkutil::createBuffer(imgSizeBytes,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto outputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
        stagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingMemory =
        vkutil::allocateDeviceMemory(imgSizeBytes, outputStagingMemoryTypeIndex,
                                     nullptr /*image*/, false /*exportable*/);
    VK_CHECK_CALL(vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory,
                                     0 /*memoryOffset*/));

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {imgWidth, imgHeight, 1};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[1], &cbbi));
    vkCmdCopyImageToBuffer(vk_transferCmdBuffers[1], vkOutputImage,
                           VK_IMAGE_LAYOUT_GENERAL, stagingBuffer,
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
    float *outputStagingData = (float *)outputVec.data();
    VK_CHECK_CALL(vkMapMemory(vk_device, stagingMemory, 0 /*offset*/,
                              imgSizeBytes, 0 /*flags*/,
                              (void **)&outputStagingData));
    for (int i = 0; i < (imgSizeElems); ++i) {
      outputVec[i] = outputStagingData[i];
    }
    vkUnmapMemory(vk_device, stagingMemory);

    // Destroy temporary staging buffer and free memory.
    vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
    vkFreeMemory(vk_device, stagingMemory, nullptr);
  }

  // Destroy images and free their memory.
  vkDestroyImage(vk_device, vkInputImage, nullptr);
  vkDestroyImage(vk_device, vkOutputImage, nullptr);
  vkFreeMemory(vk_device, vkInputImageMemory, nullptr);
  vkFreeMemory(vk_device, vkOutputImageMemory, nullptr);

  // Validate that SYCL made changes to the memory.
  bool validated = true;
  for (int i = 0; i < (imgSizeElems); ++i) {
    float expected = inputVec[i];
    // Use helper function to determine if data is accepted.
    // For floats, use default accepted error variance.
    if (!util::is_equal(outputVec[i], expected)) {
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

  auto testPassed = runTest(syclDevice, {128, 128}, {16, 16});

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  if (testPassed) {
    std::cout << "Test passed!\n";
    return EXIT_SUCCESS;
  }

  std::cerr << "Test failed\n";
  return EXIT_FAILURE;
}
