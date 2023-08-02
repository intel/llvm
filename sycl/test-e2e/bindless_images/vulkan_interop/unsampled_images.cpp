// REQUIRES: linux
// REQUIRES: cuda
// REQUIRES: vulkan

// RUN: %clangxx -fsycl -fsycl-targets=%{sycl_triple} %link-vulkan %s -o %t.out
// RUN: %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT

#include <sycl/sycl.hpp>

#include "vulkan_common.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

// Returns true if validated correctly
void run_sycl(int input_image_fd, int output_image_fd,
              int sycl_wait_semaphore_fd, int sycl_done_semaphore_fd,
              size_t width, size_t height) {
  try {
    sycl::device dev;
    sycl::queue q(dev);
    auto ctxt = q.get_context();

    // Image descriptor - mapped to Vulkan image layout
    sycl::ext::oneapi::experimental::image_descriptor desc(
        {width, height}, sycl::image_channel_order::rgba,
        sycl::image_channel_type::unsigned_int32,
        sycl::ext::oneapi::experimental::image_type::interop, 1 /*num_levels*/);

    const size_t img_size = width * height * sizeof(sycl::uint4);

    std::vector<sycl::uint4> out(width * height);

    // Extension: external memory descriptor
    sycl::ext::oneapi::experimental::external_mem_descriptor<
        sycl::ext::oneapi::experimental::external_mem_fd>
        inputExtMemDesc{input_image_fd, img_size};
    sycl::ext::oneapi::experimental::external_mem_descriptor<
        sycl::ext::oneapi::experimental::external_mem_fd>
        outputExtMemDesc{output_image_fd, img_size};

    sycl::ext::oneapi::experimental::interop_mem_handle
        input_interop_mem_handle =
            sycl::ext::oneapi::experimental::import_external_memory(
                inputExtMemDesc, dev, ctxt);

    sycl::ext::oneapi::experimental::interop_mem_handle
        output_interop_mem_handle =
            sycl::ext::oneapi::experimental::import_external_memory(
                outputExtMemDesc, dev, ctxt);

    sycl::ext::oneapi::experimental::image_mem_handle input_mapped_mem_handle =
        sycl::ext::oneapi::experimental::map_external_memory_array(
            input_interop_mem_handle, desc, dev, ctxt);
    sycl::ext::oneapi::experimental::image_mem_handle output_mapped_mem_handle =
        sycl::ext::oneapi::experimental::map_external_memory_array(
            output_interop_mem_handle, desc, dev, ctxt);

    // Extension: create the image and return the handle
    sycl::ext::oneapi::experimental::unsampled_image_handle img_input =
        sycl::ext::oneapi::experimental::create_image(input_mapped_mem_handle,
                                                      desc, dev, ctxt);
    sycl::ext::oneapi::experimental::unsampled_image_handle img_output =
        sycl::ext::oneapi::experimental::create_image(output_mapped_mem_handle,
                                                      desc, dev, ctxt);

    // Extension: import semaphores
    sycl::ext::oneapi::experimental::external_semaphore_descriptor<
        sycl::ext::oneapi::experimental::external_semaphore_fd>
        sycl_wait_external_semaphore_desc{sycl_wait_semaphore_fd};

    sycl::ext::oneapi::experimental::external_semaphore_descriptor<
        sycl::ext::oneapi::experimental::external_semaphore_fd>
        sycl_done_external_semaphore_desc{sycl_done_semaphore_fd};

    sycl::ext::oneapi::experimental::interop_semaphore_handle
        sycl_wait_interop_semaphore_handle =
            sycl::ext::oneapi::experimental::import_external_semaphore(
                sycl_wait_external_semaphore_desc, dev, ctxt);

    sycl::ext::oneapi::experimental::interop_semaphore_handle
        sycl_done_interop_semaphore_handle =
            sycl::ext::oneapi::experimental::import_external_semaphore(
                sycl_done_external_semaphore_desc, dev, ctxt);

    // Extension: wait for imported semaphore
    q.ext_oneapi_wait_external_semaphore(sycl_wait_interop_semaphore_handle);

    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for<class image_interop>(
          sycl::nd_range<2>{{width, height}, {32, 32}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_global_id(0);
            size_t dim1 = it.get_global_id(1);

            // Extension: read image data from handle (Vulkan imported)
            sycl::uint4 pixel =
                sycl::ext::oneapi::experimental::read_image<sycl::uint4>(
                    img_input, sycl::int2(dim0, dim1));

            pixel *= 10;

            // Extension: write image data using handle (Vulkan imported)
            sycl::ext::oneapi::experimental::write_image(
                img_output, sycl::int2(dim0, dim1), pixel);
          });
    });

    // Extension: signal imported semaphore
    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_signal_external_semaphore(
          sycl_done_interop_semaphore_handle);
    });

    // Wait for kernel completion before destroying external objects
    q.wait_and_throw();

    sycl::ext::oneapi::experimental::release_external_memory(
        input_interop_mem_handle, dev, ctxt);
    sycl::ext::oneapi::experimental::release_external_memory(
        output_interop_mem_handle, dev, ctxt);
    sycl::ext::oneapi::experimental::destroy_external_semaphore(
        sycl_wait_interop_semaphore_handle, dev, ctxt);
    sycl::ext::oneapi::experimental::destroy_external_semaphore(
        sycl_done_interop_semaphore_handle, dev, ctxt);
    sycl::ext::oneapi::experimental::destroy_image_handle(img_input, dev, ctxt);
    sycl::ext::oneapi::experimental::destroy_image_handle(img_output, dev,
                                                          ctxt);
  } catch (sycl::exception e) {
    std::cerr << "SYCL exception caught! : " << e.what() << "\n";
    exit(-1);
  } catch (...) {
    std::cerr << "Unknown exception caught!\n";
    exit(-1);
  }
}

// Returns true if validated correctly
bool run_test() {
  const uint32_t width = 1024 * 4, height = 1024 * 4;
  const size_t imageSizeBytes = width * height * sizeof(sycl::uint4);

  printString("Creating input image\n");
  // Create input image memory
  auto inputImage = vkutil::createImage(
      VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, {width, height, 1},
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
          VK_IMAGE_USAGE_STORAGE_BIT);
  auto inputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
      inputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  auto inputMemory =
      vkutil::allocateDeviceMemory(imageSizeBytes, inputImageMemoryTypeIndex);
  VK_CHECK_CALL(vkBindImageMemory(vk_device, inputImage, inputMemory,
                                  0 /*memoryOffset*/));

  printString("Creating output image\n");
  // Create output image memory
  auto outputImage = vkutil::createImage(
      VK_IMAGE_TYPE_2D, VK_FORMAT_R32G32B32A32_UINT, {width, height, 1},
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
          VK_IMAGE_USAGE_STORAGE_BIT);
  auto outputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
      outputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  auto outputMemory =
      vkutil::allocateDeviceMemory(imageSizeBytes, outputImageMemoryTypeIndex);
  VK_CHECK_CALL(vkBindImageMemory(vk_device, outputImage, outputMemory,
                                  0 /*memoryOffset*/));

  printString("Creating staging buffers\n");
  // Create input staging memory
  auto inputStagingBuffer = vkutil::createBuffer(
      imageSizeBytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto inputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
      inputStagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto inputStagingMemory = vkutil::allocateDeviceMemory(
      imageSizeBytes, inputStagingMemoryTypeIndex, false /*exportable*/);
  VK_CHECK_CALL(vkBindBufferMemory(vk_device, inputStagingBuffer,
                                   inputStagingMemory, 0 /*memoryOffset*/));

  // Create output staging memory
  auto outputStagingBuffer = vkutil::createBuffer(
      imageSizeBytes,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto outputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
      outputStagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  auto outputStagingMemory = vkutil::allocateDeviceMemory(
      imageSizeBytes, outputStagingMemoryTypeIndex, false /*exportable*/);
  VK_CHECK_CALL(vkBindBufferMemory(vk_device, outputStagingBuffer,
                                   outputStagingMemory, 0 /*memoryOffset*/));

  printString("Populating staging buffer\n");
  // Populate staging memory
  sycl::vec<uint32_t, 4> *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, inputStagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int i = 0; i < width * height; ++i) {
    inputStagingData[i] =
        sycl::vec<uint32_t, 4>{4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3};
  }
  vkUnmapMemory(vk_device, inputStagingMemory);

  printString("Submitting image layout transition\n");
  // Transition image layouts
  {
    VkImageMemoryBarrier barrierInput = {};
    barrierInput.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrierInput.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrierInput.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrierInput.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierInput.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierInput.image = inputImage;
    barrierInput.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrierInput.subresourceRange.levelCount = 1;
    barrierInput.subresourceRange.layerCount = 1;
    barrierInput.srcAccessMask = 0;
    barrierInput.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    VkImageMemoryBarrier barrierOutput = {};
    barrierOutput.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrierOutput.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrierOutput.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrierOutput.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierOutput.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierOutput.image = outputImage;
    barrierOutput.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrierOutput.subresourceRange.levelCount = 1;
    barrierOutput.subresourceRange.layerCount = 1;
    barrierOutput.srcAccessMask = 0;
    barrierOutput.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_computeCmdBuffer, &cbbi));
    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierInput);
    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierOutput);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_computeCmdBuffer));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_computeCmdBuffer;

    VK_CHECK_CALL(vkQueueSubmit(vk_compute_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_compute_queue));
  }

  // Create semaphore to later import in SYCL
  printString("Creating semaphores\n");
  VkSemaphore syclWaitSemaphore;
  {
    VkExportSemaphoreCreateInfo esci = {};
    esci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &esci;
    VK_CHECK_CALL(
        vkCreateSemaphore(vk_device, &sci, nullptr, &syclWaitSemaphore));
  }

  VkSemaphore syclDoneSemaphore;
  {
    VkExportSemaphoreCreateInfo esci = {};
    esci.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &esci;
    VK_CHECK_CALL(
        vkCreateSemaphore(vk_device, &sci, nullptr, &syclDoneSemaphore));
  }

  printString("Copying staging memory to images\n");
  // Copy staging to main image memory
  {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {width, height, 1};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &cbbi));
    vkCmdCopyBufferToImage(vk_transferCmdBuffers[0], inputStagingBuffer,
                           inputImage, VK_IMAGE_LAYOUT_GENERAL,
                           1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[0]));

    std::vector<VkPipelineStageFlags> stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[0];

    submission.signalSemaphoreCount = 1;
    submission.pSignalSemaphores = &syclWaitSemaphore;
    submission.pWaitDstStageMask = stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
  }

  printString("Getting memory file descriptors and calling into SYCL\n");
  // Pass memory to SYCL for modification
  int input_fd = vkutil::getMemoryOpaqueFD(inputMemory);
  int output_fd = vkutil::getMemoryOpaqueFD(outputMemory);

  // Pass semaphores to SYCL for synchronization
  int sycl_wait_semaphore_fd = vkutil::getSemaphoreOpaqueFD(syclWaitSemaphore);
  int sycl_done_semaphore_fd = vkutil::getSemaphoreOpaqueFD(syclDoneSemaphore);

  run_sycl(input_fd, output_fd, sycl_wait_semaphore_fd, sycl_done_semaphore_fd,
           width, height);

  printString("Copying image memory to staging memory\n");
  // Copy main image memory to staging
  {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {width, height, 1};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[1], &cbbi));
    vkCmdCopyImageToBuffer(vk_transferCmdBuffers[1], outputImage,
                           VK_IMAGE_LAYOUT_GENERAL, outputStagingBuffer,
                           1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[1]));

    std::vector<VkPipelineStageFlags> stages{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[1];

    submission.waitSemaphoreCount = 1;
    submission.pWaitSemaphores = &syclDoneSemaphore;
    submission.pWaitDstStageMask = stages.data();

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));
  }

  printString("Validating\n");
  // Validate that SYCL made changes to the memory
  bool validated = true;
  sycl::vec<uint32_t, 4> *outputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, outputStagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&outputStagingData));
  for (int i = 0; i < width * height; ++i) {
    sycl::vec<uint32_t, 4> expected = {4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3};
    expected *= 10;
    for (int j = 0; j < 4; ++j) {
      if (outputStagingData[i][j] != expected[j]) {
#ifdef VERBOSE_PRINT
        std::cout << "Result mismatch! actual[" << i << "][" << j
                  << "] == " << outputStagingData[i][j]
                  << " : expected == " << expected[j] << "\n";
        validated = false;
#else
        break;
#endif
      }
    }
    if (!validated)
      break;
  }
  vkUnmapMemory(vk_device, outputStagingMemory);

  if (validated) {
    std::cout << "Test passed!\n";
  } else {
    std::cout << "Test failed!\n";
  }

  // Cleanup
  vkDestroyBuffer(vk_device, inputStagingBuffer, nullptr);
  vkDestroyBuffer(vk_device, outputStagingBuffer, nullptr);
  vkDestroyImage(vk_device, inputImage, nullptr);
  vkDestroyImage(vk_device, outputImage, nullptr);
  vkFreeMemory(vk_device, inputStagingMemory, nullptr);
  vkFreeMemory(vk_device, outputStagingMemory, nullptr);
  vkFreeMemory(vk_device, inputMemory, nullptr);
  vkFreeMemory(vk_device, outputMemory, nullptr);
  vkDestroySemaphore(vk_device, syclWaitSemaphore, nullptr);
  vkDestroySemaphore(vk_device, syclDoneSemaphore, nullptr);

  return validated;
}

int main() {

  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }

  // Currently only Nvidia devices are tested
  if (vkutil::setupDevice("NVIDIA") != VK_SUCCESS) {
    std::cerr << "Device setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Command buffers setup failed!\n";
    return EXIT_FAILURE;
  }

  bool validated = run_test();

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  return validated ? EXIT_SUCCESS : EXIT_FAILURE;
}
