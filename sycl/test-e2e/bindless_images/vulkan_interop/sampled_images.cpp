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
bool run_sycl(int input_image_fd, size_t width, size_t height) {
  sycl::device dev;
  sycl::queue q(dev);
  auto ctxt = q.get_context();

  // Image descriptor - mapped to Vulkan image layout
  sycl::ext::oneapi::experimental::image_descriptor desc(
      {width, height}, sycl::image_channel_order::r,
      sycl::image_channel_type::fp32,
      sycl::ext::oneapi::experimental::image_type::interop, 1 /*num_levels*/);

  sycl::ext::oneapi::experimental::bindless_image_sampler samp(
      sycl::addressing_mode::repeat,
      sycl::coordinate_normalization_mode::normalized,
      sycl::filtering_mode::linear);

  const size_t img_size = width * height * sizeof(float);

  // Extension: external memory descriptor
  sycl::ext::oneapi::experimental::external_mem_descriptor<
      sycl::ext::oneapi::experimental::external_mem_fd>
      input_ext_mem_desc{input_image_fd, img_size};

  // Extension: interop mem handle imported from file descriptor
  sycl::ext::oneapi::experimental::interop_mem_handle input_interop_mem_handle =
      sycl::ext::oneapi::experimental::import_external_memory(
          input_ext_mem_desc, q);

  // Extension: interop mem handle imported from file descriptor
  sycl::ext::oneapi::experimental::image_mem_handle input_mapped_mem_handle =
      sycl::ext::oneapi::experimental::map_external_memory_array(
          input_interop_mem_handle, desc, q);

  // Extension: create the image and return the handle
  sycl::ext::oneapi::experimental::sampled_image_handle img_input =
      sycl::ext::oneapi::experimental::create_image(input_mapped_mem_handle,
                                                    samp, desc, q);

  std::vector<float> out(width * height);

  try {
    sycl::buffer<float, 2> buf((float *)out.data(),
                               sycl::range<2>{height, width});
    q.submit([&](sycl::handler &cgh) {
      auto outAcc = buf.get_access<sycl::access_mode::write>(
          cgh, sycl::range<2>{height, width});
      cgh.parallel_for<class image_interop>(
          sycl::nd_range<2>{{width, height}, {width, height}},
          [=](sycl::nd_item<2> it) {
            size_t dim0 = it.get_local_id(0);
            size_t dim1 = it.get_local_id(1);

            // Normalize coordinates -- +0.5 to look towards centre of pixel
            float fdim0 = float(dim0 + 0.5) / (float)width;
            float fdim1 = float(dim1 + 0.5) / (float)height;

            // Extension: read image data from handle (Vulkan imported)
            float pixel = sycl::ext::oneapi::experimental::read_image<float>(
                img_input, sycl::float2(fdim0, fdim1));

            pixel *= 10.f;
            outAcc[sycl::id<2>{dim1, dim0}] = pixel;
          });
    });
  } catch (...) {
    std::cerr << "Kernel submission failed!" << std::endl;
    exit(-1);
  }

  try {
    sycl::ext::oneapi::experimental::destroy_image_handle(img_input, q);
    sycl::ext::oneapi::experimental::release_external_memory(
        input_interop_mem_handle, q);
  } catch (...) {
    std::cerr << "Destroying interop memory failed!\n";
  }

  printString("Validating\n");
  bool validated = true;
  for (int i = 0; i < width * height; i++) {
    bool mismatch = false;
    float expected = (float)(i)*10.f;
    if (out[i] != expected) {
      mismatch = true;
      validated = false;
    }

    if (mismatch) {
#ifdef VERBOSE_PRINT
      std::cout << "Result mismatch! Expected: " << expected
                << ", Actual: " << out[i] << "\n";
#else
      break;
#endif
    }
  }
  if (validated) {
    std::cout << "Test passed!\n";
    return true;
  }
  std::cout << "Test failed!\n";
  return false;
}

// Returns true if validated correctly
bool run_test() {
  const uint32_t width = 16, height = 16;
  const size_t imageSizeBytes = width * height * sizeof(float);

  printString("Creating input image\n");
  // Create input image memory
  auto inputImage = vkutil::createImage(
      VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT, {width, height, 1},
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
          VK_IMAGE_USAGE_STORAGE_BIT);
  auto inputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
      inputImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  auto inputMemory =
      vkutil::allocateDeviceMemory(imageSizeBytes, inputImageMemoryTypeIndex);
  VK_CHECK_CALL(vkBindImageMemory(vk_device, inputImage, inputMemory,
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

  printString("Populating staging buffer\n");
  // Populate staging memory
  float *inputStagingData = nullptr;
  VK_CHECK_CALL(vkMapMemory(vk_device, inputStagingMemory, 0 /*offset*/,
                            imageSizeBytes, 0 /*flags*/,
                            (void **)&inputStagingData));
  for (int i = 0; i < width * height; ++i) {
    inputStagingData[i] = (float)i;
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

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_computeCmdBuffer, &cbbi));
    vkCmdPipelineBarrier(vk_computeCmdBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrierInput);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_computeCmdBuffer));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_computeCmdBuffer;

    VK_CHECK_CALL(vkQueueSubmit(vk_compute_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_compute_queue));
  }

  printString("Copying staging memory to images\n");
  // Copy staging to main image memory
  {
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {width, height, 1};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VK_CHECK_CALL(vkBeginCommandBuffer(vk_transferCmdBuffers[0], &cbbi));
    vkCmdCopyBufferToImage(vk_transferCmdBuffers[0], inputStagingBuffer,
                           inputImage, VK_IMAGE_LAYOUT_GENERAL,
                           1 /*regionCount*/, &copyRegion);
    VK_CHECK_CALL(vkEndCommandBuffer(vk_transferCmdBuffers[0]));

    VkSubmitInfo submission = {};
    submission.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submission.commandBufferCount = 1;
    submission.pCommandBuffers = &vk_transferCmdBuffers[0];

    VK_CHECK_CALL(vkQueueSubmit(vk_transfer_queue, 1 /*submitCount*/,
                                &submission, VK_NULL_HANDLE /*fence*/));
    VK_CHECK_CALL(vkQueueWaitIdle(vk_transfer_queue));
  }

  printString("Getting memory file descriptors and calling into SYCL\n");
  // Pass memory to SYCL for modification
  auto input_fd = vkutil::getMemoryOpaqueFD(inputMemory);
  bool validated = run_sycl(input_fd, width, height);

  // Cleanup
  vkDestroyBuffer(vk_device, inputStagingBuffer, nullptr);
  vkDestroyImage(vk_device, inputImage, nullptr);
  vkFreeMemory(vk_device, inputStagingMemory, nullptr);
  vkFreeMemory(vk_device, inputMemory, nullptr);

  return validated;
}

int main() {

  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupDevice("NVIDIA") != VK_SUCCESS) {
    std::cerr << "Device setup failed!\n";
    return EXIT_FAILURE;
  }

  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Compute pipeline setup failed!\n";
    return EXIT_FAILURE;
  }

  bool validated = run_test();

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }

  return validated ? EXIT_SUCCESS : EXIT_FAILURE;
}
