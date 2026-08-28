// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// XFAIL: windows && gpu-intel-dg2
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/21985

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} env NEOReadDebugKeys=1 UseBindlessMode=1 UseExternalAllocatorForSshAndDsh=1 %t.out

// Uncomment to print additional test information
// #define VERBOSE_PRINT
#include <iostream>

#include "../helpers/common.hpp"
#include "vulkan_setup.hpp"

#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/half_type.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// imgSizeBytes is now passed in: it must be the real (tiling-padded) import
// size, not globalSize.size()*sizeof(float).
template <typename InteropMemHandleT>
void runSycl(const sycl::device &syclDevice, sycl::range<2> globalSize,
             sycl::range<2> localSize, InteropMemHandleT extMemInHandle,
             InteropMemHandleT extMemOutHandle, size_t imgSizeBytes) {

  sycl::queue syclQueue{syclDevice};

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

bool runTest(VulkanContext &vkCtx, const sycl::device &syclDevice,
             sycl::range<2> dims, sycl::range<2> localSize) {
  const uint32_t imgWidth = static_cast<uint32_t>(dims[0]);
  const uint32_t imgHeight = static_cast<uint32_t>(dims[1]);

  const VkImageType imgType = VK_IMAGE_TYPE_2D;
  const VkFormat imgInFormat = VK_FORMAT_D32_SFLOAT;
  const VkFormat imgOutFormat = VK_FORMAT_D32_SFLOAT;

  const size_t imgSizeElems = imgWidth * imgHeight;
  const size_t imgSizeBytes = imgSizeElems * sizeof(float);

  const VkExtent3D imgExtent = {imgWidth, imgHeight, 1};

  ImageResources inputImage;
  ImageResources outputImage;

  // Real import size; set to the image memory requirement below.
  size_t importSizeBytes = imgSizeBytes;

  // Initialize image input data.
  std::vector<float> inputVec(imgSizeElems, 0.f);
  for (int i = 0; i < imgSizeElems; ++i) {
    // Default Vulkan depth textures clmap values to between 0 and 1.
    inputVec[i] = float(i) / float(imgSizeElems);
  }

  // Create/allocate device images.
  {
    // STORAGE_BIT: SYCL reads/writes this as a storage image; without it the
    // layout is transfer-only and imported reads land at the wrong offset.
    inputImage = createExportableImage(
        vkCtx, imgExtent, imgInFormat, imgType, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(vkCtx.device, inputImage.image,
                                 &memRequirements);
    // Import must describe the whole (padded) allocation the driver requires.
    importSizeBytes = std::max<size_t>(imgSizeBytes, memRequirements.size);

    // STORAGE_BIT: same as input image; the kernel writes it as a storage
    // image.
    outputImage = createExportableImage(
        vkCtx, imgExtent, imgOutFormat, imgType, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  }

  // Transition image layouts.
  std::cout << "Submitting image layout transition\n";
  {
    VkImageMemoryBarrier imgInBarrier =
        createImageMemoryBarrier(inputImage.image, 1);
    VkImageMemoryBarrier imgOutBarrier =
        createImageMemoryBarrier(outputImage.image, 1);

    // Update aspect mask for the images to VK_IMAGE_ASPECT_DEPTH_BIT.
    imgInBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    imgOutBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkCommandPool pool;
    VkCommandBuffer commandBuffer = createCommandBuffer(vkCtx, pool);
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &cbbi));
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &imgInBarrier);

    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &imgOutBarrier);
    submitCommandBuffer(vkCtx, commandBuffer, pool);
  }

  // Allocate temporary staging buffer and copy input data to device.
  std::cout << "Allocating staging memory and copying to device image\n";
  {
    auto staging = createStagingBuffer(vkCtx, imgSizeBytes,
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Copy host data to temporary staging buffer.
    float *inputStagingData = nullptr;
    VK_CHECK(vkMapMemory(vkCtx.device, staging.memory, 0 /*offset*/,
                         imgSizeBytes, 0 /*flags*/,
                         (void **)&inputStagingData));
    for (int i = 0; i < (imgSizeElems); ++i) {
      inputStagingData[i] = inputVec[i];
    }
    vkUnmapMemory(vkCtx.device, staging.memory);

    // Copy temporary staging buffer to device image memory.
    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {imgWidth, imgHeight, 1};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VkCommandPool pool;
    VkCommandBuffer commandBuffer = createCommandBuffer(vkCtx, pool);
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &cbbi));
    vkCmdCopyBufferToImage(commandBuffer, staging.buffer, inputImage.image,
                           VK_IMAGE_LAYOUT_GENERAL, 1 /*regionCount*/,
                           &copyRegion);
    submitCommandBuffer(vkCtx, commandBuffer, pool);

    // Destroy temporary staging buffer and free memory.
    cleanupBuffer(vkCtx, staging);
  }

  std::cout << "Getting memory interop handles\n";
  // Get memory interop handles.
#ifdef _WIN32
  auto imgMemIn = getMemHandle(vkCtx, inputImage.memory);
  auto imgMemOut = getMemHandle(vkCtx, outputImage.memory);
#else
  auto imgMemIn = getMemFd(vkCtx, inputImage.memory);
  auto imgMemOut = getMemFd(vkCtx, outputImage.memory);
#endif

  // Call into SYCL to fetch from input image, and populate the output image.
  std::cout << "Calling into SYCL with interop memory handles\n";
  // Pass the real import size so the SYCL import matches the Vulkan allocation.
  runSycl(syclDevice, dims, localSize, imgMemIn, imgMemOut, importSizeBytes);

  // Copy image memory to temporary staging buffer, and back to host.
  std::cout << "Copying image memory to host\n";
  std::vector<float> outputVec(imgSizeElems, 0.f);
  {
    auto staging = createStagingBuffer(vkCtx, imgSizeBytes,
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    VkCommandBufferBeginInfo cbbi = {};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkBufferImageCopy copyRegion = {};
    copyRegion.imageExtent = {imgWidth, imgHeight, 1};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    copyRegion.imageSubresource.layerCount = 1;

    VkCommandPool pool;
    VkCommandBuffer commandBuffer = createCommandBuffer(vkCtx, pool);
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &cbbi));
    vkCmdCopyImageToBuffer(commandBuffer, outputImage.image,
                           VK_IMAGE_LAYOUT_GENERAL, staging.buffer,
                           1 /*regionCount*/, &copyRegion);
    submitCommandBuffer(vkCtx, commandBuffer, pool);

    // Copy temporary staging buffer output data to host output vector.
    float *outputStagingData = (float *)outputVec.data();
    VK_CHECK(vkMapMemory(vkCtx.device, staging.memory, 0 /*offset*/,
                         imgSizeBytes, 0 /*flags*/,
                         (void **)&outputStagingData));
    for (int i = 0; i < (imgSizeElems); ++i) {
      outputVec[i] = outputStagingData[i];
    }
    vkUnmapMemory(vkCtx.device, staging.memory);

    // Destroy temporary staging buffer and free memory.
    cleanupBuffer(vkCtx, staging);
  }

  // Destroy images and free their memory.
  cleanupImageResources(vkCtx, inputImage);
  cleanupImageResources(vkCtx, outputImage);

  // Validate that SYCL made changes to the memory.
  bool validated = true;
  for (int i = 0; i < (imgSizeElems); ++i) {
    float expected = inputVec[i];
    // Use helper function to determine if data is accepted.
    // For floats, use default accepted error variance.
    if (std::abs(outputVec[i] - expected) > 0.01f) {
      std::cerr << "Result mismatch! actual[" << i << "] == " << outputVec[i]
                << " : expected == " << expected << "\n";
      validated = false;
    }
    if (!validated)
      break;
  }

  if (validated) {
    std::cout << "Results are correct!\n";
  }

  return validated;
}

int main() {
  sycl::device syclDevice;
  VulkanContext vkCtx = createVulkanContext();
  auto testPassed = runTest(vkCtx, syclDevice, {128, 128}, {16, 16});
  cleanupVulkanContext(vkCtx);

  if (testPassed) {
    std::cout << "Test passed!\n";
    return EXIT_SUCCESS;
  }

  std::cerr << "Test failed\n";
  return EXIT_FAILURE;
}
