// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan
// REQUIRES: windows

// DG2 accesses imported images as if they were uncompressed.
// XFAIL: windows && run-mode && gpu-intel-dg2
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/21985

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}

// Fills an image with Vulkan (copy or clear), reads it and writes it in place
// with a SYCL kernel, and checks the image with Vulkan. 32x33 is too small to
// be compressed, and 1366 isn't a multiple of the tile width.

// clang-format off
// RUN: %{run} %t.out --type float --channels 4 32x33
// RUN: %{run} %t.out --type unorm8 --channels 4 --clear 32x33
// RUN: %{run} %t.out --type float --channels 1 1920x1080
// RUN: %{run} %t.out --type float --channels 2 --clear 1366x768
// RUN: %{run} %t.out --type float --channels 4 1920x1080
// RUN: %{run} %t.out --type half --channels 1 1920x1080
// RUN: %{run} %t.out --type half --channels 4 --clear 3840x2160
// RUN: %{run} %t.out --type unorm8 --channels 1 1366x768
// RUN: %{run} %t.out --type unorm8 --channels 4 --clear 1920x1080
// RUN: %{run} %t.out --type snorm8 --channels 2 1920x1080
// RUN: %{run} %t.out --type snorm8 --channels 4 --clear 1366x768
// RUN: %{run} %t.out --type unorm16 --channels 2 --clear 1920x1080
// RUN: %{run} %t.out --type unorm16 --channels 4 1366x768
// RUN: %{run} %t.out --type snorm16 --channels 1 1920x1080
// RUN: %{run} %t.out --type snorm16 --channels 4 --clear 1920x1080
// RUN: %{run} %t.out --type float --channels 4 --semaphores 1920x1080
// RUN: %{run} %t.out --type unorm8 --channels 4 --clear --semaphores 1920x1080
// clang-format on

#include "../helpers/interop_read_write.hpp"
#include "sycl_vulkan_setup.hpp"

#include <sycl/properties/queue_properties.hpp>
#include <sycl/usm.hpp>

using namespace interop_read_write;

VkFormat getFormat(const std::string &type, int channels) {
  const int i = channels == 1 ? 0 : channels == 2 ? 1 : 2;
  if (type == "float")
    return std::array{VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT,
                      VK_FORMAT_R32G32B32A32_SFLOAT}[i];
  if (type == "half")
    return std::array{VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16G16_SFLOAT,
                      VK_FORMAT_R16G16B16A16_SFLOAT}[i];
  if (type == "unorm8")
    return std::array{VK_FORMAT_R8_UNORM, VK_FORMAT_R8G8_UNORM,
                      VK_FORMAT_R8G8B8A8_UNORM}[i];
  if (type == "snorm8")
    return std::array{VK_FORMAT_R8_SNORM, VK_FORMAT_R8G8_SNORM,
                      VK_FORMAT_R8G8B8A8_SNORM}[i];
  if (type == "unorm16")
    return std::array{VK_FORMAT_R16_UNORM, VK_FORMAT_R16G16_UNORM,
                      VK_FORMAT_R16G16B16A16_UNORM}[i];
  if (type == "snorm16")
    return std::array{VK_FORMAT_R16_SNORM, VK_FORMAT_R16G16_SNORM,
                      VK_FORMAT_R16G16B16A16_SNORM}[i];
  return VK_FORMAT_UNDEFINED;
}

bool isSupported(VulkanContext &ctx, VkFormat format, VkImageUsageFlags usage) {
  VkPhysicalDeviceExternalImageFormatInfo externalInfo = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
  externalInfo.handleType = PLATFORM_MEM_HANDLE_TYPE;
  VkPhysicalDeviceImageFormatInfo2 formatInfo = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &externalInfo};
  formatInfo.format = format;
  formatInfo.type = VK_IMAGE_TYPE_2D;
  formatInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  formatInfo.usage = usage;
  VkExternalImageFormatProperties externalProps = {
      VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
  VkImageFormatProperties2 props = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                                    &externalProps};
  return vkGetPhysicalDeviceImageFormatProperties2(
             ctx.physicalDevice, &formatInfo, &props) == VK_SUCCESS &&
         (externalProps.externalMemoryProperties.externalMemoryFeatures &
          VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT);
}

void beginCommandBuffer(VkCommandBuffer cmd) {
  VkCommandBufferBeginInfo beginInfo = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
}

void transitionImage(VkCommandBuffer cmd, VkImage image,
                     VkImageLayout oldLayout, VkImageLayout newLayout,
                     uint32_t srcQueueFamily = VK_QUEUE_FAMILY_IGNORED,
                     uint32_t dstQueueFamily = VK_QUEUE_FAMILY_IGNORED) {
  VkImageMemoryBarrier barrier = createImageMemoryBarrier(
      image, 1, oldLayout, newLayout, VK_ACCESS_MEMORY_WRITE_BIT,
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
  barrier.srcQueueFamilyIndex = srcQueueFamily;
  barrier.dstQueueFamilyIndex = dstQueueFamily;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);
}

void submit(VulkanContext &ctx, VkCommandBuffer cmd, VkSemaphore waitSem,
            VkSemaphore signalSem) {
  VK_CHECK(vkEndCommandBuffer(cmd));
  const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;
  if (waitSem != VK_NULL_HANDLE) {
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &waitSem;
    submitInfo.pWaitDstStageMask = &waitStage;
  }
  if (signalSem != VK_NULL_HANDLE) {
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &signalSem;
  }
  VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submitInfo, VK_NULL_HANDLE));
}

int runTest(const ChannelFormat &f, VkFormat format, int width, int height,
            bool clear, bool useSemaphores) {
  VulkanContext vkCtx = createSyclVulkanContext();
  VulkanContextGuard vkCtxGuard{vkCtx};

  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (clear)
    usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  if (!isSupported(vkCtx, format, usage)) {
    std::cout << "Format not supported, skipping" << std::endl;
    return 0;
  }

  const VkExtent3D extent = {uint32_t(width), uint32_t(height), 1};
  ImageResources imgRes = createExportableImage(
      vkCtx, extent, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage);
  const size_t bufferSize =
      size_t(width) * height * f.channels * f.channelBytes();
  BufferResources staging = createStagingBuffer(
      vkCtx, bufferSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  void *stagingData;
  VK_CHECK(vkMapMemory(vkCtx.device, staging.memory, 0, VK_WHOLE_SIZE, 0,
                       &stagingData));

  // Binary semaphores: Vulkan fill -> SYCL kernel -> Vulkan readback
  VkSemaphore fillDone = VK_NULL_HANDLE;
  VkSemaphore kernelDone = VK_NULL_HANDLE;
  if (useSemaphores) {
    fillDone = createExportableSemaphore(vkCtx);
    kernelDone = createExportableSemaphore(vkCtx);
  }

  VkCommandPool fillPool;
  VkCommandBuffer fillCmd = createCommandBuffer(vkCtx, fillPool);
  beginCommandBuffer(fillCmd);
  transitionImage(fillCmd, imgRes.image, VK_IMAGE_LAYOUT_UNDEFINED,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
  const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkBufferImageCopy region = {};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = extent;
  if (clear) {
    VkClearColorValue color;
    std::memcpy(color.float32, clearColor, sizeof(color.float32));
    vkCmdClearColorImage(fillCmd, imgRes.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1,
                         &range);
  } else {
    packPattern(f, width, height, static_cast<uint8_t *>(stagingData),
                size_t(width) * f.channels * f.channelBytes());
    vkCmdCopyBufferToImage(fillCmd, staging.buffer, imgRes.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
  }
  // Release the image to SYCL
  transitionImage(fillCmd, imgRes.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                  VK_IMAGE_LAYOUT_GENERAL, vkCtx.queueFamilyIndex,
                  VK_QUEUE_FAMILY_EXTERNAL);
  // Without semaphores, SYCL starts after the fill completed
  submit(vkCtx, fillCmd, VK_NULL_HANDLE, fillDone);
  if (!useSemaphores)
    VK_CHECK(vkQueueWaitIdle(vkCtx.queue));

  size_t errors = 0;
  try {
    // External semaphore operations require immediate command lists
    sycl::property_list queueProps =
        useSemaphores ? sycl::property_list{sycl::property::queue::in_order{},
                                            sycl::ext::intel::property::queue::
                                                immediate_command_list{}}
                      : sycl::property_list{sycl::property::queue::in_order{}};
    sycl::queue q{queueProps};

    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> memDesc{
        getMemHandle(vkCtx, imgRes.memory),
        syclexp::external_mem_handle_type::win32_nt_handle,
        imgRes.allocationSize};
    syclexp::external_mem extMem = syclexp::import_external_memory(memDesc, q);
    syclexp::image_descriptor imgDesc(sycl::range<2>(width, height), f.channels,
                                      f.channelType);
    syclexp::image_mem_handle memHandle =
        syclexp::map_external_image_memory(extMem, imgDesc, q);
    syclexp::unsampled_image_handle image =
        syclexp::create_image(memHandle, imgDesc, q);

    syclexp::external_semaphore fillDoneSem, kernelDoneSem;
    sycl::event waitEvent;
    if (useSemaphores) {
      auto importSemaphore = [&](VkSemaphore semaphore) {
        syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
            semDesc{getSemaphoreHandle(vkCtx, semaphore),
                    syclexp::external_semaphore_handle_type::win32_nt_handle};
        return syclexp::import_external_semaphore(semDesc, q);
      };
      fillDoneSem = importSemaphore(fillDone);
      kernelDoneSem = importSemaphore(kernelDone);
      waitEvent = q.ext_oneapi_wait_external_semaphore(fillDoneSem);
    }

    const size_t numValues = size_t(width) * height * f.channels;
    float *fetched = sycl::malloc_shared<float>(numValues, q);
    std::fill(fetched, fetched + numValues, -12345.f);
    sycl::event kernelEvent =
        readModifyWrite(q, f, image, width, height, fetched, waitEvent);
    if (useSemaphores)
      q.ext_oneapi_signal_external_semaphore(kernelDoneSem, kernelEvent);
    else
      q.wait_and_throw();

    VkCommandPool readbackPool;
    VkCommandBuffer readbackCmd = createCommandBuffer(vkCtx, readbackPool);
    beginCommandBuffer(readbackCmd);
    // Acquire the image from SYCL
    transitionImage(readbackCmd, imgRes.image, VK_IMAGE_LAYOUT_GENERAL,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_QUEUE_FAMILY_EXTERNAL, vkCtx.queueFamilyIndex);
    vkCmdCopyImageToBuffer(readbackCmd, imgRes.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.buffer,
                           1, &region);
    transitionImage(readbackCmd, imgRes.image,
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_IMAGE_LAYOUT_GENERAL);
    // With semaphores, the readback waits for the kernel on the device
    submit(vkCtx, readbackCmd, kernelDone, VK_NULL_HANDLE);
    VK_CHECK(vkQueueWaitIdle(vkCtx.queue));
    vkDestroyCommandPool(vkCtx.device, readbackPool, nullptr);
    q.wait_and_throw();

    const std::vector<uint32_t> written =
        unpack(f, width, height, static_cast<const uint8_t *>(stagingData),
               size_t(width) * f.channels * f.channelBytes());
    errors = checkResults(f, width, height, clear, fetched, written);

    sycl::free(fetched, q);
    syclexp::destroy_image_handle(image, q);
    syclexp::unmap_external_image_memory(memHandle,
                                         syclexp::image_type::standard, q);
    syclexp::release_external_memory(extMem, q);
    if (useSemaphores) {
      syclexp::release_external_semaphore(fillDoneSem, q);
      syclexp::release_external_semaphore(kernelDoneSem, q);
    }
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    errors = 1;
  }

  vkQueueWaitIdle(vkCtx.queue);
  vkDestroyCommandPool(vkCtx.device, fillPool, nullptr);
  if (useSemaphores) {
    vkDestroySemaphore(vkCtx.device, fillDone, nullptr);
    vkDestroySemaphore(vkCtx.device, kernelDone, nullptr);
  }
  vkUnmapMemory(vkCtx.device, staging.memory);
  cleanupBuffer(vkCtx, staging);
  cleanupImageResources(vkCtx, imgRes);

  if (errors) {
    std::cerr << "FAILURE! " << errors << " errors" << std::endl;
    return 1;
  }
  std::cout << "Test passed" << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  int width = 32;
  int height = 33;
  int channels = 4;
  bool clear = false;
  bool useSemaphores = false;
  std::string type = "float";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--clear") {
      clear = true;
    } else if (arg == "--semaphores") {
      useSemaphores = true;
    } else if (arg == "--channels" && i + 1 < argc) {
      channels = std::stoi(argv[++i]);
    } else if (arg == "--type" && i + 1 < argc) {
      type = argv[++i];
    } else if (arg.find('x') != std::string::npos) {
      size_t pos = arg.find('x');
      width = std::stoi(arg.substr(0, pos));
      height = std::stoi(arg.substr(pos + 1));
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  if (channels != 1 && channels != 2 && channels != 4) {
    std::cerr << "Error: Only 1, 2, or 4 channels supported." << std::endl;
    return 1;
  }
  std::optional<ChannelFormat> format = getChannelFormat(type, channels);
  if (!format) {
    std::cerr << "Unknown type: " << type << std::endl;
    return 1;
  }

  std::cout << "Running Vulkan 2D Read Write Test | Type: " << type
            << " | Size: " << width << "x" << height
            << " | Channels: " << channels
            << " | Fill: " << (clear ? "clear" : "copy")
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

  try {
    return runTest(*format, getFormat(type, channels), width, height, clear,
                   useSemaphores);
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }
}
