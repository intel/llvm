
// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// UNSUPPORTED: linux || windows
// UNSUPPORTED-TRACKER: VSRI-6896

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}

// clang-format off
/*
  3D UnSampled Image Write

  clang++ -fsycl  -o vsw_3d_test.bin vulkan_sycl_image_interop_write_3d_unsampled.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib


  clang++ -fsycl  -o vsw_3d_test.exe vulkan_sycl_image_interop_write_3d_unsampled.cpp -Wno-ignored-attributes -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib

  FLAGS
    --sampled      ERROR: Sampled image writes are not supported
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --channels  X  Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type  XXX    Set data type (float, half, uint32, int32, uint16, int16, uint8, int8, unorm8). 
                   Default is float
    WxHxD          Set custom Width x Height x Depth (e.g. 8x4x2)

    ./vsw_3d_test.bin
    ./vsw_3d_test.bin --semaphores --linear --channels 2 128x128x16
 */
// clang-format on

#include "vulkan_setup.hpp"

#include <optional>
#include <string>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <sycl/image.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

// ---------------------------------------------------------
// SYCL TYPE MAPPING HELPERS
// ---------------------------------------------------------
template <typename T> sycl::image_channel_type getSyclChannelType();
template <> inline sycl::image_channel_type getSyclChannelType<float>() {
  return sycl::image_channel_type::fp32;
}
template <> inline sycl::image_channel_type getSyclChannelType<int32_t>() {
  return sycl::image_channel_type::signed_int32;
}
template <> inline sycl::image_channel_type getSyclChannelType<uint32_t>() {
  return sycl::image_channel_type::unsigned_int32;
}
template <> inline sycl::image_channel_type getSyclChannelType<int16_t>() {
  return sycl::image_channel_type::signed_int16;
}
template <> inline sycl::image_channel_type getSyclChannelType<uint16_t>() {
  return sycl::image_channel_type::unsigned_int16;
}
template <> inline sycl::image_channel_type getSyclChannelType<uint8_t>() {
  return sycl::image_channel_type::unsigned_int8;
}
template <> inline sycl::image_channel_type getSyclChannelType<int8_t>() {
  return sycl::image_channel_type::signed_int8;
}

template <> inline VkFormat getVulkanFormat<sycl::half>(int channels) {
  switch (channels) {
  case 1:
    return VK_FORMAT_R16_SFLOAT;
  case 2:
    return VK_FORMAT_R16G16_SFLOAT;
  case 4:
    return VK_FORMAT_R16G16B16A16_SFLOAT;
  default:
    throw std::runtime_error("Unsupported channels for half");
  }
}
template <> inline sycl::image_channel_type getSyclChannelType<sycl::half>() {
  return sycl::image_channel_type::fp16;
}

// ---------------------------------------------------------
// TEST RUNNER
// ---------------------------------------------------------
template <typename T>
int runTest(
    int width, int height, int depth, int channels, bool useLinear,
    bool useSemaphores, VkFormat fmtOverride = VK_FORMAT_UNDEFINED,
    std::optional<sycl::image_channel_type> syclOverride = std::nullopt) {

  VkImageTiling tiling =
      useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
  VkFormat vkFormat = (fmtOverride != VK_FORMAT_UNDEFINED)
                          ? fmtOverride
                          : getVulkanFormat<T>(channels);
  std::cout << "VK Format: " << getFormatString(vkFormat) << std::endl;

  VulkanContext vkCtx = createVulkanContext();
  VkExtent3D extent = {(uint32_t)width, (uint32_t)height, (uint32_t)depth};
  ImageResources imgRes =
      createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_3D, tiling);

  // Initial Transition
  {
    VkCommandPoolCreateInfo poolInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
    VkCommandPool pool;
    vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool);
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo ca = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    vkAllocateCommandBuffers(vkCtx.device, &ca, &cmd);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bi);
    VkImageMemoryBarrier bar = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bar.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    bar.image = imgRes.image;
    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    bar.srcAccessMask = 0;
    bar.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &bar);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(vkCtx.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vkCtx.queue);
    vkDestroyCommandPool(vkCtx.device, pool, nullptr);
  }

  VkSemaphore vkSem = VK_NULL_HANDLE;
  if (useSemaphores)
    vkSem = createExportableSemaphore(vkCtx);

  // Initial clear/upload (Sanity Check)
  if (!uploadAndVerify<T>(vkCtx, imgRes, vkSem, channels)) {
    std::cerr << "Vulkan Upload Failed!" << std::endl;
    return 1;
  }

  try {
    sycl::queue q;

    // IMPORT MEMORY
#ifdef _WIN32
    HANDLE memHandle = getMemHandle(vkCtx, imgRes.memory);
    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> extMemDesc{
        memHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        imgRes.allocationSize};
#else
    int memFd = getMemFd(vkCtx, imgRes.memory);
    syclexp::external_mem_descriptor<syclexp::resource_fd> extMemDesc{
        memFd, syclexp::external_mem_handle_type::opaque_fd,
        imgRes.allocationSize};
#endif
    syclexp::external_mem extMem = syclexp::import_external_memory(
        extMemDesc, q.get_device(), q.get_context());

    // IMPORT SEMAPHORE
    syclexp::external_semaphore extSem;
    if (useSemaphores) {
#ifdef _WIN32
      HANDLE semHandle = getSemaphoreHandle(vkCtx, vkSem);
      syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>
          extSemDesc{semHandle,
                     syclexp::external_semaphore_handle_type::win32_nt_handle};
#else
      int semFd = getSemaphoreFd(vkCtx, vkSem);
      syclexp::external_semaphore_descriptor<syclexp::resource_fd> extSemDesc{
          semFd, syclexp::external_semaphore_handle_type::opaque_fd};
#endif
      extSem = syclexp::import_external_semaphore(extSemDesc, q.get_device(),
                                                  q.get_context());
    }

    sycl::image_channel_type syclType = syclOverride.has_value()
                                            ? syclOverride.value()
                                            : getSyclChannelType<T>();
    // bindless image ranges use (x,y,z) order,
    // differening from SYCL 2020 "fastest incrementing" convention.
    syclexp::image_descriptor imgDesc(sycl::range<3>(width, height, depth),
                                      channels, syclType);
    syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(
        extMem, imgDesc, q.get_device(), q.get_context());
    syclexp::unsampled_image_handle unsampledHandle = syclexp::create_image(
        devHandle, imgDesc, q.get_device(), q.get_context());

    sycl::event kernelEvent = q.submit([&](sycl::handler &h) {
      h.parallel_for(
          // ranges for parallel_for use "fastest incrementing" order (z,y,x),
          // but bindless images ranges use (x,y,z) order.
          sycl::range<3>(depth, height, width), [=](sycl::item<3> item) {
            int x = item.get_id(2);
            int y = item.get_id(1);
            int z = item.get_id(0);
            size_t index = z * width * height + y * width + x;
            size_t totalPixels = width * height * depth;

            // unorm is special snowflake
            bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
            if (isUnorm) {
              if (channels == 1) {
                float v =
                    (float)generateTestValue<T>(index, 0, totalPixels) / 255.0f;
                syclexp::write_image(unsampledHandle, sycl::int3(x, y, z), v);
              } else if (channels == 2) {
                float v1 =
                    (float)generateTestValue<T>(index, 0, totalPixels) / 255.0f;
                float v2 =
                    (float)generateTestValue<T>(index, 1, totalPixels) / 255.0f;
                syclexp::write_image(unsampledHandle, sycl::int3(x, y, z),
                                     sycl::float2(v1, v2));
              } else { // 4
                float v1 =
                    (float)generateTestValue<T>(index, 0, totalPixels) / 255.0f;
                float v2 =
                    (float)generateTestValue<T>(index, 1, totalPixels) / 255.0f;
                float v3 =
                    (float)generateTestValue<T>(index, 2, totalPixels) / 255.0f;
                float v4 =
                    (float)generateTestValue<T>(index, 3, totalPixels) / 255.0f;
                syclexp::write_image(unsampledHandle, sycl::int3(x, y, z),
                                     sycl::float4(v1, v2, v3, v4));
              }
              return;
            }

            if (channels == 1) {
              T val = generateTestValue<T>(index, 0, totalPixels);
              syclexp::write_image(unsampledHandle, sycl::int3(x, y, z), val);
            } else if (channels == 2) {
              using Vec2 = sycl::vec<T, 2>;
              Vec2 px(generateTestValue<T>(index, 0, totalPixels),
                      generateTestValue<T>(index, 1, totalPixels));
              syclexp::write_image(unsampledHandle, sycl::int3(x, y, z), px);
            } else {
              using Vec4 = sycl::vec<T, 4>;
              Vec4 px(generateTestValue<T>(index, 0, totalPixels),
                      generateTestValue<T>(index, 1, totalPixels),
                      generateTestValue<T>(index, 2, totalPixels),
                      generateTestValue<T>(index, 3, totalPixels));
              syclexp::write_image(unsampledHandle, sycl::int3(x, y, z), px);
            }
          });
    });

    if (useSemaphores)
      q.submit([&](sycl::handler &h) {
        h.depends_on(kernelEvent);
        h.ext_oneapi_signal_external_semaphore(extSem);
      });
    q.wait();

    syclexp::destroy_image_handle(unsampledHandle, q.get_device(),
                                  q.get_context());
    syclexp::release_external_memory(extMem, q.get_device(), q.get_context());
    if (useSemaphores) {
      syclexp::release_external_semaphore(extSem, q.get_device(),
                                          q.get_context());
    }
  } catch (std::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    return 1;
  }

  // Vulkan Verify
  vkDeviceWaitIdle(vkCtx.device);
  VkBuffer verifyBuffer;
  VkDeviceMemory verifyMem;
  size_t dataSize = width * height * depth * channels * sizeof(T);
  VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = dataSize;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  vkCreateBuffer(vkCtx.device, &bi, nullptr, &verifyBuffer);
  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(vkCtx.device, verifyBuffer, &req);
  VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = findMemoryType(vkCtx.physicalDevice, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  vkAllocateMemory(vkCtx.device, &ai, nullptr, &verifyMem);
  vkBindBufferMemory(vkCtx.device, verifyBuffer, verifyMem, 0);

  {
    VkCommandPoolCreateInfo poolInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
    VkCommandPool pool;
    vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool);
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo ca = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    vkAllocateCommandBuffers(vkCtx.device, &ca, &cmd);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferImageCopy reg = {};
    reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    reg.imageExtent = extent;
    vkCmdCopyImageToBuffer(cmd, imgRes.image, VK_IMAGE_LAYOUT_GENERAL,
                           verifyBuffer, 1, &reg);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    std::vector<VkPipelineStageFlags> waitStages = {
        VK_PIPELINE_STAGE_TRANSFER_BIT};
    if (useSemaphores) {
      si.waitSemaphoreCount = 1;
      si.pWaitSemaphores = &vkSem;
      si.pWaitDstStageMask = waitStages.data();
    }
    vkQueueSubmit(vkCtx.queue, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vkCtx.queue);
    vkDestroyCommandPool(vkCtx.device, pool, nullptr);
  }

  void *ptr;
  vkMapMemory(vkCtx.device, verifyMem, 0, dataSize, 0, &ptr);
  T *vData = (T *)ptr;
  bool passed = true;
  int errorCount = 0;
  size_t totalPixels = width * height * depth;
  for (size_t i = 0; i < totalPixels * channels; ++i) {
    T expected = generateTestValue<T>(i / channels, i % channels, totalPixels);
    if (!checkValue(vData[i], expected)) {
      passed = false;
      if (errorCount++ < 5)
        std::cout << "Mismatch at " << i << " Got: " << (double)vData[i]
                  << " Exp: " << (double)expected << std::endl;
    }
  }
  vkUnmapMemory(vkCtx.device, verifyMem);
  if (passed)
    std::cout << "SUCCESS!" << std::endl;
  else
    std::cout << "FAILURE! (" << errorCount << " errors)" << std::endl;

  vkDestroyBuffer(vkCtx.device, verifyBuffer, nullptr);
  vkFreeMemory(vkCtx.device, verifyMem, nullptr);
  if (useSemaphores)
    vkDestroySemaphore(vkCtx.device, vkSem, nullptr);
  cleanupVulkan(vkCtx, imgRes);
  return passed ? 0 : 1;
}

int main(int argc, char **argv) {
  int width = 4, height = 4, depth = 4, channels = 4;
  bool useLinear = false, useSemaphores = false;
  std::string type = "float";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--sampled") {
      std::cerr
          << "ERROR: --sampled flag is not supported for write operations."
          << std::endl;
      std::cerr << "Sampled images cannot be written to. Only unsampled "
                   "(storage) images support write_image()."
                << std::endl;
      return 1;
    } else if (arg == "--semaphores")
      useSemaphores = true;
    else if (arg == "--linear")
      useLinear = true;
    else if (arg == "--channels" && i + 1 < argc)
      channels = std::stoi(argv[++i]);
    else if (arg == "--type" && i + 1 < argc)
      type = argv[++i];
    else if (arg.find("x") != std::string::npos) {
      size_t x1 = arg.find("x");
      size_t x2 = arg.find("x", x1 + 1);
      try {
        width = std::stoi(arg.substr(0, x1));
        if (x2 != std::string::npos) {
          height = std::stoi(arg.substr(x1 + 1, x2 - x1 - 1));
          depth = std::stoi(arg.substr(x2 + 1));
        } else {
          height = std::stoi(arg.substr(x1 + 1));
        }
      } catch (...) {
      }
    }
  }

  std::cout << "Running UNSAMPLED 3D Write Test | Type: " << type
            << " | Size: " << width << "x" << height << "x" << depth
            << " | Channels: " << channels
            << " | Tiling: " << (useLinear ? "LINEAR" : "OPTIMAL")
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

  if (type == "float")
    return runTest<float>(width, height, depth, channels, useLinear,
                          useSemaphores);
  if (type == "half")
    return runTest<sycl::half>(width, height, depth, channels, useLinear,
                               useSemaphores);
  if (type == "int32")
    return runTest<int32_t>(width, height, depth, channels, useLinear,
                            useSemaphores);
  if (type == "uint32")
    return runTest<uint32_t>(width, height, depth, channels, useLinear,
                             useSemaphores);
  if (type == "int16")
    return runTest<int16_t>(width, height, depth, channels, useLinear,
                            useSemaphores);
  if (type == "uint16")
    return runTest<uint16_t>(width, height, depth, channels, useLinear,
                             useSemaphores);
  if (type == "uint8")
    return runTest<uint8_t>(width, height, depth, channels, useLinear,
                            useSemaphores);
  if (type == "int8")
    return runTest<int8_t>(width, height, depth, channels, useLinear,
                           useSemaphores);
  if (type == "unorm8")
    return runTest<uint8_t>(width, height, depth, channels, useLinear,
                            useSemaphores, getUnorm8Format(channels),
                            sycl::image_channel_type::unorm_int8);

  std::cerr << "Unknown type: " << type << std::endl;
  return 1;
}