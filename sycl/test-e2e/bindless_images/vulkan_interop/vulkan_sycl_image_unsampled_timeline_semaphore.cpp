// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}

// RUN: %{run} %t.out --type float --channels 1
// RUN: %{run} %t.out --type float --channels 2
// RUN: %{run} %t.out --type float --channels 4
// RUN: %{run} %t.out --type half --channels 1
// RUN: %{run} %t.out --type half --channels 2
// RUN: %{run} %t.out --type half --channels 4
// RUN: %{run} %t.out --type int32 --channels 1
// RUN: %{run} %t.out --type uint8 --channels 4
// RUN: %{run} %t.out --type int32 --channels 2
// RUN: %{run} %t.out --type int32 --channels 4
// RUN: %{run} %t.out --type uint32 --channels 1
// RUN: %{run} %t.out --type uint32 --channels 2
// RUN: %{run} %t.out --type uint32 --channels 4
// RUN: %{run} %t.out --type int16 --channels 1
// RUN: %{run} %t.out --type int16 --channels 2
// RUN: %{run} %t.out --type int16 --channels 4
// RUN: %{run} %t.out --type uint16 --channels 1
// RUN: %{run} %t.out --type uint16 --channels 2
// RUN: %{run} %t.out --type uint16 --channels 4
// RUN: %{run} %t.out --type uint8 --channels 1
// RUN: %{run} %t.out --type uint8 --channels 2
// RUN: %{run} %t.out --type int8 --channels 1
// RUN: %{run} %t.out --type int8 --channels 2
// RUN: %{run} %t.out --type int8 --channels 4
// RUN: %{run} %t.out --type unorm8 --channels 1
// RUN: %{run} %t.out --type unorm8 --channels 2
// RUN: %{run} %t.out --type unorm8 --channels 4

// clang-format off
/*
  Vulkan/SYCL 2D Image with Timeline Semaphore Interop 

  FLAGS
    --channels  X  Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type  XXX    Set data type (float, half, uint32, int32, uint16, int16,
  uint8, int8, unorm8). 

  // RUN: %{run} %t.out --type float
  // RUN: %{run} %t.out --type half --channels 2

*/
// clang-format on

#include "vulkan_setup.hpp"
#include <iostream>
#include <string>
#include <sycl/builtins.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/bindless_images.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

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

template <typename T> T generateValue(size_t x, int channel) {
  float val = static_cast<float>(x) / 100.0f;
  if constexpr (std::is_floating_point_v<T>)
    return static_cast<T>(val + channel * 0.1f);
  else
    return static_cast<T>((x + channel));
}

template <typename T>
int runTest(
    int channels, VkFormat fmtOverride = VK_FORMAT_UNDEFINED,
    std::optional<sycl::image_channel_type> syclOverride = std::nullopt) {

  constexpr int imageWidth = 32;
  constexpr int imageHeight = 32;
  VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;

  VkFormat vkFormat = (fmtOverride != VK_FORMAT_UNDEFINED)
                          ? fmtOverride
                          : getVulkanFormat<T>(channels);
  VulkanContext vkCtx = createVulkanContext();
  VkExtent3D extent = {(uint32_t)imageWidth, (uint32_t)imageHeight, 1};

  VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_STORAGE_BIT;
  ImageResources imgResc = createExportableImage(
      vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling, usage);

  size_t totalPixels =
      imgResc.extent.width * imgResc.extent.height * imgResc.extent.depth;

  VkDeviceSize imageSize = totalPixels * channels * sizeof(T);

  BufferResources stagingBuf = createStagingBuffer(
      vkCtx, imageSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  VkSemaphore timelineSemaphore = createExportableTimelineSemaphore(vkCtx);

  VkCommandPool pool;
  {
    VkCommandPoolCreateInfo poolInfo = {
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = vkCtx.queueFamilyIndex;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(vkCtx.device, &poolInfo, nullptr, &pool));
  }

  VkCommandBuffer cmds[2];
  {
    VkCommandBufferAllocateInfo alloc = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 2;
    VK_CHECK(vkAllocateCommandBuffers(vkCtx.device, &alloc, cmds));
  }

  VkFence fence;
  {
    VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(vkCtx.device, &fenceInfo, nullptr, &fence));
  }

  auto cleanupVulkanResourcesFn = [&]() {
    vkDestroyFence(vkCtx.device, fence, nullptr);
    vkDestroyCommandPool(vkCtx.device, pool, nullptr);
    vkDestroySemaphore(vkCtx.device, timelineSemaphore, nullptr);
    cleanupBuffer(vkCtx, stagingBuf);
    cleanupImageResources(vkCtx, imgResc);
    cleanupVulkanContext(vkCtx);
  };

  uint64_t vkSignalVal = 1;
  uint64_t syclSignalVal = 2;

  try {
    sycl::queue q;

#ifdef _WIN32
    HANDLE memHandle = getMemHandle(vkCtx, imgResc.memory);
    syclexp::external_mem_descriptor<syclexp::resource_win32_handle> extMemDesc{
        memHandle, syclexp::external_mem_handle_type::win32_nt_handle,
        imgResc.allocationSize};
#else
    int memFd = getMemFd(vkCtx, imgResc.memory);
    syclexp::external_mem_descriptor<syclexp::resource_fd> extMemDesc{
        memFd, syclexp::external_mem_handle_type::opaque_fd,
        imgResc.allocationSize};
#endif

    syclexp::external_mem extMem =
        syclexp::import_external_memory(extMemDesc, q);

    syclexp::external_semaphore syclSem{};
#ifdef _WIN32
    HANDLE semHandle = getSemaphoreHandle(vkCtx, timelineSemaphore);
    auto semDesc =
        syclexp::external_semaphore_descriptor<syclexp::resource_win32_handle>{
            semHandle,
            syclexp::external_semaphore_handle_type::timeline_win32_nt_handle};
#else
    int semFd = getSemaphoreFd(vkCtx, timelineSemaphore);
    auto semDesc = syclexp::external_semaphore_descriptor<syclexp::resource_fd>{
        semFd, syclexp::external_semaphore_handle_type::timeline_fd};
#endif
    syclSem = syclexp::import_external_semaphore(semDesc, q);

    sycl::image_channel_type syclType = syclOverride.has_value()
                                            ? syclOverride.value()
                                            : getSyclChannelType<T>();

    syclexp::image_descriptor imgDesc(sycl::range<2>(imageWidth, imageHeight),
                                      channels, syclType);

    syclexp::image_mem_handle imgMemHandle =
        syclexp::map_external_image_memory(extMem, imgDesc, q);

    syclexp::unsampled_image_handle unsampledHandle =
        syclexp::create_image(imgMemHandle, imgDesc, q);

    void *data;
    vkMapMemory(vkCtx.device, stagingBuf.memory, 0, imageSize, 0, &data);
    T *pixelData = static_cast<T *>(data);
    for (size_t i = 0; i < totalPixels; ++i) {
      for (int c = 0; c < channels; ++c) {
        auto testValue = generateValue<T>(i % imageWidth, c);
        pixelData[i * channels + c] = testValue;
      }
    }
    vkUnmapMemory(vkCtx.device, stagingBuf.memory);

    VkCommandBufferBeginInfo beginInfo{};
    VkCommandBufferBeginInfo begin = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmds[0], &begin);

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = imgResc.image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmds[0], VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = imgResc.extent;

    vkCmdCopyBufferToImage(cmds[0], stagingBuf.buffer, imgResc.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    VkImageMemoryBarrier barrier2 = barrier;
    barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier2.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier2.dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

    vkCmdPipelineBarrier(cmds[0], VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier2);

    vkEndCommandBuffer(cmds[0]);

    VkTimelineSemaphoreSubmitInfo timelineInfo = {
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    timelineInfo.signalSemaphoreValueCount = 1;
    timelineInfo.pSignalSemaphoreValues = &vkSignalVal;

    VkSubmitInfo uploadSubInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    uploadSubInfo.commandBufferCount = 1;
    uploadSubInfo.pCommandBuffers = &cmds[0];
    uploadSubInfo.pNext = &timelineInfo;
    uploadSubInfo.signalSemaphoreCount = 1;
    uploadSubInfo.pSignalSemaphores = &timelineSemaphore;

    VK_CHECK(vkQueueSubmit(vkCtx.queue, 1, &uploadSubInfo, VK_NULL_HANDLE));

    q.ext_oneapi_wait_external_semaphore(syclSem, vkSignalVal);
    q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::range<2>(imageWidth, imageHeight), [=](sycl::item<2>
                                                                      item) {
        int x = item.get_id(0);
        int y = item.get_id(1);
        bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
        using Vec4 = sycl::vec<float, 4>;
        Vec4 oldValue(0, 0, 0, 0);
        Vec4 newValue(0, 0, 0, 0);
        if (isUnorm) {
          oldValue =
              syclexp::fetch_image<Vec4>(unsampledHandle, sycl::int2(x, y));
        } else {
          auto fetchT = [&](auto &hdl) {
            Vec4 v(0, 0, 0, 0);
            if (channels == 1)
              v.x() = (float)syclexp::fetch_image<T>(hdl, sycl::int2(x, y));
            else {
              auto raw =
                  syclexp::fetch_image<sycl::vec<T, 4>>(hdl, sycl::int2(x, y));
              v.x() = (float)raw.x();
              v.y() = (float)raw.y();
              v.z() = (float)raw.z();
              v.w() = (float)raw.w();
            }
            return v;
          };
          oldValue = fetchT(unsampledHandle);
        }
        newValue = oldValue * (T)2;
        if (isUnorm) {
          newValue = sycl::clamp(newValue, 0.0f, 1.0f);
          if (channels == 1)
            syclexp::write_image(unsampledHandle, sycl::int2(x, y),
                                 newValue.x());
          else if (channels == 2)
            syclexp::write_image(unsampledHandle, sycl::int2(x, y),
                                 sycl::float2(newValue.x(), newValue.y()));
          else
            syclexp::write_image(unsampledHandle, sycl::int2(x, y), newValue);
        } else {
          if (channels == 1)
            syclexp::write_image(unsampledHandle, sycl::int2(x, y),
                                 static_cast<T>(newValue.x()));
          else if (channels == 2)
            syclexp::write_image(unsampledHandle, sycl::int2(x, y),
                                 sycl::vec<T, 2>(static_cast<T>(newValue.x()),
                                                 static_cast<T>(newValue.y())));
          else
            syclexp::write_image(unsampledHandle, sycl::int2(x, y),
                                 sycl::vec<T, 4>(static_cast<T>(newValue.x()),
                                                 static_cast<T>(newValue.y()),
                                                 static_cast<T>(newValue.z()),
                                                 static_cast<T>(newValue.w())));
        }
      });
    });
    q.ext_oneapi_signal_external_semaphore(syclSem, syclSignalVal);
    q.wait();

    vkBeginCommandBuffer(cmds[1], &begin);

    VkImageMemoryBarrier readbackBarrier{};
    readbackBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    readbackBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    readbackBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    readbackBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    readbackBarrier.image = imgResc.image;
    readbackBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    readbackBarrier.subresourceRange.baseMipLevel = 0;
    readbackBarrier.subresourceRange.levelCount = 1;
    readbackBarrier.subresourceRange.baseArrayLayer = 0;
    readbackBarrier.subresourceRange.layerCount = 1;
    readbackBarrier.srcAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmds[1], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &readbackBarrier);

    VkBufferImageCopy region2{};
    region2.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region2.imageSubresource.mipLevel = 0;
    region2.imageSubresource.baseArrayLayer = 0;
    region2.imageSubresource.layerCount = 1;
    region2.imageOffset = {0, 0, 0};
    region2.imageExtent = imgResc.extent;

    vkCmdCopyImageToBuffer(cmds[1], imgResc.image,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           stagingBuf.buffer, 1, &region2);
    vkEndCommandBuffer(cmds[1]);

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkTimelineSemaphoreSubmitInfo readBackTimelineInfo = {
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    readBackTimelineInfo.waitSemaphoreValueCount = 1;
    readBackTimelineInfo.pWaitSemaphoreValues = &syclSignalVal;

    VkSubmitInfo readBackSubInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    readBackSubInfo.commandBufferCount = 1;
    readBackSubInfo.pCommandBuffers = &cmds[1];

    readBackSubInfo.pNext = &readBackTimelineInfo;
    readBackSubInfo.waitSemaphoreCount = 1;
    readBackSubInfo.pWaitSemaphores = &timelineSemaphore;
    readBackSubInfo.pWaitDstStageMask = &waitStage;

    VK_CHECK(vkQueueSubmit(vkCtx.queue, 1, &readBackSubInfo, fence));
    VkResult fenceRes =
        vkWaitForFences(vkCtx.device, 1, &fence, VK_TRUE, 5000000000ULL);
    if (fenceRes == VK_TIMEOUT) {
      std::cerr << std::endl << "TIMEOUT on fence!  " << std::endl;
    }
    VK_CHECK(fenceRes);

    vkMapMemory(vkCtx.device, stagingBuf.memory, 0, imageSize, 0, &data);
    T *readbackPixelData = static_cast<T *>(data);
    bool passed = true;
    for (size_t i = 0; i < totalPixels; ++i) {
      for (int c = 0; c < channels; ++c) {
        T actual = readbackPixelData[i * channels + c];
        T expected = generateValue<T>(i % imageWidth, c) * (T)2;
        bool match = false;
        match = (actual == expected);
        if (!match) {
          passed = false;
          std::cout << "Mismatch at " << i << " ch:" << c
                    << " Got: " << (double)actual
                    << " Exp: " << (double)expected << std::endl;
        }
      }
    }
    vkUnmapMemory(vkCtx.device, stagingBuf.memory);

    syclexp::destroy_image_handle(unsampledHandle, q);
    syclexp::free_image_mem(imgMemHandle, syclexp::image_type::standard, q);
    syclexp::release_external_memory(extMem, q);
    syclexp::release_external_semaphore(syclSem, q);
    cleanupVulkanResourcesFn();

    if (passed)
      std::cout << "SUCCESS!" << std::endl;
    else
      std::cout << "FAILURE!" << std::endl;

    return passed ? 0 : 1;

  } catch (sycl::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    cleanupVulkanResourcesFn();
    return 1;
  }
}
int main(int argc, char **argv) {

  int width = 16;
  int height = 16;
  int channels = 4;
  bool useLinear = false;

  std::string type = "float";
  std::vector<int> dims;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--channels" && i + 1 < argc)
      channels = std::stoi(argv[++i]);
    else if (arg == "--type" && i + 1 < argc)
      type = argv[++i];
  }

  if (channels != 1 && channels != 2 && channels != 4) {
    std::cerr << "Error: Only 1, 2, or 4 channels supported." << std::endl;
    return 1;
  }

  std::cout << "Running 2D unsampled image timeline semaphore test | Type: "
            << type << " | Size: " << width << "x" << height
            << " | Channels: " << channels << std::endl;

  if (type == "float")
    return runTest<float>(channels);
  if (type == "half")
    return runTest<sycl::half>(channels);
  if (type == "int32")
    return runTest<int32_t>(channels);
  if (type == "uint32")
    return runTest<uint32_t>(channels);
  if (type == "int16")
    return runTest<int16_t>(channels);
  if (type == "uint16")
    return runTest<uint16_t>(channels);
  if (type == "uint8")
    return runTest<uint8_t>(channels);
  if (type == "int8")
    return runTest<int8_t>(channels);
  if (type == "unorm8") {
    return runTest<uint8_t>(channels, getUnorm8Format(channels),
                            sycl::image_channel_type::unorm_int8);
  }
  std::cerr << "Unknown type: " << type << std::endl;
  return 1;
}