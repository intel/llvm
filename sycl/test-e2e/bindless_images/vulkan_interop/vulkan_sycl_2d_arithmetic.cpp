// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}

// UNSUPPORTED: linux
// UNSUPPORTED-TRACKER:

/*
    Run ALL the vulkan formats through the gauntlet. sampled and unsampled.
    This entire test takes less than 30 seconds on a slow machine.  MUCH faster
   (and more complete coveraage) than SFINAE based approach.

    IF a particular variant is having problems on some platform, please do NOT
   just disable the whole test, instead use   // RUN-IF: (SOMETHING) yadda-yadda
    to enable/disable that variant.

    For semaphore testing, we run just a sampling. Note, that on Linux if there
   is a failure in the first section, then likely ALL semaphore tests afterwards
   will fail. This is being tracked as a separate issue.

*/

// RUN: %t.out --type float --channels 1 32x33
// RUN: %t.out --type float --channels 2 32x33
// RUN: %t.out --type float --channels 4 32x33
// RUN: %t.out --type half --channels 1 32x33
// RUN: %t.out --type half --channels 2 32x33
// RUN: %t.out --type half --channels 4 32x33
// RUN: %t.out --type int32 --channels 1 32x33
// RUN: %t.out --type int32 --channels 2 32x33
// RUN: %t.out --type int32 --channels 4 32x33
// RUN: %t.out --type uint32 --channels 1 32x33
// RUN: %t.out --type uint32 --channels 2 32x33
// RUN: %t.out --type uint32 --channels 4 32x33
// RUN: %t.out --type int16 --channels 1 32x33
// RUN: %t.out --type int16 --channels 2 32x33
// RUN: %t.out --type int16 --channels 4 32x33
// RUN: %t.out --type uint16 --channels 1 32x33
// RUN: %t.out --type uint16 --channels 2 32x33
// RUN: %t.out --type uint16 --channels 4 32x33
// RUN: %t.out --type uint8 --channels 1 32x33
// RUN: %t.out --type uint8 --channels 2 32x33
// RUN: %t.out --type uint8 --channels 4 32x33
// RUN: %t.out --type int8 --channels 1 32x33
// RUN: %t.out --type int8 --channels 2 32x33
// RUN: %t.out --type int8 --channels 4 32x33
// RUN: %t.out --type unorm8 --channels 1 32x33
// RUN: %t.out --type unorm8 --channels 2 32x33
// RUN: %t.out --type unorm8 --channels 4 32x33
// RUN: %t.out --type float --channels 1 --sampled 32x33
// RUN: %t.out --type float --channels 2 --sampled 32x33
// RUN: %t.out --type float --channels 4 --sampled 32x33
// RUN: %t.out --type half --channels 1 --sampled 32x33
// RUN: %t.out --type half --channels 2 --sampled 32x33
// RUN: %t.out --type half --channels 4 --sampled 32x33
// RUN: %t.out --type int32 --channels 1 --sampled 32x33
// RUN: %t.out --type int32 --channels 2 --sampled 32x33
// RUN: %t.out --type int32 --channels 4 --sampled 32x33
// RUN: %t.out --type uint32 --channels 1 --sampled 32x33
// RUN: %t.out --type uint32 --channels 2 --sampled 32x33
// RUN: %t.out --type uint32 --channels 4 --sampled 32x33
// RUN: %t.out --type int16 --channels 1 --sampled 32x33
// RUN: %t.out --type int16 --channels 2 --sampled 32x33
// RUN: %t.out --type int16 --channels 4 --sampled 32x33
// RUN: %t.out --type uint16 --channels 1 --sampled 32x33
// RUN: %t.out --type uint16 --channels 2 --sampled 32x33
// RUN: %t.out --type uint16 --channels 4 --sampled 32x33
// RUN: %t.out --type uint8 --channels 1 --sampled 32x33
// RUN: %t.out --type uint8 --channels 2 --sampled 32x33
// RUN: %t.out --type uint8 --channels 4 --sampled 32x33
// RUN: %t.out --type int8 --channels 1 --sampled 32x33
// RUN: %t.out --type int8 --channels 2 --sampled 32x33
// RUN: %t.out --type int8 --channels 4 --sampled 32x33
// RUN: %t.out --type unorm8 --channels 1 --sampled 32x33
// RUN: %t.out --type unorm8 --channels 2 --sampled 32x33
// RUN: %t.out --type unorm8 --channels 4 --sampled 32x33

// RUN: %t.out --type float --channels 1 32x33 --semaphores
// RUN: %t.out --type half --channels 2 32x33 --semaphores
// RUN: %t.out --type int32 --channels 4 32x33 --semaphores
// RUN: %t.out --type uint32 --channels 1 32x33 --semaphores
// RUN: %t.out --type int16 --channels 2 32x33 --semaphores
// RUN: %t.out --type uint16 --channels 4 32x33 --semaphores
// RUN: %t.out --type uint8 --channels 1 32x33 --semaphores
// RUN: %t.out --type int8 --channels 2 32x33 --semaphores
// RUN: %t.out --type unorm8 --channels 4 32x33 --semaphores
// RUN: %t.out --type float --channels 1 --sampled 32x33 --semaphores
// RUN: %t.out --type half --channels 2 --sampled 32x33 --semaphores
// RUN: %t.out --type int32 --channels 4 --sampled 32x33 --semaphores
// RUN: %t.out --type uint32 --channels 1 --sampled 32x33 --semaphores
// RUN: %t.out --type int16 --channels 2 --sampled 32x33 --semaphores
// RUN: %t.out --type uint16 --channels 4 --sampled 32x33 --semaphores
// RUN: %t.out --type uint8 --channels 1 --sampled 32x33 --semaphores
// RUN: %t.out --type int8 --channels 2 --sampled 32x33 --semaphores
// RUN: %t.out --type unorm8 --channels 4 --sampled 32x33 --semaphores

/*
  Vulkan/SYCL 2D Arithmetic (A + B = C)

  clang++ -fsycl -o vs_2d_arith.bin vulkan_sycl_2d_arithmetic.cpp -lvulkan
  -I$VULKAN_SDK/include -L$VULKAN_SDK/lib

  clang++ -fsycl -o vs_2d_arith.exe vulkan_sycl_2d_arithmetic.cpp
  -Wno-ignored-attributes -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib

  FLAGS
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --channels  X  Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type  XXX    Set data type (float, half, uint32, int32, uint16, int16,
  uint8, int8, unorm8). Default is float WxH            Set custom Width x
  Height (e.g. 8x4)
    --sampled

  // RUN: %t.out --type float --semaphores
  // RUN: %t.out --type unorm8 --sampled --semaphores
*/
#include "vulkan_setup.hpp"

#include <algorithm>
#include <optional>
#include <string>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <sycl/ext/oneapi/bindless_images_interop.hpp>
#include <sycl/sycl.hpp>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

// ---------------------------------------------------------
// SYCL TYPE MAPPING
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
// GENERATORS
// ---------------------------------------------------------
template <typename T> T generateValueA(size_t x, size_t y, int channel) {
  float val = (float)(x + y) / 100.0f;
  if constexpr (std::is_floating_point_v<T>)
    return static_cast<T>(val + channel * 0.1f);
  else
    return static_cast<T>((x + y + channel * 10) % 64);
}

template <typename T> T generateValueB(size_t x, size_t y, int channel) {
  float val = (float)(x * 2 + y) / 100.0f;
  if constexpr (std::is_floating_point_v<T>)
    return static_cast<T>(val + channel * 0.2f);
  else
    return static_cast<T>((x * 2 + y + channel * 5) % 64);
}

template <typename T>
int runTest(
    int width, int height, int channels, bool useLinear, bool useSemaphores,
    bool useSampled, VkFormat fmtOverride = VK_FORMAT_UNDEFINED,
    std::optional<sycl::image_channel_type> syclOverride = std::nullopt) {

  // --- Setup ---
  VkImageTiling tiling =
      useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
  VkFormat vkFormat = (fmtOverride != VK_FORMAT_UNDEFINED)
                          ? fmtOverride
                          : getVulkanFormat<T>(channels);

  std::cout << "  VK Format: " << getFormatString(vkFormat) << std::endl;
  std::cout << "  Mode: " << (useSampled ? "SAMPLED Input" : "UNSAMPLED Input")
            << std::endl;

  VulkanContext vkCtx = createVulkanContext();
  VkExtent3D extent = {(uint32_t)width, (uint32_t)height, 1};

  VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                            VK_IMAGE_USAGE_STORAGE_BIT;
  if (useSampled)
    usage |= VK_IMAGE_USAGE_SAMPLED_BIT;

  ImageResources imgA = createExportableImage(vkCtx, extent, vkFormat,
                                              VK_IMAGE_TYPE_2D, tiling, usage);
  ImageResources imgB = createExportableImage(vkCtx, extent, vkFormat,
                                              VK_IMAGE_TYPE_2D, tiling, usage);
  ImageResources imgOut = createExportableImage(
      vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling, usage);

  VkSemaphore semA = VK_NULL_HANDLE;
  VkSemaphore semB = VK_NULL_HANDLE;
  if (useSemaphores) {
    semA = createExportableSemaphore(vkCtx);
    semB = createExportableSemaphore(vkCtx);
  }

  uploadImage(vkCtx, imgA, channels, semA, [&](size_t i, int c) {
    return generateValueA<T>(i % width, i / width, c);
  });
  uploadImage(vkCtx, imgB, channels, semB, [&](size_t i, int c) {
    return generateValueB<T>(i % width, i / width, c);
  });

  try {
    sycl::queue q;
#ifdef _WIN32
    auto extMemA = syclexp::import_external_memory(
        syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{
            getMemHandle(vkCtx, imgA.memory),
            syclexp::external_mem_handle_type::win32_nt_handle,
            imgA.allocationSize},
        q.get_device(), q.get_context());
    auto extMemB = syclexp::import_external_memory(
        syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{
            getMemHandle(vkCtx, imgB.memory),
            syclexp::external_mem_handle_type::win32_nt_handle,
            imgB.allocationSize},
        q.get_device(), q.get_context());
    auto extMemOut = syclexp::import_external_memory(
        syclexp::external_mem_descriptor<syclexp::resource_win32_handle>{
            getMemHandle(vkCtx, imgOut.memory),
            syclexp::external_mem_handle_type::win32_nt_handle,
            imgOut.allocationSize},
        q.get_device(), q.get_context());
#else
    auto extMemA = syclexp::import_external_memory(
        syclexp::external_mem_descriptor<syclexp::resource_fd>{
            getMemFd(vkCtx, imgA.memory),
            syclexp::external_mem_handle_type::opaque_fd, imgA.allocationSize},
        q.get_device(), q.get_context());
    auto extMemB = syclexp::import_external_memory(
        syclexp::external_mem_descriptor<syclexp::resource_fd>{
            getMemFd(vkCtx, imgB.memory),
            syclexp::external_mem_handle_type::opaque_fd, imgB.allocationSize},
        q.get_device(), q.get_context());
    auto extMemOut = syclexp::import_external_memory(
        syclexp::external_mem_descriptor<syclexp::resource_fd>{
            getMemFd(vkCtx, imgOut.memory),
            syclexp::external_mem_handle_type::opaque_fd,
            imgOut.allocationSize},
        q.get_device(), q.get_context());
#endif

    syclexp::external_semaphore extSemA, extSemB, extSemOut;
    VkSemaphore semOutVk = VK_NULL_HANDLE;
    if (useSemaphores) {
      semOutVk = createExportableSemaphore(vkCtx);
#ifdef _WIN32
      extSemA = syclexp::import_external_semaphore(
          syclexp::external_semaphore_descriptor<
              syclexp::resource_win32_handle>{
              getSemaphoreHandle(vkCtx, semA),
              syclexp::external_semaphore_handle_type::win32_nt_handle},
          q.get_device(), q.get_context());
      extSemB = syclexp::import_external_semaphore(
          syclexp::external_semaphore_descriptor<
              syclexp::resource_win32_handle>{
              getSemaphoreHandle(vkCtx, semB),
              syclexp::external_semaphore_handle_type::win32_nt_handle},
          q.get_device(), q.get_context());
      extSemOut = syclexp::import_external_semaphore(
          syclexp::external_semaphore_descriptor<
              syclexp::resource_win32_handle>{
              getSemaphoreHandle(vkCtx, semOutVk),
              syclexp::external_semaphore_handle_type::win32_nt_handle},
          q.get_device(), q.get_context());
#else
      extSemA = syclexp::import_external_semaphore(
          syclexp::external_semaphore_descriptor<syclexp::resource_fd>{
              getSemaphoreFd(vkCtx, semA),
              syclexp::external_semaphore_handle_type::opaque_fd},
          q.get_device(), q.get_context());
      extSemB = syclexp::import_external_semaphore(
          syclexp::external_semaphore_descriptor<syclexp::resource_fd>{
              getSemaphoreFd(vkCtx, semB),
              syclexp::external_semaphore_handle_type::opaque_fd},
          q.get_device(), q.get_context());
      extSemOut = syclexp::import_external_semaphore(
          syclexp::external_semaphore_descriptor<syclexp::resource_fd>{
              getSemaphoreFd(vkCtx, semOutVk),
              syclexp::external_semaphore_handle_type::opaque_fd},
          q.get_device(), q.get_context());
#endif
    }

    size_t pitchA = 0; // 0 means "compute automatically" (Tight)
    if (useLinear) {
      pitchA = getRowPitch(vkCtx, imgA.image);
      // Note: If A and B are same dims/format, pitch is likely same
    }

    sycl::image_channel_type syclType = syclOverride.has_value()
                                            ? syclOverride.value()
                                            : getSyclChannelType<T>();
    syclexp::image_descriptor imgDesc(
        sycl::range<2>(width, height), // dims
        channels,                      // num_channels
        syclType,                      // channel_type
        syclexp::image_type::standard, // type (default)
        1 //,                             // num_levels (default)
          // 1,                             // array_size (default)
        // 0,                             // num_samples (default)
        // pitchA                         // pitch
    );

    auto imgMemA = syclexp::map_external_image_memory(
        extMemA, imgDesc, q.get_device(), q.get_context());
    auto imgMemB = syclexp::map_external_image_memory(
        extMemB, imgDesc, q.get_device(), q.get_context());
    auto imgMemOut = syclexp::map_external_image_memory(
        extMemOut, imgDesc, q.get_device(), q.get_context());
    auto handleOut = syclexp::create_image(imgMemOut, imgDesc, q.get_device(),
                                           q.get_context());

    // ------------------------------------------------------------
    // SEPARATE WAIT SUBMISSIONS
    // ------------------------------------------------------------
    std::vector<sycl::event> waitEvents;
    if (useSemaphores) {
      // Wait 1
      waitEvents.push_back(q.submit([&](sycl::handler &h) {
        h.ext_oneapi_wait_external_semaphore(extSemA);
      }));
      // Wait 2
      waitEvents.push_back(q.submit([&](sycl::handler &h) {
        h.ext_oneapi_wait_external_semaphore(extSemB);
      }));
    }

    sycl::event kernelEvent;

    if (useSampled) {
      // --------------------------------------------------------
      // PATH A: SAMPLED SUBMISSION
      // --------------------------------------------------------
      syclexp::bindless_image_sampler sampler(
          sycl::addressing_mode::clamp_to_edge,
          sycl::coordinate_normalization_mode::unnormalized,
          sycl::filtering_mode::nearest);
      auto handleA = syclexp::create_image(imgMemA, sampler, imgDesc,
                                           q.get_device(), q.get_context());
      auto handleB = syclexp::create_image(imgMemB, sampler, imgDesc,
                                           q.get_device(), q.get_context());

      kernelEvent = q.submit([&](sycl::handler &h) {
        h.depends_on(waitEvents);
        h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
          int x = item.get_id(0);
          int y = item.get_id(1);

          bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
          using Vec4 = sycl::vec<float, 4>;
          Vec4 valA(0, 0, 0, 0);
          Vec4 valB(0, 0, 0, 0);

          if (isUnorm) {
            valA = syclexp::sample_image<Vec4>(
                handleA, sycl::float2(x + 0.5f, y + 0.5f));
            valB = syclexp::sample_image<Vec4>(
                handleB, sycl::float2(x + 0.5f, y + 0.5f));
          } else {
            using SampleT = sycl::vec<T, 4>;
            auto rawA = syclexp::sample_image<SampleT>(
                handleA, sycl::float2(x + 0.5f, y + 0.5f));
            auto rawB = syclexp::sample_image<SampleT>(
                handleB, sycl::float2(x + 0.5f, y + 0.5f));
            valA = {(float)rawA.x(), (float)rawA.y(), (float)rawA.z(),
                    (float)rawA.w()};
            valB = {(float)rawB.x(), (float)rawB.y(), (float)rawB.z(),
                    (float)rawB.w()};
          }

          // Sum & Write
          Vec4 sum = valA + valB;
          if (isUnorm) {
            sum = sycl::clamp(sum, 0.0f, 1.0f);
            if (channels == 1)
              syclexp::write_image(handleOut, sycl::int2(x, y), sum.x());
            else if (channels == 2)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::float2(sum.x(), sum.y()));
            else
              syclexp::write_image(handleOut, sycl::int2(x, y), sum);
          } else {
            if (channels == 1)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   static_cast<T>(sum.x()));
            else if (channels == 2)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::vec<T, 2>(static_cast<T>(sum.x()),
                                                   static_cast<T>(sum.y())));
            else
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::vec<T, 4>(static_cast<T>(sum.x()),
                                                   static_cast<T>(sum.y()),
                                                   static_cast<T>(sum.z()),
                                                   static_cast<T>(sum.w())));
          }
        });
      });

      syclexp::destroy_image_handle(handleA, q.get_device(), q.get_context());
      syclexp::destroy_image_handle(handleB, q.get_device(), q.get_context());

    } else {
      // --------------------------------------------------------
      // PATH B: UNSAMPLED SUBMISSION
      // --------------------------------------------------------
      auto handleA = syclexp::create_image(imgMemA, imgDesc, q.get_device(),
                                           q.get_context());
      auto handleB = syclexp::create_image(imgMemB, imgDesc, q.get_device(),
                                           q.get_context());

      kernelEvent = q.submit([&](sycl::handler &h) {
        h.depends_on(waitEvents);
        h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
          int x = item.get_id(0);
          int y = item.get_id(1);

          bool isUnorm = (syclType == sycl::image_channel_type::unorm_int8);
          using Vec4 = sycl::vec<float, 4>;
          Vec4 valA(0, 0, 0, 0);
          Vec4 valB(0, 0, 0, 0);

          if (isUnorm) {
            valA = syclexp::fetch_image<Vec4>(handleA, sycl::int2(x, y));
            valB = syclexp::fetch_image<Vec4>(handleB, sycl::int2(x, y));
          } else {
            auto fetchT = [&](auto &hdl) {
              Vec4 v(0, 0, 0, 0);
              if (channels == 1)
                v.x() = (float)syclexp::fetch_image<T>(hdl, sycl::int2(x, y));
              else {
                auto raw = syclexp::fetch_image<sycl::vec<T, 4>>(
                    hdl, sycl::int2(x, y));
                v.x() = (float)raw.x();
                v.y() = (float)raw.y();
                v.z() = (float)raw.z();
                v.w() = (float)raw.w();
              }
              return v;
            };
            valA = fetchT(handleA);
            valB = fetchT(handleB);
          }

          // Sum & Write
          Vec4 sum = valA + valB;
          if (isUnorm) {
            sum = sycl::clamp(sum, 0.0f, 1.0f);
            if (channels == 1)
              syclexp::write_image(handleOut, sycl::int2(x, y), sum.x());
            else if (channels == 2)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::float2(sum.x(), sum.y()));
            else
              syclexp::write_image(handleOut, sycl::int2(x, y), sum);
          } else {
            if (channels == 1)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   static_cast<T>(sum.x()));
            else if (channels == 2)
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::vec<T, 2>(static_cast<T>(sum.x()),
                                                   static_cast<T>(sum.y())));
            else
              syclexp::write_image(handleOut, sycl::int2(x, y),
                                   sycl::vec<T, 4>(static_cast<T>(sum.x()),
                                                   static_cast<T>(sum.y()),
                                                   static_cast<T>(sum.z()),
                                                   static_cast<T>(sum.w())));
          }
        });
      });

      syclexp::destroy_image_handle(handleA, q.get_device(), q.get_context());
      syclexp::destroy_image_handle(handleB, q.get_device(), q.get_context());
    }

    // C. Submit Signal (Separate Action)
    if (useSemaphores) {
      q.submit([&](sycl::handler &h) {
        h.depends_on(kernelEvent);
        h.ext_oneapi_signal_external_semaphore(extSemOut);
      });
    }
    q.wait();

    // 5. Cleanup Shared Resources
    syclexp::destroy_image_handle(handleOut, q.get_device(), q.get_context());
    syclexp::free_image_mem(imgMemA, syclexp::image_type::standard,
                            q.get_device(), q.get_context());
    syclexp::free_image_mem(imgMemB, syclexp::image_type::standard,
                            q.get_device(), q.get_context());
    syclexp::free_image_mem(imgMemOut, syclexp::image_type::standard,
                            q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemA, q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemB, q.get_device(), q.get_context());
    syclexp::release_external_memory(extMemOut, q.get_device(),
                                     q.get_context());

    if (useSemaphores) {
      syclexp::release_external_semaphore(extSemA, q.get_device(),
                                          q.get_context());
      syclexp::release_external_semaphore(extSemB, q.get_device(),
                                          q.get_context());
      syclexp::release_external_semaphore(extSemOut, q.get_device(),
                                          q.get_context());
    }

    // 6. Verify (Vulkan)
    bool passed =
        verifyImage(vkCtx, imgOut, channels, semOutVk, [&](size_t i, int c) {
          size_t x = i % width;
          size_t y = i / width;
          T a = generateValueA<T>(x, y, c);
          T b = generateValueB<T>(x, y, c);

          if (syclOverride.has_value() &&
              syclOverride.value() == sycl::image_channel_type::unorm_int8) {
            float fa = (float)a / 255.0f;
            float fb = (float)b / 255.0f;
            // USE PARENTHESES TO PREVENT MACRO EXPANSION
            float sum = (std::min)(fa + fb, 1.0f);
            return static_cast<T>(sum * 255.0f + 0.5f);
          } else {
            return static_cast<T>(a + b);
          }
        });

    if (passed)
      std::cout << "SUCCESS!" << std::endl;
    else
      std::cout << "FAILURE!" << std::endl;

    if (useSemaphores) {
      vkDestroySemaphore(vkCtx.device, semA, nullptr);
      vkDestroySemaphore(vkCtx.device, semB, nullptr);
      vkDestroySemaphore(vkCtx.device, semOutVk, nullptr);
    }

    cleanupImageResources(vkCtx, imgA);
    cleanupImageResources(vkCtx, imgB);
    cleanupVulkan(vkCtx, imgOut);

    return passed ? 0 : 1;

  } catch (std::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;

    // Clean up Vulkan resources even on exception
    if (useSemaphores) {
      if (semA)
        vkDestroySemaphore(vkCtx.device, semA, nullptr);
      if (semB)
        vkDestroySemaphore(vkCtx.device, semB, nullptr);
    }
    cleanupImageResources(vkCtx, imgA);
    cleanupImageResources(vkCtx, imgB);
    cleanupVulkan(vkCtx, imgOut);
    return 1;
  }
}

int main(int argc, char **argv) {
  int width = 4;
  int height = 4;
  int channels = 4;
  bool useLinear = false;
  bool useSemaphores = false;
  bool useSampled = false;
  std::string type = "float";

  std::vector<int> dims;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--semaphores")
      useSemaphores = true;
    else if (arg == "--linear")
      useLinear = true;
    else if (arg == "--channels" && i + 1 < argc)
      channels = std::stoi(argv[++i]);
    else if (arg == "--type" && i + 1 < argc)
      type = argv[++i];
    else if (arg == "--sampled")
      useSampled = true;
    else if (arg.find("x") != std::string::npos) {
      size_t xPos = arg.find("x");
      try {
        width = std::stoi(arg.substr(0, xPos));
        height = std::stoi(arg.substr(xPos + 1));
      } catch (...) {
      }
    }
  }

  // Apply space separated args
  if (dims.size() >= 1)
    width = dims[0];
  if (dims.size() >= 2)
    height = dims[1];
  else if (dims.size() == 1)
    height = width;

  if (channels != 1 && channels != 2 && channels != 4) {
    std::cerr << "Error: Only 1, 2, or 4 channels supported." << std::endl;
    return 1;
  }

  std::cout << "Running 2D ARITHMETIC Test (C = A + B) | Type: " << type
            << " | Size: " << width << "x" << height
            << " | Channels: " << channels
            << " | Tiling: " << (useLinear ? "LINEAR" : "OPTIMAL")
            << " | Sampled: " << (useSampled ? "YES" : "NO")
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

  // Dispatcher
  if (type == "float")
    return runTest<float>(width, height, channels, useLinear, useSemaphores,
                          useSampled);
  if (type == "half")
    return runTest<sycl::half>(width, height, channels, useLinear,
                               useSemaphores, useSampled);
  if (type == "int32")
    return runTest<int32_t>(width, height, channels, useLinear, useSemaphores,
                            useSampled);
  if (type == "uint32")
    return runTest<uint32_t>(width, height, channels, useLinear, useSemaphores,
                             useSampled);
  if (type == "int16")
    return runTest<int16_t>(width, height, channels, useLinear, useSemaphores,
                            useSampled);
  if (type == "uint16")
    return runTest<uint16_t>(width, height, channels, useLinear, useSemaphores,
                             useSampled);
  if (type == "uint8")
    return runTest<uint8_t>(width, height, channels, useLinear, useSemaphores,
                            useSampled);
  if (type == "int8")
    return runTest<int8_t>(width, height, channels, useLinear, useSemaphores,
                           useSampled);

  if (type == "unorm8") {
    return runTest<uint8_t>(width, height, channels, useLinear, useSemaphores,
                            useSampled, getUnorm8Format(channels),
                            sycl::image_channel_type::unorm_int8);
  }

  std::cerr << "Unknown type: " << type << std::endl;
  return 1;
}