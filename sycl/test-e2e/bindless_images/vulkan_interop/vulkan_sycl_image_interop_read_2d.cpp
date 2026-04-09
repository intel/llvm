// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import || (windows && level_zero && aspect-ext_oneapi_bindless_images)
// REQUIRES: vulkan

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}


/*
    Run ALL the vulkan formats through the gauntlet. sampled and unsampled.
    This entire test takes less than 30 seconds on a slow machine.  MUCH faster
   (and more complete coveraage) than SFINAE based approach.

    IF a particular variant is having problems on some platform, please do NOT
   just disable the whole test, instead use   RUN~IF (SOMETHING) yadda-yadda
    to enable/disable that variant.

    For semaphore testing, we run just a sampling. Note, that on Linux if there
   is a failure in the first section, then likely ALL semaphore tests afterwards
   will fail. This is being tracked as a separate issue.

*/

// RUN: %{run} %t.out --type float --channels 1 32x33
// RUN: %{run} %t.out --type float --channels 2 32x33
// RUN: %{run} %t.out --type float --channels 4 32x33
// RUN: %{run} %t.out --type half --channels 1 32x33
// RUN: %{run} %t.out --type half --channels 2 32x33
// RUN: %{run} %t.out --type half --channels 4 32x33
// RUN: %{run} %t.out --type int32 --channels 1 32x33
// RUN: %{run} %t.out --type int32 --channels 2 32x33
// RUN: %{run} %t.out --type int32 --channels 4 32x33
// RUN: %{run} %t.out --type uint32 --channels 1 32x33
// RUN: %{run} %t.out --type uint32 --channels 2 32x33
// RUN: %{run} %t.out --type uint32 --channels 4 32x33
// RUN: %{run} %t.out --type int16 --channels 1 32x33
// RUN: %{run} %t.out --type int16 --channels 2 32x33
// RUN: %{run} %t.out --type int16 --channels 4 32x33
// RUN: %{run} %t.out --type uint16 --channels 1 32x33
// RUN: %{run} %t.out --type uint16 --channels 2 32x33
// RUN: %{run} %t.out --type uint16 --channels 4 32x33
// RUN: %{run} %t.out --type uint8 --channels 1 32x33
// RUN: %{run} %t.out --type uint8 --channels 2 32x33
// RUN: %{run} %t.out --type uint8 --channels 4 32x33
// RUN: %{run} %t.out --type int8 --channels 1 32x33
// RUN: %{run} %t.out --type int8 --channels 2 32x33
// RUN: %{run} %t.out --type int8 --channels 4 32x33
// RUN: %{run} %t.out --type unorm8 --channels 1 32x33
// RUN: %{run} %t.out --type unorm8 --channels 2 32x33
// RUN: %{run} %t.out --type unorm8 --channels 4 32x33
// RUN: %{run} %t.out --type float --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type float --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type float --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type half --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type half --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type half --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type int32 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type int32 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type int32 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type uint32 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type uint32 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type uint32 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type int16 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type int16 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type int16 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type uint16 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type uint16 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type uint16 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type uint8 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type uint8 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type uint8 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type int8 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type int8 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type int8 --channels 4 --sampled 32x33
// RUN: %{run} %t.out --type unorm8 --channels 1 --sampled 32x33
// RUN: %{run} %t.out --type unorm8 --channels 2 --sampled 32x33
// RUN: %{run} %t.out --type unorm8 --channels 4 --sampled 32x33

// RUN: %{run} %t.out --type float --channels 1 32x33 --semaphores
// RUN: %{run} %t.out --type float --channels 4 32x33 --semaphores
// RUN: %{run} %t.out --type half --channels 1 32x33 --semaphores
// RUN: %{run} %t.out --type uint32 --channels 4 32x33 --semaphores
// RUN: %{run} %t.out --type uint16 --channels 2 32x33 --semaphores
// RUN: %{run} %t.out --type int8 --channels 1 32x33 --semaphores
// RUN: %{run} %t.out --type float --channels 4 --sampled 32x33 --semaphores
// RUN: %{run} %t.out --type int32 --channels 4 --sampled 32x33 --semaphores
// RUN: %{run} %t.out --type int16 --channels 4 --sampled 32x33 --semaphores
// RUN: %{run} %t.out --type uint8 --channels 2 --sampled 32x33 --semaphores
// RUN: %{run} %t.out --type unorm8 --channels 4 --sampled 32x33 --semaphores

// clang-format off
/*
  Vulkan/SYCL 2D Image Read Test (Sampled + Unsampled)

  clang++ -fsycl  -o vsr_2d_test.bin vulkan_sycl_image_interop_read_2d.cpp -lvulkan -I$VULKAN_SDK/include -L$VULKAN_SDK/lib

  clang++ -fsycl  -o vsr_2d_test.exe vulkan_sycl_image_interop_read_2d.cpp -Wno-ignored-attributes  -lvulkan-1 -I$VULKAN_SDK/Include -L$VULKAN_SDK/Lib

  USAGE:
    ./vsr_2d_test.bin [FLAGS] [WxH]

  FLAGS:
    --sampled      Use sampled image path (default: unsampled/storage)
    --semaphores   Use Vulkan Semaphores for SYCL Interop Sync
    --linear       Use LINEAR tiling for the Vulkan Image (default is OPTIMAL)
    --channels X   Set number of channels (1, 2, or 4). Default is 4 (RGBA)
    --type XXX     Set data type (float, half, uint32, int32, uint16, int16, uint8, int8, unorm8). 
                   Default is float 
    WxH            Set custom Width x Height (e.g. 8x4)

  EXAMPLES:
    ./vsr_2d_test.bin
    ./vsr_2d_test.bin --sampled --semaphores
    ./vsr_2d_test.bin --sampled --linear --channels 2 8x4
    ./vsr_2d_test.bin --type half --channels 2
    ./vsr_2d_test.bin --linear --type unorm8 16x16

  NOTE: --linear is not currently working with 2D images on Linux
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

// ---------------------------------------------------------
// SYCL TYPE MAPPING HELPERS
// ---------------------------------------------------------

template <typename T> sycl::image_channel_type getSyclChannelType();

template <> inline sycl::image_channel_type getSyclChannelType<float>() {
  return sycl::image_channel_type::fp32;
}

template <> inline sycl::image_channel_type getSyclChannelType<sycl::half>() {
  return sycl::image_channel_type::fp16;
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

// ---------------------------------------------------------
// SYCL CHANNEL ORDER (for unsampled)
// ---------------------------------------------------------

inline sycl::image_channel_order getSyclChannelOrder(int channels) {
  switch (channels) {
  case 1:
    return sycl::image_channel_order::r;
  case 2:
    return sycl::image_channel_order::rg;
  case 4:
    return sycl::image_channel_order::rgba;
  default:
    throw std::runtime_error("Unsupported channel count for SYCL Order");
  }
}

// ---------------------------------------------------------
// VULKAN FORMAT MAPPING
// ---------------------------------------------------------

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

// ---------------------------------------------------------
// TEMPLATED RUNNER
// ---------------------------------------------------------
template <typename T>
int runTest(
    int width, int height, int channels, bool useLinear, bool useSemaphores,
    bool useSampled, VkFormat fmtOverride = VK_FORMAT_UNDEFINED,
    std::optional<sycl::image_channel_type> syclOverride = std::nullopt) {

  VkImageTiling tiling =
      useLinear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;
  VkFormat vkFormat = (fmtOverride != VK_FORMAT_UNDEFINED)
                          ? fmtOverride
                          : getVulkanFormat<T>(channels);

  std::cout << "VK Format: " << getFormatString(vkFormat) << std::endl;

  // Setup Vulkan
  VulkanContext vkCtx = createVulkanContext();
  VkExtent3D extent = {(uint32_t)width, (uint32_t)height, 1};
  ImageResources imgRes =
      createExportableImage(vkCtx, extent, vkFormat, VK_IMAGE_TYPE_2D, tiling);

  // Semaphores
  VkSemaphore vkSem = VK_NULL_HANDLE;
  if (useSemaphores)
    vkSem = createExportableSemaphore(vkCtx);

  // Upload test data
  if (!uploadAndVerify<T>(vkCtx, imgRes, vkSem, channels)) {
    std::cerr << "Vulkan Upload Failed!" << std::endl;
    return 1;
  }

  // SYCL Import and Verification
  namespace syclexp = sycl::ext::oneapi::experimental;
  try {
    sycl::queue q;

    // Import Memory (Platform Specific)
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

    // Import Semaphore (Platform Specific)
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

    // Create Image Descriptor
    sycl::image_channel_type syclType = syclOverride.has_value()
                                            ? syclOverride.value()
                                            : getSyclChannelType<T>();

    syclexp::image_descriptor imgDesc(sycl::range<2>(width, height), channels,
                                      syclType);

    // Map external memory
    syclexp::image_mem_handle devHandle = syclexp::map_external_image_memory(
        extMem, imgDesc, q.get_device(), q.get_context());

    // Branch: Sampled vs Unsampled
    syclexp::sampled_image_handle sampledHandle;
    syclexp::unsampled_image_handle unsampledHandle;

    if (useSampled) {
      // Sampler: Nearest required for Integer types
      syclexp::bindless_image_sampler sampler(
          sycl::addressing_mode::clamp_to_edge,
          sycl::coordinate_normalization_mode::unnormalized,
          sycl::filtering_mode::nearest);
      sampledHandle = syclexp::create_image(devHandle, sampler, imgDesc,
                                            q.get_device(), q.get_context());
    } else {
      // Unsampled image
      unsampledHandle = syclexp::create_image(devHandle, imgDesc,
                                              q.get_device(), q.get_context());
    }

    // Output Buffer
    size_t totalValues = width * height * channels;
    sycl::buffer<T, 1> checkBuf(totalValues);

    // Wait for Vulkan semaphore if needed
    sycl::event dependencyEvent;
    if (useSemaphores) {
      dependencyEvent = q.submit([&](sycl::handler &h) {
        h.ext_oneapi_wait_external_semaphore(extSem);
      });
    }

    // Kernel: Read image data
    q.submit([&](sycl::handler &h) {
       if (useSemaphores)
         h.depends_on(dependencyEvent);
       sycl::accessor outAcc(checkBuf, h, sycl::write_only);

       h.parallel_for(sycl::range<2>(width, height), [=](sycl::item<2> item) {
         int x = item.get_id(0);
         int y = item.get_id(1);

         if (useSampled) {
           // Sampled path: use sample_image with float coordinates
           float coordX = (float)x + 0.5f;
           float coordY = (float)y + 0.5f;

           if constexpr (std::is_floating_point_v<T> ||
                         std::is_same_v<T, sycl::half>) {
             // Float types: sample as Vec4
             sycl::float4 pixel = syclexp::sample_image<sycl::float4>(
                 sampledHandle, sycl::float2(coordX, coordY));

             size_t base = (y * width + x) * channels;
             if (channels >= 1)
               outAcc[base + 0] = static_cast<T>(pixel.x());
             if (channels >= 2)
               outAcc[base + 1] = static_cast<T>(pixel.y());
             if (channels >= 4) {
               outAcc[base + 2] = static_cast<T>(pixel.z());
               outAcc[base + 3] = static_cast<T>(pixel.w());
             }
           } else {
             // Integer types: sample as int4/uint4
             if constexpr (std::is_signed_v<T>) {
               sycl::int4 pixel = syclexp::sample_image<sycl::int4>(
                   sampledHandle, sycl::float2(coordX, coordY));

               size_t base = (y * width + x) * channels;
               if (channels >= 1)
                 outAcc[base + 0] = static_cast<T>(pixel.x());
               if (channels >= 2)
                 outAcc[base + 1] = static_cast<T>(pixel.y());
               if (channels >= 4) {
                 outAcc[base + 2] = static_cast<T>(pixel.z());
                 outAcc[base + 3] = static_cast<T>(pixel.w());
               }
             } else {
               sycl::uint4 pixel = syclexp::sample_image<sycl::uint4>(
                   sampledHandle, sycl::float2(coordX, coordY));

               size_t base = (y * width + x) * channels;
               if (channels >= 1)
                 outAcc[base + 0] = static_cast<T>(pixel.x());
               if (channels >= 2)
                 outAcc[base + 1] = static_cast<T>(pixel.y());
               if (channels >= 4) {
                 outAcc[base + 2] = static_cast<T>(pixel.z());
                 outAcc[base + 3] = static_cast<T>(pixel.w());
               }
             }
           }
         } else {
           // Unsampled path: use fetch_image with int coordinates
           sycl::int2 coords(x, y);
           size_t base = (y * width + x) * channels;

           if (channels == 1) {
             T pixel = syclexp::fetch_image<T>(unsampledHandle, coords);
             outAcc[base] = pixel;
           } else if (channels == 2) {
             sycl::vec<T, 2> pixel =
                 syclexp::fetch_image<sycl::vec<T, 2>>(unsampledHandle, coords);
             outAcc[base + 0] = pixel.x();
             outAcc[base + 1] = pixel.y();
           } else if (channels == 4) {
             sycl::vec<T, 4> pixel =
                 syclexp::fetch_image<sycl::vec<T, 4>>(unsampledHandle, coords);
             outAcc[base + 0] = pixel.x();
             outAcc[base + 1] = pixel.y();
             outAcc[base + 2] = pixel.z();
             outAcc[base + 3] = pixel.w();
           }
         }
       });
     }).wait();

    std::cout << "SYCL Kernel Executed." << std::endl;

    // Verify results
    sycl::host_accessor hostAcc(checkBuf, sycl::read_only);
    bool passed = true;
    int errorCount = 0;
    size_t totalPixels = width * height;

    for (size_t i = 0; i < totalValues; ++i) {
      size_t pixelIdx = i / channels;
      int channelIdx = i % channels;

      T expected = generateTestValue<T>(pixelIdx, channelIdx, totalPixels);

      if (!checkValue(hostAcc[i], expected)) {
        if (errorCount < 10) {
          std::cerr << "Mismatch at index " << i << " (pixel " << pixelIdx
                    << ", channel " << channelIdx << "): "
                    << "Expected " << expected << ", Got " << hostAcc[i]
                    << std::endl;
        }
        errorCount++;
        passed = false;
      }
    }

    if (passed) {
      std::cout << "SUCCESS! All " << totalValues << " values match."
                << std::endl;
    } else {
      std::cout << "FAILURE! " << errorCount << " errors out of " << totalValues
                << " values." << std::endl;
    }

    // Cleanup SYCL resources
    if (useSampled) {
      syclexp::destroy_image_handle(sampledHandle, q.get_device(),
                                    q.get_context());
    } else {
      syclexp::destroy_image_handle(unsampledHandle, q.get_device(),
                                    q.get_context());
    }

    syclexp::release_external_memory(extMem, q.get_device(), q.get_context());

    if (useSemaphores) {
      syclexp::release_external_semaphore(extSem, q.get_device(),
                                          q.get_context());
      vkDestroySemaphore(vkCtx.device, vkSem, nullptr);
    }

    cleanupVulkan(vkCtx, imgRes);
    return passed ? 0 : 1;

  } catch (std::exception &e) {
    std::cerr << "SYCL Exception: " << e.what() << std::endl;
    cleanupVulkan(vkCtx, imgRes);
    return 1;
  }
}

// ---------------------------------------------------------
// MAIN
// ---------------------------------------------------------
int main(int argc, char **argv) {
  int width = 4;
  int height = 4;
  int channels = 4;
  bool useLinear = false;
  bool useSemaphores = false;
  bool useSampled = false; // Default: unsampled
  std::string type = "float";

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--sampled") {
      useSampled = true;
    } else if (arg == "--semaphores") {
      useSemaphores = true;
    } else if (arg == "--linear") {
      useLinear = true;
    } else if (arg == "--channels" && i + 1 < argc) {
      channels = std::stoi(argv[++i]);
    } else if (arg == "--type" && i + 1 < argc) {
      type = argv[++i];
    } else if (arg.find('x') != std::string::npos) {
      // Parse WxH
      size_t pos = arg.find('x');
      width = std::stoi(arg.substr(0, pos));
      height = std::stoi(arg.substr(pos + 1));
    }
  }

  if (channels != 1 && channels != 2 && channels != 4) {
    std::cerr << "Error: Only 1, 2, or 4 channels supported." << std::endl;
    return 1;
  }

  std::cout << "Running " << (useSampled ? "SAMPLED" : "UNSAMPLED")
            << " 2D Read Test | Type: " << type << " | Size: " << width << "x"
            << height << " | Channels: " << channels
            << " | Tiling: " << (useLinear ? "LINEAR" : "OPTIMAL")
            << " | Semaphores: " << (useSemaphores ? "ON" : "OFF") << std::endl;

  // Dispatch to appropriate type
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