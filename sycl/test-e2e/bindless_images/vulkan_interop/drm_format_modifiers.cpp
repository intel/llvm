// REQUIRES: aspect-ext_oneapi_bindless_images
// REQUIRES: aspect-ext_oneapi_external_memory_import
// REQUIRES: vulkan, linux

// RUN: %{build} %link-vulkan -o %t.out %if target-spir %{ -Wno-ignored-attributes %}
// RUN: %{run} %t.out

#include "../helpers/common.hpp"
#include "vulkan_common.hpp"

#include <sycl/ext/oneapi/bindless_images.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
#include <optional>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;
using drm_format_modifier_list = std::vector<uint64_t>;

drm_format_modifier_list getDrmFormatModifiersVulkan(VkFormat format) {
  // vkGetPhysicalDeviceFormatProperties2 with
  // VkDrmFormatModifierPropertiesList2EXT as pNext
  VkDrmFormatModifierPropertiesList2EXT drmFormatModifierPropsList{
      .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
      .pNext = nullptr,
      // Below is empty for now because we query for the number first
      .drmFormatModifierCount = 0,
      .pDrmFormatModifierProperties = nullptr,
  };
  VkFormatProperties2 formatProperties{
      .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
      .pNext = &drmFormatModifierPropsList,
      .formatProperties = {},
  };
  // First call is a query for the number of properties
  vkGetPhysicalDeviceFormatProperties2(vk_physical_device, format,
                                       &formatProperties);
  std::cout << "Number of drmFormatModifiers: "
            << drmFormatModifierPropsList.drmFormatModifierCount << std::endl;
  // Second call retrieves the actual properties
  auto drmFormatModifierProps = std::vector<VkDrmFormatModifierProperties2EXT>(
      drmFormatModifierPropsList.drmFormatModifierCount);
  drmFormatModifierPropsList.pDrmFormatModifierProperties =
      drmFormatModifierProps.data();
  vkGetPhysicalDeviceFormatProperties2(vk_physical_device, format,
                                       &formatProperties);
  // Now convert the list of VkDrmFormatModifierProperties2EXT into a list
  // of just DRM format modifiers
  drm_format_modifier_list output;
  output.reserve(drmFormatModifierPropsList.drmFormatModifierCount);
  std::transform(std::begin(drmFormatModifierProps),
                 std::end(drmFormatModifierProps), std::back_inserter(output),
                 [](const VkDrmFormatModifierProperties2EXT &prop) {
                   // Ignore drmFormatModifierPlaneCount and
                   // drmFormatModifierTilingFeatures for now
                   return prop.drmFormatModifier;
                 });
  return output;
}

std::optional<VkImage>
createVulkanImage(VkImageType type, VkFormat format, VkExtent3D extent,
                  const drm_format_modifier_list &drmFormatModifiers) {
  //  VkImageCreateInfo with VkImageDrmFormatModifierListCreateInfoEXT as pNext
  VkImageDrmFormatModifierListCreateInfoEXT drmFormatModifierList{
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
      .pNext = nullptr,
      .drmFormatModifierCount =
          static_cast<uint32_t>(drmFormatModifiers.size()),
      .pDrmFormatModifiers = drmFormatModifiers.data(),
  };
  // DRM format modifiers impose restrictions on image creation
  // since they already contain a lot of the required information
  VkImageCreateInfo imgCreateInfo{
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &drmFormatModifierList,
      .flags = 0, // VkImageCreateFlags
      .imageType = type,
      .format = format,
      .extent = extent,
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT, // VkImageUsageFlags
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = nullptr,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
  };
  VkImage image;
  if (VK_CHECK_CALL(vkCreateImage(vk_device, &imgCreateInfo, nullptr,
                                  &image)) != VK_SUCCESS) {
    return std::nullopt;
  }
  return image;
}

constexpr int calcNumChannels(sycl::image_channel_order channelOrder) {
  using order = sycl::image_channel_order;
  switch (channelOrder) {
  case order::r:
    return 1;
  case order::rg:
    return 2;
  case order::rgb:
    return 3;
  case order::rgba:
    return 4;
  default:
    // We don't care about other channel orders for the test
    return 0;
  }
}

template <sycl::image_channel_order ChannelOrder,
          sycl::image_channel_type ChannelType>
bool test(const sycl::queue &syclQueue) {
  constexpr VkImageType vulkanImgType = VK_IMAGE_TYPE_2D;
  const VkFormat vulkanFormat =
      vkutil::to_vulkan_format(ChannelOrder, ChannelType);
  constexpr VkExtent3D vulkanImgExtent = {3840, 2160, 1};
  constexpr size_t numPixels = vulkanImgExtent.width * vulkanImgExtent.height;
  constexpr auto numChannels = calcNumChannels(ChannelOrder);
  constexpr size_t imgSizeBytes =
      numChannels * numPixels * sizeof(float); // TODO

  //////////////
  // Negotiation

  // 1. Query Vulkan driver for supported DRM format modifiers
  auto drmFormatModifiersVulkan = getDrmFormatModifiersVulkan(vulkanFormat);
  std::cout << std::hex;
  for (auto drmFormatMod : drmFormatModifiersVulkan) {
    std::cout << "  DRM Format Mod: 0x" << drmFormatMod << std::endl;
  }
  std::cout << std::dec;

  // 2. TODO: Query for DRM format modifiers supported by SYCL device

  // 3. TODO: Create an intersection list of DRM format modifiers
  //    from both Vulkan and SYCL
  // https://docs.kernel.org/userspace-api/dma-buf-alloc-exchange.html#negotiation

  // 4. Create Vulkan image using DRM format modifier list
  // TODO: Provide the list from step 3
  auto vulkanImageOpt = createVulkanImage(
      vulkanImgType, vulkanFormat, vulkanImgExtent, drmFormatModifiersVulkan);
  if (!vulkanImageOpt) {
    return false;
  }
  auto vulkanImage = *vulkanImageOpt;

  // 5. Query the DRM format modifier selected.
  //    This is the final step of the negotiation.
  VkImageDrmFormatModifierPropertiesEXT drmFormatModifierProp{};
  if (VK_CHECK_CALL(vkGetImageDrmFormatModifierPropertiesEXTpfn(
          vk_device, vulkanImage, &drmFormatModifierProp)) != VK_SUCCESS) {
    vkDestroyImage(vk_device, vulkanImage, nullptr);
    return false;
  }
  std::cout << "Chosen Modifier:  0x" << std::hex
            << drmFormatModifierProp.drmFormatModifier << std::dec << std::endl;

  /////////////////////
  // Export from Vulkan

  // 6. Prepare memory for the image
  [[maybe_unused]] VkMemoryRequirements memRequirements;
  auto imgMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
      vulkanImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements);
  auto vulkanImgMemory = vkutil::allocateDeviceMemory(
      imgSizeBytes, imgMemoryTypeIndex, vulkanImage);
  if (VK_CHECK_CALL(vkBindImageMemory(vk_device, vulkanImage, vulkanImgMemory,
                                      0)) != VK_SUCCESS) {
    vkDestroyImage(vk_device, vulkanImage, nullptr);
    return false;
  }

  // 7. Export the Vulkan image
  auto vulkanExportedHandle = vkutil::getMemoryOpaqueFD(vulkanImgMemory);

  ///////////////////
  // Import into SYCL

  // 8. Create a SYCL external_mem_descriptor,
  //    TODO: passing in the DRM format modifier
  auto extMemDesc = syclexp::external_mem_descriptor<syclexp::resource_fd>{
      vulkanExportedHandle, syclexp::external_mem_handle_type::opaque_fd,
      imgSizeBytes};

  bool success = true;
  try {
    // 9. Import Vulkan memory into SYCL using import_external_memory
    //    to get a SYCL external_mem handle
    syclexp::external_mem externalMem =
        syclexp::import_external_memory(extMemDesc, syclQueue);

    // 10. Create a SYCL image_descriptor based on the image information
    //     known from Vulkan
    auto desc = syclexp::image_descriptor{
        sycl::range<2>{vulkanImgExtent.width, vulkanImgExtent.height},
        numChannels, ChannelType};

    // 11. Pass the SYCL external memory descriptor and the SYCL image
    //     descriptor to map_external_image_memory
    syclexp::image_mem_handle mappedExternMemHandle =
        syclexp::map_external_image_memory(externalMem, desc, syclQueue);

    // 12. Pass the SYCL image memory handle and the SYCL image descriptor to
    //     create_image
    auto samp = syclexp::bindless_image_sampler{
        sycl::addressing_mode::repeat,
        sycl::coordinate_normalization_mode::normalized,
        sycl::filtering_mode::linear};
    syclexp::sampled_image_handle syclImage =
        syclexp::create_image(mappedExternMemHandle, samp, desc, syclQueue);

    // This test is just proof of concept for now to check that no errors occur,
    // in the future it would be good to check that the image data is correct

    //////////
    // Cleanup

    syclexp::destroy_image_handle(syclImage, syclQueue);
    syclexp::unmap_external_image_memory(
        mappedExternMemHandle, syclexp::image_type::standard, syclQueue);
    syclexp::release_external_memory(externalMem, syclQueue);
  } catch (const std::exception &ex) {
    std::cerr << "Exception caught: " << ex.what() << std::endl;
    success = false;
  } catch (...) {
    std::cerr << "Unknown exception occurred!" << std::endl;
    success = false;
  }

  vkDestroyImage(vk_device, vulkanImage, nullptr);
  vkFreeMemory(vk_device, vulkanImgMemory, nullptr);

  return success;
}

int main() {
  // 0. Setup
  if (vkutil::setupInstance() != VK_SUCCESS) {
    std::cerr << "Instance setup failed!\n";
    return EXIT_FAILURE;
  }
  sycl::device syclDevice;
  if (vkutil::setupDevice(syclDevice) != VK_SUCCESS) {
    std::cerr << "Device setup failed!\n";
    return EXIT_FAILURE;
  }
  if (!supportsDrmFormatModifiers) {
    std::cout
        << "DRM format modifiers not supported by Vulkan device, skipping."
        << std::endl;
    return EXIT_SUCCESS;
  }
  if (vkutil::setupCommandBuffers() != VK_SUCCESS) {
    std::cerr << "Command buffers setup failed!\n";
    return EXIT_FAILURE;
  }

  sycl::queue syclQueue{syclDevice};

  bool success = true;
  // TODO: More than one test
  success =
      success &&
      test<sycl::image_channel_order::rgba, sycl::image_channel_type::fp32>(
          syclQueue);

  if (vkutil::cleanup() != VK_SUCCESS) {
    std::cerr << "Cleanup failed!\n";
    return EXIT_FAILURE;
  }
  if (success) {
    std::cout << "All tests passed!\n";
    return EXIT_SUCCESS;
  }
  std::cerr << "Test failed\n";
  return EXIT_FAILURE;
}
