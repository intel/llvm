#pragma once

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
// be sure to use -Wno-ignored-attributes on Windows or the __stdcall will freak
// out when doing the device pass compilation

#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <vulkan/vulkan.h>

#ifdef _WIN32
// #define WIN32_LEAN_AND_MEAN
// #define NOMINMAX
// #include <windows.h>

// I just can't, in good conscience, bring myself to import all of windows.h
// when we only need 6 void typedefs.
typedef void *HANDLE;
typedef struct HINSTANCE__ *HINSTANCE;
typedef struct HWND__ *HWND;
typedef struct HMONITOR__ *HMONITOR;
typedef struct _SECURITY_ATTRIBUTES SECURITY_ATTRIBUTES;
typedef unsigned long DWORD;
typedef const wchar_t *LPCWSTR;

#include <vulkan/vulkan_win32.h>
#endif

// ---------------------------------------------------------
// PLATFORM ABSTRACTION
// ---------------------------------------------------------
#ifdef _WIN32
const std::vector<const char *> PLATFORM_EXTENSIONS = {
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME};
const auto PLATFORM_MEM_HANDLE_TYPE =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
const auto PLATFORM_SEM_HANDLE_TYPE =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
const std::vector<const char *> PLATFORM_EXTENSIONS = {
    VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME};
const auto PLATFORM_MEM_HANDLE_TYPE =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
const auto PLATFORM_SEM_HANDLE_TYPE =
    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

// ---------------------------------------------------------
// VULKAN HELPERS & TYPES
// ---------------------------------------------------------

struct VulkanContext {
  VkInstance instance;
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkQueue queue;
  uint32_t queueFamilyIndex;
};

struct ImageResources {
  VkImage image;
  VkDeviceMemory memory;
  VkDeviceSize allocationSize;
  VkExtent3D extent;
};

struct BufferResources {
  VkBuffer buffer;
  VkDeviceMemory memory;
  VkDeviceSize size;
};

#define VK_CHECK(f)                                                            \
  {                                                                            \
    VkResult __vk_res = (f);                                                   \
    if (__vk_res != VK_SUCCESS) {                                              \
      std::cerr << "Vulkan Error at line " << __LINE__ << ": " << __vk_res     \
                << std::endl;                                                  \
      throw std::runtime_error("Vulkan Error");                                \
    }                                                                          \
  }

// ---------------------------------------------------------
// FORMAT MAPPING & STRINGS (Unchanged)
// ---------------------------------------------------------

inline std::string getFormatString(VkFormat fmt) {
  switch (fmt) {
  case VK_FORMAT_R32_SFLOAT:
    return "VK_FORMAT_R32_SFLOAT";
  case VK_FORMAT_R32G32_SFLOAT:
    return "VK_FORMAT_R32G32_SFLOAT";
  case VK_FORMAT_R32G32B32A32_SFLOAT:
    return "VK_FORMAT_R32G32B32A32_SFLOAT";
  case VK_FORMAT_R16_SFLOAT:
    return "VK_FORMAT_R16_SFLOAT";
  case VK_FORMAT_R16G16_SFLOAT:
    return "VK_FORMAT_R16G16_SFLOAT";
  case VK_FORMAT_R16G16B16A16_SFLOAT:
    return "VK_FORMAT_R16G16B16A16_SFLOAT";
  case VK_FORMAT_R32_SINT:
    return "VK_FORMAT_R32_SINT";
  case VK_FORMAT_R32G32_SINT:
    return "VK_FORMAT_R32G32_SINT";
  case VK_FORMAT_R32G32B32A32_SINT:
    return "VK_FORMAT_R32G32B32A32_SINT";
  case VK_FORMAT_R32_UINT:
    return "VK_FORMAT_R32_UINT";
  case VK_FORMAT_R32G32_UINT:
    return "VK_FORMAT_R32G32_UINT";
  case VK_FORMAT_R32G32B32A32_UINT:
    return "VK_FORMAT_R32G32B32A32_UINT";
  case VK_FORMAT_R16_SINT:
    return "VK_FORMAT_R16_SINT";
  case VK_FORMAT_R16G16_SINT:
    return "VK_FORMAT_R16G16_SINT";
  case VK_FORMAT_R16G16B16A16_SINT:
    return "VK_FORMAT_R16G16B16A16_SINT";
  case VK_FORMAT_R16_UINT:
    return "VK_FORMAT_R16_UINT";
  case VK_FORMAT_R16G16_UINT:
    return "VK_FORMAT_R16G16_UINT";
  case VK_FORMAT_R16G16B16A16_UINT:
    return "VK_FORMAT_R16G16B16A16_UINT";
  case VK_FORMAT_R8_SINT:
    return "VK_FORMAT_R8_SINT";
  case VK_FORMAT_R8G8_SINT:
    return "VK_FORMAT_R8G8_SINT";
  case VK_FORMAT_R8G8B8A8_SINT:
    return "VK_FORMAT_R8G8B8A8_SINT";
  case VK_FORMAT_R8_UINT:
    return "VK_FORMAT_R8_UINT";
  case VK_FORMAT_R8G8_UINT:
    return "VK_FORMAT_R8G8_UINT";
  case VK_FORMAT_R8G8B8A8_UINT:
    return "VK_FORMAT_R8G8B8A8_UINT";
  case VK_FORMAT_R8_UNORM:
    return "VK_FORMAT_R8_UNORM";
  case VK_FORMAT_R8G8_UNORM:
    return "VK_FORMAT_R8G8_UNORM";
  case VK_FORMAT_R8G8B8A8_UNORM:
    return "VK_FORMAT_R8G8B8A8_UNORM";
  default:
    return "UNKNOWN_FORMAT (" + std::to_string(fmt) + ")";
  }
}

template <typename T> VkFormat getVulkanFormat(int channels);
template <> inline VkFormat getVulkanFormat<float>(int channels) {
  switch (channels) {
  case 1:
    return VK_FORMAT_R32_SFLOAT;
  case 2:
    return VK_FORMAT_R32G32_SFLOAT;
  case 4:
    return VK_FORMAT_R32G32B32A32_SFLOAT;
  default:
    throw std::runtime_error("Unsupported channels for float");
  }
}
template <> inline VkFormat getVulkanFormat<int32_t>(int channels) {
  switch (channels) {
  case 1:
    return VK_FORMAT_R32_SINT;
  case 2:
    return VK_FORMAT_R32G32_SINT;
  case 4:
    return VK_FORMAT_R32G32B32A32_SINT;
  default:
    throw std::runtime_error("Unsupported channels for int32");
  }
}
template <> inline VkFormat getVulkanFormat<uint32_t>(int channels) {
  switch (channels) {
  case 1:
    return VK_FORMAT_R32_UINT;
  case 2:
    return VK_FORMAT_R32G32_UINT;
  case 4:
    return VK_FORMAT_R32G32B32A32_UINT;
  default:
    throw std::runtime_error("Unsupported channels for uint32");
  }
}
template <> inline VkFormat getVulkanFormat<int16_t>(int channels) {
  switch (channels) {
  case 1:
    return VK_FORMAT_R16_SINT;
  case 2:
    return VK_FORMAT_R16G16_SINT;
  case 4:
    return VK_FORMAT_R16G16B16A16_SINT;
  default:
    throw std::runtime_error("Unsupported channels for int16");
  }
}
template <> inline VkFormat getVulkanFormat<uint16_t>(int channels) {
  switch (channels) {
  case 1:
    return VK_FORMAT_R16_UINT;
  case 2:
    return VK_FORMAT_R16G16_UINT;
  case 4:
    return VK_FORMAT_R16G16B16A16_UINT;
  default:
    throw std::runtime_error("Unsupported channels for uint16");
  }
}
template <> inline VkFormat getVulkanFormat<uint8_t>(int channels) {
  switch (channels) {
  case 1:
    return VK_FORMAT_R8_UINT;
  case 2:
    return VK_FORMAT_R8G8_UINT;
  case 4:
    return VK_FORMAT_R8G8B8A8_UINT;
  default:
    throw std::runtime_error("Unsupported channels for uint8");
  }
}
template <> inline VkFormat getVulkanFormat<int8_t>(int channels) {
  switch (channels) {
  case 1:
    return VK_FORMAT_R8_SINT;
  case 2:
    return VK_FORMAT_R8G8_SINT;
  case 4:
    return VK_FORMAT_R8G8B8A8_SINT;
  default:
    throw std::runtime_error("Unsupported channels for int8");
  }
}
inline VkFormat getUnorm8Format(int channels) {
  switch (channels) {
  case 1:
    return VK_FORMAT_R8_UNORM;
  case 2:
    return VK_FORMAT_R8G8_UNORM;
  case 4:
    return VK_FORMAT_R8G8B8A8_UNORM;
  default:
    throw std::runtime_error("Unsupported channels for UNORM8");
  }
}

// ---------------------------------------------------------
// some test utils
// ---------------------------------------------------------

// Generates a deterministic test value based on position and channel
template <typename T>
T generateTestValue(size_t index, int channel, size_t rangeMax) {
  if constexpr (std::is_floating_point_v<T>) {
    // Float: 0.0 -> 1.0 gradient with channel offset
    float val = (float)index / (float)(rangeMax > 1 ? rangeMax - 1 : 1);
    return static_cast<T>(val + (float)channel * 0.1f);
  } else {
    // Integer: Wrapping pattern to avoid overflow
    return static_cast<T>((index + channel * 10) % 127);
  }
}

// Compares values with appropriate tolerance for Floats
template <typename T> bool checkValue(T actual, T expected) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::abs(actual - expected) < 0.01f;
  } else {
    return actual == expected;
  }
}

// ---------------------------------------------------------
// Boilerplate
// ---------------------------------------------------------

size_t getRowPitch(VulkanContext &ctx, VkImage image) {
  VkSubresourceLayout layout;
  VkImageSubresource subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
  vkGetImageSubresourceLayout(ctx.device, image, &subResource, &layout);
  return layout.rowPitch;
}

inline uint32_t findMemoryType(VkPhysicalDevice physicalDevice,
                               uint32_t typeFilter,
                               VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags &
                                    properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find suitable memory type!");
}

inline VulkanContext createVulkanContext() {
  VulkanContext ctx;
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  VK_CHECK(vkCreateInstance(&createInfo, nullptr, &ctx.instance));

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, nullptr);
  std::vector<VkPhysicalDevice> devices(deviceCount);
  vkEnumeratePhysicalDevices(ctx.instance, &deviceCount, devices.data());
  ctx.physicalDevice = devices[0];

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(ctx.physicalDevice,
                                           &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      ctx.physicalDevice, &queueFamilyCount, queueFamilies.data());

  ctx.queueFamilyIndex = -1;
  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      ctx.queueFamilyIndex = i;
      break;
    }
  }

  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = ctx.queueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  // Enable timeline semaphore feature (Vulkan 1.2 core)
  VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures{};
  timelineFeatures.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
  timelineFeatures.timelineSemaphore = VK_TRUE;

  VkDeviceCreateInfo deviceCreateInfo{};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.pNext = &timelineFeatures;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  deviceCreateInfo.queueCreateInfoCount = 1;

  // UPDATED: Use dynamic platform extensions
  deviceCreateInfo.enabledExtensionCount =
      static_cast<uint32_t>(PLATFORM_EXTENSIONS.size());
  deviceCreateInfo.ppEnabledExtensionNames = PLATFORM_EXTENSIONS.data();

  VK_CHECK(vkCreateDevice(ctx.physicalDevice, &deviceCreateInfo, nullptr,
                          &ctx.device));
  vkGetDeviceQueue(ctx.device, ctx.queueFamilyIndex, 0, &ctx.queue);

  return ctx;
}

inline ImageResources createExportableImage(
    VulkanContext &ctx, VkExtent3D extent, VkFormat format, VkImageType type,
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL,
    VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT |
                              VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                              VK_IMAGE_USAGE_TRANSFER_DST_BIT) {
  VkImageCreateInfo imageInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  imageInfo.imageType = type;
  imageInfo.extent = extent;
  imageInfo.mipLevels = 1;
  imageInfo.arrayLayers = 1;
  imageInfo.format = format;
  imageInfo.tiling = tiling;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage = usage;
  imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

  // Export Memory Support (Image Side)
  VkExternalMemoryImageCreateInfo extMemInfo = {
      VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
  extMemInfo.handleTypes = PLATFORM_MEM_HANDLE_TYPE;
  imageInfo.pNext = &extMemInfo;

  ImageResources res;
  res.extent = extent;

  VK_CHECK(vkCreateImage(ctx.device, &imageInfo, nullptr, &res.image));

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(ctx.device, res.image, &memRequirements);

  // Export Memory Allocation (Memory Side)
  VkExportMemoryAllocateInfo exportAllocInfo = {
      VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
  exportAllocInfo.handleTypes = PLATFORM_MEM_HANDLE_TYPE;

  // Dedicated Allocation (Required for Windows Interop)
  VkMemoryDedicatedAllocateInfo dedicatedAllocInfo = {
      VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  dedicatedAllocInfo.image = res.image;
  dedicatedAllocInfo.buffer = VK_NULL_HANDLE;
  dedicatedAllocInfo.pNext = &exportAllocInfo; // Chain export info

  VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  allocInfo.pNext = &dedicatedAllocInfo; // Chain Dedicated info
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(ctx.physicalDevice, memRequirements.memoryTypeBits,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  res.allocationSize = allocInfo.allocationSize;

  VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &res.memory));
  VK_CHECK(vkBindImageMemory(ctx.device, res.image, res.memory, 0));

  return res;
}

inline VkSemaphore createExportableSemaphore(VulkanContext &ctx) {
  VkExportSemaphoreCreateInfo exportInfo{};
  exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
  exportInfo.handleTypes = PLATFORM_SEM_HANDLE_TYPE;

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphoreInfo.pNext = &exportInfo;

  VkSemaphore semaphore;
  VK_CHECK(vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &semaphore));
  return semaphore;
}

inline VkSemaphore
createExportableTimelineSemaphore(VulkanContext &ctx,
                                  uint64_t initialValue = 0) {
  VkSemaphoreTypeCreateInfo typeInfo{};
  typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  typeInfo.initialValue = initialValue;

  VkExportSemaphoreCreateInfo exportInfo{};
  exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
  exportInfo.handleTypes = PLATFORM_SEM_HANDLE_TYPE;
  exportInfo.pNext = &typeInfo;

  VkSemaphoreCreateInfo semaphoreInfo{};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphoreInfo.pNext = &exportInfo;

  VkSemaphore semaphore;
  VK_CHECK(vkCreateSemaphore(ctx.device, &semaphoreInfo, nullptr, &semaphore));
  return semaphore;
}

// ---------------------------------------------------------
// BUFFER HELPERS
// ---------------------------------------------------------

inline BufferResources createExportableBuffer(
    VulkanContext &ctx, VkDeviceSize size, VkBufferUsageFlags usage,
    VkExternalMemoryHandleTypeFlagBits handleType = PLATFORM_MEM_HANDLE_TYPE) {
  VkExternalMemoryBufferCreateInfo extMemInfo{};
  extMemInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
  extMemInfo.handleTypes = handleType;

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.pNext = &extMemInfo;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  BufferResources res;
  res.size = size;
  VK_CHECK(vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &res.buffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(ctx.device, res.buffer, &memRequirements);

  VkExportMemoryAllocateInfo exportAllocInfo{};
  exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
  exportAllocInfo.handleTypes = handleType;

  VkMemoryDedicatedAllocateInfo dedicatedAllocInfo{};
  dedicatedAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
  dedicatedAllocInfo.buffer = res.buffer;
  dedicatedAllocInfo.pNext = &exportAllocInfo;

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.pNext = &dedicatedAllocInfo;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(ctx.physicalDevice, memRequirements.memoryTypeBits,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &res.memory));
  VK_CHECK(vkBindBufferMemory(ctx.device, res.buffer, res.memory, 0));
  return res;
}

inline BufferResources createStagingBuffer(VulkanContext &ctx,
                                           VkDeviceSize size,
                                           VkBufferUsageFlags usage) {
  BufferResources res;
  res.size = size;

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  VK_CHECK(vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &res.buffer));

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx.device, res.buffer, &req);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = req.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(ctx.physicalDevice, req.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  VK_CHECK(vkAllocateMemory(ctx.device, &allocInfo, nullptr, &res.memory));
  VK_CHECK(vkBindBufferMemory(ctx.device, res.buffer, res.memory, 0));
  return res;
}

inline void cleanupBuffer(VulkanContext &ctx, BufferResources &res) {
  vkDestroyBuffer(ctx.device, res.buffer, nullptr);
  vkFreeMemory(ctx.device, res.memory, nullptr);
}

// ---------------------------------------------------------
// PLATFORM SPECIFIC GETTERS
// ---------------------------------------------------------

#ifdef _WIN32
inline HANDLE
getMemHandle(VulkanContext &ctx, VkDeviceMemory memory,
             VkExternalMemoryHandleTypeFlagBits handleType =
                 VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
  VkMemoryGetWin32HandleInfoKHR getHandleInfo{};
  getHandleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  getHandleInfo.memory = memory;
  getHandleInfo.handleType = handleType;

  HANDLE handle;
  auto func = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
      ctx.device, "vkGetMemoryWin32HandleKHR");
  if (!func)
    throw std::runtime_error("Failed to load vkGetMemoryWin32HandleKHR");
  VK_CHECK(func(ctx.device, &getHandleInfo, &handle));
  return handle;
}

inline HANDLE getSemaphoreHandle(VulkanContext &ctx, VkSemaphore semaphore) {
  VkSemaphoreGetWin32HandleInfoKHR getHandleInfo{};
  getHandleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  getHandleInfo.semaphore = semaphore;
  getHandleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

  HANDLE handle;
  auto func = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
      ctx.device, "vkGetSemaphoreWin32HandleKHR");
  if (!func)
    throw std::runtime_error("Failed to load vkGetSemaphoreWin32HandleKHR");
  VK_CHECK(func(ctx.device, &getHandleInfo, &handle));
  return handle;
}

// Stubs for compile compat if you have sloppy ifdefs elsewhere
inline int getMemFd(VulkanContext &ctx, VkDeviceMemory memory) {
  throw std::runtime_error("getMemFd called on Windows!");
}
inline int getMemFd(VulkanContext &ctx, VkDeviceMemory memory,
                    VkExternalMemoryHandleTypeFlagBits) {
  throw std::runtime_error("getMemFd called on Windows!");
}
inline int getSemaphoreFd(VulkanContext &ctx, VkSemaphore semaphore) {
  throw std::runtime_error("getSemaphoreFd called on Windows!");
}

#else
// LINUX IMPLEMENTATION
inline int getMemFd(VulkanContext &ctx, VkDeviceMemory memory) {
  VkMemoryGetFdInfoKHR getFdInfo{};
  getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  getFdInfo.memory = memory;
  getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  int fd;
  auto func =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
  if (!func)
    throw std::runtime_error("Failed to load vkGetMemoryFdKHR");
  VK_CHECK(func(ctx.device, &getFdInfo, &fd));
  return fd;
}

// Overload with explicit handle type (for DMA_BUF support)
inline int getMemFd(VulkanContext &ctx, VkDeviceMemory memory,
                    VkExternalMemoryHandleTypeFlagBits handleType) {
  VkMemoryGetFdInfoKHR getFdInfo{};
  getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  getFdInfo.memory = memory;
  getFdInfo.handleType = handleType;

  int fd;
  auto func =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(ctx.device, "vkGetMemoryFdKHR");
  if (!func)
    throw std::runtime_error("Failed to load vkGetMemoryFdKHR");
  VK_CHECK(func(ctx.device, &getFdInfo, &fd));
  return fd;
}

inline int getSemaphoreFd(VulkanContext &ctx, VkSemaphore semaphore) {
  VkSemaphoreGetFdInfoKHR getFdInfo{};
  getFdInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  getFdInfo.semaphore = semaphore;
  getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  int fd;
  auto func = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
      ctx.device, "vkGetSemaphoreFdKHR");
  if (!func)
    throw std::runtime_error("Failed to load vkGetSemaphoreFdKHR");
  VK_CHECK(func(ctx.device, &getFdInfo, &fd));
  return fd;
}
#endif

// ---------------------------------------------------------
// HELPER: Upload Data (Host -> Staging -> Device)
// ---------------------------------------------------------
template <typename Functor>
void uploadImage(VulkanContext &ctx, ImageResources &imgRes, int channels,
                 VkSemaphore signalSemaphore, Functor generator) {
  uint32_t width = imgRes.extent.width;
  uint32_t height = imgRes.extent.height;
  uint32_t depth = imgRes.extent.depth;
  size_t totalPixels = width * height * depth;

  // 1. Create Staging Buffer
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingMemory;
  VkDeviceSize dataSize = totalPixels * channels * 4; // Max safe size

  VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = dataSize;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VK_CHECK(vkCreateBuffer(ctx.device, &bi, nullptr, &stagingBuffer));

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx.device, stagingBuffer, &req);
  VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = findMemoryType(ctx.physicalDevice, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(ctx.device, &ai, nullptr, &stagingMemory));
  VK_CHECK(vkBindBufferMemory(ctx.device, stagingBuffer, stagingMemory, 0));

  // 2. Map and Fill
  void *data;
  VK_CHECK(vkMapMemory(ctx.device, stagingMemory, 0, dataSize, 0, &data));

  using T = decltype(generator(0, 0));
  T *ptr = static_cast<T *>(data);

  for (size_t i = 0; i < totalPixels; ++i) {
    for (int c = 0; c < channels; ++c) {
      ptr[i * channels + c] = generator(i, c);
    }
  }
  vkUnmapMemory(ctx.device, stagingMemory);

  // 3. Command Buffer
  VkCommandPoolCreateInfo poolInfo = {
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
  VkCommandPool pool;
  VK_CHECK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &pool));

  VkCommandBuffer cmd;
  VkCommandBufferAllocateInfo ca = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  vkAllocateCommandBuffers(ctx.device, &ca, &cmd);

  VkCommandBufferBeginInfo beginInfo = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cmd, &beginInfo);

  VkImageMemoryBarrier bar1 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  bar1.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  bar1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  bar1.image = imgRes.image;
  bar1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  bar1.srcAccessMask = 0;
  bar1.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &bar1);

  VkBufferImageCopy region = {};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = imgRes.extent; // Uses the 3D extent automatically
  vkCmdCopyBufferToImage(cmd, stagingBuffer, imgRes.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  VkImageMemoryBarrier bar2 = bar1;
  bar2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  bar2.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  bar2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  bar2.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &bar2);

  vkEndCommandBuffer(cmd);

  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  if (signalSemaphore != VK_NULL_HANDLE) {
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &signalSemaphore;
  }
  vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(ctx.queue);

  vkDestroyCommandPool(ctx.device, pool, nullptr);
  vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
  vkFreeMemory(ctx.device, stagingMemory, nullptr);
}

// ---------------------------------------------------------
// HELPER: Verify Data (Device -> Staging -> Host)
// ---------------------------------------------------------
template <typename Functor>
bool verifyImage(VulkanContext &ctx, ImageResources &imgRes, int channels,
                 VkSemaphore waitSemaphore, Functor expectedGenerator) {
  uint32_t width = imgRes.extent.width;
  uint32_t height = imgRes.extent.height;
  uint32_t depth = imgRes.extent.depth;

  // Dimensions Check
  if (width == 0 || height == 0 || depth == 0) {
    std::cerr << "[FATAL] verifyImage: Invalid Dimensions " << width << "x"
              << height << std::endl;
    return false;
  }

  size_t totalPixels = width * height * depth;

  // Create Staging Buffer
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingMemory;
  VkDeviceSize dataSize = totalPixels * channels * 4;

  VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bi.size = dataSize;
  bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VK_CHECK(vkCreateBuffer(ctx.device, &bi, nullptr, &stagingBuffer));

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(ctx.device, stagingBuffer, &req);
  VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = findMemoryType(ctx.physicalDevice, req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(ctx.device, &ai, nullptr, &stagingMemory));
  VK_CHECK(vkBindBufferMemory(ctx.device, stagingBuffer, stagingMemory, 0));

  // Command Buffer
  VkCommandPoolCreateInfo poolInfo = {
      VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
  VkCommandPool pool;
  VK_CHECK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &pool));

  VkCommandBuffer cmd;
  VkCommandBufferAllocateInfo ca = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ca.commandPool = pool;
  ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ca.commandBufferCount = 1;
  vkAllocateCommandBuffers(ctx.device, &ca, &cmd);

  VkCommandBufferBeginInfo beginInfo = {
      VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  vkBeginCommandBuffer(cmd, &beginInfo);

  // Safety Barrier: Ensure writes from SYCL (External) are visible before
  // transfer
  VkImageMemoryBarrier barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL; // Assume SYCL left it in GENERAL
  barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL; // We keep it in GENERAL
  barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;  // Wait for any writes
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT; // Ready for us to read
  barrier.image = imgRes.image;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  // Copy
  VkBufferImageCopy region = {};
  region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
  region.imageExtent = imgRes.extent;
  vkCmdCopyImageToBuffer(cmd, imgRes.image, VK_IMAGE_LAYOUT_GENERAL,
                         stagingBuffer, 1, &region);

  vkEndCommandBuffer(cmd);

  // Submit
  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  std::vector<VkPipelineStageFlags> waitStages = {
      VK_PIPELINE_STAGE_TRANSFER_BIT};
  if (waitSemaphore != VK_NULL_HANDLE) {
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &waitSemaphore;
    si.pWaitDstStageMask = waitStages.data();
  }
  vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE);
  vkQueueWaitIdle(ctx.queue);

  // Verify Data
  void *data;
  vkMapMemory(ctx.device, stagingMemory, 0, dataSize, 0, &data);

  using T = decltype(expectedGenerator(0, 0));
  T *ptr = static_cast<T *>(data);
  bool passed = true;
  int errors = 0;

  for (size_t i = 0; i < totalPixels; ++i) {
    for (int c = 0; c < channels; ++c) {
      T actual = ptr[i * channels + c];
      T expected = expectedGenerator(i, c);

      bool match = false;
      if constexpr (std::is_floating_point_v<T>) {
        match = std::abs((float)actual - (float)expected) < 0.05f;
      } else {
        match = (actual == expected);
      }

      if (!match) {
        passed = false;
        if (errors++ < 5)
          std::cout << "Mismatch at " << i << " ch:" << c
                    << " Got: " << (double)actual
                    << " Exp: " << (double)expected << std::endl;
      }
    }
  }

  vkUnmapMemory(ctx.device, stagingMemory);
  vkDestroyCommandPool(ctx.device, pool, nullptr);
  vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
  vkFreeMemory(ctx.device, stagingMemory, nullptr);
  return passed;
}

template <typename T>
bool uploadAndVerify(VulkanContext &ctx, ImageResources &imgRes,
                     VkSemaphore signalSemaphore = VK_NULL_HANDLE,
                     int channels = 4) {
  size_t texWidth = imgRes.extent.width;
  size_t texHeight = imgRes.extent.height;
  size_t texDepth = imgRes.extent.depth;
  size_t totalPixels = texWidth * texHeight * texDepth;
  VkDeviceSize imageSize = totalPixels * channels * sizeof(T);

  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;

  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = imageSize;
  bufferInfo.usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  VK_CHECK(vkCreateBuffer(ctx.device, &bufferInfo, nullptr, &stagingBuffer));

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(ctx.device, stagingBuffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memRequirements.size;
  allocInfo.memoryTypeIndex =
      findMemoryType(ctx.physicalDevice, memRequirements.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  VK_CHECK(
      vkAllocateMemory(ctx.device, &allocInfo, nullptr, &stagingBufferMemory));
  VK_CHECK(
      vkBindBufferMemory(ctx.device, stagingBuffer, stagingBufferMemory, 0));

  // GENERATE DATA
  void *data;
  vkMapMemory(ctx.device, stagingBufferMemory, 0, imageSize, 0, &data);
  T *pixelData = (T *)data;

  for (size_t i = 0; i < totalPixels; i++) {
    for (int c = 0; c < channels; ++c) {
      pixelData[i * channels + c] = generateTestValue<T>(i, c, totalPixels);
    }
  }
  vkUnmapMemory(ctx.device, stagingBufferMemory);

  // COPY TO IMAGE
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.queueFamilyIndex = ctx.queueFamilyIndex;
  VkCommandPool commandPool;
  VK_CHECK(vkCreateCommandPool(ctx.device, &poolInfo, nullptr, &commandPool));

  VkCommandBufferAllocateInfo cmdAllocInfo{};
  cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmdAllocInfo.commandPool = commandPool;
  cmdAllocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  VK_CHECK(vkAllocateCommandBuffers(ctx.device, &cmdAllocInfo, &commandBuffer));

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  VkImageMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = imgRes.image;
  barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.srcAccessMask = 0;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier);

  VkBufferImageCopy region{};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = imgRes.extent;

  vkCmdCopyBufferToImage(commandBuffer, stagingBuffer, imgRes.image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  VkImageMemoryBarrier barrier2 = barrier;
  barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier2.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier2.dstAccessMask =
      VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier2);

  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  if (signalSemaphore != VK_NULL_HANDLE) {
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &signalSemaphore;
  }

  VK_CHECK(vkQueueSubmit(ctx.queue, 1, &submitInfo, VK_NULL_HANDLE));
  vkQueueWaitIdle(ctx.queue);

  // COPY BACK (Round Trip Verify)
  vkResetCommandBuffer(commandBuffer, 0);
  vkBeginCommandBuffer(commandBuffer, &beginInfo);

  VkImageMemoryBarrier barrier3 = barrier2;
  barrier3.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier3.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier3.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
  barrier3.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier3);

  vkCmdCopyImageToBuffer(commandBuffer, imgRes.image,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingBuffer, 1,
                         &region);

  VkImageMemoryBarrier barrier4 = barrier3;
  barrier4.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
  barrier4.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrier4.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  barrier4.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                       nullptr, 1, &barrier4);

  vkEndCommandBuffer(commandBuffer);

  // Use a separate submit for readback — do NOT re-signal the binary
  // semaphore.  The first submit already signaled it; a second signal
  // without an intervening wait is invalid for binary semaphores.
  VkSubmitInfo readbackSubmit{};
  readbackSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  readbackSubmit.commandBufferCount = 1;
  readbackSubmit.pCommandBuffers = &commandBuffer;
  VK_CHECK(vkQueueSubmit(ctx.queue, 1, &readbackSubmit, VK_NULL_HANDLE));
  vkQueueWaitIdle(ctx.queue);

  vkMapMemory(ctx.device, stagingBufferMemory, 0, imageSize, 0, &data);
  T *checkData = (T *)data;

  bool valid = true;
  for (size_t i = 0; i < totalPixels * channels; i++) {
    size_t pixelIdx = i / channels;
    int channelIdx = i % channels;
    T expected = generateTestValue<T>(pixelIdx, channelIdx, totalPixels);

    if (!checkValue(checkData[i], expected)) {
      valid = false;
      // Uncomment for debugging
      // std::cout << "RoundTrip Mismatch: " << (float)checkData[i] << " != " <<
      // (float)expected << std::endl;
      break;
    }
  }
  vkUnmapMemory(ctx.device, stagingBufferMemory);

  if (valid)
    std::cout << "✓ Vulkan Data Verified (Internal Round-Trip Passed)"
              << std::endl;
  else
    std::cerr << "X Vulkan Data Verification Failed!" << std::endl;

  vkDestroyBuffer(ctx.device, stagingBuffer, nullptr);
  vkFreeMemory(ctx.device, stagingBufferMemory, nullptr);
  vkDestroyCommandPool(ctx.device, commandPool, nullptr);

  return valid;
}

// ---------------------------------------------------------
// NEW GENERIC API (The "Boss" API)
// ---------------------------------------------------------
// Used by the Boss Battle to pass custom functors
template <typename Functor>
bool uploadAndVerify(VulkanContext &ctx, ImageResources &imgRes,
                     VkSemaphore signalSemaphore, int channels,
                     Functor generator) {
  // Same logic, but using the passed generator
  uploadImage(ctx, imgRes, channels, VK_NULL_HANDLE, generator);

  if (!verifyImage(ctx, imgRes, channels, VK_NULL_HANDLE, generator)) {
    return false;
  }

  if (signalSemaphore != VK_NULL_HANDLE) {
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &signalSemaphore;
    vkQueueSubmit(ctx.queue, 1, &si, VK_NULL_HANDLE);
  }
  return true;
}

// ---------------------------------------------------------
// HELPER: Cleanup Just Image Resources (Keep Device Alive)
// ---------------------------------------------------------
inline void cleanupImageResources(VulkanContext &ctx, ImageResources &res) {
  if (res.image != VK_NULL_HANDLE) {
    vkDestroyImage(ctx.device, res.image, nullptr);
    res.image = VK_NULL_HANDLE;
  }
  if (res.memory != VK_NULL_HANDLE) {
    vkFreeMemory(ctx.device, res.memory, nullptr);
    res.memory = VK_NULL_HANDLE;
  }
}

inline void cleanupVulkan(VulkanContext &ctx, ImageResources &res) {
  vkDestroyImage(ctx.device, res.image, nullptr);
  vkFreeMemory(ctx.device, res.memory, nullptr);
  vkDestroyDevice(ctx.device, nullptr);
  vkDestroyInstance(ctx.instance, nullptr);
}
