#pragma once
#include <vulkan/vulkan.h>

#include <cstdlib>
#include <iostream>
#include <vector>

void printString(std::string str) {
#ifdef VERBOSE_PRINT
  std::cout << str;
#endif
}

template <typename DType, int NChannels>
std::ostream &operator<<(std::ostream &os,
                         const sycl::vec<DType, NChannels> &vec) {
  std::string str{""};
  for (int i = 0; i < NChannels; ++i) {
    str += std::to_string(vec[i]) + ",";
  }
  str.pop_back();
  os << str;
  return os;
}

template <typename DType, int NChannels>
bool equal_vec(sycl::vec<DType, NChannels> v1, sycl::vec<DType, NChannels> v2) {
  for (int i = 0; i < NChannels; ++i) {
    if (v1[i] != v2[i]) {
      return false;
    }
  }
  return true;
}

template <typename DType, int NChannel>
constexpr sycl::vec<DType, NChannel> initVector(DType val) {
  if constexpr (NChannel == 1) {
    return sycl::vec<DType, NChannel>{val};
  } else if constexpr (NChannel == 2) {
    return sycl::vec<DType, NChannel>{val, val};
  } else if constexpr (NChannel == 4) {
    return sycl::vec<DType, NChannel>{val, val, val, val};
  } else {
    std::cerr << "unsupported number of channels " << NChannel << "\n";
    exit(-1);
  }
}

#define VK_CHECK_CALL_RET(call)                                                \
  {                                                                            \
    VkResult err = call;                                                       \
    if (err != VK_SUCCESS)                                                     \
      return err;                                                              \
  }

#define VK_CHECK_CALL(call)                                                    \
  {                                                                            \
    VkResult err = call;                                                       \
    if (err != VK_SUCCESS)                                                     \
      std::cerr << #call << " failed. Code: " << err << "\n";                  \
  }

static VkInstance vk_instance;
static VkPhysicalDevice vk_physical_device;
static VkDebugUtilsMessengerEXT vk_debug_messenger;
static VkDevice vk_device;
static VkQueue vk_compute_queue;
static VkQueue vk_transfer_queue;

static PFN_vkGetMemoryFdKHR vk_getMemoryFdKHR;
static PFN_vkGetSemaphoreFdKHR vk_getSemaphoreFdKHR;

static uint32_t vk_computeQueueFamilyIndex;
static uint32_t vk_transferQueueFamilyIndex;

static VkCommandPool vk_computeCmdPool;
static VkCommandPool vk_transferCmdPool;

static VkCommandBuffer vk_computeCmdBuffer;
static VkCommandBuffer vk_transferCmdBuffers[2];

static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  // Only print errors from validation layer
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    std::cerr << pCallbackData->pMessage << "\n";
  }
  return VK_FALSE;
}

namespace vkutil {
VkResult setupInstance() {
  VkApplicationInfo ai = {};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "SYCL-Vulkan-Interop";
  ai.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  ai.pEngineName = "";
  ai.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  ai.apiVersion = VK_API_VERSION_1_0;

  uint32_t layerCount;
  VK_CHECK_CALL_RET(vkEnumerateInstanceLayerProperties(&layerCount, nullptr));

  std::vector<VkLayerProperties> availableLayers(layerCount);
  VK_CHECK_CALL_RET(
      vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()));

  VkInstanceCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &ai;
  std::vector<const char *> extensions = {
      VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
      VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME,
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
  ci.enabledExtensionCount = extensions.size();
  ci.ppEnabledExtensionNames = extensions.data();
  std::vector<const char *> layers = {"VK_LAYER_KHRONOS_validation"};
  ci.enabledLayerCount = layers.size();
  ci.ppEnabledLayerNames = layers.data();

  VK_CHECK_CALL_RET(vkCreateInstance(&ci, nullptr, &vk_instance));

  VkDebugUtilsMessengerCreateInfoEXT dumci = {};
  dumci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  dumci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  dumci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  dumci.pfnUserCallback = debugCallback;

  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
      vk_instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    VK_CHECK_CALL_RET(func(vk_instance, &dumci, nullptr, &vk_debug_messenger));
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }

  vk_getMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(
      vk_instance, "vkGetMemoryFdKHR");

  vk_getSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetInstanceProcAddr(
      vk_instance, "vkGetSemaphoreFdKHR");

  return VK_SUCCESS;
}

VkResult setupDevice(std::string device) {
  uint32_t physicalDeviceCount = 0;
  VK_CHECK_CALL_RET(
      vkEnumeratePhysicalDevices(vk_instance, &physicalDeviceCount, nullptr));
  if (physicalDeviceCount == 0) {
    return VK_ERROR_DEVICE_LOST;
  }
  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  VK_CHECK_CALL_RET(vkEnumeratePhysicalDevices(
      vk_instance, &physicalDeviceCount, physicalDevices.data()));

  bool foundDevice = false;

  for (int i = 0; i < physicalDeviceCount; i++) {
    vk_physical_device = physicalDevices[i];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(vk_physical_device, &props);
    std::string str(props.deviceName);

    if (str.find(device) != std::string::npos) {
      foundDevice = true;
      break;
    }
  }

  if (!foundDevice) {
    std::cerr << "Failed to find suitable device!\n";
    return VK_ERROR_DEVICE_LOST;
  }

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device,
                                           &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(
      vk_physical_device, &queueFamilyCount, queueFamilies.data());
  uint32_t i = 0;
  for (auto &qf : queueFamilies) {
    if (qf.queueFlags & VK_QUEUE_COMPUTE_BIT) {
      vk_computeQueueFamilyIndex = i;
    }
    if (qf.queueFlags & VK_QUEUE_TRANSFER_BIT) {
      vk_transferQueueFamilyIndex = i;
    }
    ++i;
  }

  float queuePriority = 1.f;

  std::vector<VkDeviceQueueCreateInfo> qcis;
  if (vk_computeQueueFamilyIndex == vk_transferQueueFamilyIndex) {
    qcis.resize(1);
    qcis[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qcis[0].queueFamilyIndex = vk_transferQueueFamilyIndex;
    qcis[0].queueCount = 1;
    qcis[0].pQueuePriorities = &queuePriority;
  } else {
    qcis.resize(2);
    qcis[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qcis[0].queueFamilyIndex = vk_transferQueueFamilyIndex;
    qcis[0].queueCount = 1;
    qcis[0].pQueuePriorities = &queuePriority;

    qcis[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qcis[1].queueFamilyIndex = vk_computeQueueFamilyIndex;
    qcis[1].queueCount = 1;
    qcis[1].pQueuePriorities = &queuePriority;
  }

  VkPhysicalDeviceFeatures deviceFeatures = {};

  std::vector<const char *> extensions = {
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME};

  VkDeviceCreateInfo dci = {};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pQueueCreateInfos = qcis.data();
  dci.queueCreateInfoCount = qcis.size();
  dci.pEnabledFeatures = &deviceFeatures;
  dci.enabledExtensionCount = extensions.size();
  dci.ppEnabledExtensionNames = extensions.data();

  VK_CHECK_CALL_RET(
      vkCreateDevice(vk_physical_device, &dci, nullptr, &vk_device));

  vkGetDeviceQueue(vk_device, vk_transferQueueFamilyIndex, 0,
                   &vk_transfer_queue);
  vkGetDeviceQueue(vk_device, vk_computeQueueFamilyIndex, 0, &vk_compute_queue);

  return VK_SUCCESS;
}

VkResult setupCommandBuffers() {
  VkCommandPoolCreateInfo cpci = {};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.queueFamilyIndex = vk_computeQueueFamilyIndex;
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VK_CHECK_CALL_RET(
      vkCreateCommandPool(vk_device, &cpci, nullptr, &vk_computeCmdPool));

  if (vk_computeQueueFamilyIndex == vk_transferQueueFamilyIndex) {
    vk_transferCmdPool = vk_computeCmdPool;
  } else {
    VkCommandPoolCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = vk_transferQueueFamilyIndex;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_CALL_RET(
        vkCreateCommandPool(vk_device, &cpci, nullptr, &vk_transferCmdPool));
  }

  {
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = vk_computeCmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VK_CHECK_CALL_RET(
        vkAllocateCommandBuffers(vk_device, &cbai, &vk_computeCmdBuffer));
  }

  {
    VkCommandBufferAllocateInfo cbai = {};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = vk_transferCmdPool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 2;
    VK_CHECK_CALL_RET(
        vkAllocateCommandBuffers(vk_device, &cbai, vk_transferCmdBuffers));
  }

  return VK_SUCCESS;
}

VkBuffer createBuffer(size_t size, VkBufferUsageFlags usage) {
  VkBufferCreateInfo bci = {};
  bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bci.size = size;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkBuffer buffer;
  if (vkCreateBuffer(vk_device, &bci, nullptr, &buffer) != VK_SUCCESS) {
    std::cerr << "Could not create buffer!\n";
    return VK_NULL_HANDLE;
  }
  return buffer;
}

VkImage createImage(VkImageType type, VkFormat format, VkExtent3D extent,
                    VkImageUsageFlags usage, size_t mipLevels,
                    bool exportable = true) {
  VkImageCreateInfo ici = {};
  ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  ici.imageType = type;
  ici.format = format;
  ici.extent = extent;
  ici.mipLevels = mipLevels;
  ici.arrayLayers = 1;
  // ici.tiling = VK_IMAGE_TILING_LINEAR;
  ici.usage = usage;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  // ici.initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;

  VkExternalMemoryImageCreateInfo emici = {};
  if (exportable) {
    emici.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    emici.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    ici.pNext = &emici;
  }

  VkImage image;
  if (vkCreateImage(vk_device, &ici, nullptr, &image)) {
    std::cerr << "Could not create image!\n";
    return VK_NULL_HANDLE;
  }
  return image;
}

VkDeviceMemory allocateDeviceMemory(size_t size, uint32_t memoryTypeIndex,
                                    bool exportable = true) {
  VkMemoryAllocateInfo mai = {};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = size;
  mai.memoryTypeIndex = memoryTypeIndex;

  VkExportMemoryAllocateInfo emai = {};
  if (exportable) {
    emai.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    emai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    mai.pNext = &emai;
  }

  VkDeviceMemory memory;
  if (vkAllocateMemory(vk_device, &mai, nullptr, &memory) != VK_SUCCESS) {
    std::cerr << "Could not allocate device memory!\n";
    return VK_NULL_HANDLE;
  }
  return memory;
}

uint32_t getImageMemoryTypeIndex(VkImage image, VkMemoryPropertyFlags flags,
                                 VkMemoryRequirements &memRequirements) {
  vkGetImageMemoryRequirements(vk_device, image, &memRequirements);

  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memRequirements.memoryTypeBits & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & flags) == flags) {
      return i;
    }
  }
  std::cerr << "Image memory type index not found!\n";
  return 0;
}

uint32_t getBufferMemoryTypeIndex(VkBuffer buffer,
                                  VkMemoryPropertyFlags flags) {
  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(vk_device, buffer, &memRequirements);

  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((memRequirements.memoryTypeBits & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & flags) == flags) {
      return i;
    }
  }
  std::cerr << "Buffer memory type index not found!\n";
  return 0;
}

VkResult cleanup() {

  if (vk_computeQueueFamilyIndex == vk_transferQueueFamilyIndex) {
    vkDestroyCommandPool(vk_device, vk_computeCmdPool, nullptr);
  } else {
    vkDestroyCommandPool(vk_device, vk_computeCmdPool, nullptr);
    vkDestroyCommandPool(vk_device, vk_transferCmdPool, nullptr);
  }

  auto destroyDebugUtilsMessenger =
      (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
          vk_instance, "vkDestroyDebugUtilsMessengerEXT");
  if (destroyDebugUtilsMessenger != nullptr) {
    destroyDebugUtilsMessenger(vk_instance, vk_debug_messenger, nullptr);
  }
  vkDestroyDevice(vk_device, nullptr);
  vkDestroyInstance(vk_instance, nullptr);
  return VK_SUCCESS;
}

int getMemoryOpaqueFD(VkDeviceMemory memory) {
  VkMemoryGetFdInfoKHR mgfi = {};
  mgfi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  mgfi.memory = memory;
  mgfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  int fd = 0;
  if (vk_getMemoryFdKHR != nullptr) {
    VK_CHECK_CALL(vk_getMemoryFdKHR(vk_device, &mgfi, &fd));
  }
  return fd;
}

int getSemaphoreOpaqueFD(VkSemaphore semaphore) {
  VkSemaphoreGetFdInfoKHR sgfi = {};
  sgfi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  sgfi.semaphore = semaphore;
  sgfi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  int fd = 0;
  if (vk_getSemaphoreFdKHR != nullptr) {
    VK_CHECK_CALL(vk_getSemaphoreFdKHR(vk_device, &sgfi, &fd));
  }
  return fd;
}

auto createImageMemoryBarrier(VkImage &img, size_t mipLevels) {
  VkImageMemoryBarrier barrierInput = {};
  barrierInput.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrierInput.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  barrierInput.newLayout = VK_IMAGE_LAYOUT_GENERAL;
  barrierInput.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrierInput.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrierInput.image = img;
  barrierInput.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  barrierInput.subresourceRange.levelCount = mipLevels;
  barrierInput.subresourceRange.layerCount = 1;
  barrierInput.srcAccessMask = 0;
  barrierInput.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  return barrierInput;
}

struct vulkan_image_test_resources_t {
  VkImage vkImage;
  VkDeviceMemory imageMemory;
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingMemory;

  vulkan_image_test_resources_t(VkImageType imgType, VkFormat format,
                                VkExtent3D ext, const size_t imageSizeBytes) {
    vkImage = vkutil::createImage(imgType, format, ext,
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                      VK_IMAGE_USAGE_STORAGE_BIT,
                                  1);
    VkMemoryRequirements memRequirements;
    auto inputImageMemoryTypeIndex = vkutil::getImageMemoryTypeIndex(
        vkImage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memRequirements);
    imageMemory =
        vkutil::allocateDeviceMemory(imageSizeBytes, inputImageMemoryTypeIndex);
    VK_CHECK_CALL(
        vkBindImageMemory(vk_device, vkImage, imageMemory, 0 /*memoryOffset*/));

    stagingBuffer = vkutil::createBuffer(imageSizeBytes,
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto inputStagingMemoryTypeIndex = vkutil::getBufferMemoryTypeIndex(
        stagingBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    stagingMemory = vkutil::allocateDeviceMemory(
        imageSizeBytes, inputStagingMemoryTypeIndex, false /*exportable*/);
    VK_CHECK_CALL(vkBindBufferMemory(vk_device, stagingBuffer, stagingMemory,
                                     0 /*memoryOffset*/));
  }

  ~vulkan_image_test_resources_t() {
    vkDestroyBuffer(vk_device, stagingBuffer, nullptr);
    vkDestroyImage(vk_device, vkImage, nullptr);
    vkFreeMemory(vk_device, stagingMemory, nullptr);
    vkFreeMemory(vk_device, imageMemory, nullptr);
  }
};

VkFormat to_vulkan_format(sycl::image_channel_order order,
                          sycl::image_channel_type channel_type) {
  if (channel_type == sycl::image_channel_type::signed_int8) {

    switch (order) {
    case sycl::image_channel_order::r:
      return VK_FORMAT_R8_SINT;
    case sycl::image_channel_order::rg:
      return VK_FORMAT_R8G8_SINT;
    case sycl::image_channel_order::rgba:
      return VK_FORMAT_R8G8B8A8_SINT;
    default: {
      std::cerr << "error in converting to vulkan format\n";
      exit(-1);
    }
    }
  } else if (channel_type == sycl::image_channel_type::unsigned_int32) {

    switch (order) {
    case sycl::image_channel_order::r:
      return VK_FORMAT_R32_UINT;
    case sycl::image_channel_order::rg:
      return VK_FORMAT_R32G32_UINT;
    case sycl::image_channel_order::rgba:
      return VK_FORMAT_R32G32B32A32_UINT;
    default: {
      std::cerr << "error in converting to vulkan format\n";
      exit(-1);
    }
    }
  } else if (channel_type == sycl::image_channel_type::signed_int32) {
    switch (order) {
    case sycl::image_channel_order::r:
      return VK_FORMAT_R32_SINT;
    case sycl::image_channel_order::rg:
      return VK_FORMAT_R32G32_SINT;
    case sycl::image_channel_order::rgba:
      return VK_FORMAT_R32G32B32A32_SINT;
    default: {
      std::cerr << "error in converting to vulkan format\n";
      exit(-1);
    }
    }
  } else if (channel_type == sycl::image_channel_type::signed_int16) {
    switch (order) {
    case sycl::image_channel_order::r:
      return VK_FORMAT_R16_SINT;
    case sycl::image_channel_order::rg:
      return VK_FORMAT_R16G16_SINT;
    case sycl::image_channel_order::rgba:
      return VK_FORMAT_R16G16B16A16_SINT;
    default: {
      std::cerr << "error in converting to vulkan format\n";
      exit(-1);
    }
    }
  } else if (channel_type == sycl::image_channel_type::fp32) {
    switch (order) {
    case sycl::image_channel_order::r:
      return VK_FORMAT_R32_SFLOAT;
    case sycl::image_channel_order::rg:
      return VK_FORMAT_R32G32_SFLOAT;
    case sycl::image_channel_order::rgba:
      return VK_FORMAT_R32G32B32A32_SFLOAT;
    default: {
      std::cerr << "error in converting to vulkan format\n";
      exit(-1);
    }
    }
  } else {
    std::cerr
        << "error in converting to vulkan format - channel type not included\n";
    exit(-1);
  }
}

} // namespace vkutil

namespace util {

template <typename DType>
bool is_equal(DType lhs, DType rhs, float epsilon = 0.0001f) {
  if constexpr (std::is_floating_point_v<DType>) {
    return (std::abs(lhs - rhs) < epsilon);
  } else {
    return lhs == rhs;
  }
}

} // namespace util
