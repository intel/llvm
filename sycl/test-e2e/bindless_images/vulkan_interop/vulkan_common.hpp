#pragma once

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <sycl/ext/oneapi/bindless_images.hpp>
#include <vector>

void printString(std::string str) {
#ifdef VERBOSE_PRINT
  std::cout << str << std::endl;
#endif
}

#define VK_CHECK_CALL_RET(call)                                                \
  {                                                                            \
    VkResult err = call;                                                       \
    if (err != VK_SUCCESS) {                                                   \
      std::cerr << #call << " failed. Code: " << err << "\n";                  \
      return err;                                                              \
    }                                                                          \
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

#ifdef _WIN32
static PFN_vkGetMemoryWin32HandleKHR vk_GetMemoryWin32HandleKHR;
static PFN_vkGetSemaphoreWin32HandleKHR vk_getSemaphoreWin32HandleKHR;
#else
static PFN_vkGetMemoryFdKHR vk_getMemoryFdKHR;
static PFN_vkGetSemaphoreFdKHR vk_getSemaphoreFdKHR;
#endif

static uint32_t vk_computeQueueFamilyIndex;
static uint32_t vk_transferQueueFamilyIndex;

static VkCommandPool vk_computeCmdPool;
static VkCommandPool vk_transferCmdPool;

static VkCommandBuffer vk_computeCmdBuffer;
static VkCommandBuffer vk_transferCmdBuffers[2];

// A static debug callback function that relays messages from the Vulkan
// validation layer to the terminal.
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
              void *pUserData) {
  // Only print errors from validation layer
  if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    std::cerr << pCallbackData->pMessage << "\n\n";
  }
  return VK_FALSE;
}

namespace vkutil {

// Returns all supported Vulkan instance extensions.
VkResult
getSupportedInstanceExtensions(std::vector<std::string> &supportedExtensions) {
  uint32_t count = 0;
  VK_CHECK_CALL_RET(
      vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));

  std::vector<VkExtensionProperties> extensionProperties(count);

  VK_CHECK_CALL_RET(vkEnumerateInstanceExtensionProperties(
      nullptr, &count, extensionProperties.data()));

  for (auto &extension : extensionProperties) {
    supportedExtensions.push_back(extension.extensionName);
  }

  return VK_SUCCESS;
}

/*
In this function we set up the Vulkan instance, which is the one of the first
steps in setting up a Vulkan application.
When creating an instance we need to specify some information about our
application, most importantly, we need to specify some extensions that we
require to perform interop operations.
*/
VkResult setupInstance() {
  // Generic application information. The specific values are not important to
  // the execution of the Vulkan program.
  VkApplicationInfo ai = {};
  ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  ai.pApplicationName = "SYCL-Vulkan-Interop";
  ai.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  ai.pEngineName = "";
  ai.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  ai.apiVersion = VK_API_VERSION_1_0;

  // Query the number of available layers and retrieve their names. One example
  // of a layer is the validation layer, this layer allows for runtime debug
  // messages to be returned if anything goes wrong in the Vulkan application.
  // We will set up a callback function to print debug information if the
  // validation layer is available.
  uint32_t layerCount;
  VK_CHECK_CALL_RET(vkEnumerateInstanceLayerProperties(&layerCount, nullptr));

  std::vector<VkLayerProperties> availableLayers(layerCount);
  VK_CHECK_CALL_RET(
      vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data()));

  // Query the supported instance extensions.
  std::vector<std::string> supportedInstanceExtensions;
  VK_CHECK_CALL_RET(
      getSupportedInstanceExtensions(supportedInstanceExtensions));

  // We have some instance extensions that we require for the tests to function.
  std::vector<const char *> requiredInstanceExtensions = {
      VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
      VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME};

  // Make sure that our required instance extensions are supported by the
  // running Vulkan instance.
  for (int i = 0; i < requiredInstanceExtensions.size(); ++i) {
    std::string requiredExtension = requiredInstanceExtensions[i];
    if (std::find(supportedInstanceExtensions.begin(),
                  supportedInstanceExtensions.end(),
                  requiredExtension) == supportedInstanceExtensions.end())
      return VK_ERROR_EXTENSION_NOT_PRESENT;
  }

  // Create the vulkan instance with our required extensions and layers.
  VkInstanceCreateInfo ci = {};
  ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ci.pApplicationInfo = &ai;
  ci.enabledExtensionCount = requiredInstanceExtensions.size();
  ci.ppEnabledExtensionNames = requiredInstanceExtensions.data();
  std::vector<const char *> layers;
  if (std::any_of(availableLayers.begin(), availableLayers.end(),
                  [](auto &layer) {
                    return layer.layerName == "VK_LAYER_KHRONOS_validation";
                  })) {
    layers.push_back("VK_LAYER_KHRONOS_validation");
  }
  ci.enabledLayerCount = layers.size();
  ci.ppEnabledLayerNames = layers.data();

  VK_CHECK_CALL_RET(vkCreateInstance(&ci, nullptr, &vk_instance));

  // Create a debug utils messenger. This will allow us to print debug
  // information from the Vulkan validation layer.
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

  return VK_SUCCESS;
}

// Returns all supported Vulkan device extensions.
VkResult
getSupportedDeviceExtensions(std::vector<VkExtensionProperties> &extensions,
                             VkPhysicalDevice device) {
  uint32_t numExtensions = 0;

  VK_CHECK_CALL_RET(vkEnumerateDeviceExtensionProperties(
      device, nullptr, &numExtensions, nullptr));

  extensions.resize(numExtensions);
  VK_CHECK_CALL_RET(vkEnumerateDeviceExtensionProperties(
      device, nullptr, &numExtensions, extensions.data()));

  return VK_SUCCESS;
}

// Set up the Vulkan device.
VkResult setupDevice(std::string device) {
  uint32_t physicalDeviceCount = 0;
  // Get all physical devices.
  VK_CHECK_CALL_RET(
      vkEnumeratePhysicalDevices(vk_instance, &physicalDeviceCount, nullptr));
  if (physicalDeviceCount == 0) {
    // If no physical devices found, return error.
    return VK_ERROR_DEVICE_LOST;
  }
  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  VK_CHECK_CALL_RET(vkEnumeratePhysicalDevices(
      vk_instance, &physicalDeviceCount, physicalDevices.data()));

  bool foundDevice = false;

  // Define the required device extensions to run the tests.
  static constexpr std::string_view requiredExtensions[] = {
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN32
      VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
      VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#endif
  };

  // Make lowercase to fix inconsistent capitalization between SYCL and Vulkan
  // device naming.
  std::transform(device.begin(), device.end(), device.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  // From all physical devices, find the first one that supports all our
  // required device extensions.
  for (int i = 0; i < physicalDeviceCount; i++) {
    vk_physical_device = physicalDevices[i];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(vk_physical_device, &props);
    std::string name(props.deviceName);

    // Make lowercase for comparision.
    std::transform(name.begin(), name.end(), name.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (name.find(device) == std::string::npos) {
      continue;
    }

    std::vector<VkExtensionProperties> supportedDeviceExtensions;
    getSupportedDeviceExtensions(supportedDeviceExtensions, vk_physical_device);
    const bool hasRequiredExtensions = std::all_of(
        std::begin(requiredExtensions), std::end(requiredExtensions),
        [&](std::string_view requiredExt) {
          auto it = std::find_if(std::begin(supportedDeviceExtensions),
                                 std::end(supportedDeviceExtensions),
                                 [&](const VkExtensionProperties &ext) {
                                   return (ext.extensionName == requiredExt);
                                 });
          return (it != std::end(supportedDeviceExtensions));
        });
    if (!hasRequiredExtensions) {
      continue;
    }

    foundDevice = true;
    std::cout << "Found suitable Vulkan device: " << props.deviceName
              << std::endl;
    break;
  }

  // If no device was found that supports all our required extensions return an
  // error.
  if (!foundDevice) {
    std::cerr << "Failed to find suitable device!\n";
    return VK_ERROR_DEVICE_LOST;
  }

  // Get queue families and assign queue family indices for compute and transfer
  // queues.
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

  // Populate queue information prior to Vulkan device creation.
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

  // Store our required device extensions. To be passed to the Vulkan device
  // creation function.
  std::vector<const char *> extensions = {
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
#ifdef _WIN32
      VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
#else
      VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
#endif
  };

  // Create the Vulkan device with the above queues, extensions, and layers.
  VkDeviceCreateInfo dci = {};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.pQueueCreateInfos = qcis.data();
  dci.queueCreateInfoCount = qcis.size();
  dci.pEnabledFeatures = &deviceFeatures;
  dci.enabledExtensionCount = extensions.size();
  dci.ppEnabledExtensionNames = extensions.data();

  VK_CHECK_CALL_RET(
      vkCreateDevice(vk_physical_device, &dci, nullptr, &vk_device));

  // Get the Vulkan queues from the device.
  vkGetDeviceQueue(vk_device, vk_transferQueueFamilyIndex, 0,
                   &vk_transfer_queue);
  vkGetDeviceQueue(vk_device, vk_computeQueueFamilyIndex, 0, &vk_compute_queue);

  // Get function pointers for memory and semaphore handle exportation.
  // Functions will depend on the OS being compiled for.
#ifdef _WIN32
  vk_GetMemoryWin32HandleKHR =
      (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(
          vk_device, "vkGetMemoryWin32HandleKHR");
  if (!vk_GetMemoryWin32HandleKHR) {
    std::cerr
        << "Could not get func pointer to \"vkGetMemoryWin32HandleKHR\"!\n";
    return VK_ERROR_UNKNOWN;
  }
  vk_getSemaphoreWin32HandleKHR =
      (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
          vk_device, "vkGetSemaphoreWin32HandleKHR");
  if (!vk_getSemaphoreWin32HandleKHR) {
    std::cerr
        << "Could not get func pointer to \"vkGetSemaphoreWin32HandleKHR\"!\n";
    return VK_ERROR_UNKNOWN;
  }
#else
  vk_getMemoryFdKHR =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(vk_device, "vkGetMemoryFdKHR");
  if (!vk_getMemoryFdKHR) {
    std::cerr << "Could not get func pointer to \"vkGetMemoryFdKHR\"!\n";
    return VK_ERROR_UNKNOWN;
  }
  vk_getSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
      vk_device, "vkGetSemaphoreFdKHR");
  if (!vk_getSemaphoreFdKHR) {
    std::cerr << "Could not get func pointer to \"vkGetSemaphoreFdKHR\"!\n";
    return VK_ERROR_UNKNOWN;
  }
#endif

  return VK_SUCCESS;
}

/*
This function sets up Vulkan command buffers.
Firstly we create command pools for each of the queues that can be used.
We have two queue types which can be used:
  - A transfer queue, used for data movement operations
  - A compute queue, used for shader invocation operations
We allocate command buffers from these command pools.
Note that some Vulkan instances may provide queues with transfer and compute
capabilities. If this is the case, we only create one command pool, and one
command buffer.
*/
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

/*
Create a Vulkan buffer with a specified size and usage.
*/
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

/*
Create a Vulkan image with a specified image type, format, extent, and usage.
This function also allows users to specify whether the image will be exportable,
in which case the appropriate extension struct is populated based on the OS the
program is compiled for.
*/
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
  ici.usage = usage;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;

  VkExternalMemoryImageCreateInfo emici = {};
  if (exportable) {
    emici.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
#ifdef _WIN32
    emici.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    emici.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

    ici.pNext = &emici;
  }

  VkImage image;
  if (vkCreateImage(vk_device, &ici, nullptr, &image)) {
    std::cerr << "Could not create image!\n";
    return VK_NULL_HANDLE;
  }
  return image;
}

/*
Allocate `size` of device memory of the specified memory type.
This function also allows users to specify whether the memory will be
exportable, in which case the appropriate extension struct is populated based on
the OS the program is compiled for.
*/
VkDeviceMemory allocateDeviceMemory(size_t size, uint32_t memoryTypeIndex,
                                    bool exportable = true) {
  VkMemoryAllocateInfo mai = {};
  mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  mai.allocationSize = size;
  mai.memoryTypeIndex = memoryTypeIndex;

  VkExportMemoryAllocateInfo emai = {};
  if (exportable) {
    emai.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
#ifdef _WIN32
    emai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
    emai.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
    mai.pNext = &emai;
  }

  VkDeviceMemory memory;
  if (vkAllocateMemory(vk_device, &mai, nullptr, &memory) != VK_SUCCESS) {
    std::cerr << "Could not allocate device memory!\n";
    return VK_NULL_HANDLE;
  }

  return memory;
}

/*
Retrieve the image memory type index for the Vulkan device based on the memory
property flags passed.
*/
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

/*
Retrieve the buffer memory type index for the Vulkan device based on the memory
property flags passed.
*/
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

/*
Destroy Vulkan objects.
This function is called towards the end of Vulkan program execution.
*/
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

#ifdef _WIN32

/*
Retrieve a win32 memory handle for a given Vulkan device memory allocation.
*/
HANDLE getMemoryWin32Handle(VkDeviceMemory memory) {

  HANDLE retHandle = 0;

  VkMemoryGetWin32HandleInfoKHR mgwhi = {};
  mgwhi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  mgwhi.memory = memory;
  mgwhi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

  if (vk_GetMemoryWin32HandleKHR != nullptr) {
    VK_CHECK_CALL(vk_GetMemoryWin32HandleKHR(vk_device, &mgwhi, &retHandle));
  } else {
    std::cerr << "Could not get win32 handle!\n";
    return 0;
  }

  return retHandle;
}

/*
Retrieve a win32 memory handle for a given Vulkan semaphore object.
*/
HANDLE getSemaphoreWin32Handle(VkSemaphore semaphore) {

  HANDLE retHandle = 0;

  VkSemaphoreGetWin32HandleInfoKHR sghwi = {};
  sghwi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  sghwi.semaphore = semaphore;
  sghwi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

  if (vk_getSemaphoreWin32HandleKHR != nullptr) {
    VK_CHECK_CALL(vk_getSemaphoreWin32HandleKHR(vk_device, &sghwi, &retHandle));
  } else {
    std::cerr << "Could not get semaphore opaque file descriptor!\n";
    return 0;
  }

  return retHandle;
}

#else

/*
Retrieve an opaque file descriptor handle for a given Vulkan memory allocation.
*/
int getMemoryOpaqueFD(VkDeviceMemory memory) {
  VkMemoryGetFdInfoKHR mgfi = {};
  mgfi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  mgfi.memory = memory;
  mgfi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

  int fd = 0;
  if (vk_getMemoryFdKHR != nullptr) {
    VK_CHECK_CALL(vk_getMemoryFdKHR(vk_device, &mgfi, &fd));
  } else {
    std::cerr << "Could not get memory opaque file descriptor!\n";
    return 0;
  }

  return fd;
}

/*
Retrieve an opaque file descriptor handle for a given Vulkan semaphore object.
*/
int getSemaphoreOpaqueFD(VkSemaphore semaphore) {
  VkSemaphoreGetFdInfoKHR sgfi = {};
  sgfi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  sgfi.semaphore = semaphore;
  sgfi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  int fd = 0;
  if (vk_getSemaphoreFdKHR != nullptr) {
    VK_CHECK_CALL(vk_getSemaphoreFdKHR(vk_device, &sgfi, &fd));
  } else {
    std::cerr << "Could not get semaphore opaque file descriptor!\n";
    return 0;
  }

  return fd;
}
#endif

/*
Populate a generic image memory barrier for a specific Vulkan image.
This function assumes we are transitioning from an undefined image layout to a
general image layout, which is sufficient for our current Vulkan tests.
*/
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

/*
This struct contains Vulkan resources used in test files, and is used to
simplify the code within these tests.
The constructor creates images, allocates device memory required for those
images, and binds that memory to the created image.
The destructor cleans up the memory allocations and destroys the image and
staging buffer used to transfer data to that image.
*/
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

/*
Convert a SYCL image channel order and image channel type to a corresponding
Vulkan format.
*/
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
