#pragma once

#include <iostream>
#include <stdexcept>
#include <string>

#include <sycl/aspects.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/intel/info/device.hpp>

#include "vulkan_setup.hpp"

inline void log_debug(const std::string &message) {
#ifdef VERBOSE_PRINT
  std::cout << message << "\n";
#endif
}

inline void log_info(const std::string &message) {
  std::cout << message << "\n";
}

inline void log_error(const std::string &message) {
  std::cerr << message << "\n";
}

inline VulkanContext createSyclVulkanContext(const sycl::device &SyclDevice) {
  if (!SyclDevice.has(sycl::aspect::ext_intel_device_info_uuid))
    throw std::runtime_error("SYCL device UUID is unavailable!");

  return createVulkanContext(
      SyclDevice.get_info<sycl::ext::intel::info::device::uuid>());
}

inline VulkanContext createSyclVulkanContext() {
  return createSyclVulkanContext(sycl::device{});
}

// RAII helper that tears down a VulkanContext when it goes out of scope.
struct VulkanContextGuard {
  VulkanContext &context;
  ~VulkanContextGuard() { cleanupVulkanContext(context); }
};
