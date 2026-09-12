#pragma once

#include <stdexcept>

#include <sycl/aspects.hpp>
#include <sycl/device.hpp>
#include <sycl/ext/intel/info/device.hpp>

#include "vulkan_setup.hpp"

inline VulkanContext createSyclVulkanContext(const sycl::device &SyclDevice) {
  if (!SyclDevice.has(sycl::aspect::ext_intel_device_info_uuid))
    throw std::runtime_error("SYCL device UUID is unavailable!");

  return createVulkanContext(
      SyclDevice.get_info<sycl::ext::intel::info::device::uuid>());
}

inline VulkanContext createSyclVulkanContext() {
  return createSyclVulkanContext(sycl::device{});
}
