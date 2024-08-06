#pragma once

#include <atomic>
#include <unordered_map>
#include <ur_api.h>
#include <OffloadAPI.h>

struct ur_context_handle_t_ {
  ur_context_handle_t_(ur_device_handle_t hDevice) : Device{hDevice} {
    urDeviceRetain(Device);
  }
  ~ur_context_handle_t_() {
    urDeviceRelease(Device);
  }

  ur_device_handle_t Device;
  std::atomic_uint32_t RefCount;
  std::unordered_map<void*, ol_alloc_type_t> AllocTypeMap;
};
