#pragma once

#include <atomic>
#include <cstdint>
#include <unordered_set>

#include <OffloadAPI.h>

#include "logger/ur_logger.hpp"

struct ur_adapter_handle_t_ {
  std::atomic_uint32_t RefCount = 0;
  logger::Logger &Logger = logger::get_logger("offload");
  ol_device_handle_t HostDevice = nullptr;
  std::unordered_set<ol_platform_handle_t> Platforms;

  ur_result_t init();
};

extern ur_adapter_handle_t_ Adapter;
