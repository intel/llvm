#pragma once

#include "common.hpp"
#include <ur_api.h>
#include <OffloadAPI.h>
#include <vector>

struct ur_platform_handle_t_ : ur::offload::handle_base {
  ur_platform_handle_t_(ol_platform_handle_t OffloadPlatform) : handle_base(), OffloadPlatform(OffloadPlatform) {};

  ol_platform_handle_t OffloadPlatform;
  std::vector<ur_device_handle_t_> Devices;
};
