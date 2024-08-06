#pragma once

#include <ur_api.h>
#include <OffloadAPI.h>

#include "common.hpp"

struct ur_queue_handle_t_ : RefCounted {
 ol_queue_handle_t OffloadQueue;
 ol_device_handle_t OffloadDevice;
};
