#pragma once

#include <ur_api.h>
#include <OffloadAPI.h>

#include "common.hpp"

struct ur_event_handle_t_ : RefCounted {
 ol_event_handle_t OffloadEvent;
};
