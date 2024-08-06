#pragma once

#include <ur_api.h>
#include <OffloadAPI.h>

#include "common.hpp"

struct ur_program_handle_t_ : RefCounted {
 ol_program_handle_t OffloadProgram;
};
