//===--------- ur_level_zero_common.hpp - Level Zero Adapter ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <sycl/detail/pi.h>
#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>
#include <zes_api.h>

#include "ur_level_zero_context.hpp"
#include "ur_level_zero_device.hpp"
#include "ur_level_zero_event.hpp"
#include "ur_level_zero_mem.hpp"
#include "ur_level_zero_module.hpp"
#include "ur_level_zero_platform.hpp"
#include "ur_level_zero_program.hpp"
#include "ur_level_zero_queue.hpp"
#include "ur_level_zero_sampler.hpp"

// Map Level Zero runtime error code to UR error code.
ur_result_t ze2urResult(ze_result_t ZeResult);
