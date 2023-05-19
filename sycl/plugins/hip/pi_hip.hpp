//===-- pi_hip.hpp - HIP Plugin -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \defgroup sycl_pi_hip HIP Plugin
/// \ingroup sycl_pi

/// \file pi_hip.hpp
/// Declarations for HIP Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying HIP runtime.
///
/// \ingroup sycl_pi_hip

#ifndef PI_HIP_HPP
#define PI_HIP_HPP

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_HIP_PLUGIN_VERSION 1

#define _PI_HIP_PLUGIN_VERSION_STRING                                          \
  _PI_PLUGIN_VERSION_STRING(_PI_HIP_PLUGIN_VERSION)

#include "sycl/detail/pi.h"
#include <array>
#include <atomic>
#include <cassert>
#include <cstring>
#include <functional>
#include <hip/hip_runtime.h>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdint.h>
#include <string>
#include <vector>

#include <ur/adapters/hip/context.hpp>
#include <ur/adapters/hip/device.hpp>
#include <ur/adapters/hip/event.hpp>
#include <ur/adapters/hip/kernel.hpp>
#include <ur/adapters/hip/memory.hpp>
#include <ur/adapters/hip/platform.hpp>
#include <ur/adapters/hip/program.hpp>
#include <ur/adapters/hip/queue.hpp>
#include <ur/adapters/hip/sampler.hpp>

#include "pi2ur.hpp"

using _pi_stream_guard = std::unique_lock<std::mutex>;

struct _pi_platform : ur_platform_handle_t_ {
  using ur_platform_handle_t_::ur_platform_handle_t_;
};

struct _pi_device : ur_device_handle_t_ {
  using ur_device_handle_t_::ur_device_handle_t_;
};

struct _pi_context : ur_context_handle_t_ {
  using ur_context_handle_t_::ur_context_handle_t_;
};

struct _pi_mem : ur_mem_handle_t_ {
  using ur_mem_handle_t_::ur_mem_handle_t_;
};

struct _pi_queue : ur_queue_handle_t_ {
  using ur_queue_handle_t_::ur_queue_handle_t_;
};

struct _pi_event : ur_event_handle_t_ {
  using ur_event_handle_t_::ur_event_handle_t_;
};

struct _pi_program : ur_program_handle_t_ {
  using ur_program_handle_t_::ur_program_handle_t_;
};

struct _pi_kernel : ur_kernel_handle_t_ {
  using ur_kernel_handle_t_::ur_kernel_handle_t_;
};

struct _pi_sampler : ur_sampler_handle_t_ {
  using ur_sampler_handle_t_::ur_sampler_handle_t_;
};

#endif // PI_HIP_HPP
