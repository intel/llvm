//===------ ur_bindings.hpp - Complete definitions of UR handles -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
#pragma once

#include "pi_level_zero.hpp"
#include <ur_api.h>

// Make the Unified Runtime handles definition complete.
// This is used in various "create" API where new handles are allocated.
struct ur_platform_handle_t_ : public _pi_platform {
  using _pi_platform::_pi_platform;
};

struct ur_device_handle_t_ : public _pi_device {
  using _pi_device::_pi_device;
};

struct ur_context_handle_t_ : public _pi_context {
  using _pi_context::_pi_context;
};

struct ur_event_handle_t_ : public _pi_event {
  using _pi_event::_pi_event;
};

struct ur_program_handle_t_ : public _pi_program {
  using _pi_program::_pi_program;
};

struct ur_kernel_handle_t_ : public _pi_kernel {
  using _pi_kernel::_pi_kernel;
};

struct ur_queue_handle_t_ : public _pi_queue {
  using _pi_queue::_pi_queue;
};

struct ur_sampler_handle_t_ : public _pi_sampler {
  using _pi_sampler::_pi_sampler;
};

struct ur_mem_handle_t_ : public _pi_mem {
  using _pi_mem::_pi_mem;
};
