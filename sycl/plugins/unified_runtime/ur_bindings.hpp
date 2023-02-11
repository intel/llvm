//===------ ur_bindings.hpp - Complete definitions of UR handles -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------===//
#pragma once

#include <ur/adapters/level_zero/ur_level_zero.hpp>
#include <ur_api.h>

// Make the Unified Runtime handles definition complete.
// This is used in various "create" API where new handles are allocated.
struct ur_platform_handle_t_ : public _ur_platform_handle_t {
  using _ur_platform_handle_t::_ur_platform_handle_t;
};
struct ur_device_handle_t_ : public _ur_device_handle_t {
  using _ur_device_handle_t::_ur_device_handle_t;
};

struct ur_context_handle_t_ : public _ur_context_handle_t {
  using _ur_context_handle_t::_ur_context_handle_t;
};

struct ur_event_handle_t_ : public _ur_event_handle_t {
  using _ur_event_handle_t::_ur_event_handle_t;
};

struct ur_program_handle_t_ : public _ur_program_handle_t {
  using _ur_program_handle_t::_ur_program_handle_t;
};

struct ur_module_handle_t_ : public _ur_module_handle_t {
  using _ur_module_handle_t::_ur_module_handle_t;
};

struct ur_kernel_handle_t_ : public _ur_kernel_handle_t {
  using _ur_kernel_handle_t::_ur_kernel_handle_t;
};

struct ur_queue_handle_t_ : public _ur_queue_handle_t {
  using _ur_queue_handle_t::_ur_queue_handle_t;
};

struct ur_native_handle_t_ : public _ur_native_handle_t {
  using _ur_native_handle_t::_ur_native_handle_t;
};

struct ur_sampler_handle_t_ : public _ur_sampler_handle_t {
  using _ur_sampler_handle_t::_ur_sampler_handle_t;
};

struct ur_mem_handle_t_ : public _ur_mem_handle_t {
  using _ur_mem_handle_t::_ur_mem_handle_t;
};
