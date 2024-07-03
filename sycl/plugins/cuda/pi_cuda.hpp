//===-- pi_cuda.hpp - CUDA Plugin -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \defgroup sycl_pi_cuda CUDA Plugin
/// \ingroup sycl_pi

/// \file pi_cuda.hpp
/// Declarations for CUDA Plugin. It is the interface between the
/// device-agnostic SYCL runtime layer and underlying CUDA runtime.
///
/// \ingroup sycl_pi_cuda

#ifndef PI_CUDA_HPP
#define PI_CUDA_HPP

// This version should be incremented for any change made to this file or its
// corresponding .cpp file.
#define _PI_CUDA_PLUGIN_VERSION 1

#define _PI_CUDA_PLUGIN_VERSION_STRING                                         \
  _PI_PLUGIN_VERSION_STRING(_PI_CUDA_PLUGIN_VERSION)

#include <adapters/cuda/command_buffer.hpp>
#include <adapters/cuda/context.hpp>
#include <adapters/cuda/device.hpp>
#include <adapters/cuda/event.hpp>
#include <adapters/cuda/kernel.hpp>
#include <adapters/cuda/memory.hpp>
#include <adapters/cuda/physical_mem.hpp>
#include <adapters/cuda/platform.hpp>
#include <adapters/cuda/program.hpp>
#include <adapters/cuda/queue.hpp>
#include <adapters/cuda/sampler.hpp>

// Share code between the PI Plugin and UR Adapter
#include <pi2ur.hpp>

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

struct _pi_ext_command_buffer : ur_exp_command_buffer_handle_t_ {
  using ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_;
};

struct _pi_physical_mem : ur_physical_mem_handle_t_ {
  using ur_physical_mem_handle_t_::ur_physical_mem_handle_t_;
};

#endif // PI_CUDA_HPP
