//===------ pi_native_cpu.hpp - Native CPU Plugin -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pi2ur.hpp>

#include <adapters/native_cpu/context.hpp>
#include <adapters/native_cpu/device.hpp>
#include <adapters/native_cpu/kernel.hpp>
#include <adapters/native_cpu/memory.hpp>
#include <adapters/native_cpu/physical_mem.hpp>
#include <adapters/native_cpu/platform.hpp>
#include <adapters/native_cpu/program.hpp>
#include <adapters/native_cpu/queue.hpp>

struct _pi_context : ur_context_handle_t_ {
  using ur_context_handle_t_::ur_context_handle_t_;
};

struct _pi_device : ur_device_handle_t_ {
  using ur_device_handle_t_::ur_device_handle_t_;
};

struct _pi_kernel : ur_kernel_handle_t_ {
  using ur_kernel_handle_t_::ur_kernel_handle_t_;
};

struct _pi_mem : ur_mem_handle_t_ {
  using ur_mem_handle_t_::ur_mem_handle_t_;
};

struct _pi_platform : ur_platform_handle_t_ {
  using ur_platform_handle_t_::ur_platform_handle_t_;
};

struct _pi_program : ur_program_handle_t_ {
  using ur_program_handle_t_::ur_program_handle_t_;
};

struct _pi_queue : ur_queue_handle_t_ {
  using ur_queue_handle_t_::ur_queue_handle_t_;
};

struct _pi_physical_mem : ur_physical_mem_handle_t_ {
  using ur_physical_mem_handle_t_::ur_physical_mem_handle_t_;
};
