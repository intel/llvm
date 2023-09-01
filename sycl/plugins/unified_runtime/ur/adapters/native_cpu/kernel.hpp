//===--------------- kernel.hpp - Native CPU Adapter ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"
#include <sycl/detail/native_cpu.hpp>
#include <ur_api.h>

using nativecpu_kernel_t = void(const sycl::detail::NativeCPUArgDesc *,
                                __nativecpu_state *);
using nativecpu_ptr_t = nativecpu_kernel_t *;
using nativecpu_task_t = std::function<nativecpu_kernel_t>;

struct ur_kernel_handle_t_ : RefCounted {

  ur_kernel_handle_t_(const char *name, nativecpu_task_t subhandler)
      : _name{name}, _subhandler{subhandler} {}

  const char *_name;
  nativecpu_task_t _subhandler;
  std::vector<sycl::detail::NativeCPUArgDesc> _args;
};
