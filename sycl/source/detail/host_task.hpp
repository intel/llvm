//==---- host_task.hpp -----------------------------------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implementation of various classes/methods related to host task support so
// that it could be excluded from <sycl/detail/core.hpp> as such support pulls
// in interop->image->vec/marray dependencies.

#pragma once

#include <detail/cg.hpp>
#include <sycl/detail/cg_types.hpp>
#include <sycl/handler.hpp>
#include <sycl/interop_handle.hpp>

namespace sycl {
inline namespace _V1 {
class interop_handle;
namespace detail {
class HostTask {
  std::function<void()> MHostTask;
  std::function<void(interop_handle)> MInteropTask;

public:
  HostTask() : MHostTask([]() {}) {}
  HostTask(std::function<void()> &&Func) : MHostTask(Func) {}
  HostTask(std::function<void(interop_handle)> &&Func) : MInteropTask(Func) {}

  bool isInteropTask() const { return !!MInteropTask; }

  void call(HostProfilingInfo *HPI) {
    if (HPI)
      HPI->start();
    MHostTask();
    if (HPI)
      HPI->end();
  }

  void call(HostProfilingInfo *HPI, interop_handle handle) {
    if (HPI)
      HPI->start();
    MInteropTask(handle);
    if (HPI)
      HPI->end();
  }

  friend class DispatchHostTask;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
