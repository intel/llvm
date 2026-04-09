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
#include <detail/global_handler.hpp>
#include <sycl/detail/cg_types.hpp>
#include <sycl/handler.hpp>
#include <sycl/interop_handle.hpp>

namespace sycl {
inline namespace _V1 {
class interop_handle;
namespace detail {
class HostTask {
  enum class HostTaskOrigin {
    SYCLCoreAPI,
    ExtEnqueueFunctionsAPI,
  };

  std::function<void()> MHostTask;
  std::function<void(interop_handle)> MInteropTask;
  HostTaskOrigin MOrigin;

public:
  HostTask() : MHostTask([]() {}), MOrigin(HostTaskOrigin::SYCLCoreAPI) {}
  HostTask(std::function<void()> &&Func,
           bool IsFromExtEnqueueFunctionsAPI = false)
      : MHostTask(std::move(Func)),
        MOrigin(IsFromExtEnqueueFunctionsAPI
                    ? HostTaskOrigin::ExtEnqueueFunctionsAPI
                    : HostTaskOrigin::SYCLCoreAPI) {}
  HostTask(std::function<void(interop_handle)> &&Func)
      : MInteropTask(std::move(Func)), MOrigin(HostTaskOrigin::SYCLCoreAPI) {}

  bool isInteropTask() const { return !!MInteropTask; }

  bool isCreatedFromEnqueueFunction() const {
    return MOrigin == HostTaskOrigin::ExtEnqueueFunctionsAPI;
  }

  void call(HostProfilingInfo *HPI) {
    if (!GlobalHandler::instance().isOkToDefer()) {
      return;
    }

    if (HPI)
      HPI->start();
    MHostTask();
    if (HPI)
      HPI->end();
  }

  void call(HostProfilingInfo *HPI, interop_handle handle) {
    if (!GlobalHandler::instance().isOkToDefer()) {
      return;
    }

    if (HPI)
      HPI->start();
    MInteropTask(handle);
    if (HPI)
      HPI->end();
  }

  friend class DispatchHostTask;
  friend class ExecCGCommand;
};

} // namespace detail
} // namespace _V1
} // namespace sycl
