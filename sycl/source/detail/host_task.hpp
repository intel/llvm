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
#include <sycl/detail/host_profiling_info.hpp>
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
  HostTask(std::function<void()> &&Func) : MHostTask(std::move(Func)) {}
  HostTask(std::function<void(interop_handle)> &&Func)
      : MInteropTask(std::move(Func)) {}

  bool isInteropTask() const { return !!MInteropTask; }

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
  friend class sycl::detail::HandlerAccess;
};

inline std::function<void()> HandlerAccess::getHostTaskFunc(HostTask &HT) {
  return std::move(HT.MHostTask);
}

struct EnqueueHostTaskData {
  explicit EnqueueHostTaskData(std::function<void()> HostTask)
      : Func(std::move(HostTask)) {}

  std::function<void()> Func;
};

template <bool OwnsData> inline void NativeHostTask(void *Data) {
  auto *HostTaskData = static_cast<EnqueueHostTaskData *>(Data);
  if constexpr (OwnsData) {
    // so it's freed if the user function throws
    std::unique_ptr<EnqueueHostTaskData> Owner(HostTaskData);
    Owner->Func();
  } else {
    HostTaskData->Func();
  }
}

} // namespace detail
} // namespace _V1
} // namespace sycl
