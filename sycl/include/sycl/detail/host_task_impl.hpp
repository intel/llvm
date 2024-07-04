//==---- host_task_impl.hpp ------------------------------------------------==//
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

#include <sycl/detail/cg.hpp>
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

class CGHostTask : public CG {
public:
  std::unique_ptr<HostTask> MHostTask;
  // queue for host-interop task
  std::shared_ptr<detail::queue_impl> MQueue;
  // context for host-interop task
  std::shared_ptr<detail::context_impl> MContext;
  std::vector<ArgDesc> MArgs;

  CGHostTask(std::unique_ptr<HostTask> HostTask,
             std::shared_ptr<detail::queue_impl> Queue,
             std::shared_ptr<detail::context_impl> Context,
             std::vector<ArgDesc> Args, CG::StorageInitHelper CGData,
             CGTYPE Type, detail::code_location loc = {})
      : CG(Type, std::move(CGData), std::move(loc)),
        MHostTask(std::move(HostTask)), MQueue(Queue), MContext(Context),
        MArgs(std::move(Args)) {}
};

} // namespace detail
template <typename FuncT>
std::enable_if_t<
    detail::check_fn_signature<std::remove_reference_t<FuncT>, void()>::value ||
    detail::check_fn_signature<std::remove_reference_t<FuncT>,
                               void(interop_handle)>::value>
handler::host_task_impl(FuncT &&Func) {
  throwIfActionIsCreated();

  MNDRDesc.set(range<1>(1));
  // Need to copy these rather than move so that we can check associated
  // accessors during finalize
  MArgs = MAssociatedAccesors;

  MHostTask.reset(new detail::HostTask(std::move(Func)));

  setType(detail::CG::CodeplayHostTask);
}

} // namespace _V1
} // namespace sycl
