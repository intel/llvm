//==-------- ipc_memory_impl.hpp --- SYCL ipc_memory implementation --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>

#include <memory>

namespace sycl {
inline namespace _V1 {
namespace detail {

class ipc_memory_impl {
  struct private_tag {
    explicit private_tag() = default;
  };

public:
  ipc_memory_impl(void *Ptr, const sycl::context &Ctx, private_tag)
      : MContext{getSyclObjImpl(Ctx)}, MPtr{Ptr} {
    MContext->getAdapter().call<UrApiKind::urIPCGetMemHandleExp>(
        MContext->getHandleRef(), Ptr, &MUrHandle.emplace(nullptr));
  }

  ipc_memory_impl(const ipc_memory_impl &) = delete;
  ipc_memory_impl(ipc_memory_impl &&) = default;

  ~ipc_memory_impl() {
    try {
      if (MUrHandle)
        MContext->getAdapter().call_nocheck<UrApiKind::urIPCPutMemHandleExp>(
            MContext->getHandleRef(), *MUrHandle, /*putBackendResource=*/false);
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~ipc_memory_impl", e);
    }
  }

  void put() {
    if (!MUrHandle)
      throw sycl::exception(make_error_code(errc::invalid),
                            "IPC memory object has already been put back.");
    MContext->getAdapter().call_nocheck<UrApiKind::urIPCPutMemHandleExp>(
        MContext->getHandleRef(), *MUrHandle, /*putBackendResource=*/true);
    MUrHandle = std::nullopt;
  }

  ipc_memory_impl &operator=(const ipc_memory_impl &) = delete;
  ipc_memory_impl &operator=(ipc_memory_impl &&) = default;

  template <typename... Ts>
  static std::shared_ptr<ipc_memory_impl> create(Ts &&...args) {
    return std::make_shared<ipc_memory_impl>(std::forward<Ts>(args)...,
                                             private_tag{});
  }

  sycl::ext::oneapi::experimental::ipc_memory_handle_data_t
  get_handle_data() const {
    if (!MUrHandle)
      throw sycl::exception(make_error_code(errc::invalid),
                            "IPC memory object has been put back and the "
                            "handle data cannot be accessed.");

    void *HandleDataPtr = nullptr;
    size_t HandleDataSize = 0;
    MContext->getAdapter().call<UrApiKind::urIPCGetMemHandleDataExp>(
        MContext->getHandleRef(), *MUrHandle, &HandleDataPtr, &HandleDataSize);
    return sycl::span<char, sycl::dynamic_extent>{
        reinterpret_cast<char *>(HandleDataPtr), HandleDataSize};
  }

  void *get_ptr() const { return MPtr; }

private:
  std::shared_ptr<context_impl> MContext;
  void *MPtr = nullptr;
  std::optional<ur_exp_ipc_mem_handle_t> MUrHandle = std::nullopt;
};

} // namespace detail
} // namespace _V1
} // namespace sycl