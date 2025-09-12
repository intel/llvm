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
#include <sycl/sycl_span.hpp>

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
      : MRelationship{HandleRelationship::Owner}, MContext{getSyclObjImpl(Ctx)},
        MPtr{Ptr} {
    adapter_impl &Adapter = MContext->getAdapter();
    Adapter.call<UrApiKind::urIPCGetMemHandleExp>(MContext->getHandleRef(), Ptr,
                                                  &MUrHandle);
  }

  ipc_memory_impl(span<const char, sycl::dynamic_extent> IPCMemoryHandleData,
                  const sycl::context &Ctx, const sycl::device &Dev,
                  private_tag)
      : MRelationship{HandleRelationship::Adopted},
        MContext{getSyclObjImpl(Ctx)} {
    adapter_impl &Adapter = MContext->getAdapter();

    // First recreate the IPC handle.
    ur_result_t UrRes =
        Adapter.call_nocheck<UrApiKind::urIPCCreateMemHandleFromDataExp>(
            MContext->getHandleRef(), getSyclObjImpl(Dev)->getHandleRef(),
            IPCMemoryHandleData.data(), IPCMemoryHandleData.size(), &MUrHandle);
    if (UrRes == UR_RESULT_ERROR_INVALID_VALUE)
      throw sycl::exception(sycl::make_error_code(errc::invalid),
                            "IPCMemoryHandleData data size does not correspond "
                            "to the target platform's IPC memory handle size.");
    Adapter.checkUrResult(UrRes);

    // Then open it and retrieve the pointer.
    Adapter.call<UrApiKind::urIPCOpenMemHandleExp>(MContext->getHandleRef(),
                                                   MUrHandle, &MPtr);
  }

  ipc_memory_impl(const ipc_memory_impl &) = delete;
  ipc_memory_impl(ipc_memory_impl &&) = default;

  ~ipc_memory_impl() {
    try {
      adapter_impl &Adapter = MContext->getAdapter();
      if (MRelationship == HandleRelationship::Owner) {
        Adapter.call_nocheck<UrApiKind::urIPCPutMemHandleExp>(
            MContext->getHandleRef(), MUrHandle);
      } else {
        Adapter.call_nocheck<UrApiKind::urIPCCloseMemHandleExp>(
            MContext->getHandleRef(), MPtr);
        Adapter.call_nocheck<UrApiKind::urIPCDestroyMemHandleExp>(
            MContext->getHandleRef(), MUrHandle);
      }
    } catch (std::exception &e) {
      __SYCL_REPORT_EXCEPTION_TO_STREAM("exception in ~ipc_memory_impl", e);
    }
  }

  ipc_memory_impl &operator=(const ipc_memory_impl &) = delete;
  ipc_memory_impl &operator=(ipc_memory_impl &&) = default;

  template <typename... Ts>
  static std::shared_ptr<ipc_memory_impl> create(Ts &&...args) {
    return std::make_shared<ipc_memory_impl>(std::forward<Ts>(args)...,
                                             private_tag{});
  }

  sycl::span<const char, sycl::dynamic_extent> get_handle_data() const {
    adapter_impl &Adapter = MContext->getAdapter();
    const void *HandleDataPtr = nullptr;
    size_t HandleDataSize = 0;
    Adapter.call<UrApiKind::urIPCGetMemHandleDataExp>(
        MContext->getHandleRef(), MUrHandle, &HandleDataPtr, &HandleDataSize);
    return sycl::span<const char, sycl::dynamic_extent>{
        reinterpret_cast<const char *>(HandleDataPtr), HandleDataSize};
  }

  void *get_ptr() const { return MPtr; }

private:
  enum class HandleRelationship { Owner, Adopted } MRelationship;
  std::shared_ptr<context_impl> MContext;
  void *MPtr = nullptr;
  ur_exp_ipc_mem_handle_t MUrHandle = nullptr;
};

} // namespace detail
} // namespace _V1
} // namespace sycl