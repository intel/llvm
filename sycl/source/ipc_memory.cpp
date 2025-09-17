//==------- ipc_memory.cpp --- SYCL inter-process communicable memory ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/ipc_memory_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

ipc_memory::ipc_memory(void *Ptr, const sycl::context &Ctx)
    : impl(detail::ipc_memory_impl::create(Ptr, Ctx)) {}

void *ipc_memory::open(ipc_memory_handle_data_t IPCMemoryHandleData,
                       const sycl::context &Ctx, const sycl::device &Dev) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  void *Ptr = nullptr;
  ur_result_t UrRes =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCOpenMemHandleExp>(
          CtxImpl->getHandleRef(), getSyclObjImpl(Dev)->getHandleRef(),
          IPCMemoryHandleData.data(), IPCMemoryHandleData.size(), &Ptr);
  if (UrRes == UR_RESULT_ERROR_INVALID_VALUE)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "IPCMemoryHandleData data size does not correspond "
                          "to the target platform's IPC memory handle size.");
  Adapter.checkUrResult(UrRes);
  return Ptr;
}

void ipc_memory::close(void *Ptr, const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();
  Adapter.call<sycl::detail::UrApiKind::urIPCCloseMemHandleExp>(
      CtxImpl->getHandleRef(), Ptr);
}

ipc_memory_handle_data_t ipc_memory::get_handle_data() const {
  return impl->get_handle_data();
}

void *ipc_memory::get_ptr() const { return impl->get_ptr(); }

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl
