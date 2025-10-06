//==------- ipc_memory.cpp --- SYCL inter-process communicable memory ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/context_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/ext/oneapi/experimental/ipc_memory.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental::ipc_memory {

__SYCL_EXPORT handle_data_t get(void *Ptr, const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  size_t HandleSize = 0;
  Adapter.call<sycl::detail::UrApiKind::urIPCGetMemHandleExp>(
      CtxImpl->getHandleRef(), Ptr, nullptr, &HandleSize);

  handle_data_t Res(HandleSize);
  Adapter.call<sycl::detail::UrApiKind::urIPCGetMemHandleExp>(
      CtxImpl->getHandleRef(), Ptr, Res.data(), nullptr);
  return Res;
}

__SYCL_EXPORT void put(handle_data_t &HandleData, const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();
  Adapter.call<sycl::detail::UrApiKind::urIPCPutMemHandleExp>(
      CtxImpl->getHandleRef(), HandleData.data());
}

__SYCL_EXPORT void *open(handle_data_t &HandleData, const sycl::context &Ctx,
                         const sycl::device &Dev) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  void *Ptr = nullptr;
  ur_result_t UrRes =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCOpenMemHandleExp>(
          CtxImpl->getHandleRef(), getSyclObjImpl(Dev)->getHandleRef(),
          HandleData.data(), HandleData.size(), &Ptr);
  if (UrRes == UR_RESULT_ERROR_INVALID_VALUE)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "HandleData data size does not correspond "
                          "to the target platform's IPC memory handle size.");
  Adapter.checkUrResult(UrRes);
  return Ptr;
}

__SYCL_EXPORT void close(void *Ptr, const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();
  Adapter.call<sycl::detail::UrApiKind::urIPCCloseMemHandleExp>(
      CtxImpl->getHandleRef(), Ptr);
}

} // namespace ext::oneapi::experimental::ipc_memory
} // namespace _V1
} // namespace sycl
