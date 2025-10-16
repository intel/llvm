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
#include <sycl/usm/usm_pointer_info.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental::ipc_memory {

__SYCL_EXPORT handle_data_t get(void *Ptr, const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  // If the API fails, check that the device actually supported it. We only do
  // this if UR fails to avoid the device-lookup overhead.
  auto CheckDeviceSupport = [Ptr, &Ctx]() {
    sycl::device Dev = get_pointer_device(Ptr, Ctx);
    if (!Dev.has(aspect::ext_oneapi_ipc_memory))
      throw sycl::exception(
          sycl::make_error_code(errc::feature_not_supported),
          "Device does not support aspect::ext_oneapi_ipc_memory.");
  };

  size_t HandleSize = 0;
  auto UrRes =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCGetMemHandleExp>(
          CtxImpl->getHandleRef(), Ptr, nullptr, &HandleSize);
  if (UrRes != UR_RESULT_SUCCESS) {
    CheckDeviceSupport();
    Adapter.checkUrResult(UrRes);
  }

  handle_data_t Res(HandleSize);
  UrRes = Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCGetMemHandleExp>(
      CtxImpl->getHandleRef(), Ptr, Res.data(), nullptr);
  if (UrRes != UR_RESULT_SUCCESS) {
    CheckDeviceSupport();
    Adapter.checkUrResult(UrRes);
  }
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
  if (!Dev.has(aspect::ext_oneapi_ipc_memory))
    throw sycl::exception(
        sycl::make_error_code(errc::feature_not_supported),
        "Device does not support aspect::ext_oneapi_ipc_memory.");

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
