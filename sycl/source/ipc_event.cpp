//==------- ipc_event.cpp -- SYCL inter-process for events -----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <sycl/context.hpp>
#include <sycl/event.hpp>
#include <sycl/ext/oneapi/experimental/ipc_event.hpp>

#include <cassert>

namespace sycl {
inline namespace _V1 {

namespace detail {

__SYCL_EXPORT sycl::event openIPCEventHandle(const std::byte *HandleData,
                                             size_t HandleDataSize,
                                             const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);

  // Consumer side: make_event's aspect check never ran in this process.
  if (!CtxImpl->supportsIPCEvents())
    throw sycl::exception(sycl::make_error_code(errc::feature_not_supported),
                          "Not all devices in the context support "
                          "aspect::ext_oneapi_ipc_event.");

  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  ur_event_handle_t UrEvent = nullptr;
  ur_result_t UrRes =
      Adapter.call_nocheck<sycl::detail::UrApiKind::urIPCOpenEventHandleExp>(
          CtxImpl->getHandleRef(), HandleData, HandleDataSize, &UrEvent);
  if (UrRes == UR_RESULT_ERROR_INVALID_VALUE)
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "HandleData size does not correspond to the target platform's "
        "IPC event handle size.");
  Adapter.checkUrResult(UrRes);
  assert(UrEvent && "urIPCOpenEventHandleExp returned success with null event");

  // event_impl adopts the UR handle; release it directly if that step
  // throws so the import doesn't leak.
  try {
    auto EventImpl =
        sycl::detail::event_impl::create_ipc_imported_event(UrEvent, Ctx);
    return sycl::detail::createSyclObjFromImpl<sycl::event>(EventImpl);
  } catch (...) {
    Adapter.call_nocheck<sycl::detail::UrApiKind::urEventRelease>(UrEvent);
    throw;
  }
}

} // namespace detail

namespace ext::oneapi::experimental::ipc::event {

__SYCL_EXPORT ipc::handle_data_t get(const sycl::event &Evt) {
  if (!Evt.ext_oneapi_ipc_enabled())
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "Event was not created with the enable_ipc property.");

  auto EvtImpl = sycl::detail::getSyclObjImpl(Evt);

  // The backend event is created lazily; materialize it if get() is called
  // before the first signal.
  EvtImpl->materializeIPCEvent();

  sycl::detail::adapter_impl &Adapter = EvtImpl->getContextImpl().getAdapter();

  // Query the handle size, allocate a buffer, then fill it.
  // The handle is a plain value type — no put() is required.
  size_t HandleSize = 0;
  Adapter.call<sycl::detail::UrApiKind::urIPCGetEventHandleExp>(
      EvtImpl->getHandle(), 0, nullptr, &HandleSize);

  ipc::handle_data_t Bytes(HandleSize);
  Adapter.call<sycl::detail::UrApiKind::urIPCGetEventHandleExp>(
      EvtImpl->getHandle(), HandleSize, Bytes.data(), &HandleSize);

  return Bytes;
}

} // namespace ext::oneapi::experimental::ipc::event
} // namespace _V1
} // namespace sycl
