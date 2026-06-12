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

#include <algorithm>
#include <cassert>

namespace sycl {
inline namespace _V1 {

namespace detail {

// All devices in the context must support the aspect; spec:
// "_Throws:_ ... `errc::feature_not_supported` ... if not all devices that
// are part of `ctx` have `aspect::ext_oneapi_ipc_event`."
static void requireIPCEventAspect(const sycl::context &Ctx) {
  for (const sycl::device &Dev : Ctx.get_devices()) {
    if (!Dev.has(aspect::ext_oneapi_ipc_event))
      throw sycl::exception(sycl::make_error_code(errc::feature_not_supported),
                            "Not all devices in the context support "
                            "aspect::ext_oneapi_ipc_event.");
  }
}

__SYCL_EXPORT sycl::event openIPCEventHandle(const std::byte *HandleData,
                                             size_t HandleDataSize,
                                             const sycl::context &Ctx) {
  requireIPCEventAspect(Ctx);

  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
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

__SYCL_EXPORT sycl::event makeIPCEvent(const sycl::context &Ctx) {
  requireIPCEventAspect(Ctx);

  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  // UR's urEventCreateExp takes a device parameter; pick the first device
  // in the context. All devices in the context have the IPC event aspect
  // (validated above), so any choice is valid.
  const std::vector<sycl::device> Devs = Ctx.get_devices();
  assert(!Devs.empty() && "context has no devices");

  ur_exp_event_desc_t Desc{UR_STRUCTURE_TYPE_EXP_EVENT_DESC, nullptr,
                           UR_EXP_EVENT_FLAG_IPC_EXP};

  ur_event_handle_t UrEvent = nullptr;
  Adapter.call<sycl::detail::UrApiKind::urEventCreateExp>(
      CtxImpl->getHandleRef(), getSyclObjImpl(Devs.front())->getHandleRef(),
      &Desc, &UrEvent);
  assert(UrEvent && "urEventCreateExp returned success with null event");

  try {
    auto EventImpl =
        sycl::detail::event_impl::create_ipc_producer_event(UrEvent, Ctx);
    return sycl::detail::createSyclObjFromImpl<sycl::event>(EventImpl);
  } catch (...) {
    Adapter.call_nocheck<sycl::detail::UrApiKind::urEventRelease>(UrEvent);
    throw;
  }
}

} // namespace detail

namespace ext::oneapi::experimental::ipc::event {

__SYCL_EXPORT handle get(const sycl::event &Evt) {
  if (!Evt.ext_oneapi_ipc_enabled())
    throw sycl::exception(
        sycl::make_error_code(errc::invalid),
        "Event was not created with the enable_ipc property.");

  auto EvtImpl = sycl::detail::getSyclObjImpl(Evt);
  sycl::detail::context_impl &CtxImpl = EvtImpl->getContextImpl();
  sycl::detail::adapter_impl &Adapter = CtxImpl.getAdapter();

  void *HandlePtr = nullptr;
  size_t HandleSize = 0;
  Adapter.call<sycl::detail::UrApiKind::urIPCGetEventHandleExp>(
      EvtImpl->getHandle(), &HandlePtr, &HandleSize);

  return {HandlePtr, HandleSize};
}

__SYCL_EXPORT void put(handle &IpcHandle, const sycl::context &Ctx) {
  auto CtxImpl = sycl::detail::getSyclObjImpl(Ctx);
  sycl::detail::adapter_impl &Adapter = CtxImpl->getAdapter();

  Adapter.call<sycl::detail::UrApiKind::urIPCPutEventHandleExp>(
      CtxImpl->getHandleRef(), IpcHandle.MData);
}

} // namespace ext::oneapi::experimental::ipc::event
} // namespace _V1
} // namespace sycl
