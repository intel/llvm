//==------------ interop_handle.cpp --- SYCL interop handle ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/accessor_impl.hpp>
#include <detail/backend_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>
#include <detail/queue_impl.hpp>
#include <sycl/exception.hpp>
#include <sycl/interop_handle.hpp>

#include <algorithm>

namespace sycl {
inline namespace _V1 {

backend interop_handle::get_backend() const noexcept {
  return detail::getImplBackend(MQueue);
}

bool interop_handle::ext_codeplay_has_graph() const noexcept {
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  // CMPLRLLVM-66082 - MGraph should become a member of this class on the
  // next ABI breaking window.
  return MGraph != nullptr;
#else
  return MQueue->getInteropGraph() != nullptr;
#endif
}

ur_native_handle_t
interop_handle::getNativeMem(detail::Requirement *Req) const {
  auto Iter = std::find_if(std::begin(MMemObjs), std::end(MMemObjs),
                           [=](ReqToMem Elem) { return (Elem.first == Req); });

  if (Iter == std::end(MMemObjs)) {
    throw exception(make_error_code(errc::invalid),
                    "Invalid memory object used inside interop");
  }

  auto Adapter = MQueue->getAdapter();
  ur_native_handle_t Handle;
  Adapter->call<detail::UrApiKind::urMemGetNativeHandle>(
      Iter->second, MDevice->getHandleRef(), &Handle);
  return Handle;
}

ur_native_handle_t interop_handle::getNativeDevice() const {
  return MDevice->getNative();
}

ur_native_handle_t interop_handle::getNativeContext() const {
  return MContext->getNative();
}

ur_native_handle_t
interop_handle::getNativeQueue(int32_t &NativeHandleDesc) const {
  return MQueue->getNative(NativeHandleDesc);
}

ur_native_handle_t interop_handle::getNativeGraph() const {
#ifdef __INTEL_PREVIEW_BREAKING_CHANGES
  // CMPLRLLVM-66082 - MGraph should become a member of this class on the
  // next ABI breaking window.
  auto Graph = MGraph;
#else
  auto Graph = MQueue->getInteropGraph();
#endif

  if (!Graph) {
    throw exception(
        make_error_code(errc::invalid),
        "No backend graph object is available for the command-group");
  }

  auto Adapter = MQueue->getAdapter();
  ur_native_handle_t Handle = 0;
  Adapter->call<detail::UrApiKind::urCommandBufferGetNativeHandleExp>(Graph,
                                                                      &Handle);
  return Handle;
}
} // namespace _V1
} // namespace sycl
