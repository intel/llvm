//==----------------- buffer_impl.cpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/xpti_registry.hpp>
#include <sycl/detail/buffer_impl.hpp>
#include <sycl/detail/memory_manager.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
uint8_t GBufferStreamID;
#endif
void *buffer_impl::allocateMem(ContextImplPtr Context, DeviceImplPtr Device,
                               bool InitFromUserData, void *HostPtr,
                               RT::PiEvent &OutEventToWait) {
  bool HostPtrReadOnly = false;
  BaseT::determineHostPtr(Context, InitFromUserData, HostPtr, HostPtrReadOnly);

  assert(!(nullptr == HostPtr && BaseT::useHostPtr() && Context->is_host()) &&
         "Internal error. Allocating memory on the host "
         "while having use_host_ptr property");
  return MemoryManager::allocateMemBuffer(
      std::move(Context), std::move(Device), this, HostPtr, HostPtrReadOnly,
      BaseT::getSize(), BaseT::MInteropEvent, BaseT::MInteropContext, MProps,
      OutEventToWait);
}
void buffer_impl::constructorNotification(const detail::code_location &CodeLoc,
                                          void *UserObj, const void *HostObj,
                                          const void *Type, uint32_t Dim,
                                          uint32_t ElemSize, size_t Range[3]) {
  XPTIRegistry::bufferConstructorNotification(UserObj, CodeLoc, HostObj, Type,
                                              Dim, ElemSize, Range);
}
// TODO: remove once ABI break is allowed
void buffer_impl::constructorNotification(const detail::code_location &CodeLoc,
                                          void *UserObj) {
  size_t r[3] = {0, 0, 0};
  constructorNotification(CodeLoc, UserObj, nullptr, "", 0, 0, r);
}

void buffer_impl::destructorNotification(void *UserObj) {
  XPTIRegistry::bufferDestructorNotification(UserObj);
}

void buffer_impl::addInteropObject(
    std::vector<pi_native_handle> &Handles) const {
  if (MOpenCLInterop) {
    if (std::find(Handles.begin(), Handles.end(),
                  pi::cast<pi_native_handle>(MInteropMemObject)) ==
        Handles.end()) {
      const plugin &Plugin = getPlugin();
      Plugin.call<PiApiKind::piMemRetain>(
          pi::cast<RT::PiMem>(MInteropMemObject));
      Handles.push_back(pi::cast<pi_native_handle>(MInteropMemObject));
    }
  }
}

std::vector<pi_native_handle>
buffer_impl::getNativeVector(backend BackendName) const {
  std::vector<pi_native_handle> Handles{};
  if (!MRecord) {
    addInteropObject(Handles);
    return Handles;
  }

  for (auto &Cmd : MRecord->MAllocaCommands) {
    RT::PiMem NativeMem = pi::cast<RT::PiMem>(Cmd->getMemAllocation());
    auto Ctx = Cmd->getWorkerContext();
    auto Platform = Ctx->getPlatformImpl();
    // If Host Shared Memory is not supported then there is alloca for host that
    // doesn't have platform
    if (!Platform)
      continue;
    auto Plugin = Platform->getPlugin();

    if (Plugin.getBackend() != BackendName)
      continue;
    if (Plugin.getBackend() == backend::opencl) {
      Plugin.call<PiApiKind::piMemRetain>(NativeMem);
    }

    pi_native_handle Handle;
    Plugin.call<PiApiKind::piextMemGetNativeHandle>(NativeMem, &Handle);
    Handles.push_back(Handle);
  }

  addInteropObject(Handles);
  return Handles;
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
