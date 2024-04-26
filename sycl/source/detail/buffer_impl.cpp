//==----------------- buffer_impl.cpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/buffer_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/memory_manager.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/xpti_registry.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
uint8_t GBufferStreamID;
#endif
void *buffer_impl::allocateMem(ContextImplPtr Context, bool InitFromUserData,
                               void *HostPtr,
                               ur_event_handle_t &OutEventToWait) {
  bool HostPtrReadOnly = false;
  BaseT::determineHostPtr(Context, InitFromUserData, HostPtr, HostPtrReadOnly);

  assert(!(nullptr == HostPtr && BaseT::useHostPtr() && Context->is_host()) &&
         "Internal error. Allocating memory on the host "
         "while having use_host_ptr property");
  return MemoryManager::allocateMemBuffer(
      std::move(Context), this, HostPtr, HostPtrReadOnly,
      BaseT::getSizeInBytes(), BaseT::MInteropEvent, BaseT::MInteropContext,
      MProps, OutEventToWait);
}
void buffer_impl::constructorNotification(const detail::code_location &CodeLoc,
                                          void *UserObj, const void *HostObj,
                                          const void *Type, uint32_t Dim,
                                          uint32_t ElemSize, size_t Range[3]) {
  XPTIRegistry::bufferConstructorNotification(UserObj, CodeLoc, HostObj, Type,
                                              Dim, ElemSize, Range);
}

void buffer_impl::destructorNotification(void *UserObj) {
  XPTIRegistry::bufferDestructorNotification(UserObj);
}

void buffer_impl::addInteropObject(
    std::vector<ur_native_handle_t> &Handles) const {
  if (MOpenCLInterop) {
    if (std::find(Handles.begin(), Handles.end(),
                  pi::cast<ur_native_handle_t>(MInteropMemObject)) ==
        Handles.end()) {
      const UrPluginPtr &Plugin = getPlugin();
      Plugin->call(urMemRetain, pi::cast<ur_mem_handle_t>(MInteropMemObject));
      Handles.push_back(pi::cast<ur_native_handle_t>(MInteropMemObject));
    }
  }
}

std::vector<ur_native_handle_t>
buffer_impl::getNativeVector(backend BackendName) const {
  std::vector<ur_native_handle_t> Handles{};
  if (!MRecord) {
    addInteropObject(Handles);
    return Handles;
  }

  for (auto &Cmd : MRecord->MAllocaCommands) {
    ur_mem_handle_t NativeMem =
        pi::cast<ur_mem_handle_t>(Cmd->getMemAllocation());
    auto Ctx = Cmd->getWorkerContext();
    auto Platform = Ctx->getPlatformImpl();
    // If Host Shared Memory is not supported then there is alloca for host that
    // doesn't have platform
    if (!Platform || (Platform->getBackend() != BackendName))
      continue;

    auto Plugin = Platform->getUrPlugin();

    if (Platform->getBackend() == backend::opencl) {
      Plugin->call(urMemRetain, NativeMem);
    }

    ur_native_handle_t Handle = nullptr;
    // When doing buffer interop we don't know what device the memory should be
    // resident on, so pass nullptr for Device param. Buffer interop may not be
    // supported by all backends.
    Plugin->call(urMemGetNativeHandle, NativeMem, /*Dev*/ nullptr, &Handle);
    Handles.push_back(Handle);
  }

  addInteropObject(Handles);
  return Handles;
}
} // namespace detail
} // namespace _V1
} // namespace sycl
