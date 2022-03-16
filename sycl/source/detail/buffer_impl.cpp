//==----------------- buffer_impl.cpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/buffer_impl.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <detail/context_impl.hpp>
#include <detail/global_handler.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/xpti_registry.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
#ifdef XPTI_ENABLE_INSTRUMENTATION
uint8_t GBufferStreamID;
#endif
void *buffer_impl::allocateMem(ContextImplPtr Context, bool InitFromUserData,
                               void *HostPtr, RT::PiEvent &OutEventToWait) {
  bool HostPtrReadOnly = false;
  BaseT::determineHostPtr(Context, InitFromUserData, HostPtr, HostPtrReadOnly);

  assert(!(nullptr == HostPtr && BaseT::useHostPtr() && Context->is_host()) &&
         "Internal error. Allocating memory on the host "
         "while having use_host_ptr property");
  return MemoryManager::allocateMemBuffer(
      std::move(Context), this, HostPtr, HostPtrReadOnly, BaseT::getSize(),
      BaseT::MInteropEvent, BaseT::MInteropContext, MProps, OutEventToWait);
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

// backend buffer_impl::getBackend() const noexcept {
//   auto &Plugin = getPlugin();
//   return Plugin.getBackend();
// }

std::vector<pi_native_handle> buffer_impl::getNative(backend BackendName) const {
  /*if (MInteropContext == nullptr) {
    static context SyclContext;
    MInteropContext = getSyclObjImpl(SyclContext);
  }*/

  //auto &Plugin = getPlugin();
  // std::vector<RT::PiMem> MemAllocations;
  // MemAllocations.reserve(MRecord->MAllocaCommands.size());
  // for (auto &Cmd : MRecord->MAllocaCommands) {
  //   MemAllocations.push_back(pi::cast<RT::PiMem>(Cmd->getMemAllocation()));
  // }
  
  // if (Plugin.getBackend() == backend::opencl) {
    // for (auto &Alloca : MemAllocations) {
    //   Plugin.call<PiApiKind::piMemRetain>(Alloca);
    // }
  std::vector<pi_native_handle> Handles{};
  if (!MRecord)
    return Handles;
  Handles.reserve(MRecord->MAllocaCommands.size());
  
  std::vector<RT::PiMem> MemAllocations;
  MemAllocations.reserve(MRecord->MAllocaCommands.size());

  for (auto &Cmd : MRecord->MAllocaCommands) {
    RT::PiMem NativeMem = pi::cast<RT::PiMem>(Cmd->getMemAllocation());
    MemAllocations.push_back(NativeMem);
    auto Ctx = Cmd->getWorkerContext();
    auto Plugin = Ctx->getPlugin();

    if (Plugin.getBackend() != BackendName) 
      continue;
    if (Plugin.getBackend() == backend::opencl) {
      Plugin.call<PiApiKind::piMemRetain>(NativeMem);
    }

    pi_native_handle Handle;
    Plugin.call<PiApiKind::piextMemGetNativeHandle>(NativeMem, &Handle);
    Handles.push_back(Handle);
  }

  //if (Plugin.getBackend() == backend::opencl)
  //  Plugin.call<PiApiKind::piMemRetain>(pi_mem mem);       // how to get RT::PiMem
  // MInteropMemObject - OpenCL's memory object handle passed by user to interoperability constructor. Should it be checked (it seems it is deprecated)?
  // Get vector<AllocaBaseCommand> from MRecord (MemObjRecord::MAllocaCommands) and use for every getMemAllocation() - it returns raw pointer
  // pi::cast<RT::PiMem>(MemAllocation)

  //std::vector<pi_native_handle> Handles;
  //Handles.reserve(MemAllocations.size());
  // for (auto &Alloc : MemAllocations) {
  //   pi_native_handle Handle;
  //   Plugin.call<PiApiKind::piextMemGetNativeHandle>(Alloc, &Handle);
  //   Handles.push_back(Handle);
  // }

  return Handles;
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
