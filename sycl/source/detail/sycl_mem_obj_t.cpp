//==------------ sycl_mem_obj_t.cpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/sycl_mem_obj_t.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/plugin.hpp>
#include <detail/scheduler/scheduler.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
SYCLMemObjT::SYCLMemObjT(cl_mem MemObject, const context &SyclContext,
                         const size_t SizeInBytes, event AvailableEvent,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator)
    : MAllocator(std::move(Allocator)), MProps(),
      MInteropEvent(detail::getSyclObjImpl(std::move(AvailableEvent))),
      MInteropContext(detail::getSyclObjImpl(SyclContext)),
      MInteropMemObject(MemObject), MOpenCLInterop(true),
      MHostPtrReadOnly(false), MNeedWriteBack(true), MSizeInBytes(SizeInBytes),
      MUserPtr(nullptr), MShadowCopy(nullptr), MUploadDataFunctor(nullptr),
      MSharedPtrStorage(nullptr) {
  if (MInteropContext->is_host())
    throw cl::sycl::invalid_parameter_error(
        "Creation of interoperability memory object using host context is "
        "not allowed",
        PI_INVALID_CONTEXT);

  RT::PiMem Mem = pi::cast<RT::PiMem>(MInteropMemObject);
  RT::PiContext Context = nullptr;
  const plugin &Plugin = getPlugin();
  Plugin.call<PiApiKind::piMemGetInfo>(Mem, CL_MEM_CONTEXT, sizeof(Context),
                                       &Context, nullptr);

  if (MInteropContext->getHandleRef() != Context)
    throw cl::sycl::invalid_parameter_error(
        "Input context must be the same as the context of cl_mem",
        PI_INVALID_CONTEXT);
  Plugin.call<PiApiKind::piMemRetain>(Mem);
}

void SYCLMemObjT::releaseMem(ContextImplPtr Context, void *MemAllocation) {
  void *Ptr = getUserPtr();
  return MemoryManager::releaseMemObj(Context, this, MemAllocation, Ptr);
}

void SYCLMemObjT::updateHostMemory(void *const Ptr) {
  const id<3> Offset{0, 0, 0};
  const range<3> AccessRange{MSizeInBytes, 1, 1};
  const range<3> MemoryRange{MSizeInBytes, 1, 1};
  const access::mode AccessMode = access::mode::read;
  SYCLMemObjI *SYCLMemObject = this;
  const int Dims = 1;
  const int ElemSize = 1;

  Requirement Req(Offset, AccessRange, MemoryRange, AccessMode, SYCLMemObject,
                  Dims, ElemSize);
  Req.MData = Ptr;

  EventImplPtr Event = Scheduler::getInstance().addCopyBack(&Req);
  if (Event)
    Event->wait(Event);
}

void SYCLMemObjT::updateHostMemory() {
  if ((MUploadDataFunctor != nullptr) && MNeedWriteBack)
    MUploadDataFunctor();

  // If we're attached to a memory record, process the deletion of the memory
  // record. We may get detached before we do this.
  if (MRecord)
    Scheduler::getInstance().removeMemoryObject(this);
  releaseHostMem(MShadowCopy);

  if (MOpenCLInterop) {
    const plugin &Plugin = getPlugin();
    Plugin.call<PiApiKind::piMemRelease>(
        pi::cast<RT::PiMem>(MInteropMemObject));
  }
}
const plugin &SYCLMemObjT::getPlugin() const {
  assert((MInteropContext != nullptr) &&
         "Trying to get Plugin from SYCLMemObjT with nullptr ContextImpl.");
  return (MInteropContext->getPlugin());
}

size_t SYCLMemObjT::getBufSizeForContext(const ContextImplPtr &Context,
                                         cl_mem MemObject) {
  size_t BufSize = 0;
  const detail::plugin &Plugin = Context->getPlugin();
  Plugin.call<detail::PiApiKind::piMemGetInfo>(
      detail::pi::cast<detail::RT::PiMem>(MemObject), CL_MEM_SIZE,
      sizeof(size_t), &BufSize, nullptr);
  return BufSize;
}

bool SYCLMemObjT::isInterop() const { return MOpenCLInterop; }

void SYCLMemObjT::determineHostPtr(const ContextImplPtr &Context,
                                   bool InitFromUserData, void *&HostPtr,
                                   bool &HostPtrReadOnly) {
  // The data for the allocation can be provided via either the user pointer
  // (InitFromUserData, can be read-only) or a runtime-allocated read-write
  // HostPtr. We can have one of these scenarios:
  // 1. The allocation is the first one and on host. InitFromUserData == true.
  // 2. The allocation is the first one and isn't on host. InitFromUserData
  // varies based on unified host memory support and whether or not the data can
  // be discarded.
  // 3. The allocation is not the first one and is on host. InitFromUserData ==
  // false, HostPtr == nullptr. This can only happen if the allocation command
  // is not linked since it would be a no-op otherwise. Attempt to reuse the
  // user pointer if it's read-write, but do not copy its contents if it's not.
  // 4. The allocation is not the first one and not on host. InitFromUserData ==
  // false, HostPtr is provided if the command is linked. The host pointer is
  // guaranteed to be reused in this case.
  if (Context->is_host() && !MOpenCLInterop && !MHostPtrReadOnly)
    InitFromUserData = true;

  if (InitFromUserData) {
    assert(!HostPtr && "Cannot init from user data and reuse host ptr provided "
                       "simultaneously");
    HostPtr = getUserPtr();
    HostPtrReadOnly = MHostPtrReadOnly;
  } else
    HostPtrReadOnly = false;
}
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
