//==------------ sycl_mem_obj_t.cpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/memory_manager.hpp>
#include <detail/plugin.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/sycl_mem_obj_t.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

SYCLMemObjT::SYCLMemObjT(pi_native_handle MemObject, const context &SyclContext,
                         const size_t, event AvailableEvent,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator)
    : SYCLMemObjT(MemObject, SyclContext, true, AvailableEvent,
                  std::move(Allocator)) {}

SYCLMemObjT::SYCLMemObjT(pi_native_handle MemObject, const context &SyclContext,
                         bool OwnNativeHandle, event AvailableEvent,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator)
    : MAllocator(std::move(Allocator)), MProps(),
      MInteropEvent(detail::getSyclObjImpl(std::move(AvailableEvent))),
      MInteropContext(detail::getSyclObjImpl(SyclContext)),
      MOpenCLInterop(true), MHostPtrReadOnly(false), MNeedWriteBack(true),
      MUserPtr(nullptr), MShadowCopy(nullptr), MUploadDataFunctor(nullptr),
      MSharedPtrStorage(nullptr), MHostPtrProvided(true),
      MOwnNativeHandle(OwnNativeHandle) {
  sycl::detail::pi::PiContext Context = nullptr;
  const PluginPtr &Plugin = getPlugin();

  Plugin->call<detail::PiApiKind::piextMemCreateWithNativeHandle>(
      MemObject, MInteropContext->getHandleRef(), OwnNativeHandle,
      &MInteropMemObject);

  // Get the size of the buffer in bytes
  Plugin->call<detail::PiApiKind::piMemGetInfo>(
      MInteropMemObject, PI_MEM_SIZE, sizeof(size_t), &MSizeInBytes, nullptr);

  Plugin->call<PiApiKind::piMemGetInfo>(MInteropMemObject, PI_MEM_CONTEXT,
                                        sizeof(Context), &Context, nullptr);

  if (MInteropContext->getHandleRef() != Context)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Input context must be the same as the context of cl_mem");

  if (MInteropContext->getBackend() == backend::opencl)
    Plugin->call<PiApiKind::piMemRetain>(MInteropMemObject);
}

sycl::detail::pi::PiMemObjectType getImageType(int Dimensions) {
  if (Dimensions == 1)
    return PI_MEM_TYPE_IMAGE1D;
  if (Dimensions == 2)
    return PI_MEM_TYPE_IMAGE2D;
  return PI_MEM_TYPE_IMAGE3D;
}

SYCLMemObjT::SYCLMemObjT(pi_native_handle MemObject, const context &SyclContext,
                         bool OwnNativeHandle, event AvailableEvent,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         sycl::detail::pi::PiMemImageChannelOrder Order,
                         sycl::detail::pi::PiMemImageChannelType Type,
                         range<3> Range3WithOnes, unsigned Dimensions,
                         size_t ElementSize)
    : MAllocator(std::move(Allocator)), MProps(),
      MInteropEvent(detail::getSyclObjImpl(std::move(AvailableEvent))),
      MInteropContext(detail::getSyclObjImpl(SyclContext)),
      MOpenCLInterop(true), MHostPtrReadOnly(false), MNeedWriteBack(true),
      MUserPtr(nullptr), MShadowCopy(nullptr), MUploadDataFunctor(nullptr),
      MSharedPtrStorage(nullptr), MHostPtrProvided(true),
      MOwnNativeHandle(OwnNativeHandle) {
  sycl::detail::pi::PiContext Context = nullptr;
  const PluginPtr &Plugin = getPlugin();

  sycl::detail::pi::PiMemImageFormat Format{Order, Type};
  sycl::detail::pi::PiMemImageDesc Desc;
  Desc.image_type = getImageType(Dimensions);
  Desc.image_width = Range3WithOnes[0];
  Desc.image_height = Range3WithOnes[1];
  Desc.image_depth = Range3WithOnes[2];
  Desc.image_array_size = 0;
  Desc.image_row_pitch = ElementSize * Desc.image_width;
  Desc.image_slice_pitch = Desc.image_row_pitch * Desc.image_height;
  Desc.num_mip_levels = 0;
  Desc.num_samples = 0;
  Desc.buffer = nullptr;

  Plugin->call<detail::PiApiKind::piextMemImageCreateWithNativeHandle>(
      MemObject, MInteropContext->getHandleRef(), OwnNativeHandle, &Format,
      &Desc, &MInteropMemObject);

  Plugin->call<PiApiKind::piMemGetInfo>(MInteropMemObject, PI_MEM_CONTEXT,
                                        sizeof(Context), &Context, nullptr);

  if (MInteropContext->getHandleRef() != Context)
    throw sycl::exception(make_error_code(errc::invalid),
        "Input context must be the same as the context of cl_mem");

  if (MInteropContext->getBackend() == backend::opencl)
    Plugin->call<PiApiKind::piMemRetain>(MInteropMemObject);
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
                  Dims, ElemSize, size_t(0));
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
  if (MRecord) {
    bool Result = Scheduler::getInstance().removeMemoryObject(this);
    std::ignore = Result; // for no assert build
    assert(
        Result &&
        "removeMemoryObject should not return false in mem object destructor");
  }
  releaseHostMem(MShadowCopy);

  if (MOpenCLInterop) {
    const PluginPtr &Plugin = getPlugin();
    Plugin->call<PiApiKind::piMemRelease>(
        pi::cast<sycl::detail::pi::PiMem>(MInteropMemObject));
  }
}
const PluginPtr &SYCLMemObjT::getPlugin() const {
  assert((MInteropContext != nullptr) &&
         "Trying to get Plugin from SYCLMemObjT with nullptr ContextImpl.");
  return (MInteropContext->getPlugin());
}

size_t SYCLMemObjT::getBufSizeForContext(const ContextImplPtr &Context,
                                         pi_native_handle MemObject) {
  size_t BufSize = 0;
  const PluginPtr &Plugin = Context->getPlugin();
  // TODO is there something required to support non-OpenCL backends?
  Plugin->call<detail::PiApiKind::piMemGetInfo>(
      detail::pi::cast<sycl::detail::pi::PiMem>(MemObject), PI_MEM_SIZE,
      sizeof(size_t), &BufSize, nullptr);
  return BufSize;
}

bool SYCLMemObjT::isInterop() const { return MOpenCLInterop; }

void SYCLMemObjT::determineHostPtr(bool InitFromUserData, void *&HostPtr,
                                   bool &HostPtrReadOnly) {
  // The data for the allocation can be provided via either the user pointer
  // (InitFromUserData, can be read-only) or a runtime-allocated read-write
  // HostPtr. We can have one of these scenarios:
  // 1. The allocation is the first one and isn't on host. InitFromUserData
  // varies based on unified host memory support and whether or not the data can
  // be discarded.
  // 2. The allocation is not the first one and not on host. InitFromUserData ==
  // false, HostPtr is provided if the command is linked. The host pointer is
  // guaranteed to be reused in this case.

  if (InitFromUserData) {
    assert(!HostPtr && "Cannot init from user data and reuse host ptr provided "
                       "simultaneously");
    HostPtr = getUserPtr();
    HostPtrReadOnly = MHostPtrReadOnly;
  } else
    HostPtrReadOnly = false;
}

void SYCLMemObjT::detachMemoryObject(
    const std::shared_ptr<SYCLMemObjT> &Self) const {
  // Check MRecord without read lock because at this point we expect that no
  // commands that operate on the buffer can be created. MRecord is nullptr on
  // buffer creation and set to meaningfull
  // value only if any operation on buffer submitted inside addCG call. addCG is
  // called from queue::submit and buffer destruction could not overlap with it.
  // For L0 context could be created with two ownership strategies - keep and
  // transfer. If user keeps ownership - we could not enable deferred buffer
  // release due to resource release conflict.
  // MRecord->MCurContext == nullptr means that last submission to buffer is on
  // host (host task), this execution doesn't depend on device context and fully
  // controlled by RT. In this case deferred buffer destruction is allowed.
  bool InteropObjectsUsed =
      !MOwnNativeHandle ||
      (MInteropContext && !MInteropContext->isOwnedByRuntime());

  if (MRecord &&
      (!MRecord->MCurContext || MRecord->MCurContext->isOwnedByRuntime()) &&
      !InteropObjectsUsed && (!MHostPtrProvided || MIsInternal)) {
    bool okToDefer = GlobalHandler::instance().isOkToDefer();
    if (okToDefer)
      Scheduler::getInstance().deferMemObjRelease(Self);
  }
}

void SYCLMemObjT::handleWriteAccessorCreation() {
  const auto InitialUserPtr = MUserPtr;
  MCreateShadowCopy();
  MCreateShadowCopy = []() -> void {};
  if (MRecord != nullptr && MUserPtr != InitialUserPtr) {
    for (auto &it : MRecord->MAllocaCommands) {
      if (it->MMemAllocation == InitialUserPtr) {
        it->MMemAllocation = MUserPtr;
      }
    }
  }
}

} // namespace detail
} // namespace _V1
} // namespace sycl
