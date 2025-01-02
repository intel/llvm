//==------------ sycl_mem_obj_t.cpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/memory_manager.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/sycl_mem_obj_t.hpp>

namespace sycl {
inline namespace _V1 {
namespace detail {

SYCLMemObjT::SYCLMemObjT(ur_native_handle_t MemObject,
                         const context &SyclContext, const size_t,
                         event AvailableEvent,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator)
    : SYCLMemObjT(MemObject, SyclContext, true, AvailableEvent,
                  std::move(Allocator)) {}

SYCLMemObjT::SYCLMemObjT(ur_native_handle_t MemObject,
                         const context &SyclContext, bool OwnNativeHandle,
                         event AvailableEvent,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator)
    : MAllocator(std::move(Allocator)), MProps(),
      MInteropEvent(detail::getSyclObjImpl(std::move(AvailableEvent))),
      MInteropContext(detail::getSyclObjImpl(SyclContext)),
      MOpenCLInterop(true), MHostPtrReadOnly(false), MNeedWriteBack(true),
      MUserPtr(nullptr), MShadowCopy(nullptr), MUploadDataFunctor(nullptr),
      MSharedPtrStorage(nullptr), MHostPtrProvided(true),
      MOwnNativeHandle(OwnNativeHandle) {
  ur_context_handle_t Context = nullptr;
  const AdapterPtr &Adapter = getAdapter();

  ur_mem_native_properties_t MemProperties = {
      UR_STRUCTURE_TYPE_MEM_NATIVE_PROPERTIES, nullptr, OwnNativeHandle};
  Adapter->call<UrApiKind::urMemBufferCreateWithNativeHandle>(
      MemObject, MInteropContext->getHandleRef(), &MemProperties,
      &MInteropMemObject);

  // Get the size of the buffer in bytes
  Adapter->call<UrApiKind::urMemGetInfo>(MInteropMemObject, UR_MEM_INFO_SIZE,
                                         sizeof(size_t), &MSizeInBytes,
                                         nullptr);

  Adapter->call<UrApiKind::urMemGetInfo>(MInteropMemObject, UR_MEM_INFO_CONTEXT,
                                         sizeof(Context), &Context, nullptr);

  if (MInteropContext->getHandleRef() != Context)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Input context must be the same as the context of cl_mem");

  if (MInteropContext->getBackend() == backend::opencl)
    Adapter->call<UrApiKind::urMemRetain>(MInteropMemObject);
}

ur_mem_type_t getImageType(int Dimensions) {
  if (Dimensions == 1)
    return UR_MEM_TYPE_IMAGE1D;
  if (Dimensions == 2)
    return UR_MEM_TYPE_IMAGE2D;
  return UR_MEM_TYPE_IMAGE3D;
}

SYCLMemObjT::SYCLMemObjT(ur_native_handle_t MemObject,
                         const context &SyclContext, bool OwnNativeHandle,
                         event AvailableEvent,
                         std::unique_ptr<SYCLMemObjAllocator> Allocator,
                         ur_image_format_t Format, range<3> Range3WithOnes,
                         unsigned Dimensions, size_t ElementSize)
    : MAllocator(std::move(Allocator)), MProps(),
      MInteropEvent(detail::getSyclObjImpl(std::move(AvailableEvent))),
      MInteropContext(detail::getSyclObjImpl(SyclContext)),
      MOpenCLInterop(true), MHostPtrReadOnly(false), MNeedWriteBack(true),
      MUserPtr(nullptr), MShadowCopy(nullptr), MUploadDataFunctor(nullptr),
      MSharedPtrStorage(nullptr), MHostPtrProvided(true),
      MOwnNativeHandle(OwnNativeHandle) {
  ur_context_handle_t Context = nullptr;
  const AdapterPtr &Adapter = getAdapter();

  ur_image_desc_t Desc = {};
  Desc.stype = UR_STRUCTURE_TYPE_IMAGE_DESC;
  Desc.type = getImageType(Dimensions);
  Desc.width = Range3WithOnes[0];
  Desc.height = Range3WithOnes[1];
  Desc.depth = Range3WithOnes[2];
  Desc.arraySize = 0;
  Desc.rowPitch = ElementSize * Desc.width;
  Desc.slicePitch = Desc.rowPitch * Desc.height;
  Desc.numMipLevel = 0;
  Desc.numSamples = 0;

  ur_mem_native_properties_t NativeProperties = {
      UR_STRUCTURE_TYPE_MEM_NATIVE_PROPERTIES, nullptr, OwnNativeHandle};

  Adapter->call<UrApiKind::urMemImageCreateWithNativeHandle>(
      MemObject, MInteropContext->getHandleRef(), &Format, &Desc,
      &NativeProperties, &MInteropMemObject);

  Adapter->call<UrApiKind::urMemGetInfo>(MInteropMemObject, UR_MEM_INFO_CONTEXT,
                                         sizeof(Context), &Context, nullptr);

  if (MInteropContext->getHandleRef() != Context)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Input context must be the same as the context of cl_mem");

  if (MInteropContext->getBackend() == backend::opencl)
    Adapter->call<UrApiKind::urMemRetain>(MInteropMemObject);
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
    const AdapterPtr &Adapter = getAdapter();
    Adapter->call<UrApiKind::urMemRelease>(MInteropMemObject);
  }
}
const AdapterPtr &SYCLMemObjT::getAdapter() const {
  assert((MInteropContext != nullptr) &&
         "Trying to get Adapter from SYCLMemObjT with nullptr ContextImpl.");
  return (MInteropContext->getAdapter());
}

size_t SYCLMemObjT::getBufSizeForContext(const ContextImplPtr &Context,
                                         ur_native_handle_t MemObject) {
  size_t BufSize = 0;
  const AdapterPtr &Adapter = Context->getAdapter();
  // TODO is there something required to support non-OpenCL backends?
  Adapter->call<UrApiKind::urMemGetInfo>(
      detail::ur::cast<ur_mem_handle_t>(MemObject), UR_MEM_INFO_SIZE,
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
  // 1. The allocation is the first one and isn't on host. InitFromUserData
  // varies based on unified host memory support and whether or not the data can
  // be discarded.
  // 2. The allocation is not the first one and not on host. InitFromUserData ==
  // false, HostPtr is provided if the command is linked. The host pointer is
  // guaranteed to be reused in this case.
  if (!Context && !MOpenCLInterop && !MHostPtrReadOnly)
    InitFromUserData = true;

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
  {
    std::lock_guard<std::mutex> Lock(MCreateShadowCopyMtx);
    MCreateShadowCopy();
    MCreateShadowCopy = []() -> void {};
  }
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
