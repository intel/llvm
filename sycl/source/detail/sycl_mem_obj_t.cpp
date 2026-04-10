//==------------ sycl_mem_obj_t.cpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/adapter_impl.hpp>
#include <detail/context_impl.hpp>
#include <detail/event_impl.hpp>
#include <detail/memory_manager.hpp>
#include <detail/scheduler/scheduler.hpp>
#include <detail/sycl_mem_obj_t.hpp>

#include <cstdint>

namespace sycl {
inline namespace _V1 {
namespace detail {

namespace {

size_t getBackendShadowCopyAlignment(context_impl *Context) {
  size_t RequiredAlign = 1;
  for (const auto &Device : Context->getDevices()) {
    const uint32_t AlignBits =
        Device.get_info<info::device::mem_base_addr_align>();
    if (AlignBits == 0)
      continue;

    // UR reports MEM_BASE_ADDR_ALIGN in bits.
    const size_t AlignBytes = (static_cast<size_t>(AlignBits) + 7) / 8;
    if (AlignBytes > RequiredAlign)
      RequiredAlign = AlignBytes;
  }
  return RequiredAlign;
}

} // namespace

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
  adapter_impl &Adapter = getAdapter();

  ur_mem_native_properties_t MemProperties = {
      UR_STRUCTURE_TYPE_MEM_NATIVE_PROPERTIES, nullptr, OwnNativeHandle};
  Adapter.call<UrApiKind::urMemBufferCreateWithNativeHandle>(
      MemObject, MInteropContext->getHandleRef(), &MemProperties,
      &MInteropMemObject);

  // Get the size of the buffer in bytes
  Adapter.call<UrApiKind::urMemGetInfo>(MInteropMemObject, UR_MEM_INFO_SIZE,
                                        sizeof(size_t), &MSizeInBytes, nullptr);

  Adapter.call<UrApiKind::urMemGetInfo>(MInteropMemObject, UR_MEM_INFO_CONTEXT,
                                        sizeof(Context), &Context, nullptr);

  if (MInteropContext->getHandleRef() != Context)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Input context must be the same as the context of cl_mem");

  if (MInteropContext->getBackend() == backend::opencl) {
    __SYCL_OCL_CALL(clRetainMemObject, ur::cast<cl_mem>(MemObject));
  }
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
  adapter_impl &Adapter = getAdapter();

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

  Adapter.call<UrApiKind::urMemImageCreateWithNativeHandle>(
      MemObject, MInteropContext->getHandleRef(), &Format, &Desc,
      &NativeProperties, &MInteropMemObject);

  Adapter.call<UrApiKind::urMemGetInfo>(MInteropMemObject, UR_MEM_INFO_CONTEXT,
                                        sizeof(Context), &Context, nullptr);

  if (MInteropContext->getHandleRef() != Context)
    throw sycl::exception(
        make_error_code(errc::invalid),
        "Input context must be the same as the context of cl_mem");

  if (MInteropContext->getBackend() == backend::opencl) {
    __SYCL_OCL_CALL(clRetainMemObject, ur::cast<cl_mem>(MemObject));
  }
}

void SYCLMemObjT::releaseMem(context_impl *Context, void *MemAllocation) {
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
    Event->wait();
}

void SYCLMemObjT::updateHostMemory() {
  // Don't try updating host memory when shutting down.
  if ((MUploadDataFunctor != nullptr) && MNeedWriteBack &&
      !MBackendOwnsWriteBack && GlobalHandler::instance().isOkToDefer())
    MUploadDataFunctor();

  // If we're attached to a memory record, process the deletion of the memory
  // record. We may get detached before we do this.
  if (MRecord) {
    // Don't strictly try holding the lock in removeMemoryObject during shutdown
    // to prevent deadlocks.
    bool Result = Scheduler::getInstance().removeMemoryObject(
        this, GlobalHandler::instance().isOkToDefer());
    std::ignore = Result; // for no assert build

    // removeMemoryObject might fail during shutdown because of not being
    // able to hold write lock. This can happen if shutdown happens due to
    // exception/termination while holding lock.
    assert(
        (Result || !GlobalHandler::instance().isOkToDefer()) &&
        "removeMemoryObject should not return false in mem object destructor");
  }
  detail::OSUtil::alignedFree(MShadowCopy);

  if (MOpenCLInterop) {
    getAdapter().call<UrApiKind::urMemRelease>(MInteropMemObject);
  }
}

void SYCLMemObjT::materializeShadowCopy(const void *SourcePtr,
                                        size_t RequiredAlign) {
  if (MPendingShadowCopyAlignment > RequiredAlign)
    RequiredAlign = MPendingShadowCopyAlignment;

  if (RequiredAlign == 0)
    RequiredAlign = 1;

  MPendingShadowCopyAlignment = RequiredAlign;

  void *OldUserPtr = MUserPtr;
  void *OldShadowCopy = MShadowCopy;
  const void *CopySource = SourcePtr;
  if (OldShadowCopy) {
    if ((reinterpret_cast<std::uintptr_t>(OldShadowCopy) % RequiredAlign) ==
        0) {
      MUserPtr = OldShadowCopy;
      return;
    }
    CopySource = OldShadowCopy;
  }

  assert(CopySource != nullptr &&
         "Cannot materialize a shadow copy without source data");

  // Allocate the shadow copy via the platform-aligned allocator directly,
  // bypassing the user-provided allocator. Shadow copies are an internal
  // runtime detail; the user allocator cannot be relied upon to satisfy
  // backend alignment requirements (e.g. CL_DEVICE_MEM_BASE_ADDR_ALIGN).
  const size_t AllocBytes =
      MSizeInBytes == 0 ? RequiredAlign
                        : ((MSizeInBytes + RequiredAlign - 1) / RequiredAlign) *
                              RequiredAlign;
  void *NewShadowCopy = detail::OSUtil::alignedAlloc(RequiredAlign, AllocBytes);
  if (!NewShadowCopy)
    throw std::bad_alloc();
  if (MSizeInBytes != 0)
    std::memcpy(NewShadowCopy, CopySource, MSizeInBytes);

  MShadowCopy = NewShadowCopy;
  MUserPtr = NewShadowCopy;
  updateRecordedMemAllocation(OldUserPtr, NewShadowCopy);

  detail::OSUtil::alignedFree(OldShadowCopy);
}

void SYCLMemObjT::updateRecordedMemAllocation(void *OldPtr, void *NewPtr) {
  if (MRecord == nullptr || OldPtr == nullptr || OldPtr == NewPtr)
    return;

  for (auto *AllocaCmd : MRecord->MAllocaCommands) {
    if (AllocaCmd->MMemAllocation == OldPtr)
      AllocaCmd->MMemAllocation = NewPtr;
  }
}

adapter_impl &SYCLMemObjT::getAdapter() const {
  assert((MInteropContext != nullptr) &&
         "Trying to get Adapter from SYCLMemObjT with nullptr ContextImpl.");
  return MInteropContext->getAdapter();
}

bool SYCLMemObjT::isInterop() const { return MOpenCLInterop; }

void SYCLMemObjT::prepareForAllocation(context_impl *Context) {
  // Context may be null for host allocations; nothing backend-specific to do.
  if (!Context)
    return;

  if (!MHasPendingAlignedShadowCopy)
    return;

  bool SkipShadowCopy = false;
  backend Backend = Context->getPlatformImpl().getBackend();
  auto Devices = Context->getDevices();
  if (Devices.size() != 0)
    Backend = Devices.front().getBackend();

  const size_t BackendRequiredAlign = getBackendShadowCopyAlignment(Context);
  if (BackendRequiredAlign > MPendingShadowCopyAlignment)
    MPendingShadowCopyAlignment = BackendRequiredAlign;

  switch (Backend) {
  case backend::ext_oneapi_level_zero:
  case backend::ext_oneapi_cuda:
  case backend::ext_oneapi_hip:
    SkipShadowCopy = true;
    break;
  case backend::opencl:
  case backend::ext_oneapi_native_cpu:
  case backend::ext_oneapi_offload:
    SkipShadowCopy = false;
    break;
  case backend::all:
  default:
    assert(false && "Unexpected SYCL backend");
    break;
  }

  std::lock_guard<std::mutex> Lock(MCreateShadowCopyMtx);
  if (SkipShadowCopy) {
    if (MShadowCopy != nullptr) {
      // A writable host accessor already forced a SYCL shadow copy. Keep using
      // that path so the final copy-back still targets the original user ptr.
      return;
    }

    // Backend (UR) will manage the misaligned host pointer through its own
    // internal staging buffer and owns the final copy-back to the original ptr.
    MCreateShadowCopy = []() -> void {};
    MBackendOwnsWriteBack = true;
    if (!MHostPtrReadOnly)
      MUploadDataFunctor = nullptr;
    MHasPendingAlignedShadowCopy = false;
    return;
  }

  materializeShadowCopy(MUserPtr, BackendRequiredAlign);
  MCreateShadowCopy = []() -> void {};
  MBackendOwnsWriteBack = false;
  MHasPendingAlignedShadowCopy = false;
}

void SYCLMemObjT::determineHostPtr(context_impl *Context, bool InitFromUserData,
                                   void *&HostPtr, bool &HostPtrReadOnly) {
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
  updateRecordedMemAllocation(InitialUserPtr, MUserPtr);
}

} // namespace detail
} // namespace _V1
} // namespace sycl
