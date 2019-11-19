//==------------ sycl_mem_obj_t.hpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/detail/sycl_mem_obj_i.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>

#include <type_traits>

namespace cl {
namespace sycl {
namespace detail {

// Forward declarations
class context_impl;
class event_impl;

using ContextImplPtr = shared_ptr_class<context_impl>;
using EventImplPtr = shared_ptr_class<event_impl>;

using sycl_memory_object_allocator = detail::aligned_allocator<char>;

// The class serves as a base for all SYCL memory objects.
template <typename AllocatorT> class SYCLMemObjT : public SYCLMemObjI {

  // The check for output iterator is commented out as it blocks set_final_data
  // with void * argument to be used.
  // TODO: Align these checks with the SYCL specification when the behaviour
  // with void * is clarified.
  template <typename T>
  using EnableIfOutputPointerT = enable_if_t<
      /*is_output_iterator<T>::value &&*/ std::is_pointer<T>::value>;

  template <typename T>
  using EnableIfOutputIteratorT = enable_if_t<
      /*is_output_iterator<T>::value &&*/ !std::is_pointer<T>::value>;

  template <typename T>
  using EnableIfDefaultAllocator =
      enable_if_t<std::is_same<T, sycl_memory_object_allocator>::value>;

  template <typename T>
  using EnableIfNonDefaultAllocator =
      enable_if_t<!std::is_same<T, sycl_memory_object_allocator>::value>;

public:
  SYCLMemObjT(const size_t SizeInBytes, const property_list &Props,
              AllocatorT Allocator)
      : MAllocator(Allocator), MProps(Props), MInteropEvent(nullptr),
        MInteropContext(nullptr), MInteropMemObject(nullptr),
        MOpenCLInterop(false), MHostPtrReadOnly(false), MNeedWriteBack(true),
        MSizeInBytes(SizeInBytes), MUserPtr(nullptr), MShadowCopy(nullptr),
        MUploadDataFunctor(nullptr), MSharedPtrStorage(nullptr) {}

  SYCLMemObjT(const property_list &Props, AllocatorT Allocator)
      : SYCLMemObjT(/*SizeInBytes*/ 0, Props, Allocator) {}

  SYCLMemObjT(const property_list &Props) : SYCLMemObjT(Props, AllocatorT()) {}

  SYCLMemObjT(cl_mem MemObject, const context &SyclContext,
              const size_t SizeInBytes, event AvailableEvent)
      : MAllocator(), MProps(),
        MInteropEvent(detail::getSyclObjImpl(std::move(AvailableEvent))),
        MInteropContext(detail::getSyclObjImpl(SyclContext)),
        MInteropMemObject(MemObject), MOpenCLInterop(true),
        MHostPtrReadOnly(false), MNeedWriteBack(true),
        MSizeInBytes(SizeInBytes), MUserPtr(nullptr), MShadowCopy(nullptr),
        MUploadDataFunctor(nullptr), MSharedPtrStorage(nullptr) {
    if (MInteropContext->is_host())
      throw cl::sycl::invalid_parameter_error(
          "Creation of interoperability memory object using host context is "
          "not allowed");

    RT::PiMem Mem = pi::cast<RT::PiMem>(MInteropMemObject);
    RT::PiContext Context = nullptr;
    PI_CALL(RT::piMemGetInfo, Mem, CL_MEM_CONTEXT, sizeof(Context), &Context,
            nullptr);

    if (MInteropContext->getHandleRef() != Context)
      throw cl::sycl::invalid_parameter_error(
          "Input context must be the same as the context of cl_mem");
    PI_CALL(RT::piMemRetain, Mem);
  }

  SYCLMemObjT(cl_mem MemObject, const context &SyclContext,
              event AvailableEvent)
      : SYCLMemObjT(MemObject, SyclContext, /*SizeInBytes*/ 0, AvailableEvent) {
  }

  size_t getSize() const override { return MSizeInBytes; }
  size_t get_count() const {
    auto constexpr AllocatorValueSize =
        sizeof(allocator_value_type_t<AllocatorT>);
    return (getSize() + AllocatorValueSize - 1) / AllocatorValueSize;
  }

  template <typename propertyT> bool has_property() const {
    return MProps.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return MProps.get_property<propertyT>();
  }

  AllocatorT get_allocator() const { return MAllocator; }

  void *allocateHostMem() override { return MAllocator.allocate(get_count()); }

  void releaseHostMem(void *Ptr) override {
    if (Ptr)
      MAllocator.deallocate(allocator_pointer_t<AllocatorT>(Ptr), get_count());
  }

  void releaseMem(ContextImplPtr Context, void *MemAllocation) override {
    void *Ptr = getUserPtr();
    return MemoryManager::releaseMemObj(Context, this, MemAllocation, Ptr);
  }

  void *getUserPtr() const {
    return MOpenCLInterop ? static_cast<void *>(MInteropMemObject) : MUserPtr;
  }

  void set_write_back(bool NeedWriteBack) { MNeedWriteBack = NeedWriteBack; }

  void set_final_data(std::nullptr_t) { MUploadDataFunctor = nullptr; }

  template <template <typename T> class PtrT, typename T>
  enable_if_t<std::is_convertible<PtrT<T>, weak_ptr_class<T>>::value>
  set_final_data(PtrT<T> FinalData) {
    weak_ptr_class<T> TempFinalData(FinalData);
    set_final_data(TempFinalData);
  }

  template <typename T> void set_final_data(weak_ptr_class<T> FinalData) {
    MUploadDataFunctor = [this, FinalData]() {
      if (shared_ptr_class<T> LockedFinalData = FinalData.lock()) {
        EventImplPtr Event = updateHostMemory(LockedFinalData.get());
        if (Event)
          Event->wait(Event);
      }
    };
  }

  void set_final_data_from_storage() {
    MUploadDataFunctor = [this]() {
      if (!MSharedPtrStorage.unique()) {
        void *FinalData = const_cast<void *>(MSharedPtrStorage.get());
        EventImplPtr Event = updateHostMemory(FinalData);
        if (Event)
          Event->wait(Event);
      }
    };
  }

  template <typename Destination>
  EnableIfOutputPointerT<Destination> set_final_data(Destination FinalData) {
    MUploadDataFunctor = [this, FinalData]() {
      EventImplPtr Event = updateHostMemory(FinalData);
      if (Event)
        Event->wait(Event);
    };
  }

  template <typename Destination>
  EnableIfOutputIteratorT<Destination> set_final_data(Destination FinalData) {
    MUploadDataFunctor = [this, FinalData]() {
      using DestinationValueT = iterator_value_type_t<Destination>;
      // TODO if Destination is ContiguousIterator then don't create
      // ContiguousStorage. updateHostMemory works only with pointer to
      // continuous data.
      const size_t Size = MSizeInBytes / sizeof(DestinationValueT);
      vector_class<DestinationValueT> ContiguousStorage(Size);
      EventImplPtr Event = updateHostMemory(ContiguousStorage.data());
      if (Event) {
        Event->wait(Event);
        std::copy(ContiguousStorage.cbegin(), ContiguousStorage.cend(),
                  FinalData);
      }
    };
  }

  EventImplPtr updateHostMemory(void *const Ptr) {
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
    return Event;
  }

  // Update host with the latest data + notify scheduler that the memory object
  // is going to die. After this method is finished no further operations with
  // the memory object is allowed. This method is executed from child's
  // destructor. This cannot be done in SYCLMemObjT's destructor as child's
  // members must be alive.
  void updateHostMemory() {
    if ((MUploadDataFunctor != nullptr) && MNeedWriteBack)
      MUploadDataFunctor();

    // If we're attached to a memory record, process the deletion of the memory
    // record. We may get detached before we do this.
    if (MRecord)
      Scheduler::getInstance().removeMemoryObject(this);
    releaseHostMem(MShadowCopy);

    if (MOpenCLInterop)
      PI_CALL(RT::piMemRelease, pi::cast<RT::PiMem>(MInteropMemObject));
  }

  bool useHostPtr() {
    return has_property<property::buffer::use_host_ptr>() ||
           has_property<property::image::use_host_ptr>();
  }

  bool canReuseHostPtr(void *HostPtr, const size_t RequiredAlign) {
    bool Aligned =
        (reinterpret_cast<std::uintptr_t>(HostPtr) % RequiredAlign) == 0;
    return Aligned || useHostPtr();
  }

  void handleHostData(void *HostPtr, const size_t RequiredAlign) {
    if (!MHostPtrReadOnly)
      set_final_data(reinterpret_cast<char *>(HostPtr));

    if (canReuseHostPtr(HostPtr, RequiredAlign)) {
      MUserPtr = HostPtr;
    } else {
      setAlign(RequiredAlign);
      MShadowCopy = allocateHostMem();
      MUserPtr = MShadowCopy;
      std::memcpy(MUserPtr, HostPtr, MSizeInBytes);
    }
  }

  void handleHostData(const void *HostPtr, const size_t RequiredAlign) {
    MHostPtrReadOnly = true;
    handleHostData(const_cast<void *>(HostPtr), RequiredAlign);
  }

  template <typename T>
  void handleHostData(const shared_ptr_class<T> &HostPtr,
                      const size_t RequiredAlign) {
    MSharedPtrStorage = HostPtr;
    MHostPtrReadOnly = std::is_const<T>::value;
    if (HostPtr) {
      if (!MHostPtrReadOnly)
        set_final_data_from_storage();

      if (canReuseHostPtr(HostPtr.get(), RequiredAlign))
        MUserPtr = HostPtr.get();
      else {
        setAlign(RequiredAlign);
        MShadowCopy = allocateHostMem();
        MUserPtr = MShadowCopy;
        std::memcpy(MUserPtr, HostPtr.get(), MSizeInBytes);
      }
    }
  }

  template <class InputIterator>
  void handleHostData(InputIterator First, InputIterator Last,
                      const size_t RequiredAlign) {
    MHostPtrReadOnly = iterator_to_const_type_t<InputIterator>::value;
    setAlign(RequiredAlign);
    if (useHostPtr())
      throw invalid_parameter_error(
          "Buffer constructor from a pair of iterator values does not support "
          "use_host_ptr property.");

    setAlign(RequiredAlign);
    MShadowCopy = allocateHostMem();
    MUserPtr = MShadowCopy;

    // We need to cast MUserPtr to pointer to the iteration type to get correct
    // offset in std::copy when it will increment destination pointer.
    using IteratorValueType = iterator_value_type_t<InputIterator>;
    using IteratorNonConstValueType = remove_const_t<IteratorValueType>;
    using IteratorPointerToNonConstValueType =
        add_pointer_t<IteratorNonConstValueType>;
    std::copy(First, Last,
              static_cast<IteratorPointerToNonConstValueType>(MUserPtr));
  }

  template <typename T = AllocatorT>
  EnableIfNonDefaultAllocator<T> setAlign(size_t RequiredAlign) {
    // Do nothing in case of user's allocator.
  }

  template <typename T = AllocatorT>
  EnableIfDefaultAllocator<T> setAlign(size_t RequiredAlign) {
    MAllocator.setAlignment(std::max<size_t>(RequiredAlign, 64));
  }

protected:
  // Allocator used for allocation memory on host.
  AllocatorT MAllocator;
  // Properties passed by user.
  property_list MProps;
  // Event passed by user to interoperability constructor.
  // Should wait on this event before start working with such memory object.
  EventImplPtr MInteropEvent;
  // Context passed by user to interoperability constructor.
  ContextImplPtr MInteropContext;
  // OpenCL's memory object handle passed by user to interoperability
  // constructor.
  cl_mem MInteropMemObject;
  // Indicates whether memory object is created using interoperability
  // constructor or not.
  bool MOpenCLInterop;
  // Indicates if user provided pointer is read only.
  bool MHostPtrReadOnly;
  // Indicates if memory object should write memory to the host on destruction.
  bool MNeedWriteBack;
  // Size of memory.
  size_t MSizeInBytes;
  // User's pointer passed to constructor.
  void *MUserPtr;
  // Copy of memory passed by user to constructor.
  void *MShadowCopy;
  // Function which update host with final data on memory object destruction.
  std::function<void(void)> MUploadDataFunctor;
  // Field which holds user's shared_ptr in case of memory object is created
  // using constructor with shared_ptr.
  shared_ptr_class<const void> MSharedPtrStorage;
};

} // namespace detail
} // namespace sycl
} // namespace cl
