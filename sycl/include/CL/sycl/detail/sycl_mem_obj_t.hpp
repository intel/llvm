//==------------ sycl_mem_obj_t.hpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/sycl_mem_obj_allocator.hpp>
#include <CL/sycl/detail/sycl_mem_obj_i.hpp>
#include <CL/sycl/detail/type_traits.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>

#include <cstring>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// Forward declarations
class context_impl;
class event_impl;
class plugin;

using ContextImplPtr = shared_ptr_class<context_impl>;
using EventImplPtr = shared_ptr_class<event_impl>;

template <typename T>
class aligned_allocator;
using sycl_memory_object_allocator = aligned_allocator<char>;

// The class serves as a base for all SYCL memory objects.
class __SYCL_EXPORT SYCLMemObjT : public SYCLMemObjI {

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
              unique_ptr_class<SYCLMemObjAllocator> Allocator)
      : MAllocator(std::move(Allocator)), MProps(Props), MInteropEvent(nullptr),
        MInteropContext(nullptr), MInteropMemObject(nullptr),
        MOpenCLInterop(false), MHostPtrReadOnly(false), MNeedWriteBack(true),
        MSizeInBytes(SizeInBytes), MUserPtr(nullptr), MShadowCopy(nullptr),
        MUploadDataFunctor(nullptr), MSharedPtrStorage(nullptr) {}

  SYCLMemObjT(const property_list &Props,
              unique_ptr_class<SYCLMemObjAllocator> Allocator)
      : SYCLMemObjT(/*SizeInBytes*/ 0, Props, std::move(Allocator)) {}

  SYCLMemObjT(cl_mem MemObject, const context &SyclContext,
              const size_t SizeInBytes, event AvailableEvent,
              unique_ptr_class<SYCLMemObjAllocator> Allocator);

  SYCLMemObjT(cl_mem MemObject, const context &SyclContext,
              event AvailableEvent,
              unique_ptr_class<SYCLMemObjAllocator> Allocator)
      : SYCLMemObjT(MemObject, SyclContext, /*SizeInBytes*/ 0, AvailableEvent,
                    std::move(Allocator)) {}

  virtual ~SYCLMemObjT() = default;

  const plugin &getPlugin() const;

  size_t getSize() const override { return MSizeInBytes; }
  size_t get_count() const {
    size_t AllocatorValueSize = MAllocator->getValueSize();
    return (getSize() + AllocatorValueSize - 1) / AllocatorValueSize;
  }

  template <typename propertyT> bool has_property() const {
    return MProps.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return MProps.get_property<propertyT>();
  }

  template <typename AllocatorT> AllocatorT get_allocator() const {
    return MAllocator->getAllocator<AllocatorT>();
  }

  void *allocateHostMem() override { return MAllocator->allocate(get_count()); }

  void releaseHostMem(void *Ptr) override {
    if (Ptr)
      MAllocator->deallocate(Ptr, get_count());
  }

  void releaseMem(ContextImplPtr Context, void *MemAllocation) override;

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
        updateHostMemory(LockedFinalData.get());
      }
    };
  }

  void set_final_data_from_storage() {
    MUploadDataFunctor = [this]() {
      if (!MSharedPtrStorage.unique()) {
        void *FinalData = const_cast<void *>(MSharedPtrStorage.get());
        updateHostMemory(FinalData);
      }
    };
  }

  template <typename Destination>
  EnableIfOutputPointerT<Destination> set_final_data(Destination FinalData) {
    if (!FinalData)
      MUploadDataFunctor = nullptr;
    else
      MUploadDataFunctor = [this, FinalData]() {
        updateHostMemory(FinalData);
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
      updateHostMemory(ContiguousStorage.data());
      std::copy(ContiguousStorage.cbegin(), ContiguousStorage.cend(),
                FinalData);
    };
  }

  void updateHostMemory(void *const Ptr);

  // Update host with the latest data + notify scheduler that the memory object
  // is going to die. After this method is finished no further operations with
  // the memory object is allowed. This method is executed from child's
  // destructor. This cannot be done in SYCLMemObjT's destructor as child's
  // members must be alive.
  void updateHostMemory();

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
      throw runtime_error(
          "Buffer constructor from a pair of iterator values does not support "
          "use_host_ptr property.",
          PI_INVALID_OPERATION);

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

  void setAlign(size_t RequiredAlign) {
    MAllocator->setAlignment(RequiredAlign);
  }

  static size_t getBufSizeForContext(const ContextImplPtr &Context,
                                     cl_mem MemObject);

  void *allocateMem(ContextImplPtr Context, bool InitFromUserData,
                    void *HostPtr, RT::PiEvent &InteropEvent) override {
    (void)Context;
    (void)InitFromUserData;
    (void)HostPtr;
    (void)InteropEvent;
    throw runtime_error("Not implemented", PI_INVALID_OPERATION);
  }

  MemObjType getType() const override { return UNDEFINED; }

protected:
  // Allocator used for allocation memory on host.
  unique_ptr_class<SYCLMemObjAllocator> MAllocator;
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
  function_class<void(void)> MUploadDataFunctor;
  // Field which holds user's shared_ptr in case of memory object is created
  // using constructor with shared_ptr.
  shared_ptr_class<const void> MSharedPtrStorage;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
