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
#include <CL/sycl/properties/buffer_properties.hpp>
#include <CL/sycl/properties/image_properties.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>

#include <cstring>
#include <memory>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

// Forward declarations
class context_impl;
class event_impl;
class plugin;

using ContextImplPtr = std::shared_ptr<context_impl>;
using EventImplPtr = std::shared_ptr<event_impl>;

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
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : MAllocator(std::move(Allocator)), MProps(Props), MInteropEvent(nullptr),
        MInteropContext(nullptr), MInteropMemObject(nullptr),
        MOpenCLInterop(false), MHostPtrReadOnly(false), MNeedWriteBack(true),
        MSizeInBytes(SizeInBytes), MUserPtr(nullptr), MShadowCopy(nullptr),
        MUploadDataFunctor(nullptr), MSharedPtrStorage(nullptr) {}

  SYCLMemObjT(const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : SYCLMemObjT(/*SizeInBytes*/ 0, Props, std::move(Allocator)) {}

  // For ABI compatibility
  SYCLMemObjT(cl_mem MemObject, const context &SyclContext,
              const size_t SizeInBytes, event AvailableEvent,
              std::unique_ptr<SYCLMemObjAllocator> Allocator);

  SYCLMemObjT(pi_native_handle MemObject, const context &SyclContext,
              const size_t SizeInBytes, event AvailableEvent,
              std::unique_ptr<SYCLMemObjAllocator> Allocator);

  SYCLMemObjT(cl_mem MemObject, const context &SyclContext,
              event AvailableEvent,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : SYCLMemObjT(MemObject, SyclContext, /*SizeInBytes*/ 0, AvailableEvent,
                    std::move(Allocator)) {}

  virtual ~SYCLMemObjT() = default;

  const plugin &getPlugin() const;

  __SYCL_DLL_LOCAL size_t getSize() const override { return MSizeInBytes; }
  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  __SYCL_DLL_LOCAL size_t get_count() const { return size(); }
  __SYCL_DLL_LOCAL size_t size() const noexcept {
    size_t AllocatorValueSize = MAllocator->getValueSize();
    return (getSize() + AllocatorValueSize - 1) / AllocatorValueSize;
  }

  template <typename propertyT> __SYCL_DLL_LOCAL bool has_property() const {
    return MProps.has_property<propertyT>();
  }

  template <typename propertyT>
  __SYCL_DLL_LOCAL propertyT get_property() const {
    return MProps.get_property<propertyT>();
  }

  __SYCL_DLL_LOCAL void
  addOrReplaceAccessorProperties(const property_list &PropertyList) {
    MProps.add_or_replace_accessor_properties(PropertyList);
  }

  __SYCL_DLL_LOCAL void
  deleteAccessorProperty(const PropWithDataKind &Kind) {
    MProps.delete_accessor_property(Kind);
  }

  template <typename AllocatorT>
  __SYCL_DLL_LOCAL AllocatorT get_allocator() const {
    return MAllocator->getAllocator<AllocatorT>();
  }

  __SYCL_DLL_LOCAL void *allocateHostMem() override {
    return MAllocator->allocate(size());
  }

  __SYCL_DLL_LOCAL void releaseHostMem(void *Ptr) override {
    if (Ptr)
      MAllocator->deallocate(Ptr, size());
  }

  void releaseMem(ContextImplPtr Context, void *MemAllocation) override;

  __SYCL_DLL_LOCAL void *getUserPtr() const {
    return MOpenCLInterop ? static_cast<void *>(MInteropMemObject) : MUserPtr;
  }

  __SYCL_DLL_LOCAL void set_write_back(bool NeedWriteBack) {
    MNeedWriteBack = NeedWriteBack;
  }

  __SYCL_DLL_LOCAL void set_final_data(std::nullptr_t) {
    MUploadDataFunctor = nullptr;
  }

  template <template <typename T> class PtrT, typename T>
  __SYCL_DLL_LOCAL
      enable_if_t<std::is_convertible<PtrT<T>, std::weak_ptr<T>>::value>
      set_final_data(PtrT<T> FinalData) {
    std::weak_ptr<T> TempFinalData(FinalData);
    set_final_data(TempFinalData);
  }

  template <typename T>
  __SYCL_DLL_LOCAL void set_final_data(std::weak_ptr<T> FinalData) {
    MUploadDataFunctor = [this, FinalData]() {
      if (std::shared_ptr<T> LockedFinalData = FinalData.lock()) {
        updateHostMemory(LockedFinalData.get());
      }
    };
  }

  __SYCL_DLL_LOCAL void set_final_data_from_storage() {
    MUploadDataFunctor = [this]() {
      if (MSharedPtrStorage.use_count() > 1) {
        void *FinalData = const_cast<void *>(MSharedPtrStorage.get());
        updateHostMemory(FinalData);
      }
    };
  }

  template <typename Destination>
  __SYCL_DLL_LOCAL EnableIfOutputPointerT<Destination>
  set_final_data(Destination FinalData) {
    if (!FinalData)
      MUploadDataFunctor = nullptr;
    else
      MUploadDataFunctor = [this, FinalData]() {
        updateHostMemory(FinalData);
      };
  }

  template <typename Destination>
  __SYCL_DLL_LOCAL EnableIfOutputIteratorT<Destination>
  set_final_data(Destination FinalData) {
    MUploadDataFunctor = [this, FinalData]() {
      using DestinationValueT = iterator_value_type_t<Destination>;
      // TODO if Destination is ContiguousIterator then don't create
      // ContiguousStorage. updateHostMemory works only with pointer to
      // continuous data.
      const size_t Size = MSizeInBytes / sizeof(DestinationValueT);
      std::unique_ptr<DestinationValueT[]> ContiguousStorage(
          new DestinationValueT[Size]);
      updateHostMemory(ContiguousStorage.get());
      std::copy(ContiguousStorage.get(), ContiguousStorage.get() + Size,
                FinalData);
    };
  }

protected:
  void updateHostMemory(void *const Ptr);

  // Update host with the latest data + notify scheduler that the memory object
  // is going to die. After this method is finished no further operations with
  // the memory object is allowed. This method is executed from child's
  // destructor. This cannot be done in SYCLMemObjT's destructor as child's
  // members must be alive.
  void updateHostMemory();

public:
  __SYCL_DLL_LOCAL bool useHostPtr() {
    return has_property<property::buffer::use_host_ptr>() ||
           has_property<property::image::use_host_ptr>();
  }

  __SYCL_DLL_LOCAL bool canReuseHostPtr(void *HostPtr,
                                        const size_t RequiredAlign) {
    bool Aligned =
        (reinterpret_cast<std::uintptr_t>(HostPtr) % RequiredAlign) == 0;
    return Aligned || useHostPtr();
  }

  __SYCL_DLL_LOCAL void handleHostData(void *HostPtr,
                                       const size_t RequiredAlign) {
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

  __SYCL_DLL_LOCAL void handleHostData(const void *HostPtr,
                                       const size_t RequiredAlign) {
    MHostPtrReadOnly = true;
    handleHostData(const_cast<void *>(HostPtr), RequiredAlign);
  }

  template <typename T>
  __SYCL_DLL_LOCAL void handleHostData(const std::shared_ptr<T> &HostPtr,
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
  __SYCL_DLL_LOCAL void handleHostData(InputIterator First, InputIterator Last,
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

  __SYCL_DLL_LOCAL void setAlign(size_t RequiredAlign) {
    MAllocator->setAlignment(RequiredAlign);
  }

  // For ABI compatibility
  static size_t getBufSizeForContext(const ContextImplPtr &Context,
                                     cl_mem MemObject);

  static size_t getBufSizeForContext(const ContextImplPtr &Context,
                                     pi_native_handle MemObject);

  __SYCL_DLL_LOCAL void *allocateMem(ContextImplPtr Context,
                                     bool InitFromUserData, void *HostPtr,
                                     RT::PiEvent &InteropEvent) override {
    (void)Context;
    (void)InitFromUserData;
    (void)HostPtr;
    (void)InteropEvent;
    throw runtime_error("Not implemented", PI_INVALID_OPERATION);
  }

  __SYCL_DLL_LOCAL MemObjType getType() const override {
    return MemObjType::Undefined;
  }

  ContextImplPtr getInteropContext() const override { return MInteropContext; }

  bool hasUserDataPtr() const { return MUserPtr != nullptr; };

  bool isInterop() const;

protected:
  // An allocateMem helper that determines which host ptr to use
  void determineHostPtr(const ContextImplPtr &Context, bool InitFromUserData,
                        void *&HostPtr, bool &HostPtrReadOnly);

  // Allocator used for allocation memory on host.
  std::unique_ptr<SYCLMemObjAllocator> MAllocator;
  // Properties passed by user.
  property_list MProps;
  // Event passed by user to interoperability constructor.
  // Should wait on this event before start working with such memory object.
  EventImplPtr MInteropEvent;
  // Context passed by user to interoperability constructor.
  ContextImplPtr MInteropContext;
  // OpenCL's memory object handle passed by user to interoperability
  // constructor.
  // TODO update this member to support other backends.
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
  std::shared_ptr<const void> MSharedPtrStorage;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
