//==------------ sycl_mem_obj_t.hpp - SYCL standard header file ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/sycl_mem_obj_i.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/sycl_mem_obj_allocator.hpp>
#include <sycl/detail/type_traits.hpp>
#include <sycl/event.hpp>
#include <sycl/properties/buffer_properties.hpp>
#include <sycl/properties/image_properties.hpp>
#include <sycl/property_list.hpp>
#include <sycl/stl.hpp>

#include <cstring>
#include <memory>
#include <type_traits>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace detail {

// Forward declarations
class context_impl;
class event_impl;
class plugin;

using ContextImplPtr = std::shared_ptr<context_impl>;
using EventImplPtr = std::shared_ptr<event_impl>;

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

public:
  SYCLMemObjT(const size_t SizeInBytes, const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : MAllocator(std::move(Allocator)), MProps(Props), MInteropEvent(nullptr),
        MInteropContext(nullptr), MInteropMemObject(nullptr),
        MOpenCLInterop(false), MHostPtrReadOnly(false), MNeedWriteBack(true),
        MSizeInBytes(SizeInBytes), MUserPtr(nullptr), MShadowCopy(nullptr),
        MUploadDataFunctor(nullptr), MSharedPtrStorage(nullptr),
        MHostPtrProvided(false) {}

  SYCLMemObjT(const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : SYCLMemObjT(/*SizeInBytes*/ 0, Props, std::move(Allocator)) {}

  SYCLMemObjT(pi_native_handle MemObject, const context &SyclContext,
              const size_t SizeInBytes, event AvailableEvent,
              std::unique_ptr<SYCLMemObjAllocator> Allocator);

  SYCLMemObjT(cl_mem MemObject, const context &SyclContext,
              event AvailableEvent,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : SYCLMemObjT(pi::cast<pi_native_handle>(MemObject), SyclContext,
                    /*SizeInBytes*/ (size_t)0, AvailableEvent,
                    std::move(Allocator)) {}

  SYCLMemObjT(pi_native_handle MemObject, const context &SyclContext,
              bool OwmNativeHandle, event AvailableEvent,
              std::unique_ptr<SYCLMemObjAllocator> Allocator);

  virtual ~SYCLMemObjT() = default;

  const plugin &getPlugin() const;

  size_t getSizeInBytes() const override { return MSizeInBytes; }
  __SYCL2020_DEPRECATED("get_count() is deprecated, please use size() instead")
  size_t get_count() const { return size(); }
  size_t size() const noexcept {
    size_t AllocatorValueSize = MAllocator->getValueSize();
    return (getSizeInBytes() + AllocatorValueSize - 1) / AllocatorValueSize;
  }

  template <typename propertyT> bool has_property() const noexcept {
    return MProps.has_property<propertyT>();
  }

  template <typename propertyT> propertyT get_property() const {
    return MProps.get_property<propertyT>();
  }

  void addOrReplaceAccessorProperties(const property_list &PropertyList) {
    MProps.add_or_replace_accessor_properties(PropertyList);
  }

  void deleteAccessorProperty(const PropWithDataKind &Kind) {
    MProps.delete_accessor_property(Kind);
  }

  const std::unique_ptr<SYCLMemObjAllocator> &get_allocator_internal() const {
    return MAllocator;
  }

  void *allocateHostMem() override { return MAllocator->allocate(size()); }

  void releaseHostMem(void *Ptr) override {
    if (Ptr)
      MAllocator->deallocate(Ptr, size());
  }

  void releaseMem(ContextImplPtr Context, void *MemAllocation) override;

  void *getUserPtr() const {
    return MOpenCLInterop ? static_cast<void *>(MInteropMemObject) : MUserPtr;
  }

  void set_write_back(bool NeedWriteBack) { MNeedWriteBack = NeedWriteBack; }

  void set_final_data(std::nullptr_t) { MUploadDataFunctor = nullptr; }

  void set_final_data_from_storage() {
    MUploadDataFunctor = [this]() {
      if (MSharedPtrStorage.use_count() > 1) {
        void *FinalData = const_cast<void *>(MSharedPtrStorage.get());
        updateHostMemory(FinalData);
      }
    };
    MHostPtrProvided = true;
  }

  void set_final_data(
      const std::function<void(const std::function<void(void *const Ptr)> &)>
          &FinalDataFunc) {

    auto UpdateFunc = [this](void *const Ptr) { updateHostMemory(Ptr); };
    MUploadDataFunctor = [FinalDataFunc, UpdateFunc]() {
      FinalDataFunc(UpdateFunc);
    };
    MHostPtrProvided = true;
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
    MHostPtrProvided = true;
    if (!MHostPtrReadOnly && HostPtr) {
      set_final_data([HostPtr](const std::function<void(void *const Ptr)> &F) {
        F(HostPtr);
      });
    }

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

  void handleHostData(const std::shared_ptr<void> &HostPtr,
                      const size_t RequiredAlign, bool IsConstPtr) {
    MHostPtrProvided = true;
    MSharedPtrStorage = HostPtr;
    MHostPtrReadOnly = IsConstPtr;
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

  void handleHostData(const std::function<void(void *)> &CopyFromInput,
                      const size_t RequiredAlign, bool IsConstPtr) {
    MHostPtrReadOnly = IsConstPtr;
    setAlign(RequiredAlign);
    if (useHostPtr())
      throw runtime_error(
          "Buffer constructor from a pair of iterator values does not support "
          "use_host_ptr property.",
          PI_ERROR_INVALID_OPERATION);

    setAlign(RequiredAlign);
    MShadowCopy = allocateHostMem();
    MUserPtr = MShadowCopy;

    CopyFromInput(MUserPtr);
  }

  void setAlign(size_t RequiredAlign) {
    MAllocator->setAlignment(RequiredAlign);
  }

  static size_t getBufSizeForContext(const ContextImplPtr &Context,
                                     pi_native_handle MemObject);

  // FIXME: Remove this when the class is removed from __SYCL_EXPORT
  virtual void *allocateMem(ContextImplPtr Context, bool InitFromUserData,
                            void *HostPtr, RT::PiEvent &InteropEvent) {
    (void)Context;
    (void)InitFromUserData;
    (void)HostPtr;
    (void)InteropEvent;
    assert(false && "Deprecated: use the overload with the device parameter");
  }

  void *allocateMem(ContextImplPtr Context, DeviceImplPtr Device,
                    bool InitFromUserData, void *HostPtr,
                    RT::PiEvent &InteropEvent) override {
    (void)Context;
    (void)Device;
    (void)InitFromUserData;
    (void)HostPtr;
    (void)InteropEvent;
    throw runtime_error("Not implemented", PI_ERROR_INVALID_OPERATION);
  }

  MemObjType getType() const override { return MemObjType::Undefined; }

  ContextImplPtr getInteropContext() const override { return MInteropContext; }

  bool hasUserDataPtr() const { return MUserPtr != nullptr; };

  bool isInterop() const;

  bool isHostPointerReadOnly() const { return MHostPtrReadOnly; }

  void detachMemoryObject(const std::shared_ptr<SYCLMemObjT> &Self) const;

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
  // Native backend memory object handle passed by user to interoperability
  // constructor.
  RT::PiMem MInteropMemObject;
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
  // Field to identify if dtor is not necessarily blocking.
  // check for MUploadDataFunctor is not enough to define it since for case when
  // we have read only HostPtr - MUploadDataFunctor is empty but delayed release
  // must be not allowed.
  bool MHostPtrProvided;
};
} // namespace detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
