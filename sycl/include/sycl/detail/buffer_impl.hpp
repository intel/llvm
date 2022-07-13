//==----------------- buffer_impl.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sycl/detail/pi.h"
#include <sycl/access/access.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/detail/export.hpp>
#include <sycl/detail/helpers.hpp>
#include <sycl/detail/sycl_mem_obj_t.hpp>
#include <sycl/property_list.hpp>
#include <sycl/stl.hpp>
#include <sycl/types.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declarations
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder,
          typename PropertyListT>
class accessor;
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;
template <typename DataT, int Dimensions, access::mode AccessMode>
class host_accessor;

namespace detail {

class __SYCL_EXPORT buffer_impl final : public SYCLMemObjT {
  using BaseT = SYCLMemObjT;
  using typename BaseT::MemObjType;

public:
  buffer_impl(size_t SizeInBytes, size_t, const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {

    if (Props.has_property<sycl::property::buffer::use_host_ptr>())
      throw sycl::invalid_object_error(
          "The use_host_ptr property requires host pointer to be provided",
          PI_ERROR_INVALID_OPERATION);
  }

  buffer_impl(void *HostData, size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {

    if (Props.has_property<
            sycl::ext::oneapi::property::buffer::use_pinned_host_memory>())
      throw sycl::invalid_object_error(
          "The use_pinned_host_memory cannot be used with host pointer",
          PI_ERROR_INVALID_OPERATION);

    BaseT::handleHostData(HostData, RequiredAlign);
  }

  buffer_impl(const void *HostData, size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {

    if (Props.has_property<
            sycl::ext::oneapi::property::buffer::use_pinned_host_memory>())
      throw sycl::invalid_object_error(
          "The use_pinned_host_memory cannot be used with host pointer",
          PI_ERROR_INVALID_OPERATION);

    BaseT::handleHostData(HostData, RequiredAlign);
  }

  template <typename T>
  buffer_impl(const std::shared_ptr<T> &HostData, const size_t SizeInBytes,
              size_t RequiredAlign, const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {

    if (Props.has_property<
            sycl::ext::oneapi::property::buffer::use_pinned_host_memory>())
      throw sycl::invalid_object_error(
          "The use_pinned_host_memory cannot be used with host pointer",
          PI_ERROR_INVALID_OPERATION);

    BaseT::handleHostData(HostData, RequiredAlign);
  }

  template <typename T>
  using EnableIfNotConstIterator =
      enable_if_t<!iterator_to_const_type_t<T>::value, T>;

  template <class InputIterator>
  buffer_impl(EnableIfNotConstIterator<InputIterator> First, InputIterator Last,
              const size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {

    if (Props.has_property<sycl::property::buffer::use_host_ptr>())
      throw sycl::invalid_object_error(
          "Buffer constructor from a pair of iterator values cannot have the "
          "use_host_ptr property.",
          PI_ERROR_INVALID_OPERATION);

    BaseT::handleHostData(First, Last, RequiredAlign);
  }

  template <typename T>
  using EnableIfConstIterator =
      enable_if_t<iterator_to_const_type_t<T>::value, T>;

  template <class InputIterator>
  buffer_impl(EnableIfConstIterator<InputIterator> First, InputIterator Last,
              const size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              std::unique_ptr<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {

    if (Props.has_property<sycl::property::buffer::use_host_ptr>())
      throw sycl::invalid_object_error(
          "Buffer constructor from a pair of iterator values cannot have the "
          "use_host_ptr property.",
          PI_ERROR_INVALID_OPERATION);

    BaseT::handleHostData(First, Last, RequiredAlign);
  }

  buffer_impl(cl_mem MemObject, const context &SyclContext,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              event AvailableEvent)
      : buffer_impl(pi::cast<pi_native_handle>(MemObject), SyclContext,
                    std::move(Allocator), /*OwnNativeHandle*/ true,
                    std::move(AvailableEvent)) {}

  buffer_impl(pi_native_handle MemObject, const context &SyclContext,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              bool OwnNativeHandle, event AvailableEvent)
      : BaseT(MemObject, SyclContext, OwnNativeHandle,
              std::move(AvailableEvent), std::move(Allocator)) {}

  // TODO: remove the following 2 constructors when it is allowed to break ABI.
  buffer_impl(cl_mem MemObject, const context &SyclContext,
              const size_t SizeInBytes,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              event AvailableEvent)
      : buffer_impl(pi::cast<pi_native_handle>(MemObject), SyclContext,
                    SizeInBytes, std::move(Allocator),
                    std::move(AvailableEvent)) {}

  buffer_impl(pi_native_handle MemObject, const context &SyclContext,
              const size_t SizeInBytes,
              std::unique_ptr<SYCLMemObjAllocator> Allocator,
              event AvailableEvent)
      : BaseT(MemObject, SyclContext, SizeInBytes, std::move(AvailableEvent),
              std::move(Allocator)) {}

  void *allocateMem(ContextImplPtr Context, bool InitFromUserData,
                    void *HostPtr, RT::PiEvent &OutEventToWait) override;
  void constructorNotification(const detail::code_location &CodeLoc,
                               void *UserObj, const void *HostObj,
                               const void *Type, uint32_t Dim,
                               uint32_t ElemType, size_t Range[3]);
  // TODO: remove once ABI break is allowed
  void constructorNotification(const detail::code_location &CodeLoc,
                               void *UserObj);
  void destructorNotification(void *UserObj);

  MemObjType getType() const override { return MemObjType::Buffer; }

  ~buffer_impl() {
    try {
      BaseT::updateHostMemory();
    } catch (...) {
    }
    destructorNotification(this);
  }

  void resize(size_t size) { BaseT::MSizeInBytes = size; }

  void addInteropObject(std::vector<pi_native_handle> &Handles) const;

  std::vector<pi_native_handle> getNativeVector(backend BackendName) const;
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
