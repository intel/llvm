//==----------------- buffer_impl.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/access/access.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/buffer_usage.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/sycl_mem_obj_t.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>
#include <CL/sycl/types.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
// Forward declarations
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class accessor;
template <typename T, int Dimensions, typename AllocatorT, typename Enable>
class buffer;
template <typename DataT, int Dimensions, access::mode AccessMode>
class host_accessor;

using buffer_allocator = detail::sycl_memory_object_allocator;

namespace detail {

class __SYCL_EXPORT buffer_impl final : public SYCLMemObjT {
  using BaseT = SYCLMemObjT;
  using typename BaseT::MemObjType;

public:
  buffer_impl(size_t SizeInBytes, size_t, const property_list &Props,
              unique_ptr_class<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {}

  buffer_impl(void *HostData, size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              unique_ptr_class<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {
    BaseT::handleHostData(HostData, RequiredAlign);
  }

  buffer_impl(const void *HostData, size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              unique_ptr_class<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {
    BaseT::handleHostData(HostData, RequiredAlign);
  }

  template <typename T>
  buffer_impl(const shared_ptr_class<T> &HostData, const size_t SizeInBytes,
              size_t RequiredAlign, const property_list &Props,
              unique_ptr_class<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {
    BaseT::handleHostData(HostData, RequiredAlign);
  }

  template <typename T>
  using EnableIfNotConstIterator =
      enable_if_t<!iterator_to_const_type_t<T>::value, T>;

  template <class InputIterator>
  buffer_impl(EnableIfNotConstIterator<InputIterator> First, InputIterator Last,
              const size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              unique_ptr_class<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {
    BaseT::handleHostData(First, Last, RequiredAlign);
  }

  template <typename T>
  using EnableIfConstIterator =
      enable_if_t<iterator_to_const_type_t<T>::value, T>;

  template <class InputIterator>
  buffer_impl(EnableIfConstIterator<InputIterator> First, InputIterator Last,
              const size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props,
              unique_ptr_class<SYCLMemObjAllocator> Allocator)
      : BaseT(SizeInBytes, Props, std::move(Allocator)) {
    BaseT::handleHostData(First, Last, RequiredAlign);
  }

  buffer_impl(cl_mem MemObject, const context &SyclContext,
              const size_t SizeInBytes,
              unique_ptr_class<SYCLMemObjAllocator> Allocator,
              event AvailableEvent)
      : BaseT(MemObject, SyclContext, SizeInBytes, std::move(AvailableEvent),
              std::move(Allocator)) {}

  void *allocateMem(ContextImplPtr Context, bool InitFromUserData,
                    void *HostPtr, RT::PiEvent &OutEventToWait) override;

  MemObjType getType() const override { return MemObjType::BUFFER; }

  ~buffer_impl() {
    if (hasSubBuffers()) {
      copyBackAnyRemainingData();
      MNeedWriteBack = false; // clear this to prevent an additional copy back
                              // when we release memory below.
    }
    try {
      BaseT::updateHostMemory(); // also releases memory and handles.
    } catch (...) {
    }
  }

  void resize(size_t size) { BaseT::MSizeInBytes = size; }

protected:
  template <typename T, int Dimensions, typename AllocatorT, typename Enable>
  friend class sycl::buffer;

  // deque of buffer_info, if any.
  std::deque<buffer_usage> MBufferUsageDQ;

  // if this MemObj is backing a buffer (and sub-buffers), provide information
  // to help with copy-back decisions.
  void recordBufferUsage(const void *const BuffPtr, const size_t Sz,
                         const size_t Offset, const bool IsSub);
  void recordAccessorUsage(const void *const BuffPtr, access::mode Mode,
                           handler &CGH);
  void recordAccessorUsage(const void *const BuffPtr, access::mode Mode);

  bool hasSubBuffers();
  void set_write_back(bool flag);
  void set_write_back(bool flag, const void *const BuffPtr);

  void copyBackSubBuffer(detail::when_copyback now, const void *const BuffPtr);
  void copyBackAnyRemainingData();
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
