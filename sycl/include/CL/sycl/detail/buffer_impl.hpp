//==----------------- buffer_impl.hpp - SYCL standard header file ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/cl.h>
#include <CL/sycl/access/access.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/aligned_allocator.hpp>
#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/detail/memory_manager.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/scheduler/scheduler.hpp>
#include <CL/sycl/detail/sycl_mem_obj_t.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/property_list.hpp>
#include <CL/sycl/stl.hpp>
#include <CL/sycl/types.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

namespace cl {
namespace sycl {
// Forward declarations
template <typename DataT, int Dimensions, access::mode AccessMode,
          access::target AccessTarget, access::placeholder IsPlaceholder>
class accessor;
template <typename T, int Dimensions, typename AllocatorT> class buffer;
class handler;

using buffer_allocator = detail::sycl_memory_object_allocator;

namespace detail {

template <typename AllocatorT>
class buffer_impl final : public SYCLMemObjT<AllocatorT> {
  using BaseT = SYCLMemObjT<AllocatorT>;
  using typename BaseT::MemObjType;

public:
  buffer_impl(size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props, AllocatorT Allocator = AllocatorT())
      : BaseT(SizeInBytes, Props, Allocator) {}

  buffer_impl(void *HostData, size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props, AllocatorT Allocator = AllocatorT())
      : BaseT(SizeInBytes, Props, Allocator) {
    BaseT::handleHostData(HostData, RequiredAlign);
  }

  buffer_impl(const void *HostData, size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props, AllocatorT Allocator = AllocatorT())
      : BaseT(SizeInBytes, Props, Allocator) {
    BaseT::handleHostData(HostData, RequiredAlign);
  }

  template <typename T>
  buffer_impl(const shared_ptr_class<T> &HostData, const size_t SizeInBytes,
              size_t RequiredAlign, const property_list &Props,
              AllocatorT Allocator = AllocatorT())
      : BaseT(SizeInBytes, Props, Allocator) {
    BaseT::handleHostData(HostData, RequiredAlign);
  }

  template <typename T>
  using EnableIfNotConstIterator =
      enable_if_t<!iterator_to_const_type_t<T>::value, T>;

  template <class InputIterator>
  buffer_impl(EnableIfNotConstIterator<InputIterator> First, InputIterator Last,
              const size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props, AllocatorT Allocator = AllocatorT())
      : BaseT(SizeInBytes, Props, Allocator) {
    BaseT::handleHostData(First, Last, RequiredAlign);
    // TODO: There is contradiction in the spec, in one place it says
    // the data is not copied back at all if the buffer is construted
    // using this c'tor, another section says that the data will be
    // copied back if iterators passed are not const ( 4.7.2.3 Buffer
    // Synchronization Rules and this constructor description)
    BaseT::set_final_data(First);
  }

  template <typename T>
  using EnableIfConstIterator =
      enable_if_t<iterator_to_const_type_t<T>::value, T>;

  template <class InputIterator>
  buffer_impl(EnableIfConstIterator<InputIterator> First, InputIterator Last,
              const size_t SizeInBytes, size_t RequiredAlign,
              const property_list &Props, AllocatorT Allocator = AllocatorT())
      : BaseT(SizeInBytes, Props, Allocator) {
    BaseT::handleHostData(First, Last, RequiredAlign);
  }

  buffer_impl(cl_mem MemObject, const context &SyclContext,
              const size_t SizeInBytes, event AvailableEvent = {})
      : BaseT(MemObject, SyclContext, SizeInBytes, std::move(AvailableEvent)) {}

  template <typename T, int Dimensions, access::mode Mode,
            access::target Target = access::target::global_buffer>
  accessor<T, Dimensions, Mode, Target, access::placeholder::false_t>
  get_access(buffer<T, Dimensions, AllocatorT> &Buffer,
             handler &CommandGroupHandler) {
    return accessor<T, Dimensions, Mode, Target, access::placeholder::false_t>(
        Buffer, CommandGroupHandler);
  }

  template <typename T, int Dimensions, access::mode Mode>
  accessor<T, Dimensions, Mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access(buffer<T, Dimensions, AllocatorT> &Buffer) {
    return accessor<T, Dimensions, Mode, access::target::host_buffer,
                    access::placeholder::false_t>(Buffer);
  }

  template <typename T, int dimensions, access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t>
  get_access(buffer<T, dimensions, AllocatorT> &Buffer,
             handler &commandGroupHandler, range<dimensions> accessRange,
             id<dimensions> accessOffset) {
    return accessor<T, dimensions, mode, target, access::placeholder::false_t>(
        Buffer, commandGroupHandler, accessRange, accessOffset);
  }

  template <typename T, int dimensions, access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access(buffer<T, dimensions, AllocatorT> &Buffer,
             range<dimensions> accessRange, id<dimensions> accessOffset) {
    return accessor<T, dimensions, mode, access::target::host_buffer,
                    access::placeholder::false_t>(Buffer, accessRange,
                                                  accessOffset);
  }

  void *allocateMem(ContextImplPtr Context, bool InitFromUserData,
                    RT::PiEvent &OutEventToWait) override {

    void *UserPtr = InitFromUserData ? BaseT::getUserPtr() : nullptr;

    return MemoryManager::allocateMemBuffer(
        std::move(Context), this, UserPtr, BaseT::MHostPtrReadOnly,
        BaseT::getSize(), BaseT::MInteropEvent, BaseT::MInteropContext,
        OutEventToWait);
  }

  MemObjType getType() const override { return MemObjType::BUFFER; }

  ~buffer_impl() { BaseT::updateHostMemory(); }
};

} // namespace detail
} // namespace sycl
} // namespace cl
