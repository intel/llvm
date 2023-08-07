//==------- sycl_mem_obj_allocator.hpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/aligned_allocator.hpp> // for aligned_allocator

#include <algorithm>   // for max
#include <cstddef>     // for size_t
#include <type_traits> // for enable_if_t

namespace sycl {
inline namespace _V1 {
namespace detail {

template <typename DataT>
using sycl_memory_object_allocator = aligned_allocator<DataT>;

class SYCLMemObjAllocator {

protected:
  virtual void *getAllocatorImpl() = 0;

public:
  virtual ~SYCLMemObjAllocator() = default;
  virtual void *allocate(std::size_t) = 0;
  virtual void deallocate(void *, std::size_t) = 0;
  virtual std::size_t getValueSize() const = 0;
  virtual void setAlignment(std::size_t RequiredAlign) = 0;
  template <typename AllocatorT> AllocatorT getAllocator() {
    return *reinterpret_cast<AllocatorT *>(getAllocatorImpl());
  }
};

template <typename AllocatorT, typename OwnerDataT>
class SYCLMemObjAllocatorHolder : public SYCLMemObjAllocator {
public:
  SYCLMemObjAllocatorHolder(AllocatorT Allocator)
      : MAllocator(Allocator),
        MValueSize(sizeof(typename AllocatorT::value_type)) {}

  SYCLMemObjAllocatorHolder()
      : MAllocator(AllocatorT()),
        MValueSize(sizeof(typename AllocatorT::value_type)) {}

  ~SYCLMemObjAllocatorHolder() = default;

  virtual void *allocate(std::size_t Count) override {
    return reinterpret_cast<void *>(MAllocator.allocate(Count));
  }

  virtual void deallocate(void *Ptr, std::size_t Count) override {
    MAllocator.deallocate(
        reinterpret_cast<typename AllocatorT::value_type *>(Ptr), Count);
  }

  void setAlignment(std::size_t RequiredAlign) override {
    setAlignImpl(RequiredAlign);
  }

  virtual std::size_t getValueSize() const override { return MValueSize; }

protected:
  virtual void *getAllocatorImpl() override { return &MAllocator; }

private:
  template <typename T>
  using EnableIfDefaultAllocator = std::enable_if_t<
      std::is_same_v<T, sycl_memory_object_allocator<OwnerDataT>>>;

  template <typename T>
  using EnableIfNonDefaultAllocator = std::enable_if_t<
      !std::is_same_v<T, sycl_memory_object_allocator<OwnerDataT>>>;

  template <typename T = AllocatorT>
  EnableIfNonDefaultAllocator<T> setAlignImpl(std::size_t) {
    // Do nothing in case of user's allocator.
  }

  template <typename T = AllocatorT>
  EnableIfDefaultAllocator<T> setAlignImpl(std::size_t RequiredAlign) {
    MAllocator.setAlignment(std::max<size_t>(RequiredAlign, 64));
  }

  AllocatorT MAllocator;
  std::size_t MValueSize;
};
} // namespace detail
} // namespace _V1
} // namespace sycl
