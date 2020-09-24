//==------------ aligned_allocator.hpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/detail/common.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/range.hpp>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {
template <typename T> class aligned_allocator {
public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

public:
  template <typename U> struct rebind { typedef aligned_allocator<U> other; };

  aligned_allocator() = default;
  ~aligned_allocator() = default;

  explicit aligned_allocator(size_t Alignment) : MAlignment(Alignment) {}

  // Construct an object
  void construct(pointer Ptr, const_reference Val) {
    new (Ptr) value_type(Val);
  }

  // Destroy an object
  void destroy(pointer Ptr) { Ptr->~value_type(); }

  pointer address(reference Val) const { return &Val; }
  const_pointer address(const_reference Val) { return &Val; }

  // Allocate memory aligned to Alignment
  pointer allocate(size_t Size) {
    size_t NumBytes = Size * sizeof(value_type);
    NumBytes = ((NumBytes - 1) | (MAlignment - 1)) + 1;
    if (Size > NumBytes)
      throw std::bad_alloc();

    pointer Result = reinterpret_cast<pointer>(
        detail::OSUtil::alignedAlloc(MAlignment, NumBytes));
    if (!Result)
      throw std::bad_alloc();
    return Result;
  }

  // Release allocated memory
  void deallocate(pointer Ptr, size_t) {
    if (Ptr)
      detail::OSUtil::alignedFree(Ptr);
  }

  bool operator==(const aligned_allocator &) { return true; }
  bool operator!=(const aligned_allocator &) { return false; }

  void setAlignment(size_t Alignment) { MAlignment = Alignment; }

private:
  // By default assume the "worst" case
  size_t MAlignment = 128;
};
} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
