//==------------ aligned_allocator.hpp - SYCL standard header file ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/os_util.hpp> // for OSUtil

#include <limits>      // for numeric_limits
#include <memory>      // for pointer_traits, allocator_traits
#include <new>         // for bad_alloc, operator new
#include <stddef.h>    // for size_t
#include <type_traits> // for false_type, is_empty, make_unsign...

namespace sycl {
inline namespace _V1 {
namespace detail {
template <typename T> class aligned_allocator {
public:
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;

public:
  template <typename U> struct rebind {
    typedef aligned_allocator<U> other;
  };

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
} // namespace _V1
} // namespace sycl

namespace std {
template <typename T>
struct allocator_traits<sycl::detail::aligned_allocator<T>> {
  using allocator_type = typename sycl::detail::aligned_allocator<T>;
  using value_type = typename allocator_type::value_type;
  using pointer = typename allocator_type::pointer;
  using const_pointer = typename allocator_type::const_pointer;
  using void_pointer =
      typename std::pointer_traits<pointer>::template rebind<void>;
  using const_void_pointer =
      typename std::pointer_traits<pointer>::template rebind<const void>;
  using difference_type =
      typename std::pointer_traits<pointer>::difference_type;
  using size_type = std::make_unsigned_t<difference_type>;
  using propagate_on_container_copy_assignment = std::false_type;
  using propagate_on_container_move_assignment = std::false_type;
  using propagate_on_container_swap = std::false_type;
  using is_always_equal = typename std::is_empty<allocator_type>::type;

  template <typename U>
  using rebind_alloc =
      typename sycl::detail::aligned_allocator<T>::template rebind<U>::other;
  template <typename U> using rebind_traits = allocator_traits<rebind_alloc<U>>;

  static pointer allocate(allocator_type &Allocator, size_type NumElems) {
    return Allocator.allocate(NumElems);
  }

  static pointer allocate(allocator_type &Allocator, size_type NumElems,
                          const_void_pointer) {
    // TODO: Utilize the locality hint argument.
    return Allocator.allocate(NumElems);
  }

  static void deallocate(allocator_type &Allocator, pointer Ptr,
                         size_type NumElems) {
    Allocator.deallocate(Ptr, NumElems);
  }

  template <class U, class... ArgsT>
  static void construct(allocator_type &Allocator, U *Ptr, ArgsT &&...Args) {
    return Allocator.construct(Ptr, Args...);
  }

  template <class U> static void destroy(allocator_type &Allocator, U *Ptr) {
    Allocator.destroy(Ptr);
  }

  static size_type max_size(const allocator_type &) noexcept {
    // max is a macro on Windows...
    return (std::numeric_limits<size_type>::max)() / sizeof(value_type);
  }

  static allocator_type
  select_on_container_copy_construction(const allocator_type &Allocator) {
    return Allocator;
  }
};
} // namespace std
