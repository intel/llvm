//==------ usm_allocator.hpp - SYCL USM Allocator ------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/queue.hpp>
#include <CL/sycl/usm/usm_enums.hpp>

#include <cstdlib>
#include <memory>

__SYCL_INLINE namespace cl {
namespace sycl {

// Forward declarations.
void *aligned_alloc(size_t alignment, size_t size, const device &dev,
                    const context &ctxt, usm::alloc kind);
void free(void *ptr, const context &ctxt);

template <typename T, usm::alloc AllocKind, size_t Alignment = 0>
class usm_allocator {
public:
  using value_type = T;
  using pointer = T *;
  using const_pointer = const T *;
  using reference = T &;
  using const_reference = const T &;

public:
  template <typename U> struct rebind {
    typedef usm_allocator<U, AllocKind, Alignment> other;
  };

  usm_allocator() = delete;
  usm_allocator(const context &Ctxt, const device &Dev)
      : mContext(Ctxt), mDevice(Dev) {}
  usm_allocator(const queue &Q)
      : mContext(Q.get_context()), mDevice(Q.get_device()) {}
  usm_allocator(const usm_allocator &Other)
      : mContext(Other.mContext), mDevice(Other.mDevice) {}

  // Construct an object
  // Note: AllocKind == alloc::device is not allowed
  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT != usm::alloc::device, int>::type = 0>
  void construct(pointer Ptr, const_reference Val) {
    new (Ptr) value_type(Val);
  }

  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT == usm::alloc::device, int>::type = 0>
  void construct(pointer Ptr, const_reference Val) {
    throw feature_not_supported(
        "Device pointers do not support construct on host");
  }

  // Destroy an object
  // Note:: AllocKind == alloc::device is not allowed
  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT != usm::alloc::device, int>::type = 0>
  void destroy(pointer Ptr) {
    Ptr->~value_type();
  }

  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT == usm::alloc::device, int>::type = 0>
  void destroy(pointer Ptr) {
    throw feature_not_supported(
        "Device pointers do not support destroy on host");
  }

  // Note:: AllocKind == alloc::device is not allowed
  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT != usm::alloc::device, int>::type = 0>
  pointer address(reference Val) const {
    return &Val;
  }

  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT == usm::alloc::device, int>::type = 0>
  pointer address(reference Val) const {
    throw feature_not_supported(
        "Device pointers do not support address on host");
  }

  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT != usm::alloc::device, int>::type = 0>
  const_pointer address(const_reference Val) const {
    return &Val;
  }

  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT == usm::alloc::device, int>::type = 0>
  const_pointer address(const_reference Val) const {
    throw feature_not_supported(
        "Device pointers do not support address on host");
  }

  /// Allocates memory.
  ///
  /// @param Size is a count of elements to allocate memory for.
  pointer allocate(size_t Size) {

    auto Result = reinterpret_cast<pointer>(
        aligned_alloc(getAlignment(), Size * sizeof(value_type),
                                 mDevice, mContext, AllocKind));
    if (!Result) {
      throw memory_allocation_error();
    }
    return Result;
  }

  /// Deallocates memory.
  ///
  /// @param Ptr is a pointer to memory being deallocated.
  void deallocate(pointer Ptr, size_t Size) {
    if (Ptr) {
      free(Ptr, mContext);
    }
  }

private:
  constexpr size_t getAlignment() const {
    /*
      // This form might be preferable if the underlying implementation
      // doesn't do the right thing when given 0 for alignment
    return ((Alignment == 0)
            ? alignof(value_type)
            : Alignment);
    */
    return Alignment;
  }

  const context mContext;
  const device mDevice;
};

} // namespace sycl
} // namespace cl
