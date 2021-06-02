//==------ usm_allocator.hpp - SYCL USM Allocator ------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/detail/export.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/queue.hpp>
#include <CL/sycl/usm/usm_enums.hpp>

#include <cstdlib>
#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declarations.
__SYCL_EXPORT void *aligned_alloc(size_t alignment, size_t size,
                                  const device &dev, const context &ctxt,
                                  usm::alloc kind,
                                  const property_list &propList);
__SYCL_EXPORT void free(void *ptr, const context &ctxt);

template <typename T, usm::alloc AllocKind, size_t Alignment = alignof(T)>
class usm_allocator {
public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

public:
  template <typename U> struct rebind {
    typedef usm_allocator<U, AllocKind, Alignment> other;
  };

  static_assert(
      AllocKind != usm::alloc::device,
      "usm_allocator does not support AllocKind == usm::alloc::device");

  usm_allocator() noexcept = delete;
  usm_allocator(const context &Ctxt, const device &Dev,
                const property_list &PropList = {}) noexcept
      : MContext(Ctxt), MDevice(Dev), MPropList(PropList) {}
  usm_allocator(const queue &Q, const property_list &PropList = {}) noexcept
      : MContext(Q.get_context()), MDevice(Q.get_device()),
        MPropList(PropList) {}
  usm_allocator(const usm_allocator &) noexcept = default;
  usm_allocator(usm_allocator &&) noexcept = default;
  usm_allocator &operator=(const usm_allocator &Other) {
    MContext = Other.MContext;
    MDevice = Other.MDevice;
    MPropList = Other.MPropList;
    return *this;
  }
  usm_allocator &operator=(usm_allocator &&Other) {
    MContext = std::move(Other.MContext);
    MDevice = std::move(Other.MDevice);
    MPropList = std::move(Other.MPropList);
    return *this;
  }

  template <class U>
  usm_allocator(const usm_allocator<U, AllocKind, Alignment> &Other) noexcept
      : MContext(Other.MContext), MDevice(Other.MDevice),
        MPropList(Other.MPropList) {}

  /// Allocates memory.
  ///
  /// \param NumberOfElements is a count of elements to allocate memory for.
  T *allocate(size_t NumberOfElements) {

    auto Result = reinterpret_cast<T *>(
        aligned_alloc(getAlignment(), NumberOfElements * sizeof(value_type),
                      MDevice, MContext, AllocKind, MPropList));
    if (!Result) {
      throw memory_allocation_error();
    }
    return Result;
  }

  /// Deallocates memory.
  ///
  /// \param Ptr is a pointer to memory being deallocated.
  /// \param Size is a number of elements previously passed to allocate.
  void deallocate(T *Ptr, size_t) {
    if (Ptr) {
      free(Ptr, MContext);
    }
  }

  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend bool operator==(const usm_allocator<T, AllocKind, Alignment> &One,
                         const usm_allocator<U, AllocKindU, AlignmentU> &Two) {
    return ((AllocKind == AllocKindU) && (One.MContext == Two.MContext) &&
            (One.MDevice == Two.MDevice));
  }

  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend bool operator!=(const usm_allocator<T, AllocKind, Alignment> &One,
                         const usm_allocator<U, AllocKindU, AlignmentU> &Two) {
    return !((AllocKind == AllocKindU) && (One.MContext == Two.MContext) &&
             (One.MDevice == Two.MDevice));
  }

private:
  constexpr size_t getAlignment() const { return Alignment; }

  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend class usm_allocator;

  context MContext;
  device MDevice;
  property_list MPropList;
};

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
