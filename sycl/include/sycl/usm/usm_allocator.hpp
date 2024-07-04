//==------ usm_allocator.hpp - SYCL USM Allocator ------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <sycl/builtins.hpp>
#include <sycl/context.hpp>
#include <sycl/detail/common.hpp>
#include <sycl/device.hpp>
#include <sycl/exception.hpp>
#include <sycl/property_list.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm.hpp>

#include <cstdlib>     // for size_t, aligned_alloc, free
#include <type_traits> // for true_type

namespace sycl {
inline namespace _V1 {
template <typename T, usm::alloc AllocKind, size_t Alignment = 0>
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

  usm_allocator() = delete;
  usm_allocator(const context &Ctxt, const device &Dev,
                const property_list &PropList = {})
      : MContext(Ctxt), MDevice(Dev), MPropList(PropList) {}
  usm_allocator(const queue &Q, const property_list &PropList = {})
      : MContext(Q.get_context()), MDevice(Q.get_device()),
        MPropList(PropList) {}
  usm_allocator(const usm_allocator &) = default;
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
  T *allocate(size_t NumberOfElements, const detail::code_location CodeLoc =
                                           detail::code_location::current()) {

    if (!NumberOfElements)
      return nullptr;

    auto Result = reinterpret_cast<T *>(
        aligned_alloc(getAlignment(), NumberOfElements * sizeof(value_type),
                      MDevice, MContext, AllocKind, MPropList, CodeLoc));
    if (!Result) {
      throw exception(make_error_code(errc::memory_allocation));
    }
    return Result;
  }

  /// Deallocates memory.
  ///
  /// \param Ptr is a pointer to memory being deallocated.
  /// \param Size is a number of elements previously passed to allocate.
  void deallocate(
      T *Ptr, size_t,
      const detail::code_location CodeLoc = detail::code_location::current()) {
    if (Ptr) {
      free(Ptr, MContext, CodeLoc);
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

  template <typename Property> bool has_property() const noexcept {
    return MPropList.has_property<Property>();
  }

  template <typename Property> Property get_property() const {
    return MPropList.get_property<Property>();
  }

private:
  constexpr size_t getAlignment() const { return max(alignof(T), Alignment); }

  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend class usm_allocator;

  context MContext;
  device MDevice;
  property_list MPropList;
};

} // namespace _V1
} // namespace sycl
