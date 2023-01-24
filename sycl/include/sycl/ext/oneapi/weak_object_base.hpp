//==------- weak_object_base.hpp --- SYCL weak objects base class ----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/detail/defines_elementary.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi::detail {
template <typename SYCLObjT> class weak_object_base;

// Helper function for getting the underlying weak_ptr from a weak_object.
template <typename SYCLObjT>
decltype(weak_object_base<SYCLObjT>::MObjWeakPtr)
getSyclWeakObjImpl(const weak_object_base<SYCLObjT> &WeakObj) {
  return WeakObj.MObjWeakPtr;
}

// Common base class for weak_object.
template <typename SYCLObjT> class weak_object_base {
public:
  using object_type = SYCLObjT;

  constexpr weak_object_base() noexcept : MObjWeakPtr() {}
  weak_object_base(const SYCLObjT &SYCLObj) noexcept
      : MObjWeakPtr(sycl::detail::getSyclObjImpl(SYCLObj)) {}
  weak_object_base(const weak_object_base &Other) noexcept = default;
  weak_object_base(weak_object_base &&Other) noexcept = default;

  void reset() noexcept { MObjWeakPtr.reset(); }
  void swap(weak_object_base &Other) noexcept {
    MObjWeakPtr.swap(Other.MObjWeakPtr);
  }

  bool expired() const noexcept { return MObjWeakPtr.expired(); }

  bool owner_before(const SYCLObjT &Other) const noexcept {
    return MObjWeakPtr.owner_before(sycl::detail::getSyclObjImpl(Other));
  }
  bool owner_before(const weak_object_base &Other) const noexcept {
    return MObjWeakPtr.owner_before(Other.MObjWeakPtr);
  }

protected:
  // Store a weak variant of the impl in the SYCLObjT.
  typename std::invoke_result_t<
      decltype(sycl::detail::getSyclObjImpl<SYCLObjT>), SYCLObjT>::weak_type
      MObjWeakPtr;

  template <class Obj>
  friend decltype(weak_object_base<Obj>::MObjWeakPtr)
  detail::getSyclWeakObjImpl(const weak_object_base<Obj> &WeakObj);
};
} // namespace ext::oneapi::detail
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
