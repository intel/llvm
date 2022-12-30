//==-------------- weak_object.hpp --- SYCL weak objects -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/buffer.hpp>
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/ext/oneapi/weak_object_base.hpp>

#include <optional>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext::oneapi {
namespace detail {
// Helper for creating ranges for empty weak_objects.
template <int Dims> static range<Dims> createDummyRange() {
  static_assert(Dims >= 0 && Dims < 4, "Invalid dimensionality in range.");
  if constexpr (Dims == 0)
    return {};
  if constexpr (Dims == 1)
    return {1};
  else if constexpr (Dims == 2)
    return {1, 1};
  else
    return {1, 1, 1};
}
} // namespace detail

// The weak_object class.
// Since we use std::shared_ptr to implementations as a way to handle common
// reference semantics in SYCL classes, the implementation of the weak_object
// class holds a weak_ptr to the implementations. The weak_ptr is in the
// weak_object_base class.
template <typename SYCLObjT>
class weak_object : public detail::weak_object_base<SYCLObjT> {
public:
  using object_type = typename detail::weak_object_base<SYCLObjT>::object_type;

  constexpr weak_object() noexcept = default;
  weak_object(const SYCLObjT &SYCLObj) noexcept
      : detail::weak_object_base<SYCLObjT>(SYCLObj) {}
  weak_object(const weak_object &Other) noexcept = default;
  weak_object(weak_object &&Other) noexcept = default;

  weak_object &operator=(const SYCLObjT &SYCLObj) noexcept {
    // Create weak_ptr from the shared_ptr to SYCLObj's implementation object.
    this->MObjWeakPtr = sycl::detail::getSyclObjImpl(SYCLObj);
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept = default;
  weak_object &operator=(weak_object &&Other) noexcept = default;

  std::optional<SYCLObjT> try_lock() const noexcept {
    auto MObjImplPtr = this->MObjWeakPtr.lock();
    if (!MObjImplPtr)
      return std::nullopt;
    return sycl::detail::createSyclObjFromImpl<SYCLObjT>(MObjImplPtr);
  }
  SYCLObjT lock() const {
    std::optional<SYCLObjT> OptionalObj = try_lock();
    if (!OptionalObj)
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Referenced object has expired.");
    return *OptionalObj;
  }
};

// Specialization of weak_object for buffer as it needs additional members
// to reconstruct the original buffer.
template <typename T, int Dimensions, typename AllocatorT>
class weak_object<buffer<T, Dimensions, AllocatorT>>
    : public detail::weak_object_base<buffer<T, Dimensions, AllocatorT>> {
private:
  using buffer_type = buffer<T, Dimensions, AllocatorT>;

public:
  using object_type =
      typename detail::weak_object_base<buffer_type>::object_type;

  constexpr weak_object() noexcept
      : detail::weak_object_base<buffer_type>(),
        MRange{detail::createDummyRange<Dimensions>()}, MOffsetInBytes{0},
        MIsSubBuffer{false} {}
  weak_object(const buffer_type &SYCLObj) noexcept
      : detail::weak_object_base<buffer_type>(SYCLObj), MRange{SYCLObj.Range},
        MOffsetInBytes{SYCLObj.OffsetInBytes},
        MIsSubBuffer{SYCLObj.IsSubBuffer} {}
  weak_object(const weak_object &Other) noexcept = default;
  weak_object(weak_object &&Other) noexcept = default;

  weak_object &operator=(const buffer_type &SYCLObj) noexcept {
    // Create weak_ptr from the shared_ptr to SYCLObj's implementation object.
    this->MObjWeakPtr = sycl::detail::getSyclObjImpl(SYCLObj);
    this->MRange = SYCLObj.Range;
    this->MOffsetInBytes = SYCLObj.OffsetInBytes;
    this->MIsSubBuffer = SYCLObj.IsSubBuffer;
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept = default;
  weak_object &operator=(weak_object &&Other) noexcept = default;

  std::optional<buffer_type> try_lock() const noexcept {
    auto MObjImplPtr = this->MObjWeakPtr.lock();
    if (!MObjImplPtr)
      return std::nullopt;
    // To reconstruct the buffer we use the reinterpret constructor.
    return buffer_type{MObjImplPtr, MRange, MOffsetInBytes, MIsSubBuffer};
  }
  buffer_type lock() const {
    std::optional<buffer_type> OptionalObj = try_lock();
    if (!OptionalObj)
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Referenced object has expired.");
    return *OptionalObj;
  }

private:
  // Additional members required for recreating buffers.
  range<Dimensions> MRange;
  size_t MOffsetInBytes;
  bool MIsSubBuffer;
};

} // namespace ext::oneapi
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
