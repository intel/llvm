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

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace detail {
// Helper for creating ranges for empty weak_objects.
template <int Dims> static range<Dims> createDummyRange() {
  static_assert(Dims > 0 && Dims < 4, "Invalid dimensionality in range.");
  if constexpr (Dims == 1)
    return {1};
  else if constexpr (Dims == 2)
    return {1, 1};
  else
    return {1, 1, 1};
}
} // namespace detail

template <typename SYCLObjT>
class weak_object : public detail::weak_object_base<SYCLObjT> {
public:
  using object_type = typename detail::weak_object_base<SYCLObjT>::object_type;

  constexpr weak_object() noexcept : detail::weak_object_base<SYCLObjT>() {}
  weak_object(const SYCLObjT &SYCLObj) noexcept
      : detail::weak_object_base<SYCLObjT>(SYCLObj) {}
  weak_object(const weak_object &Other) noexcept
      : detail::weak_object_base<SYCLObjT>(Other) {}
  weak_object(weak_object &&Other) noexcept
      : detail::weak_object_base<SYCLObjT>(Other) {}

  weak_object &operator=(const SYCLObjT &SYCLObj) noexcept {
    this->MObjWeakPtr = sycl::detail::getSyclObjImpl(SYCLObj);
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept {
    this->MObjWeakPtr = Other.MObjWeakPtr;
    return *this;
  }
  weak_object &operator=(weak_object &&Other) noexcept {
    this->MObjWeakPtr = std::move(Other.MObjWeakPtr);
    return *this;
  }

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
        MOffsetInBytes{SYCLObj.OffsetInBytes}, MIsSubBuffer{
                                                   SYCLObj.IsSubBuffer} {}
  weak_object(const weak_object &Other) noexcept
      : detail::weak_object_base<buffer_type>(Other), MRange{Other.MRange},
        MOffsetInBytes{Other.MOffsetInBytes}, MIsSubBuffer{Other.MIsSubBuffer} {
  }
  weak_object(weak_object &&Other) noexcept
      : detail::weak_object_base<buffer_type>(Other), MRange{std::move(
                                                          Other.MRange)},
        MOffsetInBytes{std::move(Other.MOffsetInBytes)},
        MIsSubBuffer{std::move(Other.MIsSubBuffer)} {}

  weak_object &operator=(const buffer_type &SYCLObj) noexcept {
    this->MObjWeakPtr = sycl::detail::getSyclObjImpl(SYCLObj);
    this->MRange = SYCLObj.Range;
    this->MOffsetInBytes = SYCLObj.OffsetInBytes;
    this->MIsSubBuffer = SYCLObj.IsSubBuffer;
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept {
    this->MObjWeakPtr = Other.MObjWeakPtr;
    this->MRange = Other.MRange;
    this->MOffsetInBytes = Other.MOffsetInBytes;
    this->MIsSubBuffer = Other.MIsSubBuffer;
    return *this;
  }
  weak_object &operator=(weak_object &&Other) noexcept {
    this->MObjWeakPtr = std::move(Other.MObjWeakPtr);
    this->MRange = std::move(Other.MRange);
    this->MOffsetInBytes = std::move(Other.MOffsetInBytes);
    this->MIsSubBuffer = std::move(Other.MIsSubBuffer);
    return *this;
  }

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
  range<Dimensions> MRange;
  size_t MOffsetInBytes;
  bool MIsSubBuffer;
};

} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl
