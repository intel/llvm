//==-------------- weak_object.hpp --- SYCL weak objects -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/access/access.hpp>     // for target, mode
#include <sycl/accessor.hpp>          // for accessor
#include <sycl/buffer.hpp>            // for buffer
#include <sycl/detail/impl_utils.hpp> // for createSyc...
#include <sycl/detail/memcpy.hpp>     // for detail
#include <sycl/exception.hpp>         // for make_erro...
#include <sycl/ext/codeplay/experimental/fusion_properties.hpp> // for buffer
#include <sycl/ext/oneapi/weak_object_base.hpp> // for weak_obje...
#include <sycl/range.hpp>                       // for range
#include <sycl/stream.hpp>                      // for stream

#include <memory>   // for shared_ptr
#include <optional> // for optional
#include <stddef.h> // for size_t

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {
namespace detail {
// Import from detail:: into ext::oneapi::detail:: to improve readability later
using namespace ::sycl::detail;

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
    this->MObjWeakPtr =
        detail::weak_object_base<SYCLObjT>::GetWeakImpl(SYCLObj);
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept = default;
  weak_object &operator=(weak_object &&Other) noexcept = default;

#ifndef __SYCL_DEVICE_ONLY__
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
#else
  // On device calls to these functions are disallowed, so declare them but
  // don't define them to avoid compilation failures.
  std::optional<SYCLObjT> try_lock() const noexcept;
  SYCLObjT lock() const;
#endif // __SYCL_DEVICE_ONLY__
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
    this->MObjWeakPtr = detail::weak_object_base<
        buffer<T, Dimensions, AllocatorT>>::GetWeakImpl(SYCLObj);
    this->MRange = SYCLObj.Range;
    this->MOffsetInBytes = SYCLObj.OffsetInBytes;
    this->MIsSubBuffer = SYCLObj.IsSubBuffer;
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept = default;
  weak_object &operator=(weak_object &&Other) noexcept = default;

  void swap(weak_object &Other) noexcept {
    this->MObjWeakPtr.swap(Other.MObjWeakPtr);
    std::swap(MRange, Other.MRange);
    std::swap(MOffsetInBytes, Other.MOffsetInBytes);
    std::swap(MIsSubBuffer, Other.MIsSubBuffer);
  }

#ifndef __SYCL_DEVICE_ONLY__
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
#else
  // On device calls to these functions are disallowed, so declare them but
  // don't define them to avoid compilation failures.
  std::optional<buffer_type> try_lock() const noexcept;
  buffer_type lock() const;
#endif // __SYCL_DEVICE_ONLY__

private:
  // Additional members required for recreating buffers.
  range<Dimensions> MRange;
  size_t MOffsetInBytes;
  bool MIsSubBuffer;
};

// Specialization of weak_object for stream as it needs additional members
// to reconstruct the original stream.
template <>
class weak_object<stream> : public detail::weak_object_base<stream> {
public:
  using object_type = typename detail::weak_object_base<stream>::object_type;

  constexpr weak_object() noexcept : detail::weak_object_base<stream>() {}
  weak_object(const stream &SYCLObj) noexcept
      : detail::weak_object_base<stream>(SYCLObj),
        MWeakGlobalBuf{SYCLObj.GlobalBuf},
        MWeakGlobalOffset{SYCLObj.GlobalOffset},
        MWeakGlobalFlushBuf{SYCLObj.GlobalFlushBuf} {}
  weak_object(const weak_object &Other) noexcept = default;
  weak_object(weak_object &&Other) noexcept = default;

  weak_object &operator=(const stream &SYCLObj) noexcept {
    // Create weak_ptr from the shared_ptr to SYCLObj's implementation object.
    this->MObjWeakPtr = detail::weak_object_base<stream>::GetWeakImpl(SYCLObj);
    MWeakGlobalBuf = SYCLObj.GlobalBuf;
    MWeakGlobalOffset = SYCLObj.GlobalOffset;
    MWeakGlobalFlushBuf = SYCLObj.GlobalFlushBuf;
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept = default;
  weak_object &operator=(weak_object &&Other) noexcept = default;

  void swap(weak_object &Other) noexcept {
    this->MObjWeakPtr.swap(Other.MObjWeakPtr);
    MWeakGlobalBuf.swap(Other.MWeakGlobalBuf);
    MWeakGlobalOffset.swap(Other.MWeakGlobalOffset);
    MWeakGlobalFlushBuf.swap(Other.MWeakGlobalFlushBuf);
  }

  void reset() noexcept {
    this->MObjWeakPtr.reset();
    MWeakGlobalBuf.reset();
    MWeakGlobalOffset.reset();
    MWeakGlobalFlushBuf.reset();
  }

#ifndef __SYCL_DEVICE_ONLY__
  std::optional<stream> try_lock() const noexcept {
    auto ObjImplPtr = this->MObjWeakPtr.lock();
    auto GlobalBuf = MWeakGlobalBuf.try_lock();
    auto GlobalOffset = MWeakGlobalOffset.try_lock();
    auto GlobalFlushBuf = MWeakGlobalFlushBuf.try_lock();
    if (!ObjImplPtr || !GlobalBuf || !GlobalOffset || !GlobalFlushBuf)
      return std::nullopt;
    return stream{ObjImplPtr, *GlobalBuf, *GlobalOffset, *GlobalFlushBuf};
  }
  stream lock() const {
    std::optional<stream> OptionalObj = try_lock();
    if (!OptionalObj)
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Referenced object has expired.");
    return *OptionalObj;
  }
#else
  // On device calls to these functions are disallowed, so declare them but
  // don't define them to avoid compilation failures.
  std::optional<stream> try_lock() const noexcept;
  stream lock() const;
#endif // __SYCL_DEVICE_ONLY__

private:
  // Additional members required for recreating stream.
  weak_object<detail::GlobalBufAccessorT> MWeakGlobalBuf;
  weak_object<detail::GlobalOffsetAccessorT> MWeakGlobalOffset;
  weak_object<detail::GlobalBufAccessorT> MWeakGlobalFlushBuf;
};

} // namespace ext::oneapi
} // namespace _V1
} // namespace sycl
