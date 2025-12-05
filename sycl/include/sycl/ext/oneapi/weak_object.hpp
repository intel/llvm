//==-------------- weak_object.hpp --- SYCL weak objects -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <optional>
#include <stddef.h>
#include <sycl/access/access.hpp>
#include <sycl/accessor.hpp>
#include <sycl/buffer.hpp>
#include <sycl/detail/impl_utils.hpp>
#include <sycl/detail/memcpy.hpp>
#include <sycl/exception.hpp>
#include <sycl/range.hpp>
#include <sycl/stream.hpp>

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi {
template <typename SYCLObjT> class weak_object;
namespace detail {
using namespace sycl::detail;
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
#ifndef __SYCL_DEVICE_ONLY__
      : MObjWeakPtr(getSyclObjImpl(SYCLObj))
#endif
  {
    (void)SYCLObj;
  }
  weak_object_base(const weak_object_base &Other) noexcept = default;
  weak_object_base(weak_object_base &&Other) noexcept = default;

  weak_object_base &operator=(const weak_object_base &Other) noexcept = default;
  weak_object_base &operator=(weak_object_base &&Other) noexcept = default;

  void reset() noexcept { MObjWeakPtr.reset(); }
  void swap(weak_object_base &Other) noexcept {
    MObjWeakPtr.swap(Other.MObjWeakPtr);
  }

  bool expired() const noexcept { return MObjWeakPtr.expired(); }

#ifndef __SYCL_DEVICE_ONLY__
  bool owner_before(const SYCLObjT &Other) const noexcept {
    return MObjWeakPtr.owner_before(getSyclObjImpl(Other));
  }
  bool owner_before(const weak_object_base &Other) const noexcept {
    return MObjWeakPtr.owner_before(Other.MObjWeakPtr);
  }
  SYCLObjT lock() const {
    std::optional<SYCLObjT> OptionalObj =
        static_cast<const weak_object<SYCLObjT> *>(this)->try_lock();
    if (!OptionalObj)
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Referenced object has expired.");
    return *std::move(OptionalObj);
  }
#else
  // On device calls to these functions are disallowed, so declare them but
  // don't define them to avoid compilation failures.
  bool owner_before(const SYCLObjT &Other) const noexcept;
  bool owner_before(const weak_object_base &Other) const noexcept;
  SYCLObjT lock() const;
#endif // __SYCL_DEVICE_ONLY__

protected:
#ifndef __SYCL_DEVICE_ONLY__
  // Store a weak variant of the impl in the SYCLObjT.
  typename std::remove_reference_t<decltype(getSyclObjImpl(
      std::declval<SYCLObjT>()))>::weak_type MObjWeakPtr;
#else
  // On device we may not have an impl, so we pad with an unused void pointer.
  std::weak_ptr<void> MObjWeakPtr;
#endif // __SYCL_DEVICE_ONLY__

  template <class Obj>
  friend decltype(weak_object_base<Obj>::MObjWeakPtr)
  getSyclWeakObjImpl(const weak_object_base<Obj> &WeakObj);
};

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
  using weak_object_base = detail::weak_object_base<SYCLObjT>;

public:
  using object_type = typename weak_object_base::object_type;

  constexpr weak_object() noexcept = default;
  weak_object(const SYCLObjT &SYCLObj) noexcept : weak_object_base(SYCLObj) {}
  weak_object(const weak_object &Other) noexcept = default;
  weak_object(weak_object &&Other) noexcept = default;

  weak_object &operator=(const SYCLObjT &SYCLObj) noexcept {
    weak_object_base::operator=(SYCLObj);
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept = default;
  weak_object &operator=(weak_object &&Other) noexcept = default;

#ifndef __SYCL_DEVICE_ONLY__
  std::optional<SYCLObjT> try_lock() const noexcept {
    auto MObjImplPtr = this->MObjWeakPtr.lock();
    if (!MObjImplPtr)
      return std::nullopt;
    return detail::createSyclObjFromImpl<SYCLObjT>(MObjImplPtr);
  }
#else
  // On device calls to these functions are disallowed, so declare them but
  // don't define them to avoid compilation failures.
  std::optional<SYCLObjT> try_lock() const noexcept;
#endif // __SYCL_DEVICE_ONLY__
};

// Specialization of weak_object for buffer as it needs additional members
// to reconstruct the original buffer.
template <typename T, int Dimensions, typename AllocatorT>
class weak_object<buffer<T, Dimensions, AllocatorT>>
    : public detail::weak_object_base<buffer<T, Dimensions, AllocatorT>> {
  using weak_object_base =
      detail::weak_object_base<buffer<T, Dimensions, AllocatorT>>;
  using buffer_type = buffer<T, Dimensions, AllocatorT>;

public:
  using object_type = typename weak_object_base::object_type;

  constexpr weak_object() noexcept
      : MRange{detail::createDummyRange<Dimensions>()}, MOffsetInBytes{0},
        MIsSubBuffer{false} {}
  weak_object(const buffer_type &SYCLObj) noexcept
      : weak_object_base(SYCLObj), MRange{SYCLObj.Range},
        MOffsetInBytes{SYCLObj.OffsetInBytes},
        MIsSubBuffer{SYCLObj.IsSubBuffer} {}
  weak_object(const weak_object &Other) noexcept = default;
  weak_object(weak_object &&Other) noexcept = default;

  weak_object &operator=(const buffer_type &SYCLObj) noexcept {
    // Create weak_ptr from the shared_ptr to SYCLObj's implementation object.
    weak_object_base::operator=(SYCLObj);
    this->MRange = SYCLObj.Range;
    this->MOffsetInBytes = SYCLObj.OffsetInBytes;
    this->MIsSubBuffer = SYCLObj.IsSubBuffer;
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept = default;
  weak_object &operator=(weak_object &&Other) noexcept = default;

  void swap(weak_object &Other) noexcept {
    weak_object_base::swap(Other);
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
#else
  // On device calls to these functions are disallowed, so declare them but
  // don't define them to avoid compilation failures.
  std::optional<buffer_type> try_lock() const noexcept;
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
  using weak_object_base = detail::weak_object_base<stream>;

public:
  using object_type = typename weak_object_base::object_type;

  constexpr weak_object() noexcept : detail::weak_object_base<stream>() {}
  weak_object(const stream &SYCLObj) noexcept
      : detail::weak_object_base<stream>(SYCLObj),
        MWeakGlobalBuf{SYCLObj.GlobalBuf},
        MWeakGlobalOffset{SYCLObj.GlobalOffset},
        MWeakGlobalFlushBuf{SYCLObj.GlobalFlushBuf} {}
  weak_object(const weak_object &Other) noexcept = default;
  weak_object(weak_object &&Other) noexcept = default;

  weak_object &operator=(const stream &SYCLObj) noexcept {
    weak_object_base::operator=(SYCLObj);
    MWeakGlobalBuf = SYCLObj.GlobalBuf;
    MWeakGlobalOffset = SYCLObj.GlobalOffset;
    MWeakGlobalFlushBuf = SYCLObj.GlobalFlushBuf;
    return *this;
  }
  weak_object &operator=(const weak_object &Other) noexcept = default;
  weak_object &operator=(weak_object &&Other) noexcept = default;

  void swap(weak_object &Other) noexcept {
    weak_object_base::swap(Other);
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
#else
  // On device calls to these functions are disallowed, so declare them but
  // don't define them to avoid compilation failures.
  std::optional<stream> try_lock() const noexcept;
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
