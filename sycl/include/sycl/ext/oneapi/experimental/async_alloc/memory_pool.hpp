//==----------- memory_pool.hpp --- SYCL asynchronous allocation -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/context.hpp> // for context
#include <sycl/device.hpp>  // for device
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool_properties.hpp>
#include <sycl/queue.hpp>         // for queue
#include <sycl/usm/usm_enums.hpp> // for usm::alloc

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {

// Forward declare memory_pool_impl.
namespace detail {
class memory_pool_impl;
} // namespace detail

/// Memory pool
class __SYCL_EXPORT memory_pool {

public:
  // NOT SUPPORTED: Host side pools unsupported.
  memory_pool(const sycl::context &, sycl::usm::alloc kind,
              const property_list & = {}) {
    if (kind == sycl::usm::alloc::device || kind == sycl::usm::alloc::shared)
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Device and shared allocation kinds are disallowed "
                            "without specifying a device!");
    if (kind == sycl::usm::alloc::unknown)
      throw sycl::exception(sycl::make_error_code(sycl::errc::invalid),
                            "Unknown allocation kinds are disallowed!");

    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Host allocated pools are unsupported!");
  }

  memory_pool(const sycl::context &ctx, const sycl::device &dev,
              sycl::usm::alloc kind, const property_list &props = {});

  memory_pool(const sycl::queue &q, sycl::usm::alloc kind,
              const property_list &props = {})
      : memory_pool(q.get_context(), q.get_device(), kind, props) {}

  // NOT SUPPORTED: Creating a pool from an existing allocation is unsupported.
  memory_pool(const sycl::context &, void *, size_t,
              const property_list & = {}) {
    throw sycl::exception(
        sycl::make_error_code(sycl::errc::feature_not_supported),
        "Creating a pool from an existing allocation is unsupported!");
  }

  ~memory_pool() = default;

  // Copy constructible/assignable, move constructible/assignable.
  memory_pool(const memory_pool &) = default;
  memory_pool(memory_pool &&) = default;
  memory_pool &operator=(const memory_pool &) = default;
  memory_pool &operator=(memory_pool &&) = default;

  // Equality comparison.
  bool operator==(const memory_pool &rhs) const { return impl == rhs.impl; }
  bool operator!=(const memory_pool &rhs) const { return !(*this == rhs); }

  // Impl handles getters and setters.
  sycl::context get_context() const;
  sycl::device get_device() const;
  sycl::usm::alloc get_alloc_kind() const;
  size_t get_threshold() const;
  size_t get_reserved_size_current() const;
  size_t get_used_size_current() const;

  void increase_threshold_to(size_t newThreshold);

  // Property getters.
  template <typename PropertyT> bool has_property() const noexcept {
    return getPropList().template has_property<PropertyT>();
  }
  template <typename PropertyT> PropertyT get_property() const {
    return getPropList().template get_property<PropertyT>();
  }

protected:
  std::shared_ptr<detail::memory_pool_impl> impl;

  memory_pool(std::shared_ptr<detail::memory_pool_impl> Impl)
      : impl(std::move(Impl)) {}

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);

  const property_list &getPropList() const;
};

} // namespace ext::oneapi::experimental
} // namespace _V1
} // namespace sycl

namespace std {
template <> struct hash<sycl::ext::oneapi::experimental::memory_pool> {
  size_t operator()(
      const sycl::ext::oneapi::experimental::memory_pool &mem_pool) const {
    return hash<std::shared_ptr<
        sycl::ext::oneapi::experimental::detail::memory_pool_impl>>()(
        sycl::detail::getSyclObjImpl(mem_pool));
  }
};
} // namespace std
