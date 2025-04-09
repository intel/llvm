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
  template <typename Properties = empty_properties_t,
            typename = std::enable_if_t<
                detail::all_are_properties_of_v<memory_pool, Properties>>>
  memory_pool(const sycl::context &, sycl::usm::alloc kind, Properties = {}) {
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

  template <typename Properties = empty_properties_t,
            typename = std::enable_if_t<
                detail::all_are_properties_of_v<memory_pool, Properties>>>
  memory_pool(const sycl::context &ctx, const sycl::device &dev,
              sycl::usm::alloc kind, Properties props = {})
      : memory_pool(ctx, dev, kind, stripProps(props)) {}

  template <typename Properties = empty_properties_t,
            typename = std::enable_if_t<
                detail::all_are_properties_of_v<memory_pool, Properties>>>
  memory_pool(const sycl::queue &q, sycl::usm::alloc kind,
              Properties props = {})
      : memory_pool(q.get_context(), q.get_device(), kind, props) {}

  // NOT SUPPORTED: Creating a pool from an existing allocation is unsupported.
  template <typename Properties = empty_properties_t,
            typename = std::enable_if_t<
                detail::all_are_properties_of_v<memory_pool, Properties>>>
  memory_pool(const sycl::context &, void *, size_t, Properties = {}) {
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
    const auto tuple = getPropsTuple();
    if constexpr (std::is_same_v<PropertyT, initial_threshold>) {
      return std::get<0>(tuple.first);
    }
    if constexpr (std::is_same_v<PropertyT, maximum_size>) {
      return std::get<1>(tuple.first);
    }
    if constexpr (std::is_same_v<PropertyT, read_only>) {
      return std::get<2>(tuple.first);
    }
    if constexpr (std::is_same_v<PropertyT, zero_init>) {
      return std::get<3>(tuple.first);
    }
  }

  template <typename PropertyT> PropertyT get_property() const {
    if (!has_property<PropertyT>())
      throw sycl::exception(make_error_code(errc::invalid),
                            "The property is not found");
    const auto tuple = getPropsTuple();
    if constexpr (std::is_same_v<PropertyT, initial_threshold>) {
      return initial_threshold(std::get<0>(tuple.second));
    }
    if constexpr (std::is_same_v<PropertyT, maximum_size>) {
      return maximum_size(std::get<1>(tuple.second));
    }
    if constexpr (std::is_same_v<PropertyT, read_only>) {
      return read_only();
    }
    if constexpr (std::is_same_v<PropertyT, zero_init>) {
      return zero_init();
    }
  }

protected:
  std::shared_ptr<detail::memory_pool_impl> impl;

  memory_pool(std::shared_ptr<detail::memory_pool_impl> Impl) : impl(Impl) {}

  memory_pool(const sycl::context &ctx, const sycl::device &dev,
              const sycl::usm::alloc kind,
              const std::pair<std::tuple<bool, bool, bool, bool>,
                              std::tuple<size_t, size_t, bool, bool>> &props);
  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_rvalue_reference_t<decltype(T::impl)> ImplObj);
  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(
      std::add_lvalue_reference_t<const decltype(T::impl)> ImplObj);

  const std::pair<std::tuple<bool, bool, bool, bool>,
                  std::tuple<size_t, size_t, bool, bool>> &
  getPropsTuple() const;

  template <typename Properties = empty_properties_t>
  std::pair<std::tuple<bool, bool, bool, bool>,
            std::tuple<size_t, size_t, bool, bool>>
  stripProps(Properties props) {

    // Pair of tuples of set properties and their values.
    // initial_threshold, maximum_size, read_only, zero_init.
    std::pair<std::tuple<bool, bool, bool, bool>,
              std::tuple<size_t, size_t, bool, bool>>
        tuple;
    bool initialThreshold = 0, maximumSize = 0, readOnly = 0, zeroInit = 0;
    size_t initialThresholdVal = 0, maximumSizeVal = 0;

    if constexpr (decltype(props)::template has_property<
                      initial_threshold_key>()) {
      initialThreshold = 1;
      initialThresholdVal =
          props.template get_property<initial_threshold>().value;
    }

    if constexpr (decltype(props)::template has_property<maximum_size_key>()) {
      maximumSize = 1;
      maximumSizeVal = props.template get_property<maximum_size>().value;
    }

    if constexpr (decltype(props)::template has_property<read_only_key>()) {
      readOnly = 1;
    }

    if constexpr (decltype(props)::template has_property<zero_init_key>()) {
      zeroInit = 1;
    }

    tuple.first = {initialThreshold, maximumSize, readOnly, zeroInit};
    tuple.second = {initialThresholdVal, maximumSizeVal, readOnly, zeroInit};
    return tuple;
  }
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
