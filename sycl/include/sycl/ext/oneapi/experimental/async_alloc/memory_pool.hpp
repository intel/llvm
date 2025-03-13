//==----------- memory_pool.hpp --- SYCL asynchronous allocation -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
// #include <detail/memory_pool_impl.hpp>
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
  memory_pool(const sycl::context &ctx, const property_list &props = {});

  memory_pool(const sycl::context &ctx, const sycl::device &dev,
              const sycl::usm::alloc kind, const property_list &props = {});

  memory_pool(const sycl::queue &q, const sycl::usm::alloc kind,
              const property_list &props = {});

  memory_pool(const sycl::context &ctx, const void *ptr, size_t size,
              const property_list &props = {});

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
  size_t get_reserved_size_high() const;
  size_t get_used_size_current() const;
  size_t get_used_size_high() const;

  void set_new_threshold(size_t newThreshold);
  void reset_reserved_size_high();
  void reset_used_size_high();
  void trim_to(size_t minBytesToKeep);

  // Property getters.
  template <typename PropertyT> bool has_property() const noexcept {
    return getPropList().template has_property<PropertyT>();
}
  template <typename PropertyT> PropertyT get_property() const {
    return getPropList().template get_property<PropertyT>();
  }

protected:
  std::shared_ptr<detail::memory_pool_impl> impl;

  template <class Obj>
  friend const decltype(Obj::impl) &
  sycl::detail::getSyclObjImpl(const Obj &SyclObject);

  template <class T>
  friend T sycl::detail::createSyclObjFromImpl(decltype(T::impl) ImplObj);

  const property_list &getPropList() const;

  memory_pool(std::shared_ptr<detail::memory_pool_impl> Impl) : impl(Impl) {}
  memory_pool(const sycl::context &ctx, const sycl::device &dev,
              const sycl::usm::alloc kind,
              const std::pair<std::tuple<bool, bool, bool, bool>,
                              std::tuple<size_t, size_t, bool, bool>> &props);

  template <typename Properties = empty_properties_t,
            typename = std::enable_if_t<
                detail::all_are_properties_of_v<memory_pool, Properties>>>
  std::pair<std::tuple<bool, bool, bool, bool>,
            std::tuple<size_t, size_t, bool, bool>>
  stripProps(const Properties &props) {

    // Pair of tuples of set properties and their values.
    // initial_threshold, maximum_size, read_only, zero_init.
    std::pair<std::tuple<bool, bool, bool, bool>,
              std::tuple<size_t, size_t, bool, bool>>
        tuple;
    bool initialThreshold = 0;
    bool maximumSize = 0;
    bool readOnly = 0;
    bool zeroInit = 0;
    // size_t initialThresholdVal = 0;
    size_t maximumSizeVal = 0;
    bool readOnlyVal = 0;
    bool zeroInitVal = 0;

    // auto a = props.template has_property<initial_threshold_key>();
    // auto a = decltype(props)::has_property<initial_threshold_key>();
    // properties P2{maximum_size{1024}};
    // if constexpr (P2.has_property<initial_threshold_key>()) {
    //   // initialThreshold = 1;
    //   constexpr size_t initialThresholdVal =
    //       P2.get_property<ext::oneapi::experimental::initial_threshold>().value;
    //   // std::cout << "Stripping initial threshold: " << initialThresholdVal
    //   //           << std::endl;
    // } else {
    //   // std::cout << "do nothing" << std::endl;
    // }

    // if (props.template has_property<
    //         ext::oneapi::experimental::maximum_size_key>()) {
    //   maximumSize = 1;
    //   maximumSizeVal =
    //       props.template
    //       get_property<ext::oneapi::experimental::maximum_size>()
    //           .value;
    //   std::cout << "Stripping maximum size: " << maximumSizeVal << std::endl;
    // }

    // if (props.template has_property<ext::oneapi::experimental::read_only>())
    // {
    //   readOnly = 1;
    //   readOnlyVal =
    //       props.template get_property<ext::oneapi::experimental::read_only>()
    //           .value;
    //   std::cout << "Stripping read only: " << readOnlyVal << std::endl;
    // }

    // if (props.template has_property<
    //         ext::oneapi::experimental::zero_init_key>()) {
    //   zeroInit = 1;
    //   zeroInitVal =
    //       props.template get_property<ext::oneapi::experimental::zero_init>()
    //           .value;
    //   std::cout << "Stripping zero init: " << zeroInitVal << std::endl;
    // }

    tuple.first = {initialThreshold, maximumSize, readOnly, zeroInit};
    tuple.second = {0, maximumSizeVal, readOnlyVal, zeroInitVal};

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
