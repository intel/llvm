//==----------- memory_pool.hpp --- SYCL asynchronous allocation -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/context.hpp>       // for context
#include <sycl/device.hpp>        // for device
#include <sycl/queue.hpp>         // for queue
#include <sycl/usm/usm_enums.hpp> // for usm::alloc

namespace sycl {
inline namespace _V1 {
namespace ext::oneapi::experimental {
namespace property {

// Property that determines the initial threshold of a memory pool.
struct initial_threshold : public sycl::detail::PropertyWithData<
                               sycl::detail::MemPoolInitialThreshold> {
public:
  initial_threshold(size_t initialThreshold)
      : initialThreshold(initialThreshold) {};
  size_t get_initial_threshold() { return initialThreshold; }

private:
  size_t initialThreshold;
};

// Property that determines the maximum size of a memory pool.
struct maximum_size
    : public sycl::detail::PropertyWithData<sycl::detail::MemPoolMaximumSize> {
public:
  maximum_size(size_t maxSize) : maxSize(maxSize) {};
  size_t get_maximum_size() { return maxSize; }

private:
  size_t maxSize;
};

// Property that provides a performance hint that all allocations from this pool
// will only be read from within SYCL kernel functions.
struct read_only
    : public sycl::detail::DataLessProperty<sycl::detail::MemPoolReadOnly> {
public:
  read_only() = default;
};

// Property that initial allocations to a pool (not subsequent allocations from
// prior frees) are iniitialised to zero.
struct zero_init
    : public sycl::detail::DataLessProperty<sycl::detail::MemPoolZeroInit> {
public:
  zero_init() = default;
};
} // namespace property

namespace detail {
class memory_pool_impl {
public:
  memory_pool_impl(const sycl::context &ctx, const sycl::device &dev,
                   const sycl::usm::alloc kind, const property_list &props);
  memory_pool_impl(const sycl::context &ctx, const sycl::device &dev,
                   const sycl::usm::alloc kind, ur_usm_pool_handle_t poolHandle,
                   const bool isDefaultPool, const property_list &props);

  ~memory_pool_impl();

  memory_pool_impl(const memory_pool_impl &) = delete;
  memory_pool_impl &operator=(const memory_pool_impl &) = delete;

  ur_usm_pool_handle_t get_handle() const { return MPoolHandle; }
  sycl::device get_device() const { return MDevice; }
  sycl::context get_context() const {
    return sycl::detail::createSyclObjFromImpl<sycl::context>(MContextImplPtr);
  }
  sycl::usm::alloc get_alloc_kind() const { return MKind; }
  const property_list &getPropList() const { return MPropList; }

  // Returns backend specific values.
  size_t get_threshold() const;
  size_t get_reserved_size_current() const;
  size_t get_reserved_size_high() const;
  size_t get_used_size_current() const;
  size_t get_used_size_high() const;

  void set_new_threshold(size_t newThreshold);
  void reset_reserved_size_high();
  void reset_used_size_high();
  void trim_to(size_t minBytesToKeep);

private:
  std::shared_ptr<sycl::detail::context_impl> MContextImplPtr;
  sycl::device MDevice;
  sycl::usm::alloc MKind;
  ur_usm_pool_handle_t MPoolHandle{0};
  bool MIsDefaultPool = false;
  property_list MPropList;
};
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
  sycl::context get_context() const { return impl->get_context(); }
  sycl::device get_device() const { return impl->get_device(); }
  sycl::usm::alloc get_alloc_kind() const { return impl->get_alloc_kind(); }

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
  template <typename propertyT> bool has_property() const noexcept {
    return getPropList().template has_property<propertyT>();
  }
  template <typename propertyT> propertyT get_property() const {
    return getPropList().template get_property<propertyT>();
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
};

} // namespace ext::oneapi::experimental

template <>
struct is_property<sycl::ext::oneapi::experimental::property::initial_threshold>
    : std::true_type {};

template <>
struct is_property<sycl::ext::oneapi::experimental::property::maximum_size>
    : std::true_type {};

template <>
struct is_property<sycl::ext::oneapi::experimental::property::read_only>
    : std::true_type {};

template <>
struct is_property<sycl::ext::oneapi::experimental::property::zero_init>
    : std::true_type {};
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
